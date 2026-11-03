"""Model/tool loop. Shared observable behavior derives from pinned Pi (see NOTICE)."""

from __future__ import annotations
from typing import Any
from .stream import checked_events, _time_message
import asyncio
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING

from .errors import ConfigurationError, ProviderProtocolError, UnsupportedCapabilityError
from .messages import (
    Message,
    AssistantMessage,
    CustomMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    message_from_dict,
    message_to_dict,
    validate_history,
    convert_application_tool_batch,
)
from .models import ModelInfo
from .provider import ModelRequest
from .tools import (
    ToolContext,
    ToolOutcome,
    aborted_result,
    error_result,
    prepare_tool_call,
    run_tool_call,
)
from .transcript import current_tools
from .tasks import _gather_owned

if TYPE_CHECKING:
    from .run import Run


def abort_error(run: Run) -> str:
    return f"Request aborted ({run.token.reason})" if run.token.reason else "Request aborted"


def without_tool_calls(message: AssistantMessage) -> AssistantMessage:
    """A failed response cannot declare calls (D7); keep its other content and say what was removed."""
    calls = [b.id for b in message.content if isinstance(b, ToolCall)]
    if calls and message.stop_reason in {"error", "aborted"}:
        message.content = [b for b in message.content if not isinstance(b, ToolCall)]
        message.diagnostics = [
            *(message.diagnostics or []),
            {"type": "removed_tool_calls", "call_ids": calls},
        ]
    return message


def failure_message(
    run: Run, stop_reason: str, error: str | None, partial: dict[str, Any] | None = None
) -> AssistantMessage:
    """Pi's record of a failed or aborted response: what streamed so far, plus the error."""
    model = run.context.model
    if partial is not None:
        message = message_from_dict(partial)
        assert isinstance(message, AssistantMessage)
    else:
        message = AssistantMessage(
            [TextContent("")],
            provider=getattr(run.provider, "name", "custom"),
            model=model.id if isinstance(model, ModelInfo) else model,
            api=getattr(run.provider, "api", None),
        )
    if run.response_started is not None and run.response_started_at is not None:
        if partial is None:
            message.timestamp = run.response_started_at
        _time_message(message, run.response_started_at, run.response_started)
    message.stop_reason, message.error = stop_reason, error
    message.thinking_level = run.context.options.get("reasoning", "off")
    return without_tool_calls(message)


async def stream_response(run: Run) -> tuple[AssistantMessage, bool]:
    context = run.context
    assert context is not None
    run.partial, run.started = None, False
    run.response_started = None
    run.response_started_at = None
    messages = deepcopy(context.messages)
    if run.hooks.transform_context:
        messages = await run.hook("transform_context", messages, run.token)
    if run.hooks.convert_to_llm:
        messages = await run.hook("convert_to_llm", deepcopy(messages))
    messages = [
        convert_application_tool_batch(m)
        if isinstance(m, CustomMessage) and m.custom_type == "application_tool_batch"
        else m
        for m in messages
    ]
    if any(isinstance(m, CustomMessage) for m in messages):
        raise UnsupportedCapabilityError("Custom messages require convert_to_llm")
    validate_history(messages)
    for message in messages:
        if isinstance(message, ToolResultMessage):
            message.details = None
            message.usage = None
            message.nested_calls = None
    model = context.model
    request = ModelRequest(
        deepcopy(messages),
        current_tools(messages),
        model.id if isinstance(model, ModelInfo) else model,
        deepcopy(context.options),
        model_info=deepcopy(model) if isinstance(model, ModelInfo) else None,
    )
    if run.identity is not None:
        request.options["identity"] = {
            **deepcopy(run.identity),
            **(request.options.get("identity") or {}),
        }
    if (
        getattr(run.provider, "supports_agent_cache_session", False) is True
        and not request.options.get("session_id")
        and request.options.get("cache_retention") != "none"
    ):
        cache_session_id = run.agent.cache_session_id
        if cache_session_id is not None:
            request._set_cache_session_id(cache_session_id)
    if run.hooks.get_api_key:
        request.api_key = await run.hook(
            "get_api_key", getattr(run.provider, "name", request.model)
        )
    request.on_payload = run.hooks.on_payload
    request.on_response = run.hooks.on_response
    request.on_provider_stream_event = run.hooks.on_provider_stream_event
    if run.token.cancelled:
        # A Pi stream function given an aborted signal answers at once; no request is sent.
        return failure_message(run, "aborted", abort_error(run)), False
    run.response_started_at, run.response_started = time.time(), time.monotonic()
    source = run.provider.stream(request, run.token)
    stream_method = getattr(run.provider, "stream", None)
    # Remote providers already check their own events; check every other source here.
    iterator = (
        source
        if getattr(stream_method, "checked", False)
        else checked_events(
            source,
            AssistantMessage(
                [],
                stop_reason="pending",
                provider=getattr(run.provider, "name", "custom"),
                model=request.model,
                api=getattr(run.provider, "api", None),
            ),
        )
    )
    final = None
    try:
        async for event in iterator:
            run.token.raise_if_cancelled()
            if event.type == "error":
                # Pi commits the provider's own failed message: partial content, usage, error.
                stop_reason = "aborted" if event.reason == "aborted" else "error"
                failed = event.message
                if isinstance(failed, AssistantMessage) and failed.stop_reason == stop_reason:
                    final = without_tool_calls(deepcopy(failed))
                    final.thinking_level = context.options.get("reasoning", "off")
                else:
                    error = getattr(failed, "error", None) or "Provider failed"
                    final = failure_message(run, stop_reason, error, run.partial)
                break
            if event.type == "done":
                assert event.message is not None
                final = without_tool_calls(deepcopy(event.message))
                final.thinking_level = context.options.get("reasoning", "off")
                continue
            assert event.partial is not None  # checked events always carry a snapshot
            run.partial = message_to_dict(event.partial)
            if not run.started:
                run.started = True
                await run.emit("message_start", message=run.partial)
            if event.type != "start":
                core_event: dict[str, Any] = {}
                if run.agent._core_contract:
                    from dataclasses import asdict

                    core_event = {
                        "assistant_message_event": {
                            "type": event.type,
                            "content_index": event.index,
                            "partial": message_to_dict(event.partial),
                            **({"delta": event.delta} if event.type.endswith("_delta") else {}),
                            **(
                                {"content": event.content}
                                if event.type.endswith("_end") and event.content is not None
                                else {}
                            ),
                            **(
                                {"tool_call": asdict(event.block)}
                                if event.type == "toolcall_end" and event.block
                                else {}
                            ),
                        }
                    }
                await run.emit(
                    "message_update",
                    delta_type=event.type,
                    block_index=event.index,
                    delta=event.delta,
                    content=event.content,
                    tool_call_id=event.call_id,
                    partial=run.partial,
                    **core_event,
                )
    finally:
        try:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()
        finally:
            # Closing the validator does not close the iterator it wraps. Own the
            # per-request stream here, without closing the shared Provider.
            if source is not iterator:
                close = getattr(source, "aclose", None)
                if close is not None:
                    await close()
    if final is None:
        raise ProviderProtocolError("Stream ended without final message")
    run.partial = None
    return final, run.started


async def execute_batch(run: Run, outcomes: list[ToolOutcome]) -> None:
    context = run.context
    assert context is not None
    tools = {t.name: t for t in context.tools}
    sequential = run.execution_mode == "sequential" or any(
        tools[o.call.name].execution_mode == "sequential" for o in outcomes if o.call.name in tools
    )
    limit = run.limits.max_concurrency
    # Unlimited by default, like Pi's Promise.all over the batch.
    semaphore = asyncio.Semaphore(limit) if limit is not None else nullcontext()

    # One snapshot per batch, shared read-only by its calls: copying the whole context
    # for every call made large batches over long histories slow.
    snapshot: dict[str, Any] = {}
    contexts: dict[int, ToolContext] = {}

    def tool_context(outcome: ToolOutcome) -> ToolContext:
        if id(outcome) in contexts:
            return contexts[id(outcome)]
        if not snapshot:
            snapshot.update(message=deepcopy(context.message), context=deepcopy(context))

        async def emit(value: Any) -> None:
            await run.emit("tool_execution_update", outcome.call.id, update=value)

        contexts[id(outcome)] = ToolContext(
            run.id,
            outcome.call.id,
            run.token,
            emit,
            assistant_message=snapshot["message"],
            agent_context=snapshot["context"],
            _on_reconciliation_required=lambda: run.require_reconciliation(),
        )
        if run.application_batch:
            contexts[id(outcome)]._set_origin(outcome.origin, outcome.source)
        return contexts[id(outcome)]

    def abort_before_start(outcome: ToolOutcome) -> bool:
        """Pi checks the abort signal before preparing and before executing each call."""
        run.checkpoint()
        if run.token.cancelled and outcome.result is None:
            outcome.result = aborted_result()
        return outcome.result is not None

    async def prepare(outcome: ToolOutcome) -> None:
        if abort_before_start(outcome):
            return
        await run.emit(
            "tool_execution_start",
            outcome.call.id,
            name=outcome.call.name,
            arguments=outcome.original_arguments,
        )
        try:
            await run.await_owned(
                prepare_tool_call(
                    tools.get(outcome.call.name),
                    outcome,
                    tool_context(outcome),
                    run.hooks.before_tool_call,
                )
            )
        except asyncio.CancelledError:
            if not run.aborted():
                raise
        abort_before_start(outcome)

    async def execute(outcome: ToolOutcome) -> None:
        def timed_out() -> None:
            # Freeze the fact known at the deadline before delivering cancellation.
            # A coroutine which suppresses cancellation cannot rewrite this decision.
            if outcome.execution_status == "running":
                outcome.execution_status = "cancelled"
                outcome.result = error_result("tool_timeout", "Tool timed out")
            else:
                outcome.result = error_result(
                    "finalization_timeout", "Execution finished; finalization timed out"
                )
            outcome._settled = True

        async with semaphore:
            if not abort_before_start(outcome):
                try:
                    await run.await_owned(
                        run_tool_call(
                            tools.get(outcome.call.name),
                            outcome.call,
                            tool_context(outcome),
                            after_tool_call=run.hooks.after_tool_call,
                            outcome=outcome,
                            prepared=True,
                        ),
                        run.limits.tool_timeout,
                        on_timeout=timed_out,
                    )
                except TimeoutError:
                    pass  # timed_out already recorded the deadline result
                except asyncio.CancelledError:
                    # Abort reached the tool. Its own result, or Pi's "Operation aborted",
                    # was recorded by run_tool_call unless it is still running.
                    if not run.aborted():
                        raise
            await run.end_tool(outcome)

    if sequential:
        for outcome in outcomes:
            await prepare(outcome)
            await execute(outcome)
            await run.commit_outcome(outcome)
            if outcome.execution_status == "unknown":
                break
    else:
        for outcome in outcomes:
            await prepare(outcome)
            if outcome.result is not None:
                await run.end_tool(outcome)
        # Every await inside a worker is owned and bounded, so an abort reaches each tool
        # directly and the batch settles; caller cancellation cancels the workers below.
        await _gather_owned(execute(o) for o in outcomes)
        for outcome in outcomes:
            await run.commit_outcome(outcome)


def exceeds_tool_budget(run: Run, outcomes: list[ToolOutcome], used: int = 0) -> bool:
    maximum = run.limits.max_tool_calls
    if maximum is None or used + len(outcomes) <= maximum:
        return False
    for outcome in outcomes:
        outcome.result = error_result("limit", "Entire tool batch exceeds remaining budget")
    return True


async def run_application_batch(run: Run) -> tuple[str, str]:
    run.checkpoint()
    await run.emit("agent_start")
    if exceeds_tool_budget(run, run.active_batch):
        return "limit_reached", "max_tool_calls"
    await execute_batch(run, run.active_batch)
    if run.agent._unknown:
        return "failed", "outcome_unknown"
    if run.stuck():
        run.fail("Tool did not stop after cancellation")
        return "cancelled" if run.token.cancelled else "failed", "tool_not_stopped"
    if run.token.cancelled:
        status, reason, _ = run.cancelled_outcome()
        return status, reason
    return "completed", "stop"


async def run_loop(run: Run, initial: list[Message]) -> tuple[str, str]:
    context = run.context
    assert context is not None
    requests = tool_count = 0
    run.checkpoint()
    await run.emit("agent_start")
    run.turn = 1
    await run.emit("turn_start")
    await run.pending(initial)
    pending = [] if run.skip_initial_steering else await run.poll_queue(True)
    while True:
        # After an abort the loop still runs to Pi's end: the next response is an
        # aborted message, recorded like any other, without a provider request.
        run.checkpoint()
        max_requests = run.limits.max_model_requests
        if max_requests is not None and requests >= max_requests:
            return "limit_reached", "max_model_requests"
        prepared = []
        if requests:
            prepared = await run.prepare("prepare_next_turn")
            if not pending:
                pending = await run.poll_queue(True)
            run.turn += 1
            await run.emit("turn_start")
        await run.apply_updates()
        await run.pending(prepared + pending)
        extra = await run.prepare("prepare_request")
        # Run hook updates persist, but never rewrite the committed transcript.
        await run.pending(extra)
        run.checkpoint()
        requests += 1
        failed_reason = None
        try:
            message, started = await run.await_owned(stream_response(run))
        except asyncio.CancelledError:
            if not run.aborted():
                raise
            message = failure_message(run, "aborted", abort_error(run), run.partial)
            started = run.started
        except Exception as exc:
            failed_reason = f"{type(exc).__name__}: {exc}"
            message = failure_message(run, "error", failed_reason, run.partial)
            started = run.started
        run.partial = None
        run.response_started = None
        outcomes = [
            ToolOutcome(deepcopy(c), original_arguments=deepcopy(c.arguments))
            for c in message.tool_calls
        ]
        run.active_batch = outcomes
        run.outcomes.extend(outcomes)
        await run.append(message, started=started)
        context.message = deepcopy(message)
        status = "completed"
        reason = message.stop_reason
        if message.stop_reason == "error":
            status = "failed"
            failed_reason = failed_reason or message.error
            run.fail(message.error or "Provider returned an error response")
        elif message.stop_reason == "aborted":
            # Pi exposes the aborted message's error as the agent's error message.
            run.fail(message.error or "Request aborted")
            if run.token.reason == "run_timeout":
                status, reason = "limit_reached", "run_timeout"
            else:
                status, reason = "cancelled", run.token.reason or "aborted"
        elif outcomes:
            if exceeds_tool_budget(run, outcomes, tool_count):
                status, reason = "limit_reached", "max_tool_calls"
            elif message.stop_reason == "length":
                for outcome in outcomes:
                    await run.emit(
                        "tool_execution_start",
                        outcome.call.id,
                        name=outcome.call.name,
                        arguments=outcome.original_arguments,
                    )
                    outcome.result = error_result(
                        "truncated",
                        f'Tool call "{outcome.call.name}" was not executed: the response hit '
                        "the output token limit, so its arguments may be truncated. Re-issue "
                        "the tool call with complete arguments.",
                    )
                    await run.end_tool(outcome)
                    await run.commit_outcome(outcome)
            else:
                tool_count += len(outcomes)
                await execute_batch(run, outcomes)
            for outcome in outcomes:
                await run.end_tool(outcome)
                await run.commit_outcome(outcome)
            if run.agent._unknown:
                status, reason = "failed", "outcome_unknown"
            elif run.stuck():
                # Never start another request while a cancelled tool keeps running.
                run.fail("Tool did not stop after cancellation")
                status = "cancelled" if run.token.cancelled else "failed"
                reason = "tool_not_stopped"
        context.tool_results = [
            ToolResultMessage(
                o.call.id,
                o.call.name,
                deepcopy(o.result.content),
                o.result.is_error,
                details=deepcopy(o.result.details),
                usage=deepcopy(o.result.usage),
                nested_calls=deepcopy(o.result.nested_calls),
                duration_ms=o.duration_ms,
            )
            for o in outcomes
            if o.result
        ]
        decision = await run.hook("finish_turn", deepcopy(context), run.token)
        if decision not in {None, "continue", "end"}:
            raise ConfigurationError("finish_turn must return continue, end or None")
        await run.emit(
            "turn_end",
            message=message_to_dict(message),
            tool_results=[message_to_dict(m) for m in context.tool_results],
        )
        if status != "completed":
            return status, failed_reason or reason
        if decision == "end":
            return "completed", "finish_turn"
        pending = await run.poll_queue(True)
        natural = bool(outcomes) and not all(o.result and o.result.terminate for o in outcomes)
        if natural or pending:
            continue
        pending = await run.poll_queue(False)
        if pending or decision == "continue":
            continue
        return "completed", reason
