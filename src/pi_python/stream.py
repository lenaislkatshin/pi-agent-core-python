"""Full Pi-style model events with defensive snapshots and strict terminal validation."""

from __future__ import annotations
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any
from .cancellation import CancelToken
from .provider import ModelRequest
import asyncio
import json
import time
from copy import deepcopy
from functools import wraps
from .errors import ProviderProtocolError
from .messages import AssistantMessage, TextContent, ThinkingContent, ToolCall, message_to_dict
from .provider import ModelEvent


def parse_tool_call_arguments(text: str | None) -> tuple[dict[str, Any], str | None]:
    """Parse the finished arguments text; malformed text is reported, never partly kept.

    Returns `(arguments, arguments_error)`. The error includes the received text so tool
    validation can hand the problem back to the model, as Pi's parseToolCallArguments does.
    """
    if not text or not text.strip():
        return {}, None
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be a JSON object")
    except ValueError as exc:
        return {}, f"{exc}\n\nReceived arguments:\n{text}"
    return parsed, None


def partial_json(value: str) -> dict:
    """Best-effort UI preview only. Final arguments always use strict JSON parsing."""
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except ValueError:
        pass
    # Bound preview work. This limit never relaxes strict final JSON validation.
    if len(value) > 262144:
        return {}
    candidates = [len(value)] + [i for i in range(len(value) - 1, -1, -1) if value[i] in ",:{["][
        :31
    ]
    for end in candidates:
        prefix = value[:end].rstrip().rstrip(",")
        stack = []
        quoted = False
        escaped = False
        for char in prefix:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "{[":
                stack.append("}" if char == "{" else "]")
            elif char in "}]" and stack:
                stack.pop()
        suffix = ('"' if quoted else "") + "".join(reversed(stack))
        try:
            parsed = json.loads(prefix + suffix)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    return {}


def terminal_content_matches(
    ended: list[TextContent | ThinkingContent | ToolCall],
    final: list[TextContent | ThinkingContent | ToolCall],
) -> bool:
    """Allow only Pi's late encrypted-reasoning backfill; never alter executable data."""
    if len(ended) != len(final):
        return False
    for old, new in zip(ended, final):
        if old == new:
            continue
        if not isinstance(old, ThinkingContent) or not isinstance(new, ThinkingContent):
            return False
        if old.thinking != new.thinking or old.redacted != new.redacted:
            return False
        try:
            before, after = (
                json.loads(old.thinking_signature or ""),
                json.loads(new.thinking_signature or ""),
            )
        except ValueError:
            return False
        if not isinstance(before, dict) or not isinstance(after, dict):
            return False
        if before.get("encrypted_content") or not after.get("encrypted_content"):
            return False
        before.pop("encrypted_content", None)
        after.pop("encrypted_content", None)
        if before != after:
            return False
    return True


_BLOCK_STARTS = {"text_start", "thinking_start", "toolcall_start"}
_BLOCK_ENDS = {"text_end", "thinking_end", "toolcall_end"}
_DELTAS = {"text_delta": TextContent, "thinking_delta": ThinkingContent, "toolcall_delta": ToolCall}


def _time_message(message: AssistantMessage, started_at: float, started: float) -> None:
    # A forwarded response may have started elsewhere. Do not measure its replay.
    if message.duration_ms is None and message.timestamp >= started_at:
        message.duration_ms = max(0, int((time.monotonic() - started) * 1000 + 0.5))


async def checked_events(
    source: AsyncIterator[ModelEvent], partial: AssistantMessage
) -> AsyncGenerator[ModelEvent, None]:
    """Enforce the ModelEvent contract and attach an independent `partial` snapshot.

    Block boundaries must pair up, deltas must fall inside an open block of their own
    kind, and the final message must equal the ended blocks (only Pi's late encrypted
    reasoning backfill may differ). `done` is released only after the source ends
    cleanly. Exceptions propagate; `error` events from a checked source pass through.
    """
    opened: set[int] = set()
    started_at, started = time.time(), time.monotonic()
    closed: set[int] = set()
    arguments: dict[int, str] = {}
    announced = False
    terminal = None
    async for source_event in source:
        if terminal is not None:
            raise ProviderProtocolError("Event after final message")
        if not isinstance(source_event, ModelEvent):
            raise ProviderProtocolError("Expected ModelEvent")
        event = deepcopy(source_event)
        kind = event.type
        if kind == "start":
            if announced:
                raise ProviderProtocolError("Duplicate start event")
            announced = True
            if event.partial is not None:
                partial = deepcopy(event.partial)
            event.partial = deepcopy(partial)
            yield event
            continue
        if kind == "error":
            if isinstance(event.message, AssistantMessage):
                _time_message(event.message, started_at, started)
            yield event
            return
        if kind == "done":
            message = event.message
            if not isinstance(message, AssistantMessage) or message.stop_reason == "pending":
                raise ProviderProtocolError("Invalid terminal message")
            message_to_dict(message)
            if opened != closed or (opened and len(message.content) != len(partial.content)):
                raise ProviderProtocolError("Terminal with unfinished blocks")
            if opened and not terminal_content_matches(partial.content, message.content):
                raise ProviderProtocolError("End/final content mismatch")
            terminal = ModelEvent("done", message=deepcopy(message), reason=message.stop_reason)
            continue  # Do not publish success before the source closes cleanly.
        if not announced:
            announced = True
            yield ModelEvent("start", partial=deepcopy(partial))
        if kind in _BLOCK_STARTS:
            if event.index != len(partial.content) or event.block is None:
                raise ProviderProtocolError("Invalid content block start")
            opened.add(event.index)
            partial.content.append(deepcopy(event.block))
        elif kind in _DELTAS:
            if not isinstance(event.delta, str):
                raise ProviderProtocolError("Delta must be text")
            if event.index not in opened or event.index in closed:
                raise ProviderProtocolError("Delta outside open block")
            block = partial.content[event.index]
            if not isinstance(block, _DELTAS[kind]):
                raise ProviderProtocolError("Delta type mismatch")
            if isinstance(block, TextContent):
                block.text += event.delta
            elif isinstance(block, ThinkingContent):
                block.thinking += event.delta
            else:
                arguments[event.index] = arguments.get(event.index, "") + event.delta
                block.arguments = partial_json(arguments[event.index])
                event.call_id, event.name = block.id, block.name
        elif kind in _BLOCK_ENDS:
            if event.index not in opened or event.index in closed or event.block is None:
                raise ProviderProtocolError("Invalid block end")
            old = partial.content[event.index]
            new = event.block
            if type(old) is not type(new):
                raise ProviderProtocolError("Block type changed")
            if isinstance(old, TextContent) and isinstance(new, TextContent):
                if old.text != new.text:
                    raise ProviderProtocolError("Text delta/end mismatch")
            elif isinstance(old, ThinkingContent) and isinstance(new, ThinkingContent):
                if old.thinking != new.thinking:
                    raise ProviderProtocolError("Thinking delta/end mismatch")
            elif isinstance(old, ToolCall) and isinstance(new, ToolCall):
                if old.id != new.id or old.name != new.name:
                    raise ProviderProtocolError("Tool identity changed")
                # Anthropic streams an empty input_json_delta for a tool without arguments.
                if event.index in arguments and arguments[event.index].strip():
                    assembled, invalid = parse_tool_call_arguments(arguments[event.index])
                    if invalid is not None:
                        if new.arguments_error is None:
                            raise ProviderProtocolError("Incomplete tool arguments")
                    elif assembled != new.arguments:
                        raise ProviderProtocolError("Tool delta/end mismatch")
            partial.content[event.index] = deepcopy(new)
            closed.add(event.index)
            event.content = (
                new.text
                if isinstance(new, TextContent)
                else new.thinking
                if isinstance(new, ThinkingContent)
                else None
            )
        else:
            raise ProviderProtocolError(f"Unknown model event: {kind}")
        event.partial = deepcopy(partial)
        yield event
    if terminal is None:
        raise ProviderProtocolError("Missing terminal message")
    if terminal.message is not None:
        _time_message(terminal.message, started_at, started)
    yield terminal


def event_contract(function: Callable[..., AsyncGenerator[ModelEvent, None]]) -> Any:
    """Decorate a remote provider: checked events, and failures as an `error` event."""

    @wraps(function)
    async def wrapped(
        self: Any, request: ModelRequest, cancel: CancelToken
    ) -> AsyncGenerator[ModelEvent, None]:
        started_at, started = time.time(), time.monotonic()
        partial = AssistantMessage(
            [],
            stop_reason="pending",
            provider=self.name,
            model=request.model,
            api=getattr(self, "api", None),
        )
        snapshot = partial
        if hasattr(self, "prepare_stream_request"):
            # Resolve inside error handling below so configuration failures still
            # obey the provider error-event contract.
            async def prepared_source() -> AsyncGenerator[ModelEvent, None]:
                provider, prepared = self.prepare_stream_request(request)
                source = function(provider, prepared, cancel)
                try:
                    async for item in source:
                        yield item
                finally:
                    await source.aclose()

            iterator = prepared_source()
        else:
            iterator = function(self, request, cancel)
        events = checked_events(iterator, partial)
        try:
            async for event in events:
                if event.partial is not None:
                    snapshot = event.partial
                yield event
        except asyncio.CancelledError:
            failed = deepcopy(snapshot)
            _time_message(failed, started_at, started)
            failed.stop_reason, failed.error = "aborted", "Request cancelled"
            yield ModelEvent("error", message=failed, reason="aborted")
        except Exception as exc:
            failed = deepcopy(snapshot)
            _time_message(failed, started_at, started)
            failed.stop_reason, failed.error = "error", f"{type(exc).__name__}: {exc}"
            if hasattr(exc, "status") and hasattr(exc, "category"):
                failed.diagnostics = [
                    {
                        "type": "provider_http_error",
                        "status": exc.status,
                        "category": exc.category,
                        "retry_after": getattr(exc, "retry_after", None),
                        "request_id": getattr(exc, "request_id", None),
                    }
                ]
            yield ModelEvent("error", message=failed, reason="error")
        finally:
            await events.aclose()
            await iterator.aclose()

    wrapped.checked = True  # type: ignore[attr-defined]
    return wrapped
