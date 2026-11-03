"""Anthropic Messages API; protocol adapted from Pi v1.0.0 (MIT)."""

from __future__ import annotations
from collections.abc import AsyncGenerator
from ..cancellation import CancelToken
from ..provider import ModelRequest
from ..messages import ToolDeclaration, now
from typing import Any
import re
from copy import deepcopy
from ..errors import ConfigurationError, ProviderProtocolError, UnsupportedCapabilityError
from ..messages import (
    AssistantMessage,
    UserMessage,
    ToolResultMessage,
    TextContent,
    ImageContent,
    ThinkingContent,
    ToolCall,
    CustomMessage,
)
from ..provider import ModelEvent
from ..messages import SystemMessage
from ..estimate import clamp_max_tokens_to_context
from ..transcript import (
    current_tools,
    declared_tools,
    initial_system_message,
    render_system_update,
    resolve_transcript,
    system_message_text,
    with_request_tools,
)
from ..tools import invoke
from ..stream import event_contract, parse_tool_call_arguments
from .common import RemoteProvider, normalize_usage, transform_messages
from .transport import stream_error
from ..identity import apply_identity

_CC_NAMES = "Read Write Edit Bash Grep Glob AskUserQuestion EnterPlanMode ExitPlanMode KillShell NotebookEdit Skill Task TaskOutput TodoWrite WebFetch WebSearch".split()
_CC = {name.lower(): name for name in _CC_NAMES}


_FINE_GRAINED_TOOL_STREAMING = "fine-grained-tool-streaming-2025-05-14"
_INTERLEAVED_THINKING = "interleaved-thinking-2025-05-14"
_SERVER_SIDE_FALLBACK = "server-side-fallback-2026-07-01"
_MID_CONVERSATION_OUTPUT_CONFIG = "mid-conversation-output-config-2026-07-01"
_THINKING_BINDING_CONTROLS = "thinking-binding-controls-2026-08-01"
_MID_CONVERSATION_TOOL_CHANGES = "inline-tools-2026-09-15"
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}
# Declared from the first request whenever native tool changes are used. Anthropic adds
# hidden scaffolding once any tool is deferred; declaring it early keeps that scaffolding
# in the cached prefix. It is never activated and the model cannot see it.
_DEFERRED_PLACEHOLDER = {
    "name": "__pi_deferred_placeholder__",
    "description": "Reserved placeholder. Never available. Never call this.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
    "defer_loading": True,
}


def content_blocks(content: list[TextContent | ImageContent]) -> str | list[dict[str, Any]]:
    """Pi convertContentBlocks for tool results: text-only content becomes one string."""
    if not any(isinstance(b, ImageContent) for b in content):
        return "\n".join(b.text for b in content if isinstance(b, TextContent))
    blocks = user_blocks(content)
    if not any(b["type"] == "text" for b in blocks):
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})
    return blocks


def user_blocks(content: list[TextContent | ImageContent]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            result.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": block.mime_type, "data": block.data},
                }
            )
        else:
            raise UnsupportedCapabilityError("Unsupported Anthropic user/tool content")
    return result


def effort_for(level: str, level_map: dict[str, str | None]) -> str:
    """Pi mapThinkingLevelToEffort; a level mapped to None falls back by name."""
    mapped = level_map.get(level)
    if isinstance(mapped, str):
        return mapped
    return {"minimal": "low", "low": "low", "medium": "medium"}.get(level, "high")


class AnthropicProvider(RemoteProvider):
    name = "anthropic"
    api = "anthropic-messages"

    def __init__(
        self, *, base_url: str = "https://api.anthropic.com", auth_mode: str = "auto", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if auth_mode not in {"auto", "api_key", "oauth"}:
            raise ConfigurationError("auth_mode must be auto, api_key or oauth")
        self.base_url = base_url.rstrip("/")
        self._explicit_base_url = base_url != "https://api.anthropic.com"
        self.auth_mode = auth_mode

    def build_request(self, request: ModelRequest, oauth: bool = False) -> dict[str, Any]:
        return self._build(request, oauth)[0]

    def _build(
        self, request: ModelRequest, oauth: bool = False
    ) -> tuple[dict[str, Any], list[str], str | None]:
        """Return the body, the anthropic-beta list and the managed effort, if any."""
        options = request.options
        model = self.model_info(request)
        compat = model.compat
        thinking_disabled = (
            model.reasoning
            and options.get("reasoning") == "off"
            and model.thinking_level_map.get("off", "") is not None
        )
        managed = compat.get("supportsMidConvoEffort") is True and not thinking_disabled
        model_max_tokens = model.max_tokens
        adaptive = compat.get("forceAdaptiveThinking") is True
        level_map = model.thinking_level_map
        retention = options.get("cache_retention", "short")
        if retention not in {"none", "short", "long"}:
            raise ConfigurationError("Invalid cache_retention")
        cache = (
            None
            if retention == "none"
            else {
                "type": "ephemeral",
                **(
                    {"ttl": "1h"}
                    if retention == "long" and compat.get("supportsLongCacheRetention", True)
                    else {}
                ),
            }
        )

        def tool_name(name: str) -> str:
            return _CC.get(name.lower(), name) if oauth else name

        def declarations(
            tools: list[ToolDeclaration], cached: dict[str, Any] | None
        ) -> list[dict[str, Any]]:
            result = [
                {
                    "name": tool_name(t.name),
                    "description": t.description,
                    **(
                        {"eager_input_streaming": True}
                        if compat.get("supportsEagerToolInputStreaming", True)
                        else {}
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": deepcopy(t.input_schema.get("properties", {})),
                        "required": deepcopy(t.input_schema.get("required", [])),
                    },
                }
                for t in tools
            ]
            if cached and result:
                result[-1]["cache_control"] = deepcopy(cached)
            return result

        source = with_request_tools(request.messages, request.tools)

        def fit(limit: int) -> int:
            # Pi clampMaxTokensToContext: leave room for the estimated prompt.
            return clamp_max_tokens_to_context(model.context_window, source, limit)

        transcript = resolve_transcript(
            source, compat.get("supportsMidConvoSystemMessages") is True
        )
        initial = initial_system_message(transcript)
        initial_tools = initial.tools_added if initial else []
        # Inline definitions keep the initial list fixed, including redefinitions.
        # An initial active tool must anchor the deferred placeholder.
        native = (
            compat.get("supportsMidConvoSystemMessages") is True
            and compat.get("supportsMidConvoToolChanges") is True
            and bool(initial_tools)
        )
        transformed = transform_messages(
            transcript,
            self.name,
            self.api,
            request.model,
            lambda value, _: re.sub(r"[^a-zA-Z0-9_-]", "_", value)[:64],
        )
        conversation = transformed[1:] if initial else transformed
        calls = [
            b.id for m in conversation if isinstance(m, AssistantMessage) for b in m.tool_calls
        ]
        if len(set(calls)) != len(calls):
            raise ConfigurationError("Tool call IDs collide after Anthropic normalization")
        messages: list[dict[str, Any]] = []
        levels: dict[int, str] = {}
        # Later system messages go directly before the next assistant message or at the
        # end, because tool_result blocks must immediately follow their tool_use.
        held: list[dict[str, Any]] = []
        index = 0
        while index < len(conversation):
            message = conversation[index]
            index += 1
            if isinstance(message, SystemMessage):
                blocks: list[dict[str, Any]] = []
                text = render_system_update(message)
                if text:
                    blocks.append({"type": "text", "text": text})
                if native:
                    blocks += [
                        {
                            "type": "tool_removal",
                            "tool": {"type": "tool_reference", "name": tool_name(name)},
                        }
                        for name in message.tools_removed
                        if name not in {t.name for t in message.tools_added}
                    ]
                    blocks += [
                        {
                            "type": "tool_addition",
                            "tool": {"type": "tool_definition", "definition": definition},
                        }
                        for definition in declarations(message.tools_added, None)
                    ]
                if blocks:
                    held.append({"role": "system", "content": blocks})
            elif isinstance(message, UserMessage):
                if isinstance(message.content, str):
                    if message.content.strip():
                        messages.append({"role": "user", "content": message.content})
                else:
                    blocks = [
                        b
                        for b in user_blocks(message.content)
                        if b["type"] != "text" or b["text"].strip()
                    ]
                    if blocks:
                        messages.append({"role": "user", "content": blocks})
            elif isinstance(message, AssistantMessage):
                blocks = []
                for b in message.content:
                    if isinstance(b, TextContent):
                        if b.text.strip():
                            blocks.append({"type": "text", "text": b.text})
                    elif isinstance(b, ThinkingContent):
                        if b.redacted:
                            blocks.append(
                                {"type": "redacted_thinking", "data": b.thinking_signature}
                            )
                        elif b.thinking_signature and b.thinking_signature.strip():
                            blocks.append(
                                {
                                    "type": "thinking",
                                    "thinking": b.thinking,
                                    "signature": b.thinking_signature,
                                }
                            )
                        elif b.thinking.strip():
                            blocks.append(
                                {"type": "thinking", "thinking": b.thinking, "signature": ""}
                                if compat.get("allowEmptySignature") is True
                                else {"type": "text", "text": b.thinking}
                            )
                    elif isinstance(b, ToolCall):
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": b.id,
                                "name": tool_name(b.name),
                                "input": b.arguments,
                            }
                        )
                if not blocks:
                    continue
                messages += held
                held.clear()
                if (
                    managed
                    and message.api == self.api
                    and message.provider == self.name
                    and message.provider_thinking_level in _EFFORTS
                ):
                    levels[len(messages)] = message.provider_thinking_level
                messages.append({"role": "assistant", "content": blocks})
            elif isinstance(message, ToolResultMessage):
                results = [message]
                while index < len(conversation):
                    following = conversation[index]
                    if not isinstance(following, ToolResultMessage):
                        break
                    results.append(following)
                    index += 1
                result_blocks: list[dict[str, Any]] = []
                attachments: list[dict[str, Any]] = []
                for result in results:
                    images = [b for b in result.content if isinstance(b, ImageContent)]
                    content = result.content
                    if result.is_error and images:
                        content = []
                        previous_image = False
                        for block in result.content:
                            if isinstance(block, ImageContent):
                                if not previous_image:
                                    content.append(
                                        TextContent("(image attached after the tool results)")
                                    )
                                previous_image = True
                            else:
                                content.append(block)
                                previous_image = False
                        attachments.append(
                            {
                                "type": "text",
                                "text": f"Images from the error result of tool call {result.call_id} ({result.name}):",
                            }
                        )
                        attachments.extend(user_blocks(list(images)))
                    result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": result.call_id,
                            "content": content_blocks(content),
                            "is_error": result.is_error,
                        }
                    )
                messages.append({"role": "user", "content": result_blocks + attachments})
            elif isinstance(message, CustomMessage):
                raise UnsupportedCapabilityError("Convert custom messages before provider boundary")
        messages += held
        if cache and messages and messages[-1]["role"] in {"user", "system"}:
            last = messages[-1]
            if isinstance(last["content"], str):
                last["content"] = [
                    {"type": "text", "text": last["content"], "cache_control": deepcopy(cache)}
                ]
            elif last["content"] and last["content"][-1]["type"] in {
                "text",
                "image",
                "tool_result",
                "tool_addition",
                "tool_removal",
            }:
                last["content"][-1]["cache_control"] = deepcopy(cache)

        reasoning = options.get("reasoning")
        if reasoning == "off":
            reasoning = None
        if reasoning is not None and reasoning not in {
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ConfigurationError("Unsupported reasoning level")
        # Managed-effort models carry the effort as a marker after the history.
        active_effort = effort_for(reasoning, level_map) if managed and reasoning else "high"
        if managed:
            marked = []
            for position, value in enumerate(messages):
                if position in levels:
                    marked.append(
                        {
                            "role": "system",
                            "content": [],
                            "output_config": {"effort": levels[position]},
                        }
                    )
                marked.append(value)
            marked.append(
                {"role": "system", "content": [], "output_config": {"effort": active_effort}}
            )
            messages = marked
        body: dict[str, Any] = {
            "model": request.model,
            "stream": True,
            "max_tokens": fit(min(options.get("max_tokens", model_max_tokens), model_max_tokens)),
            "messages": messages,
        }
        system = system_message_text(initial) if initial else ""
        if oauth:
            body["system"] = [
                {
                    "type": "text",
                    "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                }
            ]
            if system:
                body["system"].append({"type": "text", "text": system})
        elif system:
            body["system"] = [{"type": "text", "text": system}]
        for block in body.get("system", []):
            if cache:
                block["cache_control"] = deepcopy(cache)

        tool_cache = cache if compat.get("supportsCacheControlOnTools", True) else None

        current = current_tools(transcript)
        if native:
            # Later tools live only in inline definitions; the cached prefix stays fixed.
            body["tools"] = [
                *declarations(initial_tools, tool_cache),
                deepcopy(_DEFERRED_PLACEHOLDER),
            ]
        elif current:
            body["tools"] = declarations(current, tool_cache)

        thinking_enabled = False
        display = options.get("thinking_display", "summarized")
        if thinking_disabled:
            body["thinking"] = {"type": "disabled"}
        elif managed:
            # Adaptive with block binding, so a prefix mismatch drops a thinking block
            # instead of failing every later request.
            thinking_enabled = reasoning is not None
            body["thinking"] = {
                "type": "adaptive",
                "display": display,
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            }
            body["output_config"] = {"effort": "high"}
        elif model.reasoning:
            if reasoning is not None:
                thinking_enabled = True
                if adaptive:
                    body["thinking"] = {"type": "adaptive", "display": display}
                    body["output_config"] = {"effort": effort_for(reasoning, level_map)}
                else:
                    level = "high" if reasoning in {"xhigh", "max"} else reasoning
                    budget = options.get("thinking_budgets", {}).get(level, _BUDGETS[level])
                    if type(budget) is not int or budget < 0:
                        raise ConfigurationError("Invalid thinking budget")
                    ceiling = fit(
                        model_max_tokens
                        if "max_tokens" not in options
                        else min(options["max_tokens"] + budget, model_max_tokens)
                    )
                    budget = min(budget, max(0, ceiling - 1024))
                    if budget < 1024:
                        raise ConfigurationError(
                            "Thinking requires room for at least 1024 thinking and 1024 answer tokens"
                        )
                    body["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget,
                        "display": display,
                    }
                    body["max_tokens"] = ceiling
            elif level_map.get("off", "") is not None:
                body["thinking"] = {"type": "disabled"}
        if (
            "temperature" in options
            and not thinking_enabled
            and not managed
            and compat.get("supportsTemperature", True)
        ):
            body["temperature"] = options["temperature"]
        metadata = options.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("user_id"), str):
            body["metadata"] = {"user_id": metadata["user_id"]}
        if "tool_choice" in options:
            choice = options["tool_choice"]
            body["tool_choice"] = {"type": choice} if isinstance(choice, str) else deepcopy(choice)
        fallbacks = compat.get("allowedFallbackModels") or []
        if fallbacks:
            body["fallbacks"] = [{"model": f["model"]} for f in fallbacks]
        # Python extensions with no Pi equivalent; raw thinking/output_config override above.
        for key in ("top_p", "top_k", "stop_sequences", "thinking", "output_config"):
            if key in options:
                body[key] = deepcopy(options[key])

        configured = [
            v for k, v in options.get("headers", {}).items() if k.lower() == "anthropic-beta"
        ]
        if configured:
            betas = list(
                dict.fromkeys(f.strip() for f in (configured[-1] or "").split(",") if f.strip())
            )
        else:
            betas = []
            if oauth:
                betas += ["claude-code-20250219", "oauth-2025-04-20"]
            if current and compat.get("supportsEagerToolInputStreaming", True) is False:
                betas.append(_FINE_GRAINED_TOOL_STREAMING)
            if thinking_enabled and not adaptive and options.get("interleaved_thinking", True):
                betas.append(_INTERLEAVED_THINKING)
            if fallbacks:
                betas.append(_SERVER_SIDE_FALLBACK)
            if managed:
                betas += [_MID_CONVERSATION_OUTPUT_CONFIG, _THINKING_BINDING_CONTROLS]
            if native:
                betas.append(_MID_CONVERSATION_TOOL_CHANGES)
            betas = list(dict.fromkeys(betas))
        return body, betas, active_effort if managed else None

    @event_contract
    async def stream(
        self, request: ModelRequest, cancel: CancelToken
    ) -> AsyncGenerator[ModelEvent, None]:
        request_started = now()
        key = await self.credential(request, cancel)
        oauth = self.auth_mode == "oauth" or (
            self.auth_mode == "auto"
            and (self.credentials is not None or key.startswith("sk-ant-oat"))
        )
        if request.options.get("transport", "sse") not in {"sse", "auto"}:
            raise UnsupportedCapabilityError("Anthropic Messages supports SSE transport")
        built, betas, effort = self._build(request, oauth)
        headers: dict[str, str | None] = {
            **{
                k: v
                for k, v in request.options.get("headers", {}).items()
                if k.lower() != "anthropic-beta"
            },
            **({"anthropic-beta": ",".join(betas)} if betas else {}),
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
            "content-type": "application/json",
        }
        if oauth:
            headers.update(
                {
                    "authorization": f"Bearer {key}",
                    "user-agent": "claude-cli/2.1.280",
                    "x-app": "cli",
                    "anthropic-dangerous-direct-browser-access": "true",
                }
            )
        else:
            headers["x-api-key"] = key
        headers = apply_identity(headers, self.name, self.base_url, request.options)
        body = await self.payload(request, built)
        blocks: dict[int, Any] = {}
        # Wire index -> content index; a leading server-side fallback block is skipped.
        positions: dict[int, int] = {}
        skipped: set[int] = set()
        arguments = {}
        usage = {}
        reason = None
        response_meta = {}
        ended = False
        closed_blocks = set()
        names = {
            t.name.lower(): t.name
            for t in declared_tools(with_request_tools(request.messages, request.tools))
        }
        events = self.transport.stream(
            self.base_url + "/v1/messages?beta=true",
            body,
            headers,
            cancel,
            on_response=request.on_response,
            **self.transport_options(request),
        )
        announced = False
        try:
            async for event in events:
                if not announced:
                    announced = True
                    yield ModelEvent("start")
                await invoke(request.on_provider_stream_event, deepcopy(event))
                kind = event.get("type")
                if ended:
                    raise ProviderProtocolError("Anthropic event after message_stop")
                if kind == "error":
                    raise stream_error(event, headers)
                if kind == "message_start":
                    usage.update(event["message"].get("usage", {}))
                    response_meta = event["message"]
                elif kind == "content_block_start":
                    index = event["index"]
                    block = event["content_block"]
                    typ = block["type"]
                    if type(index) is not int or index in blocks or index in skipped:
                        raise ProviderProtocolError("Duplicate Anthropic block")
                    if typ == "fallback":
                        if blocks:
                            raise ProviderProtocolError(
                                "Anthropic performed an unsupported mid-output model fallback"
                            )
                        skipped.add(index)
                        continue
                    if index != len(blocks) + len(skipped):
                        raise ProviderProtocolError("Out-of-order Anthropic block")
                    positions[index] = len(blocks)
                    if typ == "text":
                        blocks[index] = TextContent(block.get("text", ""))
                        yield ModelEvent.boundary("start", positions[index], TextContent(""))
                        if blocks[index].text:
                            yield ModelEvent.text(blocks[index].text, positions[index])
                    elif typ == "thinking":
                        blocks[index] = ThinkingContent(
                            block.get("thinking", ""), block.get("signature", "")
                        )
                        yield ModelEvent.boundary("start", positions[index], ThinkingContent(""))
                        if blocks[index].thinking:
                            yield ModelEvent.thinking(blocks[index].thinking, positions[index])
                    elif typ == "redacted_thinking":
                        blocks[index] = ThinkingContent("[Reasoning redacted]", block["data"], True)
                        yield ModelEvent.boundary("start", positions[index], blocks[index])
                    elif typ == "tool_use":
                        blocks[index] = ToolCall(
                            block["id"],
                            names.get(block["name"].lower(), block["name"])
                            if oauth
                            else block["name"],
                            block.get("input", {}),
                        )
                        arguments[index] = ""
                        yield ModelEvent.boundary("start", positions[index], blocks[index])
                    else:
                        raise UnsupportedCapabilityError(f"Unsupported Anthropic block: {typ}")
                elif kind == "content_block_delta":
                    index = event["index"]
                    if index in skipped:
                        continue
                    if index in closed_blocks:
                        raise ProviderProtocolError("Delta after content block stop")
                    block = blocks[index]
                    delta = event["delta"]
                    typ = delta["type"]
                    if typ == "text_delta" and isinstance(block, TextContent):
                        block.text += delta["text"]
                        yield ModelEvent.text(delta["text"], positions[index])
                    elif typ == "thinking_delta" and isinstance(block, ThinkingContent):
                        block.thinking += delta["thinking"]
                        yield ModelEvent.thinking(delta["thinking"], positions[index])
                    elif typ == "signature_delta" and isinstance(block, ThinkingContent):
                        block.thinking_signature = (block.thinking_signature or "") + delta[
                            "signature"
                        ]
                    elif typ == "input_json_delta" and isinstance(block, ToolCall):
                        arguments[index] += delta["partial_json"]
                        yield ModelEvent.toolcall(delta["partial_json"], positions[index])
                    else:
                        raise UnsupportedCapabilityError(f"Unsupported Anthropic delta: {typ}")
                elif kind == "content_block_stop":
                    index = event["index"]
                    if index in skipped:
                        continue
                    if index not in blocks or index in closed_blocks:
                        raise ProviderProtocolError("Invalid content block stop")
                    closed_blocks.add(index)
                    if index in arguments and arguments[index]:
                        blocks[index].arguments, blocks[index].arguments_error = (
                            parse_tool_call_arguments(arguments[index])
                        )
                    yield ModelEvent.boundary("end", positions[index], blocks[index])
                elif kind == "message_delta":
                    usage.update(event.get("usage", {}))
                    reason = event["delta"].get("stop_reason", reason)
                elif kind == "message_stop":
                    ended = True
        finally:
            await events.aclose()
        if not ended or reason is None or closed_blocks != set(blocks):
            raise ProviderProtocolError("Incomplete Anthropic stream")
        mapped = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "pause_turn": "stop",
            "tool_use": "tool_use",
            "max_tokens": "length",
            "refusal": "error",
            "sensitive": "error",
        }.get(reason)
        if mapped is None:
            raise ProviderProtocolError("Unknown Anthropic stop reason")
        if mapped == "error":
            raise ProviderProtocolError(f"Anthropic stopped: {reason}")
        yield ModelEvent.done(
            AssistantMessage(
                list(blocks.values()),
                mapped,
                self.name,
                request.model,
                normalize_usage(usage, self.name),
                timestamp=request_started,
                diagnostics=None if usage else [{"type": "usage_unavailable"}],
                api="anthropic-messages",
                provider_thinking_level=effort,
                thinking_level=request.options.get("reasoning"),
                response_id=response_meta.get("id"),
                response_model=response_meta.get("model")
                if response_meta.get("model") != request.model
                else None,
                raw_stop_reason=reason,
            )
        )
