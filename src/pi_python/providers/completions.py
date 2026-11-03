"""OpenAI-compatible Chat Completions, ported from Pi v1.0.0 api/openai-completions.ts.

This is the wire protocol of local model servers (vLLM, Ollama, llama.cpp, LM Studio,
SGLang) and of many hosted services (DeepSeek, Groq, OpenRouter, Together, Qwen, ...).
Endpoint differences are described by `ModelInfo.compat` flags with Pi's names; flags
left unset are detected from the provider name and base URL exactly as upstream does.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import AsyncGenerator
from copy import deepcopy
from typing import Any

from ..cancellation import CancelToken
from ..errors import ConfigurationError, ProviderProtocolError, UnsupportedCapabilityError
from ..estimate import clamp_max_tokens_to_context, short_hash
from ..messages import (
    AssistantMessage,
    CustomMessage,
    ImageContent,
    Message,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDeclaration,
    ToolResultMessage,
    UserMessage,
    now,
)
from ..models import LEVELS, ModelCatalog, ModelInfo
from ..provider import ModelEvent, ModelRequest
from ..stream import event_contract, parse_tool_call_arguments
from ..tools import invoke
from ..transcript import (
    render_system_update,
    resolve_transcript,
    _resolve_tool_changes,
    system_message_text,
    with_request_tools,
)
from .common import RemoteProvider, transform_messages
from ..identity import apply_identity
from .transport import stream_error

API = "openai-completions"

# Delta fields that carry visible reasoning, in Pi's lookup order (first non-empty wins).
_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")
_THINKING_FORMATS = {
    "openai",
    "openrouter",
    "deepseek",
    "together",
    "baseten",
    "zai",
    "qwen",
    "chat-template",
    "qwen-chat-template",
    "string-thinking",
    "ant-ling",
}
_BUDGET_FIELDS = {"thinking_token_budget", "thinking_budget", "thinking_budget_tokens"}
_AFFINITY_FORMATS = {"openai", "openai-nosession", "openrouter"}
_BOOL_FLAGS = (
    "supportsStore",
    "supportsDeveloperRole",
    "supportsReasoningEffort",
    "supportsUsageInStreaming",
    "supportsFinishReason",
    "requiresToolResultName",
    "requiresAssistantAfterToolResult",
    "requiresThinkingAsText",
    "requiresReasoningContentOnAssistantMessages",
    "zaiToolStream",
    "supportsThinkingTokenBudget",
    "supportsStrictMode",
    "supportsOpenAIGrammarTools",
    "supportsMidConvoSystemMessages",
    "supportsMidConvoToolAdditions",
    "sendSessionAffinityHeaders",
    "supportsLongCacheRetention",
)
# Pi DEFAULT_THINKING_BUDGETS and MIN_ANSWER_TOKENS (api/simple-options.ts).
_DEFAULT_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}
_MIN_ANSWER_TOKENS = 1024
_BRIDGE = "I have processed the tool results."
# JavaScript String.prototype.trim whitespace.
_JS_SPACE = " \t\n\v\f\r                 　﻿"
_LONE_SURROGATE = re.compile(
    r"[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]"
)
_OMIT = object()


def _sanitize(text: str) -> str:
    """Pi sanitizeSurrogates: drop unpaired surrogates, which cannot be encoded."""
    return _LONE_SURROGATE.sub("", text)


def _blank(text: str) -> bool:
    return not text.strip(_JS_SPACE)


def _truthy(value: Any) -> bool:
    """JavaScript truthiness: empty lists and objects are true, 0 and "" are false."""
    if isinstance(value, (list, dict)):
        return True
    return bool(value) and value == value


def _number(value: Any) -> bool:
    return type(value) in (int, float)


def _first_defined(*values: Any) -> Any:
    """JavaScript `a ?? b ?? ...`."""
    return next((v for v in values if v is not None), None)


def detect_compat(provider: str, base_url: str, model_id: str) -> dict[str, Any]:
    """Pi detectCompat: settings inferred from the provider name and base URL."""
    is_zai = (
        provider in {"zai", "zai-coding-cn"}
        or "api.z.ai" in base_url
        or "open.bigmodel.cn" in base_url
    )
    is_together = (
        provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    )
    is_moonshot = provider in {"moonshotai", "moonshotai-cn"} or "api.moonshot." in base_url
    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_cf_workers = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cf_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url
    is_cerebras = provider == "cerebras" or "cerebras.ai" in base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url.lower()
    is_grok = provider == "xai" or "api.x.ai" in base_url
    non_standard = (
        is_nvidia
        or is_cerebras
        or is_grok
        or is_together
        or "chutes.ai" in base_url
        or is_deepseek
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cf_workers
        or is_cf_gateway
        or is_ant_ling
    )
    use_max_tokens = (
        "chutes.ai" in base_url
        or is_deepseek
        or is_moonshot
        or is_cf_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )
    developer_role_model = is_openrouter and model_id.startswith(("anthropic/", "openai/"))
    return {
        "supportsStore": not non_standard,
        "supportsDeveloperRole": developer_role_model or (not non_standard and not is_openrouter),
        "supportsReasoningEffort": not (
            is_grok
            or is_zai
            or is_moonshot
            or is_together
            or is_cf_gateway
            or is_nvidia
            or is_ant_ling
        ),
        "supportsUsageInStreaming": True,
        "supportsFinishReason": True,
        "maxTokensField": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requiresToolResultName": False,
        "requiresAssistantAfterToolResult": False,
        "requiresThinkingAsText": False,
        "requiresReasoningContentOnAssistantMessages": is_deepseek,
        "thinkingFormat": "deepseek"
        if is_deepseek
        else "zai"
        if is_zai
        else "together"
        if is_together
        else "ant-ling"
        if is_ant_ling
        else "openrouter"
        if is_openrouter
        else "openai",
        "openRouterRouting": {},
        "vercelGatewayRouting": {},
        "chatTemplateKwargs": {},
        "chatTemplateArgs": {},
        "zaiToolStream": False,
        "supportsThinkingTokenBudget": False,
        "thinkingTokenBudgetField": None,
        # OpenAI compatibility alone does not imply strict JSON-schema tool support.
        "supportsStrictMode": False,
        "supportsOpenAIGrammarTools": False,
        "supportsMidConvoSystemMessages": False,
        "supportsMidConvoToolAdditions": False,
        "cacheControlFormat": "anthropic"
        if provider == "openrouter" and model_id.startswith("anthropic/")
        else None,
        "sendSessionAffinityHeaders": is_openrouter,
        "sessionAffinityFormat": "openrouter" if is_openrouter else "openai",
        "supportsLongCacheRetention": not (
            is_together or is_cf_workers or is_cf_gateway or is_nvidia or is_ant_ling
        ),
    }


def resolve_compat(model: ModelInfo, base_url: str) -> dict[str, Any]:
    """Pi getCompat: explicit `model.compat` values over detected ones.

    Unknown keys are ignored, as for the other providers' compat records, but a known
    flag with a value of the wrong type or outside Pi's set is a configuration error.
    """
    explicit = model.compat
    for name in _BOOL_FLAGS:
        if explicit.get(name) is not None and type(explicit[name]) is not bool:
            raise ConfigurationError(f"compat.{name} must be a boolean")
    choices: list[tuple[str, set[str]]] = [
        ("thinkingFormat", _THINKING_FORMATS),
        ("maxTokensField", {"max_completion_tokens", "max_tokens"}),
        ("thinkingTokenBudgetField", _BUDGET_FIELDS),
        ("sessionAffinityFormat", _AFFINITY_FORMATS),
        ("cacheControlFormat", {"anthropic"}),
    ]
    for name, allowed in choices:
        if explicit.get(name) is not None and explicit[name] not in allowed:
            raise ConfigurationError(f"compat.{name} must be one of {sorted(allowed)}")
    for name in ("chatTemplateKwargs", "chatTemplateArgs", "openRouterRouting"):
        if explicit.get(name) is not None and not isinstance(explicit[name], dict):
            raise ConfigurationError(f"compat.{name} must be an object")
    if explicit.get("vllmPriority") is not None and not _number(explicit["vllmPriority"]):
        raise ConfigurationError("compat.vllmPriority must be a number")
    detected = detect_compat(model.provider, base_url, model.id)
    resolved = {
        name: explicit[name] if explicit.get(name) is not None else value
        for name, value in detected.items()
    }
    resolved["openRouterRouting"] = _first_defined(explicit.get("openRouterRouting"), {})
    resolved["vllmPriority"] = explicit.get("vllmPriority")
    return resolved


def _is_detail(detail: Any) -> bool:
    """Pi isOpenAIReasoningDetail."""
    if not isinstance(detail, dict):
        return False
    if not (detail.get("id") is None or isinstance(detail["id"], str)):
        return False
    if "format" in detail and not isinstance(detail["format"], str):
        return False
    if "index" in detail and not _number(detail["index"]):
        return False
    kind = detail.get("type")
    if kind == "reasoning.summary":
        return isinstance(detail.get("summary"), str)
    if kind == "reasoning.encrypted":
        return isinstance(detail.get("data"), str)
    if kind == "reasoning.text":
        return isinstance(detail.get("text"), str) and (
            detail.get("signature") is None or isinstance(detail["signature"], str)
        )
    return False


def _parse_details(signature: str | None) -> list[dict[str, Any]] | None:
    if not signature:
        return None
    try:
        parsed = json.loads(signature)
    except ValueError:
        return None
    if isinstance(parsed, list) and parsed and all(_is_detail(d) for d in parsed):
        return parsed
    return None


def _parse_legacy_detail(signature: str | None) -> dict[str, Any] | None:
    if not signature:
        return None
    try:
        parsed = json.loads(signature)
    except ValueError:
        return None
    if (
        _is_detail(parsed)
        and parsed["type"] == "reasoning.encrypted"
        and isinstance(parsed.get("id"), str)
        and parsed["id"]
        and parsed["data"]
    ):
        return dict(parsed)
    return None


def _assign_from(target: dict[str, Any], source: dict[str, Any], key: str) -> None:
    """JavaScript `target[key] = source[key]`, where an undefined value is not serialized."""
    if key in source:
        target[key] = source[key]
    else:
        target.pop(key, None)


def _append_detail(details: list[dict[str, Any]], detail: dict[str, Any]) -> None:
    """Pi appendOpenAIReasoningDetail: merge streamed text/summary pieces into entries."""
    last = details[-1] if details else None
    if last is not None and detail["type"] == last["type"] in {
        "reasoning.text",
        "reasoning.summary",
    }:
        field = "text" if detail["type"] == "reasoning.text" else "summary"
        last[field] += detail[field]
        if field == "text" and not _truthy(last.get("signature")):
            _assign_from(last, detail, "signature")
        if last.get("id") is None:
            _assign_from(last, detail, "id")
        if not _truthy(last.get("format")):
            _assign_from(last, detail, "format")
        if last.get("index") is None:
            _assign_from(last, detail, "index")
        return
    details.append(dict(detail))


def _has_usage(raw: Any) -> bool:
    """Empty/partial gateway counters are unknown, including truthy JSON objects."""
    return isinstance(raw, dict) and all(
        isinstance(value := raw.get(key), (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
        and value >= 0
        for key in ("prompt_tokens", "completion_tokens")
    )


def _usage(raw: Any) -> dict[str, Any]:
    """Pi parseChunkUsage, without cost (prices live on ModelInfo)."""
    raw = raw if isinstance(raw, dict) else {}

    def count(value: Any) -> int | float:
        return value if _number(value) and value == value else 0

    prompt = raw.get("prompt_tokens_details")
    prompt = prompt if isinstance(prompt, dict) else {}
    completion = raw.get("completion_tokens_details")
    completion = completion if isinstance(completion, dict) else {}
    read = count(
        _first_defined(
            prompt.get("cached_tokens"),
            raw.get("prompt_cache_hit_tokens"),
            raw.get("cached_tokens"),
        )
    )
    write = count(prompt.get("cache_write_tokens"))
    input_tokens = max(0, count(raw.get("prompt_tokens")) - read - write)
    output = count(raw.get("completion_tokens"))
    return {
        "input": input_tokens,
        "output": output,
        "cache_read": read,
        "cache_write": write,
        "reasoning": count(completion.get("reasoning_tokens")),
        "total_tokens": input_tokens + output + read + write,
    }


def _stop_reason(reason: Any) -> tuple[str, str | None]:
    """Pi mapStopReason, with Python stop-reason spelling."""
    if reason in {"stop", "end"}:
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in {"function_call", "tool_calls"}:
        return "tool_use", None
    return "error", f"Provider finish_reason: {reason}"


def _image_part(block: ImageContent) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
    }


def _id_chars(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


class OpenAICompletionsProvider(RemoteProvider):
    """Any OpenAI-compatible `/chat/completions` endpoint.

    `base_url` is the API root that precedes `/chat/completions`, used as given (for
    example `http://localhost:11434/v1` for Ollama, `http://localhost:8000/v1` for
    vLLM). `name` is the provider identity recorded on messages; Pi also uses it, with
    the URL, to detect known services. The API key is optional: without one no
    `authorization` header is sent, which keyless local servers accept.

    Models are never guessed. Pass a `ModelInfo` with `api="openai-completions"` and
    `provider=name`, register one in `catalog`, or call `provider.model(id, ...)`.
    `options` accepts max_tokens, reasoning, thinking_budgets, temperature,
    tool_choice, sampling_params (merged into the body last), cache_retention,
    session_id, headers (a None value removes a header) and transport ("sse").
    """

    api = API

    def __init__(self, *, base_url: str, name: str = "openai-compatible", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ConfigurationError(
                "base_url must be an http(s) URL, such as http://localhost:11434/v1"
            )
        if not isinstance(name, str) or not name:
            raise ConfigurationError("name must be a non-empty string")
        self.name = name
        self.base_url = base_url.rstrip("/")

    def model(
        self,
        id: str,
        *,
        context_window: int = 128_000,
        max_tokens: int = 16_384,
        reasoning: bool = False,
        input: tuple[str, ...] | list[str] = ("text",),
        name: str | None = None,
        thinking_level_map: dict[str, str | None] | None = None,
        compat: dict[str, Any] | None = None,
        cost: dict[str, float] | None = None,
    ) -> ModelInfo:
        """Declare a model served by this endpoint.

        The defaults are Pi's for a custom model declared with only an `id` (coding-agent
        models.json): a text-only, non-reasoning model with a 128000-token context and
        16384 output tokens. State the real limits when you know them; the output limit
        is sent with every request. For a reasoning model on Ollama, vLLM or SGLang, Pi
        suggests `compat={"supportsDeveloperRole": False, "supportsReasoningEffort": False}`.
        """
        model = ModelInfo(
            id=id,
            provider=self.name,
            api=API,
            name=name or id,
            context_window=context_window,
            max_tokens=max_tokens,
            reasoning=reasoning,
            input=tuple(input),
            thinking_level_map=dict(thinking_level_map or {}),
            cost=dict(cost or {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}),
            compat=deepcopy(compat or {}),
            base_url=self.base_url,
        )
        ModelCatalog([model])  # the catalog's checks: positive limits, JSON compat, levels
        return model

    def model_info(self, request: ModelRequest) -> ModelInfo:
        model = request.model_info or self.catalog.get(self.name, request.model)
        if model is None:
            raise ConfigurationError(
                f"Unknown model {self.name}/{request.model}. Declare it, for example "
                f"Agent(provider=provider, model=provider.model({request.model!r}, "
                "context_window=..., max_tokens=...)), or register a "
                f"ModelInfo(provider={self.name!r}, api={API!r}) in the provider catalog"
            )
        if model.provider != self.name or model.id != request.model:
            raise ConfigurationError(
                f"Model {model.provider}/{model.id} is not {self.name}/{request.model}"
            )
        if model.api != API:
            raise ConfigurationError(
                f"Model {model.provider}/{model.id} uses api {model.api!r}, not {API!r}; "
                "declare it with provider.model(...) or dataclasses.replace(model, api=...)"
            )
        model.validate_request(request)
        return deepcopy(model)

    async def _key(self, request: ModelRequest, cancel: CancelToken) -> str | None:
        """A configured credential, or None for a keyless endpoint (an extension: Pi
        requires a key or an authorization header)."""
        if request.api_key is None and self.api_key is None and self.credentials is None:
            return None
        return await self.credential(request, cancel)

    def _tools(self, tools: list[ToolDeclaration], compat: dict[str, Any]) -> list[dict[str, Any]]:
        """Pi convertTools. Tool declarations carry no constrained-sampling settings, so
        `strict` is false whenever the endpoint accepts the field."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": deepcopy(t.input_schema),
                    **({"strict": False} if compat["supportsStrictMode"] is not False else {}),
                },
            }
            for t in tools
        ]

    def _messages(
        self, model: ModelInfo, transcript: list[Message], compat: dict[str, Any], folded: int
    ) -> list[dict[str, Any]]:
        """Pi convertMessages for a resolved transcript."""

        def normalize(value: str, _source: AssistantMessage) -> str:
            if "|" in value:
                # Responses IDs are `call|item`; keep item-level uniqueness within 40 chars.
                call, _, item = value.partition("|")
                call, item = _id_chars(call), _id_chars(item)
                combined = f"{call}_{item}" if item else call
                if len(combined) <= 40:
                    return combined
                digest = short_hash(value)[:8]
                return f"{call[: max(1, 40 - len(digest) - 1)]}_{digest}"
            if model.provider == "openai":
                return value[:40]
            return value

        transformed = transform_messages(transcript, self.name, API, model.id, normalize)
        role = "developer" if model.reasoning and compat["supportsDeveloperRole"] else "system"
        bridge = compat["requiresAssistantAfterToolResult"]
        params: list[dict[str, Any]] = []
        last_role: str | None = None
        i = 0
        system_index = 0
        while i < len(transformed):
            message = transformed[i]
            if bridge and last_role == "toolResult" and isinstance(message, UserMessage):
                params.append({"role": "assistant", "content": _BRIDGE})
            if isinstance(message, SystemMessage):
                added = message.tools_added if system_index >= folded else []
                system_index += 1
                if added:
                    # Kimi-style tool addition anchored at its system message.
                    params.append({"role": "system", "tools": self._tools(added, compat)})
                text = system_message_text(message) if i == 0 else render_system_update(message)
                if text:
                    params.append({"role": role, "content": _sanitize(text)})
                last_role = "system"
            elif isinstance(message, UserMessage):
                if isinstance(message.content, str):
                    params.append({"role": "user", "content": _sanitize(message.content)})
                else:
                    parts: list[dict[str, Any]] = []
                    for b in message.content:
                        if isinstance(b, ImageContent):
                            parts.append(_image_part(b))
                        elif b.text:
                            parts.append({"type": "text", "text": _sanitize(b.text)})
                    if not parts:
                        i += 1
                        continue
                    params.append({"role": "user", "content": parts})
                last_role = "user"
            elif isinstance(message, AssistantMessage):
                item = self._assistant(model, message, compat)
                if item is None:
                    i += 1
                    continue
                params.append(item)
                last_role = "assistant"
            elif isinstance(message, ToolResultMessage):
                images: list[dict[str, Any]] = []
                j = i
                while j < len(transformed):
                    result = transformed[j]
                    if not isinstance(result, ToolResultMessage):
                        break
                    text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
                    has_images = any(isinstance(b, ImageContent) for b in result.content)
                    body = (
                        text
                        if not _blank(text)
                        else ("(see attached image)" if has_images else "(no tool output)")
                    )
                    entry: dict[str, Any] = {
                        "role": "tool",
                        "content": _sanitize(body),
                        "tool_call_id": result.call_id,
                    }
                    if compat["requiresToolResultName"] and result.name:
                        entry["name"] = result.name
                    params.append(entry)
                    if has_images and "image" in model.input:
                        images += [
                            _image_part(b) for b in result.content if isinstance(b, ImageContent)
                        ]
                    j += 1
                i = j
                if images:
                    if bridge:
                        params.append({"role": "assistant", "content": _BRIDGE})
                    params.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Attached image(s) from tool result:"},
                                *images,
                            ],
                        }
                    )
                    last_role = "user"
                else:
                    last_role = "toolResult"
                continue
            elif isinstance(message, CustomMessage):
                raise UnsupportedCapabilityError("Convert custom messages before provider boundary")
            i += 1
        return params

    def _assistant(
        self, model: ModelInfo, message: AssistantMessage, compat: dict[str, Any]
    ) -> dict[str, Any] | None:
        item: dict[str, Any] = {
            "role": "assistant",
            "content": "" if compat["requiresAssistantAfterToolResult"] else None,
        }
        text_parts = [
            {"type": "text", "text": _sanitize(b.text)}
            for b in message.content
            if isinstance(b, TextContent) and not _blank(b.text)
        ]
        text = "".join(p["text"] for p in text_parts)
        thinking = [b for b in message.content if isinstance(b, ThinkingContent)]
        calls = message.tool_calls
        signed = next(
            (d for d in (_parse_details(b.thinking_signature) for b in thinking) if d is not None),
            None,
        )
        legacy = [d for d in (_parse_legacy_detail(c.thought_signature) for c in calls) if d]
        preserved = signed if signed is not None else (legacy or None)
        visible = [b for b in thinking if not _blank(b.thinking)]
        if visible:
            if compat["requiresThinkingAsText"]:
                joined = "\n\n".join(_sanitize(b.thinking) for b in visible)
                item["content"] = [{"type": "text", "text": joined}, *text_parts]
            else:
                # Assistant content is always a plain string (some models mirror arrays).
                if text:
                    item["content"] = text
                if not preserved:
                    field = visible[0].thinking_signature
                    if model.provider == "opencode-go" and field == "reasoning":
                        field = "reasoning_content"
                    if field in _REASONING_FIELDS:
                        item[field] = "\n".join(b.thinking for b in visible)
        elif text:
            item["content"] = text
        if calls:
            item["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": json.dumps(
                            c.arguments, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                }
                for c in calls
            ]
        if preserved:
            item["reasoning_details"] = preserved
        if (
            compat["requiresReasoningContentOnAssistantMessages"]
            and model.reasoning
            and "reasoning_content" not in item
        ):
            item["reasoning_content"] = ""
        # Providers reject an assistant message with neither content nor tool calls.
        if not item["content"] and not calls:
            return None
        return item

    def build_request(self, request: ModelRequest) -> dict[str, Any]:
        """The request body (before `on_payload`), as Pi's streamSimple builds it."""
        model = self.model_info(request)
        return self._build(request, model, resolve_compat(model, self.base_url))

    def _build(
        self, request: ModelRequest, model: ModelInfo, compat: dict[str, Any]
    ) -> dict[str, Any]:
        options = request.options
        retention = options.get("cache_retention", "short")
        if retention not in {"none", "short", "long"}:
            raise ConfigurationError("Invalid cache_retention")
        reasoning = options.get("reasoning")
        if reasoning is not None and reasoning not in LEVELS:
            raise ConfigurationError("Unsupported reasoning level")
        # streamSimple: clamp the level to the model; "off" sends no effort.
        effort = model.clamp_thinking_level(reasoning) if reasoning else None
        if effort == "off":
            effort = None
        source = with_request_tools(request.messages, request.tools)
        transcript = resolve_transcript(source, compat["supportsMidConvoSystemMessages"] is True)
        tools, folded = _resolve_tool_changes(
            transcript,
            "additions"
            if compat["supportsMidConvoSystemMessages"] is True
            and compat["supportsMidConvoToolAdditions"] is True
            else "none",
        )
        messages = self._messages(model, transcript, compat, folded)
        long_cache = retention == "long" and compat["supportsLongCacheRetention"]
        body: dict[str, Any] = {"model": model.id, "messages": messages, "stream": True}
        session = options.get("session_id")
        if session is not None and (
            ("api.openai.com" in self.base_url and retention != "none") or long_cache
        ):
            body["prompt_cache_key"] = session[:64]
        if long_cache:
            body["prompt_cache_retention"] = "24h"
        if compat["supportsUsageInStreaming"] is not False:
            body["stream_options"] = {"include_usage": True}
        if compat["supportsStore"]:
            body["store"] = False
        max_tokens = clamp_max_tokens_to_context(
            model.context_window, source, options.get("max_tokens", model.max_tokens)
        )
        if max_tokens:
            body[compat["maxTokensField"]] = max_tokens
        if options.get("temperature") is not None:
            body["temperature"] = options["temperature"]
        if tools:
            body["tools"] = self._tools(tools, compat)
            if compat["zaiToolStream"]:
                body["tool_stream"] = True
        elif any(
            isinstance(m, ToolResultMessage) or (isinstance(m, AssistantMessage) and m.tool_calls)
            for m in transcript
        ):
            # Anthropic behind LiteLLM-style proxies needs `tools` once tools were used.
            body["tools"] = []
        if compat["cacheControlFormat"] == "anthropic" and retention != "none":
            cache = {"type": "ephemeral", **({"ttl": "1h"} if long_cache else {})}
            _apply_cache_control(messages, body.get("tools"), cache)
        if _truthy(options.get("tool_choice")):
            body["tool_choice"] = deepcopy(options["tool_choice"])
        if compat["vllmPriority"] is not None:
            body["priority"] = compat["vllmPriority"]
        budget = self._thinking_budget(model, effort, options, body)
        self._thinking(body, model, compat, effort, budget)
        field = compat["thinkingTokenBudgetField"] or (
            "thinking_token_budget" if compat["supportsThinkingTokenBudget"] else None
        )
        if field and budget is not None:
            body[field] = budget
        routing = model.compat.get("openRouterRouting")
        if _truthy(routing):
            body["provider"] = deepcopy(routing)
        gateway = model.compat.get("vercelGatewayRouting")
        if _truthy(gateway) and isinstance(gateway, dict):
            selected = {
                k: deepcopy(gateway[k]) for k in ("only", "order") if _truthy(gateway.get(k))
            }
            if selected:
                body["providerOptions"] = {"gateway": selected}
        # Last, so explicit sampling keys override the named fields (Pi samplingParams).
        sampling = options.get("sampling_params")
        if sampling is not None:
            if not isinstance(sampling, dict):
                raise ConfigurationError("sampling_params must be an object")
            body.update(deepcopy(sampling))
        return body

    @staticmethod
    def _thinking_budget(
        model: ModelInfo, effort: str | None, options: dict[str, Any], body: dict[str, Any]
    ) -> int | None:
        """Pi resolveClampedThinkingBudget: leave room for an answer under the ceiling."""
        if not effort or not model.reasoning:
            return None
        ceiling = _first_defined(
            body.get("max_tokens"), body.get("max_completion_tokens"), model.max_tokens
        )
        budgets = {**_DEFAULT_BUDGETS, **(options.get("thinking_budgets") or {})}
        level = "high" if effort in {"xhigh", "max"} else effort
        budget = budgets[level]
        if type(budget) is not int or budget < 0:
            raise ConfigurationError("Invalid thinking budget")
        budget = min(budget, max(0, ceiling - _MIN_ANSWER_TOKENS))
        return budget if budget > 0 else None

    @staticmethod
    def _thinking(
        body: dict[str, Any],
        model: ModelInfo,
        compat: dict[str, Any],
        effort: str | None,
        budget: int | None,
    ) -> None:
        """Pi's thinkingFormat branches, in upstream order."""
        level_map = model.thinking_level_map
        kind = compat["thinkingFormat"]
        effort_ok = compat["supportsReasoningEffort"]
        off_allowed = not ("off" in level_map and level_map["off"] is None)

        def mapped(level: str) -> str:
            # `thinkingLevelMap[level] ?? level`
            value = level_map.get(level)
            return value if value is not None else level

        def strict(level: str | None) -> Any:
            # `map[level] === undefined ? level : map[level]`, kept only when a string.
            key = level if level else "off"
            value = level_map[key] if key in level_map else level
            return value if isinstance(value, str) else _OMIT

        if kind == "zai" and model.reasoning:
            body["thinking"] = (
                {"type": "enabled", "clear_thinking": False} if effort else {"type": "disabled"}
            )
            if effort and effort_ok and (value := strict(effort)) is not _OMIT:
                body["reasoning_effort"] = value
        elif kind == "qwen" and model.reasoning:
            body["enable_thinking"] = bool(effort)
            if effort and effort_ok:
                body["reasoning_effort"] = mapped(effort)
        elif kind == "qwen-chat-template" and model.reasoning:
            body["chat_template_kwargs"] = {
                "enable_thinking": bool(effort),
                "preserve_thinking": True,
            }
        elif kind == "chat-template" and model.reasoning:
            values = _template_values(model, effort, compat["chatTemplateKwargs"], budget)
            if values:
                body["chat_template_kwargs"] = values
        elif kind == "baseten" and model.reasoning:
            values = _template_values(model, effort, compat["chatTemplateArgs"], budget)
            if values:
                body["chat_template_args"] = values
            if effort_ok and (value := strict(effort)) is not _OMIT:
                body["reasoning_effort"] = value
        elif kind == "deepseek" and model.reasoning:
            if effort:
                body["thinking"] = {"type": "enabled"}
            elif off_allowed:
                body["thinking"] = {"type": "disabled"}
            if effort and effort_ok:
                body["reasoning_effort"] = mapped(effort)
        elif kind == "openrouter" and model.reasoning:
            if effort:
                body["reasoning"] = {"effort": mapped(effort)}
            elif off_allowed:
                body["reasoning"] = {"effort": _first_defined(level_map.get("off"), "none")}
        elif kind == "ant-ling" and model.reasoning and effort:
            if isinstance(level_map.get(effort), str):
                body["reasoning"] = {"effort": level_map[effort]}
        elif kind == "together" and model.reasoning:
            body["reasoning"] = {"enabled": bool(effort)}
            if effort and effort_ok:
                body["reasoning_effort"] = mapped(effort)
        elif kind == "string-thinking" and model.reasoning:
            if effort:
                body["thinking"] = mapped(effort)
            elif off_allowed:
                body["thinking"] = _first_defined(level_map.get("off"), "none")
        elif effort and model.reasoning and effort_ok:
            body["reasoning_effort"] = mapped(effort)
        elif not effort and model.reasoning and effort_ok:
            if isinstance(level_map.get("off"), str):
                body["reasoning_effort"] = level_map["off"]

    def _headers(
        self, request: ModelRequest, key: str | None, compat: dict[str, Any]
    ) -> dict[str, str | None]:
        options = request.options
        headers: dict[str, str] = {
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        if key is not None:
            headers["authorization"] = f"Bearer {key}"
        session = options.get("session_id")
        if (
            session
            and options.get("cache_retention", "short") != "none"
            and compat["sendSessionAffinityHeaders"]
        ):
            if compat["sessionAffinityFormat"] == "openrouter":
                headers["x-session-id"] = session
            else:
                if compat["sessionAffinityFormat"] == "openai":
                    headers["session_id"] = session
                headers["x-client-request-id"] = session
                headers["x-session-affinity"] = session
        # Request headers override generated ones, credentials included (as in Pi).
        for name, value in (options.get("headers") or {}).items():
            headers.pop(name.lower(), None)
            if value is not None:
                headers[name.lower()] = value
        return apply_identity(headers, self.name, self.base_url, options)

    @event_contract
    async def stream(
        self, request: ModelRequest, cancel: CancelToken
    ) -> AsyncGenerator[ModelEvent, None]:
        request_started = now()
        if request.options.get("transport", "sse") not in {"sse", "auto"}:
            raise UnsupportedCapabilityError("Chat Completions supports SSE transport")
        key = await self._key(request, cancel)
        model = self.model_info(request)
        compat = resolve_compat(model, self.base_url)
        body = await self.payload(request, self._build(request, model, compat))
        headers = self._headers(request, key, compat)
        events = self.transport.stream(
            self.base_url + "/chat/completions",
            body,
            headers,
            cancel,
            on_response=request.on_response,
            **self.transport_options(request),
        )
        content: list[TextContent | ThinkingContent | ToolCall] = []
        # Pi streams one text and one reasoning block per response, in first-seen order.
        text: TextContent | None = None
        text_at = thinking_at = -1
        thinking: ThinkingContent | None = None
        # Tool calls are matched by stream index, else by ID, as in Pi.
        by_index: dict[int, int] = {}
        by_id: dict[str, int] = {}
        stream_indexes: dict[int, int | None] = {}
        arguments: dict[int, str] = {}
        details: list[dict[str, Any]] | None = None
        stop = "pending"
        error: str | None = None
        raw_stop: str | None = None
        finished = False
        usage = _usage({})
        usage_available = False
        response_id: str | None = None
        response_model: str | None = None
        announced = ended = False
        try:
            async for chunk in events:
                if not announced:
                    announced = True
                    yield ModelEvent("start")
                if ended:
                    continue  # the OpenAI SDK ignores data after [DONE]
                if chunk.get("type") == "transport_done":
                    ended = True
                    continue
                if _truthy(chunk.get("error")):
                    raise stream_error(chunk, headers)
                await invoke(request.on_provider_stream_event, deepcopy(chunk))
                if not response_id and isinstance(chunk.get("id"), str):
                    response_id = chunk["id"] or None
                served = chunk.get("model")
                if isinstance(served, str) and served and served != model.id:
                    response_model = response_model or served
                choices = chunk.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else None
                # Prefer complete top-level counters, then choice-level counters
                # (Moonshot). Empty/partial trailing chunks cannot erase a known
                # measurement. Preserve normalized partial fields for direct callers.
                candidates = [chunk.get("usage")]
                if isinstance(choice, dict):
                    candidates.append(choice.get("usage"))
                measured = next((raw for raw in candidates if _has_usage(raw)), None)
                if measured is not None:
                    usage = _usage(measured)
                    usage_available = True
                elif not usage_available:
                    partial = next(
                        (raw for raw in candidates if isinstance(raw, dict) and raw), None
                    )
                    if partial is not None:
                        usage = _usage(partial)
                if not isinstance(choice, dict):
                    continue
                if _truthy(choice.get("finish_reason")):
                    raw_stop = str(choice["finish_reason"])
                    stop, message = _stop_reason(choice["finish_reason"])
                    error = message or error
                    finished = True
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                # llama.cpp uses reasoning_content, others reasoning; the first wins.
                field = next(
                    (f for f in _REASONING_FIELDS if isinstance(delta.get(f), str) and delta[f]),
                    None,
                )
                if field is not None:
                    if thinking is None:
                        signature = (
                            "reasoning_content"
                            if model.provider == "opencode-go" and field == "reasoning"
                            else field
                        )
                        thinking, thinking_at = ThinkingContent("", signature), len(content)
                        content.append(thinking)
                        yield ModelEvent.boundary("start", thinking_at, ThinkingContent(""))
                    thinking.thinking += delta[field]
                    yield ModelEvent.thinking(delta[field], thinking_at)
                piece = delta.get("content")
                if piece is not None and not isinstance(piece, str):
                    raise UnsupportedCapabilityError("Chat Completions delta content must be text")
                if piece:
                    if text is None:
                        text, text_at = TextContent(""), len(content)
                        content.append(text)
                        yield ModelEvent.boundary("start", text_at, TextContent(""))
                    text.text += piece
                    yield ModelEvent.text(piece, text_at)
                calls = delta.get("tool_calls")
                if _truthy(calls):
                    if not isinstance(calls, list):
                        raise ProviderProtocolError("tool_calls delta must be a list")
                    for call in calls:
                        if not isinstance(call, dict):
                            raise ProviderProtocolError("tool call delta must be an object")
                        function = call.get("function")
                        if not isinstance(function, dict):
                            if call.get("custom") is not None:
                                raise UnsupportedCapabilityError(
                                    "Grammar (custom) tool calls are not supported"
                                )
                            function = {}
                        stream_index = call.get("index") if _number(call.get("index")) else None
                        call_id = call.get("id") if isinstance(call.get("id"), str) else ""
                        position = by_index.get(stream_index) if stream_index is not None else None
                        if position is None and call_id:
                            position = by_id.get(call_id)
                        if position is None:
                            name = function.get("name")
                            if not isinstance(name, str) or not name:
                                raise ProviderProtocolError(
                                    "Tool call delta without a function name"
                                )
                            # A call without an ID still needs one for its result (Pi keeps "").
                            block = ToolCall(call_id or f"call_{uuid.uuid4().hex[:24]}", name, {})
                            position = len(content)
                            content.append(block)
                            arguments[position] = ""
                            stream_indexes[position] = stream_index
                            if stream_index is not None:
                                by_index[stream_index] = position
                            yield ModelEvent.boundary("start", position, deepcopy(block))
                        if stream_index is not None and stream_indexes[position] is None:
                            stream_indexes[position] = stream_index
                            by_index[stream_index] = position
                        if call_id:
                            by_id[call_id] = position
                        piece = function.get("arguments")
                        if _truthy(piece):
                            if not isinstance(piece, str):
                                raise ProviderProtocolError("Tool call arguments must be a string")
                            arguments[position] += piece
                        else:
                            piece = ""
                        yield ModelEvent.toolcall(piece, position)
                found = delta.get("reasoning_details")
                if isinstance(found, list):
                    for detail in found:
                        if not _is_detail(detail):
                            continue
                        if thinking is None:
                            thinking, thinking_at = ThinkingContent("", ""), len(content)
                            content.append(thinking)
                            yield ModelEvent.boundary("start", thinking_at, ThinkingContent(""))
                        # Replay metadata, not visible deltas: merged and kept as the signature.
                        details = details if details is not None else []
                        _append_detail(details, detail)
        finally:
            await events.aclose()
        for position, item in enumerate(content):
            if isinstance(item, ThinkingContent) and details is not None:
                item.thinking_signature = json.dumps(
                    details, ensure_ascii=False, separators=(",", ":")
                )
            elif isinstance(item, ToolCall):
                raw = arguments[position]
                if _blank(raw):
                    # A call without arguments; Pi parses "" as {}.
                    yield ModelEvent.toolcall("{}", position)
                    raw = "{}"
                item.arguments, item.arguments_error = parse_tool_call_arguments(raw)
            yield ModelEvent.boundary("end", position, deepcopy(item))
        cancel.raise_if_cancelled()
        if not finished and not compat["supportsFinishReason"]:
            stop = "tool_use" if any(isinstance(b, ToolCall) for b in content) else "stop"
        if stop == "error":
            raise ProviderProtocolError(error or "Provider returned an error stop reason")
        if (compat["supportsFinishReason"] and not finished) or stop == "pending":
            raise ProviderProtocolError("Stream ended without finish_reason")
        yield ModelEvent.done(
            AssistantMessage(
                content,
                stop,
                self.name,
                request.model,
                usage,
                timestamp=request_started,
                api=API,
                diagnostics=None if usage_available else [{"type": "usage_unavailable"}],
                thinking_level=request.options.get("reasoning"),
                response_id=response_id,
                response_model=response_model,
                raw_stop_reason=raw_stop,
            )
        )


def _template_values(
    model: ModelInfo, effort: str | None, values: dict[str, Any], budget: int | None
) -> dict[str, Any] | None:
    """Pi buildChatTemplateValues: literal values, or `{"$var": ...}` thinking values."""
    result: dict[str, Any] = {}
    level_map = model.thinking_level_map
    for key, value in values.items():
        if not isinstance(value, dict):
            result[key] = value
            continue
        if not effort and value.get("omitWhenOff"):
            continue
        variable = value.get("$var")
        if variable == "thinking.enabled":
            result[key] = bool(effort)
        elif variable == "thinking.budget":
            if budget is not None:
                result[key] = budget
        else:
            level = effort if effort else "off"
            resolved = level_map[level] if level in level_map else effort
            if isinstance(resolved, str):
                result[key] = resolved
    return result or None


def _cache_text(message: dict[str, Any], cache: dict[str, Any]) -> bool:
    """Pi addCacheControlToTextContent."""
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [{"type": "text", "text": content, "cache_control": deepcopy(cache)}]
        return True
    if not isinstance(content, list):
        return False
    for part in reversed(content):
        if isinstance(part, dict) and part.get("type") == "text":
            part["cache_control"] = deepcopy(cache)
            return True
    return False


def _apply_cache_control(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, cache: dict[str, Any]
) -> None:
    """Pi applyAnthropicCacheControl: system prompt, last tool, last conversation text."""
    for message in messages:
        if message["role"] in {"system", "developer"}:
            _cache_text(message, cache)
            break
    if tools:
        tools[-1]["cache_control"] = deepcopy(cache)
    for message in reversed(messages):
        if message["role"] in {"user", "assistant", "tool"} and _cache_text(message, cache):
            break
