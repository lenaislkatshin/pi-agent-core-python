"""Versioned, JSON-only messages. All boundaries copy caller-owned values."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

from .errors import MessageValidationError, UnsupportedCapabilityError
import base64

JsonValue: TypeAlias = "None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]"
SCHEMA_VERSION = 4


def now() -> float:
    return time.time()


def validate_json(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            validate_json(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            validate_json(item)
        return
    raise MessageValidationError(f"Not a finite JSON value: {type(value).__name__}")


@dataclass
class TextContent:
    text: str
    text_signature: str | None = None
    type: Literal["text"] = field(default="text", init=False)


@dataclass
class ImageContent:
    data: str
    mime_type: str
    type: Literal["image"] = field(default="image", init=False)


@dataclass
class ThinkingContent:
    thinking: str
    thinking_signature: str | None = None
    redacted: bool = False
    type: Literal["thinking"] = field(default="thinking", init=False)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None
    namespace: str | None = None
    arguments_error: str | None = None
    type: Literal["tool_call"] = field(default="tool_call", init=False)


@dataclass
class ToolDeclaration:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class SystemMessage:
    content: str | list[TextContent] = ""
    sections: dict[str, str | None] = field(default_factory=dict)
    tools_added: list[ToolDeclaration] = field(default_factory=list)
    tools_removed: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=now)
    role: Literal["system"] = field(default="system", init=False)


@dataclass
class UserMessage:
    content: str | list[TextContent | ImageContent]
    timestamp: float = field(default_factory=now)
    role: Literal["user"] = field(default="user", init=False)


@dataclass
class AssistantMessage:
    content: list[TextContent | ThinkingContent | ToolCall]
    stop_reason: str = "stop"
    provider: str = "mock"
    model: str = "mock"
    usage: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=now)
    error: str | None = None
    api: str | None = None
    provider_thinking_level: str | None = None
    response_model: str | None = None
    response_id: str | None = None
    thinking_level: str | None = None
    diagnostics: list[dict[str, Any]] | None = None
    raw_stop_reason: str | None = None
    end_turn: bool | None = None
    deferred: dict[str, Any] | None = None
    duration_ms: int | None = None
    role: Literal["assistant"] = field(default="assistant", init=False)

    @classmethod
    def text(cls, text: str, **kwargs: Any) -> AssistantMessage:
        return cls([TextContent(text)], **kwargs)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.content if isinstance(b, ToolCall)]


@dataclass
class ToolResultMessage:
    call_id: str
    name: str
    content: list[TextContent | ImageContent]
    is_error: bool = False
    timestamp: float = field(default_factory=now)
    details: Any = None
    usage: dict[str, Any] | None = None
    nested_calls: dict[str, Any] | None = None
    duration_ms: int | None = None
    role: Literal["tool_result"] = field(default="tool_result", init=False)


@dataclass
class CustomMessage:
    custom_type: str
    data: Any
    timestamp: float = field(default_factory=now)
    role: Literal["custom"] = field(default="custom", init=False)


Message: TypeAlias = (
    SystemMessage | UserMessage | AssistantMessage | ToolResultMessage | CustomMessage
)


def message_to_dict(message: Message) -> dict[str, Any]:
    if not isinstance(
        message, (SystemMessage, UserMessage, AssistantMessage, ToolResultMessage, CustomMessage)
    ):
        raise MessageValidationError("Expected a supported Message")
    optional_fields = {
        "text_signature",
        "thought_signature",
        "namespace",
        "arguments_error",
        "api",
        "provider_thinking_level",
        "response_model",
        "response_id",
        "thinking_level",
        "diagnostics",
        "raw_stop_reason",
        "end_turn",
        "deferred",
        "details",
        "usage",
        "nested_calls",
        "duration_ms",
    }
    # dict_factory visits dataclass fields, not arbitrary user dictionaries.
    data = asdict(
        message,
        dict_factory=lambda pairs: {
            k: v for k, v in pairs if not (k in optional_fields and v is None)
        },
    )
    validate_json(data)
    # Parsing validates semantic fields as well as JSON representability.
    message_from_dict(data)
    return data


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise MessageValidationError(f"{label} must be a string")
    return value


def _blocks(values: Any, tools: bool = False, images: bool = False) -> list:
    if not isinstance(values, list):
        raise MessageValidationError("content must be a list")
    result: list = []
    for b in values:
        if not isinstance(b, dict):
            raise MessageValidationError("content block must be an object")
        if b.get("type") == "text":
            if set(b) - {"type", "text", "text_signature"} or "text" not in b:
                raise MessageValidationError("Invalid text block fields")
            signature = b.get("text_signature")
            if signature is not None:
                _text(signature, "text_signature")
            result.append(TextContent(_text(b["text"], "text"), signature))
        elif b.get("type") == "image" and images:
            if set(b) != {"type", "data", "mime_type"}:
                raise MessageValidationError("Invalid image block")
            data, mime = _text(b["data"], "image data"), _text(b["mime_type"], "mime_type")
            if mime not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
                raise UnsupportedCapabilityError("Unsupported image MIME type")
            try:
                base64.b64decode(data, validate=True)
            except ValueError as exc:
                raise MessageValidationError("Invalid image base64") from exc
            result.append(ImageContent(data, mime))
        elif b.get("type") == "thinking" and tools:
            if set(b) - {"type", "thinking", "thinking_signature", "redacted"}:
                raise MessageValidationError("Invalid thinking block")
            signature = b.get("thinking_signature")
            if signature is not None:
                _text(signature, "thinking_signature")
            if type(b.get("redacted", False)) is not bool:
                raise MessageValidationError("redacted must be boolean")
            result.append(
                ThinkingContent(
                    _text(b["thinking"], "thinking"), signature, b.get("redacted", False)
                )
            )
        elif b.get("type") == "tool_call" and tools:
            if (
                set(b)
                - {
                    "type",
                    "id",
                    "name",
                    "arguments",
                    "thought_signature",
                    "namespace",
                    "arguments_error",
                }
                or not {"id", "name", "arguments"} <= b.keys()
            ):
                raise MessageValidationError("Invalid tool call fields")
            if not b["id"] or not b["name"] or not isinstance(b["arguments"], dict):
                raise MessageValidationError("Invalid tool call ID, name or arguments")
            for key in ("thought_signature", "namespace", "arguments_error"):
                if b.get(key) is not None:
                    _text(b[key], key)
            result.append(
                ToolCall(
                    _text(b["id"], "id"),
                    _text(b["name"], "name"),
                    b["arguments"],
                    b.get("thought_signature"),
                    b.get("namespace"),
                    b.get("arguments_error"),
                )
            )
        else:
            raise UnsupportedCapabilityError(f"Unsupported content: {b.get('type')}")
    return result


def message_from_dict(data: dict[str, Any]) -> Message:
    validate_json(data)
    if not isinstance(data, dict):
        raise MessageValidationError("Message must be an object")
    d = json.loads(json.dumps(data, allow_nan=False))
    role = d.pop("role", None)
    try:
        duration = d.get("duration_ms")
        if duration is not None and (type(duration) is not int or duration < 0):
            raise MessageValidationError("duration_ms must be a non-negative integer")
        if "timestamp" in d and (type(d["timestamp"]) not in (float, int)):
            raise MessageValidationError("timestamp must be a number")
        if role == "user":
            if not isinstance(d["content"], str):
                d["content"] = _blocks(d["content"], images=True)
            return UserMessage(**d)
        if role == "system":
            if isinstance(d.get("content", ""), str):
                _text(d.get("content", ""), "content")
            else:
                d["content"] = _blocks(d["content"])
            sections = d.get("sections", {})
            if not isinstance(sections, dict) or any(
                v is not None and not isinstance(v, str) for v in sections.values()
            ):
                raise MessageValidationError("Invalid sections")
            d["tools_added"] = [ToolDeclaration(**t) for t in d.get("tools_added", [])]
            for t in d["tools_added"]:
                _text(t.name, "tool name")
                _text(t.description, "description")
                if not isinstance(t.input_schema, dict):
                    raise MessageValidationError("Invalid tool schema")
            if not isinstance(d.get("tools_removed", []), list):
                raise MessageValidationError("Invalid tool removals")
            for name in d.get("tools_removed", []):
                _text(name, "tool removal")
            return SystemMessage(**d)
        if role == "assistant":
            d["content"] = _blocks(d["content"], tools=True)
            if d.get("stop_reason", "stop") not in {
                "stop",
                "tool_use",
                "length",
                "error",
                "aborted",
                "pending",
                "deferred",
            }:
                raise UnsupportedCapabilityError("Unsupported stop reason")
            calls = [b.id for b in d["content"] if isinstance(b, ToolCall)]
            if len(set(calls)) != len(calls):
                raise MessageValidationError("Duplicate tool call ID")
            for key in ("provider", "model"):
                _text(d.get(key, "mock"), key)
            for key in (
                "api",
                "provider_thinking_level",
                "response_model",
                "response_id",
                "thinking_level",
                "raw_stop_reason",
            ):
                if d.get(key) is not None:
                    _text(d[key], key)
            if not isinstance(d.get("usage", {}), dict):
                raise MessageValidationError("Invalid usage")
            if d.get("error") is not None:
                _text(d["error"], "error")
            if d.get("end_turn") is not None and type(d["end_turn"]) is not bool:
                raise MessageValidationError("end_turn must be boolean")
            if d.get("diagnostics") is not None and (
                not isinstance(d["diagnostics"], list)
                or any(not isinstance(v, dict) for v in d["diagnostics"])
            ):
                raise MessageValidationError("diagnostics must be a list of objects")
            if d.get("deferred") is not None:
                handle = d["deferred"]
                if not isinstance(handle, dict) or not all(
                    isinstance(handle.get(k), str) for k in ("provider", "model_id", "api", "id")
                ):
                    raise MessageValidationError("Invalid deferred handle")
            return AssistantMessage(**d)
        if role == "tool_result":
            d["content"] = _blocks(d["content"], images=True)
            _text(d["call_id"], "call_id")
            _text(d["name"], "name")
            if type(d.get("is_error", False)) is not bool:
                raise MessageValidationError("is_error must be boolean")
            if d.get("usage") is not None and not isinstance(d["usage"], dict):
                raise MessageValidationError("Invalid tool usage")
            if d.get("nested_calls") is not None:
                nested = d["nested_calls"]
                if (
                    not isinstance(nested, dict)
                    or type(nested.get("complete")) is not bool
                    or not isinstance(nested.get("calls"), list)
                ):
                    raise MessageValidationError("Invalid nested calls")
                for call in nested["calls"]:
                    if (
                        not isinstance(call, dict)
                        or not all(isinstance(call.get(k), str) for k in ("id", "name"))
                        or call.get("status") not in {"ok", "error", "unfinished"}
                    ):
                        raise MessageValidationError("Invalid nested call")
            return ToolResultMessage(**d)
        if role == "custom":
            _text(d["custom_type"], "custom_type")
            return CustomMessage(**d)
        raise UnsupportedCapabilityError(f"Unsupported role: {role}")
    except (TypeError, KeyError, ValueError) as exc:
        raise MessageValidationError(str(exc)) from exc


def convert_application_tool_batch(message: CustomMessage) -> UserMessage:
    """Project built-in application feedback, never local details or raw results.

    An allowlist also protects the default visibility boundary for imported records
    with extra fields. Applications can override conversion with convert_to_llm.
    """
    data = message.data
    if not isinstance(data, dict) or not isinstance(data.get("calls"), list):
        raise MessageValidationError("Invalid application tool batch")
    calls = []
    for call in data["calls"]:
        if not isinstance(call, dict):
            raise MessageValidationError("Invalid application tool call record")
        keys = (
            "id",
            "name",
            "arguments",
            "prepared_arguments",
            "origin",
            "source",
            "execution_status",
            "is_error",
            "error_code",
            "content",
        )
        try:
            visible = {key: call[key] for key in keys}
        except KeyError as exc:
            raise MessageValidationError("Incomplete application tool call record") from exc
        validate_json(visible)
        if (
            visible["origin"] != "application"
            or not isinstance(visible["id"], str)
            or not visible["id"]
            or not isinstance(visible["name"], str)
            or not visible["name"]
            or not isinstance(visible["arguments"], dict)
            or (
                visible["prepared_arguments"] is not None
                and not isinstance(visible["prepared_arguments"], dict)
            )
            or (visible["source"] is not None and not isinstance(visible["source"], str))
            or type(visible["is_error"]) is not bool
            or visible["execution_status"]
            not in ("not_started", "running", "succeeded", "failed", "cancelled", "unknown")
            or (visible["error_code"] is not None and not isinstance(visible["error_code"], str))
        ):
            raise MessageValidationError("Invalid application tool call record")
        _blocks(visible["content"], images=True)
        calls.append(visible)
    return UserMessage(
        "Application tool execution results (application-selected, not model tool calls):\n"
        + json.dumps({"calls": calls}, ensure_ascii=False, allow_nan=False),
        timestamp=message.timestamp,
    )


def validate_history(messages: list[Message]) -> None:
    pending: dict[str, str] = {}
    for message in messages:
        message_to_dict(message)
        if isinstance(message, ToolResultMessage):
            if pending.pop(message.call_id, None) != message.name:
                raise MessageValidationError("Unmatched or duplicate tool result")
        else:
            if pending:
                raise MessageValidationError("Dangling tool calls")
            if isinstance(message, AssistantMessage):
                if message.stop_reason == "pending":
                    raise MessageValidationError("Pending response cannot be committed to history")
                if message.tool_calls and message.stop_reason in {"error", "aborted"}:
                    raise MessageValidationError("Failed assistant cannot declare tool calls")
                pending = {c.id: c.name for c in message.tool_calls}
    if pending:
        raise MessageValidationError("Dangling tool calls")


def encode_messages(messages: list[Message]) -> str:
    validate_history(messages)
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "messages": [message_to_dict(m) for m in messages]},
        ensure_ascii=False,
        allow_nan=False,
    )


def decode_messages(value: str) -> list[Message]:
    try:
        data = json.loads(value)
        validate_json(data)
        if type(data.get("schema_version")) is not int or data["schema_version"] not in {3, 4}:
            raise MessageValidationError("Unsupported message schema version")
        if set(data) != {"schema_version", "messages"} or not isinstance(data["messages"], list):
            raise MessageValidationError("Invalid message envelope")
        messages = [message_from_dict(m) for m in data["messages"]]
        validate_history(messages)
        return messages
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        raise MessageValidationError(str(exc)) from exc
