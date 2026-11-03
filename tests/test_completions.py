"""OpenAI-compatible Chat Completions provider: behavior the upstream differential cannot
cover (keyless servers, endpoint handling, failures, configuration errors)."""

import json

import httpx
import pytest

from pi_python import (
    Agent,
    CancelToken,
    ConfigurationError,
    ModelCatalog,
    ModelInfo,
    ModelRequest,
    UserMessage,
    tool,
)
from pi_python.providers import HTTPTransport, OpenAICompletionsProvider
from pi_python.providers.completions import detect_compat, resolve_compat

LOCAL = "http://localhost:11434/v1"


def sse(chunks, done=True):
    data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (data + ("data: [DONE]\n\n" if done else "")).encode()


def chunk(delta=None, finish=None, **fields):
    return {
        "id": "chatcmpl-1",
        "model": "qwen3:8b",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        **fields,
    }


def reply(text="Hello"):
    return [chunk({"role": "assistant", "content": text}), chunk({}, "stop")]


def server(*responses):
    """A mock endpoint answering each request with the next (status, body) pair."""
    requests = []
    queue = list(responses)

    async def handler(request):
        requests.append(request)
        status, body = queue.pop(0)
        if not isinstance(body, bytes):
            body = sse(body) if status < 400 else json.dumps(body).encode()
        return httpx.Response(status, headers={"content-type": "text/event-stream"}, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), requests


async def run(provider, model, prompt="hi", **request):
    events = [
        e
        async for e in provider.stream(
            ModelRequest([UserMessage(prompt)], model=model.id, model_info=model, **request),
            CancelToken(),
        )
    ]
    return events, events[-1].message


async def test_keyless_local_server_needs_no_credentials():
    client, requests = server((200, reply()))
    async with client:
        provider = OpenAICompletionsProvider(
            base_url=LOCAL + "/", name="ollama", transport=HTTPTransport(client)
        )
        model = provider.model("qwen3:8b")
        events, message = await run(provider, model)
    assert [e.type for e in events] == ["start", "text_start", "text_delta", "text_end", "done"]
    assert message.content[0].text == "Hello" and message.stop_reason == "stop"
    assert (message.provider, message.api, message.model) == (
        "ollama",
        "openai-completions",
        "qwen3:8b",
    )
    request = requests[0]
    assert str(request.url) == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == "qwen3:8b" and body["stream"] is True
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_completion_tokens"] == 16384  # the declared default output limit


@pytest.mark.parametrize(
    "usage,known",
    [
        ({}, False),
        (None, False),
        ({"unrelated": 1}, False),
        ({"prompt_tokens": 12}, False),
        ({"completion_tokens": 8}, False),
        ({"prompt_tokens": True, "completion_tokens": 8}, False),
        ({"prompt_tokens": -1, "completion_tokens": 8}, False),
        ({"prompt_tokens": 12, "completion_tokens": "8"}, False),
        ({"prompt_tokens": 0, "completion_tokens": 0}, True),
        ({"prompt_tokens": 12, "completion_tokens": 8}, True),
    ],
)
@pytest.mark.parametrize("on_choice", [False, True])
async def test_usage_requires_complete_counts_and_preserves_real_zero(usage, known, on_choice):
    chunks = reply()
    target = chunks[-1]["choices"][0] if on_choice else chunks[-1]
    target["usage"] = usage
    client, _ = server((200, chunks))
    async with client:
        provider = OpenAICompletionsProvider(base_url=LOCAL, transport=HTTPTransport(client))
        _, message = await run(provider, provider.model("qwen3:8b"))
    assert message.stop_reason == "stop"
    assert (message.diagnostics is None) is known
    if not known:
        assert {"type": "usage_unavailable"} in message.diagnostics
    else:
        assert message.usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.parametrize("empty_last", [False, True])
async def test_empty_usage_does_not_hide_valid_choice_or_earlier_usage(empty_last):
    chunks = reply()
    chunks[-1]["usage"] = {}
    usage = {"prompt_tokens": 12, "completion_tokens": 8}
    if empty_last:
        chunks[0]["usage"] = usage
    else:
        chunks[-1]["choices"][0]["usage"] = usage
    client, _ = server((200, chunks))
    async with client:
        provider = OpenAICompletionsProvider(base_url=LOCAL, transport=HTTPTransport(client))
        _, message = await run(provider, provider.model("qwen3:8b"))
    assert message.diagnostics is None
    assert message.usage["total_tokens"] == 20


async def test_partial_usage_stays_available_to_direct_callers_but_is_not_measured():
    chunks = reply()
    chunks[0]["usage"] = {"prompt_tokens": 12}
    client, _ = server((200, chunks))
    async with client:
        provider = OpenAICompletionsProvider(base_url=LOCAL, transport=HTTPTransport(client))
        _, message = await run(provider, provider.model("qwen3:8b"))
    assert message.usage["input"] == 12
    assert {"type": "usage_unavailable"} in message.diagnostics


def test_model_helper_mirrors_upstream_custom_model_defaults():
    provider = OpenAICompletionsProvider(base_url=LOCAL, name="ollama")
    model = provider.model("qwen3:8b")
    assert (model.provider, model.api, model.name) == ("ollama", "openai-completions", "qwen3:8b")
    assert (model.context_window, model.max_tokens, model.reasoning, model.input) == (
        128_000,
        16_384,
        False,
        ("text",),
    )
    assert model.base_url == LOCAL
    tuned = provider.model(
        "gpt-oss:20b", context_window=131072, max_tokens=32000, reasoning=True, compat={"x": 1}
    )
    assert tuned.reasoning and tuned.context_window == 131072 and tuned.compat == {"x": 1}
    # The model is an ordinary record: it can also be registered for lookup by name.
    provider.catalog.register(model)
    assert provider.catalog.get("ollama", "qwen3:8b") == model


async def test_api_key_header_overrides_and_removal():
    client, requests = server((200, reply()), (200, reply()), (200, reply()))
    async with client:
        provider = OpenAICompletionsProvider(
            base_url=LOCAL, api_key="sk-local", transport=HTTPTransport(client)
        )
        model = provider.model("qwen3:8b")
        await run(provider, model)
        await run(provider, model, api_key="per-request")
        await run(
            provider,
            model,
            options={"headers": {"Authorization": None, "X-Gateway": "lab"}},
        )
    assert requests[0].headers["authorization"] == "Bearer sk-local"
    assert requests[1].headers["authorization"] == "Bearer per-request"
    assert "authorization" not in requests[2].headers
    assert requests[2].headers["x-gateway"] == "lab"


def test_base_url_is_required_to_be_an_http_url():
    with pytest.raises(ConfigurationError, match="http"):
        OpenAICompletionsProvider(base_url="localhost:11434/v1")
    with pytest.raises(TypeError):
        OpenAICompletionsProvider()  # type: ignore[call-arg]
    provider = OpenAICompletionsProvider(base_url="https://api.example.com/v1///")
    assert provider.base_url == "https://api.example.com/v1"


async def test_unknown_model_error_explains_how_to_declare_it():
    provider = OpenAICompletionsProvider(base_url=LOCAL, name="ollama")
    events = [
        e
        async for e in provider.stream(
            ModelRequest([UserMessage("hi")], model="llama3.2"), CancelToken()
        )
    ]
    assert [e.type for e in events] == ["error"]
    error = events[0].message.error
    assert "Unknown model ollama/llama3.2" in error
    assert "provider.model('llama3.2'" in error and "openai-completions" in error


def test_model_with_another_api_is_rejected():
    catalog = ModelCatalog([ModelInfo("gpt-x", "openai", "openai-responses", "X", 1000, 100)])
    provider = OpenAICompletionsProvider(
        base_url="https://api.openai.com/v1", name="openai", catalog=catalog
    )
    with pytest.raises(ConfigurationError, match="uses api 'openai-responses'"):
        provider.build_request(ModelRequest([UserMessage("hi")], model="gpt-x"))


async def test_http_failure_becomes_error_event_with_diagnostics():
    client, requests = server((401, {"error": {"message": "invalid api key"}}))
    async with client:
        provider = OpenAICompletionsProvider(
            base_url=LOCAL, api_key="sk-bad-key-123", transport=HTTPTransport(client)
        )
        events, message = await run(provider, provider.model("qwen3:8b"))
    assert [e.type for e in events] == ["error"] and len(requests) == 1
    assert message.stop_reason == "error"
    assert "HTTP 401" in message.error and "invalid api key" in message.error
    assert "sk-bad-key-123" not in message.error
    assert message.diagnostics[0]["status"] == 401
    assert message.diagnostics[0]["category"] == "authentication"


async def test_server_error_is_retried_before_streaming():
    client, requests = server((503, {"error": "loading model"}), (200, reply("ok")))
    async with client:
        provider = OpenAICompletionsProvider(
            base_url=LOCAL, transport=HTTPTransport(client, max_retries=1, retry_base=0)
        )
        _, message = await run(provider, provider.model("qwen3:8b"))
    assert message.stop_reason == "stop" and len(requests) == 2


@pytest.mark.parametrize(
    "chunks, expected",
    [
        ([chunk({}, "content_filter")], "Provider finish_reason: content_filter"),
        ([chunk({"content": "cut"})], "Stream ended without finish_reason"),
        ([{"error": {"message": "model overloaded", "code": 503}}], "model overloaded"),
    ],
)
async def test_stream_failures_become_error_events(chunks, expected):
    client, _ = server((200, chunks))
    async with client:
        provider = OpenAICompletionsProvider(base_url=LOCAL, transport=HTTPTransport(client))
        events, message = await run(provider, provider.model("qwen3:8b"))
    assert events[-1].type == "error" and message.stop_reason == "error"
    assert expected in message.error


async def test_tool_call_without_id_or_arguments():
    """Pi keeps "" for both; Python needs a call ID and closes empty arguments as {}."""
    calls = [{"index": 0, "type": "function", "function": {"name": "ping"}}]
    client, _ = server((200, [chunk({"tool_calls": calls}), chunk({}, "tool_calls")]))
    async with client:
        provider = OpenAICompletionsProvider(base_url=LOCAL, transport=HTTPTransport(client))
        events, message = await run(provider, provider.model("qwen3:8b"))
    call = message.content[0]
    assert message.stop_reason == "tool_use"
    assert call.name == "ping" and call.arguments == {} and call.id.startswith("call_")
    assert [(e.type, e.delta) for e in events if e.type == "toolcall_delta"] == [
        ("toolcall_delta", ""),
        ("toolcall_delta", "{}"),
    ]


async def test_data_after_done_is_ignored_and_stream_callback_sees_chunks():
    body = sse(reply("one")) + sse([chunk({"content": "late"})], done=False)
    client, _ = server((200, body))
    seen = []
    async with client:
        provider = OpenAICompletionsProvider(base_url=LOCAL, transport=HTTPTransport(client))
        _, message = await run(
            provider, provider.model("qwen3:8b"), on_provider_stream_event=seen.append
        )
    assert message.content[0].text == "one" and len(seen) == 2


def test_compat_values_are_validated_and_detected():
    provider = OpenAICompletionsProvider(base_url=LOCAL)
    for compat, field in [
        ({"thinkingFormat": "bogus"}, "thinkingFormat"),
        ({"supportsStore": "false"}, "supportsStore"),
        ({"maxTokensField": "max_output_tokens"}, "maxTokensField"),
    ]:
        model = provider.model("m", compat=compat)
        with pytest.raises(ConfigurationError, match=field):
            provider.build_request(ModelRequest([UserMessage("hi")], model="m", model_info=model))
    # Unknown keys are ignored, as for the other providers' compat records.
    model = provider.model("m", compat={"someFutureFlag": True, "maxTokensField": "max_tokens"})
    body = provider.build_request(ModelRequest([UserMessage("hi")], model="m", model_info=model))
    assert body["max_tokens"] == 16384 and "max_completion_tokens" not in body
    assert detect_compat("ollama", LOCAL, "m")["supportsStore"] is True
    assert resolve_compat(model, "https://api.deepseek.com")["thinkingFormat"] == "deepseek"


async def test_agent_tool_conversation_against_local_server():
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    call = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
    }
    client, requests = server(
        (200, [chunk({"role": "assistant", "tool_calls": [call]}), chunk({}, "tool_calls")]),
        (200, reply("2 + 3 = 5")),
    )
    async with client:
        provider = OpenAICompletionsProvider(
            base_url=LOCAL, name="ollama", transport=HTTPTransport(client)
        )
        agent = Agent(
            provider=provider,
            model=provider.model("qwen3:8b", context_window=40960, max_tokens=4096),
            system_prompt="Use tools for arithmetic.",
            tools=[add],
        )
        result = await agent.prompt("What is 2 + 3?")
    assert result.status == "completed", result.errors
    assert result.messages[-1].content[0].text == "2 + 3 = 5"
    first, second = (json.loads(r.content) for r in requests)
    assert first["tools"][0]["function"]["name"] == "add"
    assert first["messages"][0] == {"role": "system", "content": "Use tools for arithmetic."}
    assert second["messages"][-2]["tool_calls"][0]["function"]["arguments"] == '{"a":2,"b":3}'
    assert second["messages"][-1] == {"role": "tool", "content": "5", "tool_call_id": "call_1"}


def test_model_helper_validates_like_the_catalog():
    provider = OpenAICompletionsProvider(base_url=LOCAL)
    with pytest.raises(ConfigurationError, match="positive"):
        provider.model("m", max_tokens=0)
    with pytest.raises(ConfigurationError, match="thinking level"):
        provider.model("m", thinking_level_map={"turbo": "x"})


async def test_chat_stream_error_uses_shared_redaction_and_recovery_rules():
    from pi_python import is_retryable_error

    client, _ = server((200, [{"error": {"code": "overloaded_error", "message": "token-secret"}}]))
    async with client:
        provider = OpenAICompletionsProvider(
            base_url=LOCAL, name="local", api_key="token-secret", transport=HTTPTransport(client)
        )
        _, message = await run(provider, provider.model("test"))
    assert "token-secret" not in message.error
    assert "[redacted]" in message.error
    assert is_retryable_error(message)
