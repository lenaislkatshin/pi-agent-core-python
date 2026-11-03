"""Malformed tool-call arguments reach the model as a correctable error, as in Pi 1.5."""

import json

import httpx

from pi_python import (
    Agent,
    AssistantMessage,
    ModelEvent,
    ToolCall,
    message_from_dict,
    message_to_dict,
    stream,
    tool,
)
from pi_python.providers import HTTPTransport, OpenAICompletionsProvider
from pi_python.stream import parse_tool_call_arguments

LOCAL = "http://localhost:11434/v1"


def sse(chunks):
    data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (data + "data: [DONE]\n\n").encode()


def chunk(delta=None, finish=None):
    return {
        "id": "chatcmpl-1",
        "model": "m",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }


def call_chunk(arguments, name="search"):
    calls = [{"index": 0, "id": "c1", "function": {"name": name, "arguments": arguments}}]
    return chunk({"role": "assistant", "tool_calls": calls})


FINAL = [chunk({"role": "assistant", "content": "done"}), chunk({}, "stop")]


def server(*scripts):
    requests = []
    queue = list(scripts)

    async def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse(queue.pop(0))
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), requests


async def run_agent(tools, *scripts):
    client, requests = server(*scripts)
    async with client:
        provider = OpenAICompletionsProvider(base_url=LOCAL, transport=HTTPTransport(client))
        agent = Agent(provider=provider, model=provider.model("m"), tools=tools)
        result = await agent.prompt("q")
        await agent.aclose()
    tool_messages = [m for m in requests[-1]["messages"] if m["role"] == "tool"]
    return result, tool_messages


def recording_tool():
    calls = []

    @tool
    async def search(query: str) -> str:
        """Search."""
        calls.append(query)
        return "ok"

    return search, calls


async def test_invalid_json_goes_back_to_the_model_and_the_run_continues():
    search, calls = recording_tool()
    result, tool_messages = await run_agent(
        [search], [call_chunk('{"query":"x'), chunk({}, "tool_calls")], FINAL
    )
    assert result.status == "completed" and calls == []
    assert result.messages[-1].content[0].text == "done"
    call = result.messages[-2 if result.messages[-2].role != "tool_result" else -3].tool_calls[0]
    assert call.arguments == {} and "Received arguments:" in call.arguments_error
    text = tool_messages[0]["content"]
    assert text.startswith('Invalid arguments for tool "search": ')
    assert text.endswith('Received arguments:\n{"query":"x')


async def test_arguments_that_are_not_an_object_are_reported():
    search, calls = recording_tool()
    result, tool_messages = await run_agent(
        [search], [call_chunk("[1]"), chunk({}, "tool_calls")], FINAL
    )
    assert result.status == "completed" and calls == []
    assert "Tool arguments must be a JSON object" in tool_messages[0]["content"]


async def test_truncated_call_is_not_executed_and_says_how_to_recover():
    search, calls = recording_tool()
    result, tool_messages = await run_agent(
        [search], [call_chunk('{"query":"x'), chunk({}, "length")], FINAL
    )
    assert result.status == "completed" and calls == []
    assert tool_messages[0]["content"] == (
        'Tool call "search" was not executed: the response hit the output token limit, '
        "so its arguments may be truncated. Re-issue the tool call with complete arguments."
    )


async def test_error_text_for_unknown_tool_validation_and_tool_failure():
    search, _ = recording_tool()
    _, unknown = await run_agent(
        [search], [call_chunk("{}", "no_such"), chunk({}, "tool_calls")], FINAL
    )
    assert unknown[0]["content"] == "Tool no_such not found"

    _, invalid = await run_agent(
        [search], [call_chunk('{"bad":"甲"}'), chunk({}, "tool_calls")], FINAL
    )
    assert invalid[0]["content"].startswith('Validation failed for tool "search":\n  - ')
    assert invalid[0]["content"].endswith('Received arguments:\n{\n  "bad": "甲"\n}')

    @tool
    async def broken(query: str) -> str:
        """Fail."""
        raise RuntimeError("数据库超时")

    result, failed = await run_agent(
        [broken], [call_chunk('{"query":"x"}', "broken"), chunk({}, "tool_calls")], FINAL
    )
    assert failed[0]["content"] == "数据库超时"
    assert result.tool_outcomes[0].error == "RuntimeError: 数据库超时"


def test_parse_keeps_blank_arguments_empty_and_never_keeps_a_prefix():
    assert parse_tool_call_arguments("") == ({}, None)
    assert parse_tool_call_arguments("  ") == ({}, None)
    assert parse_tool_call_arguments('{"a":1}') == ({"a": 1}, None)
    arguments, error = parse_tool_call_arguments('{"a":1,"b":')
    assert arguments == {} and error.endswith('Received arguments:\n{"a":1,"b":')


def test_arguments_error_survives_message_encoding():
    message = AssistantMessage(
        [ToolCall("c", "search", {}, arguments_error="bad\n\nReceived arguments:\n{")], "tool_use"
    )
    encoded = message_to_dict(message)
    assert encoded["content"][0]["arguments_error"].startswith("bad")
    assert message_from_dict(encoded).content[0].arguments_error.startswith("bad")
    plain = message_to_dict(AssistantMessage([ToolCall("c", "search", {})], "tool_use"))
    assert "arguments_error" not in plain["content"][0]


async def test_stream_check_still_rejects_an_unflagged_incomplete_call():
    async def source():
        yield ModelEvent.boundary("start", 0, ToolCall("c", "search", {}))
        yield ModelEvent.toolcall('{"a":', 0)
        yield ModelEvent.boundary("end", 0, ToolCall("c", "search", {}))
        yield ModelEvent.done(AssistantMessage([ToolCall("c", "search", {})], "tool_use"))

    partial = AssistantMessage([], stop_reason="pending")
    try:
        async for _ in stream.checked_events(source(), partial):
            pass
    except Exception as exc:
        assert "Incomplete tool arguments" in str(exc)
    else:
        raise AssertionError("an incomplete call without arguments_error must fail")


async def test_anthropic_reports_malformed_arguments_on_the_tool_call():
    from pi_python import CancelToken, ModelRequest, ToolDeclaration, UserMessage

    from test_providers import remote

    values = [
        {
            "type": "message_start",
            "message": {"id": "m", "usage": {"input_tokens": 1, "output_tokens": 0}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"query":"x'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    provider, _, _, client = remote("anthropic", [values])
    request = ModelRequest(
        [UserMessage("go")],
        [ToolDeclaration("search", "", {"type": "object", "properties": {}})],
        model="test-model",
    )
    async with client:
        final = [e async for e in provider.stream(request, CancelToken())][-1]
    assert final.type == "done"
    assert 'Received arguments:\n{"query":"x' in final.message.tool_calls[0].arguments_error


def test_responses_item_reports_malformed_arguments_on_the_tool_call():
    from pi_python.providers.openai import response_block

    block = response_block(
        {"type": "function_call", "call_id": "c", "id": "i", "name": "f", "arguments": "{bad"}
    )
    assert block.arguments == {} and "Received arguments:\n{bad" in block.arguments_error
