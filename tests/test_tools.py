import asyncio
import pytest
from pi_python import *
from pi_python.tools import validate_schema


def calls(*ids):
    return AssistantMessage([ToolCall(i, "t", {"i": i}) for i in ids], "tool_use")


def setup(execute, response=None, hooks=None, **kwargs):
    t = Tool("t", "", {"type": "object"}, execute)
    p = ScriptedProvider([response or calls("a", "b"), AssistantMessage.text("done")])
    a = Agent(provider=p, tools=[t], hooks=hooks, **kwargs)
    return a, p


async def test_C04_completion_vs_commit_barriers():
    entered = {k: asyncio.Event() for k in "ab"}
    releases = {k: asyncio.Event() for k in "ab"}
    finished = asyncio.Event()
    order = []
    effects = []

    async def execute(args, ctx):
        k = args["i"]
        effects.append(k)
        entered[k].set()
        await releases[k].wait()
        return ToolResult.text(k)

    a, p = setup(execute)

    def listener(e):
        if e.type == "tool_execution_end":
            order.append(e.call_id)
            if e.call_id == "b":
                finished.set()

    a.subscribe(listener)
    task = asyncio.create_task(a.prompt("go"))
    await asyncio.gather(*(e.wait() for e in entered.values()))
    releases["b"].set()
    await finished.wait()
    releases["a"].set()
    r = await task
    assert order == ["b", "a"] and effects == ["a", "b"]
    assert [m.call_id for m in r.messages if isinstance(m, ToolResultMessage)] == ["a", "b"]


@pytest.mark.parametrize("mode", ["global", "tool"])
async def test_C03_C05_serial_batch(mode):
    trace = []

    async def execute(args, ctx):
        trace.append("start" + ctx.call_id)
        await asyncio.sleep(0)
        trace.append("end" + ctx.call_id)
        return ToolResult.text("ok")

    t = Tool(
        "t",
        "",
        {"type": "object"},
        execute,
        execution_mode="sequential" if mode == "tool" else "parallel",
    )
    u = Tool("u", "", {"type": "object"}, execute)
    p = ScriptedProvider(
        [
            AssistantMessage([ToolCall("a", "u", {}), ToolCall("b", "t", {})], "tool_use"),
            AssistantMessage.text("done"),
        ]
    )
    r = await Agent(
        provider=p, tools=[t, u], execution_mode="sequential" if mode == "global" else "parallel"
    ).prompt("go")
    assert r.status == "completed" and trace == ["starta", "enda", "startb", "endb"]


async def test_C06_failure_does_not_cancel_sibling():
    async def execute(args, ctx):
        if ctx.call_id == "a":
            raise ValueError("expected")
        await asyncio.sleep(0)
        return ToolResult.text("success")

    a, p = setup(execute)
    r = await a.prompt("go")
    assert [o.execution_status for o in r.tool_outcomes] == ["failed", "succeeded"]
    assert len(p.requests) == 2


async def test_C07_C08_D1_D2_preparation_and_strict_validation():
    seen = []

    async def execute(args, ctx):
        seen.append(args)
        return ToolResult.text("ok")

    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }

    def before(call, args, ctx):
        args["n"] = "mutated"

    t = Tool("t", "", schema, execute, prepare_arguments=lambda args: {"n": int(args["n"])})
    p = ScriptedProvider(
        [
            AssistantMessage([ToolCall("x", "t", {"n": "3"})], "tool_use"),
            AssistantMessage.text("done"),
        ]
    )
    r = await Agent(provider=p, tools=[t], hooks=Hooks(before_tool_call=before)).prompt("go")
    assert seen == [{"n": 3}]
    assert r.tool_outcomes[0].original_arguments == {"n": "3"}
    t.prepare_arguments = None
    for args in [{"n": "3"}, {"n": None}, {"n": True}, {"n": 1, "extra": 1}]:
        p = ScriptedProvider(
            [
                AssistantMessage([ToolCall("x", "t", args)], "tool_use"),
                AssistantMessage.text("done"),
            ]
        )
        r = await Agent(provider=p, tools=[t]).prompt("go")
        assert r.tool_outcomes[0].result.error_code == "invalid_arguments"
    assert len(seen) == 1


async def test_C09_block_and_clear_structure():
    async def execute(args, ctx):
        return ToolResult.text("old", structured_content={"old": 1})

    a, p = setup(
        execute,
        hooks=Hooks(
            before_tool_call=lambda call, args, ctx: False if call.id == "a" else True,
            after_tool_call=lambda call, result, ctx: ToolResultUpdate(
                content=[TextContent("new")]
            ),
        ),
    )
    r = await a.prompt("go")
    assert r.tool_outcomes[0].execution_status == "not_started"
    assert r.tool_outcomes[1].result.structured_content is None
    assert r.tool_outcomes[1].raw_result.structured_content == {"old": 1}


@pytest.mark.parametrize("failure", ["hook", "output", "invalid_result"])
async def test_C22_raw_success_survives_finalization_failure(failure):
    async def execute(args, ctx):
        if failure == "invalid_result":
            return {"wrong": True}
        return ToolResult.text(
            "executed", structured_content={"n": "bad" if failure == "output" else 1}
        )

    def after(call, result, ctx):
        if failure == "hook":
            raise RuntimeError("post failed")

    t = Tool(
        "t",
        "",
        {"type": "object"},
        execute,
        output_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
    )
    p = ScriptedProvider([calls("a"), AssistantMessage.text("done")])
    r = await Agent(provider=p, tools=[t], hooks=Hooks(after_tool_call=after)).prompt("go")
    o = r.tool_outcomes[0]
    assert o.execution_status == "succeeded" and o.raw_result is not None
    assert o.result.is_error and o.result.error_code == "finalization_error"


async def test_C04_after_hook_completion_controls_end_order():
    release = asyncio.Event()
    b_done = asyncio.Event()
    order = []

    async def execute(args, ctx):
        return ToolResult.text(ctx.call_id)

    async def after(call, result, ctx):
        if call.id == "a":
            await release.wait()

    a, p = setup(execute, hooks=Hooks(after_tool_call=after))

    def listener(e):
        if e.type == "tool_execution_end":
            order.append(e.call_id)
            if e.call_id == "b":
                b_done.set()

    a.subscribe(listener)
    task = asyncio.create_task(a.prompt("go"))
    await b_done.wait()
    release.set()
    await task
    assert order == ["b", "a"]


async def test_shared_programmatic_path():
    seen = []

    async def execute(args, ctx):
        seen.append(args)
        return ToolResult.text("ok")

    t = Tool("t", "", {"type": "object"}, execute)

    async def emit(value):
        pass

    o = await run_tool_call(t, ToolCall("x", "t", {}), ToolContext("run", "x", CancelToken(), emit))
    assert o.execution_status == "succeeded" and seen == [{}]


async def test_failing_before_hook_is_a_tool_error_not_a_rejection():
    async def execute(args, ctx):
        raise AssertionError("must not run")

    def before(call, args, ctx):
        raise RuntimeError("policy service down")

    async def emit(value):
        pass

    t = Tool("t", "", {"type": "object"}, execute)
    context = ToolContext("run", "x", CancelToken(), emit)
    o = await run_tool_call(t, ToolCall("x", "t", {}), context, before_tool_call=before)
    assert o.execution_status == "not_started" and o.result.error_code == "hook_error"
    assert o.result.content[0].text == "RuntimeError: policy service down"


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "https://example.test/schema"},
        {"$defs": {"x": {"$ref": "file:///etc/passwd"}}, "$ref": "#/$defs/x"},
        {"$ref": "#/$defs/missing"},
        {"$schema": "https://example.test/custom-dialect"},
        {"type": "not-a-type"},
    ],
)
def test_C07_unsupported_schema_rejected(schema):
    with pytest.raises(ConfigurationError):
        validate_schema(schema)


async def test_standard_schemas_from_pydantic_and_mcp_register_and_validate():
    # The shapes pydantic (2020-12) and MCP servers (draft-07) emit; Pi accepts both
    # and checks `format` and `pattern` when validating arguments.
    seen = []

    async def execute(args, ctx):
        seen.append(args)
        return ToolResult.text("ok")

    mcp = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "email": {"type": "string", "format": "email"},
            "id": {"type": "string", "pattern": "^[0-9]+$"},
            "kind": {"$ref": "#/definitions/kind"},
        },
        "required": ["email", "id"],
        "definitions": {"kind": {"enum": ["a", "b"]}},
        "unevaluatedProperties": False,
    }
    calls = [
        ToolCall("good", "t", {"email": "a@b.org", "id": "42", "kind": "a"}),
        ToolCall("bad_format", "t", {"email": "nope", "id": "42"}),
        ToolCall("bad_pattern", "t", {"email": "a@b.org", "id": "x"}),
    ]
    provider = ScriptedProvider(
        [AssistantMessage(calls, "tool_use"), AssistantMessage.text("done")]
    )
    r = await Agent(provider=provider, tools=[Tool("t", "", mcp, execute)]).prompt("go")
    results = {o.call.id: o.result.error_code for o in r.tool_outcomes}
    assert results == {
        "good": None,
        "bad_format": "invalid_arguments",
        "bad_pattern": "invalid_arguments",
    }
    assert seen == [{"email": "a@b.org", "id": "42", "kind": "a"}]


def test_local_schema_refs():
    validate_schema(
        {
            "type": "object",
            "properties": {"n": {"$ref": "#/$defs/n"}},
            "$defs": {"n": {"type": "integer"}},
        }
    )


async def test_tool_resource_owner_is_preserved_across_config_snapshots():
    class Adapter:
        def __init__(self):
            self.calls = []

        def __deepcopy__(self, memo):
            raise AssertionError("External resource must not be copied")

        async def execute(self, args, ctx):
            self.calls.append(ctx.call_id)
            return ToolResult.text("ok")

    adapter = Adapter()
    t = Tool("t", "", {"type": "object"}, adapter.execute)
    p = ScriptedProvider([calls("a"), AssistantMessage.text("done")])
    a = Agent(provider=p, tools=[t])
    t.input_schema["additionalProperties"] = False
    result = await a.prompt("go")
    assert result.status == "completed" and adapter.calls == ["a"]
    assert p.requests[0].tools[0].input_schema == {"type": "object"}


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "x-vendor": {"$ref": "https://ignored.example/x"}},
        {"$schema": "http://json-schema.org/schema#", "type": "object"},
        {"$schema": "http://json-schema.org/draft-04/schema#", "type": "object", "required": []},
        {
            "$defs": {"My Type": {"type": "string"}},
            "properties": {"a": {"$ref": "#/$defs/My%20Type"}},
        },
        {
            "definitions": {"n": {"$anchor": "num", "type": "integer"}},
            "properties": {"a": {"$ref": "#num"}},
        },
    ],
    ids=["vendor-keyword", "latest-draft", "draft4-empty-required", "encoded-pointer", "anchor"],
)
def test_real_world_schema_shapes_are_accepted(schema):
    validate_schema(schema)


def test_deep_schema_and_long_validation_errors_are_reported_compactly():
    deep = {"type": "object"}
    for _ in range(2000):
        deep = {"type": "object", "properties": {"x": deep}}
    with pytest.raises(ConfigurationError, match="nested too deeply"):
        validate_schema(deep)

    async def ok(args, ctx):
        return "ok"

    schema = {
        "type": "object",
        "properties": {f"field{i}": {"type": "string", "description": "x" * 200} for i in range(5)},
        "required": ["field0"],
        "additionalProperties": False,
    }
    outcome = asyncio.run(
        run_tool_call(
            Tool("t", "", schema, ok),
            ToolCall("c", "t", {"field0": 1, "extra": True}),
            ToolContext("r", "c", CancelToken(), None),
        )
    )
    text = outcome.result.content[0].text
    assert outcome.result.error_code == "invalid_arguments" and len(text) < 300
    assert "  - field0: 1 is not of type 'string'" in text and "description" not in text


def test_schema_nesting_limit_is_counted_before_any_recursion():
    def nested(levels):  # objects nest exactly `levels` deep, counting the schema itself
        value = {}
        for _ in range(levels - 2):
            value = {"a": value}
        return {"type": "object", "default": value}

    validate_schema(nested(100))
    with pytest.raises(ConfigurationError, match="more than 100 levels"):
        validate_schema(nested(101))
    # The deepest common shapes under the limit validate on every interpreter.
    for wrap, times in [
        (lambda s: {"type": "array", "items": s}, 99),
        (lambda s: {"anyOf": [s, {"type": "null"}]}, 49),
        (lambda s: {"type": "object", "properties": {"x": s}}, 49),
    ]:
        schema = {"type": "string"}
        for _ in range(times):
            schema = wrap(schema)
        validate_schema(schema)
    cyclic = {"type": "object", "properties": {}}
    cyclic["properties"]["self"] = cyclic
    with pytest.raises(ConfigurationError, match="nested too deeply"):
        validate_schema(cyclic)


async def test_common_return_values_and_failed_conversion_keep_the_execution_fact():
    import datetime
    import uuid
    from dataclasses import dataclass

    @dataclass
    class Point:
        x: int
        when: datetime.date

    def returning(value):
        async def execute(args, ctx):
            return value

        return Tool("t", "", {"type": "object"}, execute)

    context = ToolContext("r", "c", CancelToken(), None)
    cases = [
        ((1, 2), "[1, 2]"),
        (datetime.date(2026, 1, 2), "2026-01-02"),
        (Point(1, datetime.date(2026, 1, 2)), '{"x": 1, "when": "2026-01-02"}'),
        (uuid.UUID(int=1), "00000000-0000-0000-0000-000000000001"),
    ]
    for value, text in cases:
        outcome = await run_tool_call(returning(value), ToolCall("c", "t", {}), context)
        assert outcome.result.content[0].text == text
    odd = await run_tool_call(returning(object()), ToolCall("c", "t", {}), context)
    assert odd.execution_status == "succeeded" and odd.result.error_code == "finalization_error"
