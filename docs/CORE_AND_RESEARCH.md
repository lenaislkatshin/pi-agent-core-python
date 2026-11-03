# Core contracts and the research API

[中文](zh/CORE_AND_RESEARCH.md)

Both APIs ship in `pi-python-core`. They use the same `Run`, model providers and tool execution engine. No second package or external extension is required. These entry points are available from 0.12.0; see the [release and migration notes](releases/0.12.0.md).

For ordinary research applications, keep using `from pi_python import Agent, tool`. This entry point keeps `RunResult`, `@tool`, synchronous calls, explicit tool batches, configurable limits, failed-response retry and automatic cache affinity for capable providers. Existing plugins continue to construct this Agent.

For integrations following Pi's contracts, use `from pi_python import core`. `core.Agent` and its low-level loops use the new hook/result structures and complete standard event payloads. The old top-level `Hooks` and low-level exports remain migration bridges; new upstream-oriented integrations should use `core`. There is one execution engine, with adapters at these public boundaries.

Run both offline examples:

```bash
uv run python examples/core_and_research.py
```

## Choosing an entry point

| Requirement | Research API | Core API |
|---|---|---|
| Prompt result | `RunResult` from `await agent.prompt(...)` | `None`; inspect state or `agent_end.messages` |
| Subscription | `listener(Event)` with diagnostic envelope | `listener(event_dict, signal)` |
| Finish decision | Existing `Hooks`: `"end"` / `"continue"` | `{ "action": "end" }` / `{ "action": "continue" }` |
| Tools | `@tool` or `Tool.execute(args, context)` | `core.Tool.execute(call_id, args, signal, on_update)` |
| Failed assistant continuation | Convenience retry | Low-level continue rejects an assistant tail; Agent may consume queued messages |
| Cache affinity | Automatic for providers that declare support, with opt-out | Explicit `session_id` only |
| Low-level stream | Bounded iteration; `result()` releases backpressure | Unbounded queue; result does not consume queued events |

The research Agent can also take `core.Hooks` and `core.Tool`. This allows migrating callbacks before switching event consumers. Existing root `Hooks` retain their prior signatures. Stream functions differ deliberately: research `stream_fn(request, cancel)` versus core `stream_fn(model, context, options)`, where `options["signal"]` carries the cancellation token. A core stream function returns an async event iterator, optionally through an awaitable.

## Core hooks and events

`core.Agent` accepts `hooks=core.Hooks(...)` or named callbacks such as `finish_turn=...`. The grouped Hooks use low-level context signatures. Agent's named `prepare_next_turn(signal)` follows the no-context variant; use `prepare_next_turn_with_context(context, signal)` for the context variant. `core.AgentLoopConfig` accepts hooks directly as fields, or as a grouped Hooks object.

- `before_tool_call(context, signal)` receives assistant_message, tool_call, validated args and AgentContext. Return a `block`/`reason`/`terminate` dictionary or None.
- `after_tool_call(context, signal)` additionally receives result and is_error. Return a partial result dictionary. Omitted keys preserve values; explicit None clears nullable metadata. Replacing content clears structured_content unless supplied with it.
- `prepare_request(context, signal)` receives AgentContext, model and thinking_level; return `AgentRequestUpdate` or a matching dictionary. It cannot append messages.
- The low-level `prepare_next_turn(context, signal)` receives completed-turn context; return `AgentLoopTurnUpdate`, which can append messages. Its context replacement includes messages and tools.
- `finish_turn(context, signal)` returns an action dictionary or None.
- Tool `on_update` is async in Python; await it while execute is active. Calls after execute settles are ignored.

Core events use top-level snake_case fields. `agent_end` includes messages. Tool events include tool_call_id and tool_name; start/update include args, update includes partial_result, and end includes result, is_error and optional duration_ms. `message_update` contains message and assistant_message_event. Core events do not use the research Event's versioned diagnostic envelope; do not feed them to `decode_event`.

Core state uses is_streaming, streaming_message, pending_tool_calls and error_message, alongside messages, model, thinking_level, tools and system_prompt. It is a detached snapshot. Use prepare hooks for contextual changes, or the research API's explicit configuration methods; assigning a snapshot does not change the agent.

## Shared provider configuration

The following changes affect both APIs:

- An explicitly supplied `ModelInfo.base_url` wins. Without an explicit model record, a custom Provider endpoint remains in force; default OpenAI/Anthropic endpoints can use the catalog model endpoint. Endpoint resolution also drives compatibility and identity headers. Concurrent requests never mutate the shared Provider.
- Headers merge case-insensitively: provider defaults, then model.headers, then request options.headers. Request None suppresses a default, including Authorization, Accept and Content-Type. Applications using custom authentication or proxy headers must verify the new precedence.
- Chat Completions merges model sampling_params, then sampling_params_by_thinking_level, then per-request sampling_params. These keys do not add sampling support to providers that lack it.
- Request options timeout_ms, max_retries and max_retry_delay_ms override the HTTP transport settings. Zero timeout disables the timeout; zero max_retry_delay_ms disables the retry-delay cap. Other requests retain their transport settings. WebSocket timeout covers opening and reads; HTTP fallback receives the same options. This does not add replay retries after network acceptance or model-stream failure.
- on_payload/on_response/on_provider_stream_event receive `(value, model)`. One-argument callbacks remain supported through a signature-based migration bridge. A callback is never rerun after raising TypeError.

`ModelInfo` now includes headers and sampling defaults in dataclass serialization. Update strict model-record serializers before adopting those fields. Existing message schemas 4/3 and event schemas 2/1 retain the previously documented read/write rules. A tool call that carries `arguments_error` can only be decoded by a reader that knows this field. ProxyProvider retains its separate server-owned protocol and does not resolve direct-provider configuration locally.

## Remaining Python boundaries

This is a selected Pi v1.4.0 contract implementation, not complete Pi compatibility. Python retains snake_case, the documented message/enum mapping, Unix-second timestamps, copied state, strict input/output validation and asyncio cancellation/cleanup rules. Durable replay, arbitrary custom TypeScript message roles and the complete provider/UI API remain outside this implementation. Transport constructor defaults remain the Python defaults unless explicitly overridden.

Use `require_features(["pi-core-contracts-v1", "model-request-settings-v1"])` to detect this build. The interface probes compare shared stream helpers and core entry points with pinned upstream code, and separately record research API differences. Full event payloads and request behavior have additional Python regression tests. External applications and real provider accounts need their own integration validation.
