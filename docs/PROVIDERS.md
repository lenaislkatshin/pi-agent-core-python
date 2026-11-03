# Model connectors and subscription sign-in

**English** | [中文](zh/PROVIDERS.md)

Real model connectors live in `pi_python.providers`, and the run loop keeps using the same `Agent`. Claude uses the Messages API, GPT uses the Responses API, and local models and other OpenAI-compatible services use Chat Completions. Subscription sign-in implements Claude OAuth, direct ChatGPT authorization and Codex authorization. The core loop retains its Pi `v1.0.0` baseline; provider protocol comparisons now use separately pinned `v1.4.0` sources; it is not a complete port of the `pi-mono` applications.

## Installation and API keys

```bash
pip install pi-python-core
# Signing in with a ChatGPT account (openai-chatgpt) also needs:
pip install 'pi-python-core[oauth]'
# In a source checkout:
uv sync --locked
```

```python
import asyncio
import os
from pi_python import Agent
from pi_python.providers import AnthropicProvider, OpenAIProvider, DeepSeekProvider

async def main():
    provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
    # Claude: AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
    # DeepSeek: DeepSeekProvider(api_key=os.environ["DEEPSEEK_API_KEY"])
    async with Agent(provider=provider, model=os.environ["MODEL_ID"]) as agent:
        result = await agent.prompt("Explain how this project executes tools")
        print(result.status, result.messages[-1].content)

asyncio.run(main())
```

The library does not search environment variables or read other applications' credentials; in the example above, the application reads the key explicitly. `api_key` can also be a sync or async zero-argument function. `Agent(get_api_key=callback)` calls `callback(provider_name)` before each request, and its return value takes precedence; if it returns None, the Provider's own credentials are used. The caller supplies the model name, which must be a model the account can actually use.

## Three subscription paths

| Sign-in | OAuthClient provider | Inference Provider | Request endpoint |
|---|---|---|---|
| Claude subscription | `anthropic` | `AnthropicProvider` | `api.anthropic.com/v1/messages` |
| Direct ChatGPT authorization | `openai-chatgpt` | `OpenAIProvider` | `api.openai.com/v1/responses` |
| Codex subscription authorization | `openai-codex` | `OpenAICodexProvider` | `chatgpt.com/backend-api/codex/responses` |

Direct ChatGPT authorization requires the `chatgpt.tokens.use.direct` scope in the response; the client ID returned by the dynamic-registration callback is used to exchange and refresh tokens. The program verifies the ID token's signature, issuer, audience, validity period and nonce, and the account on re-login. This verification uses PyJWT, so it needs the `[oauth]` extra. The host UUID must be persisted and reused. Source: [OpenAI's sign-in documentation](https://developers.openai.com/siwc/token-sharing-open-source/sign-in). Account availability, model access and quota depend on what the server actually grants.

The runnable example is [provider_chat.py](../examples/provider_chat.py). It opens a browser only when you explicitly pass `--login`, and it stores credentials in a file the caller names, with atomic replacement and Unix 0600 permissions.

```bash
# Set MODEL_ID to a model your account supports.
uv run python examples/provider_chat.py \
  --provider openai --model "$MODEL_ID"

uv run python examples/provider_chat.py \
  --provider anthropic --model "$MODEL_ID" \
  --credential-file ~/.config/pi-python/claude.json --login

uv run --extra oauth python examples/provider_chat.py \
  --provider openai-chatgpt --model "$MODEL_ID" \
  --credential-file ~/.config/pi-python/chatgpt.json --login

uv run python examples/provider_chat.py \
  --provider openai-codex --model "$MODEL_ID" \
  --credential-file ~/.config/pi-python/codex.json --login --device
```

For later calls, drop `--login`; the example reuses the file and refreshes the token before it expires. Do not give this example another client's credential file. It demonstrates single-process use; when several processes share credentials, the application also needs a cross-process lock.

Embedding applications can use:

- `OAuthClient.begin(...)`: generates the PKCE pair, the state and the authorization URL. `exchange(attempt, callback_url)` validates the callback and exchanges it for credentials. Claude's `method="copy_code"` accepts `code#state`.
- `await OAuthClient.login(on_auth_url, on_prompt=None, ...)`: starts listening on loopback first, then calls the UI to open the URL. If the port is busy, `on_prompt` can receive the full pasted callback. It times out after 5 minutes by default and supports `CancelToken`. Only direct ChatGPT authorization can change the loopback port through `redirect_uri`.
- `await OAuthClient("openai-codex").device_login(on_device_code)`: the device-code flow, with waiting, cancellation and timeout.
- `RefreshingCredentials(credential, persist=callback)`: serializes refreshes, so one object never spends an old refresh token twice concurrently. If a refresh succeeds but persisting fails, the new token is kept and persisting is retried first next time.

Pass `RefreshingCredentials` to a Provider's `credentials=`. The credential object's repr hides tokens, and the library never logs token response bodies.

## Reasoning, images and observation callbacks

```python
from pi_python import Agent, ImageContent

agent = Agent(
    provider=provider, model=model_id,
    thinking_level="medium", transport="sse", session_id="application-session",
    on_payload=lambda payload: None,
    on_response=lambda response: print(response["status"]),
    on_provider_stream_event=lambda event: None,
)
# Image data is a base64 string, not a path or URL.
result = await agent.prompt("Describe the image", images=[ImageContent(image_base64, "image/png")])
```

All three observation callbacks may be async. `on_payload(payload)` can modify the request in place or return a replacement dict; `on_response({status, headers})` observes HTTP/SSE responses or the WebSocket handshake; `on_provider_stream_event(event)` receives a copy of each raw provider event. They sit on the critical call path, and an exception stops the request. The requests or events they see may contain application content; the application decides how to log them.

`ThinkingContent` stores visible reasoning, the provider signature and a redacted flag. Claude supports thinking/signature deltas and redacted thinking; OpenAI keeps the full reasoning item (including encrypted_content), as well as text item IDs and phase. Signatures are replayed to the same provider/API/model; opaque signatures are not reused across models. In standard Pi content, images appear in user messages and tool results; assistant messages use text, reasoning and tool calls.

Claude's reasoning mode is decided by the model record: models with `compat["forceAdaptiveThinking"]` set use effort (adaptive reasoning), and other reasoning models use a token budget; the output limit and the level mapping come from the same record. To change them, pass a modified `ModelInfo`; the Provider itself has no override option. As upstream does, requests ask for `display: "summarized"` when reasoning is on (from Opus 4.7 on, the server returns no reasoning text by default), which `options["thinking_display"]="omitted"` turns back off; when reasoning is off and the model allows it, `thinking: {"type": "disabled"}` is sent; budget mode adds the interleaved-thinking beta. The output limit subtracts the context already used, estimated with Pi's character method, and 4096 safety tokens, with a minimum of 1.

Claude models with `supportsMidConvoEffort` use historical effort markers while thinking is enabled. An allowed explicit `off` disables thinking. Historical markers retain the earlier effort; the current marker defaults to high when unspecified. Supported OpenAI API models use `configuration_update` items before historical answers and at the end of the input, while the top-level effort stays medium. This OpenAI path requires an explicit enabled reasoning level and does not apply to Codex or direct ChatGPT sign-in. `provider_thinking_level` records the native effort; `thinking_level` records the requested level.

OpenAI maps `thinking_level` to reasoning effort. `options` supports `max_tokens`, `reasoning_summary`, `temperature`, `top_p`, `tool_choice`, `parallel_tool_calls`, `metadata`, `service_tier`, `text` and `headers`; Claude also supports `thinking_budgets`, `thinking_display`, `interleaved_thinking`, `top_k`, `stop_sequences`, and direct overrides of `thinking` and `output_config`. As upstream does, Claude forwards only `metadata.user_id`, writes a string `tool_choice` as `{"type": ...}`, and does not send `temperature` when reasoning is on, when the model does not support it, or when effort markers are in use. Which parameters a given model accepts is up to the server.

Claude uses SSE. OpenAI and Codex support `sse`, `websocket`, `websocket-cached` and `auto`. `websocket-cached` requires an application-supplied `session_id`; as upstream does, `auto` also reuses the connection when there is an application-supplied `session_id`. The Agent-managed cache ID does not enable either behavior. With `cache_retention="none"`, each request uses a one-off connection. Connections are kept apart by endpoint, session and request headers, and each connection is used by one request at a time. A connection is no longer reused after 5 idle minutes or 55 minutes after it was opened, and a connection the server has closed is rebuilt instead of failing on the old one. These limits can be changed with `HTTPTransport(websocket_idle_ttl=..., websocket_max_age=...)`.

Expired idle connections are swept on WebSocket requests and cache returns. At most 64 idle connections are retained, evicting the least recently returned first; set `HTTPTransport(websocket_cache_size=...)` to change this limit, or `0` to disable retention. Session locks are removed when their last active or waiting request exits. Sweeping is demand-driven, with no background timer. Call `await provider.aclose()` when you are done; the application manages the lifetime of a shared transport.

If cancellation races with acquiring an HTTP response, WebSocket or request lock, the transport releases the acquired resource before propagating cancellation. Repeated caller cancellation cannot interrupt that release. A supplied HTTP client remains owned by the application.

Streaming error events from Responses, Messages and Chat Completions retain their error type, code and message in `ProviderProtocolError`, so recovery logic can recognize context overflow, temporary overload and exhausted quota. HTTP and streaming errors share credential redaction and a 4,000-character detail cap.

When Codex reuses a connection, it uses upstream's incremental requests: if this request is identical to the previous one except for its input, and the input starts with the previous input followed by the previous answer, only the new part is sent, together with `previous_response_id`. In every other case the full context is sent. If the server replies that the previous answer cannot be found (`previous_response_not_found`) or that the connection limit is reached (`websocket_connection_limit_reached`) before any output, the request is retried once on a new connection with the full context; the server has not generated anything yet, so nothing is billed twice. `auto` falls back to SSE only when the handshake fails before the request is sent; a disconnect after the request is sent is not retried, which is more conservative than upstream. Python's WebSocket frames do not include the `stream` field, which upstream's do; the real Codex runs for 0.3 and 0.4 did not need it. WebSocket for the direct OpenAI API is an extension of this library with no upstream counterpart, so it still sends the full context.

`HTTPTransport(max_retries=2, retry_base=0.5, max_retry_delay=30)` retries only a 429/500/502/503/504 that the server explicitly returns, at most two extra times. It respects `Retry-After`; if the requested wait exceeds the cap, it returns the error instead of retrying early. The wait can be cancelled. Streams that have started output, connections broken after sending, authentication failures and tool execution are never retried automatically. OAuth token requests do not use this retry logic. The Business workspace usage-limit code is not retried even on HTTP 429. Once OAuth refresh starts, caller cancellation waits for the refresh and persistence to finish. Network refresh has a separate 15-second deadline; the application persistence callback must terminate. A failed save stays dirty and is retried before another refresh.

`ProviderHTTPError` provides `status`, `category`, `retry_after`, `request_id` and `retryable`. Providers turn failures into a model `error` event and put the structured HTTP information in `message.diagnostics`; the Agent's error summary is still a string. `transport.stats` returns a copy of the counters: HTTP attempts and retries, and WebSocket connections opened, reused, fallbacks, expired replacements, one-off retries and incremental requests.

`cache_retention` accepts `none` / `short` / `long`. Claude sets cache markers on the system prompt, the tool definitions and the last user or system message; long uses 1 hour. OpenAI sets the cache key through `session_id` (truncated beyond 64 characters), and long requests 24 hours. Models that support the explicit cache mode (`supportsExplicitPromptCacheMode` in the model table, such as the GPT-5.6 and GPT-6 series) use `prompt_cache_options` instead: none is the explicit mode, and long is 30 minutes. none sends no cache key and none of the related session headers. The DeepSeek Responses adapter drops the explicit cache fields DeepSeek does not support. Whether the cache hits is up to the server.

For known models, OpenAI sends the model's output limit by default (shortened by the context estimate, at least 16). When a ChatGPT sign-in token is used directly against `api.openai.com` (the token does not start with `sk-`), the library, like upstream, does not send `prompt_cache_retention`, `prompt_cache_options`, `max_output_tokens` or `temperature`, which the server rejects in that case.

Usage is normalized to `input`, `output`, `cache_read`, `cache_write`, `reasoning` and `total_tokens`. When the upstream omits usage, OpenAI Responses, Anthropic, and Chat Completions add
`{"type": "usage_unavailable"}` to `AssistantMessage.diagnostics`. Their legacy normalized
counters remain present for compatibility; those default zeros are not measured consumption.
MCP sampling observations report `usage=None` for this case. A tool's own usage is stored separately in the tool result and is not mixed into the main model's token counts.

## Automatic Codex cache sessions

From 0.11.0, `Agent` supplies a stable cache ID when using `OpenAICodexProvider` without an application-supplied `session_id`. It fills the `session-id` and `x-client-request-id` headers and `prompt_cache_key`. This applies to the ChatGPT Codex subscription channel, not to OpenAI API-key calls or Claude. The Codex CLI [identifies the session header as the cache-affinity input](https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/core/src/client.rs#L660); sending an ID does not guarantee a cache hit or a particular subscription quota saving.

The ID belongs to the Agent, not to the shared Provider or an individual request. Tool loops, later prompts, retries and configuration replacement keep it. Different Agents, including nested Agents, receive independent IDs. `reset()` discards the managed ID, including a restored one; the next access or request generates a new ID. An explicit `session_id` always takes precedence and retains its existing reset behavior. Automatic headers respect application-supplied header values, including differently cased names.

The automatic ID **never enables connection reuse or response continuation**. The default transport remains SSE. In `auto` or `websocket` mode it still uses one-off connections and full context unless the application explicitly supplies `session_id` to an eligible mode. `websocket-cached` still requires an explicit `session_id`. Cache affinity and transport session state are separate.

Use `Agent(auto_session_id=False, ...)` to keep the previous default. This does not suppress an explicit `session_id`. `options={"cache_retention": "none"}` suppresses library-generated cache keys and session headers for either kind of ID.

Save the read-only `agent.cache_session_id` alongside history and restore it through the constructor's `cache_session_id=` argument:

```python
from pi_python import Agent, encode_messages, decode_messages

# provider is an application-owned OpenAICodexProvider.
agent = Agent(provider=provider, model=model_id)
await agent.prompt("Remember the task constraints.")
saved = {
    "messages": encode_messages(list(agent.state.messages)),
    "cache_session_id": agent.cache_session_id,
}
restored = Agent(
    provider=provider, model=model_id,
    messages=decode_messages(saved["messages"]),
    cache_session_id=saved["cache_session_id"],
)
await restored.prompt("Continue the task.")
```

The application chooses how to store this object; message encoding itself has not changed. `cache_session_id` is the Agent-managed fallback, not a report of a request's effective routing ID when an explicit `session_id` or custom headers override it. Applications that manage `session_id` must persist that configuration themselves. Restoring with `cache_session_id=` retains cache-only behavior; restoring it as `session_id=` opts into the existing transport semantics. Restored cache IDs must contain 1–64 printable ASCII characters without spaces and require `auto_session_id=True`. The property returns `None` when automatic handling is disabled or the current Provider does not opt in.

The core uses an optional Provider capability, `supports_agent_cache_session = True`, rather than inspecting model names or importing a Codex adapter. Only the built-in Codex Provider enables it. Custom Providers may opt in and read `request.cache_session_id`; existing Providers and `stream_fn` implementations require no changes. Wrappers that opt in must preserve the request object or use `deepcopy`: rebuilding it from its serialized dataclass fields drops this cache-only metadata. The default-stream wrapper forwards the underlying Provider's capability. The ID is separate from `request.options`, and does not add dataclass fields to `ModelRequest` or fields to messages, events or results. Calling a Provider directly does not generate an ID.

This is an intentional difference from Pi's core Agent, which only forwards an optional ID. Pi's coding-agent layer supplies it through `SessionManager`; pi-python provides that convenience at its primary Agent entry point. The upstream baseline is unchanged. Applications can check `require_features(["codex-auto-cache-session-v1"])` before using this API. See the [0.11.0 compatibility and verification notes](releases/0.11.0.md).

## Request identity

`Agent(identity=...)` supplies JSON client identity, copied at run start. `options["identity"]` overrides it field by field on each request; replacing other options does not erase the Agent default. Direct Provider callers can use the same option. `require_features(["client-identity-v1"])` checks support.

```python
identity = {
    "user_agent": "detailed",  # or "short", or {"custom": "myapp/1"}
    "attribution": {"name": "myapp", "url": "https://example.org", "categories": ["cli-agent"]},
}
agent = Agent(provider=llm, model=model, identity=identity)
```

Defaults identify `pi-python` with its version, Python runtime and platform; attribution is absent unless supplied. This independent Python library does not attribute requests to the official Pi application. Attribution maps to OpenRouter, NVIDIA and OpenCode headers. An explicit `attribution=None` disables inherited attribution. Identity-related request headers override generated identity; a `None` header removes it, including HTTP-client defaults. Anthropic OAuth keeps its required Claude CLI user agent unless an explicit header overrides it. Proxy forwarding of identity is not implemented by the pinned upstream protocol. See [migration and exact pins](UPSTREAM_FOLLOWUP.md).

## Model table and DeepSeek

```python
from pi_python import Agent, ModelCatalog, ModelInfo
catalog = ModelCatalog.bundled()
model = catalog.get("openai", "gpt-5.4-mini")
print(model.context_window, model.max_tokens, model.input)
print(model.supported_thinking_levels())
print(model.clamp_thinking_level("minimal"))  # low
provider = OpenAIProvider(api_key=key, catalog=catalog)

# A new model not in the table: give a complete record; it is the only source of capabilities.
custom = ModelInfo("my-model", "openai", "openai-responses", "Mine",
                   context_window=200_000, max_tokens=32_000, reasoning=True)
agent = Agent(provider=provider, model=custom)
```

The package ships 68 records: Claude 17, OpenAI 42, Codex 7, DeepSeek 2. Records include context and output limits, input modalities, the reasoning-level mapping, caching, base prices and compatibility information. Providers use them to choose Claude's reasoning mode and map effort, and to reject in advance, for known models, requests that exceed the output limit or send unsupported images. The server still checks the context token count; there is no local tokenizer and no automatic compaction. When a real Provider meets a model name that is not in the table, it raises `ConfigurationError` instead of applying defaults: either pass `Agent(model=ModelInfo(...))` directly, or call `catalog.register(...)` and pass the catalog to the Provider. Both reads and writes copy the data.

The Claude/OpenAI/Codex model table is the snapshot published in `@earendil-works/pi-ai@1.4.0`; the two Python DeepSeek Responses overrides are retained. Codex `gpt-5.5` and `gpt-5.3-codex-spark` are no longer bundled; choose a supported model (for example `gpt-6.1-sol`) or supply your own verified `ModelInfo`. **It is not part of the pinned Git commit, and it does not guarantee real-time availability or reflect account permissions.** The source, collection date and digest are saved with the JSON. `estimate_cost(usage)` estimates US dollars from the snapshot's base rates only, without tiers, time-of-day pricing, service tiers or subscription billing. The compatibility information is model metadata and does not mean this library implements every provider-specific feature it mentions.

DeepSeek uses the stateless Responses interface from its current official documentation. It puts system messages in `instructions`, maps `none/low/high/max` effort, and supports reasoning_text streams and replay. This is a new adapter written for [DeepSeek's Responses documentation](https://api-docs.deepseek.com/guides/responses_api/); the pinned Pi snapshot configures DeepSeek through Chat Completions, so this adapter cannot be described as matching that upstream path item by item.

```bash
uv run python examples/provider_chat.py --provider deepseek --model deepseek-flash
```

## Local models and OpenAI-compatible services

`OpenAICompletionsProvider` connects to any OpenAI-compatible `/chat/completions` endpoint: Ollama, vLLM, llama.cpp server, LM Studio and SGLang, and cloud services such as DeepSeek, Groq, OpenRouter, Together and Qwen. It is ported item by item from upstream's `openai-completions.ts`.

```python
from pi_python import Agent
from pi_python.providers import OpenAICompletionsProvider

llm = OpenAICompletionsProvider(base_url="http://localhost:11434/v1", name="ollama")
agent = Agent(provider=llm, model=llm.model("qwen3:8b", context_window=40960, max_tokens=8192))
result = agent.prompt_sync("Introduce yourself in one sentence")
```

- `base_url` is the part before `/chat/completions` and is used as is, for example `http://localhost:11434/v1` for Ollama and `http://localhost:8000/v1` for vLLM.
- `api_key` can be omitted. Without it, no authentication header is sent; local servers usually need no key. Upstream raises an error when there is no key, and its documentation suggests a placeholder key.
- Models are never guessed. `llm.model(id, ...)` declares one model on this server. With only `id`, it uses upstream's defaults for custom models: context 128000, output 16384, text only, no reasoning, zero cost. If you know the real limits, state them; the output limit is sent with every request. A model name that is not in the table raises `ConfigurationError`, which shows the line you should write.
- Reasoning models can pass `reasoning=True`. For reasoning models on Ollama, vLLM and SGLang, upstream recommends adding `compat={"supportsDeveloperRole": False, "supportsReasoningEffort": False}`. All 27 of upstream's compatibility switches are read: a wrong value for a known switch raises `ConfigurationError`, and unknown switches are ignored, as with the Responses connector.
- `options` accepts `max_tokens`, `reasoning`, `thinking_budgets`, `temperature`, `tool_choice`, `sampling_params` (merged into the request body last), `cache_retention`, `session_id` and `headers`.
- Tools work only if the server supports tool calling. For example, vLLM must be started with `--enable-auto-tool-choice` and the `--tool-call-parser` for the model.

Differences from upstream:
- Final tool arguments are parsed as strict JSON; upstream repairs them leniently first. Arguments that are still invalid are stored in `ToolCall.arguments_error`, the tool does not run, and the error goes back to the model. A tool call cut off by the length limit is not executed either.
- A tool call with no arguments gets a `"{}"` delta.
- A missing tool-call ID is generated (`call_<hex>`).
- Malformed data chunks raise an error; upstream coerces them.
- The endpoint comes only from the Provider's `base_url`, not from `ModelInfo.base_url`, consistent with the other Python Providers.

Not ported: OpenAI grammar custom tools (receiving one raises `UnsupportedCapabilityError`), GitHub Copilot request headers, the `PI_CACHE_RETENTION` environment variable, and the model-level `headers` and `samplingParams` fields (`ModelInfo` has neither; use `options` instead). See [examples/local_model.py](../examples/local_model.py); by default it uses an in-process stand-in server, and with `--base-url` and `--model` it connects to a real one.

## Mid-session changes

The application can change tools, system instructions or reasoning effort mid-session: `agent.update_config(AgentConfigUpdate(tools=[...]))` records a system message with the tool additions and removals before the next input; `agent.prompt([SystemMessage("new instructions"), UserMessage("...")])` appends instructions; `SystemMessage(sections={"rules": "..."})` replaces or deletes sections by name. History keeps every change.

When the model supports it, these changes are sent where they occurred and the cached prefix stays unchanged; when it does not, all system messages are merged into one at the start and the tool list becomes the current set. The model table decides which:

| Model capability (model-table field) | How it is sent |
|---|---|
| Claude: `supportsMidConvoSystemMessages` and `supportsMidConvoToolChanges` | Later system messages go in the system role before the next answer; `tool_addition` carries complete inline definitions (including redefinitions), and removals use `tool_removal`; the request-level initial tool list stays fixed, and a placeholder tool that is never available is declared up front so the cached prefix is stable from the first request |
| OpenAI / Codex: `supportsMidConvoSystemMessages` | Later system messages are sent in place in the developer role |
| OpenAI / Codex: `supportsAdditionalTools` or `supportsToolSearch` | New tools are loaded in place through `additional_tools`, or through client-executed tool search records |
| None of the above | Merged into one system message at the start, with the current full tool list |

With `supportsAdditionalTools`, initial and later tools are sent through `additional_tools`; developer messages describe incremental additions, replacements and removals. With tool-search-only support, the prefix through the last removal or redefinition is folded into the initial tool list; later additions stay in place. Claude native changes require an initial tool, otherwise the current tool list is sent. Section changes still use `Updated system prompt section "name": ...` / `Removed system prompt section "name".`.

## streamProxy

```python
from pi_python import Agent, ProxyProvider

provider = ProxyProvider(
    model=server_model_descriptor,  # the full server-side model descriptor, with id/provider/api
    proxy_url="https://your-proxy.example",
    auth_token=fetch_proxy_token,   # a string, or a sync/async zero-argument function
)
agent = Agent(provider=provider, model=server_model_descriptor["id"])
```

Sends upstream's `{model, context, options}` to `/api/stream`. Python fields are converted to Pi's wire names at the boundary, and the original user fields in tool schemas and arguments are preserved. Only the options on upstream's allow list are serialized. Text, thinking, tool calls, signatures and completion events are supported; a truncated stream, data after termination, or inconsistent arguments cause a failure, and no tool is executed.

The standalone function `stream_proxy(model, context, options, cancel=None)` returns an async iterator of `ModelEvent`; context can be a message list or a `ModelRequest`, and options must include `proxy_url` and `auth_token`.

## What has been verified

The [protocol comparison](../compat/results/provider-conformance.json) contains 58 shared inputs (17 of them for Chat Completions, covering compatible configurations such as Ollama, vLLM, llama.cpp, OpenRouter and DeepSeek and all 11 reasoning-parameter formats). They are compared against the pinned upstream's actual content, signatures, stop reasons, event order and payloads, as well as the full request body and selected semantic request headers. The original twelve session cases retain their historical model descriptors (Claude Opus 5.5, Opus 4.8, Sonnet 4.5 and Fable 5; GPT-5.5, GPT-4.1 and GPT-5.6; Codex GPT-5.5) through the `streamSimple` entry point that pi-agent-core actually calls, and cover mid-session changes to system prompts, tools and reasoning effort. The [WebSocket comparison](../compat/results/websocket-conformance.json) uses scripted connections to compare every frame and every use of a connection across multi-turn requests. SDK identification headers, dynamic request IDs and request compression are outside what the comparison claims; the proxy's request body is covered separately by Python unit tests.

Testing against real accounts used existing local Claude/Codex subscription authorizations and a DeepSeek API key. It only read other clients' credentials, never refreshed them, and saved no tokens or model text. In 0.4, the [mid-session change test](../compat/results/live-session-claude.json) and the [Codex incremental test](../compat/results/live-session-codex.json) verified that tools added mid-session were called by the model, mid-session instructions were followed, effort markers were accepted, and Codex follow-up requests sent only the increment. The 0.3 matrix for text, tools, reasoning and signature replay was rerun in 0.4. New Claude and Codex sign-ins and token refresh were tested with credential files obtained through this library's own sign-in ([record](../compat/results/live-refresh.json), command `scripts/live_refresh.py`): three concurrent callers triggered only one network refresh and got the same new token; after the refresh token rotated, it was written back atomically with permissions kept at 0600; when the session expired mid-way, the Provider refreshed before sending the request; refreshing again from the file succeeded, and the example CLI reused the file without `--login`. Because refresh tokens rotate, this test can only use files from this library's own sign-in, not files from other clients. Failures such as a successful refresh followed by a failed file write, or cancellation while waiting, are covered by local tests. The OpenAI API key path, direct ChatGPT OAuth, several processes sharing one credential file, real rate limiting and recovery from network loss have not yet been verified against real accounts.

When an HTTP request fails, the error text includes the response body returned by the server, at most 4000 characters, as upstream does. Credentials carried in request headers whose names contain key, token, auth, secret, cookie or password are replaced with `[redacted]` wherever they appear in the body (including JSON-escaped and URL-encoded forms), except values shorter than 8 characters. Failed OAuth token requests carry no body.

No local model server was running on the development machine, so the Chat Completions connector has not yet been verified against a real server; its requests and parsing are covered by the comparison above and by mock-server tests.

Not yet implemented: provider-constrained sampling (OpenAI grammar tools and strict JSON-schema tools); native Gemini/Vertex, Bedrock, Mistral and Azure interfaces; background deferred polling; audio and video generation; automatic model-catalog updates; and the complete pi-mono applications. Unknown output raises an explicit error. Strict argument validation, state copies and not replaying unknown tool results are design choices, recorded separately in the coverage map.
