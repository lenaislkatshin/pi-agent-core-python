# 模型接入与订阅登录

[English](../PROVIDERS.md) | **中文**

这次扩展将真实模型接入放在 `pi_python.providers`，执行循环继续使用同一个 `Agent`。Claude 使用 Messages API，GPT 使用 Responses API，本地模型和其他 OpenAI 兼容服务使用 Chat Completions；订阅登录分别实现 Claude OAuth、ChatGPT 直接授权和 Codex 授权。核心循环保留 Pi `v1.0.0` 基线；模型协议差分使用单独固定的 `v1.4.0` 源码，不代表完整移植 `pi-mono` 应用。

## 安装和 API key

```bash
pip install pi-python-core
# 用 ChatGPT 账号登录（openai-chatgpt）时另需：
pip install 'pi-python-core[oauth]'
# 在源码目录里：
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
        result = await agent.prompt("解释这个项目的工具执行流程")
        print(result.status, result.messages[-1].content)

asyncio.run(main())
```

库本身不搜索环境变量、不读取其他应用的凭据。上例由应用显式读取密钥。`api_key` 也可传同步或异步零参数函数。`Agent(get_api_key=callback)` 在每次请求前调用 `callback(provider_name)`，返回值优先；返回 None 则使用 Provider 自身凭据。模型名称由调用者提供，须是账号实际可用的模型。

## 三种订阅路径

| 登录方式 | OAuthClient 的 provider | 推理 Provider | 请求端点 |
|---|---|---|---|
| Claude 订阅 | `anthropic` | `AnthropicProvider` | `api.anthropic.com/v1/messages` |
| ChatGPT 直接授权 | `openai-chatgpt` | `OpenAIProvider` | `api.openai.com/v1/responses` |
| Codex 订阅授权 | `openai-codex` | `OpenAICodexProvider` | `chatgpt.com/backend-api/codex/responses` |

ChatGPT 直接授权要求返回 `chatgpt.tokens.use.direct` 权限；动态注册回调返回的 client ID 用于换取和刷新令牌。程序验证 ID token 的签名、发行者、接收方、有效期、nonce 和重登录账号。这项校验使用 PyJWT，需要安装 `[oauth]`。主机 UUID 要持久化并复用。依据：[OpenAI 注册与登录说明](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)。账号可用性、模型权限和配额以服务端实际授权为准。

可运行的例子是 [provider_chat.py](../../examples/provider_chat.py)。它只在明确执行 `--login` 时打开浏览器，凭据存到调用者指定的文件，采用原子替换和 Unix 0600 权限。

```bash
# MODEL_ID 由你设置为账号支持的型号。
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

后续调用去掉 `--login`，例子会复用该文件并在到期前刷新。不要将其他客户端的凭据文件交给此例子。它演示单进程使用；多个进程共享凭据时，应用还需要跨进程锁。

嵌入应用可以使用：

- `OAuthClient.begin(...)`：生成 PKCE、state、授权 URL。`exchange(attempt, callback_url)` 校验回调并换取凭据。Claude 的 `method="copy_code"` 支持 `code#state`。
- `await OAuthClient.login(on_auth_url, on_prompt=None, ...)`：先监听 loopback，再调用 UI 打开 URL。端口被占用时可通过 `on_prompt` 接收粘贴的完整回调。默认 5 分钟超时；支持 `CancelToken`。只有 ChatGPT 直接授权可通过 `redirect_uri` 更换 loopback 端口。
- `await OAuthClient("openai-codex").device_login(on_device_code)`：设备码流程，支持等待、取消和超时。
- `RefreshingCredentials(credential, persist=callback)`：串行刷新，避免同一对象并发消耗旧 refresh token。刷新成功但持久化失败时保留新令牌，下次先重试持久化。

将 `RefreshingCredentials` 传给 Provider 的 `credentials=` 即可。凭据对象的 repr 隐藏令牌，库不会记录令牌响应正文。

## 推理、图片、观察接口

```python
from pi_python import Agent, ImageContent

agent = Agent(
    provider=provider, model=model_id,
    thinking_level="medium", transport="sse", session_id="application-session",
    on_payload=lambda payload: None,
    on_response=lambda response: print(response["status"]),
    on_provider_stream_event=lambda event: None,
)
# 图片数据是 base64 字符串，不是路径或 URL。
result = await agent.prompt("描述图片", images=[ImageContent(image_base64, "image/png")])
```

三个观察回调都支持 async。`on_payload(payload)` 可以原地修改请求或返回替代 dict；`on_response({status, headers})` 观察 HTTP/SSE 或 WebSocket 握手；`on_provider_stream_event(event)` 接收原始供应商事件的副本。它们属于关键调用路径，异常会停止本次请求。回调看到的请求或事件可能包含应用内容，应用自行决定如何记录。

`ThinkingContent` 保存可见推理、供应商签名和 redacted 标记。Claude 支持 thinking/signature delta 与 redacted thinking；OpenAI 保存完整 reasoning item（含 encrypted_content），以及文本 item ID / phase。向相同 provider/API/model 回放签名；跨模型不复用不透明签名。标准 Pi 内容中的图片用于用户消息和工具结果，助手消息使用文本、推理和工具调用。

Claude 的推理方式由模型记录决定：`compat["forceAdaptiveThinking"]` 为真的模型使用 effort（自适应推理），其他推理模型使用 token 预算；输出上限和等级映射也来自同一条记录。要改变它们，传一个修改过的 `ModelInfo`，Provider 本身没有覆盖选项。与上游一致，开启推理时请求 `display: "summarized"`（Opus 4.7 起服务端默认不返回推理文字），可用 `options["thinking_display"]="omitted"` 改回；推理关闭且模型允许时发送 `thinking: {"type": "disabled"}`；预算模式附带 interleaved-thinking beta。输出上限按 Pi 的字符估算扣除已用上下文和 4096 个安全 token，最少为 1。

Claude 模型若支持 `supportsMidConvoEffort`，在开启推理时使用历史强度标记；模型允许时，显式 `off` 会关闭推理。历史标记保留此前强度，末尾追加本次强度（未指定时为 high）。支持此能力的 OpenAI API 模型使用 `configuration_update`，在历史回答前及输入末尾记录强度，顶层 effort 保持 medium。OpenAI 此路径要求显式启用推理，不用于 Codex 或直接 ChatGPT 登录。`provider_thinking_level` 记录原生强度；`thinking_level` 记录请求等级。

OpenAI 将 `thinking_level` 映射到 reasoning effort。`options` 支持 `max_tokens`、`reasoning_summary`、`temperature`、`top_p`、`tool_choice`、`parallel_tool_calls`、`metadata`、`service_tier`、`text`、`headers`；Claude 另支持 `thinking_budgets`、`thinking_display`、`interleaved_thinking`、`top_k`、`stop_sequences`，以及直接覆盖的 `thinking` 和 `output_config`。与上游一致，Claude 只转发 `metadata.user_id`，字符串形式的 `tool_choice` 写成 `{"type": ...}`；推理开启、模型不支持或使用强度标记时不发送 `temperature`。具体模型允许哪些参数由服务端决定。

Claude 使用 SSE。OpenAI/Codex 支持 `sse`、`websocket`、`websocket-cached` 和 `auto`。`websocket-cached` 要求应用显式提供 `session_id`；与上游一致，`auto` 在有显式 `session_id` 时也复用连接。Agent 自动管理的缓存标识不会启用上述行为。`cache_retention="none"` 时每次请求使用一次性连接。连接按端点、会话和请求头隔离，同一连接串行使用；空闲 5 分钟或建立满 55 分钟后不再复用，已被服务端关闭的连接也会重建，而不是在旧连接上失败。时限可用 `HTTPTransport(websocket_idle_ttl=..., websocket_max_age=...)` 调整。

每次 WebSocket 请求及归还连接时清扫过期的空闲连接。默认最多保留 64 条空闲连接，超限时先关闭最早归还的连接；可用 `HTTPTransport(websocket_cache_size=...)` 调整，设为 `0` 则不保留连接。会话锁在最后一个运行或等待中的请求退出后移除。清扫由请求触发，没有后台定时器。结束后调用 `await provider.aclose()`；共享 transport 由应用管理生命周期。

如果取消与取得 HTTP 响应、WebSocket 或请求锁同时发生，transport 会先释放已取得的资源，再传播取消异常。调用方重复取消不会打断这次释放。传入的 HTTP 客户端仍由应用管理。

Responses、Messages 和 Chat Completions 的流内错误会在 `ProviderProtocolError` 中保留错误类型、代码和消息，让恢复逻辑识别上下文超限、暂时过载和额度耗尽。HTTP 和流内错误共用凭据脱敏规则，错误详情最多保留 4,000 个字符。

Codex 复用连接时采用上游的增量请求：如果本次请求除输入外完全相同，且输入以上次输入加上次回答开头，只发送新增部分和 `previous_response_id`。任何其他情况都发送完整上下文。服务端回复找不到上一回答（`previous_response_not_found`）或连接数已满（`websocket_connection_limit_reached`）且尚未输出内容时，换新连接以完整上下文重试一次；此时服务端尚未生成内容，不会重复计费。`auto` 只在握手失败、请求尚未发出时回退 SSE；请求发出后的断线不重发，这比上游保守。Python 的 WebSocket 帧不带 `stream` 字段，上游会带；0.3 和 0.4 的实际 Codex 联调均不需要它。直接调用 OpenAI API 的 WebSocket 是本库扩展，上游没有对应路径，因此仍发送完整上下文。

`HTTPTransport(max_retries=2, retry_base=0.5, max_retry_delay=30)` 只重试服务器明确返回的 429/500/502/503/504。最多额外两次；遵守 `Retry-After`，超过等待上限就返回错误，不提前重试。等待支持取消。已开始流式输出、发送后连接中断、鉴权失败及工具执行均不自动重试。OAuth 换令牌请求不采用这套重试。Business 工作区用量限制即使返回 HTTP 429 也不重试。OAuth 刷新开始后，调用方取消会等待刷新和持久化完成；网络刷新单独限时 15 秒，应用的持久化回调必须能结束。写回失败保留 dirty 状态，下次调用先重试写回。

`ProviderHTTPError` 提供 `status`、`category`、`retry_after`、`request_id`、`retryable`。Provider 将失败转换为模型 `error` 事件，结构化 HTTP 信息放入 `message.diagnostics`；Agent 的错误摘要仍为字符串。`transport.stats` 返回计数副本：HTTP 尝试/重试，WebSocket 建连、复用、回退、过期替换、一次性重试和增量请求。

`cache_retention` 接受 `none` / `short` / `long`。Claude 在系统、工具定义和最后一条用户或系统消息上设置缓存标记；long 使用 1 小时。OpenAI 通过 `session_id` 设置缓存键（超过 64 字符时截断），long 请求 24 小时；支持显式缓存模式的模型（模型表 `supportsExplicitPromptCacheMode`，如 GPT-5.6、GPT-6 系列）改用 `prompt_cache_options`：none 为显式模式，long 为 30 分钟。none 不发送缓存键及关联会话请求头。DeepSeek 的 Responses 适配去掉不支持的显式缓存字段。缓存是否命中由服务端决定。

OpenAI 对已知模型默认发送模型输出上限（按上下文估算截短，最少 16）。使用 ChatGPT 登录令牌直连 `api.openai.com` 时（令牌不以 `sk-` 开头），与上游一样不发送服务端拒绝的 `prompt_cache_retention`、`prompt_cache_options`、`max_output_tokens` 和 `temperature`。

用量归一化为 `input`、`output`、`cache_read`、`cache_write`、`reasoning`、`total_tokens`。工具自己的用量独立保存在工具结果中，不混入主模型 token 计数。

## Codex 自动缓存会话

从 0.11.0 起，使用 `OpenAICodexProvider` 的 Agent 在应用未传 `session_id` 时，自动提供稳定的缓存标识，用于 `session-id`、`x-client-request-id` 请求头和 `prompt_cache_key`。这针对 ChatGPT Codex 订阅通道，不适用于 OpenAI API key 通道或 Claude。Codex CLI [明确说明会话头用于缓存亲和性](https://github.com/openai/codex/blob/e4568a3538c9cab6627d64ff8cdfc7d98b79b7db/codex-rs/core/src/client.rs#L660)；发送标识不保证命中，也不保证具体的订阅配额节省。

标识属于 Agent，不属于共享 Provider 或单次请求。同一 Agent 的工具循环、后续 prompt、重试和配置替换保持标识；不同 Agent（包括子 Agent）各自生成。`reset()` 丢弃自动管理的标识，包括恢复时传入的标识，下次读取或请求时生成新值。应用显式传入的 `session_id` 始终优先，并保持原有 reset 行为。自动请求头尊重应用自行设置的同名头，包括不同大小写形式。

自动标识**不会启用连接复用或上一响应续接**。默认传输仍为 SSE。使用 `auto` 或 `websocket` 时，除非应用显式传入 `session_id` 并选择支持复用的模式，否则仍使用一次性连接、发送完整上下文。`websocket-cached` 仍要求显式 `session_id`。缓存亲和性与传输会话状态分开管理。

用 `Agent(auto_session_id=False, ...)` 恢复旧默认行为；该开关不屏蔽显式 `session_id`。`options={"cache_retention": "none"}` 对两类标识都停止自动设置缓存键和关联会话头。

将只读属性 `agent.cache_session_id` 与历史一起保存，并通过构造参数 `cache_session_id=` 恢复：

```python
from pi_python import Agent, encode_messages, decode_messages

# provider 是应用管理的 OpenAICodexProvider。
agent = Agent(provider=provider, model=model_id)
await agent.prompt("记住任务约束。")
saved = {
    "messages": encode_messages(list(agent.state.messages)),
    "cache_session_id": agent.cache_session_id,
}
restored = Agent(
    provider=provider, model=model_id,
    messages=decode_messages(saved["messages"]),
    cache_session_id=saved["cache_session_id"],
)
await restored.prompt("继续任务。")
```

上述对象由应用自行持久化，消息编码格式没有变化。`cache_session_id` 表示 Agent 管理的后备标识；当显式 `session_id` 或自定义请求头覆盖它时，该属性不代表实际请求的路由标识。自行管理 `session_id` 的应用仍需保存自己的配置。通过 `cache_session_id=` 恢复保持“只用于缓存”；改用 `session_id=` 恢复则会采用原有传输语义。恢复值须为 1–64 个不含空格的可打印 ASCII 字符，且 `auto_session_id` 须为 True。关闭自动处理或当前 Provider 未声明支持时，该属性返回 None。

core 通过可选的 Provider 能力 `supports_agent_cache_session = True` 判断，不检查模型名，也不导入 Codex 适配器。内置 Provider 中只有 Codex 开启它。自定义 Provider 可主动声明支持并读取 `request.cache_session_id`；已有 Provider 和 `stream_fn` 不需要修改。主动声明支持的包装器应保留原请求或使用 `deepcopy`；通过序列化的 dataclass 字段重建请求会丢失这项缓存专用元数据。默认流包装器会转发底层 Provider 的能力。标识不放入 `request.options`，不增加 `ModelRequest` 的 dataclass 字段，也不增加消息、事件或结果字段。直接调用底层 Provider 不会自动生成标识。

这是与官方 Pi core 的有意差异：上游底层 Agent 只转发可选标识，由 coding-agent 的 `SessionManager` 提供；pi-python 在主要的 Agent 接入入口补齐该便利功能。上游参照版本不变。应用可用 `require_features(["codex-auto-cache-session-v1"])` 检查支持情况。兼容性与验证范围见 [0.11.0 发布说明](../releases/0.11.0.md)。

## 请求身份

`Agent(identity=...)` 接收 JSON 客户端身份，在运行开始时复制。每次请求的 `options["identity"]` 逐字段覆盖它；替换其他 options 不会清掉 Agent 的默认身份。直接调用 Provider 也可以传此选项。可用 `require_features(["client-identity-v1"])` 检查支持情况。

```python
identity = {
    "user_agent": "detailed",  # 也可用 "short" 或 {"custom": "myapp/1"}
    "attribution": {"name": "myapp", "url": "https://example.org", "categories": ["cli-agent"]},
}
agent = Agent(provider=llm, model=model, identity=identity)
```

默认 User-Agent 包含 `pi-python` 版本、Python 运行时和平台；默认不发送应用归属信息。本库是独立 Python 实现，不将请求归属于官方 Pi 应用。显式 attribution 映射到 OpenRouter、NVIDIA、OpenCode 请求头；`attribution=None` 关闭继承的归属信息。身份相关显式 headers 优先于生成值，值为 None 时移除该头，包括 HTTP 客户端默认值。Anthropic OAuth 保留必需的 Claude CLI User-Agent，除非显式 headers 覆盖。固定版本的代理协议未实现 identity 转发。迁移和精确来源见 [跟进说明](../UPSTREAM_FOLLOWUP.md)。

## 模型能力表和 DeepSeek

```python
from pi_python import Agent, ModelCatalog, ModelInfo
catalog = ModelCatalog.bundled()
model = catalog.get("openai", "gpt-5.4-mini")
print(model.context_window, model.max_tokens, model.input)
print(model.supported_thinking_levels())
print(model.clamp_thinking_level("minimal"))  # low
provider = OpenAIProvider(api_key=key, catalog=catalog)

# 表外的新模型：给出完整记录，能力只有这一个来源。
custom = ModelInfo("my-model", "openai", "openai-responses", "Mine",
                   context_window=200_000, max_tokens=32_000, reasoning=True)
agent = Agent(provider=provider, model=custom)
```

随包提供 68 条记录：Claude 17、OpenAI 42、Codex 7、DeepSeek 2。记录含上下文和输出上限、输入模态、推理等级映射、缓存、基础价格和兼容信息。Provider 用它选择 Claude 推理模式、映射 effort，并提前拒绝已知模型超输出上限或不支持的图片。上下文 token 数仍由服务端检查，没有本地 tokenizer 或自动压缩。真实 Provider 遇到表中没有的模型名会报 `ConfigurationError`，不会套用默认值：可以直接 `Agent(model=ModelInfo(...))`，也可以 `catalog.register(...)` 后把 catalog 传给 Provider；读写均复制数据。

Claude/OpenAI/Codex 模型表来自 `@earendil-works/pi-ai@1.4.0` 发布包，保留两条 Python DeepSeek Responses 覆盖记录。Codex 不再内置 `gpt-5.5` 和 `gpt-5.3-codex-spark`；应用需选择受支持的模型（例如 `gpt-6.1-sol`）或自行提供核实过的 `ModelInfo`。**不是固定 Git 提交的一部分，也不保证实时可用或代表账号权限**。来源、采集日期及摘要随 JSON 保存。`estimate_cost(usage)` 只用快照基础费率估算美元，不包括阶梯、时段、服务等级或订阅计费。兼容信息是模型元数据，不表示本库实现了其中所有供应商专有功能。

DeepSeek 使用官方当前文档中的无状态 Responses 接口。它把系统消息放入 `instructions`，映射 `none/low/high/max` effort，并支持 reasoning_text 流和重放。这是针对 [DeepSeek Responses 文档](https://api-docs.deepseek.com/guides/responses_api/) 的新增适配；固定 Pi 快照为 DeepSeek 配置的是 Chat Completions，不能把此适配说成与该上游路径逐项相同。

```bash
uv run python examples/provider_chat.py --provider deepseek --model deepseek-flash
```

## 本地模型与 OpenAI 兼容服务

`OpenAICompletionsProvider` 接入任何 OpenAI 兼容的 `/chat/completions` 接口：Ollama、vLLM、llama.cpp server、LM Studio、SGLang，以及 DeepSeek、Groq、OpenRouter、Together、通义等云服务。它逐项移植自上游 `openai-completions.ts`。

```python
from pi_python import Agent
from pi_python.providers import OpenAICompletionsProvider

llm = OpenAICompletionsProvider(base_url="http://localhost:11434/v1", name="ollama")
agent = Agent(provider=llm, model=llm.model("qwen3:8b", context_window=40960, max_tokens=8192))
result = agent.prompt_sync("用一句话介绍你自己")
```

- `base_url` 是 `/chat/completions` 之前的部分，按原样使用，例如 Ollama 的 `http://localhost:11434/v1`、vLLM 的 `http://localhost:8000/v1`。
- `api_key` 可省略。省略时不发送认证头，本地服务通常不需要密钥。上游在没有密钥时直接报错，它的文档建议填一个占位密钥。
- 模型不会被猜测。`llm.model(id, ...)` 声明这个服务上的一个模型；只给 `id` 时使用上游为自定义模型定的默认值：上下文 128000、输出 16384、只收文本、不推理、零费用。知道真实上限时请写明，输出上限会随每次请求发送。表外模型名会报 `ConfigurationError`，并给出这一行该怎么写。
- 推理模型可以传 `reasoning=True`；Ollama、vLLM、SGLang 上的推理模型，上游建议加 `compat={"supportsDeveloperRole": False, "supportsReasoningEffort": False}`。上游的 27 个兼容开关都会读取：已知开关的取值写错时报 `ConfigurationError`，未知开关忽略，与 Responses 接入相同。
- `options` 接受 `max_tokens`、`reasoning`、`thinking_budgets`、`temperature`、`tool_choice`、`sampling_params`（最后合并进请求体）、`cache_retention`、`session_id` 和 `headers`。
- 服务端要支持工具调用才能用工具。例如 vLLM 需要用 `--enable-auto-tool-choice` 和对应模型的 `--tool-call-parser` 启动。

与上游的差异：
- 最终的工具参数按严格 JSON 解析，上游会先宽松地修补。不合法的参数记入 `ToolCall.arguments_error`，工具不执行，错误返回给模型。因长度截断的工具调用同样不执行。
- 没有参数的工具调用会补一个 `"{}"` 增量。
- 缺少工具调用 ID 时会生成一个（`call_<hex>`）。
- 格式错误的数据块会报错，上游则会强行转换。
- 端点只看 Provider 的 `base_url`，不看 `ModelInfo.base_url`，与其他 Python Provider 一致。

以下内容没有移植：OpenAI grammar 自定义工具（收到时报 `UnsupportedCapabilityError`）、GitHub Copilot 请求头、`PI_CACHE_RETENTION` 环境变量，以及模型级的 `headers` 和 `samplingParams` 字段（`ModelInfo` 没有这两个字段，可用 `options` 代替）。示例见 [examples/local_model.py](../../examples/local_model.py)；它默认使用进程内的替身服务器，加 `--base-url` 和 `--model` 即连接真实服务。

## 会话中途的变更

应用可以在会话中途改变工具、系统指令或推理强度：`agent.update_config(AgentConfigUpdate(tools=[...]))` 会在下一次输入前记录一条工具增删的系统消息；`agent.prompt([SystemMessage("新的指令"), UserMessage("...")])` 追加指令；`SystemMessage(sections={"rules": "..."})` 按名称替换或删除段落。历史里保留每一次变化。

模型支持时，这些变化在原位置发送，已缓存的前缀保持不变；不支持时，所有系统消息合并成开头的一条，工具列表换成当前集合。具体由模型表决定：

| 模型能力（模型表字段） | 发送方式 |
|---|---|
| Claude：`supportsMidConvoSystemMessages` 与 `supportsMidConvoToolChanges` | 后续系统消息以 system 角色放在下一条回答之前；`tool_addition` 携带完整内联定义，包括同名重定义；删除用 `tool_removal`；请求级初始工具列表固定，并预先声明一个永不可用的占位工具，使缓存前缀从第一个请求起就稳定 |
| OpenAI / Codex：`supportsMidConvoSystemMessages` | 后续系统消息以 developer 角色原位发送 |
| OpenAI / Codex：`supportsAdditionalTools` 或 `supportsToolSearch` | 新增工具通过 `additional_tools`，或客户端执行的 tool search 记录，在原位置加载 |
| 以上均不支持 | 合并为开头一条系统消息，发送当前完整工具列表 |

支持 `supportsAdditionalTools` 时，初始和后续工具都通过 `additional_tools` 发送，developer 消息解释增量添加、替换和删除。仅支持 tool search 时，将最后一次删除或重定义之前的工具折叠为初始列表，之后的添加仍原位发送。Claude 原生变更要求初始工具，否则发送当前完整列表。后续系统段落仍写成 `Updated system prompt section "名称": ...` 或 `Removed system prompt section "名称".`。

## streamProxy

```python
from pi_python import Agent, ProxyProvider

provider = ProxyProvider(
    model=server_model_descriptor,  # 包含 id/provider/api 的完整服务端模型描述
    proxy_url="https://your-proxy.example",
    auth_token=fetch_proxy_token,   # 字符串或同步/异步零参数函数
)
agent = Agent(provider=provider, model=server_model_descriptor["id"])
```

发送上游 `{model, context, options}` 到 `/api/stream`。Python 字段在边界转换为 Pi wire 命名，保留工具 schema 和参数中的原始用户字段。只序列化上游白名单选项。支持文本、thinking、工具调用、签名和完成事件；流截断、终止后数据或参数不一致导致失败，不执行工具。

独立函数 `stream_proxy(model, context, options, cancel=None)` 返回 `ModelEvent` 异步迭代器，context 可以是消息列表或 `ModelRequest`，options 必须包含 `proxy_url` 与 `auth_token`。

## 验证边界

[协议差分](../../compat/results/provider-conformance.json) 包含 58 组共享输入（其中 17 组是 Chat Completions，覆盖 Ollama、vLLM、llama.cpp、OpenRouter、DeepSeek 等兼容配置和全部 11 种推理参数格式），对照实际固定上游的内容、签名、停止原因、事件顺序和载荷，以及完整请求体和选定语义请求头。原有 12 组会话样例保留当时的模型描述（Claude Opus 5.5、Opus 4.8、Sonnet 4.5、Fable 5，GPT-5.5、GPT-4.1、GPT-5.6，Codex GPT-5.5），走 pi-agent-core 实际调用的 `streamSimple` 入口，覆盖会话中途的系统、工具和推理强度变化。[WebSocket 差分](../../compat/results/websocket-conformance.json) 用脚本化连接对照多轮请求的每一帧和连接使用。SDK 标识头、动态请求 ID 和请求压缩不在差分声明中；代理的请求体另由 Python 单测覆盖。

真实联调使用本地已有 Claude/Codex 订阅授权和 DeepSeek API key，只读取、不刷新其他客户端的凭据，不保存令牌或模型正文。0.4 的[会话变更联调](../../compat/results/live-session-claude.json)与 [Codex 增量联调](../../compat/results/live-session-codex.json)验证：中途加入的工具被模型调用、中途指令被遵守、推理强度标记被接受、Codex 后续请求只发送增量。0.3 的文本、工具、推理及签名重放矩阵已在 0.4 重跑。Claude 与 Codex 的新登录和令牌刷新已用本库自己登录得到的凭据文件实测（[记录](../../compat/results/live-refresh.json)，命令 `scripts/live_refresh.py`）：三个并发调用者只触发一次网络刷新并拿到同一新令牌；刷新令牌轮换后原子写回、权限保持 0600；会话中途过期时 Provider 先刷新再发请求；从文件再次刷新成功，例子 CLI 不加 `--login` 即可复用。刷新令牌会轮换，所以这项测试只能用本库登录得到的文件，不能用其他客户端的文件。刷新成功但写文件失败、等待中取消等故障由本地测试覆盖。OpenAI API key、ChatGPT 直接 OAuth、多进程共享同一凭据文件、真实限流和断网恢复仍未做账号侧验证。

HTTP 失败时，错误文字带上服务器返回的正文，与上游一样最多 4000 字符；名字里含 key、token、auth、secret、cookie 或 password 的请求头所带的凭据，在正文中出现时（包括 JSON 转义和 URL 编码的形式）替换为 `[redacted]`，短于 8 个字符的不替换。OAuth 换令牌请求的失败不带正文。

本机没有运行中的本地模型服务，Chat Completions 接入尚未连接真实服务器验证；它的请求和解析由上面的差分以及模拟服务器测试覆盖。

仍未实现供应商约束采样（OpenAI grammar 工具与严格 JSON schema 工具）、Gemini/Vertex、Bedrock、Mistral 和 Azure 的原生接口、后台 deferred 轮询、音视频生成、模型目录自动更新，以及完整 pi-mono 应用。未知输出显式报错。严格参数验证、状态副本、未知工具结果不重放属于设计选择，单独记在验收映射中。

OpenAI Responses、Anthropic 和 Chat Completions 在上游未报告用量时，会在
`AssistantMessage.diagnostics` 加入 `{"type": "usage_unavailable"}`。为兼容旧接口保留的
归一化默认零值不代表实测消耗；MCP 的用量观察事件在此情况下返回 `usage=None`。
