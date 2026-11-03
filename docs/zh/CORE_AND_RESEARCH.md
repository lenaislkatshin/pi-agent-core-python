# 核心合同与研究入口

[English](../CORE_AND_RESEARCH.md)

两种入口都在 `pi-python-core` 包内，共用同一个 Run、模型接入和工具执行引擎。没有移除研究便利能力，不需要安装第二个包。这两个入口从 0.12.0 起提供，升级注意事项见[发布与迁移说明](../releases/0.12.0.md)。

日常研究继续使用 `from pi_python import Agent, tool`。它保留 RunResult、普通函数工具、同步调用、直接执行工具批次、资源限制、失败回答重试和支持该能力的 Provider 自动缓存标识。现有插件也继续构造这个 Agent。

需要跟随 Pi 合同的集成使用 `from pi_python import core`。`core.Agent` 和核心循环使用上游对应的钩子结构、结果及事件载荷。旧顶层 Hooks 和低层函数保留为迁移入口；新的上游对齐代码使用 core。两个入口通过适配共用引擎，不维护两套循环。

可直接运行两个无网络示例：

```bash
uv run python examples/core_and_research.py
```

## 选择入口

| 使用方式 | 研究入口 | 核心入口 |
|---|---|---|
| 等待 prompt | 得到 RunResult | 返回 None，通过 state 或 agent_end.messages 取结果 |
| 事件订阅 | listener(Event)，带运行编号等诊断封装 | listener(event_dict, signal) |
| 结束钩子 | 旧 Hooks 返回字符串 end/continue | 返回 `{ "action": "end" }` 等对象 |
| 工具执行函数 | @tool，或 execute(args, context) | execute(call_id, args, signal, on_update) |
| 最后回答失败后继续 | 支持便利重试 | 低层 continue 拒绝 assistant 结尾；Agent 可以消费排队消息 |
| 缓存会话 | 对声明支持的 Provider 自动管理，可关闭 | 仅显式 session_id |
| 低层事件流 | 仅迭代时有界；等待 result 后解除背压 | 无界；等待结果不消费已排队事件 |

研究 Agent 也接受 core.Hooks 和 core.Tool，可以先迁移钩子，再迁移事件消费者。旧顶层 Hooks 保留原签名。研究入口的 stream_fn 接收 request、cancel；核心入口的 stream_fn 接收 model、context、options，取消令牌在 options["signal"] 中。核心自定义流返回异步事件迭代器，也可以先异步取得迭代器。

## 钩子与事件迁移

core.Agent 可以传 hooks=core.Hooks(...)，也可以直接传 finish_turn 等命名回调。成组 Hooks 使用低层的上下文签名。Agent 的命名 prepare_next_turn 仅接收 signal；需要上下文时使用 prepare_next_turn_with_context(context, signal)。core.AgentLoopConfig 可以直接设置钩子字段，或使用成组 Hooks。

- before_tool_call 接收上下文和 signal；上下文含 assistant_message、tool_call、验证后的 args 和 AgentContext。返回 block/reason/terminate 字典或 None。
- after_tool_call 还会收到 result、is_error。返回部分结果字典；省略键表示保留，显式 None 可以清空可空元数据。替换 content 而未同时提供 structured_content 时，旧结构化内容被清除。
- prepare_request 收到 AgentContext、model、thinking_level；返回 AgentRequestUpdate 或对应字典，不能追加 messages。
- 低层 prepare_next_turn 收到已完成轮次上下文；返回 AgentLoopTurnUpdate，可以追加消息，context 替换包含 messages 和 tools。
- finish_turn 返回 action 字典或 None。
- Python 的工具 on_update 是异步回调，执行期间应 await；工具结束后的更新调用被忽略。

核心事件使用顶层 snake_case 字段。agent_end 含 messages；工具事件含 tool_call_id、tool_name，start/update 含 args，update 含 partial_result，end 含 result、is_error 和可选 duration_ms。message_update 保留 message 与 assistant_message_event。核心事件不采用研究入口 Event 的版本化诊断封装，不能交给 decode_event 读取。

核心 state 提供 messages、model、thinking_level、tools、system_prompt、is_streaming、streaming_message、pending_tool_calls、error_message，仍是脱离内部状态的副本。上下文变更可通过 prepare 钩子完成；需要直接配置管理时可使用研究入口。修改副本不会修改 Agent。

## 两种入口共用的请求配置

- 显式传入的 ModelInfo.base_url 优先；未传显式模型记录时，Provider 的自定义地址仍有效，默认 OpenAI/Anthropic 地址可采用目录模型地址。兼容性检测和身份头也使用解析后的地址；并发请求不会修改共享 Provider。
- header 按大小写无关的“Provider 默认 → 模型 headers → 单次 options.headers”合并。请求值 None 可以移除默认头，包括 Authorization、Accept、Content-Type。自定义认证和代理需要验证新优先级。
- Chat Completions 依次合并模型 sampling_params、对应推理等级的 sampling_params_by_thinking_level、单次 sampling_params。不会因此给其他 Provider 增加其原本不支持的采样参数。
- timeout_ms、max_retries、max_retry_delay_ms 覆盖单次 HTTP 请求的传输设置。timeout_ms=0 关闭超时，max_retry_delay_ms=0 关闭重试等待上限；不影响其他请求。WebSocket 超时用于建连和读取，回退 HTTP 时保留请求配置。没有增加网络请求已被接受后的自动重放或模型流重试。
- on_payload、on_response、on_provider_stream_event 接收 value、model。单参数旧回调通过签名适配继续支持；回调内部抛出 TypeError 后不会换一种签名再次执行。

ModelInfo 的 dataclass 序列化现在包含 headers 和采样默认值；使用严格模型记录解析器的应用需要接受这些字段。消息/事件历史继续遵守此前 schema 4/3、2/1 的读写规则。带 `arguments_error` 的工具调用只有认识这个字段的读取器才能解码。ProxyProvider 仍由代理服务管理模型和请求策略，不在客户端应用直接 Provider 的配置规则。

## 保留的边界和验证范围

这是选择性的 Pi v1.4.0 合同实现，不是整个 Pi 的完整移植。保留 Python 命名、已说明的消息/枚举映射、Unix 秒时间戳、状态副本、严格参数/输出验证和 asyncio 取消清理规则。持久化重放、任意 TypeScript 自定义消息角色及完整 Provider/UI API 不在此次实现范围。没有显式覆盖时，传输构造器仍使用 Python 版本的默认值。

可用 require_features 检查 pi-core-contracts-v1、model-request-settings-v1。接口探针对照固定上游，分别记录 core 入口与研究入口；完整事件和请求行为另有 Python 回归。外部应用、真实数据库、代理和模型账户仍需独立集成验证。
