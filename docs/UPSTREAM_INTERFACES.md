# Pi 接口对齐状态（0.12.0，2026-10-28）

用户在审查后批准“核心合同尽量跟随官方，Python 的额外价值放在明确的扩展层”，并强调研究使用的易用性。本次已实施同包内的两个公开入口：`pi_python.core` 提供官方对应合同，`pi_python.Agent` 保留研究便利能力。两者共用同一执行引擎，没有拆包或删除研究能力。

完整使用和迁移步骤见 [核心合同与研究入口](zh/CORE_AND_RESEARCH.md)（[English](CORE_AND_RESEARCH.md)）。原先“仅修复流等待、其余暂缓”的阶段性决定已被后续实施授权替代。

## 参照与证据范围

接口参照固定为 [Pi v1.4.0 / 4a080252](https://github.com/earendil-works/pi/tree/4a080252f06caf63a525d7873c5d94d3c23887e0)。原有 core 行为差分仍保留 v1.0.0 基线，Provider/插件选择性跟随 v1.4.0。这里不声明全量 Pi API 等价。

[接口探针](../scripts/interface_audit.py) 核验上游源码后，运行实际 [官方实现](../reference/interface-audit.ts)，比较共享流规则和 core 入口，同时记录研究入口的差异。结果见 [interface-audit.json](../compat/results/interface-audit.json)。若核心观测与上游不同，命令失败；该命令已纳入完整验证。测试不访问真实模型账户。

| 原先发现的问题 | 当前处理 | 研究入口的兼容安排 |
|---|---|---|
| 历史转发被错误计时、新 deferred 被排除 | 按 timestamp 判定，保留已有耗时；内置 Provider 记录请求开始时间 | 两层共用修正，Python 时间戳仍用秒 |
| 部分消费后等待 result 卡住 | result 不依赖继续消费，保留未读事件 | 研究流仅迭代时仍有界；core 流无界 |
| 低层预先取消没有事件 | core 循环发出生命周期并保留取消回答 | 旧顶层低层入口保留作迁移桥接 |
| 从失败 assistant 调用低层 continue | core 低层拒绝所有 assistant 结尾 | 研究 Agent 保留便利重试；core Agent 可消费排队消息 |
| finish-turn 返回字符串 | core 接受 action 对象 | 旧 Hooks 保留字符串；研究 Agent 也接受 core.Hooks |
| before/after 钩子参数和结果不同 | core 提供上下文+signal、block 决定及部分结果覆盖；区分省略与显式 None | 通过边界适配进入同一工具执行引擎 |
| prepare-request 与 next-turn 共用更新类型 | core 区分 AgentRequestUpdate 与 AgentLoopTurnUpdate；仅后者可追加消息 | 旧 TurnUpdate 仍服务既有调用方 |
| 事件缺少信息或结构不一致 | core 的 agent_end 含 messages，工具事件含名称/参数/结果，message_update 含完整对应模型事件 | 旧 Event 的诊断封装和 schema 不被静默替换 |
| 自定义流签名不同 | core 的 stream_fn 接收 model、context、options | 研究入口保留 request、cancel |
| ModelInfo 地址未贯通 | 显式模型记录的地址贯通请求、兼容性与身份；请求独立解析 | 自定义 Provider 端点在未传显式模型记录时保留 |
| 模型 headers 和采样默认值缺失 | ModelInfo 支持 headers、sampling_params、sampling_params_by_thinking_level | 新 dataclass 字段需要严格解析器迁移；采样合并用于 Completions |
| 请求头覆盖顺序不统一 | 默认→模型→请求，大小写无关，None 可移除默认头 | 会改变旧认证/代理覆盖行为，需集成验证 |
| 逐请求超时/重试选项未转发 | 支持 timeout_ms、max_retries、max_retry_delay_ms，包含 HTTP 回退；不修改共享传输设置 | 未覆盖时保留 Python transport 构造器默认值 |
| 回调缺少模型参数 | 三类 Provider 观察回调接收 value、model | 单参数旧回调按签名适配；异常后不重试回调 |
| 自动缓存便利行为进入低层 | core 不生成自动缓存标识 | 研究 Agent 保留经过用户批准的自动缓存能力 |

## 保留的有意差异与未实现范围

- Python 的 snake_case、消息/枚举映射和 Unix 秒时间戳仍保留。队列模式同时接受官方 one-at-a-time 与旧 one_at_a_time。
- core state 的字段名称对应官方，但仍是副本；没有引入对内部消息的任意原地修改。研究 API 继续提供配置更新、RunResult、同步调用、普通函数工具、应用工具批次、预算与插件。
- 两层仍共用严格输入/输出检查、状态复制、Python Task 取消、清理期限和未知结果处理。这些是 [已有设计差异](../compat/COVERAGE.md)，不能算成已消除的偏差。
- core.Tool 提供 label、parameters 和官方对应执行参数；Python 的 on_update 为异步回调。没有实现 durable replay，不能把 replay 标记当作可恢复执行保证。最终工具参数不合法时记入 `ToolCall.arguments_error`，由工具校验把错误返回给模型，对应 Pi 1.5 的 `argumentsError`。Pi 的 `repairJson` 没有移植，解析仍是严格 JSON。
- ProxyProvider 继续遵循代理服务自己的请求合同，不套用直接 Provider 的本地模型配置解析。
- 未移植完整 UI、持久任务、全部供应商及 TypeScript 开放自定义消息类型。请求身份保持 pi-python 品牌和默认无 attribution，不冒用官方应用身份。

## 测试与迁移边界

[新增回归](../tests/test_core_contracts.py) 检查钩子输入与覆盖语义、事件字段、层间默认行为隔离、模型地址/header/采样优先级、回调次数、并发隔离、真实本地 HTTP 超时及核心流。它们是 Python 行为测试；不能把全部断言都称为双端差分。

[双入口示例](../examples/core_and_research.py) 离线执行同一个工具任务。完整检查命令、退出状态及输出由 [verification.json](../compat/results/verification.json) 记录。Linux 矩阵的旧记录不自动证明本轮新增代码。

旧消息 schema 3 和事件 schema 1 仍按此前规则读取；写出保持消息 4、研究事件 2。core 事件是独立的标准字典入口，不交给 decode_event。外部数据库、严格订阅器、代理部署和真实模型账户尚未逐一验证，不能据库内回归作全面兼容保证。

后续维护同时检查字段合同与行为样例。原有差分运行器会省略部分事件字段；接口探针与完整载荷回归用于补上这类盲点。新增差异应明确属于语言适配、便利能力、未移植范围还是待修复问题。
