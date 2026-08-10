# Changelog

本文件记录 PRP Runtime 的显著变更。

版本阶段 `0.0.1` 到 `0.0.4` 属于 pre-0.1 开发期，不保留向后兼容，不提供数据迁移。

## [0.0.1] - 未发布

### Added

- 项目元数据 `pyproject.toml`，Python 要求 `>=3.12`。
- 双许可证 `MIT OR Apache-2.0` 与 `NOTICE`。
- Python 包骨架 `src/prp_runtime`，暴露包身份信息。
- 严格设置层 `Settings`：仅接受白名单 `PRP_` 环境变量，未识别变量直接报错；模型 profile 经 JSON 环境变量注入，`api_key` 以 `SecretStr` 持有，不出现在 `repr` 或序列化中。
- 原生领域合同：`NativeRunRequest`、`Run`、`WorkUnit`、`Attempt`、`Artifact`、`Evidence`、`Budget`、`Usage`、`ControllerDecision`，全部 frozen 且拒绝未知字段，无 chain-of-thought 字段。
- 四执行策略枚举与独立的 `AUTO`/`MANUAL` 路由策略；三张显式状态机转换表（Run/WorkUnit/Attempt），非法跳转抛结构化错误。
- 结构化错误层：稳定 `code` + `family` + `retryable`，不含堆栈或凭据。
- 追加式事件账本 `RunEvent`：Run 内 `sequence` 单调唯一，payload 为受限 JSON。
- SQLite 持久化：单一 schema、外键与 WAL、`user_version` 版本闸门（不兼容即明确失败并提示删除开发库）；状态变更与事件同事务提交；事件序号由数据库原子分配。
- 重启恢复：仍在 `RUNNING` 的 Attempt 标记为 `INTERRUPTED`，不假定成功或失败；恢复幂等，已完成数据不被改写。
- 出站 Provider 契约与一个 OpenAI-compatible 文本适配器：归一化 usage / finish reason / 错误分类，错误正文截断并脱敏；上游未报告 token 时记为不可用。
- `DIRECT` 执行策略：单 WorkUnit、单 Attempt，不调用 Planner 或 Verifier；Worker 只写 Attempt 层事实，Controller 是 Run/WorkUnit 状态的唯一写入口。
- PRP Native API：`POST /v1/runs`、`GET /v1/runs/{run_id}`、`POST /v1/runs/{run_id}/cancel`、`GET /v1/runs/{run_id}/events`（SSE 回放，支持 `Last-Event-ID` 与 `?after=` 游标）、`GET /health`。
- 请求体与输入长度上限，超限返回 `413` 与稳定错误码。

### Not included

- `CASCADE`、`PLANNED`、`PROGRESSIVE` 策略；显式请求时返回结构化 `400`，不静默降级为 `DIRECT`。
- Planner、Verifier、预算强制、图调度与并行执行。
- OpenAI Responses / Chat Completions 与 Anthropic Messages 入站绑定。
- 出站流式调用；`/events` 是持久账本回放，不是上游流的转发。
