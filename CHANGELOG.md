# Changelog

本文件记录 PRP Runtime 的显著变更。

`0.0.2` 是当前包身份。数据库 schema 直接替换，pre-0.1 数据不提供向后兼容或迁移。

## [0.0.2] - 2026-08-22

### Legal

- License is Apache-2.0. The earlier dual-license wording was replaced; see
  `NOTICE` and `TRADEMARKS.md` for attribution and branding boundaries.

### Added

- 四种正交 Agent mode：`NORMAL`、`AUTO`、`PLAN`、`YOLO`；确定性 Policy 对工具产生可审计的 `ALLOW`、`ASK` 或 `DENY`。
- Cloud/Bridge 两种执行位置、`SANDBOXED`/`HOST` 隔离边界、owner-scoped Workspace、Snapshot、ToolCall/ToolResult、Approval、Lease 和 ChangeSet 合同。
- `list_files`、`read_file`、`search_text`、`apply_patch`、`run_targeted_test`、`get_diff`、`get_status` 受限工具子集；写入基于 Snapshot 生成 ChangeSet，测试命令使用预注册结构化 argv。
- Native Session/Run API、单租户服务令牌、SSE 事件回放与 Bridge 的无模型断线游标恢复。
- Responses、Chat Completions 和 Anthropic Messages 的声明子集映射到共享 Agent 核心；不声称完整第三方协议兼容。
- final node、revision lineage、内容指纹、原子 reservation、Provider readiness/恢复、隔离并行和 Progressive 证据修订。

### Security and limits

- `PLAN` 无副作用；`AUTO` 只自动放行低风险命令；模型不能自批权限、访问任意主机路径、保存秘密或启用通用网络。
- `SANDBOXED` 必须有真实 Linux `bubblewrap`；Bridge 的本地 Workspace path boundary 不等于 OS sandbox。
- 本版是单实例 SQLite 参考实现，不提供生产 SLA、无限推理、完整 Codex/Claude Code 兼容或多租户计费。

### Verification

- v0.0.2 package identity、最小云端代码任务（list/search/read/patch/targeted test/diff）和 API/Bridge 定向门禁的源码与定向证据已记录。
- 相关证据来自仓库 conformance/integration tests 与 `ai/RELEASE-EVIDENCE-MATRIX.md`、`ai/PROTOCOL-VALIDATION.json`、`ai/PROVIDER-CAPABILITIES.json`；这些记录不是持续 CI、benchmark、SLA 或所有 provider 通过证明。

## [0.0.1] - 未发布

### Added

- 项目元数据 `pyproject.toml`，Python 要求 `>=3.12`。
- Apache-2.0 许可证与 `NOTICE`。
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
