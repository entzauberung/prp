# Changelog

本文件记录 PRP Runtime 的显著变更。

`0.0.4` 是当前包身份。数据库 schema 直接替换，pre-0.1 数据不提供向后兼容或迁移。

## [0.0.4] - 2026-09-04

### Delivered

- 包身份改为 `0.0.4`。BRIDGE 协议版本与包身份对齐。
- 服务器脑 + 本机手：BRIDGE 不再走服务器工具 handler，也不解析客户端根目录。
- 持久化 Bridge 客户端身份、能力、心跳和 snapshot；claim 绑定 `client_id`，跨客户端领取被拒绝。
- Progressive 将具体 ToolCall 派发给服务器选定的 Bridge 客户端，远程等待与审批暂停分离，同一 Run 可恢复。
- Bridge 结果成为 Artifact/Evidence/ChangeSet；AST/静态事实在服务器侧对有界返回源运行。
- CLOUD bundle 绑定被请求 Run 的已验证 snapshot，并从 `snapshot_files.content` 读取，不读 live workspace 根。Git merge、fact merge、local snapshot 和 Bridge patch 在创建时写入这份内容。
- Analyzer/Verifier 角色不再映射到 Worker profile。

### Limits

- Bridge manifest 默认不保存客户端文件字节。CLOUD 导出只使用创建 snapshot 时捕获的内容。
- 仍是单实例 SQLite 参考实现，无 Docker、分布式队列或生产 SLA。

### Not included

- 完整第三方协议实现、迁移框架、计费、SSO 或 Kubernetes。

## [0.0.3] - 2026-08-30

### Delivered

- 包身份改为 `0.0.3`。`prp local run` 是本地主路径：同一进程内执行 DIRECT 任务，不要求可达 HTTP 服务器。
- 显式 `ExecutionLocation.LOCAL`，不会静默改成 `CLOUD` 或 `BRIDGE`。本地默认 `HOST`。
- 顺序 `LOCAL + HOST + DIRECT + concurrency=1` 就地使用已授权工作区，不把整树 `copytree` 当作隐藏前置条件。
- 真正并行拷贝隔离仍可用，默认 `2` 槽 / `256 MiB`，上限 `8` 槽 / `512 MiB`。
- `prp local approve` / `prp local deny` 继续同一条 ASK 暂停的本地 run；LOCAL 不创建 Bridge claim。
- 可选 `prp serve` 默认绑定 `127.0.0.1`，复用现有 `create_app` 接线；该命令在同步边界交给 ASGI runner，不再嵌套 `asyncio.run`。
- 顺序 `prp local run --isolation-mode SANDBOXED` 明确拒绝，不会静默改成 HOST。
- HOST/LOCAL `/ready` 不要求 bubblewrap；`SANDBOXED` 仍报告自身能力要求。
- 进程级槽位、拷贝字节、并发、attempt 和 token 信封；耗尽返回结构化错误，成功、失败和取消会释放占用。

### Limits

- `HOST` 是路径边界，不是 OS sandbox。HOST YOLO 仍需要显式用户事实和配置。
- 没有 Docker、cgroup-per-agent 或每 Agent 守护进程。
- 模型不能自行提高进程信封。模型质量取决于配置的 provider。
- 不声称完整 Codex/Claude Code/MCP/A2A 兼容、SLA 或 benchmark 优势。

### Not included

- 完整第三方协议实现、分布式队列、计费、SSO 或 Kubernetes。
- 通用 shell、未注册网络或模型控制的权限升级。

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
