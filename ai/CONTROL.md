# PRP 交付闭环修复施工蓝图

## 蓝图标识

- BLUEPRINT_ID: PRP-DELIVERY-CLOSEOUT-20260811-001
- CREATED_AT: 2026-08-11
- TARGET: v0.0.1 delivery evidence and strict JSON boundary closeout
- SOURCE_OF_TRUTH: 本文件、`ai/STATE.md`、当前 WO/ST、`ai/EXECUTION-PROTOCOL.md`

## 用户目标

以当前项目文件为准，修复最近交付审查仍未关闭的问题：重新建立可信的 Git staged/unstaged/untracked 基线，消除旧 CONTROL 与追加工单的矛盾，补齐 Settings、Provider、SQLite 三处严格 JSON 行为级测试，并让最终 Diff、定向验证和唯一人工 R2 对应同一个工作树快照。

## 用户场景

- 审查者能从报告中直接读取完整文件清单和 numstat，而不是看到“已附”但找不到内容。
- staged、unstaged、untracked 分别统计，不再用单个“78/81 文件”描述不同 Git 视图。
- 非标准 JSON 在环境配置、Provider schema payload 和事件账本写入边界均有实际调用测试。
- R2 只验证最终工作树；门禁后只允许协议要求的 `STATE.md`、`LATEST-REPORT.md` 控制面更新。

## 需求范围

1. 将当前新蓝图作为唯一全局范围来源，废止旧 4 WO/11 ST 与追加 WO-005 的冲突状态。
2. 首先采集 staged、unstaged、untracked 的真实清单、numstat 和 whitespace 状态。
3. 对当前修复交付文件做范围分类；不以 mtime 作为授权或 Diff 证据。
4. 为 `Settings.from_env` 增加非标准数字拒绝测试。
5. 为 OpenAI-compatible Provider 增加非法 schema 在出站 HTTP 前被拒绝的测试。
6. 为 `SqliteStore.append_event` 增加非有限 payload 写入拒绝且不落库的测试。
7. 保持 Schema compile cache、Evidence 三值合同、SQLite Schema 3 和现有 API 行为不回退。
8. 生成最终 Git 三视图证据并运行定向回归。
9. 设置唯一 R2 人工门禁，运行全量 pytest、ruff、mypy 和双 Diff whitespace 检查。

## 当前证据基线

- `compile_schema` 当前使用 `@lru_cache(maxsize=128)`，已有 cache hit/miss 测试。
- 生产源码中直接 `json.loads` 当前仅存在于 `src/prp_runtime/json_support.py`。
- Settings、Provider 和 SQLite 读侧已调用 `strict_json_loads`；SQLite 写侧已使用 `allow_nan=False`。
- Schema 版本拒绝测试当前参数化覆盖 1、2 和未来版本。
- 旧报告的 78 与用户提供的 81 不一致；两者均不得作为本蓝图事实，必须由 WO-001-ST-001 重采。
- 仓库历史据旧报告称仅跟踪 README；该陈述也必须由 Git 三视图证据约束，不用 mtime 补强。

## 明确非目标

- 不修改 README、CHANGELOG、许可证、版本号或创建 tag/commit/branch/push。
- 不实现 Budget、CASCADE、PLANNED、PROGRESSIVE、Planner、并行或外部入站绑定。
- 不修改领域模型、Evidence 合同、SQLite schema、migration 或 API 契约。
- 不增加、升级或删除依赖，不修改 `pyproject.toml`、`uv.lock` 或工具配置。
- 不优化 JSON Schema 功能、缓存策略或验证关键字集合。
- 不启动服务、不访问真实网络、不调用真实模型。

## 产品总约束

- `CONSTITUTION.md` 优先；只修复已证实的剩余交付缺口。
- 当前代码是新蓝图起始基线，不追认旧报告中无法复现的文件数量。
- 报告必须包含实际命令输出中的文件行，不能只写结论性数字。
- 后续实现只允许改动 WO-002 明列的三个边界模块及其测试；测试已通过时不得顺手改源码。

## 架构总约束

- `strict_json_loads` 继续作为 JSON 文本到 Python 值的唯一生产入口。
- Provider schema 必须在 HTTP 调用前严格解析；Settings 环境 JSON 必须严格解析。
- 事件 payload 必须在 SQLite 写入时拒绝非有限值，读回继续使用严格解析。
- `compile_schema` 缓存、JSON Schema 三值语义和 Verifier/Evidence 行为不在本轮重构范围。

## 数据约束

- `SCHEMA_VERSION` 保持 3；evidence 表形状不变。
- 不执行 migration、不删除数据库、不修改 schema.sql。
- 非有限事件 payload 拒绝后不得新增事件行；已有有限 payload 往返语义保持。

## 兼容性约束

- 包版本和公开目标保持 `0.0.1`。
- Native API、Provider 请求形状和稳定错误合同不新增字段。
- 不为非法 JSON 增加 fallback、宽松解析或旧行为兼容。

## 安全约束

- 测试全部使用临时数据库和 mock HTTP；禁止真实网络与凭据。
- 错误断言不得记录完整密钥、内部堆栈或真实环境值。
- 不执行 Git 写操作，包括 add、restore、reset、commit、branch、tag、push。

## 性能约束

- 保持 128 项 Schema LRU cache；本轮不添加基准或更大缓存。
- 测试使用小型 payload/schema，禁止压力测试和并发负载扩张。

## 允许修改总范围

- `src/prp_runtime/settings.py`
- `src/prp_runtime/providers/openai_compatible.py`
- `src/prp_runtime/storage/sqlite.py`
- `tests/unit/providers/test_base.py`
- `tests/unit/providers/test_openai_compatible.py`
- `tests/unit/storage/test_store.py`
- `ai/STATE.md`
- `ai/LATEST-REPORT.md`

除当前 ST 明确列出者外，上述文件也不得提前或跨任务修改。当前其余 staged/unstaged 文件只允许审计，不允许借本蓝图改写。

## 资源保护规则

- WO-001 和 WO-003-ST-001 只允许只读 Git 审计与定向验证。
- WO-002 只允许当前边界的定向 pytest/ruff/mypy，串行执行，无网络。
- `WO-003-ST-002` 是唯一 R2，必须停止并等待本蓝图下的用户明确批准；旧门禁批准不自动沿用。
- 每个 R1 命令一次一条；无 Node/Rust/Cargo；不设置后台进程。
- 每个可修复 ST 最多两轮；同一失败两次即 BLOCKED。

## 工单依赖图

```text
WO-001 Git 三视图基线与范围冻结
  -> WO-002 严格 JSON 三边界行为证据
    -> WO-003 最终 Diff 审计与人工 R2 门禁
```

## 完整推进顺序

1. WO-001：重采 Git 三视图，分类当前基线并冻结本蓝图允许改动。
2. WO-002：分别补齐 Settings、Provider、SQLite 行为测试，再做联合定向回归。
3. WO-003：输出最终三视图和 numstat，对比基线，获批后执行唯一 R2。

## 全局完成定义

- 3 个 WO、8 个 ST 全部完成。
- staged、unstaged、untracked 基线和最终状态均有逐文件证据，数字一致且不混用。
- 三个严格 JSON 边界均有实际调用测试，目标测试与联合回归通过。
- 最终改动仅限本文件和当前 ST 的允许范围，无构建产物、无未登记文件。
- 唯一 R2 获得用户明确批准并全绿；否则停在待批准状态，不伪造 COMPLETED。
- R2 后只有协议写入的 `ai/STATE.md`、`ai/LATEST-REPORT.md` 可形成预期控制面差异。

## 全局禁止操作

- 修改当前 ST 白名单外文件，或运行当前 ST 未逐条列出的命令。
- 修改依赖、锁文件、配置、数据库 schema、版本、README 或架构。
- 用 mtime、报告自述或文件总数代替 Git 三视图和逐文件清单。
- 运行网络、服务、watch、真实 API、安装、构建、发布或数据库迁移。
- 执行任何 Git 写操作或删除当前用户改动。
- 未获新批准执行 R2，或在 R2 失败后现场修复并继续。

## 全局硬停止条件

- Git 三视图无法读取、文件清单与 numstat 自相矛盾或出现构建产物。
- 发现必须修改本蓝图总范围外的源码、测试、依赖、配置或数据模型。
- 边界测试暴露需要改变公开 API/错误合同才能修复的问题。
- 同一失败两次、两轮修复耗尽、命令超时、OOM、磁盘不足或系统严重卡顿。
- 最终 Diff 相比基线出现未登记文件，或无法说明 staged/unstaged/untracked 差异。
- R2 未获批准，或任一 R2 命令失败。
