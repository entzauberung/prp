# PRP Runtime

Progressive Reasoning Protocol 的参考运行时。当前包版本为 `0.0.2`，面向单实例、SQLite 和受控 Workspace 的云端代码 Agent。

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Package v0.0.2](https://img.shields.io/badge/package-v0.0.2-green)](https://github.com)
[![Tests 1867 passed](https://img.shields.io/badge/tests-1867%20passed-brightgreen)](https://github.com)
[![License MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache-2.0-orange)](https://github.com)

[简体中文](README.md) | [English](README.en.md)

## v0.0.2 — 三方向交付

> 82 源码模块 · 82 测试文件 · 1867 tests passed · ruff clean · mypy 82 files passed

| 方向 | 核心交付 | 门禁 |
|------|----------|------|
| **A. 协议加固** | final_node · lineage · fingerprint · reservation · supervisor · event replay · recovery | conformance ✓ |
| **B. 云端 Agent** | Session/Run · ToolCall/Approval 循环 · Policy ALLOW/ASK/DENY · Sandbox · Workspace · Bridge | Agent E2E ✓ |
| **C. 可控并行与真正 Progressive** | 独立 Slot · Git 三方合并 · AST 冲突检测 · 安全复用 · 有限修订 | conflict/merge/revision ✓ |

---

## Progressive Reasoning 如何真正工作

PRP 的独特价值不是"多步推理"——是**有证据、可回退、可并行的确定性推理控制**。

```
Planner 提案 (final_node 声明终态)
 │
 ├─ compile → DAG (content + dependency fingerprint)
 │
 ├─ Coordinator 分批 (冲突检测 → 读写分离 → 并行调度)
 │   ├── Slot A ──┐
 │   ├── Slot B ──┼── 独立隔离写入 → ChangeSet
 │   └── Slot C ──┘
 │
 ├─ Git 三方合并 → 统一 Snapshot
 │
 ├─ Verifier 全局验证 (targeted test + rule check)
 │
 └─ 失败？
     ├── lineage + fingerprint → 安全复用已通过节点
     ├── Planner revision (有限次数, 预算控制)
     └── 新 graph version → 回到 Coordinator
```

关键区别：每个写节点在独立 Slot 执行，合并由 Git 三方合并而非覆盖，复用由内容指纹而非时间戳决定，停止由预算和修订上限而非模型判断决定。

---

## 四策略说明

| 策略 | 何时使用 | 做什么 |
|------|----------|--------|
| DIRECT | 简单请求 | 单 WorkUnit、单 Attempt；Worker 返回 Artifact 后执行 RuleVerifier |
| CASCADE | 需要分层 | 按 profile 链条执行，逐层验证 |
| PLANNED | 需要图调度 | 编译执行图，Planner 提出计划，Worker 执行 |
| PROGRESSIVE | 需要证据 | 逐步推进；独立 Slot 并行 → Git 合并 → 验证 → 复用/修订循环 |

## Agent 工作流

v0.0.2 的 Native Session 将一次授权 Workspace 与多个 Run 关联。云端最小代码任务的受控工具子集为：

- `list_files`、`read_file`、`search_text`：只读 Workspace 观察。
- `apply_patch`：基于 Snapshot 产生新的 Snapshot 和持久化 ChangeSet，写入必须经过策略和批准。
- `run_targeted_test`：只执行服务端预注册的 TEST 命令，使用结构化 argv、固定 Workspace cwd、超时和输出上限。
- `get_diff`、`get_status`：校验 ChangeSet 与当前 Snapshot 后返回受限 diff/status。

Agent mode、隔离和执行位置是独立维度：

| 维度 | 可选值 | 事实 |
| --- | --- | --- |
| Agent mode | `NORMAL`、`AUTO`、`PLAN`、`YOLO` | `PLAN` 无副作用；`AUTO` 仅自动放行低风险命令；未知或越界请求由确定性 Policy 拒绝或请求批准。 |
| Isolation | `SANDBOXED`、`HOST` | `SANDBOXED` 必须使用真实 Linux `bubblewrap`；缺少能力不能伪装成沙箱。`HOST` 只在用户明确选择且服务端允许时可用。 |
| Location | `CLOUD`、`BRIDGE` | Cloud 使用服务端授权 Workspace；Bridge 是无模型的本地传输/游标客户端，不把本地绝对路径发送给服务端。 |

每次 ToolCall、Approval、ToolResult、Snapshot、ChangeSet、Evidence、Event 和 Usage 都是可审计事实。Provider 只能通过公开 ToolCall/ToolResult 推进，不保存 CoT，也不能自行提升权限。

### DEV readiness 与生产 handoff

DEV 轨只用于合同、账本、Agent、ChangeSet、冲突、AST、merge 和 Progressive dry-run 开发。DEV 结果必须带 `dev_only=true`；临时 HOST 目录、Bridge 路径边界和 text-only transport 都不是 OS sandbox，也不能证明 mount、user namespace、network 或 pid 隔离。

生产 handoff 的前置条件保持独立：

- 当前合格 runner 的真实 `SANDBOXED` gate 已通过：bubblewrap staged loader、mount、network、pid、workspace sentinel、runtime RO、reap 和资源定向事实均已验证。
- Cloud Agent 定向 E2E 已通过 `create_app()` production composition，覆盖 read/search、approval、patch、targeted test、diff、merge 和 final failure semantics。
- DEV 结果仍必须带 `dev_only=true`；`DEV_READY_FOR_PROD_ENV` 只表示 DEV 合同和定向事实已整理，不能替代真实 L3 或生产 E2E。
- handoff 不携带绝对临时路径、秘密、CoT 或生产 promotion；生产 promotion 仍须经过真实生产 gate 和既有审批。

## 配置

- **PRP Native API**
  - `POST /v1/sessions`：创建 owner-scoped Workspace Session。
  - `POST /v1/sessions/{session_id}/runs`：异步创建 Run，Session 的 Agent options 由服务端绑定。
  - `GET /v1/sessions/{session_id}/runs/{run_id}`：读取 Run 状态、结果文本、用量与错误。
  - `GET /v1/sessions/{session_id}/runs/{run_id}/events`：SSE 事件流 + 游标回放。
  - `POST /v1/sessions/{session_id}/runs/{run_id}/approve`：同意/拒绝 Policy 要求批准的 ToolCall。
- **OpenAI Responses Binding**: `POST /v1/responses`（声明子集）
- **OpenAI Chat Completions Binding**: `POST /v1/chat/completions`（声明子集）
- **Anthropic Messages Binding**: `POST /v1/messages`（声明子集）

所有配置通过服务端环境变量 `PRP_*`；不认识的变量报错而非忽略。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PRP_HOST` | `0.0.0.0` | 监听地址 |
| `PRP_PORT` | `8000` | 监听端口 |
| `PRP_DATABASE` | `./prp.db` | SQLite 路径 |
| `PRP_LOG_LEVEL` | `INFO` | 日志级别 |
| `PRP_OPENAI_API_KEY` | — | Provider key |
| `PRP_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Provider endpoint |
| `PRP_OPENAI_MODEL` | `gpt-4o` | Provider model |
| `PRP_PROFILES` | — | JSON profile chain |
| `PRP_AUTH_TOKEN` | — | Bearer token for single-tenant auth |
| `PRP_WORKSPACE_ROOT` | — | Server Workspace base path |
| `PRP_SANDBOX_BINARY` | `bwrap` | bubblewrap binary |
| `PRP_TOOL_TIMEOUT` | `30` | Tool execution timeout (seconds) |
| `PRP_MAX_AGENT_TURNS` | `20` | Max Agent loop turns per Run |

## 安装与运行

```bash
uv pip install .                  # 从源码安装
uv run python -m prp_runtime      # 打印版本 (0.0.2)
```

ASGI 启动：

```bash
PRP_OPENAI_API_KEY=sk-... \
PRP_AUTH_TOKEN=secret \
PRP_WORKSPACE_ROOT=/srv/workspaces \
uvicorn prp_runtime.app:create_app --factory --host 0.0.0.0 --port 8000
```

**注意**：`python -m prp_runtime` 只打印包版本，**不是**服务启动器。使用上面的 ASGI 示例启动。

**配置安全边界**：LLM profile 和凭据只从 `PRP_*` 环境变量读取；请求只能引用服务端已注册的 profile 和 Workspace。

### 无模型 Bridge

Bridge 不运行本地模型，只保存服务端返回的 Session/Run 标识、事件游标和幂等结果指纹。它可以为本地 Workspace 提供路径边界，但这不是 OS sandbox；实际工具授权仍由服务端 Policy 和 Approval 控制。

```bash
uv run prp --base-url http://127.0.0.1:8000 --token-stdin connect workspace-alias --access READ --agent-mode PLAN
uv run prp --base-url http://127.0.0.1:8000 run "inspect the authorized workspace"
uv run prp --base-url http://127.0.0.1:8000 resume
```

Bridge 命令只提交 Workspace alias 和相对任务输入；本地绝对路径不会发给服务端。`HOST YOLO` 需要交互式显式确认，不能由模型升级。

### API 概览
- `POST /v1/sessions`、`POST /v1/sessions/{session_id}/runs`：授权 Session 和异步 Run
- `GET /v1/sessions/{session_id}/runs/{run_id}/events`：持久事件 SSE 与游标回放
- `POST /v1/responses`：声明子集的 OpenAI Responses 映射
- `POST /v1/chat/completions`：声明子集的 OpenAI Chat Completions 映射
- `POST /v1/messages`：声明子集的 Anthropic Messages 映射

### 测试与验证
定向示例（测试已实现路径）：

```bash
uv run pytest tests/unit/control/test_direct.py -q
```

测试只覆盖已实现路径，禁止真实网络调用；本 README 不把静态历史结果或本地 benchmark 写成实时 CI/SLA。

### 项目边界
- 非模型训练、微调或 GPU serving 平台
- 不实现完整 Codex/Claude Code、MCP、A2A、浏览器或任意网络协议
- Tool 只接受注册的相对路径、结构化 argv 和受限结果；没有任意 shell
- Cloud 使用服务端 Workspace；Bridge 是无模型传输层，路径边界不等于 OS sandbox
- 无多租户计费、SSO、分布式队列或 Kubernetes 编排
- 无生产 SLA、无限推理或完整第三方 API 字段承诺

## 许可证

采用 `MIT OR Apache-2.0` 双许可，任选其一。详见 [LICENSE-MIT](LICENSE-MIT)、[LICENSE-APACHE](LICENSE-APACHE) 和 [NOTICE](NOTICE)。
