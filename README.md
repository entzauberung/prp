# PRP Runtime

> 让模型推理不再只是一次回答，而成为可验证、可恢复、可审批的执行过程。

PRP Runtime 是 Progressive Reasoning Protocol 的参考运行时。它把模型、工具、权限、工作区和验证组织成一条可追踪的执行链，面向受控 Workspace 中的云端代码 Agent。

当前版本：`0.0.2` · Python `3.12+` · 单实例 SQLite · Apache-2.0

[English](README.en.md)

## 为什么需要 PRP

普通模型 API 解决的是：

```text
发送消息 -> 获得回答
```

真实工程任务还需要回答：这一步为什么可以执行？什么证据证明它完成了？失败后应该重试、切换、回滚还是停止？写文件前谁批准？重启后能否恢复？

PRP 管的不是模型“想了多少”，而是推理如何被计划、执行、验证、修订和停止。

## 三个方向

### 协议与事实

Responses、Chat Completions 和 Anthropic Messages 的声明子集进入同一个 Native Runtime。Run、Attempt、Artifact、Evidence、Event 和 Usage 都能成为可审计事实，并支持 SSE 回放、取消、恢复、预算和错误分类。

### Agent 与工具

Agent 只能使用注册工具：`list_files`、`read_file`、`search_text`、`apply_patch`、`run_targeted_test`、`get_diff`、`get_status`。

写操作经过 Policy 和 Approval，patch 产生 Snapshot 与 ChangeSet，测试只能运行预注册的结构化命令。模型不能自行提升权限，也不能获得任意 shell。

### 真正的 Progressive Reasoning

```text
Planner 提案 -> 编译 DAG -> 独立 Slot 执行 -> Evidence 验证
       -> Git 三方合并 -> 有限修订 -> 新 graph version
```

Progressive 不是“再问模型一次”：新证据出现才允许修订；通过节点按 lineage 和内容 fingerprint 复用；写入在独立 Slot 中进行；合并使用三方合并；revision、attempt、token 和预算都有上限；最终状态由规则和验证决定。

## 四种策略

| 策略 | 适合场景 | 核心行为 |
|---|---|---|
| `DIRECT` | 简单任务 | 一个 WorkUnit、一次 Attempt、一次验证 |
| `CASCADE` | 需要备用模型 | 仅在可重试失败时进入下一 profile |
| `PLANNED` | 需要任务图 | Planner 提案 DAG，Worker 按依赖执行 |
| `PROGRESSIVE` | 需要证据和修订 | 执行、合并、验证、复用和有限 revision |

## 最小使用

```bash
uv pip install .
```

```bash
PRP_PROFILES='[{"alias":"worker","provider":"openai","model":"gpt-4o"}]' \
PRP_OPENAI_API_KEY='<YOUR_PROVIDER_KEY>' \
PRP_AUTH_TOKEN='<YOUR_SERVICE_TOKEN>' \
PRP_WORKSPACE_ROOT='/srv/workspaces' \
uvicorn prp_runtime.app:create_app --factory --host 0.0.0.0 --port 8000
```

`python -m prp_runtime` 只打印包版本，不启动服务。

## API

- `POST /v1/sessions`：创建授权 Session 与 Workspace 绑定
- `POST /v1/sessions/{session_id}/runs`：创建异步 Run
- `GET /v1/sessions/{session_id}/runs/{run_id}`：读取状态、结果和用量
- `GET /v1/sessions/{session_id}/runs/{run_id}/events`：SSE 事件与游标回放
- `POST /v1/sessions/{session_id}/runs/{run_id}/approve`：审批 ToolCall
- `POST /v1/responses`：OpenAI Responses 声明子集
- `POST /v1/chat/completions`：OpenAI Chat Completions 声明子集
- `POST /v1/messages`：Anthropic Messages 声明子集

## 适用场景

代码仓库只读审查、问题定位、审批式单文件修复、预注册测试任务、多模型 fallback、预算控制、可回放 Agent 流程，以及需要并行、冲突检测和有限修订的工程任务。

PRP 不是任意 shell，不是模型训练平台，不提供生产 SLA，也不承诺完整第三方 API 兼容。`SANDBOXED` 是需要真实 Linux `bubblewrap` 的 Sandbox isolation 模式；Bridge 的路径边界不等于 OS sandbox。

## 与普通协议的不同

- 模型 API 规定消息格式；PRP 规定一次任务如何推进。
- 工具协议描述“能调用什么”；PRP 还记录审批、证据、快照、变更集和停止条件。
- 普通多轮反思依赖模型继续判断；PRP 让 Evidence、状态机和预算决定是否修订。
- 普通并行容易互相覆盖；PRP 用 Slot、ChangeSet、fingerprint 和三方合并控制并行。

## 研究脉络

以下是相关研究方向参考，不是 PRP 的原创来源，也不代表本项目复现了论文或完成了 benchmark：

- *ReAct: Synergizing Reasoning and Acting in Language Models*，Shunyu Yao et al., 2023：<https://arxiv.org/abs/2210.03629>
- *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*，Shunyu Yao et al., 2023：<https://arxiv.org/abs/2305.10601>
- *Reflexion: Language Agents with Verbal Reinforcement Learning*，Noah Shinn et al., 2023：<https://arxiv.org/abs/2303.11366>
- *Self-Refine: Iterative Refinement with Self-Feedback*，Aman Madaan et al., 2023：<https://arxiv.org/abs/2303.17651>
- *Graph of Thoughts: Solving Elaborate Problems with Large Language Models*，Maciej Besta et al., 2024：<https://arxiv.org/abs/2308.09687>
- *Toolformer: Language Models Can Teach Themselves to Use Tools*，Timo Schick et al., 2023：<https://arxiv.org/abs/2302.04761>
- *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*，John Yang et al., 2024：<https://arxiv.org/abs/2405.15793>

PRP 的侧重点是工程协议：把行动、修订、图和工具变成持久、可验证、可恢复的事实。

## 开源与现实 / Open Source & Reality

PRP 由一名独立开发者维护。接下来我将从甘肃农村前往外地租房生活，并通过普通工作承担生活和 API 测试成本。工作会切碎开发时间，v0.0.3 可能不会很快到来，但开发不会停止。我无法始终把全部时间投入开源，却会长期维护这个项目。使用、反馈、贡献或赞助，都会直接转化为开发时间和测试资源。

## 许可证与边界

本项目采用 Apache-2.0，详见 [LICENSE-APACHE](LICENSE-APACHE)、[NOTICE](NOTICE) 和 [TRADEMARKS.md](TRADEMARKS.md)。Apache-2.0 允许商业使用和分发修改版，但项目名称和官方身份不得被用于制造冒充或背书暗示。当前版本是单实例 SQLite 参考实现，不提供多租户计费、SSO、分布式队列、Kubernetes、完整 Codex/Claude Code/MCP/A2A 兼容或生产 SLA。

---

<p align="center">
  <strong>二十七步天注定，逆流河上任我行。</strong><br>
  <sub>Искров · 甘肃</sub>
</p>
