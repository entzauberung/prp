# PRP Runtime

> Turn model reasoning from a one-shot answer into a verifiable, recoverable, approval-aware execution process.

PRP Runtime is a reference runtime for the Progressive Reasoning Protocol. It organizes models, tools, permissions, Workspaces and verification into one traceable execution chain for cloud code Agents working in authorized Workspaces.

Current version: `0.0.2` · Python `3.12+` · single-instance SQLite · Apache-2.0

[简体中文](README.md)

## Why PRP

A model API solves:

```text
send a message -> receive an answer
```

An engineering task also needs to answer: why is this action allowed, what evidence says it is complete, who approved the write, what should happen after failure, and can a restart recover?

PRP does not try to measure how much a model “thinks”. It controls how reasoning is planned, executed, verified, revised and stopped.

## Three Directions

### Protocols and Facts

The declared subsets of Responses, Chat Completions and Anthropic Messages enter one Native Runtime. Runs, Attempts, Artifacts, Evidence, Events and Usage become auditable facts, with SSE replay, cancellation, recovery, budgets and error classification.

### Agent and Tools

The Agent can use only registered tools: `list_files`, `read_file`, `search_text`, `apply_patch`, `run_targeted_test`, `get_diff` and `get_status`.

Writes pass through Policy and Approval. A patch creates a Snapshot and ChangeSet. Tests run only registered structured commands. A model cannot raise its own permissions or obtain an arbitrary shell.

### Progressive Reasoning

```text
Planner proposal -> compile to DAG -> isolated Slot execution -> Evidence
       -> Git three-way merge -> bounded revision -> new graph version
```

Progressive is not “ask the model again”. It requires new evidence before revision, fingerprint-based reuse, isolated writes, three-way merge, bounded revision/attempt/token budgets and rule-based final state.

## Four Strategies

| Strategy | Use it for | Core behavior |
|---|---|---|
| `DIRECT` | Simple tasks | One WorkUnit, one Attempt, one verification |
| `CASCADE` | Fallbacks | Move to the next profile only for retryable failures |
| `PLANNED` | Graph scheduling | Planner proposes a DAG; Workers execute dependencies |
| `PROGRESSIVE` | Evidence and revision | Execute, merge, verify, reuse and revise within limits |

## Minimal Use

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

`python -m prp_runtime` prints the package version. It does not start a server.

## API

- `POST /v1/sessions`: create an authorized Session bound to a Workspace
- `POST /v1/sessions/{session_id}/runs`: create an asynchronous Run
- `GET /v1/sessions/{session_id}/runs/{run_id}`: read status, result and Usage
- `GET /v1/sessions/{session_id}/runs/{run_id}/events`: SSE events with cursor replay
- `POST /v1/sessions/{session_id}/runs/{run_id}/approve`: approve a ToolCall
- `POST /v1/responses`: declared OpenAI Responses subset
- `POST /v1/chat/completions`: declared OpenAI Chat Completions subset
- `POST /v1/messages`: declared Anthropic Messages subset

## What It Is Good For

Read-only repository inspection, diagnosis, approval-aware small repairs, registered verification tasks, model fallback, budget control, replayable Agent flows and engineering tasks that need parallel work, conflict detection and bounded revision.

PRP is not an arbitrary shell, a model-training platform or a production SLA. It does not promise complete third-party API compatibility. `SANDBOXED` is the Sandbox isolation mode and requires real Linux `bubblewrap`; a Bridge path boundary is not an OS sandbox.

## How It Differs

- A model API defines message format; PRP defines how a task advances.
- A tool protocol says what can be called; PRP also records approval, evidence, snapshots, changesets and stop conditions.
- Open-ended reflection depends on the model deciding to continue; PRP uses Evidence, state machines and budgets to decide revision.
- Naive parallel work can overwrite changes; PRP uses Slots, ChangeSets, fingerprints and three-way merge.

## Research Context

These are related research directions, not PRP's original sources and not evidence that this project reproduced the papers or completed a benchmark:

- *ReAct: Synergizing Reasoning and Acting in Language Models*, Shunyu Yao et al., 2023: <https://arxiv.org/abs/2210.03629>
- *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, Shunyu Yao et al., 2023: <https://arxiv.org/abs/2305.10601>
- *Reflexion: Language Agents with Verbal Reinforcement Learning*, Noah Shinn et al., 2023: <https://arxiv.org/abs/2303.11366>
- *Self-Refine: Iterative Refinement with Self-Feedback*, Aman Madaan et al., 2023: <https://arxiv.org/abs/2303.17651>
- *Graph of Thoughts: Solving Elaborate Problems with Large Language Models*, Maciej Besta et al., 2024: <https://arxiv.org/abs/2308.09687>
- *Toolformer: Language Models Can Teach Themselves to Use Tools*, Timo Schick et al., 2023: <https://arxiv.org/abs/2302.04761>
- *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*, John Yang et al., 2024: <https://arxiv.org/abs/2405.15793>

PRP focuses on the engineering layer: turning actions, revisions, graphs and tools into persistent, verifiable and recoverable protocol facts.

## Open Source & Reality

PRP is maintained by one independent developer. I am leaving rural Gansu to rent a place elsewhere and take regular work to cover living expenses and API testing costs. That work will fragment development time, so v0.0.3 may not arrive soon, but development will not stop. I cannot devote all of my time to open source, yet I intend to maintain this project for the long term. Use, feedback, contributions, or sponsorship directly become development time and testing resources.

## License and Boundaries

PRP is licensed under Apache-2.0; see [LICENSE-APACHE](LICENSE-APACHE), [NOTICE](NOTICE) and [TRADEMARKS.md](TRADEMARKS.md). Apache-2.0 permits commercial use and distribution of modified versions, but the project name and official identity must not be used to create confusion or imply endorsement. This version is a single-instance SQLite reference runtime. It does not provide multi-tenant billing, SSO, distributed queues, Kubernetes, complete Codex/Claude Code/MCP/A2A compatibility or a production SLA.

---

<p align="center">
  <strong>二十七步天注定，逆流河上任我行。</strong><br>
  <sub>Искров · 甘肃</sub>
</p>
