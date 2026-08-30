# PRP Runtime

> Turn model reasoning from a one-shot answer into a verifiable, recoverable, approval-aware execution process.

PRP Runtime is a reference runtime for the Progressive Reasoning Protocol and a local-first Agent OS: controller, policy, tools, approval, evidence, budget and recovery run in one process. Any configured model can plug into the same loop. Model quality depends on the chosen provider, not on PRP itself.

Current version: `0.0.3` · Python `3.12+` · single-instance SQLite · no Docker · Apache-2.0

[简体中文](README.md)

## Why PRP

A model API solves:

```text
send a message -> receive an answer
```

An engineering task also needs to answer: why is this action allowed, what evidence says it is complete, who approved the write, what should happen after failure, and can a restart recover?

PRP does not try to measure how much a model “thinks”. It controls how reasoning is planned, executed, verified, revised and stopped.

## Minimal Use

The primary path is in-process `prp local run`. No HTTP server is required.

```bash
uv pip install .
```

```bash
export PRP_WORKER_PROFILE='{"alias":"worker","provider":"openai_compatible","model":"your-model","role":"WORKER","base_url":"https://models.example/v1","context_window_tokens":32000,"max_output_tokens":4000}'
prp local run "summarise this repository" --workspace .
```

When ASK pauses, continue against the same workspace:

```bash
prp local approve <request_id> --workspace .
prp local deny <request_id> --workspace . --reason "not allowed"
```

`python -m prp_runtime` prints the package version. It does not start a server.

## Optional HTTP and Other Topologies

`prp serve` is an optional loopback interface for other programs. It binds `127.0.0.1:8000` by default. Local run does not use this command. A wider bind must be explicit.

```bash
prp serve
```

The three execution locations stay distinct and are never silently translated:

| Location | Behavior |
|---|---|
| `LOCAL` | Controller, provider adapter, tools and workspace share one process; default `HOST` |
| `CLOUD` | Controller and tools execute inside the server process |
| `BRIDGE` | The server keeps planning; a model-free local client executes tools |

## Three Directions

### Protocols and Facts

The declared subsets of Responses, Chat Completions and Anthropic Messages enter one Native Runtime. Runs, Attempts, Artifacts, Evidence, Events and Usage become auditable facts, with SSE replay, cancellation, recovery, budgets and error classification. These are declared subsets, not complete third-party protocol compatibility.

### Agent and Tools

The Agent can use only registered tools: `list_files`, `read_file`, `search_text`, `apply_patch`, `run_targeted_test`, `get_diff` and `get_status`.

Writes pass through Policy and Approval. A patch creates a Snapshot and ChangeSet. Tests run only registered structured commands. A model cannot raise its own permissions or obtain an arbitrary shell or general network access.

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

## Isolation, Capacity and Security Boundaries

- Sequential `LOCAL + HOST + DIRECT + concurrency=1` operates on the granted workspace root in place, guarded by descriptor/path-boundary checks, without a hidden full-tree `copytree`.
- `HOST` is a path boundary, not an operating-system sandbox. HOST YOLO still requires the explicit user fact and the configured setting.
- `SANDBOXED` is optional and requires real Linux `bubblewrap` when selected. HOST/LOCAL readiness does not require it. Sequential `prp local run` defaults to `HOST`; `--isolation-mode SANDBOXED` is rejected explicitly and is never silently converted to HOST.
- True parallel work, `PLANNED` or `PROGRESSIVE` uses copied slots; defaults are `2` slots and `256 MiB`, capped at `8` slots and `512 MiB`.
- The process envelope also bounds concurrency, attempts and tokens. Exhaustion returns a structured error and does not silently change location, strategy or isolation.
- There is no Docker, Podman, cgroup-per-agent or per-agent daemon.

## What It Is Good For

Local one-process code tasks, read-only repository inspection, diagnosis, approval-aware small repairs, registered verification tasks, model fallback, budget control, replayable Agent flows and engineering tasks that need parallel work, conflict detection and bounded revision.

PRP is not an arbitrary shell, a model-training platform or a production SLA. It does not promise complete Codex, Claude Code, MCP or A2A compatibility. Whether a model is strong enough depends on the configured provider.

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

PRP is maintained by one independent developer. I am leaving rural Gansu to rent a place elsewhere and take regular work to cover living expenses and API testing costs. That work will fragment development time, so later versions may not arrive soon, but development will not stop. I cannot devote all of my time to open source, yet I intend to maintain this project for the long term. Use, feedback, contributions, or sponsorship directly become development time and testing resources.

## License and Boundaries

PRP is licensed under Apache-2.0; see [LICENSE-APACHE](LICENSE-APACHE), [NOTICE](NOTICE) and [TRADEMARKS.md](TRADEMARKS.md). Apache-2.0 permits commercial use and distribution of modified versions, but the project name and official identity must not be used to create confusion or imply endorsement. This version is a single-instance SQLite reference runtime. It does not provide multi-tenant billing, SSO, distributed queues, Kubernetes, complete Codex/Claude Code/MCP/A2A compatibility or a production SLA.

---

<p align="center">
  <strong>二十七步天注定，逆流河上任我行。</strong><br>
  <sub>Искров · 甘肃</sub>
</p>
