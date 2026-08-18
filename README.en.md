# PRP Runtime

The reference runtime for the Progressive Reasoning Protocol. The current package is `0.0.2`, a single-instance cloud code-agent runtime built around SQLite and authorized Workspaces.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Package v0.0.2](https://img.shields.io/badge/package-v0.0.2-green)](https://github.com)
[![Tests 1867 passed](https://img.shields.io/badge/tests-1867%20passed-brightgreen)](https://github.com)
[![License MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache-2.0-orange)](https://github.com)

[简体中文](README.md) | English

## v0.0.2 — Three-Direction Delivery

> 82 source modules · 82 test files · 1867 tests passed · ruff clean · mypy 82 files passed

| Direction | Core Delivery | Gate |
|-----------|---------------|------|
| **A. Protocol Hardening** | final_node · lineage · fingerprint · reservation · supervisor · event replay · recovery | conformance ✓ |
| **B. Cloud Agent Workflow** | Session/Run · ToolCall/Approval loop · Policy ALLOW/ASK/DENY · Sandbox · Workspace · Bridge | Agent E2E ✓ |
| **C. Controllable Parallelism & True Progressive** | isolated Slots · Git three-way merge · AST conflict detection · safe reuse · bounded revision | conflict/merge/revision ✓ |

---

## How Progressive Reasoning Actually Works

PRP's unique value is not "multi-step reasoning" — it is **deterministic reasoning control with evidence, rollback, and parallelism**.

```
Planner proposal (final_node declares end-state)
 │
 ├─ compile → DAG (content + dependency fingerprints)
 │
 ├─ Coordinator batching (conflict detection → read/write separation → parallel dispatch)
 │   ├── Slot A ──┐
 │   ├── Slot B ──┼── isolated writes → ChangeSet
 │   └── Slot C ──┘
 │
 ├─ Git three-way merge → unified Snapshot
 │
 ├─ Verifier global validation (targeted test + rule check)
 │
 └─ Failed?
     ├── lineage + fingerprint → safely reuse passing nodes
     ├── Planner revision (bounded count, budget control)
     └── new graph version → back to Coordinator
```

Key distinctions: each write node executes in an isolated Slot; merging uses Git three-way merge not overwrites; reuse is decided by content fingerprints not timestamps; stopping is governed by budget and revision limits not model judgment.

---

## Four Execution Strategies

| Strategy | When to use | What it does |
|----------|-------------|-------------|
| DIRECT | Simple requests | Single WorkUnit, single Attempt; Worker returns Artifact then RuleVerifier |
| CASCADE | Requires layering | Executes profile chains with progressive verification |
| PLANNED | Requires graph scheduling | Compiles execution graph; Planner proposes plans, Worker executes |
| PROGRESSIVE | Requires evidence | Stepwise advancement; isolated Slots in parallel → Git merge → verify → reuse/revise loop |

## Agent Workflow

In v0.0.2, a Native Session binds an authorized Workspace to one or more Runs. The bounded cloud code-task tool subset is:

- `list_files`, `read_file`, `search_text`: read-only Workspace inspection.
- `apply_patch`: creates a new Snapshot and a persisted ChangeSet; writes require policy authorization and approval.
- `run_targeted_test`: runs only a server-registered TEST command with structured argv, a fixed Workspace cwd, a timeout and output limits.
- `get_diff`, `get_status`: verify a ChangeSet against the current Snapshot before returning a bounded view.

Agent mode, isolation and execution location are independent dimensions:

| Dimension | Values | Contract |
| --- | --- | --- |
| Agent mode | `NORMAL`, `AUTO`, `PLAN`, `YOLO` | `PLAN` has no side effects; `AUTO` auto-allows only low-risk commands; unknown or out-of-scope requests are deterministically denied or require approval. |
| Isolation | `SANDBOXED`, `HOST` | `SANDBOXED` requires real Linux `bubblewrap`; missing capability is a hard failure, not a fake sandbox. `HOST` requires explicit user selection and server enablement. |
| Location | `CLOUD`, `BRIDGE` | Cloud uses a server-authorized Workspace; Bridge is a model-free local transport/cursor client and never sends local absolute paths to the server. |

ToolCall, Approval, ToolResult, Snapshot, ChangeSet, Evidence, Event and Usage are auditable facts. Providers advance only through public ToolCall/ToolResult turns; CoT is not stored and a model cannot self-upgrade permissions.

### DEV Readiness and Production Handoff

The DEV lane is limited to contract, ledger, Agent, ChangeSet, conflict, AST, merge and Progressive dry-run development. Every DEV result must carry `dev_only=true`; a temporary HOST directory, a Bridge path boundary and a text-only transport are not an OS sandbox and cannot prove mount, user-namespace, network or pid isolation.

Production handoff remains independent:

- The qualified runner's real `SANDBOXED` gate has passed: bubblewrap staged loading, mount, network, pid, workspace sentinel, runtime read-only, reap and resource facts were verified by targeted checks.
- The Cloud Agent targeted E2E passed through the `create_app()` production composition, covering read/search, approval, patch, targeted test, diff, merge and final failure semantics.
- DEV results must still carry `dev_only=true`; `DEV_READY_FOR_PROD_ENV` only describes organized DEV contracts and targeted facts and cannot replace real L3 or production E2E evidence.
- Handoff must not carry absolute temporary paths, secrets or CoT, and must not perform production promotion; promotion still requires the real production gate and existing approvals.

## Configuration

All configuration comes from server-side environment variables with prefix `PRP_`; unrecognized variables are reported as errors instead of being ignored.

| Variable | Default | Description |
|----------|---------|-------------|
| `PRP_HOST` | `0.0.0.0` | Listen address |
| `PRP_PORT` | `8000` | Listen port |
| `PRP_DATABASE` | `./prp.db` | SQLite path |
| `PRP_LOG_LEVEL` | `INFO` | Log level |
| `PRP_OPENAI_API_KEY` | — | Provider key |
| `PRP_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Provider endpoint |
| `PRP_OPENAI_MODEL` | `gpt-4o` | Provider model |
| `PRP_PROFILES` | — | JSON profile chain |
| `PRP_AUTH_TOKEN` | — | Bearer token for single-tenant auth |
| `PRP_WORKSPACE_ROOT` | — | Server Workspace base path |
| `PRP_SANDBOX_BINARY` | `bwrap` | bubblewrap binary |
| `PRP_TOOL_TIMEOUT` | `30` | Tool execution timeout (seconds) |
| `PRP_MAX_AGENT_TURNS` | `20` | Max Agent loop turns per Run |

## Install & Run

```bash
uv pip install .                  # Install from source
uv run python -m prp_runtime      # Print version (0.0.2)
```

ASGI launch:

```bash
PRP_OPENAI_API_KEY=sk-... \
PRP_AUTH_TOKEN=secret \
PRP_WORKSPACE_ROOT=/srv/workspaces \
uvicorn prp_runtime.app:create_app --factory --host 0.0.0.0 --port 8000
```

**Note**: `python -m prp_runtime` is an identity command that only prints package version; it is **not** a server launcher. Use the ASGI example above.

**Configuration safety boundary**: LLM profiles and credentials are provided exclusively through `PRP_*` environment variables; requests may reference only server-registered profiles and Workspaces.

### Model-free Bridge

The Bridge does not run a local model. It stores only server-issued Session/Run identifiers, an event cursor and idempotent result fingerprints. A local Workspace root can be used as a path boundary, but it is not an OS sandbox; server Policy and Approval still control tool authorization.

```bash
uv run prp --base-url http://127.0.0.1:8000 --token-stdin connect workspace-alias --access READ --agent-mode PLAN
uv run prp --base-url http://127.0.0.1:8000 run "inspect the authorized workspace"
uv run prp --base-url http://127.0.0.1:8000 resume
```

Bridge commands submit only a Workspace alias and relative task input; local absolute paths are rejected before transport. `HOST YOLO` requires explicit interactive confirmation and cannot be enabled by a model.

### API Overview
- `POST /v1/sessions`, `POST /v1/sessions/{session_id}/runs`: authorized Sessions and asynchronous Runs
- `GET /v1/sessions/{session_id}/runs/{run_id}/events`: persistent SSE events with cursor replay
- `POST /v1/responses`: declared OpenAI Responses subset
- `POST /v1/chat/completions`: declared OpenAI Chat Completions subset
- `POST /v1/messages`: declared Anthropic Messages subset

### Testing & Verification

Tests cover implemented paths and block real network calls. Static historical results and local benchmarks are evidence, not a CI or SLA claim.

### Project Boundaries
- Not a model training, fine-tuning or GPU serving platform
- Not full Codex/Claude Code, MCP, A2A, browser or arbitrary network compatibility
- Tools accept only registered relative paths, structured argv and bounded results; there is no arbitrary shell
- Cloud uses server-owned Workspaces; Bridge is a model-free transport layer, and its path boundary is not an OS sandbox
- No multi-tenancy billing, SSO, distributed queues or Kubernetes orchestration
- No production SLA, unlimited reasoning or complete third-party API field guarantee

## License
Adopted `MIT OR Apache-2.0` dual license, either one. See [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE) and [NOTICE](NOTICE).
