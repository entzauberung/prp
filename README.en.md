# Progressive Reasoning Protocol

[简体中文](README.md)

> Evidence-gated reasoning for work that must be inspectable, revisable, and stoppable.

Progressive Reasoning Protocol (PRP) is an independent open protocol and research kernel for turning a model-driven task into a sequence of public, verifiable facts.

**Use this repository if you are designing or studying the protocol.** If you need a cloud controller, model provider integration, approval service, or local tool client, use [Iskrov Agent](https://github.com/entzauberung/iskrov-agent) instead.

PRP does not define a cloud topology, model API, or tool runtime. It defines how a task advances:

```text
plan -> execute -> observe -> verify -> revise or stop
```

## What PRP Specifies

- **Vocabulary**: closed values for strategies, lifecycle states, verdicts, and stop reasons.
- **Facts**: typed records for runs, work units, attempts, artifacts, evidence, and events.
- **State machines**: legal transitions with immutable terminal history.
- **Revision laws**: deterministic rules for opening a new graph version or stopping.

The protocol does not store private reasoning traces. It stores the facts needed to audit what was allowed, what was produced, what verified it, and why the process continued or stopped.

## The Progressive Principle

Progressive reasoning is not “ask the model again”. A new graph version requires a recorded trigger, an explicit revision budget, and sufficient remaining resources.

```text
current graph -> verification or provider fact
                       |
                       +--> PASS / NO_GAIN / REGRESSION / BUDGET -> STOP
                       |
                       +--> deterministic or retryable failure -> new graph version
```

`INCONCLUSIVE` is a first-class verdict. A check that cannot decide is neither silently accepted nor misclassified as a failure.

## Core Invariants

1. A terminal state cannot return to a running state.
2. A revision creates a new graph version; it does not rewrite history.
3. A successful result may be reused only when its public lineage, fingerprints, and dependency facts match.
4. Missing evidence or malformed fingerprints force recomputation.
5. Cancellation, budget exhaustion, no gain, and regression are explicit stop reasons.
6. Planner proposals are bounded, closed graphs; a proposal is not execution.

## Strategies

| Strategy | Role |
|---|---|
| `DIRECT` | One work unit and one verification |
| `CASCADE` | Move to a fallback after a retryable failure |
| `PLANNED` | Execute a bounded dependency graph |
| `PROGRESSIVE` | Compare evidence across graph versions and revise within limits |

The research focus of this repository is `PROGRESSIVE`. The other values define implementation boundaries and conformance vocabulary.

## Repository Layout

```text
spec/     normative protocol notes and boundaries
paper/    thesis, hypotheses, and related work
src/prp/  dependency-free reference kernel
tests/    conformance tests for protocol laws
```

The kernel intentionally contains no HTTP server, database, CLI, provider adapter, bridge client, workspace tool, or scheduler.

## Minimal Example

```python
from prp.revision import decide_revision
from prp.vocabulary import VerificationResult

decision = decide_revision(
    verification_result=VerificationResult.FAIL,
    revision_count=0,
    graph_version=1,
    max_plan_revisions=3,
)

assert decision.next_graph_version == 2
```

This is a pure decision: it performs no model call and changes no files.

## PRP and Iskrov Agent

PRP is an independent protocol project. [Iskrov Agent](https://github.com/entzauberung/iskrov-agent) is a separate AGPL-3.0-only product that implements an agent runtime and uses progressive execution as one of its strategies.

Other runtimes can implement PRP without using Iskrov Agent. Iskrov Agent can evolve its product topology without changing the identity of this protocol.

## Research Status

This repository contains a reference kernel and conformance tests. The hypotheses and evaluation questions are documented in [paper/thesis.md](paper/thesis.md). Related papers are context, not claims that PRP reproduces their methods or results.

The current kernel is intentionally small. It is a protocol reference and an executable set of laws, not a claim of production readiness or benchmark superiority.

## License

PRP is licensed under [Apache-2.0](LICENSE-APACHE). See [NOTICE](NOTICE) for attribution and branding boundaries.
