# Progressive Reasoning Protocol

> An evidence-gated protocol for progressive revision.

Progressive Reasoning Protocol (PRP) is an independent protocol research project and dependency-free Python reference kernel. It specifies when an Agent may advance to the next version of an execution graph using public facts, deterministic verification, comparison, and a finite resource budget.

PRP studies **when continuation is allowed**, not how to make a model think longer. It defines no model, provider, HTTP transport, database, tool runtime, Bridge, or cloud topology. Use [Iskrov Agent](https://github.com/entzauberung/iskrov-agent) when you need a complete cloud-controlled Agent product.

## The problem

A plain model call is:

```text
request -> answer
```

An auditable Agent must also establish whether the result is supported by evidence, whether a failure permits another attempt, whether a candidate added value, whether historical work is safe to reuse, and when the process must stop.

PRP represents those decisions as public, durable, replayable facts. It does not store private chain-of-thought traces.

## Minimal lifecycle

```text
graph version N
      |
      v
public facts -> deterministic verification -> comparison
                                      |
              PASS / NO_GAIN / REGRESSION / BUDGET -> STOP
                                      |
              trigger + revision budget -> graph version N + 1
```

A revision requires a recorded deterministic failure, a comparable inconclusive result, or a retryable provider failure; an explicit finite revision budget; available resources; and a comparison without `NO_GAIN` or `REGRESSION`. `INCONCLUSIVE` remains a first-class verdict.

## What PRP defines

- Public facts for runs, work units, attempts, artifacts, evidence, and events.
- Closed state machines with immutable terminal history.
- Versioned revision without rewriting historical work.
- Conservative reuse using lineage, fingerprints, dependency facts, and evidence.
- Stable revision decisions and explicit stop reasons.

`DIRECT`, `CASCADE`, and `PLANNED` are runtime routing strategies, not PRP research objects. Progressive revision is the protocol core.

## Reference implementation

```python
from prp import VerificationResult, decide_progressive

decision = decide_progressive(
    verification_result=VerificationResult.FAIL,
    max_plan_revisions=3,
)
```

The function is pure: it performs no model call, file write, network request, or database mutation.

## Repository layout

```text
spec/     normative identity, facts, state machines, and boundaries
paper/    thesis, hypotheses, and related work
src/prp/  dependency-free reference kernel
tests/    conformance tests
docs/     implementation plans and supporting notes
```

This repository intentionally contains no HTTP server, database, provider adapter, Bridge, workspace runtime, tool executor, or scheduler.

## Research status

Version `0.0.2` is a reference implementation and executable set of laws. It makes no production SLA, benchmark-superiority, or model-quality claim. See [paper/thesis.md](paper/thesis.md) for hypotheses and [README.md](README.md) for the Chinese project description.

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
```

Python 3.12 or newer is required.

## License

PRP is licensed under [Apache-2.0](LICENSE-APACHE). See [NOTICE](NOTICE) for attribution and branding boundaries.
