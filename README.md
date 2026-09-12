# Progressive Reasoning Protocol

> Evidence-gated revision for Agent work that must be inspectable, revisable, and stoppable.

Progressive Reasoning Protocol (PRP) is a small, independent protocol and research kernel. It defines when an execution graph may advance to a new version based on public facts and evidence. It does not define a model, provider, transport, tool runtime, or cloud topology.

Use this repository to study or implement evidence-gated progressive execution. Use [Iskrov Agent](https://github.com/entzauberung/iskrov-agent) when you need a complete cloud-controlled Agent runtime.

## Protocol lifecycle

```text
graph version N
      |
      v
public facts -> deterministic verification -> comparison
                                      |
              PASS / NO_GAIN / REGRESSION / BUDGET -> STOP
                                      |
              trigger + declared budget -> graph version N + 1
```

Progressive revision is not an instruction to ask a model again. A revision requires a recorded trigger, a finite revision budget, available resources, and a comparison that has not regressed. `INCONCLUSIVE` remains its own verdict and cannot be treated as success.

## What the protocol defines

- Closed vocabulary for lifecycle states, verification, comparison, revision, reuse, and stop reasons.
- Public facts for runs, work units, attempts, artifacts, evidence, and events.
- State machines whose terminal states cannot return to running states.
- Immutable graph versions: revision creates new facts and does not rewrite historical work.
- Conservative reuse based on lineage, fingerprints, dependency facts, round facts, and proven attempt history.

`DIRECT`, `CASCADE`, and `PLANNED` are runtime routing strategies that may host PRP. They are compatibility vocabulary, not protocol research objects.

## Reference implementation

The package is a dependency-free Python reference kernel:

```python
from prp import VerificationResult, decide_progressive

decision = decide_progressive(
    verification_result=VerificationResult.FAIL,
    revision_count=0,
    graph_version=1,
    max_plan_revisions=3,
)

assert decision.next_graph_version == 2
```

The function is pure: it performs no model call, file write, network request, or persistence operation.

## Repository layout

```text
spec/     normative protocol identity, facts, machines, and boundaries
paper/    thesis, hypotheses, and related work
src/prp/  dependency-free reference kernel
tests/    conformance tests for protocol laws
```

The repository intentionally contains no HTTP server, database, provider adapter, Bridge client, workspace tool, scheduler, or private reasoning trace.

## Research status

PRP 0.0.2 is a reference kernel and executable set of laws. Its hypotheses are documented in [paper/thesis.md](paper/thesis.md); related work is context, not a reproduction or benchmark claim. The project makes no production SLA or model-quality claim.

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check .
```

Python 3.12 or newer is required.

## License

PRP is licensed under [Apache-2.0](LICENSE-APACHE). See [NOTICE](NOTICE) for attribution and branding boundaries.
