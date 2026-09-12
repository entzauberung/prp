# Progressive Reasoning Protocol

PRP is a small, independent protocol and research kernel for evidence-gated progressive revision. It defines when a versioned execution graph may advance using public facts, deterministic verification, comparison, and a finite revision budget.

The repository contains a dependency-free Python reference kernel, normative specification notes, a thesis, and conformance tests. It deliberately does not contain a server, provider adapter, database, Bridge, workspace runtime, or scheduler. Those belong to an Agent product such as Iskrov Agent.

The public entry point is:

```python
from prp import VerificationResult, decide_progressive

decision = decide_progressive(
    verification_result=VerificationResult.FAIL,
    max_plan_revisions=3,
)
```

See [README.md](README.md) for the full project description.
