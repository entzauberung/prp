# 1. Facts

A protocol fact is durable, typed, and free of private model text.

## Run

One task. Status starts at `PENDING`. Strategy is empty until a controller
assigns it. `graph_version` starts at 1 and only increases.

A terminal Run has `completed_at`. A `FAILED` Run carries an error. No
other status may carry an error.

## WorkUnit

One node in one graph version. It may declare:

- `lineage_key`
- `content_fingerprint`
- `dependency_fingerprint`
- dependencies
- resource claims

Lineage and both fingerprints are present together, or all absent.
A unit cannot depend on itself. Dependencies and claims are unique.

Revision does not mutate a historical WorkUnit. It creates a new unit in
a new graph version.

## Attempt

One provider call for one WorkUnit. Raw provider payloads are not facts.
Produced content lives in an Artifact.

`INTERRUPTED` means the process stopped while the attempt was running.
`UNKNOWN` means the upstream outcome cannot be confirmed.
Neither is success or failure.

## Artifact

A produced result. Completion is driven by artifacts, not self-reports.
This version admits `TEXT` and `JSON` only.

## Evidence

A verdict about one Artifact of one WorkUnit.

`result` is the stored truth: `PASS`, `FAIL`, or `INCONCLUSIVE`.
`passed` is a derived projection, never stored.

`FAIL` and `INCONCLUSIVE` both project to `passed is false`. The stored
result is what prevents "not proven" from being read as "proven bad".

`DETERMINISTIC_CHECK` must name its rule. `MODEL_REVIEW` is a weaker
kind of evidence, not a substitute for a missing deterministic check.

## Event

An append-only ledger entry. Sequence is monotonic and unique inside a
Run. Payload is restricted JSON. Events are not rewritten.

## Budget

A finite envelope: tokens, deadline, and `max_plan_revisions`.
Progressive revision is forbidden unless `max_plan_revisions` is explicit.
Exhaustion is a stop, not a silent strategy change.
