# 3. Progressive revision

Progressive is not "ask the model again".

A revision is allowed only when all of the following hold:

1. The Run is not terminal and not cancelling.
2. The current fact is a deterministic failure, an inconclusive
   verification that still has a comparison trigger, or a retryable
   provider failure.
3. `max_plan_revisions` is explicitly set.
4. `revision_count < max_plan_revisions`.
5. Budget tokens and deadline are not exhausted.
6. A comparison, if present, is not `NO_GAIN` or `REGRESSION`.

Then, and only then, the protocol may open `graph_version + 1`.

## Stop reasons

| Reason | Meaning |
|---|---|
| `PASS` | Deterministic verification passed |
| `TERMINAL` | The Run already ended |
| `CANCELLED` | Cancellation outranks revision |
| `BUDGET` | Tokens or deadline exhausted |
| `REVISION_LIMIT` | Finite revision ceiling reached |
| `NO_REVISION_BUDGET` | Ceiling was never declared |
| `NO_TRIGGER` | No deterministic trigger was recorded |
| `NO_GAIN` | Candidate added no new evidence and did not improve |
| `REGRESSION` | Candidate is worse on recorded facts |
| `INCONCLUSIVE` | The check could not decide |

## Round

One Progressive round is a snapshot boundary: base snapshot, merged
snapshot, change sets, evidence ids, and an immutable status
(`PLANNED`, `VERIFIED`, `FAILED`, `CANCELLED`).

A planned round cannot carry terminal facts. A verified round requires
merged snapshot and evidence. Failed and cancelled rounds cannot carry
evidence or a merged snapshot.

A round cannot revise itself.

## Reuse

A historical successful node may be reused only when public facts match:

- historical unit succeeded
- lineage and both fingerprints are present and equal
- dependency artifact hashes are known SHA-256 values and equal
- if round facts are supplied, snapshots, merge digest, change sets,
  and evidence ids also match
- attempt history is proven

Anything missing or changed forces `RECOMPUTE`. Reuse never inspects
model text.

## Planner proposal

A proposal is not executable. Node keys are proposal-local and must not
look like persisted ids. The graph is closed: unknown dependencies are
rejections. Maximum node count is 64.
