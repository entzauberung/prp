# 2. State machines

Every status change goes through a closed table. A terminal status never
returns to a running status.

Recovery creates a new Attempt or performs an explicit terminal
transition. Graph revision creates a new WorkUnit in a new graph version.

## Run

```text
PENDING    -> RUNNING | CANCELLED | FAILED
RUNNING    -> CANCELLING | SUCCEEDED | FAILED | CANCELLED
CANCELLING -> CANCELLED
SUCCEEDED, FAILED, CANCELLED -> ∅
```

## WorkUnit

```text
PENDING  -> READY | BLOCKED | CANCELLED | INVALIDATED
READY    -> RUNNING | BLOCKED | CANCELLED | INVALIDATED
RUNNING  -> SUCCEEDED | FAILED | CANCELLED
BLOCKED  -> READY | CANCELLED | INVALIDATED
SUCCEEDED, FAILED, CANCELLED, INVALIDATED -> ∅
```

`INVALIDATED` is the historical unit's fate after a later graph version
replaces it. It is not revived.

## Attempt

```text
PENDING  -> RUNNING | CANCELLED | FAILED
RUNNING  -> SUCCEEDED | FAILED | CANCELLED | INTERRUPTED | UNKNOWN
terminal -> ∅
```

Restart of a `RUNNING` Attempt becomes `INTERRUPTED`. The protocol does
not guess that the upstream call succeeded.

## Strategy strength

```text
DIRECT = 0
CASCADE = 1
PLANNED = 2
PROGRESSIVE = 3
```

Escalation is one-directional. A manually pinned strategy cannot be
replaced. Exhaustion does not downgrade the strategy.

## Recovery

A restart decision is one of: `continue`, `interrupt`, `terminate`,
`preserve`, `release`, `block`.

`BLOCK` keeps the recorded status and requires diagnosis. It does not
invent a retry or an upstream outcome.
