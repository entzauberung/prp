# Budget Configuration (Example)

This example shows the bounded budget structure. Actual budget enforcement is
implemented in the unified runner.

```json
{
  "global_max_attempts": 24,
  "per_alias_max_attempts": 6,
  "per_alias_max_successes": 6,
  "max_output_tokens": {
    "default": 128,
    "planner": 256,
    "progressive": 256,
    "agent": 256
  },
  "timeout_seconds": {
    "provider": 600,
    "integration": 600,
    "strategy": 900,
    "agent": 900,
    "regression": 600
  }
}
```

## Budget enforcement

- Total real provider attempts capped at 24 globally
- Each alias capped at 6 attempts
- Serialized execution (concurrency = 1)
- Remaining scenarios after budget exhaustion marked `BUDGET_NOT_RUN`
- Budget tracking persisted in ledger
