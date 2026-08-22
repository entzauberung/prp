# Budget Configuration (Example)

This example shows the bounded budget structure. Actual budget enforcement is
implemented in the unified runner.

```json
{
  "global_max_attempts": 30,
  "per_alias_max_attempts": 8,
  "per_alias_max_successes": 8,
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
  },
  "concurrency": 1,
  "one_transient_confirmation_per_scenario": true
}
```

## Budget enforcement

- Total real provider attempts capped at 30 globally
- Each alias capped at 8 attempts
- Serialized execution (concurrency = 1)
- Default output is 128 tokens; Planner, Progressive, and Agent output is 256 tokens
- Provider/protocol stages are capped at 600 seconds; reasoning and Agent stages at 900 seconds
- Only one transient confirmation is allowed per scenario; auth and invalid-request failures are not retried
- Remaining scenarios after budget exhaustion marked `BUDGET_NOT_RUN`
- Budget tracking persisted in ledger

## Fixed Campaign Logs

The campaign writes only these bounded files under the caller-selected result
directory (`PRP_EXTERNAL_RESULT_DIR`, default `external_tests/.results/`):

- `00-baseline.md`
- `10-repair-gate.log`
- `20-providers.jsonl`
- `30-protocols.jsonl`
- `40-reasoning.jsonl`
- `50-agent-engineering.jsonl`
- `60-resilience.log`
- `70-regression.log`
- `80-cleanup.json`
- `99-final-report.md`

Each actual attempt is recorded with alias, model, protocol, host, status,
classification, run/attempt IDs, usage, latency, output hash, and provenance.
Secrets are redacted by default. Logs are final evidence and are preserved by
cleanup.

## Cleanup Manifest Contract

The cleanup manifest is created before the campaign and updated with each
campaign-owned temporary resource. Each removable entry must contain an exact
absolute path, owner `campaign`, creation marker, resource kind, and parent
boundary. The only removable kinds are:

- temporary SQLite database plus its explicitly observed `-wal` and `-shm` files;
- temporary Workspace/Git directories created by `temporary_external_resources`;
- campaign-created cache directories or bytecode files at explicitly recorded paths;
- runner scratch files and truncated intermediate files at explicitly recorded paths.

Cleanup may remove an entry only when the recorded path still resolves inside its
recorded campaign-owned temporary root. It must never use recursive globs, a broad
user/project root, or an unresolved environment variable. Pre-existing paths,
unknown ownership, path-boundary failures, and every path not in the manifest are
`SKIPPED` and preserved.

The preserve list is explicit: project source, tests, configuration,
dependencies and lock files, `dist/`, `uv.lock`, local databases, ignored
result directories, historical evidence, and every fixed campaign log. Cleanup
writes `80-cleanup.json` with `removed`, `skipped`, and `preserved` entries. No
cleanup is performed by this task.
