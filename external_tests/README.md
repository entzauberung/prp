# External validation

The external validation campaign uses five active profiles and a bounded request
budget. Profile definitions are hardcoded in `credential_loader.PROFILE_CONTRACTS`.

**Authoritative credential source**: `/home/bruce/文档/测试key.md`

The runner reads explicitly authorized credentials from the above Markdown file.
Keys are authorized for test use and visibility in test context. The deprecated
`credentials.json` file contains stale keys and incorrect endpoints and MUST NOT
be used.

The active profiles are:

| Alias | Model | Protocol | Base URL |
| --- | --- | --- | --- |
| `DEEPSEEK_FLASH_CHAT` | `deepseek-v4-flash` | `OPENAI_CHAT` | `https://api.deepseek.com` |
| `DEEPSEEK_FLASH_RESPONSES` | `deepseek-v4-flash` | `OPENAI_RESPONSES` | `https://api.deepseek.com` |
| `DEEPSEEK_FLASH_ANTHROPIC` | `deepseek-v4-flash` | `ANTHROPIC_MESSAGES` | `https://api.deepseek.com/anthropic` |
| `LUNA_GPT_56` | `gpt-5.6-luna` | `OPENAI_RESPONSES` | `https://fast.vanyospace.com` |
| `CLAUDE_SONNET_5` | `claude-sonnet-5` | `ANTHROPIC_MESSAGES` | `https://fast.vanyospace.com` |

Only `api.deepseek.com` and `fast.vanyospace.com` are in the active host
allowlist. TLS remains enabled and the harness does not use ambient proxy
settings.

The bounded budget is 24 provider attempts globally, at most 6 attempts per
alias, with 128 output tokens for ordinary requests and 256 for
planner/progressive/agent requests. Requests are serialized.

## Live Success Definition

A test result counts as live provider success ONLY when:

1. A genuine socket request reached the provider endpoint
2. The request used correct host, model, and protocol per the contract
3. A non-empty, parseable response was returned
4. The corresponding ledger entry was persisted

Collection-only tests, fake adapters, and mocked HTTP do NOT qualify as live success.

## Result Classification

Test scenarios classify their outcomes into eight categories:

- `PASS`: Test passed, product behavior is correct.
- `PRODUCT_DEFECT`: PRP runtime defect confirmed.
- `UPSTREAM_UNSUPPORTED`: Upstream provider does not support the protocol.
- `UPSTREAM_AUTH_OR_PERMISSION`: Credential invalid or quota exhausted.
- `UPSTREAM_TRANSIENT`: Upstream timeout or 5xx, may succeed on retry.
- `ENVIRONMENT_LIMITATION`: Local environment constraint (network block, missing file).
- `BUDGET_NOT_RUN`: Not executed due to budget limit.
- `NOT_APPLICABLE`: Scenario not applicable to this configuration.

Classified failures do not block independent test groups. The runner continues
execution after classification and records all results in bounded log files.

## Log Files

All logs are written to `/home/bruce/文档/prp测试日志/real-gap-closure/` with fixed names:

- `00-harness.log`: Runner startup and stage transitions.
- `10-providers.jsonl`: Provider smoke tests.
- `20-protocols.jsonl`: Runtime integration tests.
- `30-strategies.jsonl`: Strategy and reasoning tests.
- `40-agent.jsonl`: Agent tool workflow tests.
- `50-regression.log`: Regression and final tests.
- `99-final-report.md`: Campaign summary and findings.

Each file is capped at 2 MiB; total directory is capped at 16 MiB. Logs contain
redacted command output, structured results, and classification rationale.
Credentials are redacted where possible but visibility is not a blocker.

The external runner reads explicitly authorized credentials from
`/home/bruce/文档/测试key.md` in its child process. Keys are never placed in
this documentation, command arguments, reports, ledgers, events, or the Agent
environment.
