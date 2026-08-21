# External Test Harness Migration Notes

## Credential Source Contract

**AUTHORITATIVE SOURCE**: `/home/bruce/文档/测试key.md`

The runner reads authorized credentials from the above Markdown file only. User has explicitly authorized these isolated test keys for visibility in test context.

**DEPRECATED SOURCE**: `external_tests/credentials.json`

This file contains stale keys, incorrect endpoints, and model names inconsistent with the current contract. It MUST NOT be used by any test or runner going forward.

## Active Profile Contract (v0.0.2)

| Alias | Model | Protocol | Base URL |
|---|---|---|---|
| `DEEPSEEK_FLASH_CHAT` | `deepseek-v4-flash` | `OPENAI_CHAT` | `https://api.deepseek.com` |
| `DEEPSEEK_FLASH_RESPONSES` | `deepseek-v4-flash` | `OPENAI_RESPONSES` | `https://api.deepseek.com` |
| `DEEPSEEK_FLASH_ANTHROPIC` | `deepseek-v4-flash` | `ANTHROPIC_MESSAGES` | `https://api.deepseek.com/anthropic` |
| `LUNA_GPT_56` | `gpt-5.6-luna` | `OPENAI_RESPONSES` | `https://fast.vanyospace.com` |
| `CLAUDE_SONNET_5` | `claude-sonnet-5` | `ANTHROPIC_MESSAGES` | `https://fast.vanyospace.com` |

These five profiles are hardcoded in `credential_loader.PROFILE_CONTRACTS`.

## Live Success Definition

A test result counts as live provider success ONLY when:

1. A genuine socket request reached the provider endpoint
2. The request used correct host, model, and protocol per the contract
3. A non-empty, parseable response was returned
4. The corresponding ledger entry was persisted

Collection-only tests, fake adapters, and mocked HTTP do NOT qualify as live success.

## Budget and Classification

- Maximum 24 real provider attempts across all stages
- Maximum 6 attempts per alias
- Serialized execution (concurrency = 1)
- Failures are classified but do NOT block independent scenarios:
  - `PASS`
  - `PRODUCT_DEFECT`
  - `UPSTREAM_UNSUPPORTED`
  - `UPSTREAM_AUTH_OR_PERMISSION`
  - `UPSTREAM_TRANSIENT`
  - `ENVIRONMENT_LIMITATION`
  - `NOT_APPLICABLE`
  - `BUDGET_NOT_RUN`

## Log Structure

All logs written to `/home/bruce/文档/prp测试日志/real-gap-closure/`:

- `00-harness.log` - Runner startup and transitions
- `10-providers.jsonl` - Provider smoke results
- `20-protocols.jsonl` - API protocol composition results
- `30-strategies.jsonl` - Strategy and reasoning results
- `40-agent.jsonl` - Agent tool workflow results
- `50-regression.log` - Regression test results
- `99-final-report.md` - Campaign summary

## Stage Registry

The unified runner implements these stages:

- `preflight` - Zero-network collection validation (implemented in unattended_runner.py)
- `provider` - Real provider smoke tests (not yet implemented)
- `integration` - Runtime integration tests (not yet implemented)
- `deep` - Deep workflow tests (not yet implemented)
- `stability` - Stability and security tests (not yet implemented)
- `regression` - Regression and final tests (not yet implemented)

## Credential Visibility

Keys are authorized for test use and MAY appear in test context. Credential visibility is NOT a blocker for this validation campaign. Logs still attempt redaction for readability.

## Forbidden Operations

- Using `external_tests/credentials.json` as credential source
- Recording collection-only or fake adapter results as live success
- Parallel test execution
- Unbounded retry loops
- Modifying production source code during validation
