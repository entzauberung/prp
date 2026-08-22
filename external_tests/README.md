# Optional External Validation

This directory contains optional, manually started provider and integration
checks. It is not required for the core package test suite and never runs as a
side effect of importing the project.

## Configuration

The harness reads provider credentials from environment variables only. It does
not read credential files, repository files, home-directory files, or command
arguments containing secrets.

Set the required variables outside the repository when running a live stage:

```text
PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY=<YOUR_PROVIDER_KEY>
PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_API_KEY=<YOUR_PROVIDER_KEY>
PRP_EXTERNAL_DEEPSEEK_FLASH_ANTHROPIC_API_KEY=<YOUR_PROVIDER_KEY>
PRP_EXTERNAL_LUNA_GPT_56_API_KEY=<YOUR_PROVIDER_KEY>
PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY=<YOUR_PROVIDER_KEY>
```

The values above are placeholders. Never commit real keys. Optional Terra
fallback metadata is also environment-only:

- `PRP_EXTERNAL_TERRA_GPT_MODEL`
- `PRP_EXTERNAL_TERRA_GPT_BASE_URL`
- `PRP_EXTERNAL_TERRA_GPT_API_KEY`
- `PRP_EXTERNAL_TERRA_GPT_ALLOWED_HOST`

The runner accepts `--stage`, `--select`, `--interface`, `--case`, and an
optional `--result-file`. It does not accept a credential-file argument.

## Results and Safety

Results default to `external_tests/.results/` and can be redirected with
`PRP_EXTERNAL_RESULT_DIR` or `--result-file`. Capability evidence can be
redirected with `PRP_LIVE_CAPABILITY_FILE`. These outputs are local validation
artifacts and are ignored by Git.

Live success requires an actual HTTPS provider response and a persisted,
redacted ledger entry. Collection-only checks, fake adapters, and mocked HTTP
are never reported as live success. Requests are serialized, budgets are
bounded, and provider failures are classified without exposing credentials.

The matrix, budget, and migration notes describe the optional checks only; they
do not expand the production package boundary or claim universal provider/API
compatibility.
