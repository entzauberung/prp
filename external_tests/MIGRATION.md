# External Harness Notes

## Credential Contract

The optional harness accepts provider keys only through
`PRP_EXTERNAL_*_API_KEY` environment variables. It does not support Markdown,
JSON, TOML, home-directory, or repository credential files. The repository
contains placeholders and contract metadata only.

The active profile contract is defined by `credential_loader.PROFILE_CONTRACTS`
and the environment names are listed in `README.md` and `model_matrix.example.json`.
Optional Terra fallback metadata is accepted only through its explicit
`PRP_EXTERNAL_TERRA_GPT_*` variables and only after a retryable Luna failure.

## Results Contract

The runner writes redacted JSONL and capability evidence to the directory named
by `PRP_EXTERNAL_RESULT_DIR`, defaulting to the ignored local
`external_tests/.results/` directory. A caller may provide an explicit
`--result-file` or `PRP_LIVE_CAPABILITY_FILE`; no personal absolute path is
embedded in the harness.

Each live result must identify the profile, protocol, endpoint host, outcome,
bounded usage, and a redacted error classification. Secrets, raw provider
responses, absolute local paths, and chain-of-thought are not valid ledger data.

## Stage Contract

The unified runner provides collection, provider, protocol, strategy, Agent,
and regression stages. Commands are serialized and use structured argv. Live
stages are opt-in and require the caller to provide environment configuration.
The core package does not depend on these stages.

## Optional Terra Fallback

Terra is not active by default. Its model, HTTPS base URL, API key, and allowed
host must all be supplied through environment variables, and the host must be
explicitly admitted by the active matrix. Missing or invalid metadata produces
`TERRA_NOT_CONFIGURED` and sends no request.
