# Active Profile Matrix

This matrix defines the five active provider profiles for external validation.
These are hardcoded in `credential_loader.PROFILE_CONTRACTS`.

Terra is an optional, pending fallback contract. It is not an active profile until
all required metadata has been discovered and explicitly admitted to this matrix.

## Profiles

| Alias | Model | Protocol | Base URL | Host |
|---|---|---|---|---|
| `DEEPSEEK_FLASH_CHAT` | `deepseek-v4-flash` | `OPENAI_CHAT` | `https://api.deepseek.com` | `api.deepseek.com` |
| `DEEPSEEK_FLASH_RESPONSES` | `deepseek-v4-flash` | `OPENAI_RESPONSES` | `https://api.deepseek.com` | `api.deepseek.com` |
| `DEEPSEEK_FLASH_ANTHROPIC` | `deepseek-v4-flash` | `ANTHROPIC_MESSAGES` | `https://api.deepseek.com/anthropic` | `api.deepseek.com` |
| `LUNA_GPT_56` | `gpt-5.6-luna` | `OPENAI_RESPONSES` | `https://fast.vanyospace.com` | `fast.vanyospace.com` |
| `CLAUDE_SONNET_5` | `claude-sonnet-5` | `ANTHROPIC_MESSAGES` | `https://fast.vanyospace.com` | `fast.vanyospace.com` |

## Protocol Coverage

- **OpenAI Chat Completions**: `DEEPSEEK_FLASH_CHAT`
- **OpenAI Responses**: `DEEPSEEK_FLASH_RESPONSES`, `LUNA_GPT_56`
- **Anthropic Messages**: `DEEPSEEK_FLASH_ANTHROPIC`, `CLAUDE_SONNET_5`

## Interface Completion Matrix

An interface is complete only after one candidate has both an actual outbound
provider PASS and an actual PRP ingress composition PASS. A candidate failure is
recorded against that candidate and does not copy success to another candidate.
The matrix does not require every candidate to pass.

| Interface | Candidate order | Outbound scenario IDs | Ingress scenario ID | Completion requirement |
|---|---|---|---|---|
| `OPENAI_CHAT` | `DEEPSEEK_FLASH_CHAT` | `wo-001-st-003-deepseek_flash_chat` | `wo-003-st-001-chat` | Actual provider PASS, then actual `/v1/chat/completions` PASS with persisted Run/Attempt/Artifact/Evidence/Event/Usage facts. |
| `OPENAI_RESPONSES` | `DEEPSEEK_FLASH_RESPONSES`, `LUNA_GPT_56`, optional `TERRA_GPT` | `wo-001-st-003-deepseek_flash_responses`; `wo-002-st-002-luna_gpt_56` | `wo-003-st-001-responses` | Any one candidate with actual provider PASS and actual `/v1/responses` PASS. Preserve lifecycle IDs `wo-003-st-002-real-lifecycle` and `wo-003-st-002-real-lifecycle-*` as regression evidence. |
| `ANTHROPIC_MESSAGES` | `DEEPSEEK_FLASH_ANTHROPIC`, `CLAUDE_SONNET_5` | `wo-001-st-003-deepseek_flash_anthropic`; `wo-002-st-002-claude_sonnet_5` | `wo-003-st-001-messages` | Any one candidate with actual provider PASS, then actual `/v1/messages` PASS with persisted production facts. |

### Gate and failure semantics

- `PASS` is valid only for an actual HTTPS provider call with a non-empty parsed
  response and a persisted ledger record. Simulated, local, collection-only, or
  prerequisite records never satisfy an interface gate.
- A candidate with auth, invalid-request, unsupported, transient, network, or
  product failure remains a classified finding. The next eligible candidate may
  be attempted according to its fallback rules.
- An interface remains `OPEN` while no candidate has completed both outbound and
  ingress gates. Once one candidate completes both, the interface is `COMPLETE`
  even if other candidates have findings.
- `OPENAI_CHAT` has no fallback candidate in the active matrix.
- `OPENAI_RESPONSES` may select Terra only after Luna has a retryable
  upstream/environment failure and all exact Terra metadata and allowlist checks
  pass. Auth, invalid-request, unsupported, or product failures do not select
  Terra.
- `TERRA_NOT_CONFIGURED` means no Terra request is sent, its host is not added
  to the allowlist, and the missing values are not inferred from the alias.

### Current reconciled gate snapshot

- `OPENAI_RESPONSES`: `COMPLETE` retained by actual DeepSeek outbound and
  `/v1/responses` ingress evidence.
- `OPENAI_CHAT`: `OPEN`, because the current DeepSeek candidate has no actual
  provider PASS.
- `ANTHROPIC_MESSAGES`: `OPEN`, because current DeepSeek and Claude candidates
  have no actual provider PASS.

## Host Allowlist

Only these two hosts are authorized for external requests:

- `api.deepseek.com`
- `fast.vanyospace.com`

## Model Metadata

DeepSeek `DeepSeek-V4-Flash-0731` is service version metadata; actual request
model ID is `deepseek-v4-flash`.

## Roles

All five profiles can serve as WORKER. PLANNER and PROGRESSIVE roles select from
these profiles at runtime based on strategy requirements.

## Credentials

All profiles read provider keys from `PRP_EXTERNAL_*_API_KEY` environment
variables. The repository contains placeholders only; no credential file or
personal source is supported. The three DeepSeek profiles may share one value,
as may the Luna/OpenAI and Claude/Anthropic profiles.

## Optional Terra Contract

| Field | Contract |
|---|---|
| Alias candidate | `TERRA_GPT` |
| Expected protocol | `OPENAI_RESPONSES` |
| Model environment placeholder | `PRP_EXTERNAL_TERRA_GPT_MODEL` |
| Base URL environment placeholder | `PRP_EXTERNAL_TERRA_GPT_BASE_URL` |
| API key environment placeholder | `PRP_EXTERNAL_TERRA_GPT_API_KEY` |
| Allowed host environment placeholder | `PRP_EXTERNAL_TERRA_GPT_ALLOWED_HOST` |
| Current status | `TERRA_NOT_CONFIGURED` |

The exact Terra model ID, HTTPS base URL, host and credential mapping are unknown.
They must be supplied through the explicit environment placeholders; no value is
inferred from the alias. Terra remains outside `profiles` and outside `allowed_hosts`
while any field is missing, malformed, or unverified. The host may enter the active
allowlist only in the same explicit matrix change that records the verified base URL.

Luna remains the primary profile. A Terra request may be selected only after an
active `LUNA_GPT_56` attempt has a classified retryable upstream or environment
failure, Terra has passed the complete metadata and allowlist checks, and the
fallback is recorded as a separate alias/model/protocol/host attempt. Auth,
invalid-request, product, or unsupported failures do not silently switch to Terra.
If the metadata contract is incomplete, record `TERRA_NOT_CONFIGURED`, send no
Terra request, and continue the remaining scenarios.
