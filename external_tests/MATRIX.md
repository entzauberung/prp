# Active Profile Matrix

This matrix defines the five active provider profiles for external validation.
These are hardcoded in `credential_loader.PROFILE_CONTRACTS`.

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

All profiles read from the single authoritative source: `/home/bruce/文档/测试key.md`

The loader extracts three credential classes:
- `deepseek` - Used by all three DEEPSEEK_FLASH_* profiles
- `openai` - Used by LUNA_GPT_56
- `anthropic` - Used by CLAUDE_SONNET_5
