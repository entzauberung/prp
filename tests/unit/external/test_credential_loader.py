"""Unit tests for credential_loader module.

This test suite validates the credential parsing, profile contract enforcement,
and secure credential handling logic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "prp_external_credential_loader",
    _ROOT / "external_tests" / "credential_loader.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_LOADER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOADER)


def test_parse_credentials_text_extracts_all_three_keys() -> None:
    """Parser should extract DeepSeek, OpenAI, and Anthropic keys."""
    text = """\
## DeepSeek API
This is the DeepSeek credential.
sk-deepseek-test-key-1234567890abcdef1234

## Vanyospace OpenAI / Luna GPT-5.6
This is the Luna/OpenAI credential for Vanyospace.
sk-openai-test-key-1234567890abcdef1234

## Vanyospace Anthropic / Claude Sonnet 5
This is the Claude/Anthropic credential for Vanyospace.
sk-ant-api03-test-key-1234567890abcdef1234
"""
    creds = _LOADER.parse_credentials_text(text)

    assert len(creds.aliases) == 5
    assert "DEEPSEEK_FLASH_CHAT" in creds.aliases
    assert "LUNA_GPT_56" in creds.aliases
    assert "CLAUDE_SONNET_5" in creds.aliases


def test_profile_env_generates_correct_vars() -> None:
    """profile_env should generate PRP_EXTERNAL_* environment variables."""
    text = """\
## DeepSeek API
sk-deepseek-test-key-12345678901234567890

## Vanyospace OpenAI / Luna GPT-5.6
sk-openai-test-key-12345678901234567890

## Vanyospace Anthropic / Claude Sonnet 5
sk-ant-api03-test-key-12345678901234567890
"""
    creds = _LOADER.parse_credentials_text(text)

    # Test DeepSeek profile
    deepseek_env = creds.profile_env("DEEPSEEK_FLASH_CHAT")
    assert "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY" in deepseek_env
    assert "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_BASE_URL" in deepseek_env
    assert "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_MODEL" in deepseek_env
    assert deepseek_env["PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY"].startswith("sk-deepseek-")
    assert deepseek_env["PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_BASE_URL"] == "https://api.deepseek.com"

    # Test Luna profile
    luna_env = creds.profile_env("LUNA_GPT_56")
    assert "PRP_EXTERNAL_LUNA_GPT_56_API_KEY" in luna_env
    assert luna_env["PRP_EXTERNAL_LUNA_GPT_56_API_KEY"].startswith("sk-openai-")
    assert luna_env["PRP_EXTERNAL_LUNA_GPT_56_BASE_URL"] == "https://fast.vanyospace.com"

    # Test Claude profile
    claude_env = creds.profile_env("CLAUDE_SONNET_5")
    assert "PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY" in claude_env
    assert claude_env["PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY"].startswith("sk-ant-")


def test_profile_contracts_define_exactly_five_profiles() -> None:
    """PROFILE_CONTRACTS should contain exactly five profiles."""
    assert len(_LOADER.PROFILE_CONTRACTS) == 5

    expected_profiles = {
        "DEEPSEEK_FLASH_CHAT",
        "DEEPSEEK_FLASH_RESPONSES",
        "DEEPSEEK_FLASH_ANTHROPIC",
        "LUNA_GPT_56",
        "CLAUDE_SONNET_5",
    }
    assert set(_LOADER.PROFILE_CONTRACTS.keys()) == expected_profiles


def test_profile_contracts_only_allow_two_hosts() -> None:
    """PROFILE_CONTRACTS should only reference authorized hosts."""
    authorized_hosts = {"api.deepseek.com", "fast.vanyospace.com"}

    for profile_name, contract in _LOADER.PROFILE_CONTRACTS.items():
        from urllib.parse import urlsplit
        hostname = urlsplit(contract["base_url"]).hostname
        assert hostname in authorized_hosts, f"Profile {profile_name} uses unauthorized host {hostname}"


def test_missing_credential_raises_error() -> None:
    """Parser should raise CredentialError when required keys are missing."""
    text = """\
## DeepSeek API
sk-deepseek-test-key-12345678901234567890
"""

    with pytest.raises(_LOADER.CredentialError) as exc_info:
        _LOADER.parse_credentials_text(text)

    assert "MISSING_KEY" in exc_info.value.code
    assert "anthropic" in exc_info.value.code
    assert "openai" in exc_info.value.code


def test_load_credentials_from_file(tmp_path) -> None:
    """load_credentials should read from file and parse."""
    cred_file = tmp_path / "test_creds.md"
    cred_file.write_text("""\
## DeepSeek API
sk-deepseek-test-key-12345678901234567890

## Vanyospace OpenAI / Luna GPT-5.6
sk-openai-test-key-12345678901234567890

## Vanyospace Anthropic / Claude Sonnet 5
sk-ant-api03-test-key-12345678901234567890
""")

    creds = _LOADER.load_credentials(str(cred_file))
    assert len(creds.aliases) == 5
    assert "DEEPSEEK_FLASH_CHAT" in creds.aliases


def test_invalid_file_raises_error() -> None:
    """load_credentials should raise CredentialError for invalid file."""
    with pytest.raises(_LOADER.CredentialError) as exc_info:
        _LOADER.load_credentials("/nonexistent/file.md")

    assert exc_info.value.code == "FILE_UNREADABLE"


def test_deepseek_profiles_share_same_credential() -> None:
    """All three DeepSeek profiles should use the same credential."""
    text = """\
## DeepSeek API
sk-deepseek-shared-key-12345678901234567890

## Vanyospace OpenAI / Luna GPT-5.6
sk-openai-test-key-12345678901234567890

## Vanyospace Anthropic / Claude Sonnet 5
sk-ant-api03-test-key-12345678901234567890
"""
    creds = _LOADER.parse_credentials_text(text)

    chat_key = creds.profile_env("DEEPSEEK_FLASH_CHAT")["PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY"]
    responses_key = creds.profile_env("DEEPSEEK_FLASH_RESPONSES")["PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_API_KEY"]
    anthropic_key = creds.profile_env("DEEPSEEK_FLASH_ANTHROPIC")["PRP_EXTERNAL_DEEPSEEK_FLASH_ANTHROPIC_API_KEY"]

    assert chat_key == responses_key == anthropic_key == "sk-deepseek-shared-key-12345678901234567890"


def test_credentials_are_not_logged_in_repr() -> None:
    """CredentialSet repr should not leak actual keys."""
    text = """\
## DeepSeek API
sk-deepseek-secret-key-12345678901234567890

## Vanyospace OpenAI / Luna GPT-5.6
sk-openai-secret-key-12345678901234567890

## Vanyospace Anthropic / Claude Sonnet 5
sk-ant-api03-secret-key-12345678901234567890
"""
    creds = _LOADER.parse_credentials_text(text)

    repr_str = repr(creds)
    assert "secret-key" not in repr_str
    assert "CredentialSet" in repr_str
