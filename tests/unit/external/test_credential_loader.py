"""Unit tests for the environment-only external credential contract."""

from __future__ import annotations

import importlib.util
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


def _environment() -> dict[str, str]:
    return {
        "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY": "deepseek-test-key",
        "PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_API_KEY": "deepseek-test-key",
        "PRP_EXTERNAL_DEEPSEEK_FLASH_ANTHROPIC_API_KEY": "deepseek-test-key",
        "PRP_EXTERNAL_LUNA_GPT_56_API_KEY": "openai-test-key",
        "PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY": "anthropic-test-key",
    }


def test_environment_loader_exposes_all_profiles() -> None:
    creds = _LOADER.load_credentials_from_env(_environment())
    assert len(creds.aliases) == 5
    assert "DEEPSEEK_FLASH_CHAT" in creds.aliases
    assert creds.profile_env("DEEPSEEK_FLASH_CHAT")[
        "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY"
    ] == "deepseek-test-key"
    assert creds.profile_env("LUNA_GPT_56")[
        "PRP_EXTERNAL_LUNA_GPT_56_BASE_URL"
    ] == "https://fast.vanyospace.com"


def test_profile_contracts_define_exactly_five_profiles() -> None:
    assert set(_LOADER.PROFILE_CONTRACTS) == {
        "DEEPSEEK_FLASH_CHAT",
        "DEEPSEEK_FLASH_RESPONSES",
        "DEEPSEEK_FLASH_ANTHROPIC",
        "LUNA_GPT_56",
        "CLAUDE_SONNET_5",
    }


def test_profile_contracts_only_allow_two_hosts() -> None:
    authorized_hosts = {"api.deepseek.com", "fast.vanyospace.com"}
    from urllib.parse import urlsplit

    for profile_name, contract in _LOADER.PROFILE_CONTRACTS.items():
        hostname = urlsplit(contract["base_url"]).hostname
        assert hostname in authorized_hosts, profile_name


def test_missing_environment_key_raises_error() -> None:
    environment = _environment()
    del environment["PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY"]
    with pytest.raises(_LOADER.CredentialError, match="MISSING_ENV"):
        _LOADER.load_credentials_from_env(environment)


def test_placeholder_environment_key_is_rejected() -> None:
    environment = _environment()
    environment["PRP_EXTERNAL_LUNA_GPT_56_API_KEY"] = "<YOUR_PROVIDER_KEY>"
    with pytest.raises(_LOADER.CredentialError, match="MISSING_ENV"):
        _LOADER.load_credentials_from_env(environment)


def test_conflicting_shared_provider_keys_are_rejected() -> None:
    environment = _environment()
    environment["PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_API_KEY"] = "other-key"
    with pytest.raises(_LOADER.CredentialError, match="CONFLICTING_ENV"):
        _LOADER.load_credentials_from_env(environment)


def test_credentials_are_not_logged_in_repr() -> None:
    creds = _LOADER.load_credentials_from_env(_environment())
    representation = repr(creds)
    assert "deepseek-test-key" not in representation
    assert "CredentialSet" in representation


def test_terra_is_not_configured_without_complete_metadata() -> None:
    result = _LOADER.resolve_terra_profile(
        {},
        allowed_hosts=("terra.example",),
        fallback_from="LUNA_GPT_56",
        failure_classification="NETWORK",
    )
    assert result.alias == "TERRA_GPT"
    assert result.status == "TERRA_NOT_CONFIGURED"
    assert result.redacted()["api_key"] is None


def test_terra_requires_explicit_luna_fallback_and_admitted_host() -> None:
    environment = {
        "PRP_EXTERNAL_TERRA_GPT_MODEL": "terra-fixture-model",
        "PRP_EXTERNAL_TERRA_GPT_BASE_URL": "https://terra.example/v1",
        "PRP_EXTERNAL_TERRA_GPT_API_KEY": "terra-fixture-key",
        "PRP_EXTERNAL_TERRA_GPT_ALLOWED_HOST": "terra.example",
    }
    direct = _LOADER.resolve_terra_profile(environment, allowed_hosts=("terra.example",))
    assert direct.status == "FALLBACK_NOT_ALLOWED"
    ready = _LOADER.resolve_terra_profile(
        environment,
        allowed_hosts=("terra.example",),
        fallback_from="LUNA_GPT_56",
        failure_classification="NETWORK",
    )
    assert ready.status == "READY"
    assert ready.api_key == "terra-fixture-key"


def test_terra_fallback_rejects_non_retryable_luna_failure() -> None:
    result = _LOADER.resolve_terra_profile(
        {
            "PRP_EXTERNAL_TERRA_GPT_MODEL": "terra-fixture-model",
            "PRP_EXTERNAL_TERRA_GPT_BASE_URL": "https://terra.example/v1",
            "PRP_EXTERNAL_TERRA_GPT_API_KEY": "terra-fixture-key",
            "PRP_EXTERNAL_TERRA_GPT_ALLOWED_HOST": "terra.example",
        },
        allowed_hosts=("terra.example",),
        fallback_from="LUNA_GPT_56",
        failure_classification="UPSTREAM_AUTH_OR_PERMISSION",
    )
    assert result.status == "FALLBACK_NOT_ALLOWED"
