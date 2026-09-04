"""Targeted Settings parsing tests for the outbound protocol contract."""

import json

import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import ModelRole
from prp_runtime.providers.base import ProviderProtocol
from prp_runtime.settings import (
    DEFAULT_PROCESS_MAX_ATTEMPTS,
    DEFAULT_PROCESS_MAX_CONCURRENCY,
    DEFAULT_PROCESS_MAX_TOTAL_TOKENS,
    MAX_PROCESS_MAX_ATTEMPTS,
    MAX_PROCESS_MAX_CONCURRENCY,
    MAX_PROCESS_MAX_TOTAL_TOKENS,
    Settings,
)


def _profile(protocol: str | None = None, **extra: object) -> dict[str, object]:
    profile: dict[str, object] = {
        "alias": "worker",
        "provider": "test-provider",
        "model": "test-model",
        "role": ModelRole.WORKER.value,
        "base_url": "https://models.example/v1",
        "context_window_tokens": 1_024,
        "max_output_tokens": 128,
    }
    if protocol is not None:
        profile["protocol"] = protocol
    profile.update(extra)
    return profile


def test_settings_keep_legacy_profile_defaulting_to_openai_chat() -> None:
    settings = Settings.from_env({"PRP_WORKER_PROFILE": json.dumps(_profile())})
    assert settings.worker_profile is not None
    assert settings.worker_profile.protocol is ProviderProtocol.OPENAI_CHAT


def test_settings_parse_responses_and_anthropic_profiles() -> None:
    responses = Settings.from_env(
        {
            "PRP_WORKER_PROFILE": json.dumps(
                _profile(ProviderProtocol.OPENAI_RESPONSES.value)
            )
        }
    )
    assert responses.worker_profile is not None
    assert responses.worker_profile.protocol is ProviderProtocol.OPENAI_RESPONSES

    anthropic = Settings.from_env(
        {
            "PRP_WORKER_PROFILE": json.dumps(
                _profile(
                    ProviderProtocol.ANTHROPIC_MESSAGES.value,
                    anthropic_version="2023-06-01",
                )
            )
        }
    )
    assert anthropic.worker_profile is not None
    assert anthropic.worker_profile.anthropic_version == "2023-06-01"


def test_settings_reject_protocol_version_mismatch_and_unknown_protocol() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env(
            {
                "PRP_WORKER_PROFILE": json.dumps(
                    _profile(
                        ProviderProtocol.OPENAI_CHAT.value,
                        anthropic_version="2023-06-01",
                    )
                )
            }
        )
    with pytest.raises(ValidationError):
        Settings.from_env(
            {"PRP_WORKER_PROFILE": json.dumps(_profile("UNKNOWN_PROTOCOL"))}
        )


def test_settings_expose_bounded_isolation_capacity() -> None:
    settings = Settings()
    assert settings.isolation_max_slots == 2
    assert settings.isolation_max_bytes == 256 * 1024 * 1024
    from_env = Settings.from_env(
        {
            "PRP_ISOLATION_MAX_SLOTS": "1",
            "PRP_ISOLATION_MAX_BYTES": "1048576",
        }
    )
    assert from_env.isolation_max_slots == 1
    assert from_env.isolation_max_bytes == 1_048_576
    with pytest.raises(ValidationError):
        Settings(isolation_max_slots=0)
    with pytest.raises(ValidationError):
        Settings(isolation_max_bytes=0)
    with pytest.raises(ValidationError):
        Settings(isolation_max_slots=9)
    with pytest.raises(ValidationError):
        Settings(isolation_max_bytes=512 * 1024 * 1024 + 1)


def test_settings_expose_bounded_process_resource_envelope() -> None:
    settings = Settings()
    envelope = settings.resource_envelope
    assert settings.process_max_concurrency == DEFAULT_PROCESS_MAX_CONCURRENCY
    assert settings.process_max_attempts == DEFAULT_PROCESS_MAX_ATTEMPTS
    assert settings.process_max_total_tokens == DEFAULT_PROCESS_MAX_TOTAL_TOKENS
    assert envelope.max_slots == settings.isolation_max_slots
    assert envelope.max_copied_bytes == settings.isolation_max_bytes
    assert envelope.max_concurrency == 1
    assert envelope.max_attempts == 8
    assert envelope.max_total_tokens == 250_000
    from_env = Settings.from_env(
        {
            "PRP_PROCESS_MAX_CONCURRENCY": "2",
            "PRP_PROCESS_MAX_ATTEMPTS": "4",
            "PRP_PROCESS_MAX_TOTAL_TOKENS": "1024",
        }
    )
    assert from_env.process_max_concurrency == 2
    assert from_env.process_max_attempts == 4
    assert from_env.process_max_total_tokens == 1024
    with pytest.raises(ValidationError):
        Settings(process_max_concurrency=0)
    with pytest.raises(ValidationError):
        Settings(process_max_attempts=-1)
    with pytest.raises(ValidationError):
        Settings(process_max_total_tokens=0)
    with pytest.raises(ValidationError):
        Settings(process_max_concurrency=MAX_PROCESS_MAX_CONCURRENCY + 1)
    with pytest.raises(ValidationError):
        Settings(process_max_attempts=MAX_PROCESS_MAX_ATTEMPTS + 1)
    with pytest.raises(ValidationError):
        Settings(process_max_total_tokens=MAX_PROCESS_MAX_TOTAL_TOKENS + 1)


def _role_profile(role: str, alias: str) -> dict[str, object]:
    return {
        "alias": alias,
        "provider": "test-provider",
        "model": f"{alias}-model",
        "role": role,
        "base_url": "https://models.example/v1",
        "context_window_tokens": 1_024,
        "max_output_tokens": 128,
    }


def test_analyzer_and_verifier_profiles_are_not_worker_aliases() -> None:
    settings = Settings.from_env(
        {
            "PRP_WORKER_PROFILE": json.dumps(_profile()),
            "PRP_ANALYZER_PROFILE": json.dumps(_role_profile("ANALYZER", "analyzer")),
            "PRP_VERIFIER_PROFILE": json.dumps(_role_profile("VERIFIER", "verifier")),
        }
    )
    assert settings.profile_for_role(ModelRole.ANALYZER) is not settings.worker_profile
    assert settings.require_profile(ModelRole.ANALYZER).role is ModelRole.ANALYZER
    assert settings.require_profile(ModelRole.VERIFIER).role is ModelRole.VERIFIER
    assert settings.require_profile(ModelRole.WORKER).role is ModelRole.WORKER


def test_missing_analyzer_or_verifier_is_not_mapped_to_worker() -> None:
    settings = Settings.from_env({"PRP_WORKER_PROFILE": json.dumps(_profile())})
    assert settings.profile_for_role(ModelRole.ANALYZER) is None
    assert settings.profile_for_role(ModelRole.VERIFIER) is None
    with pytest.raises(Exception, match="ANALYZER"):
        settings.require_profile(ModelRole.ANALYZER)
    with pytest.raises(Exception, match="VERIFIER"):
        settings.require_profile(ModelRole.VERIFIER)


def test_settings_reject_analyzer_or_verifier_role_mismatch() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env(
            {"PRP_ANALYZER_PROFILE": json.dumps(_profile(alias="analyzer"))}
        )
    with pytest.raises(ValidationError):
        Settings.from_env(
            {
                "PRP_VERIFIER_PROFILE": json.dumps(
                    _role_profile("WORKER", "verifier")
                )
            }
        )
