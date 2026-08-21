"""Targeted Settings parsing tests for the outbound protocol contract."""

import json

import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import ModelRole
from prp_runtime.providers.base import ProviderProtocol
from prp_runtime.settings import Settings


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
