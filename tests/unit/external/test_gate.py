"""Offline tests for the explicit external-validation configuration gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_EXTERNAL_TESTS_DIR = Path(__file__).resolve().parents[3] / "external_tests"
if str(_EXTERNAL_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_TESTS_DIR))

from support import ExternalGateError, load_external_config  # noqa: E402


def _matrix(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "allowed_hosts": ["models.example"],
                "profiles": [
                    {
                        "alias": "TEST_PROFILE",
                        "vendor": "TEST",
                        "model_id": None,
                        "protocol": "OPENAI_CHAT",
                        "base_url_env": "PRP_EXTERNAL_TEST_BASE_URL",
                        "model_env": "PRP_EXTERNAL_TEST_MODEL",
                        "api_key_env": "PRP_EXTERNAL_TEST_API_KEY",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _valid_environment() -> dict[str, str]:
    return {
        "PRP_EXTERNAL_TESTS": "1",
        "PRP_EXTERNAL_PROFILE_ALIASES": " TEST_PROFILE ",
        "PRP_EXTERNAL_TEST_BASE_URL": "https://models.example/v1",
        "PRP_EXTERNAL_TEST_MODEL": "test-model",
        "PRP_EXTERNAL_TEST_API_KEY": "fixture-value",
    }


def _scoped_matrix(path: Path) -> None:
    profiles = [
        {
            "alias": "DEEPSEEK_FLASH_CHAT",
            "vendor": "DEEPSEEK",
            "protocol": "OPENAI_CHAT",
            "base_url_env": "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_BASE_URL",
            "model_env": "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_MODEL",
            "api_key_env": "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY",
        },
        {
            "alias": "DEEPSEEK_FLASH_RESPONSES",
            "vendor": "DEEPSEEK",
            "protocol": "OPENAI_RESPONSES",
            "base_url_env": "PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_BASE_URL",
            "model_env": "PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_MODEL",
            "api_key_env": "PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_API_KEY",
        },
        {
            "alias": "DEEPSEEK_FLASH_ANTHROPIC",
            "vendor": "DEEPSEEK",
            "protocol": "ANTHROPIC_MESSAGES",
            "base_url_env": "PRP_EXTERNAL_DEEPSEEK_FLASH_ANTHROPIC_BASE_URL",
            "model_env": "PRP_EXTERNAL_DEEPSEEK_FLASH_ANTHROPIC_MODEL",
            "api_key_env": "PRP_EXTERNAL_DEEPSEEK_FLASH_ANTHROPIC_API_KEY",
        },
        {
            "alias": "LUNA_GPT_56",
            "vendor": "INTERMEDIARY",
            "protocol": "OPENAI_RESPONSES",
            "base_url_env": "PRP_EXTERNAL_LUNA_GPT_56_BASE_URL",
            "model_env": "PRP_EXTERNAL_LUNA_GPT_56_MODEL",
            "api_key_env": "PRP_EXTERNAL_LUNA_GPT_56_API_KEY",
        },
        {
            "alias": "CLAUDE_SONNET_5",
            "vendor": "CLAUDE_GROUP",
            "protocol": "ANTHROPIC_MESSAGES",
            "base_url_env": "PRP_EXTERNAL_CLAUDE_SONNET_5_BASE_URL",
            "model_env": "PRP_EXTERNAL_CLAUDE_SONNET_5_MODEL",
            "api_key_env": "PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY",
        },
    ]
    path.write_text(
        json.dumps({"allowed_hosts": ["deepseek.example"], "profiles": profiles}),
        encoding="utf-8",
    )


def test_gate_requires_exact_opt_in() -> None:
    with pytest.raises(ExternalGateError, match="PRP_EXTERNAL_TESTS"):
        load_external_config({})


def test_gate_requires_explicit_profile_selection(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _matrix(matrix_path)
    environ = _valid_environment()
    del environ["PRP_EXTERNAL_PROFILE_ALIASES"]

    with pytest.raises(ExternalGateError, match="PRP_EXTERNAL_PROFILE_ALIASES"):
        load_external_config(environ, matrix_path)


def test_gate_reads_only_selected_profiles_and_hosts(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _scoped_matrix(matrix_path)
    environ = {
        "PRP_EXTERNAL_TESTS": "1",
        "PRP_EXTERNAL_PROFILE_ALIASES": (
            "DEEPSEEK_FLASH_CHAT, DEEPSEEK_FLASH_RESPONSES, "
            "DEEPSEEK_FLASH_ANTHROPIC"
        ),
    }
    for alias in (
        "DEEPSEEK_FLASH_CHAT",
        "DEEPSEEK_FLASH_RESPONSES",
        "DEEPSEEK_FLASH_ANTHROPIC",
    ):
        prefix = f"PRP_EXTERNAL_{alias}"
        environ.update(
            {
                f"{prefix}_BASE_URL": "https://deepseek.example/v1",
                f"{prefix}_MODEL": "confirmed-flash-fixture",
                f"{prefix}_API_KEY": "isolated-fixture-value",
            }
        )

    config = load_external_config(environ, matrix_path)

    assert [profile.alias for profile in config.profiles] == [
        "DEEPSEEK_FLASH_CHAT",
        "DEEPSEEK_FLASH_RESPONSES",
        "DEEPSEEK_FLASH_ANTHROPIC",
    ]
    assert config.allowed_hosts == ("deepseek.example",)
    assert "LUNA_GPT_56" not in repr(config)
    assert "CLAUDE_SONNET_5" not in repr(config)
    assert "isolated-fixture-value" not in repr(config)


@pytest.mark.parametrize(
    "selection, expected",
    [
        ("", "empty profile alias"),
        ("DEEPSEEK_FLASH_CHAT,,DEEPSEEK_FLASH_RESPONSES", "empty profile alias"),
        ("DEEPSEEK_FLASH_CHAT,DEEPSEEK_FLASH_CHAT", "duplicate profile alias"),
        ("UNKNOWN", "unknown profile alias"),
    ],
)
def test_gate_rejects_invalid_profile_selection(
    tmp_path: Path, selection: str, expected: str
) -> None:
    matrix_path = tmp_path / "matrix.json"
    _scoped_matrix(matrix_path)
    environ = {
        "PRP_EXTERNAL_TESTS": "1",
        "PRP_EXTERNAL_PROFILE_ALIASES": selection,
    }

    with pytest.raises(ExternalGateError, match=expected):
        load_external_config(environ, matrix_path)


def test_gate_reports_missing_environment_name_without_value(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _matrix(matrix_path)
    environ = _valid_environment()
    del environ["PRP_EXTERNAL_TEST_API_KEY"]

    with pytest.raises(ExternalGateError) as excinfo:
        load_external_config(environ, matrix_path)

    message = str(excinfo.value)
    assert "PRP_EXTERNAL_TEST_API_KEY" in message
    assert "fixture-value" not in message


def test_gate_rejects_malformed_opt_in_and_url(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _matrix(matrix_path)
    environ = _valid_environment()
    environ["PRP_EXTERNAL_TESTS"] = "yes"
    with pytest.raises(ExternalGateError, match="PRP_EXTERNAL_TESTS"):
        load_external_config(environ, matrix_path)

    environ["PRP_EXTERNAL_TESTS"] = "1"
    environ["PRP_EXTERNAL_TEST_BASE_URL"] = "http://models.example/v1"
    with pytest.raises(ExternalGateError, match="PRP_EXTERNAL_TEST_BASE_URL"):
        load_external_config(environ, matrix_path)

    environ["PRP_EXTERNAL_TEST_BASE_URL"] = "https://models.example/v1?token=fixture"
    with pytest.raises(ExternalGateError, match="PRP_EXTERNAL_TEST_BASE_URL"):
        load_external_config(environ, matrix_path)


def test_valid_profile_masks_secret_and_private_url_in_diagnostics(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _matrix(matrix_path)
    config = load_external_config(_valid_environment(), matrix_path)
    profile = config.profiles[0]

    assert profile.model_id == "test-model"
    assert profile.api_key.get_secret_value() == "fixture-value"
    assert "fixture-value" not in repr(profile)
    assert "/v1" not in repr(profile)
    assert "fixture-value" not in json.dumps(profile.redacted())
    assert "https://models.example/v1" not in json.dumps(profile.redacted())


def test_gate_rejects_malformed_matrix_without_echoing_contents(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text("{ malformed", encoding="utf-8")

    with pytest.raises(ExternalGateError, match="model matrix"):
        load_external_config(_valid_environment(), matrix_path)


def test_gate_validates_unselected_environment_name_collisions(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    _scoped_matrix(matrix_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["profiles"][3]["api_key_env"] = "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    environ = _valid_environment()

    with pytest.raises(ExternalGateError, match="duplicate api_key_env"):
        load_external_config(environ, matrix_path)
