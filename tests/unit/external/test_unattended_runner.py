"""Targeted tests for the authoritative unattended runner entry points."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_ROOT / "external_tests"))
import unattended_runner  # noqa: E402
from credential_loader import load_credentials_from_env  # noqa: E402


@pytest.fixture
def dummy_credentials():
    return load_credentials_from_env(
        {
            "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY": "deepseek-dummy-key",
            "PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_API_KEY": "deepseek-dummy-key",
            "PRP_EXTERNAL_DEEPSEEK_FLASH_ANTHROPIC_API_KEY": "deepseek-dummy-key",
            "PRP_EXTERNAL_LUNA_GPT_56_API_KEY": "openai-dummy-key",
            "PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY": "anthropic-dummy-key",
        }
    )


def test_stage_result_file_routing_is_protocol_specific() -> None:
    assert unattended_runner.LOG_ROOT.as_posix().endswith("external_tests/.results")
    assert unattended_runner.STAGE_RESULT_FILES["protocols"].name == "30-responses.jsonl"
    assert unattended_runner.STAGE_RESULT_FILES["strategies"].name == "50-reasoning.jsonl"
    assert unattended_runner.STAGE_RESULT_FILES["agent"].name == "60-agent.jsonl"
    assert (
        unattended_runner.INTERFACE_RESULT_FILES[
            unattended_runner.PROTOCOL_CASE_INTERFACES["chat"]
        ].name
        == "20-chat.jsonl"
    )
    assert (
        unattended_runner.INTERFACE_RESULT_FILES[
            unattended_runner.PROTOCOL_CASE_INTERFACES["anthropic"]
        ].name
        == "40-anthropic.jsonl"
    )


def test_first_actual_pass_candidate_uses_interface_order() -> None:
    entries = (
        SimpleNamespace(
            alias="DEEPSEEK_FLASH_ANTHROPIC",
            protocol="ANTHROPIC_MESSAGES",
            status="PASS",
            actual_or_simulated="SIMULATED",
        ),
        SimpleNamespace(
            alias="CLAUDE_SONNET_5",
            protocol="ANTHROPIC_MESSAGES",
            status="PASS",
            actual_or_simulated="ACTUAL",
        ),
    )

    selected = unattended_runner._first_actual_pass_candidate(
        "ANTHROPIC_MESSAGES",
        ("DEEPSEEK_FLASH_ANTHROPIC", "CLAUDE_SONNET_5"),
        entries,
    )

    assert selected == "CLAUDE_SONNET_5"


def test_first_actual_pass_candidate_rejects_wrong_protocol() -> None:
    entries = (
        SimpleNamespace(
            alias="CLAUDE_SONNET_5",
            protocol="OPENAI_RESPONSES",
            status="PASS",
            actual_or_simulated="ACTUAL",
        ),
    )

    with pytest.raises(ValueError, match="prior actual PASS"):
        unattended_runner._first_actual_pass_candidate(
            "ANTHROPIC_MESSAGES",
            ("DEEPSEEK_FLASH_ANTHROPIC", "CLAUDE_SONNET_5"),
            entries,
        )


def test_capability_candidates_can_use_actual_capability_pass(
    monkeypatch, tmp_path
) -> None:
    result_file = tmp_path / "provider.json"
    monkeypatch.setattr(
        unattended_runner.CapabilityStore,
        "read",
        lambda self: (
            SimpleNamespace(
                alias="DEEPSEEK_FLASH_RESPONSES",
                protocol="OPENAI_RESPONSES",
                status="PASS",
                actual_or_simulated="ACTUAL",
            ),
        ),
    )

    selected = unattended_runner._capability_probe_candidates(
        "OPENAI_RESPONSES",
        ("DEEPSEEK_FLASH_RESPONSES", "LUNA_GPT_56"),
        result_file,
    )

    assert selected == ("DEEPSEEK_FLASH_RESPONSES",)


def test_interface_selection_is_deterministic_and_scoped(
    monkeypatch, dummy_credentials
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(unattended_runner.subprocess, "run", fake_run)
    result = unattended_runner.run_stage(
        "provider", dummy_credentials, interface="OPENAI_RESPONSES"
    )

    assert result.interface == "OPENAI_RESPONSES"
    assert result.candidate_aliases == (
        "DEEPSEEK_FLASH_RESPONSES",
        "LUNA_GPT_56",
    )
    env = captured["env"]
    assert env["PRP_EXTERNAL_PROFILE_ALIASES"] == (
        "DEEPSEEK_FLASH_RESPONSES,LUNA_GPT_56"
    )
    assert env["PRP_EXTERNAL_INTERFACE"] == "OPENAI_RESPONSES"
    assert "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY" not in env
    assert "PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY" not in env


def test_interface_rejects_cross_interface_selection(monkeypatch, dummy_credentials) -> None:
    monkeypatch.setattr(unattended_runner.subprocess, "run", pytest.fail)

    with pytest.raises(ValueError, match="not a candidate"):
        unattended_runner.run_stage(
            "provider",
            dummy_credentials,
            interface="OPENAI_CHAT",
            select=("CLAUDE_SONNET_5",),
        )


def test_interface_cli_accepts_documented_lower_case_name() -> None:
    args = unattended_runner._parse_args(
        ["--stage", "providers", "--interface", "openai_chat"]
    )
    assert args.interface == "OPENAI_CHAT"


def test_anthropic_case_maps_to_messages(monkeypatch, dummy_credentials) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(unattended_runner.subprocess, "run", fake_run)
    unattended_runner.run_stage("protocols", dummy_credentials, case="anthropic")

    argv = captured["argv"]
    assert argv[-2:] == ["-k", "messages"]


def test_run_stage_uses_structured_single_child(monkeypatch, dummy_credentials) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(unattended_runner.subprocess, "run", fake_run)
    result = unattended_runner.run_stage("strategy", dummy_credentials)

    assert result.status == "PASS"
    assert captured["argv"][:3] == [unattended_runner.sys.executable, "-m", "pytest"]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["timeout"] == 900


def test_run_stage_selection_injects_only_selected_active_profile(
    monkeypatch, dummy_credentials
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(unattended_runner.subprocess, "run", fake_run)
    unattended_runner.run_stage(
        "provider", dummy_credentials, select=("DEEPSEEK_FLASH_RESPONSES",)
    )

    child_env = captured["kwargs"]["env"]
    assert child_env["PRP_EXTERNAL_PROFILE_ALIASES"] == "DEEPSEEK_FLASH_RESPONSES"
    assert "PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_API_KEY" in child_env
    assert "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY" not in child_env
    assert "PRP_EXTERNAL_LUNA_GPT_56_API_KEY" not in child_env


def test_run_stage_classifies_unconfigured_terra_without_starting_process(
    monkeypatch, dummy_credentials
) -> None:
    monkeypatch.setattr(unattended_runner.subprocess, "run", pytest.fail)

    result = unattended_runner.run_stage(
        "provider", dummy_credentials, select=("TERRA_GPT",)
    )

    assert result.status == "TERRA_NOT_CONFIGURED"
    assert result.exit_code == 0


def test_run_stage_allows_terra_only_after_retryable_luna_failure(
    monkeypatch, dummy_credentials
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(unattended_runner.subprocess, "run", fake_run)
    monkeypatch.setenv("PRP_EXTERNAL_TERRA_GPT_MODEL", "terra-fixture-model")
    monkeypatch.setenv("PRP_EXTERNAL_TERRA_GPT_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("PRP_EXTERNAL_TERRA_GPT_API_KEY", "terra-fixture-key")
    monkeypatch.setenv("PRP_EXTERNAL_TERRA_GPT_ALLOWED_HOST", "api.deepseek.com")

    result = unattended_runner.run_stage(
        "provider",
        dummy_credentials,
        select=("TERRA_GPT",),
        fallback_from="LUNA_GPT_56",
        fallback_failure="NETWORK",
    )

    assert result.status == "PASS"
    child_env = captured["kwargs"]["env"]
    assert child_env["PRP_EXTERNAL_PROFILE_ALIASES"] == "TERRA_GPT"
    assert child_env["PRP_EXTERNAL_TERRA_GPT_MODEL"] == "terra-fixture-model"
    assert child_env["PRP_EXTERNAL_TERRA_GPT_API_KEY"] == "terra-fixture-key"
    assert "PRP_EXTERNAL_LUNA_GPT_56_API_KEY" not in child_env
