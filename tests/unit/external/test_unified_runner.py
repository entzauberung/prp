"""Unit tests for the unified unattended_runner module.

This test suite validates the complete stage registry, credential injection,
output redaction, and subprocess invocation mechanics.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).parents[3]
_LOADER_SPEC = importlib.util.spec_from_file_location(
    "prp_external_credential_loader_for_runner",
    _ROOT / "external_tests" / "credential_loader.py",
)
assert _LOADER_SPEC is not None and _LOADER_SPEC.loader is not None
_LOADER = importlib.util.module_from_spec(_LOADER_SPEC)
_LOADER_SPEC.loader.exec_module(_LOADER)

_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "prp_external_unattended_runner",
    _ROOT / "external_tests" / "unattended_runner.py",
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)


@pytest.fixture
def dummy_credentials():
    """Create dummy credentials for testing."""
    return _LOADER.load_credentials_from_env(
        {
            "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY": "deepseek-dummy-key",
            "PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_API_KEY": "deepseek-dummy-key",
            "PRP_EXTERNAL_DEEPSEEK_FLASH_ANTHROPIC_API_KEY": "deepseek-dummy-key",
            "PRP_EXTERNAL_LUNA_GPT_56_API_KEY": "openai-dummy-key",
            "PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY": "anthropic-dummy-key",
        }
    )


def test_stage_registry_contains_authoritative_stages() -> None:
    """Stage registry should contain canonical names and safe aliases."""
    expected_stages = {
        "preflight",
        "provider",
        "providers",
        "integration",
        "protocols",
        "strategy",
        "strategies",
        "agent",
        "regression",
    }
    assert set(_RUNNER.STAGE_REGISTRY.keys()) == expected_stages

    # Verify preflight is still collection-only
    assert _RUNNER.STAGE_REGISTRY["preflight"].pytest_args[:2] == ("--collect-only", "-q")

    # Verify live stages use markers
    assert "-m" in _RUNNER.STAGE_REGISTRY["provider"].pytest_args
    assert "live_provider" in _RUNNER.STAGE_REGISTRY["provider"].pytest_args

    # Verify strategy and agent have longer timeouts
    assert _RUNNER.STAGE_REGISTRY["strategy"].timeout_seconds == 900
    assert _RUNNER.STAGE_REGISTRY["agent"].timeout_seconds == 900

    # Verify no stage uses shell separators
    for stage_spec in _RUNNER.STAGE_REGISTRY.values():
        assert "--" not in stage_spec.pytest_args


def test_interface_candidate_matrix_is_protocol_scoped() -> None:
    assert _RUNNER.INTERFACE_CANDIDATES["OPENAI_CHAT"] == (
        "DEEPSEEK_FLASH_CHAT",
    )
    assert _RUNNER.INTERFACE_CANDIDATES["OPENAI_RESPONSES"] == (
        "DEEPSEEK_FLASH_RESPONSES",
        "LUNA_GPT_56",
        "TERRA_GPT",
    )
    assert _RUNNER.INTERFACE_CANDIDATES["ANTHROPIC_MESSAGES"] == (
        "DEEPSEEK_FLASH_ANTHROPIC",
        "CLAUDE_SONNET_5",
    )


def test_child_env_clears_ambient_profiles_and_proxies(dummy_credentials) -> None:
    """Child env should remove ambient PRP_EXTERNAL_ vars and proxies."""
    env = _RUNNER.build_child_env(
        dummy_credentials,
        {
            "PATH": "/usr/bin",
            "PRP_EXTERNAL_OLD_API_KEY": "ambient-secret",
            "HTTPS_PROXY": "http://proxy.invalid",
            "http_proxy": "http://proxy.invalid",
            "KEEP": "yes",
        },
    )

    assert env["KEEP"] == "yes"
    assert "PRP_EXTERNAL_OLD_API_KEY" not in env
    assert "HTTPS_PROXY" not in env
    assert "http_proxy" not in env
    assert len([name for name in env if name.startswith("PRP_EXTERNAL_")]) == 17
    assert "ambient-secret" not in repr(env)


def test_child_env_adds_all_profiles(dummy_credentials) -> None:
    """Child env should add all five profile credential sets."""
    env = _RUNNER.build_child_env(dummy_credentials)

    # Check DeepSeek profiles (all use deepseek credential)
    assert "PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY" in env
    assert env["PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY"] == "deepseek-dummy-key"
    assert "PRP_EXTERNAL_DEEPSEEK_FLASH_RESPONSES_API_KEY" in env
    assert "PRP_EXTERNAL_DEEPSEEK_FLASH_ANTHROPIC_API_KEY" in env

    # Check Luna profile (uses openai credential)
    assert "PRP_EXTERNAL_LUNA_GPT_56_API_KEY" in env
    assert env["PRP_EXTERNAL_LUNA_GPT_56_API_KEY"] == "openai-dummy-key"

    # Check Claude profile (uses anthropic credential)
    assert "PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY" in env
    assert env["PRP_EXTERNAL_CLAUDE_SONNET_5_API_KEY"] == "anthropic-dummy-key"


def test_stage_uses_one_structured_child_and_redacts_output(monkeypatch, dummy_credentials) -> None:
    """run_stage should invoke subprocess with structured argv and redact secrets."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="child says deepseek-dummy-key",
            stderr="child err openai-dummy-key",
        )

    monkeypatch.setattr(_RUNNER.subprocess, "run", fake_run)
    result = _RUNNER.run_stage("preflight", dummy_credentials)

    assert result.status == "PASS"
    assert result.exit_code == 0
    assert captured["argv"][:4] == [
        _RUNNER.sys.executable,
        "-m",
        "pytest",
        "--collect-only",
    ]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["cwd"] == _RUNNER.REPO_ROOT
    assert (
        captured["kwargs"]["env"]["PRP_EXTERNAL_LUNA_GPT_56_API_KEY"]
        == "openai-dummy-key"
    )
    assert "deepseek-dummy-key" not in result.stdout
    assert "openai-dummy-key" not in result.stderr


def test_child_exit_code_and_timeout_are_propagated(monkeypatch, dummy_credentials) -> None:
    """Child exit code and timeout should propagate correctly."""
    monkeypatch.setattr(
        _RUNNER.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout="", stderr=""),
    )
    failed = _RUNNER.run_stage("preflight", dummy_credentials)
    assert failed.status == "FAIL"
    assert failed.exit_code == 7

    def timeout(*args, **kwargs):
        raise _RUNNER.subprocess.TimeoutExpired(kwargs["timeout"], args[0])

    monkeypatch.setattr(_RUNNER.subprocess, "run", timeout)
    timed_out = _RUNNER.run_stage("preflight", dummy_credentials)
    assert timed_out.status == "TIMEOUT"
    assert timed_out.exit_code == 124


def test_unknown_stage_fails_without_starting_process(monkeypatch, dummy_credentials) -> None:
    """Unknown stage should raise ValueError without starting a subprocess."""
    monkeypatch.setattr(_RUNNER.subprocess, "run", pytest.fail)

    with pytest.raises(ValueError, match="unknown stage"):
        _RUNNER.run_stage("unknown_stage", dummy_credentials)


def test_stage_specific_timeouts_override_default(monkeypatch, dummy_credentials) -> None:
    """Each stage should use its registered timeout unless overridden."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_RUNNER.subprocess, "run", fake_run)

    # Test preflight default
    _RUNNER.run_stage("preflight", dummy_credentials)
    assert captured["timeout"] == _RUNNER.DEFAULT_TIMEOUT_SECONDS

    # Test strategy long timeout
    _RUNNER.run_stage("strategy", dummy_credentials)
    assert captured["timeout"] == 900

    # Test explicit override
    _RUNNER.run_stage("provider", dummy_credentials, timeout_seconds=999)
    assert captured["timeout"] == 999


def test_redaction_removes_all_credential_types(dummy_credentials) -> None:
    """Output redaction should remove all three credential classes."""
    text = (
        "DeepSeek: deepseek-dummy-key\n"
        "OpenAI: openai-dummy-key\n"
        "Anthropic: anthropic-dummy-key"
    )
    redacted = _RUNNER._redact_output(text, dummy_credentials)

    assert "deepseek-dummy-key" not in redacted
    assert "openai-dummy-key" not in redacted
    assert "anthropic-dummy-key" not in redacted
    assert "<redacted>" in redacted


def test_host_aliases_extracted_from_profile_contracts() -> None:
    """Host aliases should be extracted from PROFILE_CONTRACTS base URLs."""
    host_aliases = _RUNNER._host_aliases()

    # Should have exactly two hosts
    assert len(host_aliases) == 2
    assert "api.deepseek.com" in host_aliases
    assert "fast.vanyospace.com" in host_aliases

    # Should be sorted
    assert host_aliases == tuple(sorted(host_aliases))


def test_stage_result_includes_presence_count_and_hosts(monkeypatch, dummy_credentials) -> None:
    """StageResult should include presence_count and host_aliases."""
    monkeypatch.setattr(
        _RUNNER.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = _RUNNER.run_stage("provider", dummy_credentials)

    # Five profiles use three fields each, plus the two external-test controls.
    assert result.presence_count == 17

    # Should have exactly two host aliases
    assert len(result.host_aliases) == 2
    assert "api.deepseek.com" in result.host_aliases
    assert "fast.vanyospace.com" in result.host_aliases
