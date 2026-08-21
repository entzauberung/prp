"""Targeted tests for the authoritative unattended runner entry points."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_ROOT / "external_tests"))
from credential_loader import parse_credentials_text  # noqa: E402
import unattended_runner  # noqa: E402


@pytest.fixture
def dummy_credentials():
    return parse_credentials_text(
        """\
## DeepSeek API
sk-deepseek-dummy-key-12345678901234567890

## Vanyospace OpenAI / Luna GPT-5.6
sk-openai-dummy-key-12345678901234567890

## Vanyospace Anthropic / Claude Sonnet 5
sk-anthropic-dummy-key-12345678901234567890
"""
    )


def test_stage_result_file_routing_is_protocol_specific() -> None:
    assert unattended_runner.STAGE_RESULT_FILES["protocols"].name == "20-protocols.jsonl"
    assert unattended_runner.STAGE_RESULT_FILES["strategies"].name == "30-strategies.jsonl"
    assert unattended_runner.STAGE_RESULT_FILES["agent"].name == "40-agent.jsonl"


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
