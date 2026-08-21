"""Compatibility coverage for the retired relaxed-runner test path."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_ROOT / "external_tests"))
import unattended_runner  # noqa: E402


def test_current_runner_replaces_relaxed_runner_stage_api() -> None:
    assert "provider" in unattended_runner.STAGE_REGISTRY
    assert "strategy" in unattended_runner.STAGE_REGISTRY
    assert "agent" in unattended_runner.STAGE_REGISTRY
    assert all(
        spec.pytest_args[0] in {"--collect-only", "-m"}
        for spec in unattended_runner.STAGE_REGISTRY.values()
    )
