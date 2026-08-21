"""Capability-gated real Agent workflow checks."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from external_tests.capability_ledger import CapabilityStore
from external_tests.result_ledger import LedgerEntry, LedgerStore
from external_tests.support import ExternalConfig, ExternalGateError
from external_tests.test_live_protocols import _profile_by_alias

CAPABILITY_FILE = Path("/home/bruce/prp/ai/PROVIDER-CAPABILITIES.json")
ALIAS = "DEEPSEEK_FLASH_RESPONSES"


def _result_path() -> Path:
    value = os.environ.get("PRP_LIVE_RESULT_FILE")
    if value is None or not value.strip():
        raise ExternalGateError("PRP_LIVE_RESULT_FILE is required for Agent evidence")
    return Path(value)


def _tool_capability_passed() -> bool:
    return any(
        entry.alias == ALIAS
        and entry.capability == "tool_call"
        and entry.status == "PASS"
        for entry in CapabilityStore(CAPABILITY_FILE).read()
    )


@pytest.mark.live_agent
def test_read_tool_loop_requires_real_tool_capability(
    external_config: ExternalConfig,
) -> None:
    profile = _profile_by_alias(external_config, ALIAS)
    if not _tool_capability_passed():
        LedgerStore(_result_path()).merge(
            [
                LedgerEntry(
                    scenario_id="wo-005-st-001-read-loop",
                    alias=profile.alias,
                    model_id=profile.model_id,
                    protocol="AGENT_READ->TOOL_CALL",
                    endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
                    run_id="not-run",
                    attempt_id="not-run",
                    status="NOT_APPLICABLE",
                    actual_or_simulated="SIMULATED",
                    input_tokens=None,
                    output_tokens=None,
                    known_cost="unknown",
                    latency_ms=None,
                    error_code="PREREQUISITE_NO_TOOL_CALL_PASS",
                    output_sha256=None,
                    recorded_at="not-run",
                )
            ]
        )
        pytest.skip("real Agent read loop requires a prior tool_call capability PASS")

    raise AssertionError(
        "real tool capability is available, but the bounded production Agent read "
        "loop has not been implemented in this test"
    )


@pytest.mark.live_agent
def test_write_tool_loop_requires_read_capability(
    external_config: ExternalConfig,
) -> None:
    profile = _profile_by_alias(external_config, ALIAS)
    if not _tool_capability_passed():
        LedgerStore(_result_path()).merge(
            [
                LedgerEntry(
                    scenario_id="wo-005-st-002-write-loop",
                    alias=profile.alias,
                    model_id=profile.model_id,
                    protocol="AGENT_WRITE->APPROVAL_PATCH_TEST_DIFF",
                    endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
                    run_id="not-run",
                    attempt_id="not-run",
                    status="NOT_APPLICABLE",
                    actual_or_simulated="SIMULATED",
                    input_tokens=None,
                    output_tokens=None,
                    known_cost="unknown",
                    latency_ms=None,
                    error_code="PREREQUISITE_NO_TOOL_CALL_PASS",
                    output_sha256=None,
                    recorded_at="not-run",
                )
            ]
        )
        pytest.skip("real Agent write loop requires a prior tool_call capability PASS")

    raise AssertionError(
        "real Agent write execution must be implemented only after the read loop "
        "has a verified real ToolCall path"
    )
