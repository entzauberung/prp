"""Capability-gated Agent workflow checks and local production gates."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from external_tests.capability_ledger import CapabilityStore
from external_tests.result_ledger import LedgerEntry, LedgerStore
from external_tests.support import (
    ExternalConfig,
    ExternalGateError,
    ExternalProfile,
    create_external_http_client,
    validate_external_url,
)
from external_tests.test_live_deepseek import _create_adapter, _profile_for_runtime
from prp_runtime.app import create_app
from prp_runtime.domain.enums import RunStatus, ToolCallStatus, ToolEffect
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

_OWNER_ID = "prn_agent_live"
_WORKSPACE_ID = "ws_project"
_SERVICE_TOKEN = "agent-live-service-token"
_READ_MARKER = "agent-read-evidence-731"
_AGENT_MAX_OUTPUT_TOKENS = 512


def _result_path() -> Path:
    value = os.environ.get("PRP_LIVE_RESULT_FILE")
    if value is None or not value.strip():
        raise ExternalGateError("PRP_LIVE_RESULT_FILE is required for Agent evidence")
    return Path(value)


def _capability_path() -> Path:
    value = os.environ.get("PRP_LIVE_CAPABILITY_FILE")
    if value is None or not value.strip():
        raise ExternalGateError("PRP_LIVE_CAPABILITY_FILE is required for Agent evidence")
    return Path(value)


def _tool_capable_aliases() -> set[str]:
    return {
        entry.alias for entry in CapabilityStore(_capability_path()).read()
        if entry.capability == "tool_call"
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
    }


def _tool_capable_profile(external_config: ExternalConfig):
    """Select the first configured profile with actual tool-call evidence."""
    capable_aliases = _tool_capable_aliases()
    for profile in external_config.profiles:
        if profile.alias in capable_aliases:
            return profile
    return None


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_SERVICE_TOKEN}"}


async def _snapshot_id(database_path: Path) -> str:
    async with SqliteStore(database_path) as store:
        snapshots = await store.list_snapshots(_WORKSPACE_ID, owner_id=_OWNER_ID)
        if len(snapshots) != 1:
            raise AssertionError("live Agent fixture must seed exactly one snapshot")
        return snapshots[0].snapshot_id


def _build_live_app(
    *,
    database_path: Path,
    workspace_root: Path,
    runtime_profile: Any,
    adapter: Any,
) -> tuple[Any, str]:
    from tests.integration.test_agent_api import seed_workspace

    seed_workspace(database_path, _OWNER_ID, workspace_root)
    snapshot_id = asyncio.run(_snapshot_id(database_path))
    settings = Settings(
        database_path=database_path,
        worker_profile=runtime_profile,
        service_token=SecretStr(_SERVICE_TOKEN),
        service_principal=_OWNER_ID,
        workspace_roots={"project-main": str(workspace_root)},
    )
    return create_app(settings, adapters={runtime_profile.alias: adapter}), snapshot_id


def _wait_for_terminal(
    client: TestClient,
    session_id: str,
    run_id: str,
    *,
    timeout_seconds: float,
    approval_limit: int | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    terminal = {status.value for status in RunStatus if status.is_terminal}
    while time.monotonic() < deadline:
        response = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}", headers=_auth_headers()
        )
        body = response.json()
        if body.get("status") in terminal:
            return body
        if approval_limit is not None:
            approvals_response = client.get(
                f"/v1/sessions/{session_id}/approvals",
                headers=_auth_headers(),
            )
            if len(approvals_response.json()) > approval_limit:
                raise AssertionError("real Agent exceeded the bounded approval count")
        time.sleep(0.02)
    raise AssertionError("real Agent run did not reach a terminal state")


def _wait_for_approval(
    client: TestClient,
    session_id: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(
            f"/v1/sessions/{session_id}/approvals", headers=_auth_headers()
        )
        approvals = response.json()
        if approvals:
            if len(approvals) != 1:
                raise AssertionError("real Agent write must request one bounded approval")
            return approvals[0]
        time.sleep(0.02)
    raise AssertionError("real Agent write did not request approval")


async def _run_facts(database_path: Path, run_id: str) -> dict[str, Any]:
    async with SqliteStore(database_path) as store:
        run = await store.get_run(run_id)
        attempts = await store.list_run_attempts(run_id)
        calls = await store.list_tool_calls(run_id)
        results = []
        for call in calls:
            results.append(await store.get_tool_result(call.call_id))
        approvals = await store.list_approvals(owner_id=_OWNER_ID, run_id=run_id)
        decisions = []
        for approval in approvals:
            decisions.append(
                await store.get_approval_decision(
                    approval.request_id,
                    owner_id=_OWNER_ID,
                )
            )
        change_sets = await store.list_change_sets(run_id=run_id)
        snapshots = await store.list_snapshots(_WORKSPACE_ID, owner_id=_OWNER_ID)
        return {
            "run": run,
            "attempts": attempts,
            "calls": calls,
            "results": tuple(results),
            "approvals": approvals,
            "decisions": tuple(decisions),
            "change_sets": change_sets,
            "snapshots": snapshots,
        }


def _record_actual_agent(
    *,
    profile: ExternalProfile,
    scenario: str,
    protocol: str,
    run_id: str,
    facts: dict[str, Any],
    output_text: str,
) -> None:
    attempts = facts["attempts"]
    first_attempt = attempts[0] if attempts else None
    run = facts["run"]
    base_scenario = f"wo-014-{scenario}"
    existing_ids = {
        entry.scenario_id for entry in LedgerStore(_result_path()).read()
    }
    scenario_id = (
        base_scenario
        if base_scenario not in existing_ids
        else f"{base_scenario}-retry-{run_id}"
    )
    LedgerStore(_result_path()).merge(
        [
            LedgerEntry(
                scenario_id=scenario_id,
                alias=profile.alias,
                model_id=profile.model_id,
                protocol=protocol,
                endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
                run_id=run_id,
                attempt_id=(
                    first_attempt.attempt_id
                    if first_attempt is not None
                    else "not-recorded"
                ),
                status="PASS",
                actual_or_simulated="ACTUAL",
                input_tokens=run.usage.input_tokens,
                output_tokens=run.usage.output_tokens,
                known_cost="unknown",
                latency_ms=run.usage.elapsed_ms,
                error_code=None,
                output_sha256=hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
                recorded_at=(
                    run.completed_at.isoformat()
                    if run.completed_at is not None
                    else "completed"
                ),
            )
        ],
        secret_values=[profile.api_key.get_secret_value()],
    )


def _agent_read_passed() -> bool:
    return any(
        entry.protocol == "AGENT_READ->SEARCH->READ"
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
        for entry in LedgerStore(_result_path()).read()
    )


def _agent_runtime_profile(profile: ExternalProfile) -> Any:
    return _profile_for_runtime(profile).model_copy(
        update={"max_output_tokens": _AGENT_MAX_OUTPUT_TOKENS}
    )


def _record_simulated(
    path: Path,
    *,
    scenario_id: str,
    protocol: str,
    error_code: str | None = None,
) -> None:
    LedgerStore(path).merge(
        [
            LedgerEntry(
                scenario_id=scenario_id,
                alias="LOCAL_AGENT",
                model_id="local-agent-fixture",
                protocol=protocol,
                endpoint_host="local.invalid",
                run_id=f"{scenario_id}-run",
                attempt_id=f"{scenario_id}-attempt",
                status="PASS" if error_code is None else "NOT_APPLICABLE",
                actual_or_simulated="SIMULATED",
                input_tokens=None,
                output_tokens=None,
                known_cost="unknown",
                latency_ms=None,
                error_code=error_code,
                output_sha256=None,
                recorded_at="simulated",
            )
        ]
    )


@pytest.mark.asyncio
async def test_local_agent_read_search_loop(tmp_path: Path) -> None:
    """Exercise the production read/search loop with isolated local adapters."""
    from tests.integration.test_agent_code_task import (
        test_read_search_agent_path_uses_workspace_tools_and_audited_session_run,
    )

    from prp_runtime.storage.sqlite import SqliteStore

    async with SqliteStore(tmp_path / "agent-read.db") as store:
        await test_read_search_agent_path_uses_workspace_tools_and_audited_session_run(
            store,
            tmp_path,
        )
    result_path = tmp_path / "agent-read-results.jsonl"
    _record_simulated(
        result_path,
        scenario_id="wo-004-st-001-local-read-search",
        protocol="AGENT_READ->SEARCH->TOOL_CALL",
    )
    assert LedgerStore(result_path).read()[0].actual_or_simulated == "SIMULATED"


@pytest.mark.asyncio
async def test_local_agent_write_approval_test_diff_loop(tmp_path: Path) -> None:
    """Exercise patch approval, registered test execution, and diff persistence."""
    from tests.integration.test_agent_code_task import (
        test_patch_test_agent_path_requires_write_approval_and_persists_changeset,
    )

    from prp_runtime.storage.sqlite import SqliteStore

    async with SqliteStore(tmp_path / "agent-write.db") as store:
        await test_patch_test_agent_path_requires_write_approval_and_persists_changeset(
            store,
            tmp_path,
        )
    result_path = tmp_path / "agent-write-results.jsonl"
    _record_simulated(
        result_path,
        scenario_id="wo-004-st-001-local-write-approval",
        protocol="AGENT_WRITE->APPROVAL->PATCH->TEST->DIFF",
    )
    assert LedgerStore(result_path).read()[0].actual_or_simulated == "SIMULATED"


@pytest.mark.live_agent
def test_read_tool_loop_requires_real_tool_capability(
    external_config: ExternalConfig,
    tmp_path: Path,
) -> None:
    profile = _tool_capable_profile(external_config)
    if profile is None:
        raise AssertionError("real Agent read requires an ACTUAL/PASS ToolCall profile")
    validate_external_url(profile.base_url, external_config.allowed_hosts)
    runtime_profile = _agent_runtime_profile(profile)
    workspace_root = tmp_path / "live-read-workspace"
    evidence_root = workspace_root / "evidence"
    evidence_root.mkdir(parents=True)
    evidence_file = evidence_root / "private-note.txt"
    evidence_file.write_text(
        f"verification_value: {_READ_MARKER}\n",
        encoding="utf-8",
    )
    database_path = tmp_path / "live-read.db"
    external_client = create_external_http_client(external_config.allowed_hosts)
    adapter = _create_adapter(runtime_profile, external_client)
    run_id: str | None = None
    final: dict[str, Any] | None = None
    try:
        app, _ = _build_live_app(
            database_path=database_path,
            workspace_root=workspace_root,
            runtime_profile=runtime_profile,
            adapter=adapter,
        )
        with TestClient(app) as client:
            session_response = client.post(
                "/v1/sessions",
                headers=_auth_headers(),
                json={
                    "workspace_id": _WORKSPACE_ID,
                    "access": ["READ"],
                    "agent_options": {
                        "agent_mode": "PLAN",
                        "isolation_mode": "HOST",
                        "execution_location": "CLOUD",
                    },
                },
            )
            assert session_response.status_code == 201
            session_id = session_response.json()["session_id"]
            instructions = (
                "Follow this finite state machine exactly and use one tool per turn. "
                "State 1: if no tool result exists, call search_text with pattern "
                "'verification_value', root 'evidence', and glob '**/*.txt'. "
                "State 2: after a successful search_text result, call read_file for "
                "'evidence/private-note.txt'. State 3: after a successful read_file "
                "result, call no more tools and answer with only the value after the "
                "'verification_value:' label. Never repeat a completed state."
            )
            run_response = client.post(
                f"/v1/sessions/{session_id}/runs",
                headers=_auth_headers(),
                json={
                    "input": "Retrieve the verification value from the isolated workspace.",
                    "instructions": instructions,
                    "routing_policy": "MANUAL",
                    "strategy": "DIRECT",
                    "budget": {"max_attempts": 8, "max_total_tokens": 50_000},
                },
            )
            assert run_response.status_code == 202
            run_id = run_response.json()["run_id"]
            final = _wait_for_terminal(
                client,
                session_id,
                run_id,
                timeout_seconds=240.0,
            )
            if client.portal is None:
                raise AssertionError("live Agent TestClient portal is unavailable")
            client.portal.call(external_client.aclose)
    finally:
        if not external_client.is_closed:
            try:
                asyncio.run(external_client.aclose())
            except RuntimeError:
                pass

    assert run_id is not None and final is not None
    assert final.get("status") == RunStatus.SUCCEEDED.value
    output_text = final.get("output_text")
    assert isinstance(output_text, str) and _READ_MARKER in output_text
    facts = asyncio.run(_run_facts(database_path, run_id))
    calls = facts["calls"]
    results = facts["results"]
    assert [call.tool_name for call in calls] == ["search_text", "read_file"]
    assert all(call.status is ToolCallStatus.SUCCEEDED for call in calls)
    assert all(result.status is ToolCallStatus.SUCCEEDED for result in results)
    assert results[0].result is not None and results[0].result["matches"]
    assert results[1].result is not None
    assert _READ_MARKER in results[1].result["content"]
    assert len(facts["attempts"]) == 1
    assert not facts["approvals"] and not facts["change_sets"]
    assert len(facts["snapshots"]) == 1
    _record_actual_agent(
        profile=profile,
        scenario="read-search-read",
        protocol="AGENT_READ->SEARCH->READ",
        run_id=run_id,
        facts=facts,
        output_text=output_text,
    )


@pytest.mark.live_agent
def test_write_tool_loop_requires_actual_prerequisite(
    external_config: ExternalConfig,
    tmp_path: Path,
) -> None:
    if not _agent_read_passed():
        raise AssertionError("real Agent write requires a prior ACTUAL/PASS read loop")
    profile = _tool_capable_profile(external_config)
    if profile is None:
        raise AssertionError("real Agent write requires an ACTUAL/PASS ToolCall profile")
    validate_external_url(profile.base_url, external_config.allowed_hosts)
    runtime_profile = _agent_runtime_profile(profile)
    workspace_root = tmp_path / "live-write-workspace"
    source_root = workspace_root / "src"
    source_root.mkdir(parents=True)
    target = source_root / "main.py"
    target.write_text('def answer():\n    return "before"\n', encoding="utf-8")
    (workspace_root / "test_fixture.py").write_text(
        'from src.main import answer\n\n\ndef test_answer():\n    assert answer() == "after"\n',
        encoding="utf-8",
    )
    database_path = tmp_path / "live-write.db"
    external_client = create_external_http_client(external_config.allowed_hosts)
    adapter = _create_adapter(runtime_profile, external_client)
    run_id: str | None = None
    final: dict[str, Any] | None = None
    expected_diff = (
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def answer():\n"
        '-    return "before"\n'
        '+    return "after"\n'
    )
    try:
        app, base_snapshot_id = _build_live_app(
            database_path=database_path,
            workspace_root=workspace_root,
            runtime_profile=runtime_profile,
            adapter=adapter,
        )
        with TestClient(app) as client:
            session_response = client.post(
                "/v1/sessions",
                headers=_auth_headers(),
                json={
                    "workspace_id": _WORKSPACE_ID,
                    "access": ["READ", "WRITE"],
                    "agent_options": {
                        "agent_mode": "AUTO",
                        "isolation_mode": "HOST",
                        "execution_location": "CLOUD",
                    },
                },
            )
            assert session_response.status_code == 201
            session_id = session_response.json()["session_id"]
            instructions = (
                "Follow this finite state machine exactly and use one tool per turn. "
                "State 1: if no tool result exists, call apply_patch once with "
                f"patch.base_snapshot_id '{base_snapshot_id}' and patch.unified_diff "
                "exactly equal to this diff:\n"
                f"{expected_diff}"
                "State 2: if history contains a successful apply_patch result but no "
                "successful run_targeted_test result, never patch again; call "
                "run_targeted_test with spec_name 'pytest' and parameters.targets "
                "containing only 'test_fixture.py'. State 3: if history contains a "
                "successful run_targeted_test result but no successful get_diff result, "
                "never patch or test again; call get_diff with an empty object. State 4: "
                "after a successful get_diff result, call no more tools and reply exactly "
                "agent-write-complete. Never repeat a completed state or simulate results."
            )
            run_response = client.post(
                f"/v1/sessions/{session_id}/runs",
                headers=_auth_headers(),
                json={
                    "input": "Apply, verify and inspect the isolated fixture change.",
                    "instructions": instructions,
                    "routing_policy": "MANUAL",
                    "strategy": "DIRECT",
                    "budget": {"max_attempts": 8, "max_total_tokens": 80_000},
                },
            )
            assert run_response.status_code == 202
            run_id = run_response.json()["run_id"]
            approval = _wait_for_approval(
                client,
                session_id,
                timeout_seconds=180.0,
            )
            assert approval["run_id"] == run_id
            assert approval["tool_name"] == "apply_patch"
            assert approval["effect"] == ToolEffect.WRITE.value
            assert approval["workspace_id"] == _WORKSPACE_ID
            assert approval["scope"]["tools"] == ["apply_patch"]
            assert approval["scope"]["paths"] == ["src/main.py"]
            assert target.read_text(encoding="utf-8").endswith('return "before"\n')
            decision_response = client.post(
                f"/v1/sessions/{session_id}/approvals/{approval['request_id']}/decision",
                headers=_auth_headers(),
                json={"outcome": "ALLOW"},
            )
            assert decision_response.status_code == 200
            assert decision_response.json()["outcome"] == "ALLOW"
            final = _wait_for_terminal(
                client,
                session_id,
                run_id,
                timeout_seconds=360.0,
                approval_limit=1,
            )
            if client.portal is None:
                raise AssertionError("live Agent TestClient portal is unavailable")
            client.portal.call(external_client.aclose)
    finally:
        if not external_client.is_closed:
            try:
                asyncio.run(external_client.aclose())
            except RuntimeError:
                pass

    assert run_id is not None and final is not None
    assert final.get("status") == RunStatus.SUCCEEDED.value
    output_text = final.get("output_text")
    assert isinstance(output_text, str) and output_text.strip()
    facts = asyncio.run(_run_facts(database_path, run_id))
    calls = facts["calls"]
    results = facts["results"]
    assert [call.tool_name for call in calls] == [
        "apply_patch",
        "run_targeted_test",
        "get_diff",
    ]
    assert all(call.status is ToolCallStatus.SUCCEEDED for call in calls)
    assert all(result.status is ToolCallStatus.SUCCEEDED for result in results)
    assert results[0].changed_paths == ("src/main.py",)
    assert results[1].exit_code == 0
    assert results[2].result is not None
    assert results[2].result["entries"][0]["path"] == "src/main.py"
    assert results[2].result["entries"][0]["status"] == "MODIFIED"
    assert len(facts["attempts"]) == 2
    assert len(facts["approvals"]) == len(facts["decisions"]) == 1
    assert facts["decisions"][0].outcome.value == "ALLOW"
    assert len(facts["change_sets"]) == 1
    assert facts["change_sets"][0].base_snapshot_id == base_snapshot_id
    assert len(facts["snapshots"]) == 2
    assert target.read_text(encoding="utf-8").endswith('return "after"\n')
    _record_actual_agent(
        profile=profile,
        scenario="write-approval-patch-test-diff",
        protocol="AGENT_WRITE->APPROVAL->PATCH->TEST->DIFF",
        run_id=run_id,
        facts=facts,
        output_text=output_text,
    )
