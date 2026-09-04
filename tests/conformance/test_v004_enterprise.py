"""v0.0.4 enterprise security and observability regression matrix.

This file adds focused assertions for global hard-stop invariants. Historical
tests remain the compatibility floor and are not rewritten here.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from prp_runtime.api.bindings import reject_unsupported_fields
from prp_runtime.client.bridge import (
    BridgeProtocolError,
    _assert_remote_safe,
    _public_event,
    public_dispatch_payload,
)
from prp_runtime.control.progressive import (
    ComparisonOutcome,
    ReuseDisposition,
    ReuseReason,
    compare_rounds,
    decide_reuse,
)
from prp_runtime.domain.enums import ExecutionLocation, RunStatus, ToolEffect, WorkUnitStatus
from prp_runtime.domain.errors import ErrorCode, PrpError, StateError
from prp_runtime.domain.events import EventType, RunEvent, assert_sequence_chain
from prp_runtime.domain.models import (
    GlobalCheck,
    GlobalVerificationReport,
    NativeRunRequest,
    Session,
    Usage,
    VerificationResult,
    WorkUnit,
    WorkspaceGrant,
    new_client_id,
)
from prp_runtime.domain.values import (
    new_run_id,
    new_session_id,
    new_tool_call_id,
    new_work_unit_id,
    new_workspace_id,
    utc_now,
)
from prp_runtime.runtime.conflicts import ConflictFacts, ConflictKind, classify_facts
from prp_runtime.tools.models import BridgeClaim, BridgeClaimStatus
from prp_runtime.workspace.backend import _export_path_is_excluded

T0 = datetime(2026, 9, 2, tzinfo=UTC)


def _work_unit(*, lineage: str = "stable-node", status: WorkUnitStatus = WorkUnitStatus.SUCCEEDED) -> WorkUnit:
    return WorkUnit(
        work_unit_id="wu_reuse_candidate",
        run_id="run_reuse",
        graph_version=2,
        lineage_key=lineage,
        dependency_fingerprint="1" * 64,
        content_fingerprint="a" * 64,
        name="node",
        instruction="produce node",
        status=status,
    )


def _global_report(result: VerificationResult, *, evidence_ids: tuple[str, ...] = ()) -> GlobalVerificationReport:
    return GlobalVerificationReport(
        run_id="run_progressive",
        round_id="round_" + "a" * 32,
        graph_version=2,
        result=result,
        checks=(
            GlobalCheck(
                kind="EVIDENCE",
                result=result,
                detail="deterministic round fact",
                evidence_ids=evidence_ids,
                fact_ids=evidence_ids,
            ),
        ),
        evidence_ids=evidence_ids,
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _claim(*, expires_in: int = 60) -> BridgeClaim:
    claimed_at = T0
    return BridgeClaim(
        call_id=new_tool_call_id(),
        run_id=new_run_id(),
        session_id=new_session_id(),
        workspace_id="ws_project",
        snapshot_id="snap_" + "a" * 32,
        owner_id="prn_operator",
        client_id=new_client_id(),
        idempotency_key="claim-key",
        fingerprint="b" * 64,
        claimed_at=claimed_at,
        expires_at=claimed_at + timedelta(seconds=expires_in),
    )


def test_session_rejects_cross_principal_and_cross_workspace_grants() -> None:
    with pytest.raises(ValidationError):
        Session(
            session_id=new_session_id(),
            principal_id="prn_operator",
            workspace_id="ws_project",
            grant=WorkspaceGrant(principal_id="prn_other", workspace_id="ws_project"),
        )
    with pytest.raises(ValidationError):
        Session(
            session_id=new_session_id(),
            principal_id="prn_operator",
            workspace_id="ws_project",
            grant=WorkspaceGrant(principal_id="prn_operator", workspace_id="ws_other"),
        )


def test_cloud_default_stays_distinct_from_bridge_location() -> None:
    request = NativeRunRequest(input="hello")
    assert request.agent_options.execution_location is ExecutionLocation.CLOUD
    bridged = NativeRunRequest(
        input="hello",
        agent_options={"execution_location": ExecutionLocation.BRIDGE},
    )
    assert bridged.agent_options.execution_location is ExecutionLocation.BRIDGE
    assert bridged.agent_options.execution_location is not request.agent_options.execution_location


def test_bridge_public_payload_rejects_strategy_credentials_and_roots() -> None:
    payload = {
        "call_id": new_tool_call_id(),
        "work_unit_id": new_work_unit_id(),
        "tool_name": "read_file",
        "effect": ToolEffect.READ.value,
        "scope": {
            "session_id": new_session_id(),
            "run_id": new_run_id(),
            "workspace_id": new_workspace_id(),
            "client_id": new_client_id(),
        },
        "arguments": {"path": "src/app.py"},
        "requested_at": "2026-08-10T12:00:00Z",
        "location": "BRIDGE",
    }
    public = public_dispatch_payload(payload)
    assert "strategy" not in public
    assert "token" not in public
    dumped = str(public)
    assert "agent-secret" not in dumped
    with pytest.raises(BridgeProtocolError):
        public_dispatch_payload({**payload, "strategy": "PROGRESSIVE"})
    with pytest.raises(BridgeProtocolError):
        public_dispatch_payload({**payload, "token": "agent-secret"})
    with pytest.raises(BridgeProtocolError):
        public_dispatch_payload({**payload, "arguments": {"path": "/var/prp/src/app.py"}})


def test_public_events_redact_secrets_and_roots_but_keep_ids(tmp_path: Path) -> None:
    local_root = tmp_path / "workspace"
    local_root.mkdir()
    event = {
        "sequence": 4,
        "call_id": "tc_keep",
        "event_type": "TOOL_CALL_RUNNING",
        "status": "RUNNING",
        "result": {
            "api_key": "sk-secret",
            "path": str(local_root / "src" / "app.py"),
        },
    }
    public = _public_event(event, local_root=local_root)
    serialized = str(public)
    assert public["sequence"] == 4
    assert public["call_id"] == "tc_keep"
    assert "sk-secret" not in serialized
    assert str(local_root) not in serialized
    assert public["result"]["api_key"] == "<redacted>"
    with pytest.raises(BridgeProtocolError):
        _public_event({**event, "strategy": "PROGRESSIVE"}, local_root=local_root)
    with pytest.raises(BridgeProtocolError):
        _assert_remote_safe({"output": "/etc/shadow"})


def test_claim_lease_expiry_cannot_settle_or_mutate() -> None:
    claim = _claim(expires_in=30)
    inside = T0 + timedelta(seconds=10)
    after = T0 + timedelta(seconds=30)
    assert claim.is_active_at(inside) is True
    assert claim.is_active_at(after) is False
    with pytest.raises(ValueError, match="cannot expire before"):
        claim.expire(at=inside)
    expired = claim.expire(at=after)
    assert expired.status is BridgeClaimStatus.EXPIRED
    with pytest.raises(ValueError, match="only an active"):
        expired.settle(at=after)
    with pytest.raises(ValueError, match="lease window"):
        claim.settle(at=after)


def test_event_cursor_chain_is_strict_and_traceable() -> None:
    run_id = new_run_id()
    events = (
        RunEvent(
            run_id=run_id,
            sequence=1,
            event_type=EventType.RUN_CREATED,
            timestamp=T0,
            payload={"request": {"input": "hello"}},
        ),
        RunEvent(
            run_id=run_id,
            sequence=2,
            event_type=EventType.RUN_STARTED,
            timestamp=T0,
            payload={},
        ),
    )
    assert_sequence_chain(events)
    with pytest.raises(StateError):
        assert_sequence_chain(
            (
                events[0],
                events[1].model_copy(update={"sequence": 4}),
            )
        )
    with pytest.raises(ValidationError, match="reasoning"):
        RunEvent(
            run_id=run_id,
            sequence=1,
            event_type=EventType.RUN_STARTED,
            timestamp=T0,
            payload={"reasoning": "private thought"},
        )


def test_reuse_and_comparison_require_verified_facts_not_self_assertion() -> None:
    matching = decide_reuse(
        _work_unit(),
        _work_unit(),
        historical_dependency_artifact_hashes=(),
        candidate_dependency_artifact_hashes=(),
        historical_base_snapshot_id="snap_base",
        candidate_base_snapshot_id="snap_base",
        historical_merged_snapshot_id="snap_merged",
        candidate_merged_snapshot_id="snap_merged",
        historical_merge_input_digest="digest",
        candidate_merge_input_digest="digest",
        historical_change_set_ids=("cs_same",),
        candidate_change_set_ids=("cs_same",),
        historical_evidence_ids=("ev_same",),
        candidate_evidence_ids=("ev_same",),
        historical_attempts_proven=True,
    )
    assert matching.disposition is ReuseDisposition.REUSE
    changed = decide_reuse(
        _work_unit(),
        _work_unit(lineage="other"),
        historical_dependency_artifact_hashes=(),
        candidate_dependency_artifact_hashes=(),
    )
    assert changed.disposition is ReuseDisposition.RECOMPUTE
    assert changed.reason is ReuseReason.LINEAGE_CHANGED
    passed = _global_report(VerificationResult.PASS, evidence_ids=("ev_" + "1" * 32,))
    failed = _global_report(VerificationResult.FAIL, evidence_ids=("ev_" + "2" * 32,))
    improved = compare_rounds(failed, passed)
    assert improved.outcome is ComparisonOutcome.IMPROVED
    assert improved.accepted is True
    assert compare_rounds(passed, passed).accepted is False
    assert compare_rounds(passed, failed).accepted is False


def test_conflicting_writes_are_explicit_and_disjoint_writes_are_compatible() -> None:
    left = ConflictFacts(write_paths=("src/main.py",), base_snapshot_id="snap_" + "a" * 32)
    overlap = ConflictFacts(write_paths=("src/main.py",), base_snapshot_id="snap_" + "a" * 32)
    other = ConflictFacts(write_paths=("src/other.py",), base_snapshot_id="snap_" + "a" * 32)
    conflict = classify_facts(left, overlap)
    assert conflict.kind is ConflictKind.PATH
    assert conflict.conflict is True
    assert classify_facts(left, other).kind is ConflictKind.NO_CONFLICT


def test_export_path_exclusion_keeps_secrets_and_git_out_of_cloud_bundle() -> None:
    assert _export_path_is_excluded(".env") is True
    assert _export_path_is_excluded(".git/config") is True
    assert _export_path_is_excluded("opencode.json") is True
    assert _export_path_is_excluded("src/result.txt") is False


def test_declared_protocol_subset_rejects_stream_tools_and_unknown_fields() -> None:
    reject_unsupported_fields(
        {"model": "gpt", "messages": []},
        allowed=frozenset({"model", "messages"}),
    )
    with pytest.raises(PrpError) as stream_error:
        reject_unsupported_fields(
            {"model": "gpt", "stream": True},
            allowed=frozenset({"model"}),
        )
    assert stream_error.value.detail.code is ErrorCode.UNSUPPORTED_STREAM_MODE
    with pytest.raises(PrpError) as tools_error:
        reject_unsupported_fields(
            {"model": "gpt", "tools": []},
            allowed=frozenset({"model"}),
        )
    assert tools_error.value.detail.code is ErrorCode.UNSUPPORTED_TOOLS
    with pytest.raises(PrpError) as unknown_error:
        reject_unsupported_fields(
            {"model": "gpt", "temperature_top": 1},
            allowed=frozenset({"model"}),
        )
    assert unknown_error.value.detail.code is ErrorCode.UNSUPPORTED_FIELD


def test_run_status_and_usage_remain_the_only_success_facts() -> None:
    assert RunStatus.SUCCEEDED.is_terminal is True
    assert RunStatus.FAILED.is_terminal is True
    assert RunStatus.RUNNING.is_terminal is False
    usage = Usage(input_tokens=2, output_tokens=3)
    assert usage.total_tokens == 5
    assert "reasoning" not in usage.model_dump()


def test_fake_enterprise_two_client_server_brain_loop(tmp_path: Path) -> None:
    """One real BRIDGE+PROGRESSIVE run through production APIs, fake adapters only."""
    import json
    import time

    from fastapi.testclient import TestClient
    from pydantic import SecretStr

    from prp_runtime.app import create_app
    from prp_runtime.client.executor import BridgeExecutor
    from prp_runtime.domain.enums import (
        ExecutionStrategy,
        ModelRole,
        RoutingPolicy,
        ToolCallStatus,
    )
    from prp_runtime.domain.events import EventType
    from prp_runtime.domain.models import AgentToolCall, Usage
    from prp_runtime.planning.models import PlanProposal
    from prp_runtime.providers.base import FinishReason, ModelProfile, ProviderResponse
    from prp_runtime.settings import Settings
    from prp_runtime.storage.sqlite import SqliteStore
    from prp_runtime.tools import ToolRegistry, build_filesystem_registry
    from prp_runtime.tools.patch import LocalPatchStore, PatchRunner, build_patch_definition
    from prp_runtime.workspace import WorkspaceBackend
    from prp_runtime.workspace.models import Snapshot, SnapshotStatus
    from tests.integration.test_agent_api import (
        _handshake_body,
        _registration_body,
        auth_headers,
        seed_workspace,
        wait_for_terminal,
    )

    workspace_root = tmp_path / "enterprise-workspace"
    workspace_root.mkdir()
    (workspace_root / "app.py").write_text(
        "def answer() -> int:\n    return 1\n", encoding="utf-8"
    )
    (workspace_root / "README.md").write_text("enterprise\n", encoding="utf-8")
    database_path = tmp_path / "agent.db"
    seed_workspace(database_path, "prn_operator", workspace_root)

    async def original_snapshot() -> str:
        async with SqliteStore(database_path) as store:
            snapshots = await store.list_snapshots("ws_project", owner_id="prn_operator")
            assert snapshots
            return snapshots[0].snapshot_id

    snapshot_id = __import__("asyncio").run(original_snapshot())
    patch_diff = (
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def answer() -> int:\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    result_schema = json.dumps(
        {
            "type": "object",
            "properties": {"ok": {"const": True}},
            "required": ["ok"],
            "additionalProperties": False,
        }
    )
    proposal = PlanProposal(
        summary="patch the client workspace",
        final_node="answer",
        nodes=(
            {
                "key": "answer",
                "name": "Answer",
                "instruction": "patch app.py then return ok",
                "output": {"kind": "JSON", "json_schema": result_schema},
            },
        ),
    )

    def _profile(alias: str, role: ModelRole) -> ModelProfile:
        return ModelProfile(
            alias=alias,
            provider="fake",
            model=f"{alias}-model",
            role=role,
            base_url="https://models.invalid/v1",
            supports_structured_output=True,
            context_window_tokens=8_000,
            max_output_tokens=1_000,
        )

    class Planner:
        name = "enterprise-planner"

        def __init__(self) -> None:
            self.requests: list[object] = []

        async def complete(self, request: object) -> ProviderResponse:
            self.requests.append(request)
            return ProviderResponse(
                text=proposal.model_dump_json(),
                usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
                finish_reason=FinishReason.STOP,
            )

        async def aclose(self) -> None:
            return None

    class Worker:
        name = "enterprise-worker"

        def __init__(self) -> None:
            self.requests: list[object] = []
            self._tool_emitted = False

        async def complete(self, request: object) -> ProviderResponse:
            self.requests.append(request)
            if self._tool_emitted:
                return ProviderResponse(
                    text='{"ok": true}',
                    usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
                    finish_reason=FinishReason.STOP,
                )
            self._tool_emitted = True
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id=new_tool_call_id(),
                        tool_name="apply_patch",
                        arguments={
                            "patch": {
                                "base_snapshot_id": snapshot_id,
                                "unified_diff": patch_diff,
                            }
                        },
                    ),
                ),
                usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
                finish_reason=FinishReason.TOOL_CALLS,
            )

        async def aclose(self) -> None:
            return None

    planner = Planner()
    worker = Worker()
    settings = Settings(
        database_path=database_path,
        leader_profile=_profile("planner", ModelRole.PLANNER),
        worker_profile=_profile("worker", ModelRole.WORKER),
        service_token=SecretStr("agent-secret"),
        service_principal="prn_operator",
        workspace_roots={"project-main": str(workspace_root)},
    )
    app = create_app(settings, adapters={"planner": planner, "worker": worker})
    live = _handshake_body(
        "apply_patch",
        "list_files",
        "read_file",
        effects=("READ", "WRITE"),
    )
    other = _handshake_body("read_file")
    headers = auth_headers()

    def wait_for_running_call(client: TestClient, session_id: str, run_id: str) -> dict[str, object]:
        for _ in range(200):
            run = client.get(
                f"/v1/sessions/{session_id}/runs/{run_id}", headers=headers
            ).json()
            if run["status"] in {status.value for status in RunStatus if status.is_terminal}:
                raise AssertionError(f"run finished before Bridge claim: {run}")
            calls = client.get(
                f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls",
                headers=headers,
            ).json()
            running = [item for item in calls if item["status"] == ToolCallStatus.RUNNING.value]
            if running:
                return running[0]
            time.sleep(0.05)
        raise AssertionError("Bridge tool call did not become RUNNING")

    with TestClient(app) as client:
        assert (
            client.post("/v1/bridge/clients", json=_registration_body(live), headers=headers).status_code
            == 201
        )
        assert client.post("/v1/bridge/handshake", json=live, headers=headers).status_code == 200
        beat = client.post(
            f"/v1/bridge/clients/{live['client_id']}/heartbeat",
            json={"fingerprint": live["fingerprint"]},
            headers=headers,
        )
        assert beat.status_code == 200
        assert beat.json()["liveness"] == "LIVE"
        session = client.post(
            "/v1/sessions",
            headers=headers,
            json={
                "workspace_id": "ws_project",
                "access": ["READ", "WRITE"],
                "agent_options": {
                    "execution_location": "BRIDGE",
                    "agent_mode": "YOLO",
                },
            },
        )
        assert session.status_code == 201, session.text
        session_id = session.json()["session_id"]
        assert session.json()["agent_options"]["execution_location"] == "BRIDGE"
        created = client.post(
            f"/v1/sessions/{session_id}/runs",
            headers=headers,
            json={
                "input": "patch app.py on the assigned Bridge client",
                "routing_policy": RoutingPolicy.MANUAL.value,
                "strategy": ExecutionStrategy.PROGRESSIVE.value,
                "budget": {"max_plan_revisions": 1, "max_attempts": 4},
            },
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]
        call = wait_for_running_call(client, session_id, run_id)
        call_id = call["call_id"]
        assert call["tool_name"] == "apply_patch"
        assert call["snapshot_id"] == snapshot_id
        assert (
            client.post("/v1/bridge/clients", json=_registration_body(other), headers=headers).status_code
            == 201
        )
        assert client.post("/v1/bridge/handshake", json=other, headers=headers).status_code == 200
        unauthorized = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim",
            json={"call_id": call_id, "client_id": live["client_id"]},
        )
        assert unauthorized.status_code == 401
        crossed = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim",
            json={"call_id": call_id, "client_id": other["client_id"]},
            headers={**headers, "Idempotency-Key": "claim-other"},
        )
        assert crossed.status_code in {403, 409}
        stale = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim",
            json={
                "call_id": call_id,
                "client_id": live["client_id"],
                "snapshot_id": "snap_" + "b" * 32,
            },
            headers={**headers, "Idempotency-Key": "claim-stale"},
        )
        assert stale.status_code == 409
        claimed = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim",
            json={
                "call_id": call_id,
                "client_id": live["client_id"],
                "snapshot_id": snapshot_id,
            },
            headers={**headers, "Idempotency-Key": "claim-live"},
        )
        assert claimed.status_code == 201, claimed.text
        assert claimed.json()["client_id"] == live["client_id"]
        replay_claim = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim",
            json={
                "call_id": call_id,
                "client_id": live["client_id"],
                "snapshot_id": snapshot_id,
            },
            headers={**headers, "Idempotency-Key": "claim-live-2"},
        )
        assert replay_claim.status_code == 201, replay_claim.text
        assert replay_claim.json()["claim_id"] == claimed.json()["claim_id"]
        beat_claimed = client.post(
            f"/v1/bridge/clients/{live['client_id']}/heartbeat",
            json={"fingerprint": live["fingerprint"]},
            headers=headers,
        )
        assert beat_claimed.status_code == 200, beat_claimed.text
        view = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}",
            headers=headers,
        ).json()
        before = (workspace_root / "app.py").read_text(encoding="utf-8")

        async def run_local_patch() -> dict[str, object]:
            with WorkspaceBackend(workspace_root) as backend:
                manifest = backend.snapshot_manifest()
                snapshot = Snapshot(
                    snapshot_id=view["snapshot_id"],
                    workspace_id="ws_project",
                    status=SnapshotStatus.READY,
                    created_at=utc_now(),
                    completed_at=utc_now(),
                    file_count=len(manifest.entries),
                    total_size=manifest.total_size,
                )
                local_store = LocalPatchStore()
                runner = PatchRunner(
                    backend,
                    local_store,
                    owner_id="prn_operator",
                    base_snapshot=snapshot,
                    base_manifest=manifest,
                )
                registry = ToolRegistry(
                    (
                        *build_filesystem_registry(backend).definitions,
                        build_patch_definition(runner),
                    )
                )
                executor = BridgeExecutor(registry, workspace_root)
                dispatch = {
                    "call_id": call_id,
                    "run_id": run_id,
                    "work_unit_id": view["work_unit_id"],
                    "session_id": session_id,
                    "workspace_id": "ws_project",
                    "client_id": live["client_id"],
                    "tool_name": view["tool_name"],
                    "effect": view["effect"],
                    "arguments": view["arguments"],
                    "scope": {"paths": ["**"]},
                    "snapshot_id": view["snapshot_id"],
                    "requested_at": view["requested_at"],
                    "claimed_at": claimed.json()["claimed_at"],
                    "expires_at": claimed.json()["expires_at"],
                    "status": claimed.json()["status"],
                }
                return await executor.execute(dispatch)

        payload = __import__("asyncio").run(run_local_patch())
        assert (workspace_root / "app.py").read_text(encoding="utf-8") != before
        mutated = (workspace_root / "app.py").read_text(encoding="utf-8")
        assert "return 2" in mutated
        result_facts = payload.get("result")
        assert isinstance(result_facts, dict)
        assert isinstance(result_facts.get("patch"), dict)
        assert isinstance(result_facts.get("files"), list)
        assert result_facts["files"]
        payload["client_id"] = live["client_id"]
        payload["snapshot_id"] = snapshot_id
        payload["dev_only"] = True
        first = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/result",
            json=payload,
            headers={**headers, "Idempotency-Key": "result-live"},
        )
        assert first.status_code == 200, first.text
        async def _change_sets() -> tuple[str, ...]:
            items = await app.state.store.list_change_sets(run_id=run_id)
            return tuple(item.change_set_id for item in items)

        assert __import__("asyncio").run(_change_sets())
        replay = client.post(
            f"/v1/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/result",
            json=payload,
            headers={**headers, "Idempotency-Key": "result-live-2"},
        )
        assert replay.status_code == 200, replay.text
        assert (workspace_root / "app.py").read_text(encoding="utf-8") == mutated
        finished = wait_for_terminal(client, session_id, run_id)
        assert finished["status"] == RunStatus.SUCCEEDED.value, finished
        assert finished["strategy"] == ExecutionStrategy.PROGRESSIVE.value
        events = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/events",
            headers=headers,
        )
        assert events.status_code == 200
        event_names = [
            line.split(" ", 1)[1]
            for line in events.text.splitlines()
            if line.startswith("event: ")
        ]
        assert EventType.BRIDGE_CLAIM_CREATED.value in events.text
        assert EventType.BRIDGE_CLAIM_SETTLED.value in events.text
        assert EventType.ARTIFACT_PRODUCED.value in events.text
        assert EventType.EVIDENCE_RECORDED.value in events.text
        assert EventType.MERGE_MERGED.value in events.text, ",".join(event_names)
        assert EventType.MERGE_PROMOTED.value in events.text
        assert EventType.CONTROLLER_DECISION.value in events.text
        assert "CHANGE_SET" in events.text
        assert "\"AST\"" in events.text or "AST" in events.text
        assert "agent-secret" not in events.text
        assert str(workspace_root) not in events.text
        assert "https://models.invalid" not in events.text
        first_event_id = None
        for line in events.text.splitlines():
            if line.startswith("id: "):
                first_event_id = int(line.removeprefix("id: ").strip())
                break
        assert first_event_id is not None
        replay_events = client.get(
            f"/v1/sessions/{session_id}/runs/{run_id}/events",
            params={"after": first_event_id},
            headers=headers,
        )
        assert replay_events.status_code == 200
        assert EventType.RUN_SUCCEEDED.value in replay_events.text
        assert f"id: {first_event_id}\n" not in replay_events.text
        assert client.get("/v1/bridge/bundle", headers=headers).status_code == 404

    assert len(planner.requests) == 1
    assert len(worker.requests) >= 2
    assert (workspace_root / "app.py").read_text(encoding="utf-8") == (
        "def answer() -> int:\n    return 2\n"
    )


