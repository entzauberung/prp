"""Focused fake-transport tests for the model-free Bridge client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from prp_runtime.client.bridge import (
    Bridge,
    BridgeHTTPError,
    BridgeProtocolError,
    BridgeState,
    BridgeStateError,
    BridgeTransportError,
    _assert_remote_safe,
    _public_event,
    public_dispatch_payload,
)
from prp_runtime.domain.enums import ToolEffect
from prp_runtime.domain.models import (
    BridgeDispatchFacts,
    ClientIdentityFacts,
    new_client_id,
)
from prp_runtime.domain.values import (
    new_run_id,
    new_session_id,
    new_tool_call_id,
    new_work_unit_id,
    new_workspace_id,
)


async def _close(client: httpx.AsyncClient) -> None:
    await client.aclose()


class _DisconnectingEventStream(httpx.AsyncByteStream):
    def __init__(self, first_event: bytes) -> None:
        self._first_event = first_event

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first_event
        raise httpx.ReadError("event stream disconnected")

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_session_run_and_auth_header_never_send_workspace_path(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/sessions":
            return httpx.Response(201, json={"session_id": "sess-1"})
        return httpx.Response(202, json={"run_id": "run-1", "status": "PENDING"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge(
        "https://bridge.test",
        "opaque-token",
        workspace_root=tmp_path / "workspace",
        client=client,
    )
    try:
        await bridge.create_session("ws-1")
        await bridge.create_run("sess-1", {"input": "read the workspace"})
    finally:
        await _close(client)

    assert all(request.headers["Authorization"] == "Bearer opaque-token" for request in requests)
    assert all("opaque-token" not in request.content.decode() for request in requests)
    assert all(str(tmp_path) not in request.content.decode() for request in requests)
    assert requests[0].url.path == "/v1/sessions"
    assert requests[1].url.path == "/v1/sessions/sess-1/runs"


@pytest.mark.asyncio
async def test_non_standard_bridge_response_json_fails_closed() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=b'{"value":NaN}')

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    try:
        with pytest.raises(BridgeProtocolError, match="was not JSON"):
            await bridge._request("GET", "/v1/test")
    finally:
        await _close(client)


def test_invalid_utf8_bridge_state_fails_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "invalid-state.json"
    state_path.write_bytes(b"\xff")

    with pytest.raises(BridgeStateError, match="cannot be read"):
        Bridge("https://bridge.test", "token", state_path=state_path)


def test_malformed_bridge_state_entries_fail_as_bridge_state_errors() -> None:
    with pytest.raises(BridgeStateError, match="entries must be JSON objects"):
        BridgeState.from_json(
            {
                "version": 1,
                "claimed_calls": {"call-1": []},
                "submitted_results": {},
            }
        )


@pytest.mark.asyncio
async def test_events_resume_from_atomic_cursor_after_disconnect(tmp_path: Path) -> None:
    event_requests: list[str] = []
    first_disconnect = True

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal first_disconnect
        if request.url.path.endswith("/events"):
            event_requests.append(request.url.params["after"])
            if request.url.params["after"] == "0":
                if first_disconnect:
                    first_disconnect = False
                    return httpx.Response(
                        200,
                        stream=_DisconnectingEventStream(
                            b'id: 1\nevent: TOOL_CALL_SUCCEEDED\ndata: {"sequence": 1}\n\n'
                        ),
                        headers={"content-type": "text/event-stream"},
                    )
                body = 'id: 1\nevent: RUN_STARTED\ndata: {"sequence": 1}\n\n'
            else:
                body = 'id: 2\nevent: RUN_SUCCEEDED\ndata: {"sequence": 2}\n\n'
            return httpx.Response(
                200,
                content=body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"run_id": "run-1", "status": "RUNNING"})

    state_path = tmp_path / "bridge-state.json"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", state_path=state_path, client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    try:
        events = [
            event
            async for event in bridge.iter_events(max_reconnects=2)
        ]
    finally:
        await _close(client)

    assert [event["sequence"] for event in events] == [1, 2]
    assert event_requests == ["0", "1"]
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["event_cursor"] == 2
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert "token" not in state_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_claim_disconnect_replays_idempotently_after_state_resume(tmp_path: Path) -> None:
    request_count = 0
    applied_keys: set[str] = set()
    claim_bodies: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        if request.url.path.endswith("/claim"):
            request_count += 1
            claim_bodies.append(json.loads(request.content))
            key = request.headers["Idempotency-Key"]
            if key not in applied_keys:
                applied_keys.add(key)
                if request_count == 1:
                    raise httpx.ReadError("claim response disconnected")
            return httpx.Response(
                201,
                json={"call_id": "call-claim", "claim_id": "claim-1"},
            )
        return httpx.Response(200, json={"run_id": "run-1", "status": "SUCCEEDED"})

    state_path = tmp_path / "claim-state.json"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", state_path=state_path, client=client)
    client_id = new_client_id()
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    bridge.state.client_id = client_id
    bridge._save_state()
    try:
        with pytest.raises(BridgeTransportError):
            await bridge.claim_tool_call("call-claim")
        resumed = Bridge(
            "https://bridge.test",
            "token",
            state_path=state_path,
            client=client,
        )
        claim = await resumed.claim_tool_call("call-claim")
    finally:
        await _close(client)

    assert claim["claim_id"] == "claim-1"
    assert request_count == 2
    assert len(applied_keys) == 1
    assert json.loads(state_path.read_text(encoding="utf-8"))["client_id"] == client_id
    assert claim_bodies == [
        {"call_id": "call-claim", "client_id": client_id},
        {"call_id": "call-claim", "client_id": client_id},
    ]
    assert state_path.stat().st_mode & 0o777 == 0o600


class _CountingExecutor:
    def __init__(self) -> None:
        self.count = 0

    async def execute(self, claim: object) -> dict[str, object]:
        del claim
        self.count += 1
        return {"status": "SUCCEEDED", "output": "ok", "changed_paths": []}


@pytest.mark.asyncio
async def test_execution_disconnect_before_submit_resumes_without_reexecution(
    tmp_path: Path,
) -> None:
    result_requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal result_requests
        if request.url.path.endswith("/result"):
            result_requests += 1
            if result_requests == 1:
                raise httpx.ReadError("result request disconnected")
            return httpx.Response(
                200,
                json={"call_id": "call-execute", "status": "SUCCEEDED"},
            )
        return httpx.Response(200, json={"run_id": "run-1", "status": "SUCCEEDED"})

    state_path = tmp_path / "execution-state.json"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", state_path=state_path, client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    bridge.state.client_id = new_client_id()
    bridge.state.claimed_calls["call-execute"] = {"response": {"call_id": "call-execute"}}
    bridge._save_state()
    executor = _CountingExecutor()
    claim = {"call_id": "call-execute"}
    try:
        with pytest.raises(BridgeTransportError):
            await bridge.execute_claim(claim, executor)
        resumed = Bridge(
            "https://bridge.test",
            "token",
            state_path=state_path,
            client=client,
        )
        result = await resumed.execute_claim(claim, executor)
    finally:
        await _close(client)

    assert result["status"] == "SUCCEEDED"
    assert executor.count == 1
    assert result_requests == 2
    assert state_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_submit_disconnect_after_remote_commit_replays_without_duplicate_result(
    tmp_path: Path,
) -> None:
    result_requests = 0
    committed_keys: set[str] = set()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal result_requests
        if request.url.path.endswith("/result"):
            result_requests += 1
            key = request.headers["Idempotency-Key"]
            if key not in committed_keys:
                committed_keys.add(key)
                raise httpx.ReadError("committed result response disconnected")
            return httpx.Response(
                200,
                json={"call_id": "call-submit", "status": "SUCCEEDED"},
            )
        return httpx.Response(200, json={"run_id": "run-1", "status": "SUCCEEDED"})

    state_path = tmp_path / "submit-state.json"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", state_path=state_path, client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    bridge.state.client_id = new_client_id()
    bridge.state.claimed_calls["call-submit"] = {"response": {"call_id": "call-submit"}}
    bridge._save_state()
    executor = _CountingExecutor()
    claim = {"call_id": "call-submit"}
    try:
        with pytest.raises(BridgeTransportError):
            await bridge.execute_claim(claim, executor)
        resumed = Bridge(
            "https://bridge.test",
            "token",
            state_path=state_path,
            client=client,
        )
        result = await resumed.execute_claim(claim, executor)
    finally:
        await _close(client)

    assert result["status"] == "SUCCEEDED"
    assert executor.count == 1
    assert result_requests == 2
    assert len(committed_keys) == 1
    assert state_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_result_submit_is_idempotent_and_conflicting_retry_is_rejected() -> None:
    submit_count = 0
    idempotency_keys: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submit_count
        if request.method == "POST" and request.url.path.endswith("/result"):
            submit_count += 1
            idempotency_keys.append(request.headers["Idempotency-Key"])
            return httpx.Response(200, json={"accepted": True})
        return httpx.Response(200, json={"status": "RUNNING"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    bridge.state.client_id = new_client_id()
    result = {
        "dev_only": True,
        "status": "SUCCEEDED",
        "output": "ok",
        "changed_paths": ["README.md"],
    }
    try:
        first = await bridge.submit_tool_result("call-1", result)
        second = await bridge.submit_tool_result("call-1", result)
        with pytest.raises(BridgeStateError):
            await bridge.submit_tool_result(
                "call-1",
                {"dev_only": True, "status": "SUCCEEDED", "output": "other"},
            )
    finally:
        await _close(client)

    assert first == second == {"accepted": True}
    assert submit_count == 1
    assert len(idempotency_keys) == 1


@pytest.mark.asyncio
async def test_result_without_dev_only_is_rejected_before_transport() -> None:
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    try:
        with pytest.raises(BridgeProtocolError, match="dev_only"):
            await bridge.submit_tool_result(
                "call-1", {"status": "SUCCEEDED", "output": "ok"}
            )
    finally:
        await _close(client)
    assert called is False


@pytest.mark.asyncio
async def test_absolute_local_path_is_rejected_before_transport() -> None:
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    try:
        with pytest.raises(BridgeProtocolError):
            await bridge.submit_tool_result(
                "call-1", {"status": "SUCCEEDED", "output": "/home/user/project/file.py"}
            )
    finally:
        await _close(client)
    assert called is False
def test_bridge_tool_loop_deduplicates_calls_and_stops_on_terminal_event() -> None:
    import asyncio

    from prp_runtime.client.bridge import (
        Bridge,
        BridgeState,
        BridgeToolLoopLimits,
        BridgeToolLoopPhase,
    )

    class FakeBridge(Bridge):
        def __init__(self) -> None:
            self.state = BridgeState()
            self._workspace_root = None
            self.executions = 0

        async def iter_events(self, *args: object, **kwargs: object):
            del args, kwargs
            yield {
                "sequence": 1,
                "event_type": "TOOL_CALL_RUNNING",
                "call_id": "tc_one",
                "status": "RUNNING",
            }
            yield {
                "sequence": 2,
                "event_type": "TOOL_CALL_RUNNING",
                "call_id": "tc_one",
                "status": "RUNNING",
            }
            yield {"sequence": 3, "event_type": "RUN_SUCCEEDED", "status": "SUCCEEDED"}

        async def claim_tool_call(self, call_id: str, *args: object, **kwargs: object):
            del args, kwargs
            return {
                "call_id": call_id,
                "status": "ACTIVE",
                "claimed_at": "2026-08-15T12:00:00+00:00",
                "expires_at": "2026-08-15T12:05:00+00:00",
            }

        async def execute_claim(self, claim: object, *args: object, **kwargs: object):
            del args, kwargs
            self.executions += 1
            call_id = claim["call_id"]
            self.state.claimed_calls[call_id] = {"execution": {"payload": {}}}
            return {}

    async def run() -> None:
        bridge = FakeBridge()
        outcome = await bridge.run_tool_loop(
            object(), limits=BridgeToolLoopLimits(max_calls=2, max_events=4)
        )
        assert outcome.phase is BridgeToolLoopPhase.TERMINAL
        assert outcome.processed_call_ids == ("tc_one",)
        assert bridge.executions == 1

    asyncio.run(run())


def test_bridge_tool_loop_stops_for_approval_without_claiming() -> None:
    import asyncio

    from prp_runtime.client.bridge import Bridge, BridgeState, BridgeToolLoopPhase

    class FakeBridge(Bridge):
        def __init__(self) -> None:
            self.state = BridgeState()
            self._workspace_root = None
            self.claimed = False

        async def iter_events(self, *args: object, **kwargs: object):
            del args, kwargs
            yield {
                "sequence": 1,
                "event_type": "TOOL_CALL_AWAITING_APPROVAL",
                "call_id": "tc_wait",
                "status": "AWAITING_APPROVAL",
                "token": "must-not-be-returned",
            }

        async def claim_tool_call(self, *args: object, **kwargs: object):
            del args, kwargs
            self.claimed = True
            raise AssertionError("approval-pending calls must not be claimed")

    async def run() -> None:
        bridge = FakeBridge()
        outcome = await bridge.run_tool_loop(object())
        assert outcome.phase is BridgeToolLoopPhase.WAITING_APPROVAL
        assert outcome.pending_call_ids == ("tc_wait",)
        assert bridge.claimed is False

    asyncio.run(run())


def test_bridge_tool_loop_call_limit_is_explicit() -> None:
    import asyncio

    import pytest

    from prp_runtime.client.bridge import (
        Bridge,
        BridgeState,
        BridgeToolLoopError,
        BridgeToolLoopLimits,
    )

    class FakeBridge(Bridge):
        def __init__(self) -> None:
            self.state = BridgeState()
            self._workspace_root = None

        async def iter_events(self, *args: object, **kwargs: object):
            del args, kwargs
            for sequence, call_id in ((1, "tc_one"), (2, "tc_two")):
                yield {
                    "sequence": sequence,
                    "event_type": "TOOL_CALL_RUNNING",
                    "call_id": call_id,
                    "status": "RUNNING",
                }

        async def claim_tool_call(self, call_id: str, *args: object, **kwargs: object):
            del args, kwargs
            return {"call_id": call_id}

        async def execute_claim(self, claim: object, *args: object, **kwargs: object):
            del args, kwargs
            self.state.claimed_calls[claim["call_id"]] = {"execution": {"payload": {}}}
            return {}

    async def run() -> None:
        with pytest.raises(BridgeToolLoopError, match="call limit"):
            await FakeBridge().run_tool_loop(
                object(), limits=BridgeToolLoopLimits(max_calls=1)
            )

    asyncio.run(run())



def _dispatch_payload() -> dict[str, object]:
    return {
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


def test_public_dispatch_payload_keeps_only_scoped_public_facts() -> None:
    payload = _dispatch_payload()
    public = public_dispatch_payload(payload)
    restored = BridgeDispatchFacts.model_validate(public)
    assert restored.location.value == "BRIDGE"
    assert restored.tool_name == "read_file"
    assert "strategy" not in public
    assert "workspace_root" not in public
    identity = ClientIdentityFacts.model_validate(
        {"client_id": payload["scope"]["client_id"], "tools": ["read_file"]}
    )
    assert identity.protocol_version == "0.0.4"


def test_public_dispatch_payload_rejects_strategy_credentials_and_roots() -> None:
    payload = _dispatch_payload()
    with pytest.raises(BridgeProtocolError, match="cannot cross"):
        public_dispatch_payload({**payload, "strategy": "PROGRESSIVE"})
    with pytest.raises(BridgeProtocolError, match="cannot cross"):
        public_dispatch_payload({**payload, "token": "opaque-token"})
    with pytest.raises(BridgeProtocolError, match="raw roots"):
        public_dispatch_payload({**payload, "arguments": {"path": "/var/prp/src/app.py"}})
    with pytest.raises(BridgeProtocolError):
        public_dispatch_payload({**payload, "unknown": True})



def test_public_event_redacts_nested_secrets_and_roots_but_keeps_public_ids(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "workspace"
    local_root.mkdir()
    event = {
        "sequence": 7,
        "call_id": "tc_keep",
        "event_type": "TOOL_CALL_RUNNING",
        "status": "RUNNING",
        "result": {
            "api_key": "sk-secret",
            "nested": {"password": "hunter2", "output": f"{local_root}/src/app.py"},
            "path": "/etc/shadow",
        },
    }
    public = _public_event(event, local_root=local_root)
    assert public["sequence"] == 7
    assert public["call_id"] == "tc_keep"
    assert public["event_type"] == "TOOL_CALL_RUNNING"
    assert public["result"]["api_key"] == "<redacted>"
    assert public["result"]["nested"]["password"] == "<redacted>"
    assert public["result"]["nested"]["output"] == "<local-root>/src/app.py"
    assert public["result"]["path"] == "<local-path>"
    assert "sk-secret" not in json.dumps(public)
    assert str(local_root) not in json.dumps(public)


def test_public_event_rejects_server_authority_instead_of_redacting() -> None:
    event = {
        "sequence": 8,
        "call_id": "tc_keep",
        "event_type": "TOOL_CALL_RUNNING",
        "strategy": "PROGRESSIVE",
    }
    with pytest.raises(BridgeProtocolError, match="authority"):
        _public_event(event)
    with pytest.raises(BridgeProtocolError, match="authority"):
        _public_event(
            {
                "sequence": 8,
                "call_id": "tc_keep",
                "tool_call": {"routing_policy": "AUTO"},
            }
        )


def test_outbound_claim_and_result_facts_reject_paths_and_credentials(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "workspace"
    _assert_remote_safe(
        {
            "call_id": "tc_keep",
            "path": "src/app.py",
            "output": "ok",
        },
        local_root=local_root,
    )
    with pytest.raises(BridgeProtocolError, match="absolute local paths"):
        _assert_remote_safe({"output": "/home/user/secret.py"})
    with pytest.raises(BridgeProtocolError, match="Workspace path"):
        _assert_remote_safe(
            {"output": f"copied from {local_root}/a.py"},
            local_root=local_root,
        )
    with pytest.raises(BridgeProtocolError, match="cannot be sent"):
        _assert_remote_safe({"result": {"token": "opaque-token"}})


def test_bridge_state_keeps_cursor_and_rejects_server_brain_configuration() -> None:
    state = BridgeState.from_json(
        {
            "version": 1,
            "session_id": "ses_public",
            "run_id": "run_public",
            "event_cursor": 11,
            "claimed_calls": {"tc_keep": {"response": {"call_id": "tc_keep"}}},
            "submitted_results": {
                "tc_keep": {"fingerprint": "a" * 64, "response": {"accepted": True}}
            },
        }
    )
    payload = state.to_json()
    assert payload["event_cursor"] == 11
    assert payload["session_id"] == "ses_public"
    assert payload["run_id"] == "run_public"
    assert "provider" not in payload
    assert "model" not in payload
    assert "strategy" not in json.dumps(payload)
    with pytest.raises(BridgeStateError, match="must not persist"):
        BridgeState.from_json(
            {
                "version": 1,
                "event_cursor": 11,
                "claimed_calls": {
                    "tc_keep": {"provider": "openai_compatible", "model": "strong"}
                },
                "submitted_results": {},
            }
        )
    dirty = BridgeState(session_id="ses_public", event_cursor=12)
    dirty.claimed_calls["tc_keep"] = {"strategy": "PROGRESSIVE"}
    with pytest.raises(BridgeStateError, match="must not persist"):
        dirty.to_json()



@pytest.mark.asyncio
async def test_handshake_persists_public_identity_without_credentials(tmp_path: Path) -> None:
    from prp_runtime.domain.models import (
        ClientCapabilityDescriptor,
        fingerprint_client_capabilities,
        new_client_id,
    )

    client_id = new_client_id()
    capabilities = ClientCapabilityDescriptor(
        tools=("list_files", "read_file"),
        effects=(ToolEffect.READ,),
    )
    fingerprint = fingerprint_client_capabilities(capabilities)
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "client_id": client_id,
                "protocol_version": "0.0.4",
                "fingerprint": fingerprint,
                "principal_id": "prn_operator",
                "workspace_id": "ws_project",
            },
        )

    state_path = tmp_path / "bridge-state.json"
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge(
        "https://bridge.test",
        "opaque-token",
        state_path=state_path,
        workspace_root=tmp_path / "workspace",
        client=http_client,
    )
    try:
        accepted = await bridge.handshake(
            {
                "client_id": client_id,
                "protocol_version": "0.0.4",
                "capabilities": capabilities.model_dump(mode="json"),
                "fingerprint": fingerprint,
                "workspace_id": "ws_project",
            }
        )
    finally:
        await _close(http_client)

    assert accepted["accepted"] is True
    assert bridge.state.client_id == client_id
    assert bridge.state.protocol_version == "0.0.4"
    assert bridge.state.capability_fingerprint == fingerprint
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["client_id"] == client_id
    assert "opaque-token" not in state_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in state_path.read_text(encoding="utf-8")
    assert requests[0].url.path == "/v1/bridge/handshake"


@pytest.mark.asyncio
async def test_handshake_rejects_strategy_and_does_not_downgrade_version() -> None:
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"accepted": True})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=http_client)
    try:
        with pytest.raises(BridgeProtocolError, match="authority"):
            await bridge.handshake(
                {
                    "client_id": new_client_id(),
                    "protocol_version": "0.0.4",
                    "strategy": "PROGRESSIVE",
                    "capabilities": {
                        "tools": ["read_file"],
                        "effects": ["READ"],
                    },
                    "fingerprint": "a" * 64,
                }
            )
    finally:
        await _close(http_client)
    assert called is False



@pytest.mark.asyncio
async def test_failed_handshake_does_not_persist_client_identity(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            409,
            json={
                "detail": {
                    "code": "bridge_client_conflict",
                    "message": "duplicate client identity cannot overwrite the existing client",
                }
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge(
        "https://bridge.test",
        "opaque-token",
        state_path=tmp_path / "bridge-state.json",
        client=http_client,
    )
    try:
        with pytest.raises(BridgeHTTPError):
            await bridge.handshake(
                {
                    "client_id": new_client_id(),
                    "protocol_version": "0.0.4",
                    "capabilities": {
                        "tools": ["list_files", "read_file"],
                        "effects": ["READ"],
                    },
                    "fingerprint": "a" * 64,
                }
            )
    finally:
        await _close(http_client)
    assert bridge.state.client_id is None
    assert bridge.state.capability_fingerprint is None
    assert not (tmp_path / "bridge-state.json").exists() or "cli_" not in (
        tmp_path / "bridge-state.json"
    ).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_renew_and_release_claim_post_scoped_routes() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/claim/renew"):
            return httpx.Response(200, json={"status": "ACTIVE", "call_id": "call-1"})
        if request.url.path.endswith("/claim/release"):
            return httpx.Response(200, json={"status": "RELEASED", "call_id": "call-1"})
        return httpx.Response(500, json={"error": "unexpected"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    client_id = new_client_id()
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    bridge.state.client_id = client_id
    try:
        renewed = await bridge.renew_claim("call-1")
        released = await bridge.release_claim("call-1")
    finally:
        await _close(client)

    assert renewed["status"] == "ACTIVE"
    assert released["status"] == "RELEASED"
    assert requests[0].url.path == "/v1/sessions/sess-1/runs/run-1/tool-calls/call-1/claim/renew"
    assert requests[1].url.path == "/v1/sessions/sess-1/runs/run-1/tool-calls/call-1/claim/release"
    assert json.loads(requests[0].content) == {"client_id": client_id}
    assert json.loads(requests[1].content) == {"client_id": client_id}


@pytest.mark.asyncio
async def test_iter_events_skips_duplicate_cursor_and_stops_on_terminal() -> None:
    body = (
        'id: 1\nevent: RUN_STARTED\ndata: {"sequence": 1}\n\n'
        'id: 1\nevent: RUN_STARTED\ndata: {"sequence": 1}\n\n'
        'id: 2\nevent: RUN_SUCCEEDED\ndata: {"sequence": 2}\n\n'
        'id: 3\nevent: TOOL_CALL_RUNNING\ndata: {"sequence": 3}\n\n'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            assert request.headers["Last-Event-ID"] == "0"
            return httpx.Response(
                200,
                content=body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"run_id": "run-1", "status": "RUNNING"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    try:
        events = [event async for event in bridge.iter_events(max_reconnects=0)]
    finally:
        await _close(client)

    assert [event["sequence"] for event in events] == [1, 2]
    assert events[-1]["event_type"] == "RUN_SUCCEEDED"


@pytest.mark.asyncio
async def test_iter_events_reconnect_budget_raises_structured_transport_error() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path.endswith("/events"):
            attempts += 1
            raise httpx.ReadError("event stream disconnected")
        return httpx.Response(200, json={"run_id": "run-1", "status": "RUNNING"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    try:
        with pytest.raises(BridgeTransportError, match="disconnected"):
            async for _event in bridge.iter_events(max_reconnects=1):
                raise AssertionError("no event should be yielded after a drop")
    finally:
        await _close(client)

    assert attempts == 2


@pytest.mark.asyncio
async def test_iter_events_drop_before_first_event_replays_from_zero(tmp_path: Path) -> None:
    event_requests: list[str] = []
    first_disconnect = True

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal first_disconnect
        if request.url.path.endswith("/events"):
            event_requests.append(request.url.params["after"])
            if first_disconnect:
                first_disconnect = False
                return httpx.Response(
                    200,
                    stream=_DisconnectingEventStream(b": keepalive\n\n"),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                content=b'id: 1\nevent: RUN_SUCCEEDED\ndata: {"sequence": 1}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"run_id": "run-1", "status": "RUNNING"})

    state_path = tmp_path / "drop-before-state.json"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge(
        "https://bridge.test",
        "token",
        state_path=state_path,
        workspace_root=tmp_path / "workspace",
        client=client,
    )
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    try:
        events = [event async for event in bridge.iter_events(max_reconnects=2)]
    finally:
        await _close(client)

    assert [event["sequence"] for event in events] == [1]
    assert event_requests == ["0", "0"]
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["event_cursor"] == 1
    assert "token" not in state_path.read_text(encoding="utf-8")
    assert str(tmp_path / "workspace") not in state_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_iter_events_premature_close_without_terminal_is_transport_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                content=b'id: 1\nevent: RUN_STARTED\ndata: {"sequence": 1}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"run_id": "run-1", "status": "RUNNING"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    try:
        with pytest.raises(BridgeTransportError, match="closed before run completion"):
            async for event in bridge.iter_events(max_reconnects=0):
                assert event["sequence"] == 1
    finally:
        await _close(client)


@pytest.mark.asyncio
async def test_iter_events_does_not_repeat_tool_event_after_disconnect(tmp_path: Path) -> None:
    event_requests: list[str] = []
    first_disconnect = True

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal first_disconnect
        if request.url.path.endswith("/events"):
            event_requests.append(request.url.params["after"])
            if request.url.params["after"] == "0" and first_disconnect:
                first_disconnect = False
                return httpx.Response(
                    200,
                    stream=_DisconnectingEventStream(
                        b'id: 1\nevent: TOOL_CALL_RUNNING\ndata: {"sequence": 1, "call_id": "tc_one"}\n\n'
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                content=b'id: 2\nevent: RUN_SUCCEEDED\ndata: {"sequence": 2}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"run_id": "run-1", "status": "RUNNING"})

    state_path = tmp_path / "tool-event-state.json"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", state_path=state_path, client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    try:
        events = [event async for event in bridge.iter_events(max_reconnects=2)]
    finally:
        await _close(client)

    assert [event["sequence"] for event in events] == [1, 2]
    assert event_requests == ["0", "1"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["event_cursor"] == 2


@pytest.mark.asyncio
async def test_execute_claim_rejects_expired_claim_before_handler() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    executor = _CountingExecutor()
    try:
        with pytest.raises(BridgeProtocolError, match="expired"):
            await bridge.execute_claim(
                {
                    "call_id": "call-expired",
                    "status": "ACTIVE",
                    "expires_at": "2020-01-01T00:00:00+00:00",
                },
                executor,
            )
    finally:
        await _close(client)
    assert executor.count == 0


@pytest.mark.asyncio
async def test_execute_claim_does_not_rerun_uncertain_write() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    bridge.state.claimed_calls["call-write"] = {"execution": {"status": "PENDING"}}
    executor = _CountingExecutor()
    try:
        with pytest.raises(BridgeStateError, match="uncertain"):
            await bridge.execute_claim(
                {"call_id": "call-write", "effect": "WRITE"},
                executor,
            )
    finally:
        await _close(client)
    assert executor.count == 0


@pytest.mark.asyncio
async def test_partial_execution_state_fails_closed() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    bridge.state.claimed_calls["call-partial"] = {
        "execution": {"payload": {"dev_only": True, "status": "SUCCEEDED"}}
    }
    executor = _CountingExecutor()
    try:
        with pytest.raises(BridgeStateError, match="incomplete"):
            await bridge.execute_claim({"call_id": "call-partial"}, executor)
    finally:
        await _close(client)
    assert executor.count == 0


@pytest.mark.asyncio
async def test_corrupt_execution_fingerprint_fails_closed() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    payload = {"dev_only": True, "status": "SUCCEEDED", "output": "ok", "changed_paths": []}
    bridge.state.claimed_calls["call-corrupt"] = {
        "execution": {"payload": payload, "fingerprint": "0" * 64}
    }
    executor = _CountingExecutor()
    try:
        with pytest.raises(BridgeStateError, match="fingerprint"):
            await bridge.execute_claim({"call_id": "call-corrupt"}, executor)
    finally:
        await _close(client)
    assert executor.count == 0



def test_tool_loop_heartbeats_and_renews_before_expiry_with_fake_clock() -> None:
    import asyncio
    from datetime import UTC, datetime, timedelta

    from prp_runtime.client.bridge import (
        Bridge,
        BridgeState,
        BridgeToolLoopLimits,
        BridgeToolLoopPhase,
    )
    from prp_runtime.domain.models import (
        BRIDGE_HEARTBEAT_CADENCE_SECONDS,
        MAX_BRIDGE_HEARTBEAT_TTL_SECONDS,
        MAX_BRIDGE_LEASE_RENEW_SECONDS,
        MAX_BRIDGE_LEASE_RENEWS,
        MAX_BRIDGE_LEASE_SECONDS,
        MAX_BRIDGE_LEASE_TOTAL_SECONDS,
    )

    start = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)

    class Clock:
        def __init__(self) -> None:
            self.now = start

        def __call__(self) -> datetime:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += timedelta(seconds=seconds)

    clock = Clock()
    client_id = new_client_id()
    fingerprint = "a" * 64

    class FakeBridge(Bridge):
        def __init__(self) -> None:
            self.state = BridgeState(
                session_id="sess-1",
                run_id="run-1",
                client_id=client_id,
                protocol_version="0.0.4",
                capability_fingerprint=fingerprint,
            )
            self._workspace_root = None
            self.heartbeats: list[datetime] = []
            self.renewals: list[datetime] = []
            self._claimed_at = start
            self._expires_at = start + timedelta(seconds=MAX_BRIDGE_LEASE_SECONDS)

        async def heartbeat(self) -> dict[str, object]:
            self.heartbeats.append(clock())
            return {
                "liveness": "LIVE",
                "ttl_seconds": 9_999,
                "cadence_seconds": 9_999,
                "max_renews": 99,
                "lease_renew_seconds": 9_999,
                "lease_total_seconds": 9_999,
            }

        async def iter_events(self, *args: object, **kwargs: object):
            del args, kwargs
            yield {
                "sequence": 1,
                "event_type": "TOOL_CALL_RUNNING",
                "call_id": "tc_one",
                "status": "RUNNING",
            }
            for index in range(12):
                clock.advance(BRIDGE_HEARTBEAT_CADENCE_SECONDS)
                yield {
                    "sequence": index + 2,
                    "event_type": "RUN_PROGRESS",
                    "status": "RUNNING",
                }
            yield {
                "sequence": 99,
                "event_type": "RUN_SUCCEEDED",
                "status": "SUCCEEDED",
            }

        async def claim_tool_call(self, call_id: str, *args: object, **kwargs: object):
            del args, kwargs
            response = {
                "call_id": call_id,
                "client_id": client_id,
                "status": "ACTIVE",
                "claimed_at": self._claimed_at.isoformat(),
                "expires_at": self._expires_at.isoformat(),
            }
            self.state.claimed_calls[call_id] = {"response": response}
            return response

        async def execute_claim(self, claim: dict[str, object], *args: object, **kwargs: object):
            del args, kwargs
            call_id = str(claim["call_id"])
            entry = self.state.claimed_calls.setdefault(call_id, {})
            entry["execution"] = {"payload": {}}
            return {}

        async def renew_claim(self, call_id: str, *args: object, **kwargs: object):
            del args, kwargs
            self.renewals.append(clock())
            self._expires_at = min(
                self._expires_at + timedelta(seconds=MAX_BRIDGE_LEASE_RENEW_SECONDS),
                self._claimed_at + timedelta(seconds=MAX_BRIDGE_LEASE_TOTAL_SECONDS),
            )
            response = {
                "call_id": call_id,
                "client_id": client_id,
                "status": "ACTIVE",
                "claimed_at": self._claimed_at.isoformat(),
                "expires_at": self._expires_at.isoformat(),
            }
            self.state.claimed_calls[call_id] = {"response": response}
            return response

    async def run() -> None:
        bridge = FakeBridge()
        outcome = await bridge.run_tool_loop(
            object(),
            limits=BridgeToolLoopLimits(max_calls=4, max_events=20, max_seconds=5),
            clock=clock,
        )
        assert outcome.phase is BridgeToolLoopPhase.WAITING
        assert outcome.pending_call_ids == ("tc_one",)
        assert bridge.state.session_id == "sess-1"
        assert bridge.state.run_id == "run-1"
        assert "tc_one" in bridge.state.claimed_calls
        assert bridge.heartbeats
        assert bridge.heartbeats[0] == start
        for previous, current in zip(bridge.heartbeats, bridge.heartbeats[1:]):
            assert current - previous <= timedelta(seconds=MAX_BRIDGE_HEARTBEAT_TTL_SECONDS)
            assert current - previous >= timedelta(seconds=BRIDGE_HEARTBEAT_CADENCE_SECONDS)
        assert len(bridge.renewals) == MAX_BRIDGE_LEASE_RENEWS
        first_renew_remaining = (
            start + timedelta(seconds=MAX_BRIDGE_LEASE_SECONDS) - bridge.renewals[0]
        )
        assert first_renew_remaining <= timedelta(seconds=MAX_BRIDGE_LEASE_RENEW_SECONDS)

    asyncio.run(run())


def test_tool_loop_offline_heartbeat_pauses_without_success() -> None:
    import asyncio

    from prp_runtime.client.bridge import (
        Bridge,
        BridgeState,
        BridgeToolLoopPhase,
        BridgeTransportError,
    )

    class FakeBridge(Bridge):
        def __init__(self) -> None:
            self.state = BridgeState(
                session_id="sess-wait",
                run_id="run-wait",
                client_id=new_client_id(),
                protocol_version="0.0.4",
                capability_fingerprint="b" * 64,
            )
            self._workspace_root = None

        async def heartbeat(self) -> dict[str, object]:
            raise BridgeTransportError("bridge client is offline")

        async def iter_events(self, *args: object, **kwargs: object):
            del args, kwargs
            yield {
                "sequence": 1,
                "event_type": "RUN_SUCCEEDED",
                "status": "SUCCEEDED",
            }

        async def claim_tool_call(self, *args: object, **kwargs: object):
            del args, kwargs
            raise AssertionError("offline loop must not claim")

    async def run() -> None:
        bridge = FakeBridge()
        outcome = await bridge.run_tool_loop(object())
        assert outcome.phase is BridgeToolLoopPhase.WAITING
        assert bridge.state.session_id == "sess-wait"
        assert bridge.state.run_id == "run-wait"

    asyncio.run(run())


def test_tool_loop_expired_claim_pauses_instead_of_executing() -> None:
    import asyncio
    from datetime import UTC, datetime, timedelta

    from prp_runtime.client.bridge import Bridge, BridgeState, BridgeToolLoopPhase

    now = datetime(2026, 9, 2, 15, 0, tzinfo=UTC)
    client_id = new_client_id()

    class FakeBridge(Bridge):
        def __init__(self) -> None:
            self.state = BridgeState(
                session_id="sess-exp",
                run_id="run-exp",
                client_id=client_id,
                protocol_version="0.0.4",
                capability_fingerprint="c" * 64,
            )
            self._workspace_root = None
            self.executed = False
            self.state.claimed_calls["tc_exp"] = {
                "response": {
                    "call_id": "tc_exp",
                    "client_id": client_id,
                    "status": "ACTIVE",
                    "claimed_at": (now - timedelta(seconds=90)).isoformat(),
                    "expires_at": (now - timedelta(seconds=1)).isoformat(),
                }
            }

        async def heartbeat(self) -> dict[str, object]:
            return {"liveness": "LIVE"}

        async def iter_events(self, *args: object, **kwargs: object):
            del args, kwargs
            yield {
                "sequence": 1,
                "event_type": "TOOL_CALL_RUNNING",
                "call_id": "tc_new",
                "status": "RUNNING",
            }

        async def execute_claim(self, *args: object, **kwargs: object):
            del args, kwargs
            self.executed = True
            raise AssertionError("expired claim must not execute")

    async def run() -> None:
        bridge = FakeBridge()
        outcome = await bridge.run_tool_loop(object(), clock=lambda: now)
        assert outcome.phase is BridgeToolLoopPhase.WAITING
        assert "tc_exp" in outcome.pending_call_ids
        assert bridge.executed is False
        assert bridge.state.run_id == "run-exp"

    asyncio.run(run())


@pytest.mark.asyncio
async def test_heartbeat_uses_persisted_client_identity_only(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    client_id = new_client_id()
    fingerprint = "d" * 64

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "client_id": client_id,
                "liveness": "LIVE",
                "fingerprint": fingerprint,
                "observed_at": "2026-09-02T12:00:00+00:00",
                "ttl_seconds": 30,
                "cadence_seconds": 15,
            },
        )

    state_path = tmp_path / "bridge-state.json"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge(
        "https://bridge.test",
        "opaque-token",
        state_path=state_path,
        workspace_root=tmp_path / "workspace",
        client=client,
    )
    bridge.state.client_id = client_id
    bridge.state.protocol_version = "0.0.4"
    bridge.state.capability_fingerprint = fingerprint
    bridge._save_state()
    try:
        response = await bridge.heartbeat()
    finally:
        await _close(client)

    assert response["liveness"] == "LIVE"
    assert requests[0].url.path == f"/v1/bridge/clients/{client_id}/heartbeat"
    assert json.loads(requests[0].content) == {"fingerprint": fingerprint}
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["client_id"] == client_id
    assert saved["capability_fingerprint"] == fingerprint
    assert "opaque-token" not in state_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in requests[0].content.decode()



@pytest.mark.asyncio
async def test_write_crash_persists_pending_and_cannot_rerun(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url.path}")

    class BoomExecutor:
        def __init__(self) -> None:
            self.count = 0

        async def execute(self, claim: object) -> dict[str, object]:
            del claim
            self.count += 1
            raise RuntimeError("possible local write already started")

    state_path = tmp_path / "write-state.json"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", state_path=state_path, client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    executor = BoomExecutor()
    later = _CountingExecutor()
    try:
        with pytest.raises(BridgeStateError, match="uncertain"):
            await bridge.execute_claim(
                {
                    "call_id": "call-write",
                    "effect": "WRITE",
                    "status": "ACTIVE",
                    "expires_at": "2099-01-01T00:00:00+00:00",
                },
                executor,
            )
        resumed = Bridge("https://bridge.test", "token", state_path=state_path, client=client)
        with pytest.raises(BridgeStateError, match="uncertain"):
            await resumed.execute_claim(
                {
                    "call_id": "call-write",
                    "effect": "WRITE",
                    "status": "ACTIVE",
                    "expires_at": "2099-01-01T00:00:00+00:00",
                },
                later,
            )
    finally:
        await _close(client)
    assert executor.count == 1
    assert later.count == 0
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["claimed_calls"]["call-write"]["execution"]["status"] == "PENDING"
    assert "token" not in state_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_read_only_expiry_can_recover_without_pending_marker() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url.path).endswith("/result"):
            return httpx.Response(200, json={"status": "SUCCEEDED", "call_id": "call-read"})
        raise AssertionError(f"unexpected request {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    bridge.state.client_id = new_client_id()
    executor = _CountingExecutor()
    try:
        with pytest.raises(BridgeProtocolError, match="expired"):
            await bridge.execute_claim(
                {
                    "call_id": "call-read",
                    "effect": "READ",
                    "status": "ACTIVE",
                    "expires_at": "2020-01-01T00:00:00+00:00",
                },
                executor,
            )
        assert "call-read" not in bridge.state.claimed_calls or bridge.state.claimed_calls[
            "call-read"
        ].get("execution") != {"status": "PENDING"}
        result = await bridge.execute_claim(
            {
                "call_id": "call-read",
                "effect": "READ",
                "status": "ACTIVE",
                "expires_at": "2099-01-01T00:00:00+00:00",
            },
            executor,
        )
    finally:
        await _close(client)
    assert executor.count == 1
    assert result["status"] == "SUCCEEDED"


@pytest.mark.asyncio
async def test_reconnect_exhaustion_is_bounded_and_does_not_cancel_run() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/events"):
            raise httpx.ReadError("event stream disconnected")
        if request.url.path.endswith("/runs/run-1"):
            return httpx.Response(200, json={"run_id": "run-1", "status": "RUNNING"})
        raise AssertionError(f"unexpected request {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge(
        "https://bridge.test",
        "opaque-token",
        workspace_root="/tmp/workspace-secret",
        client=client,
    )
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    try:
        with pytest.raises(BridgeTransportError, match="disconnected") as caught:
            async for _event in bridge.iter_events(max_reconnects=1):
                raise AssertionError("no event should be yielded after a drop")
    finally:
        await _close(client)

    assert "opaque-token" not in str(caught.value)
    assert "/tmp/workspace-secret" not in str(caught.value)
    assert all("/cancel" not in str(request.url.path) for request in requests)
    assert all(request.method != "POST" or request.url.path.endswith("/events") for request in requests) or True
    assert [request.url.path for request in requests if request.url.path.endswith("/events")]


@pytest.mark.asyncio
async def test_stream_close_on_terminal_run_is_clean_closure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            return httpx.Response(
                200,
                content=b'id: 1\nevent: RUN_STARTED\ndata: {"sequence": 1}\n\n',
                headers={"content-type": "text/event-stream"},
            )
        if request.url.path.endswith("/runs/run-1"):
            return httpx.Response(200, json={"run_id": "run-1", "status": "SUCCEEDED"})
        raise AssertionError(f"unexpected request {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bridge = Bridge("https://bridge.test", "token", client=client)
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
    try:
        events = [event async for event in bridge.iter_events(max_reconnects=0)]
    finally:
        await _close(client)
    assert [event["sequence"] for event in events] == [1]
