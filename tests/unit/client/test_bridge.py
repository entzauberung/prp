"""Focused fake-transport tests for the model-free Bridge client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from prp_runtime.client.bridge import (
    Bridge,
    BridgeProtocolError,
    BridgeState,
    BridgeStateError,
    BridgeTransportError,
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

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        if request.url.path.endswith("/claim"):
            request_count += 1
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
    bridge.state.session_id = "sess-1"
    bridge.state.run_id = "run-1"
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
