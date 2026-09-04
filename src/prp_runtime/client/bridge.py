"""Small, model-free client for an authorized local Workspace Bridge.

The bridge is a transport and durable-cursor library. It does not execute a
model, shell command, or arbitrary host path. A caller claims a persisted tool
call, performs its own bounded local operation, and submits the result with a
stable idempotency key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime, timedelta
from dataclasses import dataclass, field
from enum import StrEnum, unique
from pathlib import Path
from typing import Any, Self, cast

import httpx

from prp_runtime.domain.models import (
    BRIDGE_HEARTBEAT_CADENCE_SECONDS,
    MAX_BRIDGE_HEARTBEAT_TTL_SECONDS,
    MAX_BRIDGE_LEASE_RENEW_SECONDS,
    MAX_BRIDGE_LEASE_RENEWS,
    MAX_BRIDGE_LEASE_TOTAL_SECONDS,
    project_public_bridge_dispatch,
)
from prp_runtime.json_support import StrictJsonError, strict_json_loads

__all__ = [
    "Bridge",
    "BridgeClient",
    "BridgeError",
    "BridgeHTTPError",
    "BridgeProtocolError",
    "BridgeState",
    "BridgeStateError",
    "BridgeToolLoopError",
    "BridgeToolLoopLimits",
    "BridgeToolLoopPhase",
    "BridgeToolLoopResult",
    "BridgeTransportError",
    "public_dispatch_payload",
]

_ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_TERMINAL_EVENT_TYPES = frozenset(
    {"RUN_SUCCEEDED", "RUN_FAILED", "RUN_CANCELLED"}
)
_STATE_VERSION = 1


class BridgeError(RuntimeError):
    """Base error for client-side bridge failures."""


class BridgeHTTPError(BridgeError):
    """An HTTP response rejected by the remote API."""

    def __init__(self, status_code: int, *, method: str, path: str) -> None:
        super().__init__(f"bridge request {method} {path} failed with HTTP {status_code}")
        self.status_code = status_code
        self.method = method
        self.path = path


class BridgeTransportError(BridgeError):
    """A request could not reach the remote API or an event stream broke."""


class BridgeProtocolError(BridgeError):
    """The remote API returned a response outside the bridge contract."""


class BridgeStateError(BridgeError):
    """The local resume state is corrupt or conflicts with a retry."""


class BridgeToolLoopError(BridgeError):
    """A bounded Bridge tool loop cannot continue safely."""


@unique
class BridgeToolLoopPhase(StrEnum):
    """Observable stopping points for one resumable tool loop."""

    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING = "WAITING"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class BridgeToolLoopLimits:
    """Finite budgets that keep event-driven local execution from busy looping."""

    max_calls: int = 32
    max_events: int = 256
    max_reconnects: int = 3
    max_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.max_calls <= 0 or self.max_events <= 0:
            raise ValueError("Bridge tool loop counts must be positive")
        if self.max_reconnects < 0:
            raise ValueError("Bridge tool loop reconnects must not be negative")
        if self.max_seconds <= 0:
            raise ValueError("Bridge tool loop time must be positive")


@dataclass(frozen=True)
class BridgeToolLoopResult:
    """A loop outcome that contains no local paths or credentials."""

    phase: BridgeToolLoopPhase
    processed_call_ids: tuple[str, ...] = ()
    pending_call_ids: tuple[str, ...] = ()
    terminal_event: dict[str, Any] | None = None


def _state_entries(
    value: Mapping[Any, Any], field_name: str
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise BridgeStateError(f"bridge {field_name} keys must be non-empty strings")
        if not isinstance(item, dict):
            raise BridgeStateError(f"bridge {field_name} entries must be JSON objects")
        entries[key] = dict(item)
    return entries


@dataclass
class BridgeState:
    """The only state persisted by a bridge; credentials and host paths are absent."""

    session_id: str | None = None
    run_id: str | None = None
    event_cursor: int = 0
    claimed_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    submitted_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    client_id: str | None = None
    protocol_version: str | None = None
    capability_fingerprint: str | None = None

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> BridgeState:
        """Load and validate the small, versioned state document."""
        if not isinstance(value, Mapping):
            raise BridgeStateError("bridge state must be a JSON object")
        if value.get("version") != _STATE_VERSION:
            raise BridgeStateError("unsupported bridge state version")
        cursor = value.get("event_cursor", 0)
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise BridgeStateError("bridge event_cursor must be a non-negative integer")
        session_id = value.get("session_id")
        run_id = value.get("run_id")
        client_id = value.get("client_id")
        protocol_version = value.get("protocol_version")
        capability_fingerprint = value.get("capability_fingerprint")
        if session_id is not None and not isinstance(session_id, str):
            raise BridgeStateError("bridge session_id must be a string or null")
        if run_id is not None and not isinstance(run_id, str):
            raise BridgeStateError("bridge run_id must be a string or null")
        if client_id is not None and (
            not isinstance(client_id, str) or not client_id or _is_absolute_path(client_id)
        ):
            raise BridgeStateError("bridge client_id is invalid")
        if protocol_version is not None and protocol_version != "0.0.4":
            raise BridgeStateError("unsupported bridge protocol version")
        if capability_fingerprint is not None and (
            not isinstance(capability_fingerprint, str)
            or len(capability_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in capability_fingerprint)
        ):
            raise BridgeStateError("bridge capability fingerprint is invalid")
        claimed = value.get("claimed_calls", {})
        submitted = value.get("submitted_results", {})
        if not isinstance(claimed, dict) or not isinstance(submitted, dict):
            raise BridgeStateError("bridge call state must be JSON objects")
        state = cls(
            session_id=session_id,
            run_id=run_id,
            event_cursor=cursor,
            claimed_calls=_state_entries(claimed, "claimed_calls"),
            submitted_results=_state_entries(submitted, "submitted_results"),
            client_id=client_id,
            protocol_version=protocol_version,
            capability_fingerprint=capability_fingerprint,
        )
        _assert_state_safe(state.to_json())
        return state

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe state document without secrets or local paths."""
        payload = {
            "version": _STATE_VERSION,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "event_cursor": self.event_cursor,
            "claimed_calls": self.claimed_calls,
            "submitted_results": self.submitted_results,
            "client_id": self.client_id,
            "protocol_version": self.protocol_version,
            "capability_fingerprint": self.capability_fingerprint,
        }
        _assert_state_safe(payload)
        return payload


def _json_payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        dumped = dict(value)
    else:
        raise TypeError("bridge payload must be a mapping or a Pydantic model")
    if not isinstance(dumped, dict):
        raise TypeError("bridge payload must serialize to a JSON object")
    return dumped


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_absolute_path(value: str) -> bool:
    return value.startswith(("/", "\\")) or bool(_ABSOLUTE_WINDOWS_PATH.match(value))


def _assert_remote_safe(value: Any, *, local_root: Path | None = None) -> None:
    """Reject host paths and credential keys before serialization."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _OUTBOUND_FORBIDDEN_KEYS:
                raise BridgeProtocolError(
                    f"{key} cannot be sent through the public Bridge boundary"
                )
            _assert_remote_safe(nested, local_root=local_root)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_remote_safe(nested, local_root=local_root)
    elif isinstance(value, str):
        if _is_absolute_path(value):
            raise BridgeProtocolError("absolute local paths must not be sent to the server")
        if local_root is not None and str(local_root) in value:
            raise BridgeProtocolError("the configured local Workspace path must not be sent")


class Bridge:
    """Authenticated transport with resumable event and tool-result state."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        state_path: str | Path | None = None,
        workspace_root: str | Path | None = None,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not token or not token.strip():
            raise ValueError("token must not be empty")
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._state_path = None if state_path is None else Path(state_path)
        self._workspace_root = (
            None if workspace_root is None else Path(workspace_root).expanduser().resolve()
        )
        self._client = client or httpx.AsyncClient(trust_env=False)
        self._owns_client = client is None
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self.state = self._load_state()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _load_state(self) -> BridgeState:
        if self._state_path is None or not self._state_path.exists():
            return BridgeState()
        try:
            value = strict_json_loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, StrictJsonError) as error:
            raise BridgeStateError("bridge state cannot be read") from error
        if not isinstance(value, dict):
            raise BridgeStateError("bridge state must be a JSON object")
        return BridgeState.from_json(value)

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        parent = self._state_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        descriptor: int | None = None
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                dir=parent, prefix=f".{self._state_path.name}.", suffix=".tmp"
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = None
                json.dump(self.state.to_json(), stream, ensure_ascii=True, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self._state_path)
            temporary_path = None
        except OSError as error:
            raise BridgeStateError("bridge state cannot be written atomically") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
        }
        if extra is not None:
            headers.update(extra)
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        try:
            response = await self._client.request(
                method,
                self._url(path),
                headers=self._headers(headers),
                params=params,
                json=None if body is None else dict(body),
            )
        except httpx.HTTPError as error:
            raise BridgeTransportError(f"bridge request {method} {path} failed") from error
        if not 200 <= response.status_code < 300:
            raise BridgeHTTPError(response.status_code, method=method, path=path)
        if not response.content:
            return {}
        try:
            return strict_json_loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, StrictJsonError) as error:
            raise BridgeProtocolError(f"bridge response {method} {path} was not JSON") from error

    @staticmethod
    def _session_path(session_id: str) -> str:
        return f"/v1/sessions/{session_id}"

    def _require_session(self, session_id: str | None) -> str:
        resolved = session_id or self.state.session_id
        if not resolved:
            raise BridgeStateError("a session is required before using the bridge")
        return resolved

    def _require_run(self, run_id: str | None) -> str:
        resolved = run_id or self.state.run_id
        if not resolved:
            raise BridgeStateError("a run is required before using the bridge")
        return resolved

    def _require_client_id(self) -> str:
        client_id = self.state.client_id
        if not client_id:
            raise BridgeStateError("a client identity is required before using the bridge")
        return client_id

    async def register_client(self, payload: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Register one model-free client identity without credentials or roots."""
        body = _json_payload(payload)
        _assert_remote_safe(body, local_root=self._workspace_root)
        _reject_authority_fields(body)
        response = await self._request("POST", "/v1/bridge/clients", body=body)
        if not isinstance(response, dict):
            raise BridgeProtocolError("client registration response must be a JSON object")
        return response

    async def handshake(self, payload: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Advertise local capabilities and persist public identity facts only."""
        body = _json_payload(payload)
        _assert_remote_safe(body, local_root=self._workspace_root)
        _reject_authority_fields(body)
        response = await self._request("POST", "/v1/bridge/handshake", body=body)
        if not isinstance(response, dict) or response.get("accepted") is not True:
            raise BridgeProtocolError("handshake was not accepted")
        client_id = response.get("client_id")
        protocol_version = response.get("protocol_version")
        fingerprint = response.get("fingerprint")
        if (
            not isinstance(client_id, str)
            or not isinstance(protocol_version, str)
            or not isinstance(fingerprint, str)
        ):
            raise BridgeProtocolError("handshake response is missing identity facts")
        if protocol_version != "0.0.4":
            raise BridgeProtocolError("unsupported Bridge protocol version")
        if _is_absolute_path(client_id):
            raise BridgeProtocolError("handshake client_id must not be a host path")
        self.state.client_id = client_id
        self.state.protocol_version = protocol_version
        self.state.capability_fingerprint = fingerprint
        self._save_state()
        return response

    async def heartbeat(self) -> dict[str, Any]:
        """Refresh server liveness using persisted client identity only."""
        client_id = self._require_client_id()
        fingerprint = self.state.capability_fingerprint
        if not isinstance(fingerprint, str) or not fingerprint:
            raise BridgeStateError("a capability fingerprint is required before heartbeat")
        response = await self._request(
            "POST",
            f"/v1/bridge/clients/{client_id}/heartbeat",
            body={"fingerprint": fingerprint},
        )
        if not isinstance(response, dict):
            raise BridgeProtocolError("heartbeat response must be a JSON object")
        _assert_remote_safe(response, local_root=self._workspace_root)
        return response

    async def create_session(
        self,
        workspace_id: str,
        *,
        access: tuple[str, ...] | list[str] = ("READ",),
        agent_options: Mapping[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"workspace_id": workspace_id, "access": list(access)}
        if agent_options is not None:
            body["agent_options"] = dict(agent_options)
        if expires_at is not None:
            body["expires_at"] = expires_at
        _assert_remote_safe(body, local_root=self._workspace_root)
        response = await self._request("POST", "/v1/sessions", body=body)
        if not isinstance(response, dict) or not isinstance(response.get("session_id"), str):
            raise BridgeProtocolError("session response is missing session_id")
        self.state.session_id = response["session_id"]
        self.state.run_id = None
        self.state.event_cursor = 0
        self.state.claimed_calls.clear()
        self.state.submitted_results.clear()
        self._save_state()
        return response

    async def get_session(self, session_id: str | None = None) -> dict[str, Any]:
        resolved = self._require_session(session_id)
        response = await self._request("GET", self._session_path(resolved))
        if not isinstance(response, dict):
            raise BridgeProtocolError("session response must be a JSON object")
        return response

    async def create_run(
        self,
        session_id: str | None,
        payload: Mapping[str, Any] | str,
        **fields: Any,
    ) -> dict[str, Any]:
        resolved = self._require_session(session_id)
        body = {"input": payload} if isinstance(payload, str) else _json_payload(payload)
        body.update(fields)
        _assert_remote_safe(body, local_root=self._workspace_root)
        response = await self._request(
            "POST", f"{self._session_path(resolved)}/runs", body=body
        )
        if not isinstance(response, dict) or not isinstance(response.get("run_id"), str):
            raise BridgeProtocolError("run response is missing run_id")
        self.state.run_id = response["run_id"]
        self.state.event_cursor = 0
        self.state.claimed_calls.clear()
        self.state.submitted_results.clear()
        self._save_state()
        return response

    start_run = create_run

    async def get_run(
        self, session_id: str | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        session = self._require_session(session_id)
        run = self._require_run(run_id)
        response = await self._request(
            "GET", f"{self._session_path(session)}/runs/{run}"
        )
        if not isinstance(response, dict):
            raise BridgeProtocolError("run response must be a JSON object")
        return response

    async def cancel_run(
        self, session_id: str | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        session = self._require_session(session_id)
        run = self._require_run(run_id)
        response = await self._request(
            "POST", f"{self._session_path(session)}/runs/{run}/cancel"
        )
        if not isinstance(response, dict):
            raise BridgeProtocolError("cancel response must be a JSON object")
        return response

    async def list_tool_calls(
        self,
        session_id: str | None = None,
        run_id: str | None = None,
        *,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        session = self._require_session(session_id)
        run = self._require_run(run_id)
        params = None if status is None else {"status": status}
        response = await self._request(
            "GET", f"{self._session_path(session)}/runs/{run}/tool-calls", params=params
        )
        if not isinstance(response, list) or not all(isinstance(item, dict) for item in response):
            raise BridgeProtocolError("tool call response must be a JSON object list")
        return response

    async def claim_tool_call(
        self,
        call_id: str,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._require_session(session_id)
        run = self._require_run(run_id)
        existing = self.state.claimed_calls.get(call_id)
        if existing is not None:
            response = existing.get("response")
            if isinstance(response, dict):
                return response
            raise BridgeStateError("stored tool claim has an invalid response")
        path = f"{self._session_path(session)}/runs/{run}/tool-calls/{call_id}/claim"
        response = await self._request(
            "POST",
            path,
            body={"call_id": call_id, "client_id": self._require_client_id()},
            headers={"Idempotency-Key": self._idempotency_key("claim", call_id)},
        )
        if not isinstance(response, dict):
            raise BridgeProtocolError("tool claim response must be a JSON object")
        self.state.claimed_calls[call_id] = {"response": response}
        self._save_state()
        return response

    async def renew_claim(
        self,
        call_id: str,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._require_session(session_id)
        run = self._require_run(run_id)
        path = f"{self._session_path(session)}/runs/{run}/tool-calls/{call_id}/claim/renew"
        response = await self._request(
            "POST", path, body={"client_id": self._require_client_id()}
        )
        if not isinstance(response, dict):
            raise BridgeProtocolError("tool claim renewal response must be a JSON object")
        entry = self.state.claimed_calls.setdefault(call_id, {})
        entry["response"] = response
        self._save_state()
        return response

    async def release_claim(
        self,
        call_id: str,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._require_session(session_id)
        run = self._require_run(run_id)
        path = f"{self._session_path(session)}/runs/{run}/tool-calls/{call_id}/claim/release"
        response = await self._request(
            "POST", path, body={"client_id": self._require_client_id()}
        )
        if not isinstance(response, dict):
            raise BridgeProtocolError("tool claim release response must be a JSON object")
        entry = self.state.claimed_calls.setdefault(call_id, {})
        entry["response"] = response
        self._save_state()
        return response

    async def submit_tool_result(
        self,
        call_id: str,
        result: Mapping[str, Any] | Any,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._require_session(session_id)
        run = self._require_run(run_id)
        body = _json_payload(result)
        if body.get("dev_only") is not True:
            raise BridgeProtocolError("Bridge tool results must be marked dev_only=true")
        _assert_remote_safe(body, local_root=self._workspace_root)
        digest = _fingerprint(body)
        existing = self.state.submitted_results.get(call_id)
        if existing is not None:
            if existing.get("fingerprint") != digest:
                raise BridgeStateError("a tool result retry conflicts with persisted state")
            response = existing.get("response")
            if isinstance(response, dict):
                return response
            raise BridgeStateError("stored tool result has an invalid response")
        path = f"{self._session_path(session)}/runs/{run}/tool-calls/{call_id}/result"
        payload = dict(body)
        payload["client_id"] = self._require_client_id()
        response = await self._request(
            "POST",
            path,
            body=payload,
            headers={"Idempotency-Key": self._idempotency_key("result", call_id)},
        )
        if not isinstance(response, dict):
            raise BridgeProtocolError("tool result response must be a JSON object")
        self.state.submitted_results[call_id] = {
            "fingerprint": digest,
            "response": response,
        }
        self._save_state()
        return response

    async def execute_claim(
        self,
        claim: Mapping[str, Any],
        executor: Any,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a locally planned claim and submit its bounded payload once."""
        call_id = claim.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise BridgeProtocolError("claim response is missing call_id")
        submitted = self.state.submitted_results.get(call_id)
        if submitted is not None:
            response = submitted.get("response")
            if isinstance(response, dict):
                return response
            raise BridgeStateError("stored tool result has an invalid response")
        stored_claim = self.state.claimed_calls.get(call_id)
        stored_execution = None if stored_claim is None else stored_claim.get("execution")
        if stored_execution is not None:
            if not isinstance(stored_execution, dict):
                raise BridgeStateError("stored local execution has an invalid shape")
            if stored_execution.get("status") == "PENDING":
                raise BridgeStateError(
                    "a previous local write is uncertain and must not be rerun"
                )
            payload = stored_execution.get("payload")
            fingerprint = stored_execution.get("fingerprint")
            if not isinstance(payload, dict) or not isinstance(fingerprint, str):
                raise BridgeStateError("stored local execution is incomplete")
            if fingerprint != _fingerprint(payload):
                raise BridgeStateError("stored local execution fingerprint is invalid")
        else:
            self._assert_claim_executable(claim)
            entry = self.state.claimed_calls.setdefault(
                call_id, {"response": {"call_id": call_id}}
            )
            if str(claim.get("effect", "")).upper() == "WRITE":
                entry["execution"] = {"status": "PENDING"}
                self._save_state()
            try:
                payload = await executor.execute(_local_execution_claim(claim))
            except AttributeError as error:
                raise BridgeProtocolError("bridge executor does not provide execute") from error
            except BridgeProtocolError:
                raise
            except Exception as error:
                if str(claim.get("effect", "")).upper() == "WRITE":
                    raise BridgeStateError(
                        "a previous local write is uncertain and must not be rerun"
                    ) from error
                raise
            if not isinstance(payload, Mapping):
                raise BridgeProtocolError("bridge executor returned a non-object payload")
            payload = dict(payload)
            if payload.get("dev_only") is False:
                raise BridgeProtocolError("Bridge executor cannot opt out of dev_only=true")
            payload["dev_only"] = True
            _assert_remote_safe(payload, local_root=self._workspace_root)
            entry["execution"] = {
                "fingerprint": _fingerprint(payload),
                "payload": payload,
            }
            self._save_state()
        return await self.submit_tool_result(
            call_id,
            payload,
            session_id=session_id,
            run_id=run_id,
        )

    def _loop_clock(self, clock: Callable[[], datetime] | None) -> Callable[[], datetime]:
        if clock is not None:
            return clock
        existing = getattr(self, "_clock", None)
        if callable(existing):
            return existing
        return lambda: datetime.now(tz=UTC)

    def _active_assigned_claims(self) -> list[tuple[str, dict[str, Any]]]:
        client_id = self.state.client_id
        active: list[tuple[str, dict[str, Any]]] = []
        for call_id, entry in self.state.claimed_calls.items():
            if call_id in self.state.submitted_results:
                continue
            if not isinstance(entry, dict):
                continue
            response = entry.get("response")
            if not isinstance(response, dict):
                continue
            status = response.get("status")
            if status not in (None, "ACTIVE"):
                continue
            if response.get("closed_at") is not None:
                continue
            owner = response.get("client_id")
            if owner is not None and owner != client_id:
                continue
            active.append((call_id, response))
        return active

    @staticmethod
    def _capped_server_int(value: object, default: int, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return default
        return min(value, maximum)

    def _apply_keepalive_limits(
        self, response: Mapping[str, Any], limits: dict[str, int]
    ) -> dict[str, int]:
        updated = dict(limits)
        updated["ttl"] = self._capped_server_int(
            response.get("ttl_seconds"), limits["ttl"], MAX_BRIDGE_HEARTBEAT_TTL_SECONDS
        )
        updated["cadence"] = self._capped_server_int(
            response.get("cadence_seconds"), limits["cadence"], updated["ttl"]
        )
        updated["max_renews"] = self._capped_server_int(
            response.get("max_renews"), limits["max_renews"], MAX_BRIDGE_LEASE_RENEWS
        )
        updated["lease_renew"] = self._capped_server_int(
            response.get("lease_renew_seconds"),
            limits["lease_renew"],
            MAX_BRIDGE_LEASE_RENEW_SECONDS,
        )
        updated["lease_total"] = self._capped_server_int(
            response.get("lease_total_seconds"),
            limits["lease_total"],
            MAX_BRIDGE_LEASE_TOTAL_SECONDS,
        )
        return updated

    def _wait_loop_result(
        self, processed: list[str], extra_pending: tuple[str, ...] = ()
    ) -> BridgeToolLoopResult:
        pending = [call_id for call_id, _claim in self._active_assigned_claims()]
        for call_id in extra_pending:
            if call_id not in pending:
                pending.append(call_id)
        return BridgeToolLoopResult(
            phase=BridgeToolLoopPhase.WAITING,
            processed_call_ids=tuple(processed),
            pending_call_ids=tuple(pending),
        )

    @staticmethod
    def _claim_timestamp(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return None
        return parsed

    def _claim_expired(self, claim: Mapping[str, Any], *, now: datetime) -> bool:
        expires_at = self._claim_timestamp(claim.get("expires_at"))
        return expires_at is not None and expires_at <= now

    def _claim_needs_renew(
        self, claim: Mapping[str, Any], *, now: datetime, limits: Mapping[str, int]
    ) -> bool:
        expires_at = self._claim_timestamp(claim.get("expires_at"))
        if expires_at is None:
            return False
        remaining = (expires_at - now).total_seconds()
        return remaining <= float(limits["lease_renew"])

    def _claim_past_ceiling(
        self, claim: Mapping[str, Any], *, now: datetime, limits: Mapping[str, int]
    ) -> bool:
        claimed_at = self._claim_timestamp(claim.get("claimed_at"))
        expires_at = self._claim_timestamp(claim.get("expires_at"))
        if claimed_at is None:
            return False
        ceiling = claimed_at + timedelta(seconds=int(limits["lease_total"]))
        if now >= ceiling:
            return True
        return expires_at is not None and expires_at <= now

    async def _maintain_loop_keepalive(
        self,
        *,
        now_fn: Callable[[], datetime],
        last_heartbeat_at: datetime | None,
        limits: dict[str, int],
        processed: list[str],
        session_id: str | None,
        run_id: str | None,
        renew_counts: dict[str, int],
    ) -> tuple[BridgeToolLoopResult | None, datetime | None, dict[str, int]]:
        if not self.state.client_id or not self.state.capability_fingerprint:
            return None, last_heartbeat_at, limits
        now = now_fn()
        due = last_heartbeat_at is None
        if last_heartbeat_at is not None:
            elapsed = (now - last_heartbeat_at).total_seconds()
            if elapsed >= float(limits["cadence"]) or elapsed >= float(limits["ttl"]):
                due = True
        if due:
            try:
                response = await self.heartbeat()
            except (BridgeTransportError, BridgeHTTPError, BridgeProtocolError, BridgeStateError):
                return self._wait_loop_result(processed), last_heartbeat_at, limits
            liveness = str(response.get("liveness", "")).upper()
            if liveness and liveness != "LIVE":
                return self._wait_loop_result(processed), last_heartbeat_at, limits
            limits = self._apply_keepalive_limits(response, limits)
            last_heartbeat_at = now_fn()
        for call_id, claim in self._active_assigned_claims():
            now = now_fn()
            if self._claim_expired(claim, now=now) or self._claim_past_ceiling(
                claim, now=now, limits=limits
            ):
                return (
                    self._wait_loop_result(processed, (call_id,)),
                    last_heartbeat_at,
                    limits,
                )
            if not self._claim_needs_renew(claim, now=now, limits=limits):
                continue
            if renew_counts.get(call_id, 0) >= int(limits["max_renews"]):
                return (
                    self._wait_loop_result(processed, (call_id,)),
                    last_heartbeat_at,
                    limits,
                )
            try:
                renewed = await self.renew_claim(call_id, session_id, run_id)
            except (
                BridgeTransportError,
                BridgeHTTPError,
                BridgeProtocolError,
                BridgeStateError,
            ):
                return (
                    self._wait_loop_result(processed, (call_id,)),
                    last_heartbeat_at,
                    limits,
                )
            if str(renewed.get("status", "")).upper() not in {"", "ACTIVE"}:
                return (
                    self._wait_loop_result(processed, (call_id,)),
                    last_heartbeat_at,
                    limits,
                )
            renew_counts[call_id] = renew_counts.get(call_id, 0) + 1
        return None, last_heartbeat_at, limits

    async def run_tool_loop(
        self,
        executor: Any,
        session_id: str | None = None,
        run_id: str | None = None,
        *,
        limits: BridgeToolLoopLimits | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> BridgeToolLoopResult:
        """Process approved tool events until approval, wait, or run termination.

        Keepalive runs inside this loop. The current event wait is bounded by the
        server heartbeat cadence; no detached daemon or client-chosen lease is
        created.
        """
        import asyncio

        budget = limits or BridgeToolLoopLimits()
        now_fn = self._loop_clock(clock)
        processed: list[str] = []
        pending: list[str] = []
        seen_calls = set(self.state.submitted_results)
        seen_calls.update(
            call_id
            for call_id, entry in self.state.claimed_calls.items()
            if isinstance(entry.get("execution"), dict)
        )
        event_count = 0
        last_heartbeat_at: datetime | None = None
        keepalive_limits = {
            "ttl": MAX_BRIDGE_HEARTBEAT_TTL_SECONDS,
            "cadence": BRIDGE_HEARTBEAT_CADENCE_SECONDS,
            "max_renews": MAX_BRIDGE_LEASE_RENEWS,
            "lease_renew": MAX_BRIDGE_LEASE_RENEW_SECONDS,
            "lease_total": MAX_BRIDGE_LEASE_TOTAL_SECONDS,
        }
        renew_counts: dict[str, int] = {}
        event_iter = self.iter_events(
            session_id,
            run_id,
            max_reconnects=budget.max_reconnects,
        ).__aiter__()
        pending_event: asyncio.Task[Any] | None = None
        try:
            async with asyncio.timeout(budget.max_seconds):
                while True:
                    pause, last_heartbeat_at, keepalive_limits = (
                        await self._maintain_loop_keepalive(
                            now_fn=now_fn,
                            last_heartbeat_at=last_heartbeat_at,
                            limits=keepalive_limits,
                            processed=processed,
                            session_id=session_id,
                            run_id=run_id,
                            renew_counts=renew_counts,
                        )
                    )
                    if pause is not None:
                        return pause
                    if pending_event is None:
                        pending_event = asyncio.create_task(anext(event_iter))
                    wait_for = float(keepalive_limits["cadence"])
                    if last_heartbeat_at is not None:
                        remaining = wait_for - (
                            now_fn() - last_heartbeat_at
                        ).total_seconds()
                        wait_for = max(0.01, remaining)
                    done, _pending = await asyncio.wait(
                        {pending_event}, timeout=wait_for
                    )
                    if not done:
                        continue
                    try:
                        event = pending_event.result()
                    except StopAsyncIteration:
                        pending_event = None
                        raise BridgeToolLoopError(
                            "Bridge event stream ended before a terminal event"
                        ) from None
                    pending_event = None
                    event_count += 1
                    if event_count > budget.max_events:
                        raise BridgeToolLoopError(
                            "Bridge tool loop event limit exceeded"
                        )
                    if _is_terminal_event(event):
                        return BridgeToolLoopResult(
                            phase=BridgeToolLoopPhase.TERMINAL,
                            processed_call_ids=tuple(processed),
                            terminal_event=_public_event(
                                event, local_root=self._workspace_root
                            ),
                        )
                    call_id = _event_call_id(event)
                    if call_id is None or call_id in seen_calls:
                        continue
                    status = _event_status(event)
                    if status in {
                        "REQUESTED",
                        "AWAITING_APPROVAL",
                        "APPROVAL_REQUIRED",
                    }:
                        pending.append(call_id)
                        return BridgeToolLoopResult(
                            phase=BridgeToolLoopPhase.WAITING_APPROVAL,
                            processed_call_ids=tuple(processed),
                            pending_call_ids=tuple(pending),
                        )
                    if status not in {"RUNNING", "APPROVED", "CLAIMABLE"}:
                        raise BridgeToolLoopError(
                            f"Bridge tool call {call_id} has an ambiguous state"
                        )
                    if len(processed) >= budget.max_calls:
                        raise BridgeToolLoopError(
                            "Bridge tool loop call limit exceeded"
                        )
                    try:
                        claim = await self.claim_tool_call(call_id, session_id, run_id)
                    except (
                        BridgeTransportError,
                        BridgeHTTPError,
                        BridgeProtocolError,
                        BridgeStateError,
                    ):
                        return self._wait_loop_result(processed, (call_id,))
                    dispatch_claim = _merge_dispatch_facts(claim, event)
                    try:
                        await self.execute_claim(
                            dispatch_claim,
                            executor,
                            session_id=session_id,
                            run_id=run_id,
                        )
                    except (
                        BridgeTransportError,
                        BridgeHTTPError,
                        BridgeProtocolError,
                    ) as error:
                        message = str(error).lower()
                        if "expired" in message or "offline" in message:
                            return self._wait_loop_result(processed, (call_id,))
                        raise
                    seen_calls.add(call_id)
                    processed.append(call_id)
        except TimeoutError as error:
            raise BridgeToolLoopError("Bridge tool loop time limit exceeded") from error
        finally:
            if pending_event is not None and not pending_event.done():
                pending_event.cancel()
        raise BridgeToolLoopError("Bridge event stream ended before a terminal event")

    @staticmethod
    def _idempotency_key(kind: str, call_id: str) -> str:
        digest = hashlib.sha256(f"{kind}:{call_id}".encode()).hexdigest()
        return f"prp-bridge-{kind}-{digest}"

    async def iter_events(
        self,
        session_id: str | None = None,
        run_id: str | None = None,
        *,
        max_reconnects: int = 3,
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay SSE events from the stored cursor and reconnect after a drop."""
        if max_reconnects < 0:
            raise ValueError("max_reconnects must not be negative")
        session = self._require_session(session_id)
        run = self._require_run(run_id)
        reconnects = 0
        while True:
            path = f"{self._session_path(session)}/runs/{run}/events"
            try:
                async with self._client.stream(
                    "GET",
                    self._url(path),
                    params={"after": str(self.state.event_cursor)},
                    headers=self._headers(
                        {
                            "Accept": "text/event-stream",
                            "Last-Event-ID": str(self.state.event_cursor),
                        }
                    ),
                ) as response:
                    if not 200 <= response.status_code < 300:
                        raise BridgeHTTPError(response.status_code, method="GET", path=path)
                    event_type: str | None = None
                    data_lines: list[str] = []
                    async for line in response.aiter_lines():
                        if line == "":
                            parsed = self._finish_event(event_type, data_lines)
                            event_type = None
                            data_lines = []
                            if parsed is not None:
                                yield parsed
                                if self._event_cursor_is_terminal():
                                    return
                        elif line.startswith(":"):
                            continue
                        elif line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                    parsed = self._finish_event(event_type, data_lines)
                    if parsed is not None:
                        yield parsed
                        if self._event_cursor_is_terminal():
                            return
            except BridgeHTTPError:
                raise
            except httpx.HTTPError as error:
                if reconnects >= max_reconnects:
                    raise BridgeTransportError("bridge event stream disconnected") from error
                reconnects += 1
                continue

            if self._event_cursor_is_terminal():
                return
            current = await self.get_run(session, run)
            if str(current.get("status", "")) in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                return
            if reconnects >= max_reconnects:
                raise BridgeTransportError("bridge event stream closed before run completion")
            reconnects += 1

    watch_events = iter_events

    @staticmethod
    def _assert_claim_executable(claim: Mapping[str, Any]) -> None:
        status = claim.get("status")
        if status is not None and status != "ACTIVE":
            raise BridgeProtocolError("Bridge claim is not active")
        if claim.get("closed_at") is not None:
            raise BridgeProtocolError("Bridge claim is not active")
        expires_at = claim.get("expires_at")
        if expires_at is None:
            return
        if not isinstance(expires_at, str):
            raise BridgeProtocolError("Bridge claim lease is invalid")
        try:
            parsed = datetime.fromisoformat(expires_at)
        except ValueError as error:
            raise BridgeProtocolError("Bridge claim lease is invalid") from error
        if parsed.tzinfo is None:
            raise BridgeProtocolError("Bridge claim lease is invalid")
        if parsed <= datetime.now(tz=UTC):
            raise BridgeProtocolError("Bridge claim lease has expired")

    def _event_cursor_is_terminal(self) -> bool:

        return getattr(self, "_last_event_type", None) in _TERMINAL_EVENT_TYPES

    def _finish_event(
        self, event_type: str | None, data_lines: list[str]
    ) -> dict[str, Any] | None:
        if not data_lines:
            return None
        try:
            value = strict_json_loads("\n".join(data_lines))
        except StrictJsonError as error:
            raise BridgeProtocolError("bridge event data is not JSON") from error
        if not isinstance(value, dict):
            raise BridgeProtocolError("bridge event data must be a JSON object")
        if event_type is not None and "event_type" not in value:
            value["event_type"] = event_type
        sequence = value.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise BridgeProtocolError("bridge event is missing a positive sequence")
        if sequence <= self.state.event_cursor:
            return None
        self.state.event_cursor = sequence
        self._last_event_type = str(value.get("event_type", ""))
        self._save_state()
        return value


BridgeClient = Bridge


_DISPATCH_FACT_KEYS = frozenset(
    {
        "tool_name",
        "effect",
        "arguments",
        "scope",
        "snapshot_id",
        "work_unit_id",
        "requested_at",
    }
)
_AUTHORITY_EVENT_KEYS = frozenset(
    {
        "provider",
        "routing_policy",
        "strategy",
    }
)
_SENSITIVE_EVENT_KEYS = frozenset(
    {
        "authorization",
        "access_token",
        "api_key",
        "credential",
        "host_path",
        "password",
        "secret",
        "server_root",
        "token",
        "workspace_path",
        "workspace_root",
    }
)
_OUTBOUND_FORBIDDEN_KEYS = _SENSITIVE_EVENT_KEYS
_STATE_FORBIDDEN_KEYS = _AUTHORITY_EVENT_KEYS | _SENSITIVE_EVENT_KEYS | frozenset({"model"})


def _event_call_id(event: Mapping[str, Any]) -> str | None:
    candidates: list[Mapping[str, Any]] = [event]
    for key in ("data", "tool_call", "call"):
        nested = event.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    for candidate in candidates:
        value = candidate.get("call_id")
        if isinstance(value, str) and value:
            return value
    return None


def _event_status(event: Mapping[str, Any]) -> str:
    candidates: list[Mapping[str, Any]] = [event]
    for key in ("data", "tool_call", "call"):
        nested = event.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    for candidate in candidates:
        value = candidate.get("status")
        if isinstance(value, str):
            return value.upper()
    event_type = event.get("event_type")
    if isinstance(event_type, str):
        normalized = event_type.upper()
        if "APPROVAL" in normalized:
            return "APPROVAL_REQUIRED"
        for status in ("RUNNING", "REQUESTED", "SUCCEEDED", "FAILED", "CANCELLED"):
            if status in normalized:
                return status
    return ""


def _is_terminal_event(event: Mapping[str, Any]) -> bool:
    event_type = event.get("event_type")
    if isinstance(event_type, str) and event_type.upper() in _TERMINAL_EVENT_TYPES:
        return True
    return _event_status(event) in {"RUN_SUCCEEDED", "RUN_FAILED", "RUN_CANCELLED"}


def _local_execution_claim(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Drop durable client identity before a model-free local executor."""
    return {key: value for key, value in claim.items() if key != "client_id"}


def _merge_dispatch_facts(
    claim: Mapping[str, Any], event: Mapping[str, Any]
) -> dict[str, Any]:
    call_id = _event_call_id(event)
    if call_id is None or claim.get("call_id") != call_id:
        raise BridgeToolLoopError("claim and tool event call ids do not match")
    merged = dict(claim)
    sources: list[Mapping[str, Any]] = [event]
    nested = event.get("tool_call")
    if isinstance(nested, Mapping):
        sources.insert(0, nested)
    for key in _DISPATCH_FACT_KEYS:
        if key in merged:
            continue
        for source in sources:
            if key in source:
                merged[key] = source[key]
                break
    return merged


def public_dispatch_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project one public Bridge dispatch and reject private authority."""
    try:
        facts = project_public_bridge_dispatch(value)
    except (TypeError, ValueError) as error:
        raise BridgeProtocolError(str(error)) from error
    return facts.model_dump(mode="json")


def _reject_authority_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _AUTHORITY_EVENT_KEYS:
                raise BridgeProtocolError(
                    f"{key} is a server authority field and cannot cross the Bridge boundary"
                )
            _reject_authority_fields(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_authority_fields(nested)


def _assert_state_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _STATE_FORBIDDEN_KEYS:
                raise BridgeStateError(f"bridge state must not persist {key}")
            _assert_state_safe(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_state_safe(nested)


def _public_event(
    value: Mapping[str, Any], *, local_root: Path | None = None
) -> dict[str, Any]:
    _reject_authority_fields(value)
    return cast(dict[str, Any], _redact_event_value(value, local_root=local_root))


def _redact_event_value(
    value: Any, *, key: str | None = None, local_root: Path | None = None
) -> Any:
    if key is not None and key.lower() in _SENSITIVE_EVENT_KEYS:
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(nested_key): _redact_event_value(
                nested, key=str(nested_key), local_root=local_root
            )
            for nested_key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_event_value(nested, local_root=local_root) for nested in value]
    if isinstance(value, str):
        if local_root is not None and os.fspath(local_root) in value:
            return value.replace(os.fspath(local_root), "<local-root>")
        if _is_absolute_path(value):
            return "<local-path>"
    return value
