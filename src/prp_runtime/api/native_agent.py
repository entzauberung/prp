"""Authenticated Native Agent Session and durable lifecycle routes.

These routes are deliberately scoped below a Session.  The bearer dependency
supplies the principal and every Store read repeats that scope at the database
boundary; a client never supplies an owner id.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import Field, JsonValue, model_validator

from prp_runtime.api.auth import AuthContext, require_auth
from prp_runtime.api.native import (
    EVENT_STREAM_MEDIA_TYPE,
    RunEnvelope,
    RunEventEnvelope,
)
from prp_runtime.control.controller import RunController
from prp_runtime.control.progressive import RoundStatus
from prp_runtime.domain.enums import (
    BridgeClientLiveness,
    ExecutionLocation,
    ExecutionStrategy,
    MergeLedgerStatus,
    RunStatus,
    ToolCallStatus,
    ToolEffect,
)
from prp_runtime.domain.errors import (
    DomainValidationError,
    ErrorCode,
    StateError,
)
from prp_runtime.domain.models import (
    BRIDGE_HEARTBEAT_CADENCE_SECONDS,
    MAX_BRIDGE_HEARTBEAT_TTL_SECONDS,
    MAX_BRIDGE_LEASE_RENEW_SECONDS,
    MAX_BRIDGE_LEASE_RENEWS,
    MAX_BRIDGE_LEASE_SECONDS,
    MAX_BRIDGE_LEASE_TOTAL_SECONDS,
    BridgeClientStatus,
    BridgeHeartbeatFacts,
    BridgeHeartbeatView,
    ClientHandshakeAcceptance,
    ClientHandshakeRequest,
    DomainModel,
    ErrorInfo,
    NativeRunRequest,
    ClientCapabilityDescriptor,
    RegisteredBridgeClient,
    Session,
    SessionCreateRequest,
    SessionStatus,
    WorkspaceGrant,
)
from prp_runtime.domain.values import SnapshotId, ToolCallId, WorkspaceId, new_session_id, utc_now
from prp_runtime.domain.models import ClientId, Fingerprint, RoundId
from prp_runtime.policy.models import (
    ApprovalDecision,
    ApprovalIssuer,
    ApprovalOutcome,
    ApprovalRequest,
)
from prp_runtime.runtime.assembler import (
    CloudBundleError,
    CloudResultBundle,
    assemble_cloud_bundle,
    assemble_run_result,
    select_verified_cloud_snapshot,
)
from prp_runtime.runtime.bridge import BridgeAssignmentCoordinator
from prp_runtime.runtime.event_bus import EventBus
from prp_runtime.runtime.supervisor import RunSupervisor
from prp_runtime.storage.sqlite import DuplicateEntityError, MissingEntityError, SqliteStore
from prp_runtime.tools.models import (
    MAX_CHANGED_PATHS,
    MAX_TOOL_OUTPUT_BYTES,
    BridgeClaim,
    ToolCall,
    ToolResult,
)
from prp_runtime.workspace.backend import WorkspaceBackendError, select_snapshot_export_files
from prp_runtime.workspace.models import (
    BridgeManifestAcceptance,
    BridgeManifestPublication,
    SnapshotStatus,
    WorkspaceStatus,
)

__all__ = [
    "ApprovalDecisionRequest",
    "ApprovalView",
    "BridgeClaimRequest",
    "BridgeToolResultRequest",
    "CloudResultBundle",
    "create_router",
]

_MAX_EVENT_PAGE = 500
Auth = Annotated[AuthContext, Depends(require_auth)]


class ApprovalDecisionRequest(DomainModel):
    """Client input for one approval decision; issuer is server-injected."""

    outcome: ApprovalOutcome
    reason: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def _deny_has_a_reason(self) -> ApprovalDecisionRequest:
        if self.outcome is ApprovalOutcome.DENY and not self.reason:
            raise ValueError("DENY decisions require a reason")
        return self


class ApprovalView(DomainModel):
    """One approval request plus its optional immutable decision projection."""

    request_id: str
    call_id: ToolCallId
    run_id: str
    workspace_id: str
    tool_name: str
    effect: str
    scope: object
    reason: str
    issuer: ApprovalIssuer
    requested_at: object
    outcome: ApprovalOutcome | None = None
    decision_issuer: ApprovalIssuer | None = None
    decision_reason: str | None = None
    decided_at: object | None = None

    @classmethod
    def from_request(
        cls,
        request: ApprovalRequest,
        decision: ApprovalDecision | None = None,
    ) -> ApprovalView:
        return cls(
            request_id=request.request_id,
            call_id=request.call_id,
            run_id=request.run_id,
            workspace_id=request.workspace_id,
            tool_name=request.tool_name,
            effect=request.effect.value,
            scope=request.scope,
            reason=request.reason,
            issuer=request.issuer,
            requested_at=request.requested_at,
            outcome=None if decision is None else decision.outcome,
            decision_issuer=None if decision is None else decision.issuer,
            decision_reason=None if decision is None else decision.reason,
            decided_at=None if decision is None else decision.decided_at,
        )


class BridgeHeartbeatRequest(DomainModel):
    """Authenticated liveness refresh for one registered client."""

    fingerprint: Fingerprint


class BridgeClientRegistrationRequest(DomainModel):
    """Owner-scoped registration facts. No root or credential."""

    client_id: ClientId
    workspace_id: WorkspaceId
    fingerprint: Fingerprint
    capabilities: ClientCapabilityDescriptor


class BridgeClaimRequest(DomainModel):
    """Client-safe claim input; owner and lease window are server-derived."""

    call_id: ToolCallId
    client_id: ClientId
    snapshot_id: SnapshotId | None = None


class BridgeClientIdentityRequest(DomainModel):
    """Exact assigned client identity for claim renew, release and result."""

    client_id: ClientId


_BRIDGE_RESULT_STATUSES = frozenset(
    {ToolCallStatus.SUCCEEDED, ToolCallStatus.FAILED, ToolCallStatus.CANCELLED}
)


class BridgeToolResultRequest(DomainModel):
    """A result observation that a Bridge client can prove locally."""

    dev_only: Literal[True]
    status: ToolCallStatus
    result: dict[str, JsonValue] | None = None
    output: str = Field(default="", max_length=MAX_TOOL_OUTPUT_BYTES)
    truncated: bool = False
    changed_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        max_length=MAX_CHANGED_PATHS,
    )
    exit_code: int | None = None
    error: ErrorInfo | None = None
    client_id: ClientId
    snapshot_id: SnapshotId | None = None
    round_id: RoundId | None = None

    @model_validator(mode="after")
    def _bridge_result_is_safe(self) -> BridgeToolResultRequest:
        if self.status not in _BRIDGE_RESULT_STATUSES:
            raise ValueError("Bridge may submit only SUCCEEDED, FAILED or CANCELLED results")
        if len(self.output.encode("utf-8")) > MAX_TOOL_OUTPUT_BYTES:
            raise ValueError("tool output exceeds the size limit")
        ToolResult(
            call_id="tc_bridge_validation",
            status=self.status,
            result=self.result,
            output=self.output,
            truncated=self.truncated,
            changed_paths=self.changed_paths,
            exit_code=self.exit_code,
            error=self.error,
            completed_at=utc_now(),
        )
        return self

    def to_tool_result(self, call_id: ToolCallId, *, completed_at: datetime) -> ToolResult:
        """Bind the URL-scoped call id and server timestamp to this candidate."""
        return ToolResult(
            call_id=call_id,
            status=self.status,
            result=self.result,
            output=self.output,
            truncated=self.truncated,
            changed_paths=self.changed_paths,
            exit_code=self.exit_code,
            error=self.error,
            completed_at=completed_at,
        )

    def fingerprint(self, call_id: ToolCallId) -> str:
        """Hash canonical public result facts, including the URL-scoped call."""
        payload = self.model_dump(mode="json")
        payload["call_id"] = call_id
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _bridge_forbidden(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=403, detail={"code": code, "message": message})


async def _assigned_bridge_client(
    request: Request,
    *,
    principal_id: str,
    client_id: str,
    workspace_id: str | None = None,
    tool_name: str | None = None,
    require_live: bool = True,
) -> RegisteredBridgeClient:
    try:
        client = await _store(request).get_bridge_client(
            client_id, principal_id=principal_id
        )
    except MissingEntityError as error:
        raise _not_found("bridge client is not available") from error
    if client.status is BridgeClientStatus.DISABLED:
        raise _bridge_forbidden(
            "bridge_client_disabled",
            "disabled Bridge client cannot claim",
        )
    if workspace_id is not None and client.workspace_id != workspace_id:
        raise _bridge_forbidden(
            "bridge_workspace_mismatch",
            "Bridge client workspace does not match the call",
        )
    if tool_name is not None and tool_name not in client.capabilities.tools:
        raise _bridge_forbidden(
            "bridge_capability_mismatch",
            "invalid capability cannot claim a tool or lease",
        )
    if require_live:
        live = _supervisor(request).is_bridge_client_eligible(
            client.client_id, fingerprint=client.capability_fingerprint
        )
        if not live:
            if client.last_seen_at is None:
                raise _bridge_forbidden(
                    "bridge_handshake_required",
                    "Bridge claim requires an accepted client handshake",
                )
            raise _bridge_forbidden(
                "bridge_client_offline",
                "offline or expired Bridge client cannot receive new claims",
            )
    return client


def _store(request: Request) -> SqliteStore:
    store = getattr(request.app.state, "store", None)
    if not isinstance(store, SqliteStore):
        raise RuntimeError("the application was built without a store")
    return store


def _controller(request: Request) -> RunController:
    controller = getattr(request.app.state, "controller", None)
    if not isinstance(controller, RunController):
        raise RuntimeError("the application was built without a run controller")
    return controller


def _supervisor(request: Request) -> RunSupervisor:
    supervisor = getattr(request.app.state, "supervisor", None)
    if not isinstance(supervisor, RunSupervisor):
        raise RuntimeError("the application was built without a run supervisor")
    return supervisor


def _coordinator(request: Request) -> BridgeAssignmentCoordinator:
    return BridgeAssignmentCoordinator(_store(request), _supervisor(request))


def _event_bus(request: Request) -> EventBus:
    event_bus = getattr(request.app.state, "event_bus", None)
    if not isinstance(event_bus, EventBus):
        raise RuntimeError("the application was built without an event bus")
    return event_bus


def _not_found(message: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": "not_found", "message": message},
    )


async def _session(request: Request, session_id: str, auth: Auth) -> Session:
    try:
        session = await _store(request).get_session(
            session_id, principal_id=auth.principal_id
        )
    except MissingEntityError as error:
        raise _not_found("session is not available") from error
    if session.status is not SessionStatus.ACTIVE:
        raise HTTPException(
            status_code=409,
            detail={"code": "session_not_active", "message": "session is not active"},
        )
    if session.expires_at is not None and session.expires_at <= utc_now():
        raise HTTPException(
            status_code=409,
            detail={"code": "session_expired", "message": "session has expired"},
        )
    return session


async def _run(
    request: Request, session_id: str, run_id: str, auth: Auth
) -> object:
    await _session(request, session_id, auth)
    try:
        return await _store(request).get_run_for_session(
            session_id, run_id, principal_id=auth.principal_id
        )
    except MissingEntityError as error:
        raise _not_found("run is not available in this session") from error


async def _envelope(request: Request, run_id: str) -> RunEnvelope:
    store = _store(request)
    run = await store.get_run(run_id)
    result = await assemble_run_result(store, run_id)
    return RunEnvelope(
        run_id=run.run_id,
        status=run.status,
        strategy=run.strategy,
        graph_version=run.graph_version,
        output_text=result.output_text,
        output_kind=result.output_kind,
        usage=run.usage,
        error=run.error,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
    )


def _cursor(last_event_id: str | None, after: int | None) -> int:
    """Prefer the SSE reconnect header, then the explicit run-scoped cursor."""
    if last_event_id is not None:
        try:
            resolved = int(last_event_id)
        except (TypeError, ValueError) as error:
            raise DomainValidationError(
                "Last-Event-ID must be an event sequence number",
                code=ErrorCode.INVALID_REQUEST,
                field="Last-Event-ID",
            ) from error
    elif after is not None:
        resolved = after
    else:
        resolved = 0
    if resolved < 0:
        raise DomainValidationError(
            "the event cursor must not be negative",
            code=ErrorCode.INVALID_REQUEST,
            field="Last-Event-ID" if last_event_id is not None else "after",
        )
    return resolved


async def _approval(
    request: Request,
    request_id: str,
    auth: Auth,
    *,
    session_id: str | None = None,
) -> ApprovalRequest:
    if session_id is not None:
        await _session(request, session_id, auth)
    try:
        approval = await _store(request).get_approval(
            request_id, owner_id=auth.principal_id
        )
        if session_id is not None:
            await _run(request, session_id, approval.run_id, auth)
        return approval
    except MissingEntityError as error:
        raise _not_found("approval is not available") from error


async def _approval_view(
    store: SqliteStore, approval: ApprovalRequest, *, owner_id: str
) -> ApprovalView:
    try:
        decision = await store.get_approval_decision(
            approval.request_id, owner_id=owner_id
        )
    except MissingEntityError:
        decision = None
    return ApprovalView.from_request(approval, decision)


async def _decide(
    request: Request,
    request_id: str,
    body: ApprovalDecisionRequest,
    auth: Auth,
    *,
    session_id: str | None = None,
) -> ApprovalView:
    approval = await _approval(request, request_id, auth, session_id=session_id)
    store = _store(request)
    try:
        existing = await store.get_approval_decision(
            request_id, owner_id=auth.principal_id
        )
    except MissingEntityError:
        existing = None
    if existing is not None:
        if existing.outcome is not body.outcome or existing.reason != body.reason:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "approval_decision_conflict",
                    "message": "approval decision is immutable and conflicts",
                },
            )
        decision = existing
    else:
        decision = await store.decide_approval(
            request_id,
            ApprovalDecision(
                approval_request_id=request_id,
                outcome=body.outcome,
                issuer=ApprovalIssuer.USER,
                reason=body.reason,
                decided_at=utc_now(),
            ),
            owner_id=auth.principal_id,
        )
    # A decision is a durable wake-up fact. The supervisor remains the only
    # component that may decide whether and how the run resumes.
    await _supervisor(request).enqueue(approval.run_id)
    return ApprovalView.from_request(approval, decision)


def create_router() -> APIRouter:
    """Build authenticated Native Agent routes."""
    router = APIRouter(prefix="/v1", tags=["agent"])

    @router.post(
        "/bridge/clients",
        response_model=RegisteredBridgeClient,
        status_code=201,
    )
    async def register_bridge_client(
        body: BridgeClientRegistrationRequest, request: Request, auth: Auth
    ) -> RegisteredBridgeClient:
        client = RegisteredBridgeClient(
            client_id=body.client_id,
            principal_id=auth.principal_id,
            workspace_id=body.workspace_id,
            capabilities=body.capabilities,
            capability_fingerprint=body.fingerprint,
        )
        try:
            return await _store(request).register_bridge_client(client)
        except MissingEntityError as error:
            raise _not_found("workspace is not available") from error
        except DuplicateEntityError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bridge_client_conflict",
                    "message": str(error),
                },
            ) from error

    @router.get("/bridge/clients", response_model=list[RegisteredBridgeClient])
    async def list_bridge_clients(
        request: Request, auth: Auth, workspace_id: str | None = Query(default=None)
    ) -> list[RegisteredBridgeClient]:
        return list(
            await _store(request).list_bridge_clients(
                principal_id=auth.principal_id,
                workspace_id=workspace_id,
            )
        )

    @router.get("/bridge/clients/{client_id}", response_model=RegisteredBridgeClient)
    async def get_bridge_client(
        client_id: str, request: Request, auth: Auth
    ) -> RegisteredBridgeClient:
        try:
            return await _store(request).get_bridge_client(
                client_id, principal_id=auth.principal_id
            )
        except MissingEntityError as error:
            raise _not_found("bridge client is not available") from error

    @router.post(
        "/bridge/clients/{client_id}/disable",
        response_model=RegisteredBridgeClient,
    )
    async def disable_bridge_client(
        client_id: str, request: Request, auth: Auth
    ) -> RegisteredBridgeClient:
        try:
            return await _store(request).disable_bridge_client(
                client_id, principal_id=auth.principal_id, at=utc_now()
            )
        except MissingEntityError as error:
            raise _not_found("bridge client is not available") from error
        except DuplicateEntityError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bridge_client_conflict",
                    "message": str(error),
                },
            ) from error

    @router.post(
        "/bridge/clients/{client_id}/heartbeat",
        response_model=BridgeHeartbeatView,
    )
    async def heartbeat_bridge_client(
        client_id: str,
        body: BridgeHeartbeatRequest,
        request: Request,
        auth: Auth,
    ) -> BridgeHeartbeatView:
        store = _store(request)
        try:
            await store.record_bridge_heartbeat(
                client_id,
                principal_id=auth.principal_id,
                fingerprint=body.fingerprint,
                at=utc_now(),
            )
        except MissingEntityError as error:
            raise _not_found("bridge client is not registered") from error
        except StateError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "bridge_heartbeat_rejected", "message": str(error)},
            ) from error
        _supervisor(request).record_bridge_heartbeat(
            client_id, fingerprint=body.fingerprint
        )
        observed_at = utc_now()
        return BridgeHeartbeatView(
            client_id=client_id,
            liveness=BridgeClientLiveness.LIVE,
            fingerprint=body.fingerprint,
            observed_at=observed_at,
            ttl_seconds=MAX_BRIDGE_HEARTBEAT_TTL_SECONDS,
            cadence_seconds=BRIDGE_HEARTBEAT_CADENCE_SECONDS,
            lease_seconds=MAX_BRIDGE_LEASE_SECONDS,
            lease_total_seconds=MAX_BRIDGE_LEASE_TOTAL_SECONDS,
            lease_renew_seconds=MAX_BRIDGE_LEASE_RENEW_SECONDS,
            max_renews=MAX_BRIDGE_LEASE_RENEWS,
        )

    @router.post(
        "/bridge/handshake",
        response_model=ClientHandshakeAcceptance,
        status_code=200,
    )
    async def handshake(
        body: ClientHandshakeRequest, request: Request, auth: Auth
    ) -> ClientHandshakeAcceptance:
        store = _store(request)
        workspace_id = body.workspace_id
        if workspace_id is not None:
            try:
                workspace = await store.get_workspace(
                    workspace_id, owner_id=auth.principal_id
                )
            except MissingEntityError as error:
                raise _not_found("workspace is not available") from error
            if workspace.status is not WorkspaceStatus.ACTIVE:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "workspace_not_active",
                        "message": "workspace is not active",
                    },
                )
        try:
            registered = await store.get_bridge_client(
                body.client_id, principal_id=auth.principal_id
            )
        except MissingEntityError as error:
            raise _not_found("bridge client is not registered") from error
        if registered.status is BridgeClientStatus.DISABLED:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bridge_client_disabled",
                    "message": "disabled Bridge client cannot handshake",
                },
            )
        if registered.capability_fingerprint != body.fingerprint:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bridge_client_conflict",
                    "message": "handshake fingerprint does not match the registered client",
                },
            )
        if workspace_id is not None and registered.workspace_id != workspace_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bridge_client_conflict",
                    "message": "handshake workspace does not match the registered client",
                },
            )
        if workspace_id is None:
            workspace_id = registered.workspace_id
        if registered.capabilities != body.capabilities:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bridge_client_conflict",
                    "message": "handshake capabilities do not match the registered client",
                },
            )
        acceptance = ClientHandshakeAcceptance(
            accepted=True,
            client_id=registered.client_id,
            protocol_version=body.protocol_version,
            fingerprint=registered.capability_fingerprint,
            principal_id=auth.principal_id,
            workspace_id=workspace_id,
        )
        try:
            await store.record_bridge_heartbeat(
                body.client_id,
                principal_id=auth.principal_id,
                fingerprint=body.fingerprint,
                at=utc_now(),
            )
        except StateError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "bridge_heartbeat_rejected", "message": str(error)},
            ) from error
        _supervisor(request).record_bridge_heartbeat(
            body.client_id, fingerprint=body.fingerprint
        )
        return acceptance

    @router.post(
        "/bridge/manifests",
        response_model=BridgeManifestAcceptance,
        status_code=201,
    )
    async def publish_bridge_manifest(
        body: BridgeManifestPublication, request: Request, auth: Auth
    ) -> BridgeManifestAcceptance:
        await _assigned_bridge_client(
            request,
            principal_id=auth.principal_id,
            client_id=body.client_id,
            workspace_id=body.workspace_id,
            require_live=True,
        )
        try:
            return await _store(request).publish_bridge_manifest(
                body, principal_id=auth.principal_id, published_at=utc_now()
            )
        except MissingEntityError as error:
            raise _not_found("bridge client or workspace is not available") from error
        except DuplicateEntityError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "bridge_manifest_conflict", "message": str(error)},
            ) from error
        except StateError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "stale_bridge_snapshot", "message": str(error)},
            ) from error

    @router.post("/sessions", response_model=Session, status_code=201)
    async def create_session(body: SessionCreateRequest, request: Request, auth: Auth) -> Session:
        store = _store(request)
        try:
            workspace = await store.get_workspace(
                body.workspace_id, owner_id=auth.principal_id
            )
        except MissingEntityError as error:
            raise _not_found("workspace is not available") from error
        if workspace.status is not WorkspaceStatus.ACTIVE:
            raise HTTPException(
                status_code=409,
                detail={"code": "workspace_not_active", "message": "workspace is not active"},
            )
        created_at = utc_now()
        session = Session(
            session_id=new_session_id(),
            principal_id=auth.principal_id,
            workspace_id=body.workspace_id,
            grant=WorkspaceGrant(
                principal_id=auth.principal_id,
                workspace_id=body.workspace_id,
                access=body.access,
                expires_at=body.expires_at,
            ),
            agent_options=body.agent_options,
            created_at=created_at,
            expires_at=body.expires_at,
        )
        return await store.create_session(session)

    @router.get("/sessions/{session_id}", response_model=Session)
    async def get_session(session_id: str, request: Request, auth: Auth) -> Session:
        return await _session(request, session_id, auth)

    @router.post("/sessions/{session_id}/runs", response_model=RunEnvelope, status_code=202)
    async def create_run(
        session_id: str, body: NativeRunRequest, request: Request, auth: Auth
    ) -> RunEnvelope:
        session = await _session(request, session_id, auth)
        settings = request.app.state.settings
        if len(body.input) > settings.max_input_chars:
            raise DomainValidationError(
                f"input exceeds the {settings.max_input_chars} character limit",
                code=ErrorCode.INPUT_TOO_LARGE,
                field="input",
            )
        scoped_body = body.model_copy(update={"agent_options": session.agent_options})
        controller = _controller(request)
        created = await controller.create_run(scoped_body)
        await _store(request).attach_run_to_session(
            session_id, created.run_id, principal_id=auth.principal_id
        )
        await _supervisor(request).enqueue(created.run_id)
        return RunEnvelope(
            run_id=created.run_id,
            status=created.status,
            strategy=created.strategy,
            graph_version=created.graph_version,
            usage=created.usage,
            error=created.error,
            created_at=created.created_at,
            started_at=created.started_at,
            completed_at=created.completed_at,
        )

    @router.get("/sessions/{session_id}/runs/{run_id}", response_model=RunEnvelope)
    async def get_run(session_id: str, run_id: str, request: Request, auth: Auth) -> RunEnvelope:
        await _run(request, session_id, run_id, auth)
        return await _envelope(request, run_id)

    @router.get(
        "/sessions/{session_id}/runs/{run_id}/bundle",
        response_model=CloudResultBundle,
    )
    async def get_cloud_bundle(
        session_id: str, run_id: str, request: Request, auth: Auth
    ) -> CloudResultBundle:
        session = await _session(request, session_id, auth)
        run = await _run(request, session_id, run_id, auth)
        if (
            session.agent_options.execution_location is not ExecutionLocation.CLOUD
            or run.request.agent_options.execution_location is not ExecutionLocation.CLOUD
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bundle_not_cloud",
                    "message": "CLOUD export is only available for CLOUD runs",
                },
            )
        store = _store(request)
        settings = request.app.state.settings
        if run.status is not RunStatus.SUCCEEDED:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bundle_unverified",
                    "message": "only a succeeded run can export a verified bundle",
                },
            )
        snapshots = await store.list_snapshots(
            session.workspace_id, owner_id=auth.principal_id
        )
        rounds = await store.list_rounds(run.run_id)
        verified = tuple(
            round_ for round_ in rounds if round_.status is RoundStatus.VERIFIED
        )
        change_set_ids = () if not verified else verified[-1].change_set_ids
        evidence_ids = () if not verified else verified[-1].evidence_ids
        verified_snapshot_id = None if not verified else verified[-1].merged_snapshot_id
        if verified_snapshot_id is None:
            ledgers = await store.list_merge_ledgers(
                run_id=run.run_id,
                statuses=(MergeLedgerStatus.PROMOTED, MergeLedgerStatus.MERGED),
            )
            if ledgers:
                verified_snapshot_id = ledgers[-1].merged_snapshot_id
        try:
            snapshot = select_verified_cloud_snapshot(
                run=run,
                session=session,
                snapshots=snapshots,
                verified_snapshot_id=verified_snapshot_id,
            )
        except CloudBundleError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": error.code, "message": error.message},
            ) from error
        if run.strategy is ExecutionStrategy.PROGRESSIVE and not evidence_ids:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bundle_unverified",
                    "message": "Progressive export requires verified evidence",
                },
            )
        try:
            manifest = await store.get_snapshot_manifest(
                snapshot.snapshot_id, owner_id=auth.principal_id
            )
            contents = await store.get_snapshot_file_contents(
                snapshot.snapshot_id, owner_id=auth.principal_id
            )
            selection = select_snapshot_export_files(
                manifest,
                contents,
                max_files=settings.export_max_files,
                max_bytes=settings.export_max_bytes,
                max_nesting=settings.export_max_nesting,
            )
            return assemble_cloud_bundle(
                run=run,
                session=session,
                snapshot=snapshot,
                selection=selection,
                change_set_ids=change_set_ids,
                evidence_ids=evidence_ids,
            )
        except CloudBundleError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": error.code, "message": error.message},
            ) from error
        except WorkspaceBackendError as error:
            message = str(error)
            code = (
                "bundle_oversize"
                if "exceeds" in message
                else "bundle_stale"
                if "stale" in message
                else "bundle_rejected"
            )
            raise HTTPException(
                status_code=409,
                detail={"code": code, "message": message},
            ) from error

    @router.post("/sessions/{session_id}/runs/{run_id}/cancel", response_model=RunEnvelope)
    async def cancel_run(session_id: str, run_id: str, request: Request, auth: Auth) -> RunEnvelope:
        await _run(request, session_id, run_id, auth)
        await _controller(request).cancel(run_id)
        await _supervisor(request).enqueue(run_id)
        return await _envelope(request, run_id)

    @router.get("/sessions/{session_id}/runs/{run_id}/tool-calls", response_model=list[ToolCall])
    async def list_tool_calls(
        session_id: str,
        run_id: str,
        request: Request,
        auth: Auth,
        status: ToolCallStatus | None = Query(default=None),
    ) -> list[ToolCall]:
        await _run(request, session_id, run_id, auth)
        statuses = None if status is None else [status]
        return list(
            await _store(request).list_tool_calls_for_session(
                session_id,
                run_id,
                principal_id=auth.principal_id,
                statuses=statuses,
            )
        )

    @router.get(
        "/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}",
        response_model=ToolCall,
    )
    async def get_tool_call(
        session_id: str, run_id: str, call_id: str, request: Request, auth: Auth
    ) -> ToolCall:
        await _run(request, session_id, run_id, auth)
        try:
            return await _store(request).get_tool_call_for_session(
                session_id, run_id, call_id, principal_id=auth.principal_id
            )
        except MissingEntityError as error:
            raise _not_found("tool call is not available") from error

    @router.post(
        "/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim",
        response_model=BridgeClaim,
        status_code=201,
    )
    async def claim_tool_call(
        session_id: str,
        run_id: str,
        call_id: str,
        body: BridgeClaimRequest,
        request: Request,
        auth: Auth,
        idempotency_key: str = Header(..., alias="Idempotency-Key"),
    ) -> BridgeClaim:
        session = await _session(request, session_id, auth)
        if body.call_id != call_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "claim_scope_mismatch",
                    "message": "claim body call_id does not match the route",
                },
            )
        client = await _assigned_bridge_client(
            request,
            principal_id=auth.principal_id,
            client_id=body.client_id,
            workspace_id=session.workspace_id,
        )
        await _run(request, session_id, run_id, auth)
        try:
            current_call = await _store(request).get_tool_call_for_session(
                session_id, run_id, call_id, principal_id=auth.principal_id
            )
        except MissingEntityError as error:
            raise _not_found("tool call is not available for Bridge claim") from error
        if current_call.tool_name not in client.capabilities.tools:
            raise _bridge_forbidden(
                "bridge_capability_mismatch",
                "invalid capability cannot claim a tool or lease",
            )
        if current_call.effect in {ToolEffect.WRITE, ToolEffect.COMMAND}:
            if (
                body.snapshot_id is not None
                and body.snapshot_id != current_call.snapshot_id
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "stale_bridge_snapshot",
                        "message": "mutating Bridge claim requires the immutable base snapshot",
                    },
                )
            try:
                snapshot = await _store(request).get_snapshot(
                    current_call.snapshot_id, owner_id=auth.principal_id
                )
            except MissingEntityError as error:
                raise _not_found("Bridge snapshot is not available") from error
            if snapshot.status is not SnapshotStatus.READY:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "stale_bridge_snapshot",
                        "message": "mutating Bridge claim requires a ready base snapshot",
                    },
                )
        if (
            not idempotency_key
            or idempotency_key != idempotency_key.strip()
            or len(idempotency_key) > 128
        ):
            raise DomainValidationError(
                "Idempotency-Key must be 1 to 128 non-whitespace characters",
                code=ErrorCode.INVALID_REQUEST,
                field="Idempotency-Key",
            )
        try:
            return await _store(request).claim_tool_call(
                session_id,
                run_id,
                call_id,
                principal_id=auth.principal_id,
                client_id=client.client_id,
                idempotency_key=idempotency_key,
                claimed_at=utc_now(),
            )
        except MissingEntityError as error:
            raise _not_found("tool call is not available for Bridge claim") from error
        except DuplicateEntityError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bridge_claim_conflict",
                    "message": str(error),
                },
            ) from error
        except StateError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bridge_claim_unavailable",
                    "message": str(error),
                },
            ) from error

    @router.post(
        "/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim/renew",
        response_model=BridgeClaim,
    )
    async def renew_tool_claim(
        session_id: str,
        run_id: str,
        call_id: str,
        body: BridgeClientIdentityRequest,
        request: Request,
        auth: Auth,
    ) -> BridgeClaim:
        session = await _session(request, session_id, auth)
        await _run(request, session_id, run_id, auth)
        await _assigned_bridge_client(
            request,
            principal_id=auth.principal_id,
            client_id=body.client_id,
            workspace_id=session.workspace_id,
            require_live=True,
        )
        try:
            current = await _store(request).get_bridge_claim_for_session(
                session_id, run_id, call_id, principal_id=auth.principal_id
            )
            if current.client_id != body.client_id:
                raise _bridge_forbidden(
                    "bridge_client_mismatch",
                    "only the assigned Bridge client can renew this claim",
                )
            return await _store(request).renew_bridge_claim(
                current.claim_id, principal_id=auth.principal_id, at=utc_now()
            )
        except MissingEntityError as error:
            raise _not_found("Bridge claim is not available") from error
        except (DuplicateEntityError, StateError) as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "bridge_claim_unavailable", "message": str(error)},
            ) from error

    @router.post(
        "/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/claim/release",
        response_model=BridgeClaim,
    )
    async def release_tool_claim(
        session_id: str,
        run_id: str,
        call_id: str,
        body: BridgeClientIdentityRequest,
        request: Request,
        auth: Auth,
    ) -> BridgeClaim:
        session = await _session(request, session_id, auth)
        await _run(request, session_id, run_id, auth)
        await _assigned_bridge_client(
            request,
            principal_id=auth.principal_id,
            client_id=body.client_id,
            workspace_id=session.workspace_id,
            require_live=False,
        )
        try:
            current = await _store(request).get_bridge_claim_for_session(
                session_id, run_id, call_id, principal_id=auth.principal_id
            )
            if current.client_id != body.client_id:
                raise _bridge_forbidden(
                    "bridge_client_mismatch",
                    "only the assigned Bridge client can release this claim",
                )
            return await _store(request).release_bridge_claim(
                current.claim_id, principal_id=auth.principal_id, at=utc_now()
            )
        except MissingEntityError as error:
            raise _not_found("Bridge claim is not available") from error
        except (DuplicateEntityError, StateError) as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "bridge_claim_unavailable", "message": str(error)},
            ) from error

    @router.post(
        "/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/result",
        response_model=ToolResult,
    )
    async def submit_tool_result(
        session_id: str,
        run_id: str,
        call_id: str,
        body: BridgeToolResultRequest,
        request: Request,
        auth: Auth,
        idempotency_key: str = Header(..., alias="Idempotency-Key"),
    ) -> ToolResult:
        session = await _session(request, session_id, auth)
        await _run(request, session_id, run_id, auth)
        await _assigned_bridge_client(
            request,
            principal_id=auth.principal_id,
            client_id=body.client_id,
            workspace_id=session.workspace_id,
            require_live=False,
        )
        try:
            current = await _store(request).get_bridge_claim_for_session(
                session_id, run_id, call_id, principal_id=auth.principal_id
            )
        except MissingEntityError as error:
            raise _not_found("Bridge claim is not available") from error
        if current.client_id != body.client_id:
            raise _bridge_forbidden(
                "bridge_client_mismatch",
                "only the assigned Bridge client can submit this result",
            )
        if not idempotency_key or idempotency_key != idempotency_key.strip():
            raise DomainValidationError(
                "Idempotency-Key must be 1 to 128 non-whitespace characters",
                code=ErrorCode.INVALID_REQUEST,
                field="Idempotency-Key",
            )
        if len(idempotency_key) > 128:
            raise DomainValidationError(
                "Idempotency-Key must be 1 to 128 non-whitespace characters",
                code=ErrorCode.INVALID_REQUEST,
                field="Idempotency-Key",
            )
        candidate = body.to_tool_result(call_id, completed_at=utc_now())
        try:
            completed, replayed = await _store(request).submit_bridge_tool_result(
                session_id,
                run_id,
                call_id,
                candidate,
                principal_id=auth.principal_id,
                client_id=body.client_id,
                snapshot_id=body.snapshot_id,
                round_id=body.round_id,
            )
        except MissingEntityError as error:
            raise _not_found("Bridge claim is not available") from error
        except (DuplicateEntityError, StateError) as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "bridge_result_conflict",
                    "message": str(error),
                },
            ) from error
        if not replayed:
            await _coordinator(request).wake_after_settlement(run_id, replayed=replayed)
        return completed

    @router.get(
        "/sessions/{session_id}/runs/{run_id}/tool-calls/{call_id}/result",
        response_model=ToolResult,
    )
    async def get_tool_result(
        session_id: str, run_id: str, call_id: str, request: Request, auth: Auth
    ) -> ToolResult:
        await _run(request, session_id, run_id, auth)
        try:
            return await _store(request).get_tool_result_for_session(
                session_id, run_id, call_id, principal_id=auth.principal_id
            )
        except MissingEntityError as error:
            raise _not_found("tool result is not available") from error

    @router.get("/sessions/{session_id}/approvals", response_model=list[ApprovalView])
    async def list_session_approvals(
        session_id: str,
        request: Request,
        auth: Auth,
        run_id: str | None = Query(default=None),
    ) -> list[ApprovalView]:
        await _session(request, session_id, auth)
        if run_id is not None:
            await _run(request, session_id, run_id, auth)
        approvals = await _store(request).list_approvals(
            owner_id=auth.principal_id, run_id=run_id
        )
        return [
            await _approval_view(
                _store(request), approval, owner_id=auth.principal_id
            )
            for approval in approvals
        ]

    @router.get("/approvals", response_model=list[ApprovalView])
    async def list_approvals(
        request: Request,
        auth: Auth,
        run_id: str | None = Query(default=None),
        call_id: str | None = Query(default=None),
    ) -> list[ApprovalView]:
        approvals = await _store(request).list_approvals(
            owner_id=auth.principal_id, run_id=run_id, call_id=call_id
        )
        return [
            await _approval_view(
                _store(request), approval, owner_id=auth.principal_id
            )
            for approval in approvals
        ]

    @router.get("/approvals/{request_id}", response_model=ApprovalView)
    async def get_approval(request_id: str, request: Request, auth: Auth) -> ApprovalView:
        approval = await _approval(request, request_id, auth)
        return await _approval_view(
            _store(request), approval, owner_id=auth.principal_id
        )

    @router.post("/approvals/{request_id}/decision", response_model=ApprovalView)
    async def decide_approval(
        request_id: str,
        body: ApprovalDecisionRequest,
        request: Request,
        auth: Auth,
    ) -> ApprovalView:
        return await _decide(request, request_id, body, auth)

    @router.post(
        "/sessions/{session_id}/approvals/{request_id}/decision",
        response_model=ApprovalView,
    )
    async def decide_session_approval(
        session_id: str,
        request_id: str,
        body: ApprovalDecisionRequest,
        request: Request,
        auth: Auth,
    ) -> ApprovalView:
        return await _decide(
            request, request_id, body, auth, session_id=session_id
        )

    @router.get("/sessions/{session_id}/runs/{run_id}/events")
    async def stream_events(
        session_id: str,
        run_id: str,
        request: Request,
        auth: Auth,
        after: int | None = Query(default=None, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> Response:
        await _run(request, session_id, run_id, auth)
        cursor = _cursor(last_event_id, after)
        event_bus = _event_bus(request)
        subscription = await event_bus.subscribe(run_id)
        store = _store(request)

        async def body() -> AsyncIterator[bytes]:
            stream_cursor = cursor
            try:
                while True:
                    events = await store.list_events(
                        run_id, after_sequence=stream_cursor, limit=_MAX_EVENT_PAGE
                    )
                    for event in events:
                        envelope = RunEventEnvelope(
                            run_id=event.run_id,
                            sequence=event.sequence,
                            event_type=event.event_type.value,
                            timestamp=event.timestamp,
                            payload=dict(event.payload),
                        )
                        data = json.dumps(
                            envelope.model_dump(mode="json"), sort_keys=True
                        )
                        stream_cursor = event.sequence
                        yield (
                            f"id: {event.sequence}\n"
                            f"event: {event.event_type.value}\n"
                            f"data: {data}\n\n"
                        ).encode()
                    if len(events) == _MAX_EVENT_PAGE:
                        continue
                    current = await store.get_run(run_id)
                    if current.status.is_terminal:
                        return
                    hint = await subscription.get()
                    if hint is None:
                        return
            finally:
                await subscription.close()

        return StreamingResponse(
            body(),
            media_type=EVENT_STREAM_MEDIA_TYPE,
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return router
