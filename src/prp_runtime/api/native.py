"""PRP Native run API.

The native binding is a thin shell over the domain: it accepts a
``NativeRunRequest``, hands it to the controller, and reads persisted facts back
out. It never holds run state of its own, so a disconnected client never loses a
run.
"""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import ConfigDict, Field

from prp_runtime.control.controller import RunController
from prp_runtime.control.routing import facts_from_request
from prp_runtime.domain.enums import ExecutionStrategy, RunStatus
from prp_runtime.domain.errors import DomainValidationError, ErrorCode
from prp_runtime.domain.models import (
    ArtifactKind,
    DomainModel,
    ErrorInfo,
    NativeRunRequest,
    Usage,
)
from prp_runtime.domain.values import RunId, UtcTimestamp
from prp_runtime.runtime.assembler import assemble_run_result
from prp_runtime.storage.sqlite import SqliteStore

__all__ = ["EVENT_STREAM_MEDIA_TYPE", "RunEnvelope", "RunEventEnvelope", "create_router"]

EVENT_STREAM_MEDIA_TYPE = "text/event-stream"

_MAX_EVENT_PAGE = 500


class RunEnvelope(DomainModel):
    """The client view of one run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: RunId
    status: RunStatus
    strategy: ExecutionStrategy | None = None
    graph_version: int = Field(default=1, ge=1)
    output_text: str | None = None
    output_kind: ArtifactKind | None = None
    usage: Usage = Usage()
    error: ErrorInfo | None = None
    created_at: UtcTimestamp
    started_at: UtcTimestamp | None = None
    completed_at: UtcTimestamp | None = None


class RunEventEnvelope(DomainModel):
    """The client view of one ledger entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: RunId
    sequence: int = Field(ge=1)
    event_type: str
    timestamp: UtcTimestamp
    payload: dict[str, object] = Field(default_factory=dict)


def _controller(request: Request) -> RunController:
    controller = getattr(request.app.state, "controller", None)
    if not isinstance(controller, RunController):
        raise RuntimeError("the application was built without a run controller")
    return controller


def _store(request: Request) -> SqliteStore:
    store = getattr(request.app.state, "store", None)
    if not isinstance(store, SqliteStore):
        raise RuntimeError("the application was built without a store")
    return store


async def _envelope(store: SqliteStore, run_id: str) -> RunEnvelope:
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


def _resolve_cursor(last_event_id: str | None, after: int | None) -> int:
    """Prefer the SSE reconnect header, then the explicit cursor."""
    if last_event_id is not None:
        try:
            resolved = int(last_event_id)
        except ValueError as error:
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
            field="after",
        )
    return resolved


def create_router() -> APIRouter:
    """Build the native run router."""
    router = APIRouter(prefix="/v1/runs", tags=["runs"])

    @router.post("", response_model=RunEnvelope, status_code=201)
    async def create_run(request: Request, body: NativeRunRequest) -> RunEnvelope:
        """Create a run and execute it to a terminal status.

        The run is persisted before execution starts, so it survives a client
        disconnect and stays queryable afterwards.
        """
        settings = request.app.state.settings
        if len(body.input) > settings.max_input_chars:
            raise DomainValidationError(
                f"input exceeds the {settings.max_input_chars} character limit",
                code=ErrorCode.INPUT_TOO_LARGE,
                field="input",
            )
        controller = _controller(request)
        created = await controller.create_run(body)
        await controller.execute(
            created.run_id,
            routing_facts=facts_from_request(body),
        )
        return await _envelope(_store(request), created.run_id)

    @router.get("/{run_id}", response_model=RunEnvelope)
    async def get_run(request: Request, run_id: str) -> RunEnvelope:
        """Read one run. A missing run is a 404 with a stable code."""
        return await _envelope(_store(request), run_id)

    @router.post("/{run_id}/cancel", response_model=RunEnvelope)
    async def cancel_run(request: Request, run_id: str) -> RunEnvelope:
        """Request cancellation. Cancelling a terminal run changes nothing."""
        controller = _controller(request)
        await controller.cancel(run_id)
        return await _envelope(_store(request), run_id)

    @router.get("/{run_id}/events")
    async def stream_events(
        request: Request,
        run_id: str,
        after: int | None = Query(default=None, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> Response:
        """Replay the run's ledger as Server-Sent Events.

        The stream is a replay from the store, not a subscription to a live
        connection: a client that drops reconnects with ``Last-Event-ID`` (or
        ``?after=``) and continues from exactly where it stopped.
        """
        cursor = _resolve_cursor(last_event_id, after)
        store = _store(request)
        # Read the run first so an unknown id is a structured 404 rather than an
        # empty stream.
        await store.get_run(run_id)
        events = await store.list_events(run_id, after_sequence=cursor, limit=_MAX_EVENT_PAGE)

        async def body() -> AsyncIterator[bytes]:
            for event in events:
                envelope = RunEventEnvelope(
                    run_id=event.run_id,
                    sequence=event.sequence,
                    event_type=event.event_type.value,
                    timestamp=event.timestamp,
                    payload=dict(event.payload),
                )
                data = json.dumps(envelope.model_dump(mode="json"), sort_keys=True)
                yield (
                    f"id: {event.sequence}\n"
                    f"event: {event.event_type.value}\n"
                    f"data: {data}\n\n"
                ).encode()

        return StreamingResponse(
            body(),
            media_type=EVENT_STREAM_MEDIA_TYPE,
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return router
