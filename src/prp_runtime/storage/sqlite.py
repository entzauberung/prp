"""SQLite store: schema, connection lifecycle and the current operation set.

There is exactly one schema and no migration path. A database created by a
different schema version is rejected with an instruction to delete the
development database.

Every mutating method joins the caller's transaction when one is open and
commits on its own when it is the outermost call, so a state change and its
events are never half committed. All JSON columns pass through the domain
models on the way in and on the way out.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self, SupportsInt

import aiosqlite
from pydantic import JsonValue

from prp_runtime.domain.enums import (
    AttemptStatus,
    BridgeClaimStatus,
    ExecutionStrategy,
    MergeLedgerStatus,
    ModelRole,
    ReservationStatus,
    ResourceAccess,
    RunStatus,
    ToolCallStatus,
    ToolEffect,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import ErrorCode, InternalError, StateError
from prp_runtime.domain.events import (
    EventType,
    RunEvent,
    next_sequence,
    payload_from_agent_history,
    payload_from_bridge_claim,
    payload_from_merge_ledger,
    payload_from_tool_call,
    payload_from_tool_result,
)
from prp_runtime.domain.models import (
    AgentHistoryRecord,
    AgentRequestOptions,
    Artifact,
    ArtifactKind,
    Attempt,
    AttemptCost,
    ErrorCategory,
    ErrorInfo,
    Evidence,
    EvidenceKind,
    ExecutionScope,
    MergeLedger,
    NativeRunRequest,
    OutputRequirement,
    Run,
    RunMetrics,
    Session,
    SessionStatus,
    Usage,
    VerificationResult,
    WorkspaceGrant,
    WorkUnit,
)
from prp_runtime.domain.transitions import transition_merge
from prp_runtime.domain.values import ModelRef, ResourceClaim, new_reservation_id, utc_now
from prp_runtime.json_support import strict_json_loads
from prp_runtime.policy.models import (
    ApprovalDecision,
    ApprovalIssuer,
    ApprovalOutcome,
    ApprovalRequest,
    CapabilityScope,
    Lease,
    LeaseStatus,
)
from prp_runtime.tools.models import (
    BridgeClaim,
    ToolCall,
    ToolResult,
    validate_tool_rejection_reason,
)
from prp_runtime.workspace.changes import (
    ChangeSet,
    FileChange,
    FileChangeAction,
    FileContent,
    Patch,
)
from prp_runtime.workspace.models import (
    Snapshot,
    SnapshotEntry,
    SnapshotEntryType,
    SnapshotManifest,
    SnapshotStatus,
    Workspace,
    WorkspaceSource,
    WorkspaceSourceType,
    WorkspaceStatus,
)

if TYPE_CHECKING:
    from prp_runtime.control.progressive import ProgressiveRound
    from prp_runtime.control.reservations import Reservation, ReservationRequest

__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "DanglingReferenceError",
    "DuplicateEntityError",
    "IncompatibleSchemaError",
    "MissingEntityError",
    "SequenceConflictError",
    "SqliteStore",
]

SCHEMA_VERSION = 11

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_DEFAULT_BUSY_TIMEOUT_MS = 5_000

_MAX_SEQUENCE_RETRIES = 8

_TOOL_EVENT_BY_STATUS: Mapping[ToolCallStatus, EventType] = {
    ToolCallStatus.REQUESTED: EventType.TOOL_CALL_REQUESTED,
    ToolCallStatus.AWAITING_APPROVAL: EventType.TOOL_CALL_AWAITING_APPROVAL,
    ToolCallStatus.RUNNING: EventType.TOOL_CALL_STARTED,
    ToolCallStatus.SUCCEEDED: EventType.TOOL_CALL_SUCCEEDED,
    ToolCallStatus.FAILED: EventType.TOOL_CALL_FAILED,
    ToolCallStatus.CANCELLED: EventType.TOOL_CALL_CANCELLED,
    ToolCallStatus.REJECTED: EventType.TOOL_CALL_REJECTED,
    ToolCallStatus.INTERRUPTED: EventType.TOOL_CALL_INTERRUPTED,
    ToolCallStatus.UNKNOWN: EventType.TOOL_CALL_UNKNOWN,
}
_MERGE_EVENT_BY_STATUS: Mapping[MergeLedgerStatus, EventType] = {
    MergeLedgerStatus.PLANNED: EventType.MERGE_PLANNED,
    MergeLedgerStatus.RUNNING: EventType.MERGE_STARTED,
    MergeLedgerStatus.MERGED: EventType.MERGE_MERGED,
    MergeLedgerStatus.PROMOTED: EventType.MERGE_PROMOTED,
    MergeLedgerStatus.CONFLICT: EventType.MERGE_CONFLICT,
    MergeLedgerStatus.UNKNOWN: EventType.MERGE_UNKNOWN,
}
_POLICY_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|token|password|secret|credential|authorization)\b"
)


class EventBusProtocol(Protocol):
    async def publish(self, run_id: str, sequence: int) -> None:
        """Publish one committed sequence hint."""


class IncompatibleSchemaError(InternalError):
    """The database on disk was created by a different schema version."""


class DuplicateEntityError(StateError):
    """An entity with the same identifier is already persisted."""


class MissingEntityError(StateError):
    """The requested entity is not persisted."""


class DanglingReferenceError(StateError):
    """A write referenced an entity that does not exist."""


class SequenceConflictError(StateError):
    """A run's next event sequence could not be allocated."""


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _require_iso(value: datetime) -> str:
    return value.isoformat()


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value))


def _require_datetime(value: object) -> datetime:
    parsed = _parse_datetime(value)
    if parsed is None:
        raise InternalError("expected a stored timestamp but found NULL")
    return parsed


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: SupportsInt | str | None) -> int | None:
    return None if value is None else int(value)


def _optional_decimal(value: object) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _error_info(category: object, message: object) -> ErrorInfo | None:
    if category is None:
        return None
    if message is None:
        raise InternalError("stored error has a category but no message")
    return ErrorInfo(category=ErrorCategory(str(category)), message=str(message))


def _safe_policy_text(value: str, field: str) -> str:
    if _POLICY_SECRET_RE.search(value):
        raise ValueError(f"{field} must not contain secret material")
    return value


def _scope_json(scope: CapabilityScope) -> str:
    return json.dumps(scope.model_dump(mode="json"), sort_keys=True, allow_nan=False)


def _scope_from_json(value: object) -> CapabilityScope:
    parsed = strict_json_loads(str(value))
    if not isinstance(parsed, dict):
        raise InternalError("stored capability scope is not a JSON object")
    return CapabilityScope.model_validate(parsed)


def _bridge_claim_fingerprint(
    *,
    call_id: str,
    run_id: str,
    session_id: str,
    workspace_id: str,
    owner_id: str,
    claimant_id: str,
    idempotency_key: str,
) -> str:
    """Hash only public claim identity; never persist a bearer secret."""
    material = json.dumps(
        {
            "call_id": call_id,
            "run_id": run_id,
            "session_id": session_id,
            "workspace_id": workspace_id,
            "owner_id": owner_id,
            "claimant_id": claimant_id,
            "idempotency_key": idempotency_key,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _bridge_results_match(existing: ToolResult, candidate: ToolResult) -> bool:
    """Compare client-proven result facts without trusting server timestamps."""
    return existing.model_copy(update={"completed_at": candidate.completed_at}) == candidate


def _usage_or_none(row: aiosqlite.Row) -> Usage | None:
    if row["usage_input_tokens"] is None:
        return None
    return Usage(
        input_tokens=int(row["usage_input_tokens"]),
        output_tokens=int(row["usage_output_tokens"] or 0),
        strong_model_tokens=int(row["usage_strong_model_tokens"] or 0),
        elapsed_ms=int(row["usage_elapsed_ms"] or 0),
    )


def _translate_integrity_error(error: sqlite3.IntegrityError, entity: str) -> StateError:
    message = str(error)
    if "FOREIGN KEY" in message:
        return DanglingReferenceError(
            f"{entity} references an entity that does not exist",
            code=ErrorCode.ILLEGAL_STATE_TRANSITION,
        )
    if "UNIQUE" in message or "PRIMARY KEY" in message:
        return DuplicateEntityError(
            f"{entity} is already persisted", code=ErrorCode.ILLEGAL_STATE_TRANSITION
        )
    return StateError(
        f"{entity} violates a database constraint: {message}",
        code=ErrorCode.ILLEGAL_STATE_TRANSITION,
    )


class SqliteStore:
    """Owns one SQLite connection and the current schema.

    Opening is idempotent, closing is idempotent, and no connection is created at
    import time.
    """

    def __init__(
        self,
        database_path: Path,
        *,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
        event_bus: EventBusProtocol | None = None,
    ) -> None:
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        self._database_path = database_path
        self._busy_timeout_ms = busy_timeout_ms
        self._connection: aiosqlite.Connection | None = None
        self._transaction_depth = 0
        self._capacity_limits: dict[str, int] = {}
        self._event_bus = event_bus
        self._pending_event_hints: list[tuple[str, int]] = []

    @property
    def database_path(self) -> Path:
        return self._database_path

    @property
    def is_open(self) -> bool:
        return self._connection is not None

    @property
    def connection(self) -> aiosqlite.Connection:
        """The live connection. Raises when the store is closed."""
        if self._connection is None:
            raise InternalError("sqlite store is not open")
        return self._connection

    async def open(self) -> None:
        """Connect, apply pragmas and ensure the current schema is present."""
        if self._connection is not None:
            return
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self._database_path)
        connection.row_factory = aiosqlite.Row
        try:
            await self._apply_pragmas(connection)
            await self._ensure_schema(connection)
        except BaseException:
            await connection.close()
            raise
        self._connection = connection

    async def close(self) -> None:
        """Close the connection if one is open."""
        connection, self._connection = self._connection, None
        self._transaction_depth = 0
        if connection is not None:
            await connection.close()

    def set_event_bus(self, event_bus: EventBusProtocol | None) -> None:
        """Attach the process-local hint bus used after committed events."""
        self._event_bus = event_bus

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Run a unit of work so state and events commit together.

        Nesting joins the outermost transaction: the commit or rollback happens
        exactly once, when the outermost block leaves.
        """
        connection = self.connection
        outermost = self._transaction_depth == 0
        if outermost:
            self._pending_event_hints.clear()
        self._transaction_depth += 1
        try:
            yield connection
        except BaseException:
            self._transaction_depth -= 1
            if self._transaction_depth == 0:
                await connection.rollback()
                self._pending_event_hints.clear()
            raise
        self._transaction_depth -= 1
        if self._transaction_depth == 0:
            try:
                await connection.commit()
            except BaseException:
                self._pending_event_hints.clear()
                raise
            hints = tuple(self._pending_event_hints)
            self._pending_event_hints.clear()
            if self._event_bus is not None:
                for run_id, sequence in hints:
                    try:
                        await self._event_bus.publish(run_id, sequence)
                    except Exception:
                        # Hints are best-effort; the committed Store remains truth.
                        continue

    @property
    def in_transaction(self) -> bool:
        return self._transaction_depth > 0

    def register_capacity_limit(self, capacity_key: str, limit: int) -> None:
        """Register a server-side capacity limit used by atomic reservation holds."""
        if not capacity_key.strip() or limit < 1:
            raise ValueError("capacity key must be non-blank and limit must be positive")
        self._capacity_limits[capacity_key] = limit

    async def _apply_pragmas(self, connection: aiosqlite.Connection) -> None:
        await connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.execute("PRAGMA synchronous = NORMAL")
        async with connection.execute("PRAGMA journal_mode = WAL") as cursor:
            await cursor.fetchone()

    async def _ensure_schema(self, connection: aiosqlite.Connection) -> None:
        version = await self._read_user_version(connection)
        if version == SCHEMA_VERSION:
            return
        if version != 0:
            raise IncompatibleSchemaError(
                f"database schema version {version} does not match the current "
                f"version {SCHEMA_VERSION}; delete the development database at "
                f"{self._database_path.name} and start again"
            )
        await connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        await connection.commit()

    @staticmethod
    async def _read_user_version(connection: aiosqlite.Connection) -> int:
        async with connection.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise InternalError("sqlite did not report a user_version")
        return int(row[0])

    # --- low level helpers ------------------------------------------------------

    async def _write(self, sql: str, params: Mapping[str, Any], entity: str) -> int:
        """Execute one write inside the current or a new transaction."""
        async with self.transaction() as connection:
            try:
                cursor = await connection.execute(sql, params)
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, entity) from error
            return cursor.rowcount

    async def _fetch_one(
        self, sql: str, params: Mapping[str, Any] | Sequence[Any]
    ) -> aiosqlite.Row | None:
        async with self.connection.execute(sql, params) as cursor:
            return await cursor.fetchone()

    async def _fetch_all(self, sql: str, params: Sequence[Any]) -> list[aiosqlite.Row]:
        async with self.connection.execute(sql, params) as cursor:
            return list(await cursor.fetchall())

    # --- runs -------------------------------------------------------------------

    async def create_run(self, run: Run) -> None:
        """Insert a new run. A duplicate run id is rejected."""
        await self._write(
            """
            INSERT INTO runs (run_id, status, routing_policy, strategy, graph_version,
                              final_work_unit_id, request_json, usage_input_tokens,
                              usage_output_tokens,
                              usage_strong_model_tokens, usage_elapsed_ms,
                              metrics_usage_observed, metrics_usage_known,
                              metrics_provider_elapsed_ms, metrics_wall_clock_ms,
                              metrics_cost_known, metrics_cost,
                              error_category, error_message,
                              created_at, started_at, completed_at)
            VALUES (:run_id, :status, :routing_policy, :strategy, :graph_version,
                    :final_work_unit_id, :request_json, :usage_input_tokens, :usage_output_tokens,
                    :usage_strong_model_tokens, :usage_elapsed_ms,
                    :metrics_usage_observed, :metrics_usage_known,
                    :metrics_provider_elapsed_ms, :metrics_wall_clock_ms,
                    :metrics_cost_known, :metrics_cost,
                    :error_category, :error_message,
                    :created_at, :started_at, :completed_at)
            """,
            {
                "run_id": run.run_id,
                "status": run.status.value,
                "routing_policy": run.request.routing_policy.value,
                "strategy": None if run.strategy is None else run.strategy.value,
                "graph_version": run.graph_version,
                "final_work_unit_id": run.final_work_unit_id,
                "request_json": run.request.model_dump_json(),
                "usage_input_tokens": run.usage.input_tokens,
                "usage_output_tokens": run.usage.output_tokens,
                "usage_strong_model_tokens": run.usage.strong_model_tokens,
                "usage_elapsed_ms": run.usage.elapsed_ms,
                "metrics_usage_observed": int(run.metrics.usage is not None),
                "metrics_usage_known": int(run.metrics.usage is not None),
                "metrics_provider_elapsed_ms": run.metrics.provider_elapsed_ms,
                "metrics_wall_clock_ms": run.metrics.wall_clock_ms,
                "metrics_cost_known": int(run.metrics.cost is not None),
                "metrics_cost": None if run.metrics.cost is None else str(run.metrics.cost),
                "error_category": None if run.error is None else run.error.category.value,
                "error_message": None if run.error is None else run.error.message,
                "created_at": _require_iso(run.created_at),
                "started_at": _iso(run.started_at),
                "completed_at": _iso(run.completed_at),
            },
            "run",
        )

    async def update_run(self, run: Run) -> None:
        """Persist a run's mutable state. Usage moves only through ``add_run_usage``."""
        changed = await self._write(
            """
            UPDATE runs
               SET status = :status,
                   strategy = :strategy,
                   graph_version = :graph_version,
                   final_work_unit_id = :final_work_unit_id,
                   metrics_provider_elapsed_ms = :metrics_provider_elapsed_ms,
                   metrics_wall_clock_ms = :metrics_wall_clock_ms,
                   metrics_cost = :metrics_cost,
                   metrics_usage_known = CASE
                       WHEN :metrics_usage_known = 1 THEN 1 ELSE metrics_usage_known END,
                   metrics_cost_known = CASE
                       WHEN :metrics_cost_known = 1 THEN 1 ELSE metrics_cost_known END,
                   error_category = :error_category,
                   error_message = :error_message,
                   started_at = :started_at,
                   completed_at = :completed_at
             WHERE run_id = :run_id
            """,
            {
                "run_id": run.run_id,
                "status": run.status.value,
                "strategy": None if run.strategy is None else run.strategy.value,
                "graph_version": run.graph_version,
                "final_work_unit_id": run.final_work_unit_id,
                "metrics_provider_elapsed_ms": run.metrics.provider_elapsed_ms,
                "metrics_wall_clock_ms": run.metrics.wall_clock_ms,
                "metrics_cost_known": int(run.metrics.cost is not None),
                "metrics_cost": None if run.metrics.cost is None else str(run.metrics.cost),
                "metrics_usage_known": int(run.metrics.usage is not None),
                "error_category": None if run.error is None else run.error.category.value,
                "error_message": None if run.error is None else run.error.message,
                "started_at": _iso(run.started_at),
                "completed_at": _iso(run.completed_at),
            },
            "run",
        )
        if changed == 0:
            raise MissingEntityError(
                f"run {run.run_id} is not persisted", code=ErrorCode.RUN_NOT_FOUND
            )

    async def get_run(self, run_id: str) -> Run:
        """Read one run, or raise ``MissingEntityError``."""
        row = await self._fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            raise MissingEntityError(f"run {run_id} is not persisted", code=ErrorCode.RUN_NOT_FOUND)
        return self._row_to_run(row)

    # --- workspaces and snapshots ---------------------------------------------

    async def create_workspace(self, workspace: Workspace) -> None:
        """Insert one server-owned workspace identity."""
        try:
            await self._write(
                """
                INSERT INTO workspaces (
                    workspace_id, owner_id, alias, source_type, server_alias,
                    bridge_grant, status, created_at, closed_at
                ) VALUES (:workspace_id, :owner_id, :alias, :source_type,
                          :server_alias, :bridge_grant, :status, :created_at, :closed_at)
                """,
                {
                    "workspace_id": workspace.workspace_id,
                    "owner_id": workspace.owner_id,
                    "alias": workspace.alias,
                    "source_type": workspace.source.source_type.value,
                    "server_alias": workspace.source.server_alias,
                    "bridge_grant": workspace.source.bridge_grant,
                    "status": workspace.status.value,
                    "created_at": _require_iso(workspace.created_at),
                    "closed_at": _iso(workspace.closed_at),
                },
                "workspace",
            )
        except sqlite3.IntegrityError as error:
            raise _translate_integrity_error(error, "workspace") from error

    async def get_workspace(self, workspace_id: str, *, owner_id: str) -> Workspace:
        """Read a workspace only when it belongs to ``owner_id``."""
        row = await self._fetch_one(
            "SELECT * FROM workspaces WHERE workspace_id = ? AND owner_id = ?",
            (workspace_id, owner_id),
        )
        if row is None:
            raise MissingEntityError(f"workspace {workspace_id} is not persisted")
        return self._row_to_workspace(row)

    async def list_workspaces(self, *, owner_id: str) -> tuple[Workspace, ...]:
        """List only the caller principal's workspaces."""
        rows = await self._fetch_all(
            "SELECT * FROM workspaces WHERE owner_id = ? ORDER BY created_at, workspace_id",
            (owner_id,),
        )
        return tuple(self._row_to_workspace(row) for row in rows)

    # --- sessions and API ownership ------------------------------------------

    async def create_session(self, session: Session) -> Session:
        """Persist one principal-owned session after checking its workspace."""
        async with self.transaction() as connection:
            workspace = await self._fetch_one(
                "SELECT workspace_id FROM workspaces WHERE workspace_id = ? AND owner_id = ?",
                (session.workspace_id, session.principal_id),
            )
            if workspace is None:
                raise MissingEntityError("workspace is outside the authenticated principal scope")
            existing = await self._fetch_one(
                "SELECT * FROM sessions WHERE session_id = ?", (session.session_id,)
            )
            if existing is not None:
                if str(existing["principal_id"]) != session.principal_id:
                    raise MissingEntityError("session is outside the authenticated principal scope")
                persisted = self._row_to_session(existing)
                if persisted == session:
                    return persisted
                raise DuplicateEntityError("session id is already persisted")
            try:
                await connection.execute(
                    """
                    INSERT INTO sessions (
                        session_id, principal_id, workspace_id, access_json,
                        agent_options_json, status, created_at, expires_at, revoked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.session_id,
                        session.principal_id,
                        session.workspace_id,
                        json.dumps(
                            [access.value for access in session.grant.access], sort_keys=True
                        ),
                        session.agent_options.model_dump_json(),
                        session.status.value,
                        _require_iso(session.created_at),
                        _iso(session.expires_at),
                        _iso(session.revoked_at),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "session") from error
        return session

    async def get_session(self, session_id: str, *, principal_id: str) -> Session:
        """Read a session only through its authenticated principal boundary."""
        row = await self._fetch_one(
            """
            SELECT s.* FROM sessions AS s
            JOIN workspaces AS w ON w.workspace_id = s.workspace_id
            WHERE s.session_id = ? AND s.principal_id = ? AND w.owner_id = ?
            """,
            (session_id, principal_id, principal_id),
        )
        if row is None:
            raise MissingEntityError("session is outside the authenticated principal scope")
        return self._row_to_session(row)

    async def attach_run_to_session(
        self, session_id: str, run_id: str, *, principal_id: str
    ) -> None:
        """Durably associate a newly-created run with its authenticated session."""
        async with self.transaction() as connection:
            session = await self._fetch_one(
                """
                SELECT s.session_id FROM sessions AS s
                JOIN workspaces AS w ON w.workspace_id = s.workspace_id
                WHERE s.session_id = ? AND s.principal_id = ? AND w.owner_id = ?
                """,
                (session_id, principal_id, principal_id),
            )
            if session is None:
                raise MissingEntityError("session is outside the authenticated principal scope")
            run = await self._fetch_one("SELECT run_id FROM runs WHERE run_id = ?", (run_id,))
            if run is None:
                raise MissingEntityError(
                    f"run {run_id} is not persisted", code=ErrorCode.RUN_NOT_FOUND
                )
            existing = await self._fetch_one(
                "SELECT session_id FROM session_runs WHERE run_id = ?", (run_id,)
            )
            if existing is not None:
                if str(existing["session_id"]) == session_id:
                    return
                raise DuplicateEntityError("run is already attached to another session")
            try:
                await connection.execute(
                    "INSERT INTO session_runs (session_id, run_id, created_at) VALUES (?, ?, ?)",
                    (session_id, run_id, _require_iso(utc_now())),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "session run") from error

    async def get_run_for_session(
        self, session_id: str, run_id: str, *, principal_id: str
    ) -> Run:
        """Read a run only when the session and run share the caller's scope."""
        row = await self._fetch_one(
            """
            SELECT r.* FROM runs AS r
            JOIN session_runs AS sr ON sr.run_id = r.run_id
            JOIN sessions AS s ON s.session_id = sr.session_id
            JOIN workspaces AS w ON w.workspace_id = s.workspace_id
            WHERE sr.session_id = ? AND sr.run_id = ?
              AND s.principal_id = ? AND w.owner_id = ?
            """,
            (session_id, run_id, principal_id, principal_id),
        )
        if row is None:
            raise MissingEntityError(f"run {run_id} is not in the authenticated session")
        return self._row_to_run(row)

    async def get_execution_scope(
        self, run_id: str, *, principal_id: str
    ) -> ExecutionScope | None:
        """Restore one active owner-scoped Session execution scope.

        A run without a ``session_runs`` row is a legacy/non-Session run and
        returns ``None``. Scope facts are checked again at read time so a
        revoked or expired Session and an inactive Workspace cannot be reused.
        """
        rows = await self._fetch_all(
            """
            SELECT r.run_id, s.*, w.status AS workspace_status, w.owner_id AS workspace_owner_id
              FROM runs AS r
              JOIN session_runs AS sr ON sr.run_id = r.run_id
              JOIN sessions AS s ON s.session_id = sr.session_id
              JOIN workspaces AS w ON w.workspace_id = s.workspace_id
             WHERE r.run_id = ? AND s.principal_id = ? AND w.owner_id = ?
            """,
            (run_id, principal_id, principal_id),
        )
        if not rows:
            return None
        if len(rows) != 1:
            raise InternalError("run execution scope is ambiguous")
        row = rows[0]
        session = self._row_to_session(row)
        if session.status is not SessionStatus.ACTIVE:
            raise StateError("session is not active")
        if session.expires_at is not None and session.expires_at <= utc_now():
            raise StateError("session grant has expired")
        if WorkspaceStatus(str(row["workspace_status"])) is not WorkspaceStatus.ACTIVE:
            raise StateError("workspace is not active")
        return ExecutionScope(
            run_id=str(row["run_id"]),
            session_id=session.session_id,
            principal_id=session.principal_id,
            workspace_id=session.workspace_id,
            grant=session.grant,
            agent_options=session.agent_options,
        )

    async def create_snapshot(
        self,
        snapshot: Snapshot,
        manifest: SnapshotManifest,
        *,
        owner_id: str,
    ) -> Snapshot:
        """Insert or idempotently replay an owner-scoped immutable snapshot."""
        workspace = await self._fetch_one(
            "SELECT workspace_id FROM workspaces WHERE workspace_id = ? AND owner_id = ?",
            (snapshot.workspace_id, owner_id),
        )
        if workspace is None:
            raise MissingEntityError(f"workspace {snapshot.workspace_id} is not persisted")

        existing = await self._fetch_one(
            "SELECT s.* FROM snapshots AS s "
            "JOIN workspaces AS w ON w.workspace_id = s.workspace_id "
            "WHERE s.manifest_hash = ? AND w.owner_id = ?",
            (manifest.manifest_hash, owner_id),
        )
        if existing is not None:
            if str(existing["workspace_id"]) != snapshot.workspace_id:
                raise DuplicateEntityError("manifest hash belongs to another workspace")
            return self._row_to_snapshot(existing)

        persisted = snapshot.model_copy(
            update={
                "file_count": len(manifest.entries),
                "total_size": manifest.total_size,
            }
        )
        async with self.transaction() as connection:
            try:
                await connection.execute(
                    """
                    INSERT INTO snapshots (
                        snapshot_id, workspace_id, status, manifest_hash, file_count,
                        total_size, created_at, completed_at
                    ) VALUES (:snapshot_id, :workspace_id, :status, :manifest_hash,
                              :file_count, :total_size, :created_at, :completed_at)
                    """,
                    {
                        "snapshot_id": persisted.snapshot_id,
                        "workspace_id": persisted.workspace_id,
                        "status": persisted.status.value,
                        "manifest_hash": manifest.manifest_hash,
                        "file_count": persisted.file_count,
                        "total_size": persisted.total_size,
                        "created_at": _require_iso(persisted.created_at),
                        "completed_at": _iso(persisted.completed_at),
                    },
                )
                for entry in manifest.entries:
                    await connection.execute(
                        """
                        INSERT INTO snapshot_files (
                            snapshot_id, path, sha256, size, entry_type
                        ) VALUES (:snapshot_id, :path, :sha256, :size, :entry_type)
                        """,
                        {
                            "snapshot_id": persisted.snapshot_id,
                            "path": entry.path,
                            "sha256": entry.sha256,
                            "size": entry.size,
                            "entry_type": entry.entry_type.value,
                        },
                    )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "snapshot") from error
        return persisted

    async def get_snapshot(self, snapshot_id: str, *, owner_id: str) -> Snapshot:
        """Read a snapshot only through its owner's workspace."""
        row = await self._fetch_one(
            "SELECT s.* FROM snapshots AS s "
            "JOIN workspaces AS w ON w.workspace_id = s.workspace_id "
            "WHERE s.snapshot_id = ? AND w.owner_id = ?",
            (snapshot_id, owner_id),
        )
        if row is None:
            raise MissingEntityError(f"snapshot {snapshot_id} is not persisted")
        return self._row_to_snapshot(row)

    async def list_snapshots(
        self, workspace_id: str, *, owner_id: str
    ) -> tuple[Snapshot, ...]:
        """List snapshots only for an owner-scoped workspace."""
        rows = await self._fetch_all(
            """
            SELECT s.*
              FROM snapshots AS s
              JOIN workspaces AS w ON w.workspace_id = s.workspace_id
             WHERE s.workspace_id = ? AND w.owner_id = ?
             ORDER BY s.created_at, s.snapshot_id
            """,
            (workspace_id, owner_id),
        )
        return tuple(self._row_to_snapshot(row) for row in rows)

    async def get_snapshot_manifest(
        self, snapshot_id: str, *, owner_id: str
    ) -> SnapshotManifest:
        """Round-trip manifest entries through the owner's snapshot scope."""
        snapshot = await self.get_snapshot(snapshot_id, owner_id=owner_id)
        rows = await self._fetch_all(
            "SELECT path, sha256, size, entry_type FROM snapshot_files "
            "WHERE snapshot_id = ? ORDER BY path",
            (snapshot.snapshot_id,),
        )
        return SnapshotManifest(
            entries=tuple(
                SnapshotEntry(
                    path=str(row["path"]),
                    sha256=str(row["sha256"]),
                    size=int(row["size"]),
                    entry_type=SnapshotEntryType(str(row["entry_type"])),
                )
                for row in rows
            )
        )

    # --- ChangeSets ------------------------------------------------------------

    async def create_change_set(self, change_set: ChangeSet) -> ChangeSet:
        """Persist one immutable, tool-bound ChangeSet or replay the same fact."""
        async with self.transaction() as connection:
            context = await self._fetch_one(
                """
                SELECT run_id, workspace_id, base_snapshot_id
                  FROM tool_calls
                 WHERE call_id = ?
                """,
                (change_set.tool_call_id,),
            )
            if context is None:
                raise MissingEntityError(
                    f"tool call {change_set.tool_call_id} is not persisted"
                )
            if (
                str(context["run_id"]) != change_set.run_id
                or str(context["workspace_id"]) != change_set.workspace_id
                or str(context["base_snapshot_id"]) != change_set.base_snapshot_id
            ):
                raise StateError("ChangeSet does not match its tool call context")
            new_snapshot = await self._fetch_one(
                "SELECT snapshot_id FROM snapshots WHERE snapshot_id = ? AND workspace_id = ?",
                (change_set.new_snapshot_id, change_set.workspace_id),
            )
            if new_snapshot is None:
                raise StateError("ChangeSet new snapshot is not in the authorized workspace")

            existing = await self._fetch_one(
                "SELECT * FROM change_sets WHERE tool_call_id = ?",
                (change_set.tool_call_id,),
            )
            if existing is not None:
                persisted = await self._row_to_change_set(existing)
                if (
                    persisted.model_copy(update={"change_set_id": change_set.change_set_id})
                    == change_set
                ):
                    return persisted
                raise DuplicateEntityError("tool call already has a different ChangeSet")

            try:
                await connection.execute(
                    """
                    INSERT INTO change_sets (
                        change_set_id, run_id, tool_call_id, workspace_id,
                        base_snapshot_id, new_snapshot_id, patch_text, patch_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        change_set.change_set_id,
                        change_set.run_id,
                        change_set.tool_call_id,
                        change_set.workspace_id,
                        change_set.base_snapshot_id,
                        change_set.new_snapshot_id,
                        change_set.patch.unified_diff,
                        change_set.patch_sha256,
                        _require_iso(change_set.created_at),
                    ),
                )
                for file_change in change_set.files:
                    await connection.execute(
                        """
                        INSERT INTO change_set_files (
                            change_set_id, path, action, before_sha256, before_size,
                            after_sha256, after_size
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            change_set.change_set_id,
                            file_change.path,
                            file_change.action.value,
                            None if file_change.before is None else file_change.before.sha256,
                            None if file_change.before is None else file_change.before.size,
                            None if file_change.after is None else file_change.after.sha256,
                            None if file_change.after is None else file_change.after.size,
                        ),
                    )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "ChangeSet") from error
        return change_set

    async def get_change_set(self, change_set_id: str) -> ChangeSet:
        """Read one immutable ChangeSet and its ordered file facts."""
        row = await self._fetch_one(
            "SELECT * FROM change_sets WHERE change_set_id = ?", (change_set_id,)
        )
        if row is None:
            raise MissingEntityError(f"ChangeSet {change_set_id} is not persisted")
        return await self._row_to_change_set(row)

    async def list_change_sets(
        self,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
        workspace_id: str | None = None,
    ) -> tuple[ChangeSet, ...]:
        """List ChangeSets through at least one narrow, auditable identity."""
        filters = {
            "run_id": run_id,
            "tool_call_id": tool_call_id,
            "workspace_id": workspace_id,
        }
        clauses = [f"{name} = ?" for name, value in filters.items() if value is not None]
        values = [value for value in filters.values() if value is not None]
        if not clauses:
            raise ValueError("list_change_sets requires a run, tool call, or workspace scope")
        rows = await self._fetch_all(
            "SELECT * FROM change_sets WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at, change_set_id",
            values,
        )
        return tuple([await self._row_to_change_set(row) for row in rows])

    async def _row_to_change_set(self, row: aiosqlite.Row) -> ChangeSet:
        file_rows = await self._fetch_all(
            """
            SELECT path, action, before_sha256, before_size, after_sha256, after_size
              FROM change_set_files
             WHERE change_set_id = ?
             ORDER BY path
            """,
            (str(row["change_set_id"]),),
        )
        files = tuple(
            FileChange(
                path=str(file_row["path"]),
                action=FileChangeAction(str(file_row["action"])),
                before=(
                    None
                    if file_row["before_sha256"] is None
                    else FileContent(
                        sha256=str(file_row["before_sha256"]),
                        size=int(file_row["before_size"]),
                    )
                ),
                after=(
                    None
                    if file_row["after_sha256"] is None
                    else FileContent(
                        sha256=str(file_row["after_sha256"]),
                        size=int(file_row["after_size"]),
                    )
                ),
            )
            for file_row in file_rows
        )
        patch = Patch(
            base_snapshot_id=str(row["base_snapshot_id"]),
            unified_diff=str(row["patch_text"]),
        )
        if patch.sha256 != str(row["patch_sha256"]):
            raise InternalError("stored ChangeSet patch hash does not match its content")
        return ChangeSet(
            change_set_id=str(row["change_set_id"]),
            run_id=str(row["run_id"]),
            tool_call_id=str(row["tool_call_id"]),
            workspace_id=str(row["workspace_id"]),
            base_snapshot_id=str(row["base_snapshot_id"]),
            new_snapshot_id=str(row["new_snapshot_id"]),
            patch=patch,
            files=files,
            created_at=_require_datetime(row["created_at"]),
        )

    # --- Merge ledger ----------------------------------------------------------

    async def create_merge_ledger(self, ledger: MergeLedger) -> MergeLedger:
        """Persist one planned merge, replaying the same input idempotently."""
        change_set_ids_json = json.dumps(
            list(ledger.change_set_ids), ensure_ascii=True, separators=(",", ":")
        )
        async with self.transaction() as connection:
            existing = await self._fetch_one(
                "SELECT * FROM merge_ledger WHERE merge_id = ?",
                (ledger.merge_id,),
            )
            if existing is not None:
                persisted = self._row_to_merge_ledger(existing)
                if persisted == ledger:
                    return persisted
                raise DuplicateEntityError("merge ledger id already has a different fact")

            existing_input = await self._fetch_one(
                "SELECT * FROM merge_ledger WHERE run_id = ? AND input_digest = ?",
                (ledger.run_id, ledger.input_digest),
            )
            if existing_input is not None:
                persisted = self._row_to_merge_ledger(existing_input)
                same_input = (
                    persisted.workspace_id == ledger.workspace_id
                    and persisted.base_snapshot_id == ledger.base_snapshot_id
                    and persisted.change_set_ids == ledger.change_set_ids
                )
                if same_input and ledger.status is MergeLedgerStatus.PLANNED:
                    return persisted
                raise DuplicateEntityError("merge input already has a lifecycle fact")

            try:
                await connection.execute(
                    """
                    INSERT INTO merge_ledger (
                        merge_id, run_id, workspace_id, base_snapshot_id,
                        change_set_ids_json, input_digest, status,
                        merged_snapshot_id, merged_content_hash, promoted_content_hash,
                        created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ledger.merge_id,
                        ledger.run_id,
                        ledger.workspace_id,
                        ledger.base_snapshot_id,
                        change_set_ids_json,
                        ledger.input_digest,
                        ledger.status.value,
                        ledger.merged_snapshot_id,
                        ledger.merged_content_hash,
                        ledger.promoted_content_hash,
                        _require_iso(ledger.created_at),
                        _iso(ledger.completed_at),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "merge ledger") from error
            await self.append_event(
                ledger.run_id,
                _MERGE_EVENT_BY_STATUS[ledger.status],
                payload_from_merge_ledger(ledger),
                timestamp=ledger.created_at,
            )
        return ledger

    async def get_merge_ledger(self, merge_id: str) -> MergeLedger:
        """Read one merge lifecycle fact by its opaque identity."""
        row = await self._fetch_one(
            "SELECT * FROM merge_ledger WHERE merge_id = ?", (merge_id,)
        )
        if row is None:
            raise MissingEntityError(f"merge ledger {merge_id} is not persisted")
        return self._row_to_merge_ledger(row)

    async def list_merge_ledgers(
        self,
        *,
        run_id: str | None = None,
        statuses: Sequence[MergeLedgerStatus] | None = None,
    ) -> tuple[MergeLedger, ...]:
        """List merge facts through optional run and lifecycle filters."""
        clauses: list[str] = []
        values: list[object] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            values.extend(status.value for status in statuses)
        query = "SELECT * FROM merge_ledger"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, merge_id"
        rows = await self._fetch_all(query, values)
        return tuple(self._row_to_merge_ledger(row) for row in rows)

    async def update_merge_ledger(self, ledger: MergeLedger) -> MergeLedger:
        """Apply one legal merge transition and append its lifecycle event."""
        async with self.transaction() as connection:
            row = await self._fetch_one(
                "SELECT * FROM merge_ledger WHERE merge_id = ?", (ledger.merge_id,)
            )
            if row is None:
                raise MissingEntityError(f"merge ledger {ledger.merge_id} is not persisted")
            current = self._row_to_merge_ledger(row)
            if current == ledger:
                return current
            if (
                current.run_id != ledger.run_id
                or current.workspace_id != ledger.workspace_id
                or current.base_snapshot_id != ledger.base_snapshot_id
                or current.change_set_ids != ledger.change_set_ids
                or current.input_digest != ledger.input_digest
                or current.created_at != ledger.created_at
            ):
                raise StateError("merge ledger identity is immutable")
            transition_merge(current.status, ledger.status)
            try:
                await connection.execute(
                    """
                    UPDATE merge_ledger
                       SET status = ?, merged_snapshot_id = ?, merged_content_hash = ?,
                           promoted_content_hash = ?, completed_at = ?
                     WHERE merge_id = ?
                    """,
                    (
                        ledger.status.value,
                        ledger.merged_snapshot_id,
                        ledger.merged_content_hash,
                        ledger.promoted_content_hash,
                        _iso(ledger.completed_at),
                        ledger.merge_id,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "merge ledger") from error
            await self.append_event(
                ledger.run_id,
                _MERGE_EVENT_BY_STATUS[ledger.status],
                payload_from_merge_ledger(ledger),
                timestamp=ledger.completed_at or ledger.created_at,
            )
        return ledger

    async def mark_merge_unknown(
        self, merge_id: str, *, completed_at: datetime | None = None
    ) -> MergeLedger:
        """Close a running merge as UNKNOWN without guessing its filesystem outcome."""
        current = await self.get_merge_ledger(merge_id)
        if current.status is not MergeLedgerStatus.RUNNING:
            return current
        closed_at = completed_at or utc_now()
        if closed_at < current.created_at:
            closed_at = current.created_at
        unknown = current.model_copy(
            update={
                "status": MergeLedgerStatus.UNKNOWN,
                "completed_at": closed_at,
            }
        )
        return await self.update_merge_ledger(unknown)

    async def mark_merge_promoted(
        self,
        merge_id: str,
        promoted_content_hash: str,
        *,
        completed_at: datetime | None = None,
    ) -> MergeLedger:
        """Record verified promotion once, replaying the same hash idempotently."""
        current = await self.get_merge_ledger(merge_id)
        if current.status is MergeLedgerStatus.PROMOTED:
            if current.promoted_content_hash != promoted_content_hash:
                raise StateError("promotion hash does not match the terminal merge fact")
            return current
        if current.status is not MergeLedgerStatus.MERGED:
            raise StateError("only a merged candidate can be promoted")
        if current.merged_content_hash != promoted_content_hash:
            raise StateError("promotion hash does not match the merged candidate")
        closed_at = completed_at or utc_now()
        if current.completed_at is not None and closed_at < current.completed_at:
            closed_at = current.completed_at
        promoted = current.model_copy(
            update={
                "status": MergeLedgerStatus.PROMOTED,
                "promoted_content_hash": promoted_content_hash,
                "completed_at": closed_at,
            }
        )
        return await self.update_merge_ledger(promoted)

    @staticmethod
    def _row_to_merge_ledger(row: aiosqlite.Row) -> MergeLedger:
        parsed_ids = strict_json_loads(str(row["change_set_ids_json"]))
        if not isinstance(parsed_ids, list) or not all(
            isinstance(item, str) for item in parsed_ids
        ):
            raise InternalError("stored merge ChangeSet ids are invalid")
        return MergeLedger(
            merge_id=str(row["merge_id"]),
            run_id=str(row["run_id"]),
            workspace_id=str(row["workspace_id"]),
            base_snapshot_id=str(row["base_snapshot_id"]),
            change_set_ids=tuple(parsed_ids),
            input_digest=str(row["input_digest"]),
            status=MergeLedgerStatus(str(row["status"])),
            merged_snapshot_id=_optional_str(row["merged_snapshot_id"]),
            merged_content_hash=_optional_str(row["merged_content_hash"]),
            promoted_content_hash=_optional_str(row["promoted_content_hash"]),
            created_at=_require_datetime(row["created_at"]),
            completed_at=_parse_datetime(row["completed_at"]),
        )

    # --- Progressive rounds ----------------------------------------------------

    async def create_round(self, progressive_round: ProgressiveRound) -> ProgressiveRound:
        """Insert one immutable round fact and its recovery event."""
        from prp_runtime.control.progressive import RoundStatus

        async with self.transaction() as connection:
            run = await self._fetch_one(
                "SELECT run_id FROM runs WHERE run_id = ?", (progressive_round.run_id,)
            )
            if run is None:
                raise MissingEntityError(f"run {progressive_round.run_id} is not persisted")
            base = await self._fetch_one(
                (
                    "SELECT snapshot_id, workspace_id, status "
                    "FROM snapshots WHERE snapshot_id = ?"
                ),
                (progressive_round.base_snapshot_id,),
            )
            if base is None:
                raise MissingEntityError(
                    f"base snapshot {progressive_round.base_snapshot_id} is not persisted"
                )
            if str(base["status"]) != SnapshotStatus.READY.value:
                raise StateError("Progressive round base snapshot is not READY")
            base_workspace_id = str(base["workspace_id"])
            if progressive_round.merged_snapshot_id is not None:
                merged = await self._fetch_one(
                    (
                        "SELECT snapshot_id, workspace_id, status "
                        "FROM snapshots WHERE snapshot_id = ?"
                    ),
                    (progressive_round.merged_snapshot_id,),
                )
                if merged is None:
                    raise MissingEntityError(
                        f"merged snapshot {progressive_round.merged_snapshot_id} is not persisted"
                    )
                if str(merged["status"]) != SnapshotStatus.READY.value:
                    raise StateError("Progressive round merged snapshot is not READY")
                if str(merged["workspace_id"]) != base_workspace_id:
                    raise StateError(
                        "Progressive round snapshots do not share one workspace"
                    )
            if progressive_round.change_set_ids:
                placeholders = ",".join("?" for _ in progressive_round.change_set_ids)
                row = await self._fetch_one(
                    f"SELECT COUNT(*) AS count FROM change_sets WHERE run_id = ? "
                    f"AND base_snapshot_id = ? AND change_set_id IN ({placeholders})",
                    (
                        progressive_round.run_id,
                        progressive_round.base_snapshot_id,
                        *progressive_round.change_set_ids,
                    ),
                )
                if row is None or int(row["count"]) != len(progressive_round.change_set_ids):
                    raise StateError(
                        "round ChangeSets do not share the persisted base snapshot"
                    )
            if progressive_round.evidence_ids:
                placeholders = ",".join("?" for _ in progressive_round.evidence_ids)
                row = await self._fetch_one(
                    f"SELECT COUNT(*) AS count FROM evidence WHERE run_id = ? "
                    f"AND evidence_id IN ({placeholders})",
                    (progressive_round.run_id, *progressive_round.evidence_ids),
                )
                if row is None or int(row["count"]) != len(progressive_round.evidence_ids):
                    raise MissingEntityError("round references an unpersisted Evidence")
            existing = await self._fetch_one(
                "SELECT * FROM progressive_rounds WHERE round_id = ?",
                (progressive_round.round_id,),
            )
            if existing is not None:
                restored = self._row_to_round(existing)
                if restored == progressive_round:
                    return restored
                raise DuplicateEntityError("round already has a different immutable fact")
            try:
                await connection.execute(
                    """
                    INSERT INTO progressive_rounds (
                        round_id, run_id, round_index, graph_version, base_snapshot_id,
                        merged_snapshot_id, change_set_ids_json, evidence_ids_json, status,
                        revision_of_round_id, revision_reason, failure_reason, created_at,
                        completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        progressive_round.round_id,
                        progressive_round.run_id,
                        progressive_round.round_index,
                        progressive_round.graph_version,
                        progressive_round.base_snapshot_id,
                        progressive_round.merged_snapshot_id,
                        json.dumps(list(progressive_round.change_set_ids), sort_keys=True),
                        json.dumps(list(progressive_round.evidence_ids), sort_keys=True),
                        progressive_round.status.value,
                        progressive_round.revision_of_round_id,
                        progressive_round.revision_reason,
                        progressive_round.failure_reason,
                        _require_iso(progressive_round.created_at),
                        _iso(progressive_round.completed_at),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "Progressive round") from error
            event_type = (
                EventType.ROUND_VERIFIED
                if progressive_round.status is RoundStatus.VERIFIED
                else EventType.ROUND_FAILED
                if progressive_round.status in (RoundStatus.FAILED, RoundStatus.CANCELLED)
                else EventType.ROUND_CREATED
            )
            payload: dict[str, JsonValue] = {
                "round_id": progressive_round.round_id,
                "round_index": progressive_round.round_index,
                "graph_version": progressive_round.graph_version,
                "status": progressive_round.status.value,
            }
            if progressive_round.merged_snapshot_id is not None:
                payload["merged_snapshot_id"] = progressive_round.merged_snapshot_id
            if progressive_round.failure_reason is not None:
                payload["reason"] = progressive_round.failure_reason
            await self.append_event(progressive_round.run_id, event_type, payload)
        return progressive_round

    async def create_progressive_round(
        self, progressive_round: ProgressiveRound
    ) -> ProgressiveRound:
        """Descriptive alias for ``create_round``."""
        return await self.create_round(progressive_round)

    async def update_round(self, progressive_round: ProgressiveRound) -> ProgressiveRound:
        """Close one planned round exactly once with its verified outcome."""
        from prp_runtime.control.progressive import RoundStatus

        async with self.transaction() as connection:
            row = await self._fetch_one(
                "SELECT * FROM progressive_rounds WHERE round_id = ?",
                (progressive_round.round_id,),
            )
            if row is None:
                raise MissingEntityError(
                    f"round {progressive_round.round_id} is not persisted"
                )
            current = self._row_to_round(row)
            if current == progressive_round:
                return current
            if (
                current.run_id != progressive_round.run_id
                or current.round_index != progressive_round.round_index
                or current.graph_version != progressive_round.graph_version
                or current.base_snapshot_id != progressive_round.base_snapshot_id
                or current.change_set_ids != progressive_round.change_set_ids
                or current.revision_of_round_id != progressive_round.revision_of_round_id
                or current.revision_reason != progressive_round.revision_reason
                or current.created_at != progressive_round.created_at
            ):
                raise StateError("Progressive round identity is immutable")
            if current.status is not RoundStatus.PLANNED:
                raise StateError("terminal Progressive round is immutable")
            if progressive_round.status is RoundStatus.PLANNED:
                raise StateError("planned Progressive round has no close transition")
            if progressive_round.merged_snapshot_id is not None:
                merged = await self._fetch_one(
                    (
                        "SELECT workspace_id, status FROM snapshots "
                        "WHERE snapshot_id = ?"
                    ),
                    (progressive_round.merged_snapshot_id,),
                )
                if merged is None:
                    raise MissingEntityError(
                        "Progressive round references a missing merged snapshot"
                    )
                base = await self._fetch_one(
                    (
                        "SELECT workspace_id, status FROM snapshots "
                        "WHERE snapshot_id = ?"
                    ),
                    (progressive_round.base_snapshot_id,),
                )
                if base is None:
                    raise MissingEntityError(
                        "Progressive round references a missing base snapshot"
                    )
                if str(merged["status"]) != SnapshotStatus.READY.value:
                    raise StateError("Progressive round merged snapshot is not READY")
                if (
                    str(base["status"]) != SnapshotStatus.READY.value
                    or str(merged["workspace_id"]) != str(base["workspace_id"])
                ):
                    raise StateError(
                        "Progressive round snapshots do not share one READY workspace"
                    )
            if progressive_round.evidence_ids:
                placeholders = ",".join("?" for _ in progressive_round.evidence_ids)
                evidence = await self._fetch_one(
                    f"SELECT COUNT(*) AS count FROM evidence WHERE run_id = ? "
                    f"AND evidence_id IN ({placeholders})",
                    (progressive_round.run_id, *progressive_round.evidence_ids),
                )
                if evidence is None or int(evidence["count"]) != len(
                    progressive_round.evidence_ids
                ):
                    raise MissingEntityError(
                        "Progressive round references an unpersisted Evidence"
                    )
            try:
                await connection.execute(
                    """
                    UPDATE progressive_rounds
                       SET merged_snapshot_id = ?, evidence_ids_json = ?, status = ?,
                           failure_reason = ?, completed_at = ?
                     WHERE round_id = ?
                    """,
                    (
                        progressive_round.merged_snapshot_id,
                        json.dumps(
                            list(progressive_round.evidence_ids), sort_keys=True
                        ),
                        progressive_round.status.value,
                        progressive_round.failure_reason,
                        _iso(progressive_round.completed_at),
                        progressive_round.round_id,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "Progressive round") from error
            event_type = (
                EventType.ROUND_VERIFIED
                if progressive_round.status is RoundStatus.VERIFIED
                else EventType.ROUND_FAILED
            )
            payload: dict[str, JsonValue] = {
                "round_id": progressive_round.round_id,
                "round_index": progressive_round.round_index,
                "graph_version": progressive_round.graph_version,
                "status": progressive_round.status.value,
            }
            if progressive_round.merged_snapshot_id is not None:
                payload["merged_snapshot_id"] = progressive_round.merged_snapshot_id
            if progressive_round.failure_reason is not None:
                payload["reason"] = progressive_round.failure_reason
            await self.append_event(progressive_round.run_id, event_type, payload)
        return progressive_round

    async def update_progressive_round(
        self, progressive_round: ProgressiveRound
    ) -> ProgressiveRound:
        """Descriptive alias for ``update_round``."""
        return await self.update_round(progressive_round)

    async def get_round(self, round_id: str) -> ProgressiveRound:
        """Restore one immutable Progressive round fact."""
        row = await self._fetch_one(
            "SELECT * FROM progressive_rounds WHERE round_id = ?", (round_id,)
        )
        if row is None:
            raise MissingEntityError(f"round {round_id} is not persisted")
        return self._row_to_round(row)

    async def get_progressive_round(self, round_id: str) -> ProgressiveRound:
        """Descriptive alias for ``get_round``."""
        return await self.get_round(round_id)

    async def list_rounds(self, run_id: str) -> tuple[ProgressiveRound, ...]:
        """Restore all rounds in stable round order for one run."""
        rows = await self._fetch_all(
            "SELECT * FROM progressive_rounds WHERE run_id = ? "
            "ORDER BY round_index, graph_version, round_id",
            (run_id,),
        )
        return tuple(self._row_to_round(row) for row in rows)

    async def list_progressive_rounds(self, run_id: str) -> tuple[ProgressiveRound, ...]:
        """Descriptive alias for ``list_rounds``."""
        return await self.list_rounds(run_id)

    @staticmethod
    def _row_to_round(row: aiosqlite.Row) -> ProgressiveRound:
        from prp_runtime.control.progressive import ProgressiveRound, RoundStatus

        change_set_ids = strict_json_loads(str(row["change_set_ids_json"]))
        evidence_ids = strict_json_loads(str(row["evidence_ids_json"]))
        if not isinstance(change_set_ids, list) or not all(
            isinstance(value, str) for value in change_set_ids
        ):
            raise InternalError("stored round ChangeSet ids are malformed")
        if not isinstance(evidence_ids, list) or not all(
            isinstance(value, str) for value in evidence_ids
        ):
            raise InternalError("stored round Evidence ids are malformed")
        return ProgressiveRound(
            round_id=str(row["round_id"]),
            run_id=str(row["run_id"]),
            round_index=int(row["round_index"]),
            graph_version=int(row["graph_version"]),
            base_snapshot_id=str(row["base_snapshot_id"]),
            merged_snapshot_id=_optional_str(row["merged_snapshot_id"]),
            change_set_ids=tuple(change_set_ids),
            evidence_ids=tuple(evidence_ids),
            status=RoundStatus(str(row["status"])),
            revision_of_round_id=_optional_str(row["revision_of_round_id"]),
            revision_reason=_optional_str(row["revision_reason"]),
            failure_reason=_optional_str(row["failure_reason"]),
            created_at=_require_datetime(row["created_at"]),
            completed_at=_parse_datetime(row["completed_at"]),
        )

    # --- tool calls and results -----------------------------------------------

    async def create_tool_call(
        self,
        call: ToolCall,
        *,
        workspace_id: str,
        idempotency_key: str,
    ) -> ToolCall:
        """Persist one requested tool call and its request event atomically."""
        if call.status is not ToolCallStatus.REQUESTED:
            raise StateError("a new tool call must start in REQUESTED state")
        if call.snapshot_id is None:
            raise ValueError("a tool call requires a base snapshot")
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must not be blank")

        arguments_json = json.dumps(call.arguments, sort_keys=True, allow_nan=False)
        async with self.transaction() as connection:
            existing = await self._fetch_one(
                "SELECT * FROM tool_calls WHERE run_id = ? AND idempotency_key = ?",
                (call.run_id, idempotency_key),
            )
            if existing is not None:
                existing_call = self._row_to_tool_call(existing)
                same_request = (
                    str(existing["workspace_id"]) == workspace_id
                    and existing_call.work_unit_id == call.work_unit_id
                    and existing_call.snapshot_id == call.snapshot_id
                    and existing_call.tool_name == call.tool_name
                    and existing_call.effect is call.effect
                    and existing_call.arguments == call.arguments
                )
                if same_request:
                    return existing_call
                raise DuplicateEntityError(
                    "idempotency key already belongs to a different tool call"
                )
            try:
                await connection.execute(
                    """
                    INSERT INTO tool_calls (
                        call_id, run_id, work_unit_id, workspace_id, base_snapshot_id,
                        idempotency_key, tool_name, effect, arguments_json, status,
                        requested_at
                    ) VALUES (:call_id, :run_id, :work_unit_id, :workspace_id,
                              :base_snapshot_id, :idempotency_key, :tool_name,
                              :effect, :arguments_json, :status, :requested_at)
                    """,
                    {
                        "call_id": call.call_id,
                        "run_id": call.run_id,
                        "work_unit_id": call.work_unit_id,
                        "workspace_id": workspace_id,
                        "base_snapshot_id": call.snapshot_id,
                        "idempotency_key": idempotency_key,
                        "tool_name": call.tool_name,
                        "effect": call.effect.value,
                        "arguments_json": arguments_json,
                        "status": call.status.value,
                        "requested_at": _require_iso(call.requested_at),
                    },
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "tool call") from error
            await self.append_event(
                call.run_id,
                EventType.TOOL_CALL_REQUESTED,
                payload_from_tool_call(call),
                timestamp=call.requested_at,
            )
        return call

    async def await_tool_call(
        self,
        call_id: str,
        *,
        reason: str = "approval required",
        timestamp: datetime | None = None,
    ) -> ToolCall:
        """Move a requested call to approval wait and append its audit event."""
        async with self.transaction() as connection:
            row = await self._fetch_one(
                "SELECT * FROM tool_calls WHERE call_id = ?", (call_id,)
            )
            if row is None:
                raise MissingEntityError(f"tool call {call_id} is not persisted")
            current = self._row_to_tool_call(row)
            transitioned = current.transition(ToolCallStatus.AWAITING_APPROVAL)
            await connection.execute(
                "UPDATE tool_calls SET status = ? WHERE call_id = ?",
                (transitioned.status.value, call_id),
            )
            payload = payload_from_tool_call(transitioned)
            payload["reason"] = reason
            await self.append_event(
                current.run_id,
                EventType.TOOL_CALL_AWAITING_APPROVAL,
                payload,
                timestamp=timestamp or utc_now(),
            )
            return transitioned

    async def start_tool_call(
        self,
        call_id: str,
        *,
        approved: bool | None = None,
        started_at: datetime | None = None,
    ) -> ToolCall:
        """Start a requested call, requiring approval after an approval wait."""
        async with self.transaction() as connection:
            row = await self._fetch_one(
                "SELECT * FROM tool_calls WHERE call_id = ?", (call_id,)
            )
            if row is None:
                raise MissingEntityError(f"tool call {call_id} is not persisted")
            current = self._row_to_tool_call(row)
            transitioned = current.transition(ToolCallStatus.RUNNING, approved=approved)
            started = started_at or utc_now()
            await connection.execute(
                "UPDATE tool_calls SET status = ?, started_at = ? WHERE call_id = ?",
                (transitioned.status.value, _require_iso(started), call_id),
            )
            await self.append_event(
                current.run_id,
                EventType.TOOL_CALL_STARTED,
                payload_from_tool_call(transitioned),
                timestamp=started,
            )
            return transitioned

    async def complete_tool_call(
        self,
        call_or_result: str | ToolResult,
        result: ToolResult | None = None,
    ) -> ToolResult:
        """Persist one terminal result, its call state and event atomically.

        Passing the same result again is an idempotent replay. A different
        result for the same call is rejected rather than overwriting history.
        """
        if isinstance(call_or_result, ToolResult):
            completed = call_or_result
        else:
            if result is None:
                raise ValueError("complete_tool_call requires a result")
            completed = result
            if completed.call_id != call_or_result:
                raise ValueError("tool result call_id does not match the requested call")
        for attempt in range(3):
            try:
                return await self._complete_tool_call_once(completed)
            except DuplicateEntityError as error:
                existing = await self._get_tool_result_if_present(completed.call_id)
                if existing is not None:
                    if existing == completed:
                        return existing
                    raise StateError("tool call already has a conflicting result") from error
                raise
            except sqlite3.OperationalError as error:
                if "locked" not in str(error).lower() or attempt == 2:
                    raise
        raise InternalError("tool result completion retry loop exhausted")

    async def reject_tool_call(
        self,
        call_id: str,
        *,
        reason: str,
        completed_at: datetime | None = None,
    ) -> ToolResult:
        """Reject a pre-execution call without manufacturing a RUNNING state."""
        safe_reason = validate_tool_rejection_reason(reason)
        rejected_at = completed_at or utc_now()
        for attempt in range(3):
            try:
                return await self._reject_tool_call_once(
                    call_id,
                    reason=safe_reason,
                    completed_at=rejected_at,
                )
            except DuplicateEntityError as error:
                existing = await self._get_tool_result_if_present(call_id)
                if (
                    existing is not None
                    and existing.status is ToolCallStatus.REJECTED
                    and existing.error is not None
                    and existing.error.message == safe_reason
                ):
                    return existing
                raise StateError("tool call already has a conflicting result") from error
            except sqlite3.OperationalError as error:
                if "locked" not in str(error).lower() or attempt == 2:
                    raise
        raise InternalError("tool rejection retry loop exhausted")

    async def _reject_tool_call_once(
        self,
        call_id: str,
        *,
        reason: str,
        completed_at: datetime,
    ) -> ToolResult:
        """Apply one atomic pre-execution rejection attempt."""
        async with self.transaction() as connection:
            row = await self._fetch_one(
                "SELECT * FROM tool_calls WHERE call_id = ?", (call_id,)
            )
            if row is None:
                raise MissingEntityError(f"tool call {call_id} is not persisted")
            existing_result_row = await self._fetch_one(
                "SELECT * FROM tool_results WHERE call_id = ?", (call_id,)
            )
            if existing_result_row is not None:
                existing = self._row_to_tool_result(existing_result_row)
                if (
                    existing.status is ToolCallStatus.REJECTED
                    and existing.error is not None
                    and existing.error.message == reason
                ):
                    return existing
                raise StateError("tool call already has a conflicting result")

            current = self._row_to_tool_call(row)
            if current.status not in (
                ToolCallStatus.REQUESTED,
                ToolCallStatus.AWAITING_APPROVAL,
            ):
                raise StateError(
                    f"cannot reject tool call in {current.status.value} state"
                )
            transitioned = current.transition(ToolCallStatus.REJECTED)
            rejected = ToolResult.from_rejected_call(
                current,
                reason=reason,
                completed_at=completed_at,
            )
            assert rejected.error is not None
            try:
                await connection.execute(
                    """
                    UPDATE tool_calls
                       SET status = ?, completed_at = ?
                     WHERE call_id = ?
                    """,
                    (
                        transitioned.status.value,
                        _require_iso(rejected.completed_at),
                        call_id,
                    ),
                )
                await connection.execute(
                    """
                    INSERT INTO tool_results (
                        call_id, status, result_json, output, truncated,
                        changed_paths_json, exit_code, error_category,
                        error_message, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        call_id,
                        rejected.status.value,
                        json.dumps(rejected.result, sort_keys=True, allow_nan=False),
                        rejected.output,
                        int(rejected.truncated),
                        json.dumps(list(rejected.changed_paths), sort_keys=True),
                        rejected.exit_code,
                        rejected.error.category.value,
                        rejected.error.message,
                        _require_iso(rejected.completed_at),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "tool result") from error
            await self.append_event(
                current.run_id,
                EventType.TOOL_CALL_REJECTED,
                payload_from_tool_result(rejected),
                timestamp=rejected.completed_at,
            )
            return rejected

    async def _get_tool_result_if_present(self, call_id: str) -> ToolResult | None:
        row = await self._fetch_one("SELECT * FROM tool_results WHERE call_id = ?", (call_id,))
        return None if row is None else self._row_to_tool_result(row)

    async def _complete_tool_call_once(self, completed: ToolResult) -> ToolResult:
        """Apply one completion attempt; the caller retries transient SQLite locks."""
        call_id = completed.call_id
        async with self.transaction() as connection:
            row = await self._fetch_one(
                "SELECT * FROM tool_calls WHERE call_id = ?", (call_id,)
            )
            if row is None:
                raise MissingEntityError(f"tool call {call_id} is not persisted")
            existing_result_row = await self._fetch_one(
                "SELECT * FROM tool_results WHERE call_id = ?", (call_id,)
            )
            if existing_result_row is not None:
                existing_result = self._row_to_tool_result(existing_result_row)
                if existing_result == completed:
                    return existing_result
                raise StateError("tool call already has a conflicting result")

            current = self._row_to_tool_call(row)
            if current.status is not ToolCallStatus.RUNNING:
                raise StateError(
                    f"cannot complete tool call in {current.status.value} state"
                )
            transitioned = current.transition(completed.status)
            try:
                await connection.execute(
                    """
                    UPDATE tool_calls
                       SET status = ?, completed_at = ?
                     WHERE call_id = ?
                    """,
                    (
                        transitioned.status.value,
                        _require_iso(completed.completed_at),
                        call_id,
                    ),
                )
                await connection.execute(
                    """
                    INSERT INTO tool_results (
                        call_id, status, result_json, output, truncated,
                        changed_paths_json, exit_code, error_category,
                        error_message, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        call_id,
                        completed.status.value,
                        None
                        if completed.result is None
                        else json.dumps(completed.result, sort_keys=True, allow_nan=False),
                        completed.output,
                        int(completed.truncated),
                        json.dumps(list(completed.changed_paths), sort_keys=True),
                        completed.exit_code,
                        None if completed.error is None else completed.error.category.value,
                        None if completed.error is None else completed.error.message,
                        _require_iso(completed.completed_at),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "tool result") from error
            await self.append_event(
                current.run_id,
                _TOOL_EVENT_BY_STATUS[completed.status],
                payload_from_tool_result(completed),
                timestamp=completed.completed_at,
            )
            return completed

    async def mark_tool_call_unknown(
        self,
        call_id: str,
        *,
        completed_at: datetime | None = None,
        message: str = "tool outcome is unconfirmed after restart",
    ) -> ToolResult:
        """Close a running call as UNKNOWN without guessing its side effect."""
        call = await self.get_tool_call(call_id)
        if call.status is not ToolCallStatus.RUNNING:
            raise StateError(f"cannot mark tool call in {call.status.value} state as UNKNOWN")
        result = ToolResult.from_call(
            call,
            status=ToolCallStatus.UNKNOWN,
            error=ErrorInfo(category=ErrorCategory.UNKNOWN, message=message),
            completed_at=completed_at or utc_now(),
        )
        return await self.complete_tool_call(result)

    async def get_tool_call(self, call_id: str) -> ToolCall:
        """Read one persisted tool call."""
        row = await self._fetch_one("SELECT * FROM tool_calls WHERE call_id = ?", (call_id,))
        if row is None:
            raise MissingEntityError(f"tool call {call_id} is not persisted")
        return self._row_to_tool_call(row)

    async def get_tool_result(self, call_id: str) -> ToolResult:
        """Read the one terminal result for a tool call."""
        row = await self._fetch_one("SELECT * FROM tool_results WHERE call_id = ?", (call_id,))
        if row is None:
            raise MissingEntityError(f"tool result for {call_id} is not persisted")
        return self._row_to_tool_result(row)

    async def list_tool_calls(
        self,
        run_id: str,
        *,
        work_unit_id: str | None = None,
        statuses: Iterable[ToolCallStatus] | None = None,
    ) -> tuple[ToolCall, ...]:
        """List one run's calls in request order, optionally filtered."""
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if work_unit_id is not None:
            clauses.append("work_unit_id = ?")
            params.append(work_unit_id)
        if statuses is not None:
            values = [status.value for status in statuses]
            if not values:
                return ()
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"status IN ({placeholders})")
            params.extend(values)
        rows = await self._fetch_all(
            "SELECT * FROM tool_calls WHERE "
            + " AND ".join(clauses)
            + " ORDER BY requested_at, call_id",
            params,
        )
        return tuple(self._row_to_tool_call(row) for row in rows)

    async def list_tool_calls_with_status(
        self, statuses: Iterable[ToolCallStatus]
    ) -> tuple[ToolCall, ...]:
        """List tool calls in the supplied lifecycle states across all runs."""
        values = [status.value for status in statuses]
        if not values:
            return ()
        placeholders = ", ".join("?" for _ in values)
        rows = await self._fetch_all(
            f"SELECT * FROM tool_calls WHERE status IN ({placeholders}) "
            "ORDER BY requested_at, call_id",
            values,
        )
        return tuple(self._row_to_tool_call(row) for row in rows)

    async def list_tool_calls_for_attempt(
        self,
        attempt_id: str,
        *,
        statuses: Iterable[ToolCallStatus] | None = None,
    ) -> tuple[ToolCall, ...]:
        """List calls in the run/work-unit scope recorded by one attempt.

        ToolCall intentionally has no attempt foreign key. Callers must still
        reconcile these facts with the attempt's public history before acting.
        """
        attempt = await self.get_attempt(attempt_id)
        calls = await self.list_tool_calls(
            attempt.run_id,
            work_unit_id=attempt.work_unit_id,
            statuses=statuses,
        )
        return tuple(call for call in calls if call.requested_at >= attempt.created_at)

    async def list_tool_calls_for_session(
        self,
        session_id: str,
        run_id: str,
        *,
        principal_id: str,
        statuses: Iterable[ToolCallStatus] | None = None,
    ) -> tuple[ToolCall, ...]:
        """List tool calls after enforcing the Session owner boundary."""
        clauses = [
            "tc.run_id = ?",
            "sr.session_id = ?",
            "s.principal_id = ?",
            "w.owner_id = ?",
        ]
        params: list[Any] = [run_id, session_id, principal_id, principal_id]
        if statuses is not None:
            values = [status.value for status in statuses]
            if not values:
                return ()
            placeholders = ", ".join("?" for _ in values)
            clauses.append(f"tc.status IN ({placeholders})")
            params.extend(values)
        rows = await self._fetch_all(
            """
            SELECT tc.* FROM tool_calls AS tc
            JOIN session_runs AS sr ON sr.run_id = tc.run_id
            JOIN sessions AS s ON s.session_id = sr.session_id
            JOIN workspaces AS w ON w.workspace_id = tc.workspace_id
            WHERE """
            + " AND ".join(clauses)
            + " ORDER BY tc.requested_at, tc.call_id",
            params,
        )
        return tuple(self._row_to_tool_call(row) for row in rows)

    async def get_tool_call_for_session(
        self, session_id: str, run_id: str, call_id: str, *, principal_id: str
    ) -> ToolCall:
        """Read one ToolCall only through its Session owner scope."""
        row = await self._fetch_one(
            """
            SELECT tc.* FROM tool_calls AS tc
            JOIN session_runs AS sr ON sr.run_id = tc.run_id
            JOIN sessions AS s ON s.session_id = sr.session_id
            JOIN workspaces AS w ON w.workspace_id = tc.workspace_id
            WHERE tc.call_id = ? AND tc.run_id = ? AND sr.session_id = ?
              AND s.principal_id = ? AND w.owner_id = ?
            """,
            (call_id, run_id, session_id, principal_id, principal_id),
        )
        if row is None:
            raise MissingEntityError(f"tool call {call_id} is not in the authenticated session")
        return self._row_to_tool_call(row)

    async def get_tool_result_for_session(
        self, session_id: str, run_id: str, call_id: str, *, principal_id: str
    ) -> ToolResult:
        """Read one ToolResult only through its Session owner scope."""
        row = await self._fetch_one(
            """
            SELECT tr.* FROM tool_results AS tr
            JOIN tool_calls AS tc ON tc.call_id = tr.call_id
            JOIN session_runs AS sr ON sr.run_id = tc.run_id
            JOIN sessions AS s ON s.session_id = sr.session_id
            JOIN workspaces AS w ON w.workspace_id = tc.workspace_id
            WHERE tr.call_id = ? AND tc.run_id = ? AND sr.session_id = ?
              AND s.principal_id = ? AND w.owner_id = ?
            """,
            (call_id, run_id, session_id, principal_id, principal_id),
        )
        if row is None:
            raise MissingEntityError(
                f"tool result for {call_id} is not in the authenticated session"
            )
        return self._row_to_tool_result(row)

    # --- Native Bridge claims -----------------------------------------------

    async def claim_tool_call(
        self,
        session_id: str,
        run_id: str,
        call_id: str,
        *,
        principal_id: str,
        claimant_id: str,
        idempotency_key: str,
        claimed_at: datetime | None = None,
        expires_at: datetime | None = None,
        fingerprint: str | None = None,
    ) -> BridgeClaim:
        """Atomically create or replay one owner-scoped Bridge claim.

        The call must already be RUNNING in a BRIDGE Session. An expired active
        claim is closed before a new idempotency key can claim the call, so two
        clients cannot observe simultaneous lease ownership.
        """
        if (
            not idempotency_key
            or idempotency_key != idempotency_key.strip()
            or len(idempotency_key) > 128
        ):
            raise ValueError("Idempotency-Key must be 1 to 128 non-whitespace characters")
        if not claimant_id or claimant_id != claimant_id.strip() or len(claimant_id) > 128:
            raise ValueError("claimant_id must be 1 to 128 non-whitespace characters")
        now = claimed_at or utc_now()
        expiry = expires_at or (now + timedelta(seconds=60))
        if now.tzinfo is None or expiry.tzinfo is None:
            raise ValueError("Bridge claim timestamps must be timezone-aware")
        if expiry <= now:
            raise ValueError("Bridge claim expiry must be after claimed_at")
        async with self.transaction() as connection:
            existing = await self._fetch_one(
                """
                SELECT bc.*
                  FROM bridge_claims AS bc
                  JOIN sessions AS s ON s.session_id = bc.session_id
                  JOIN workspaces AS w ON w.workspace_id = bc.workspace_id
                 WHERE bc.session_id = ? AND bc.run_id = ?
                   AND bc.idempotency_key = ?
                   AND s.principal_id = ? AND w.owner_id = ?
                """,
                (session_id, run_id, idempotency_key, principal_id, principal_id),
            )
            if existing is not None:
                replay = self._row_to_bridge_claim(existing)
                if replay.call_id != call_id or replay.claimant_id != claimant_id:
                    raise DuplicateEntityError(
                        "Idempotency-Key already belongs to a different Bridge claim"
                    )
                if fingerprint is not None and replay.fingerprint != fingerprint:
                    raise DuplicateEntityError(
                        "Bridge claim fingerprint conflicts with the Idempotency-Key"
                    )
                if replay.status is BridgeClaimStatus.ACTIVE and replay.expires_at <= now:
                    expired = replay.expire(at=now)
                    await connection.execute(
                        "UPDATE bridge_claims SET status = ?, closed_at = ? "
                        "WHERE claim_id = ? AND status = 'ACTIVE'",
                        (expired.status.value, _require_iso(now), replay.claim_id),
                    )
                    await self.append_event(
                        replay.run_id,
                        EventType.BRIDGE_CLAIM_EXPIRED,
                        payload_from_bridge_claim(expired),
                        timestamp=now,
                    )
                    return expired
                return replay

            row = await self._fetch_one(
                """
                SELECT tc.*,
                       s.status AS session_status,
                       s.principal_id AS session_principal_id,
                       s.workspace_id AS session_workspace_id,
                       s.agent_options_json AS session_agent_options_json,
                       s.expires_at AS session_expires_at,
                       w.status AS workspace_status,
                       w.owner_id AS workspace_owner_id
                  FROM tool_calls AS tc
                  JOIN session_runs AS sr ON sr.run_id = tc.run_id
                  JOIN sessions AS s ON s.session_id = sr.session_id
                  JOIN workspaces AS w ON w.workspace_id = tc.workspace_id
                 WHERE tc.call_id = ? AND tc.run_id = ?
                   AND sr.session_id = ? AND s.workspace_id = tc.workspace_id
                   AND s.principal_id = ? AND w.owner_id = ?
                """,
                (call_id, run_id, session_id, principal_id, principal_id),
            )
            if row is None:
                raise MissingEntityError("tool call is outside the authenticated Bridge scope")
            if SessionStatus(str(row["session_status"])) is not SessionStatus.ACTIVE:
                raise StateError("session is not active")
            session_expires_at = _parse_datetime(row["session_expires_at"])
            if session_expires_at is not None and session_expires_at <= now:
                raise StateError("session grant has expired")
            if WorkspaceStatus(str(row["workspace_status"])) is not WorkspaceStatus.ACTIVE:
                raise StateError("workspace is not active")
            options_value = strict_json_loads(str(row["session_agent_options_json"]))
            if not isinstance(options_value, dict):
                raise InternalError("stored session agent options are not an object")
            options = AgentRequestOptions.model_validate(options_value)
            if options.execution_location.value != "BRIDGE":
                raise StateError("Bridge claim requires a BRIDGE execution location")

            call = self._row_to_tool_call(row)
            if call.status is not ToolCallStatus.RUNNING:
                raise StateError(
                    f"cannot claim tool call in {call.status.value} state"
                )

            active_row = await self._fetch_one(
                "SELECT * FROM bridge_claims WHERE call_id = ? AND status = 'ACTIVE'",
                (call_id,),
            )
            if active_row is not None:
                active = self._row_to_bridge_claim(active_row)
                if active.expires_at <= now:
                    expired = active.expire(at=now)
                    await connection.execute(
                        "UPDATE bridge_claims SET status = ?, closed_at = ? "
                        "WHERE claim_id = ? AND status = 'ACTIVE'",
                        (expired.status.value, _require_iso(now), active.claim_id),
                    )
                    await self.append_event(
                        active.run_id,
                        EventType.BRIDGE_CLAIM_EXPIRED,
                        payload_from_bridge_claim(expired),
                        timestamp=now,
                    )
                else:
                    raise DuplicateEntityError("tool call already has an active Bridge claim")

            claim_fingerprint = fingerprint or _bridge_claim_fingerprint(
                call_id=call_id,
                run_id=run_id,
                session_id=session_id,
                workspace_id=str(row["workspace_id"]),
                owner_id=principal_id,
                claimant_id=claimant_id,
                idempotency_key=idempotency_key,
            )
            claim = BridgeClaim(
                call_id=call_id,
                run_id=run_id,
                session_id=session_id,
                workspace_id=str(row["workspace_id"]),
                owner_id=principal_id,
                claimant_id=claimant_id,
                idempotency_key=idempotency_key,
                fingerprint=claim_fingerprint,
                claimed_at=now,
                expires_at=expiry,
            )
            try:
                await connection.execute(
                    """
                    INSERT INTO bridge_claims (
                        claim_id, call_id, run_id, session_id, workspace_id, owner_id,
                        claimant_id, idempotency_key, fingerprint, status, claimed_at,
                        expires_at, closed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim.claim_id,
                        claim.call_id,
                        claim.run_id,
                        claim.session_id,
                        claim.workspace_id,
                        claim.owner_id,
                        claim.claimant_id,
                        claim.idempotency_key,
                        claim.fingerprint,
                        claim.status.value,
                        _require_iso(claim.claimed_at),
                        _require_iso(claim.expires_at),
                        None,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "Bridge claim") from error
            await self.append_event(
                claim.run_id,
                EventType.BRIDGE_CLAIM_CREATED,
                payload_from_bridge_claim(claim),
                timestamp=claim.claimed_at,
            )
            return claim

    async def get_bridge_claim(self, claim_id: str, *, principal_id: str) -> BridgeClaim:
        """Read a claim only through its authenticated owner boundary."""
        row = await self._fetch_one(
            """
            SELECT bc.*
              FROM bridge_claims AS bc
              JOIN sessions AS s ON s.session_id = bc.session_id
              JOIN workspaces AS w ON w.workspace_id = bc.workspace_id
             WHERE bc.claim_id = ? AND bc.owner_id = ?
               AND s.principal_id = ? AND w.owner_id = ?
            """,
            (claim_id, principal_id, principal_id, principal_id),
        )
        if row is None:
            raise MissingEntityError("Bridge claim is outside the authenticated scope")
        return self._row_to_bridge_claim(row)

    async def get_bridge_claim_for_session(
        self,
        session_id: str,
        run_id: str,
        call_id: str,
        *,
        principal_id: str,
    ) -> BridgeClaim:
        """Read the claim for one session/run/call scope."""
        row = await self._fetch_one(
            """
            SELECT bc.*
              FROM bridge_claims AS bc
              JOIN sessions AS s ON s.session_id = bc.session_id
              JOIN workspaces AS w ON w.workspace_id = bc.workspace_id
             WHERE bc.session_id = ? AND bc.run_id = ? AND bc.call_id = ?
               AND bc.owner_id = ? AND s.principal_id = ? AND w.owner_id = ?
            """,
            (session_id, run_id, call_id, principal_id, principal_id, principal_id),
        )
        if row is None:
            raise MissingEntityError("Bridge claim is outside the authenticated scope")
        return self._row_to_bridge_claim(row)

    async def expire_bridge_claim(
        self,
        claim_id: str,
        *,
        principal_id: str,
        at: datetime,
    ) -> BridgeClaim:
        """Expire one owner-scoped active claim idempotently."""
        if at.tzinfo is None:
            raise ValueError("Bridge claim expiry time must be timezone-aware")
        async with self.transaction() as connection:
            row = await self._fetch_one(
                """
                SELECT bc.*
                  FROM bridge_claims AS bc
                  JOIN sessions AS s ON s.session_id = bc.session_id
                  JOIN workspaces AS w ON w.workspace_id = bc.workspace_id
                 WHERE bc.claim_id = ? AND bc.owner_id = ?
                   AND s.principal_id = ? AND w.owner_id = ?
                """,
                (claim_id, principal_id, principal_id, principal_id),
            )
            if row is None:
                raise MissingEntityError("Bridge claim is outside the authenticated scope")
            current = self._row_to_bridge_claim(row)
            if current.status is BridgeClaimStatus.EXPIRED:
                return current
            expired = current.expire(at=at)
            await connection.execute(
                "UPDATE bridge_claims SET status = ?, closed_at = ? "
                "WHERE claim_id = ? AND status = 'ACTIVE'",
                (expired.status.value, _require_iso(at), claim_id),
            )
            await self.append_event(
                current.run_id,
                EventType.BRIDGE_CLAIM_EXPIRED,
                payload_from_bridge_claim(expired),
                timestamp=at,
            )
            return expired

    async def submit_bridge_tool_result(
        self,
        session_id: str,
        run_id: str,
        call_id: str,
        result: ToolResult,
        *,
        principal_id: str,
        claimant_id: str,
        settled_at: datetime | None = None,
    ) -> tuple[ToolResult, bool]:
        """Complete a claimed call and settle its lease in one transaction.

        The boolean is true only for a replay of the already durable result.
        Callers use it to avoid emitting a second in-memory wake-up.
        """
        if result.call_id != call_id:
            raise ValueError("tool result call_id does not match the route")
        if result.status not in (
            ToolCallStatus.SUCCEEDED,
            ToolCallStatus.FAILED,
            ToolCallStatus.CANCELLED,
        ):
            raise StateError("Bridge may submit only a provable terminal result")
        if settled_at is not None and settled_at.tzinfo is None:
            raise ValueError("Bridge settlement time must be timezone-aware")
        for attempt in range(3):
            try:
                return await self._submit_bridge_tool_result_once(
                    session_id,
                    run_id,
                    call_id,
                    result,
                    principal_id=principal_id,
                    claimant_id=claimant_id,
                    settled_at=settled_at,
                )
            except DuplicateEntityError as error:
                existing = await self._get_tool_result_if_present(call_id)
                if existing is not None:
                    if _bridge_results_match(existing, result):
                        return existing, True
                    raise StateError("tool call already has a conflicting result") from error
                raise
            except sqlite3.OperationalError as error:
                if "locked" not in str(error).lower() or attempt == 2:
                    raise
        raise InternalError("Bridge result completion retry loop exhausted")

    async def _submit_bridge_tool_result_once(
        self,
        session_id: str,
        run_id: str,
        call_id: str,
        result: ToolResult,
        *,
        principal_id: str,
        claimant_id: str,
        settled_at: datetime | None,
    ) -> tuple[ToolResult, bool]:
        """Apply one atomic Bridge result submission attempt."""
        async with self.transaction() as connection:
            claim_row = await self._fetch_one(
                """
                SELECT bc.*, s.status AS session_status,
                       s.expires_at AS session_expires_at,
                       w.status AS workspace_status
                  FROM bridge_claims AS bc
                  JOIN sessions AS s ON s.session_id = bc.session_id
                  JOIN workspaces AS w ON w.workspace_id = bc.workspace_id
                 WHERE bc.session_id = ? AND bc.run_id = ? AND bc.call_id = ?
                   AND bc.owner_id = ? AND bc.claimant_id = ?
                   AND s.principal_id = ? AND w.owner_id = ?
                """,
                (
                    session_id,
                    run_id,
                    call_id,
                    principal_id,
                    claimant_id,
                    principal_id,
                    principal_id,
                ),
            )
            if claim_row is None:
                raise MissingEntityError("Bridge claim is outside the authenticated scope")
            claim = self._row_to_bridge_claim(claim_row)
            settlement_time = settled_at or result.completed_at
            if SessionStatus(str(claim_row["session_status"])) is not SessionStatus.ACTIVE:
                raise StateError("session is not active")
            session_expires_at = _parse_datetime(claim_row["session_expires_at"])
            if session_expires_at is not None and session_expires_at <= settlement_time:
                raise StateError("session grant has expired")
            if WorkspaceStatus(str(claim_row["workspace_status"])) is not WorkspaceStatus.ACTIVE:
                raise StateError("workspace is not active")
            existing_result_row = await self._fetch_one(
                "SELECT * FROM tool_results WHERE call_id = ?", (call_id,)
            )
            if existing_result_row is not None:
                existing = self._row_to_tool_result(existing_result_row)
                if not _bridge_results_match(existing, result):
                    raise StateError("tool call already has a conflicting result")
                if claim.status is BridgeClaimStatus.SETTLED:
                    return existing, True
                if claim.status is not BridgeClaimStatus.ACTIVE:
                    raise StateError("expired or released Bridge claim cannot submit a result")
                if not claim.is_active_at(settlement_time):
                    raise StateError("Bridge claim lease has expired")
                settled = claim.settle(at=settlement_time)
                await connection.execute(
                    "UPDATE bridge_claims SET status = ?, closed_at = ? "
                    "WHERE claim_id = ? AND status = 'ACTIVE'",
                    (settled.status.value, _require_iso(settlement_time), claim.claim_id),
                )
                await self.append_event(
                    claim.run_id,
                    EventType.BRIDGE_CLAIM_SETTLED,
                    payload_from_bridge_claim(settled),
                    timestamp=settlement_time,
                )
                return existing, True

            if claim.status is not BridgeClaimStatus.ACTIVE:
                raise StateError("expired or released Bridge claim cannot submit a result")
            if not claim.is_active_at(settlement_time):
                raise StateError("Bridge claim lease has expired")
            call_row = await self._fetch_one(
                "SELECT * FROM tool_calls WHERE call_id = ? AND run_id = ?",
                (call_id, run_id),
            )
            if call_row is None:
                raise MissingEntityError("tool call is not persisted")
            current = self._row_to_tool_call(call_row)
            if current.status is not ToolCallStatus.RUNNING:
                raise StateError(
                    f"cannot complete tool call in {current.status.value} state"
                )
            transitioned = current.transition(result.status)
            try:
                await connection.execute(
                    "UPDATE tool_calls SET status = ?, completed_at = ? WHERE call_id = ?",
                    (
                        transitioned.status.value,
                        _require_iso(result.completed_at),
                        call_id,
                    ),
                )
                await connection.execute(
                    """
                    INSERT INTO tool_results (
                        call_id, status, result_json, output, truncated,
                        changed_paths_json, exit_code, error_category,
                        error_message, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        call_id,
                        result.status.value,
                        None
                        if result.result is None
                        else json.dumps(result.result, sort_keys=True, allow_nan=False),
                        result.output,
                        int(result.truncated),
                        json.dumps(list(result.changed_paths), sort_keys=True),
                        result.exit_code,
                        None if result.error is None else result.error.category.value,
                        None if result.error is None else result.error.message,
                        _require_iso(result.completed_at),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "Bridge tool result") from error
            await self.append_event(
                run_id,
                _TOOL_EVENT_BY_STATUS[result.status],
                payload_from_tool_result(result),
                timestamp=result.completed_at,
            )
            settled = claim.settle(at=settlement_time)
            await connection.execute(
                "UPDATE bridge_claims SET status = ?, closed_at = ? "
                "WHERE claim_id = ? AND status = 'ACTIVE'",
                (settled.status.value, _require_iso(settlement_time), claim.claim_id),
            )
            await self.append_event(
                run_id,
                EventType.BRIDGE_CLAIM_SETTLED,
                payload_from_bridge_claim(settled),
                timestamp=settlement_time,
            )
            return result, False

    # --- approvals and leases --------------------------------------------------

    async def create_approval(
        self,
        request: ApprovalRequest,
        *,
        owner_id: str,
    ) -> ApprovalRequest:
        """Persist an owner-scoped approval request and its audit event."""
        _safe_policy_text(request.reason, "approval reason")
        async with self.transaction() as connection:
            existing = await self._fetch_one(
                "SELECT * FROM approvals WHERE request_id = ?", (request.request_id,)
            )
            if existing is not None:
                if str(existing["owner_id"]) != owner_id:
                    raise MissingEntityError("approval request is outside the owner scope")
                existing_request = self._row_to_approval_request(existing)
                if existing_request == request:
                    return existing_request
                raise DuplicateEntityError("approval request is already persisted")

            context = await self._fetch_one(
                """
                SELECT tc.workspace_id, tc.run_id, tc.tool_name, tc.effect, w.owner_id
                  FROM tool_calls AS tc
                  JOIN workspaces AS w ON w.workspace_id = tc.workspace_id
                 WHERE tc.call_id = ? AND tc.run_id = ? AND tc.workspace_id = ?
                """,
                (request.call_id, request.run_id, request.workspace_id),
            )
            if context is None or str(context["owner_id"]) != owner_id:
                raise MissingEntityError("tool call is outside the owner scope")
            if (
                str(context["tool_name"]) != request.tool_name
                or str(context["effect"]) != request.effect.value
            ):
                raise ValueError("approval request does not match the persisted tool call")
            try:
                await connection.execute(
                    """
                    INSERT INTO approvals (
                        request_id, call_id, run_id, workspace_id, owner_id,
                        tool_name, effect, scope_json, reason, issuer, requested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.request_id,
                        request.call_id,
                        request.run_id,
                        request.workspace_id,
                        owner_id,
                        request.tool_name,
                        request.effect.value,
                        _scope_json(request.scope),
                        request.reason,
                        request.issuer.value,
                        _require_iso(request.requested_at),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "approval request") from error
            await self.append_event(
                request.run_id,
                EventType.APPROVAL_REQUESTED,
                {
                    "approval_id": request.request_id,
                    "request_id": request.request_id,
                    "call_id": request.call_id,
                    "workspace_id": request.workspace_id,
                    "tool_name": request.tool_name,
                    "effect": request.effect.value,
                    "issuer": request.issuer.value,
                },
                timestamp=request.requested_at,
            )
        return request

    async def create_approval_request(
        self, request: ApprovalRequest, *, owner_id: str
    ) -> ApprovalRequest:
        """Explicit-name alias for ``create_approval``."""
        return await self.create_approval(request, owner_id=owner_id)

    async def get_approval(self, request_id: str, *, owner_id: str) -> ApprovalRequest:
        row = await self._fetch_one(
            """
            SELECT a.*
              FROM approvals AS a
              JOIN workspaces AS w ON w.workspace_id = a.workspace_id
             WHERE a.request_id = ? AND a.owner_id = ? AND w.owner_id = ?
            """,
            (request_id, owner_id, owner_id),
        )
        if row is None:
            raise MissingEntityError(f"approval request {request_id} is not persisted")
        return self._row_to_approval_request(row)

    async def get_approval_request(
        self, request_id: str, *, owner_id: str
    ) -> ApprovalRequest:
        """Explicit-name alias for ``get_approval``."""
        return await self.get_approval(request_id, owner_id=owner_id)

    async def get_approval_decision(
        self, request_id: str, *, owner_id: str
    ) -> ApprovalDecision:
        """Read the immutable decision through the same owner boundary."""
        row = await self._fetch_one(
            """
            SELECT a.* FROM approvals AS a JOIN workspaces AS w
              ON w.workspace_id = a.workspace_id
             WHERE a.request_id = ? AND a.owner_id = ? AND w.owner_id = ?
            """,
            (request_id, owner_id, owner_id),
        )
        if row is None or row["outcome"] is None:
            raise MissingEntityError(f"approval decision for {request_id} is not persisted")
        return self._row_to_approval_decision(row)

    async def list_approvals(
        self,
        *,
        owner_id: str,
        run_id: str | None = None,
        call_id: str | None = None,
    ) -> tuple[ApprovalRequest, ...]:
        clauses = ["a.owner_id = ?", "w.owner_id = ?"]
        params: list[Any] = [owner_id, owner_id]
        if run_id is not None:
            clauses.append("a.run_id = ?")
            params.append(run_id)
        if call_id is not None:
            clauses.append("a.call_id = ?")
            params.append(call_id)
        rows = await self._fetch_all(
            "SELECT a.* FROM approvals AS a JOIN workspaces AS w "
            "ON w.workspace_id = a.workspace_id WHERE "
            + " AND ".join(clauses)
            + " ORDER BY a.requested_at, a.request_id",
            params,
        )
        return tuple(self._row_to_approval_request(row) for row in rows)

    async def decide_approval(
        self,
        request_id: str,
        decision: ApprovalDecision,
        *,
        owner_id: str,
    ) -> ApprovalDecision:
        """Record one immutable decision, idempotently replaying the same fact."""
        if decision.approval_request_id != request_id:
            raise ValueError("approval decision does not match the request")
        if decision.reason is not None:
            _safe_policy_text(decision.reason, "approval decision reason")
        async with self.transaction() as connection:
            row = await self._fetch_one(
                """
                SELECT a.* FROM approvals AS a JOIN workspaces AS w
                  ON w.workspace_id = a.workspace_id
                 WHERE a.request_id = ? AND a.owner_id = ? AND w.owner_id = ?
                """,
                (request_id, owner_id, owner_id),
            )
            if row is None:
                raise MissingEntityError(f"approval request {request_id} is not persisted")
            current = self._row_to_approval_request(row)
            if row["outcome"] is not None:
                existing = self._row_to_approval_decision(row)
                if (
                    existing.outcome is decision.outcome
                    and existing.issuer is decision.issuer
                    and existing.reason == decision.reason
                ):
                    return existing
                raise StateError("approval decision is immutable and conflicts")
            try:
                await connection.execute(
                    """
                    UPDATE approvals
                       SET outcome = ?, decision_issuer = ?, decision_reason = ?, decided_at = ?
                     WHERE request_id = ? AND outcome IS NULL
                    """,
                    (
                        decision.outcome.value,
                        decision.issuer.value,
                        decision.reason,
                        _require_iso(decision.decided_at),
                        request_id,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "approval decision") from error
            await self.append_event(
                current.run_id,
                EventType.APPROVAL_DECIDED,
                {
                    "approval_id": request_id,
                    "request_id": request_id,
                    "outcome": decision.outcome.value,
                    "issuer": decision.issuer.value,
                    "reason": decision.reason,
                },
                timestamp=decision.decided_at,
            )
            return decision

    async def create_lease(self, lease: Lease, *, owner_id: str) -> Lease:
        """Persist an active lease only for an owner-approved request."""
        if lease.status is not LeaseStatus.ACTIVE:
            raise StateError("a new lease must be ACTIVE")
        if lease.close_reason is not None:
            _safe_policy_text(lease.close_reason, "lease close reason")
        async with self.transaction() as connection:
            existing = await self._fetch_one(
                "SELECT * FROM leases WHERE lease_id = ?", (lease.lease_id,)
            )
            if existing is not None:
                if str(existing["owner_id"]) != owner_id:
                    raise MissingEntityError("lease is outside the owner scope")
                existing_lease = self._row_to_lease(existing)
                if existing_lease == lease:
                    return existing_lease
                raise DuplicateEntityError("lease is already persisted")
            approval_row = await self._fetch_one(
                """
                SELECT a.* FROM approvals AS a JOIN workspaces AS w
                  ON w.workspace_id = a.workspace_id
                 WHERE a.request_id = ? AND a.owner_id = ? AND w.owner_id = ?
                """,
                (lease.approval_request_id, owner_id, owner_id),
            )
            if approval_row is None:
                raise MissingEntityError("approval request is outside the owner scope")
            approval = self._row_to_approval_request(approval_row)
            if approval_row["outcome"] != ApprovalOutcome.ALLOW.value:
                raise StateError("lease requires an ALLOW approval decision")
            if (
                approval.call_id != lease.call_id
                or not lease.scope.is_subset_of(approval.scope)
                or approval.workspace_id != lease.scope.workspace_id
            ):
                raise StateError("lease exceeds its approval scope")
            try:
                await connection.execute(
                    """
                    INSERT INTO leases (
                        lease_id, request_id, call_id, workspace_id, owner_id,
                        scope_json, issuer, issued_at, expires_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lease.lease_id,
                        lease.approval_request_id,
                        lease.call_id,
                        approval.workspace_id,
                        owner_id,
                        _scope_json(lease.scope),
                        lease.issuer.value,
                        _require_iso(lease.issued_at),
                        _require_iso(lease.expires_at),
                        lease.status.value,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "lease") from error
            await self.append_event(
                approval.run_id,
                EventType.LEASE_CREATED,
                {
                    "lease_id": lease.lease_id,
                    "approval_id": lease.approval_request_id,
                    "call_id": lease.call_id,
                    "status": lease.status.value,
                    "issuer": lease.issuer.value,
                },
                timestamp=lease.issued_at,
            )
        return lease

    async def get_lease(self, lease_id: str, *, owner_id: str) -> Lease:
        row = await self._fetch_one(
            "SELECT l.* FROM leases AS l JOIN workspaces AS w "
            "ON w.workspace_id = l.workspace_id "
            "WHERE l.lease_id = ? AND l.owner_id = ? AND w.owner_id = ?",
            (lease_id, owner_id, owner_id),
        )
        if row is None:
            raise MissingEntityError(f"lease {lease_id} is not persisted")
        return self._row_to_lease(row)

    async def get_active_lease(
        self,
        lease_id: str,
        *,
        owner_id: str,
        at: datetime,
    ) -> Lease:
        """Read a lease only when it is usable at the injected clock time."""
        lease = await self.get_lease(lease_id, owner_id=owner_id)
        if not lease.is_active_at(at):
            raise StateError("lease is expired or revoked")
        return lease

    async def list_leases(
        self,
        *,
        owner_id: str,
        run_id: str | None = None,
        call_id: str | None = None,
    ) -> tuple[Lease, ...]:
        clauses = ["l.owner_id = ?", "w.owner_id = ?"]
        params: list[Any] = [owner_id, owner_id]
        joins = "JOIN workspaces AS w ON w.workspace_id = l.workspace_id"
        if run_id is not None:
            joins += " JOIN tool_calls AS tc ON tc.call_id = l.call_id"
            clauses.append("tc.run_id = ?")
            params.append(run_id)
        if call_id is not None:
            clauses.append("l.call_id = ?")
            params.append(call_id)
        rows = await self._fetch_all(
            "SELECT l.* FROM leases AS l "
            + joins
            + " WHERE "
            + " AND ".join(clauses)
            + " ORDER BY l.issued_at, l.lease_id",
            params,
        )
        return tuple(self._row_to_lease(row) for row in rows)

    async def revoke_lease(
        self,
        lease_id: str,
        *,
        owner_id: str,
        at: datetime,
        reason: str,
    ) -> Lease:
        """Revoke one active lease and append a terminal event."""
        _safe_policy_text(reason, "lease close reason")
        async with self.transaction() as connection:
            row = await self._fetch_one(
                "SELECT l.* FROM leases AS l JOIN workspaces AS w "
                "ON w.workspace_id = l.workspace_id "
                "WHERE l.lease_id = ? AND l.owner_id = ? AND w.owner_id = ?",
                (lease_id, owner_id, owner_id),
            )
            if row is None:
                raise MissingEntityError(f"lease {lease_id} is not persisted")
            current = self._row_to_lease(row)
            if current.status is LeaseStatus.REVOKED:
                if current.closed_at == at and current.close_reason == reason:
                    return current
                raise StateError("lease revocation is immutable and conflicts")
            revoked = current.revoke(at=at, reason=reason)
            await connection.execute(
                "UPDATE leases SET status = ?, closed_at = ?, close_reason = ? "
                "WHERE lease_id = ? AND status = 'ACTIVE'",
                (revoked.status.value, _require_iso(at), reason, lease_id),
            )
            approval = await self.get_approval(current.approval_request_id, owner_id=owner_id)
            await self.append_event(
                approval.run_id,
                EventType.LEASE_REVOKED,
                {
                    "lease_id": lease_id,
                    "approval_id": current.approval_request_id,
                    "call_id": current.call_id,
                    "status": revoked.status.value,
                },
                timestamp=at,
            )
            return revoked

    async def expire_lease(
        self,
        lease_id: str,
        *,
        owner_id: str,
        at: datetime,
    ) -> Lease:
        """Close an active lease at or after its declared expiry."""
        async with self.transaction() as connection:
            row = await self._fetch_one(
                "SELECT l.* FROM leases AS l JOIN workspaces AS w "
                "ON w.workspace_id = l.workspace_id "
                "WHERE l.lease_id = ? AND l.owner_id = ? AND w.owner_id = ?",
                (lease_id, owner_id, owner_id),
            )
            if row is None:
                raise MissingEntityError(f"lease {lease_id} is not persisted")
            current = self._row_to_lease(row)
            if current.status is LeaseStatus.EXPIRED:
                return current
            expired = current.expire(at=at)
            await connection.execute(
                "UPDATE leases SET status = ?, closed_at = ? WHERE lease_id = ? "
                "AND status = 'ACTIVE'",
                (expired.status.value, _require_iso(at), lease_id),
            )
            approval = await self.get_approval(current.approval_request_id, owner_id=owner_id)
            await self.append_event(
                approval.run_id,
                EventType.LEASE_EXPIRED,
                {
                    "lease_id": lease_id,
                    "approval_id": current.approval_request_id,
                    "call_id": current.call_id,
                    "status": expired.status.value,
                },
                timestamp=at,
            )
            return expired

    @staticmethod
    def _row_to_approval_request(row: aiosqlite.Row) -> ApprovalRequest:
        return ApprovalRequest(
            request_id=str(row["request_id"]),
            call_id=str(row["call_id"]),
            run_id=str(row["run_id"]),
            workspace_id=str(row["workspace_id"]),
            tool_name=str(row["tool_name"]),
            effect=ToolEffect(str(row["effect"])),
            scope=_scope_from_json(row["scope_json"]),
            reason=str(row["reason"]),
            issuer=ApprovalIssuer(str(row["issuer"])),
            requested_at=_require_datetime(row["requested_at"]),
        )

    @staticmethod
    def _row_to_approval_decision(row: aiosqlite.Row) -> ApprovalDecision:
        if row["outcome"] is None or row["decision_issuer"] is None:
            raise InternalError("stored approval has no decision")
        return ApprovalDecision(
            approval_request_id=str(row["request_id"]),
            outcome=ApprovalOutcome(str(row["outcome"])),
            issuer=ApprovalIssuer(str(row["decision_issuer"])),
            reason=_optional_str(row["decision_reason"]),
            decided_at=_require_datetime(row["decided_at"]),
        )

    @staticmethod
    def _row_to_lease(row: aiosqlite.Row) -> Lease:
        return Lease(
            lease_id=str(row["lease_id"]),
            approval_request_id=str(row["request_id"]),
            call_id=str(row["call_id"]),
            scope=_scope_from_json(row["scope_json"]),
            issuer=ApprovalIssuer(str(row["issuer"])),
            issued_at=_require_datetime(row["issued_at"]),
            expires_at=_require_datetime(row["expires_at"]),
            status=LeaseStatus(str(row["status"])),
            closed_at=_parse_datetime(row["closed_at"]),
            close_reason=_optional_str(row["close_reason"]),
        )

    @staticmethod
    def _row_to_tool_call(row: aiosqlite.Row) -> ToolCall:
        arguments = strict_json_loads(str(row["arguments_json"]))
        if not isinstance(arguments, dict):
            raise InternalError("stored tool call arguments are not a JSON object")
        return ToolCall(
            call_id=str(row["call_id"]),
            run_id=str(row["run_id"]),
            work_unit_id=str(row["work_unit_id"]),
            tool_name=str(row["tool_name"]),
            effect=ToolEffect(str(row["effect"])),
            arguments=arguments,
            status=ToolCallStatus(str(row["status"])),
            snapshot_id=str(row["base_snapshot_id"]),
            requested_at=_require_datetime(row["requested_at"]),
        )

    @staticmethod
    def _row_to_tool_result(row: aiosqlite.Row) -> ToolResult:
        result_value = (
            None
            if row["result_json"] is None
            else strict_json_loads(str(row["result_json"]))
        )
        if result_value is not None and not isinstance(result_value, dict):
            raise InternalError("stored tool result is not a JSON object")
        changed_paths = strict_json_loads(str(row["changed_paths_json"]))
        if not isinstance(changed_paths, list) or not all(
            isinstance(path, str) for path in changed_paths
        ):
            raise InternalError("stored tool changed paths are not a string list")
        return ToolResult(
            call_id=str(row["call_id"]),
            status=ToolCallStatus(str(row["status"])),
            result=result_value,
            output=str(row["output"]),
            truncated=bool(int(row["truncated"])),
            changed_paths=tuple(changed_paths),
            exit_code=_optional_int(row["exit_code"]),
            error=_error_info(row["error_category"], row["error_message"]),
            completed_at=_require_datetime(row["completed_at"]),
        )

    @staticmethod
    def _row_to_bridge_claim(row: aiosqlite.Row) -> BridgeClaim:
        return BridgeClaim(
            claim_id=str(row["claim_id"]),
            call_id=str(row["call_id"]),
            run_id=str(row["run_id"]),
            session_id=str(row["session_id"]),
            workspace_id=str(row["workspace_id"]),
            owner_id=str(row["owner_id"]),
            claimant_id=str(row["claimant_id"]),
            idempotency_key=str(row["idempotency_key"]),
            fingerprint=str(row["fingerprint"]),
            status=BridgeClaimStatus(str(row["status"])),
            claimed_at=_require_datetime(row["claimed_at"]),
            expires_at=_require_datetime(row["expires_at"]),
            closed_at=_parse_datetime(row["closed_at"]),
        )

    @staticmethod
    def _row_to_workspace(row: aiosqlite.Row) -> Workspace:
        source_type = WorkspaceSourceType(str(row["source_type"]))
        source = WorkspaceSource(
            source_type=source_type,
            server_alias=_optional_str(row["server_alias"]),
            bridge_grant=_optional_str(row["bridge_grant"]),
        )
        return Workspace(
            workspace_id=str(row["workspace_id"]),
            owner_id=str(row["owner_id"]),
            alias=str(row["alias"]),
            source=source,
            status=WorkspaceStatus(str(row["status"])),
            created_at=_require_datetime(row["created_at"]),
            closed_at=_parse_datetime(row["closed_at"]),
        )

    @staticmethod
    def _row_to_session(row: aiosqlite.Row) -> Session:
        access = strict_json_loads(str(row["access_json"]))
        if not isinstance(access, list):
            raise InternalError("stored session access is not a JSON list")
        try:
            access_values = tuple(ResourceAccess(str(value)) for value in access)
        except ValueError as error:
            raise InternalError("stored session access contains an invalid value") from error
        options = strict_json_loads(str(row["agent_options_json"]))
        if not isinstance(options, dict):
            raise InternalError("stored session agent options are not a JSON object")
        created_at = _require_datetime(row["created_at"])
        return Session(
            session_id=str(row["session_id"]),
            principal_id=str(row["principal_id"]),
            workspace_id=str(row["workspace_id"]),
            grant=WorkspaceGrant(
                principal_id=str(row["principal_id"]),
                workspace_id=str(row["workspace_id"]),
                access=access_values,
                expires_at=_parse_datetime(row["expires_at"]),
            ),
            agent_options=AgentRequestOptions.model_validate(options),
            status=SessionStatus(str(row["status"])),
            created_at=created_at,
            expires_at=_parse_datetime(row["expires_at"]),
            revoked_at=_parse_datetime(row["revoked_at"]),
        )

    @staticmethod
    def _row_to_snapshot(row: aiosqlite.Row) -> Snapshot:
        return Snapshot(
            snapshot_id=str(row["snapshot_id"]),
            workspace_id=str(row["workspace_id"]),
            status=SnapshotStatus(str(row["status"])),
            created_at=_require_datetime(row["created_at"]),
            completed_at=_parse_datetime(row["completed_at"]),
            file_count=_optional_int(row["file_count"]),
            total_size=_optional_int(row["total_size"]),
        )

    async def list_runs_with_status(
        self, statuses: Iterable[RunStatus]
    ) -> tuple[Run, ...]:
        """Read runs in durable status order for recovery and supervision."""
        values = [status.value for status in statuses]
        if not values:
            return ()
        placeholders = ", ".join("?" for _ in values)
        rows = await self._fetch_all(
            f"SELECT * FROM runs WHERE status IN ({placeholders}) "
            "ORDER BY created_at, run_id",
            values,
        )
        return tuple(self._row_to_run(row) for row in rows)

    async def list_pending_runs(self) -> tuple[Run, ...]:
        """Read every pending run eligible for the normal supervisor scan."""
        return await self.list_runs_with_status([RunStatus.PENDING])

    async def list_recoverable_runs(self) -> tuple[Run, ...]:
        """Read pending runs and runs waiting on a durable approval decision."""
        rows = await self._fetch_all(
            """
            SELECT r.*
              FROM runs AS r
             WHERE r.status = 'PENDING'
                OR (
                    r.status = 'RUNNING'
                    AND EXISTS (
                        SELECT 1 FROM tool_calls AS tc
                         WHERE tc.run_id = r.run_id
                           AND tc.status = 'AWAITING_APPROVAL'
                    )
                )
             ORDER BY r.created_at, r.run_id
            """,
            (),
        )
        return tuple(self._row_to_run(row) for row in rows)

    async def get_run_usage(self, run_id: str) -> Usage:
        """Read a run's aggregated usage."""
        row = await self._fetch_one(
            """
            SELECT usage_input_tokens, usage_output_tokens,
                   usage_strong_model_tokens, usage_elapsed_ms
              FROM runs WHERE run_id = ?
            """,
            (run_id,),
        )
        if row is None:
            raise MissingEntityError(f"run {run_id} is not persisted", code=ErrorCode.RUN_NOT_FOUND)
        usage = _usage_or_none(row)
        if usage is None:
            raise InternalError(f"run {run_id} has no usage row")
        return usage

    async def add_run_usage(self, run_id: str, usage: Usage) -> Usage:
        """Add measured usage while preserving the metrics contract."""
        changed = await self._write(
            """
            UPDATE runs
               SET usage_input_tokens = usage_input_tokens + :input_tokens,
                   usage_output_tokens = usage_output_tokens + :output_tokens,
                   usage_strong_model_tokens = usage_strong_model_tokens + :strong_model_tokens,
                   usage_elapsed_ms = usage_elapsed_ms + :elapsed_ms
             WHERE run_id = :run_id
            """,
            {
                "run_id": run_id,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "strong_model_tokens": usage.strong_model_tokens,
                "elapsed_ms": usage.elapsed_ms,
            },
            "run usage",
        )
        if changed == 0:
            raise MissingEntityError(
                f"run {run_id} is not persisted", code=ErrorCode.RUN_NOT_FOUND
            )
        return await self.get_run_usage(run_id)

    async def add_run_metrics(
        self,
        run_id: str,
        *,
        usage: Usage | None,
        cost: AttemptCost | None,
    ) -> Usage:
        """Atomically add one attempt's usage and exact cost facts.

        Unknown usage permanently taints aggregate usage/cost availability for
        the run, while measured zero remains a known value.
        """
        if usage is None and cost is not None:
            raise ValueError("unknown usage cannot carry a known cost")
        async with self.transaction() as connection:
            row = await self._fetch_one(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            )
            if row is None:
                raise MissingEntityError(
                    f"run {run_id} is not persisted", code=ErrorCode.RUN_NOT_FOUND
                )
            observed = int(row["metrics_usage_observed"])
            usage_known = int(row["metrics_usage_known"])
            cost_known = int(row["metrics_cost_known"])
            observed_before = observed
            observed += 1
            if usage is None:
                usage_known = 0
                provider_elapsed = None
            elif usage_known or observed_before == 0:
                usage_known = 1
                provider_elapsed = (
                    usage.elapsed_ms
                    if observed_before == 0
                    else int(row["metrics_provider_elapsed_ms"] or 0) + usage.elapsed_ms
                )
            else:
                provider_elapsed = None

            new_cost: Decimal | None
            if cost is None:
                cost_known = 0
                new_cost = None
            elif cost_known or observed_before == 0:
                cost_known = 1
                previous = _optional_decimal(row["metrics_cost"])
                new_cost = (
                    cost.total_cost
                    if observed_before == 0
                    else (previous or Decimal(0)) + cost.total_cost
                )
            else:
                new_cost = None

            usage_sql = {
                "input_tokens": 0 if usage is None else usage.input_tokens,
                "output_tokens": 0 if usage is None else usage.output_tokens,
                "strong_model_tokens": 0 if usage is None else usage.strong_model_tokens,
                "elapsed_ms": 0 if usage is None else usage.elapsed_ms,
            }
            await connection.execute(
                """
                UPDATE runs
                   SET usage_input_tokens = usage_input_tokens + :input_tokens,
                       usage_output_tokens = usage_output_tokens + :output_tokens,
                       usage_strong_model_tokens = usage_strong_model_tokens + :strong_model_tokens,
                       usage_elapsed_ms = usage_elapsed_ms + :elapsed_ms,
                       metrics_usage_observed = :metrics_usage_observed,
                       metrics_usage_known = :metrics_usage_known,
                       metrics_provider_elapsed_ms = :metrics_provider_elapsed_ms,
                       metrics_cost_known = :metrics_cost_known,
                       metrics_cost = :metrics_cost
                 WHERE run_id = :run_id
                """,
                {
                    "run_id": run_id,
                    **usage_sql,
                    "metrics_usage_observed": observed,
                    "metrics_usage_known": usage_known,
                    "metrics_provider_elapsed_ms": provider_elapsed,
                    "metrics_cost_known": cost_known,
                    "metrics_cost": None if new_cost is None else str(new_cost),
                },
            )
        return await self.get_run_usage(run_id)

    async def record_run_metrics(
        self,
        run_id: str,
        *,
        usage: Usage | None,
        cost: AttemptCost | None,
        usage_already_accumulated: bool = False,
    ) -> None:
        """Record one attempt's metric availability and exact cost atomically."""
        if usage is None and cost is not None:
            raise ValueError("unknown usage cannot carry a known cost")
        async with self.transaction() as connection:
            row = await self._fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            if row is None:
                raise MissingEntityError(
                    f"run {run_id} is not persisted", code=ErrorCode.RUN_NOT_FOUND
                )
            observed_before = int(row["metrics_usage_observed"])
            usage_known_before = int(row["metrics_usage_known"])
            cost_known_before = int(row["metrics_cost_known"])
            usage_known: int = int(usage_known_before and usage is not None)
            provider_elapsed = _optional_int(row["metrics_provider_elapsed_ms"])
            if usage is None:
                usage_known = 0
                provider_elapsed = None
            elif not usage_known_before:
                usage_known = 0
                provider_elapsed = None
            elif usage_already_accumulated:
                provider_elapsed = int(row["usage_elapsed_ms"])
            else:
                provider_elapsed = (provider_elapsed or 0) + usage.elapsed_ms

            cost_known: int = int(cost_known_before and cost is not None)
            previous_cost = _optional_decimal(row["metrics_cost"])
            new_cost = previous_cost
            if cost is None:
                cost_known = 0
                new_cost = None
            elif not cost_known_before:
                cost_known = 0
                new_cost = None
            else:
                cost_known = 1
                new_cost = (previous_cost or Decimal(0)) + cost.total_cost

            # The first measured fact starts a known aggregate; an unknown fact
            # permanently makes that dimension unavailable for the run.
            if observed_before == 0 and usage is not None:
                usage_known = 1
                provider_elapsed = (
                    int(row["usage_elapsed_ms"])
                    if usage_already_accumulated
                    else usage.elapsed_ms
                )
            if observed_before == 0 and cost is not None:
                cost_known = 1
                new_cost = cost.total_cost
            await connection.execute(
                """
                UPDATE runs
                   SET metrics_usage_observed = 1,
                       metrics_usage_known = :metrics_usage_known,
                       metrics_provider_elapsed_ms = :metrics_provider_elapsed_ms,
                       metrics_cost_known = :metrics_cost_known,
                       metrics_cost = :metrics_cost
                 WHERE run_id = :run_id
                """,
                {
                    "run_id": run_id,
                    "metrics_usage_known": int(bool(usage_known)),
                    "metrics_provider_elapsed_ms": provider_elapsed,
                    "metrics_cost_known": int(bool(cost_known)),
                    "metrics_cost": None if new_cost is None else str(new_cost),
                },
            )

    @staticmethod
    def _row_to_run(row: aiosqlite.Row) -> Run:
        usage = _usage_or_none(row)
        strategy = _optional_str(row["strategy"])
        return Run(
            run_id=str(row["run_id"]),
            request=NativeRunRequest.model_validate_json(str(row["request_json"])),
            status=RunStatus(str(row["status"])),
            strategy=None if strategy is None else ExecutionStrategy(strategy),
            graph_version=int(row["graph_version"]),
            final_work_unit_id=_optional_str(row["final_work_unit_id"]),
            usage=Usage() if usage is None else usage,
            metrics=RunMetrics(
                usage=None
                if int(row["metrics_usage_known"]) == 0
                else (Usage() if usage is None else usage),
                provider_elapsed_ms=_optional_int(row["metrics_provider_elapsed_ms"]),
                wall_clock_ms=_optional_int(row["metrics_wall_clock_ms"]),
                cost=(
                    _optional_decimal(row["metrics_cost"])
                    if int(row["metrics_cost_known"]) == 1
                    else None
                ),
            ),
            error=_error_info(row["error_category"], row["error_message"]),
            created_at=_require_datetime(row["created_at"]),
            started_at=_parse_datetime(row["started_at"]),
            completed_at=_parse_datetime(row["completed_at"]),
        )

    # --- work units -------------------------------------------------------------

    async def create_graph(self, work_units: Sequence[WorkUnit]) -> None:
        """Atomically insert one complete same-run, same-version dependency graph."""
        if not work_units:
            raise StateError(
                "a work graph must contain at least one work unit",
                code=ErrorCode.ILLEGAL_STATE_TRANSITION,
            )
        run_ids = {unit.run_id for unit in work_units}
        versions = {unit.graph_version for unit in work_units}
        if len(run_ids) != 1 or len(versions) != 1:
            raise StateError(
                "a work graph must use one run_id and graph_version",
                code=ErrorCode.ILLEGAL_STATE_TRANSITION,
            )
        work_unit_ids = {unit.work_unit_id for unit in work_units}
        if len(work_unit_ids) != len(work_units):
            raise DuplicateEntityError(
                "a work graph contains duplicate work unit ids",
                code=ErrorCode.ILLEGAL_STATE_TRANSITION,
            )
        lineage_keys = [unit.lineage_key for unit in work_units if unit.lineage_key is not None]
        if len(set(lineage_keys)) != len(lineage_keys):
            raise DuplicateEntityError(
                "a work graph contains duplicate lineage keys",
                code=ErrorCode.ILLEGAL_STATE_TRANSITION,
            )
        for unit in work_units:
            if not set(unit.depends_on) <= work_unit_ids:
                raise DanglingReferenceError(
                    "a work graph dependency points outside the graph",
                    code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                )
        await self.create_work_units(work_units)

    async def create_work_units(self, work_units: Sequence[WorkUnit]) -> None:
        """Insert work units and their edges as one graph commit."""
        async with self.transaction():
            for unit in work_units:
                await self._insert_work_unit_row(unit)
            for unit in work_units:
                await self._insert_work_unit_edges(unit)

    async def create_work_unit(self, work_unit: WorkUnit) -> None:
        """Insert one work unit whose dependencies are already persisted."""
        await self.create_work_units([work_unit])

    async def update_work_unit(self, work_unit: WorkUnit) -> None:
        """Persist a work unit's status. Instructions and edges are immutable."""
        changed = await self._write(
            "UPDATE work_units SET status = :status WHERE work_unit_id = :work_unit_id",
            {"work_unit_id": work_unit.work_unit_id, "status": work_unit.status.value},
            "work unit",
        )
        if changed == 0:
            raise MissingEntityError(
                f"work unit {work_unit.work_unit_id} is not persisted",
                code=ErrorCode.WORK_UNIT_NOT_FOUND,
            )

    async def get_work_unit(self, work_unit_id: str) -> WorkUnit:
        """Read one work unit with its edges and claims."""
        row = await self._fetch_one(
            "SELECT * FROM work_units WHERE work_unit_id = ?", (work_unit_id,)
        )
        if row is None:
            raise MissingEntityError(
                f"work unit {work_unit_id} is not persisted",
                code=ErrorCode.WORK_UNIT_NOT_FOUND,
            )
        dependencies = await self._fetch_all(
            "SELECT depends_on_id FROM work_unit_dependencies WHERE work_unit_id = ? "
            "ORDER BY depends_on_id",
            (work_unit_id,),
        )
        claims = await self._fetch_all(
            "SELECT resource, access FROM work_unit_resource_claims WHERE work_unit_id = ? "
            "ORDER BY resource, access",
            (work_unit_id,),
        )
        return self._row_to_work_unit(
            row,
            tuple(str(edge["depends_on_id"]) for edge in dependencies),
            tuple(
                ResourceClaim(
                    resource=str(claim["resource"]), access=ResourceAccess(str(claim["access"]))
                )
                for claim in claims
            ),
        )

    async def list_work_units(
        self, run_id: str, *, graph_version: int | None = None
    ) -> tuple[WorkUnit, ...]:
        """Read a run's work units, optionally restricted to one graph version."""
        if graph_version is None:
            rows = await self._fetch_all(
                "SELECT * FROM work_units WHERE run_id = ? ORDER BY graph_version, created_at, "
                "work_unit_id",
                (run_id,),
            )
        else:
            rows = await self._fetch_all(
                "SELECT * FROM work_units WHERE run_id = ? AND graph_version = ? "
                "ORDER BY created_at, work_unit_id",
                (run_id, graph_version),
            )
        edge_rows = await self._fetch_all(
            """
            SELECT d.work_unit_id AS work_unit_id, d.depends_on_id AS depends_on_id
              FROM work_unit_dependencies AS d
              JOIN work_units AS w ON w.work_unit_id = d.work_unit_id
             WHERE w.run_id = ?
             ORDER BY d.depends_on_id
            """,
            (run_id,),
        )
        claim_rows = await self._fetch_all(
            """
            SELECT c.work_unit_id AS work_unit_id, c.resource AS resource, c.access AS access
              FROM work_unit_resource_claims AS c
              JOIN work_units AS w ON w.work_unit_id = c.work_unit_id
             WHERE w.run_id = ?
             ORDER BY c.resource, c.access
            """,
            (run_id,),
        )
        edges: dict[str, list[str]] = {}
        for edge in edge_rows:
            edges.setdefault(str(edge["work_unit_id"]), []).append(str(edge["depends_on_id"]))
        claims: dict[str, list[ResourceClaim]] = {}
        for claim in claim_rows:
            claims.setdefault(str(claim["work_unit_id"]), []).append(
                ResourceClaim(
                    resource=str(claim["resource"]), access=ResourceAccess(str(claim["access"]))
                )
            )
        return tuple(
            self._row_to_work_unit(
                row,
                tuple(edges.get(str(row["work_unit_id"]), ())),
                tuple(claims.get(str(row["work_unit_id"]), ())),
            )
            for row in rows
        )

    async def _insert_work_unit_row(self, work_unit: WorkUnit) -> None:
        await self._write(
            """
            INSERT INTO work_units (
                work_unit_id, run_id, graph_version, lineage_key,
                dependency_fingerprint, content_fingerprint, name, instruction,
                acceptance_criteria, output_json, status, created_at
            )
            VALUES (
                :work_unit_id, :run_id, :graph_version, :lineage_key,
                :dependency_fingerprint, :content_fingerprint, :name, :instruction,
                :acceptance_criteria, :output_json, :status, :created_at
            )
            """,
            {
                "work_unit_id": work_unit.work_unit_id,
                "run_id": work_unit.run_id,
                "graph_version": work_unit.graph_version,
                "lineage_key": work_unit.lineage_key,
                "dependency_fingerprint": work_unit.dependency_fingerprint,
                "content_fingerprint": work_unit.content_fingerprint,
                "name": work_unit.name,
                "instruction": work_unit.instruction,
                "acceptance_criteria": work_unit.acceptance_criteria,
                "output_json": work_unit.output.model_dump_json(),
                "status": work_unit.status.value,
                "created_at": _require_iso(work_unit.created_at),
            },
            "work unit",
        )

    async def _insert_work_unit_edges(self, work_unit: WorkUnit) -> None:
        for dependency in work_unit.depends_on:
            await self._write(
                "INSERT INTO work_unit_dependencies (work_unit_id, depends_on_id) "
                "VALUES (:work_unit_id, :depends_on_id)",
                {"work_unit_id": work_unit.work_unit_id, "depends_on_id": dependency},
                "work unit dependency",
            )
        for claim in work_unit.resource_claims:
            await self._write(
                "INSERT INTO work_unit_resource_claims (work_unit_id, resource, access) "
                "VALUES (:work_unit_id, :resource, :access)",
                {
                    "work_unit_id": work_unit.work_unit_id,
                    "resource": claim.resource,
                    "access": claim.access.value,
                },
                "work unit resource claim",
            )

    @staticmethod
    def _row_to_work_unit(
        row: aiosqlite.Row,
        depends_on: tuple[str, ...],
        resource_claims: tuple[ResourceClaim, ...],
    ) -> WorkUnit:
        return WorkUnit(
            work_unit_id=str(row["work_unit_id"]),
            run_id=str(row["run_id"]),
            graph_version=int(row["graph_version"]),
            lineage_key=_optional_str(row["lineage_key"]),
            dependency_fingerprint=_optional_str(row["dependency_fingerprint"]),
            content_fingerprint=_optional_str(row["content_fingerprint"]),
            name=str(row["name"]),
            instruction=str(row["instruction"]),
            acceptance_criteria=_optional_str(row["acceptance_criteria"]),
            output=OutputRequirement.model_validate_json(str(row["output_json"])),
            status=WorkUnitStatus(str(row["status"])),
            depends_on=depends_on,
            resource_claims=resource_claims,
            created_at=_require_datetime(row["created_at"]),
        )

    # --- attempts ---------------------------------------------------------------

    async def create_attempt(self, attempt: Attempt) -> None:
        """Insert a new attempt. Duplicate index within a work unit is rejected."""
        await self._write(
            """
            INSERT INTO attempts (attempt_id, run_id, work_unit_id, attempt_index, role,
                                  provider, model, status, provider_request_id,
                                  usage_input_tokens, usage_output_tokens,
                                  usage_strong_model_tokens, usage_elapsed_ms,
                                  cost_input_price_per_million_tokens,
                                  cost_output_price_per_million_tokens,
                                  cost_input, cost_output, cost_total,
                                  error_category, error_message,
                                  created_at, started_at, completed_at)
            VALUES (:attempt_id, :run_id, :work_unit_id, :attempt_index, :role,
                    :provider, :model, :status, :provider_request_id,
                    :usage_input_tokens, :usage_output_tokens,
                    :usage_strong_model_tokens, :usage_elapsed_ms,
                    :cost_input_price_per_million_tokens,
                    :cost_output_price_per_million_tokens,
                    :cost_input, :cost_output, :cost_total,
                    :error_category, :error_message,
                    :created_at, :started_at, :completed_at)
            """,
            self._attempt_params(attempt),
            "attempt",
        )

    async def update_attempt(self, attempt: Attempt) -> None:
        """Persist an attempt's outcome, usage and timestamps."""
        changed = await self._write(
            """
            UPDATE attempts
               SET status = :status,
                   provider_request_id = :provider_request_id,
                   usage_input_tokens = :usage_input_tokens,
                   usage_output_tokens = :usage_output_tokens,
                   usage_strong_model_tokens = :usage_strong_model_tokens,
                   usage_elapsed_ms = :usage_elapsed_ms,
                   cost_input_price_per_million_tokens = :cost_input_price_per_million_tokens,
                   cost_output_price_per_million_tokens = :cost_output_price_per_million_tokens,
                   cost_input = :cost_input,
                   cost_output = :cost_output,
                   cost_total = :cost_total,
                   error_category = :error_category,
                   error_message = :error_message,
                   started_at = :started_at,
                   completed_at = :completed_at
             WHERE attempt_id = :attempt_id
            """,
            self._attempt_params(attempt),
            "attempt",
        )
        if changed == 0:
            raise MissingEntityError(f"attempt {attempt.attempt_id} is not persisted")

    async def get_attempt(self, attempt_id: str) -> Attempt:
        """Read one attempt."""
        row = await self._fetch_one("SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,))
        if row is None:
            raise MissingEntityError(f"attempt {attempt_id} is not persisted")
        return self._row_to_attempt(row)

    async def list_attempts(self, work_unit_id: str) -> tuple[Attempt, ...]:
        """Read a work unit's attempts in escalation order."""
        rows = await self._fetch_all(
            "SELECT * FROM attempts WHERE work_unit_id = ? ORDER BY attempt_index",
            (work_unit_id,),
        )
        return tuple(self._row_to_attempt(row) for row in rows)

    async def list_attempts_with_status(
        self, statuses: Iterable[AttemptStatus]
    ) -> tuple[Attempt, ...]:
        """Read every attempt in one of the given statuses, across all runs."""
        values = [status.value for status in statuses]
        if not values:
            return ()
        placeholders = ", ".join("?" for _ in values)
        rows = await self._fetch_all(
            f"SELECT * FROM attempts WHERE status IN ({placeholders}) "
            "ORDER BY run_id, created_at, attempt_id",
            values,
        )
        return tuple(self._row_to_attempt(row) for row in rows)

    async def list_run_attempts(
        self, run_id: str, *, statuses: Iterable[AttemptStatus] | None = None
    ) -> tuple[Attempt, ...]:
        """Read a run's attempts, optionally filtered by status."""
        if statuses is None:
            rows = await self._fetch_all(
                "SELECT * FROM attempts WHERE run_id = ? ORDER BY created_at, attempt_id",
                (run_id,),
            )
        else:
            values = [status.value for status in statuses]
            if not values:
                return ()
            placeholders = ", ".join("?" for _ in values)
            rows = await self._fetch_all(
                f"SELECT * FROM attempts WHERE run_id = ? AND status IN ({placeholders}) "
                "ORDER BY created_at, attempt_id",
                [run_id, *values],
            )
        return tuple(self._row_to_attempt(row) for row in rows)

    # --- agent history ----------------------------------------------------------

    async def append_agent_history(self, record: AgentHistoryRecord) -> AgentHistoryRecord:
        """Append one public history item and its metadata event atomically.

        History is keyed by both sequence and idempotency key. Replaying the
        same item returns the persisted row without another event; any reuse of
        either key with different content is a state conflict.
        """
        item_json = json.dumps(
            record.item.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        async with self.transaction() as connection:
            attempt_row = await self._fetch_one(
                "SELECT run_id, work_unit_id FROM attempts WHERE attempt_id = ?",
                (record.attempt_id,),
            )
            if attempt_row is None:
                raise MissingEntityError(f"attempt {record.attempt_id} is not persisted")
            if (
                str(attempt_row["run_id"]) != record.run_id
                or str(attempt_row["work_unit_id"]) != record.work_unit_id
            ):
                raise StateError(
                    f"agent history scope does not match attempt {record.attempt_id}",
                    code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                )

            sequence_row = await self._fetch_one(
                "SELECT * FROM agent_history WHERE attempt_id = ? AND sequence = ?",
                (record.attempt_id, record.sequence),
            )
            key_row = await self._fetch_one(
                "SELECT * FROM agent_history WHERE attempt_id = ? AND idempotency_key = ?",
                (record.attempt_id, record.idempotency_key),
            )
            existing_rows = [row for row in (sequence_row, key_row) if row is not None]
            if existing_rows:
                first = existing_rows[0]
                same_row = all(
                    int(row["sequence"]) == int(first["sequence"])
                    for row in existing_rows
                )
                same_content = (
                    same_row
                    and int(first["sequence"]) == record.sequence
                    and str(first["idempotency_key"]) == record.idempotency_key
                    and str(first["item_json"]) == item_json
                )
                if same_content:
                    return self._row_to_agent_history(first)
                raise StateError(
                    "agent history sequence or idempotency key conflicts with persisted content",
                    code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                )

            try:
                await connection.execute(
                    """
                    INSERT INTO agent_history (
                        run_id, work_unit_id, attempt_id, sequence,
                        idempotency_key, item_kind, item_json, created_at
                    )
                    VALUES (
                        :run_id, :work_unit_id, :attempt_id, :sequence,
                        :idempotency_key, :item_kind, :item_json, :created_at
                    )
                    """,
                    {
                        "run_id": record.run_id,
                        "work_unit_id": record.work_unit_id,
                        "attempt_id": record.attempt_id,
                        "sequence": record.sequence,
                        "idempotency_key": record.idempotency_key,
                        "item_kind": record.item.kind,
                        "item_json": item_json,
                        "created_at": _require_iso(record.created_at),
                    },
                )
            except sqlite3.IntegrityError as error:
                raise _translate_integrity_error(error, "agent history") from error
            await self.append_event(
                record.run_id,
                EventType.AGENT_HISTORY_RECORDED,
                payload_from_agent_history(record),
                timestamp=record.created_at,
            )
        return record

    async def list_agent_history(
        self,
        attempt_id: str,
        *,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> tuple[AgentHistoryRecord, ...]:
        """Read one attempt's public history in append order."""
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if after_sequence is not None and after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        cursor_value = 0 if after_sequence is None else after_sequence
        sql = (
            "SELECT * FROM agent_history WHERE attempt_id = ? AND sequence > ? "
            "ORDER BY sequence"
        )
        params: list[Any] = [attempt_id, cursor_value]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = await self._fetch_all(sql, params)
        return tuple(self._row_to_agent_history(row) for row in rows)

    @staticmethod
    def _row_to_agent_history(row: aiosqlite.Row) -> AgentHistoryRecord:
        return AgentHistoryRecord(
            run_id=str(row["run_id"]),
            work_unit_id=str(row["work_unit_id"]),
            attempt_id=str(row["attempt_id"]),
            sequence=int(row["sequence"]),
            idempotency_key=str(row["idempotency_key"]),
            item=strict_json_loads(str(row["item_json"])),
            created_at=_require_datetime(row["created_at"]),
        )

    @staticmethod
    def _attempt_params(attempt: Attempt) -> dict[str, Any]:
        usage = attempt.usage
        return {
            "attempt_id": attempt.attempt_id,
            "run_id": attempt.run_id,
            "work_unit_id": attempt.work_unit_id,
            "attempt_index": attempt.attempt_index,
            "role": attempt.role.value,
            "provider": attempt.model.provider,
            "model": attempt.model.model,
            "status": attempt.status.value,
            "provider_request_id": attempt.provider_request_id,
            "usage_input_tokens": None if usage is None else usage.input_tokens,
            "usage_output_tokens": None if usage is None else usage.output_tokens,
            "usage_strong_model_tokens": None if usage is None else usage.strong_model_tokens,
            "usage_elapsed_ms": None if usage is None else usage.elapsed_ms,
            "cost_input_price_per_million_tokens": (
                None if attempt.cost is None else str(attempt.cost.input_price_per_million_tokens)
            ),
            "cost_output_price_per_million_tokens": (
                None if attempt.cost is None else str(attempt.cost.output_price_per_million_tokens)
            ),
            "cost_input": None if attempt.cost is None else str(attempt.cost.input_cost),
            "cost_output": None if attempt.cost is None else str(attempt.cost.output_cost),
            "cost_total": None if attempt.cost is None else str(attempt.cost.total_cost),
            "error_category": None if attempt.error is None else attempt.error.category.value,
            "error_message": None if attempt.error is None else attempt.error.message,
            "created_at": _require_iso(attempt.created_at),
            "started_at": _iso(attempt.started_at),
            "completed_at": _iso(attempt.completed_at),
        }

    @staticmethod
    def _row_to_attempt(row: aiosqlite.Row) -> Attempt:
        return Attempt(
            attempt_id=str(row["attempt_id"]),
            run_id=str(row["run_id"]),
            work_unit_id=str(row["work_unit_id"]),
            attempt_index=int(row["attempt_index"]),
            role=ModelRole(str(row["role"])),
            model=ModelRef(provider=str(row["provider"]), model=str(row["model"])),
            status=AttemptStatus(str(row["status"])),
            provider_request_id=_optional_str(row["provider_request_id"]),
            usage=_usage_or_none(row),
            cost=(
                None
                if row["cost_total"] is None
                else AttemptCost(
                    input_tokens=int(row["usage_input_tokens"]),
                    output_tokens=int(row["usage_output_tokens"]),
                    input_price_per_million_tokens=Decimal(str(row["cost_input_price_per_million_tokens"])),
                    output_price_per_million_tokens=Decimal(str(row["cost_output_price_per_million_tokens"])),
                    input_cost=Decimal(str(row["cost_input"])),
                    output_cost=Decimal(str(row["cost_output"])),
                    total_cost=Decimal(str(row["cost_total"])),
                )
            ),
            error=_error_info(row["error_category"], row["error_message"]),
            created_at=_require_datetime(row["created_at"]),
            started_at=_parse_datetime(row["started_at"]),
            completed_at=_parse_datetime(row["completed_at"]),
        )

    # --- artifacts and evidence -------------------------------------------------

    async def add_artifact(self, artifact: Artifact) -> None:
        """Persist a produced artifact."""
        await self._write(
            """
            INSERT INTO artifacts (artifact_id, run_id, work_unit_id, attempt_id, name,
                                   kind, content, created_at)
            VALUES (:artifact_id, :run_id, :work_unit_id, :attempt_id, :name,
                    :kind, :content, :created_at)
            """,
            {
                "artifact_id": artifact.artifact_id,
                "run_id": artifact.run_id,
                "work_unit_id": artifact.work_unit_id,
                "attempt_id": artifact.attempt_id,
                "name": artifact.name,
                "kind": artifact.kind.value,
                "content": artifact.content,
                "created_at": _require_iso(artifact.created_at),
            },
            "artifact",
        )

    async def get_artifact(self, artifact_id: str) -> Artifact:
        """Read one artifact."""
        row = await self._fetch_one(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        )
        if row is None:
            raise MissingEntityError(f"artifact {artifact_id} is not persisted")
        return self._row_to_artifact(row)

    async def list_artifacts(self, work_unit_id: str) -> tuple[Artifact, ...]:
        """Read a work unit's artifacts in production order."""
        rows = await self._fetch_all(
            "SELECT * FROM artifacts WHERE work_unit_id = ? ORDER BY created_at, artifact_id",
            (work_unit_id,),
        )
        return tuple(self._row_to_artifact(row) for row in rows)

    async def add_evidence(self, evidence: Evidence) -> None:
        """Persist a verdict about an artifact.

        Only ``result`` is written. ``Evidence.passed`` is derived, so there is no
        boolean column that could drift out of step with the verdict.
        """
        await self._write(
            """
            INSERT INTO evidence (evidence_id, run_id, work_unit_id, artifact_id, kind,
                                  rule, result, detail, created_at)
            VALUES (:evidence_id, :run_id, :work_unit_id, :artifact_id, :kind,
                    :rule, :result, :detail, :created_at)
            """,
            {
                "evidence_id": evidence.evidence_id,
                "run_id": evidence.run_id,
                "work_unit_id": evidence.work_unit_id,
                "artifact_id": evidence.artifact_id,
                "kind": evidence.kind.value,
                "rule": evidence.rule,
                "result": evidence.result.value,
                "detail": evidence.detail,
                "created_at": _require_iso(evidence.created_at),
            },
            "evidence",
        )

    async def list_evidence(self, work_unit_id: str) -> tuple[Evidence, ...]:
        """Read a work unit's evidence in record order."""
        rows = await self._fetch_all(
            "SELECT * FROM evidence WHERE work_unit_id = ? ORDER BY created_at, evidence_id",
            (work_unit_id,),
        )
        return tuple(self._row_to_evidence(row) for row in rows)

    # --- reservations -----------------------------------------------------------

    async def reserve_reservation(
        self,
        request: ReservationRequest,
        *,
        reservation_id: str | None = None,
        created_at: datetime | None = None,
        held_at: datetime | None = None,
        capacity_limit: int | None = None,
    ) -> Reservation:
        """Atomically hold one dispatch reservation or return its idempotent fact."""
        from prp_runtime.control.reservations import Reservation

        created = utc_now() if created_at is None else created_at
        held = created if held_at is None else held_at
        if capacity_limit is None and request.capacity_key is not None:
            capacity_limit = self._capacity_limits.get(request.capacity_key)
        if capacity_limit is not None and capacity_limit < 1:
            raise ValueError("capacity_limit must be at least 1")
        reservation = Reservation(
            reservation_id=new_reservation_id() if reservation_id is None else reservation_id,
            request=request,
            status=ReservationStatus.HELD,
            created_at=created,
            held_at=held,
        )
        async with self.transaction() as connection:
            try:
                existing_row = await self._fetch_one(
                    """
                    SELECT * FROM reservations
                     WHERE run_id = :run_id AND dispatch_key = :dispatch_key
                    """,
                    {
                        "run_id": request.run_id,
                        "dispatch_key": request.dispatch_key,
                    },
                )
                if existing_row is not None:
                    existing = self._row_to_reservation(existing_row)
                    if existing.request != request:
                        raise StateError(
                            "dispatch key already belongs to a different reservation request",
                            code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                        )
                    return existing
                if request.capacity_key is not None and capacity_limit is not None:
                    async with connection.execute(
                        """
                        SELECT COUNT(*) AS count FROM reservations
                         WHERE capacity_key = ? AND status = 'HELD'
                        """,
                        (request.capacity_key,),
                    ) as cursor:
                        row = await cursor.fetchone()
                    if row is not None and int(row["count"]) >= capacity_limit:
                        raise StateError(
                            f"capacity {request.capacity_key!r} has no available slot",
                            code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                        )
                await connection.execute(
                    """
                    INSERT INTO reservations (
                        reservation_id, run_id, work_unit_id, dispatch_key,
                        attempt_units, estimated_input_tokens,
                        estimated_output_tokens, token_upper_bound,
                        strong_token_upper_bound, capacity_key, status,
                        created_at, held_at
                    ) VALUES (
                        :reservation_id, :run_id, :work_unit_id, :dispatch_key,
                        :attempt_units, :estimated_input_tokens,
                        :estimated_output_tokens, :token_upper_bound,
                        :strong_token_upper_bound, :capacity_key, :status,
                        :created_at, :held_at
                    )
                    """,
                    self._reservation_insert_params(reservation),
                )
            except sqlite3.IntegrityError as error:
                if "UNIQUE" not in str(error) and "PRIMARY KEY" not in str(error):
                    raise _translate_integrity_error(error, "reservation") from error
                existing_row = await self._fetch_one(
                    """
                    SELECT * FROM reservations
                     WHERE run_id = :run_id AND dispatch_key = :dispatch_key
                    """,
                    {
                        "run_id": request.run_id,
                        "dispatch_key": request.dispatch_key,
                    },
                )
                if existing_row is None:
                    raise DuplicateEntityError(
                        "reservation is already persisted",
                        code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                    ) from error
                existing = self._row_to_reservation(existing_row)
                if existing.request != request:
                    raise StateError(
                        "dispatch key already belongs to a different reservation request",
                        code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                    ) from error
                return existing
            await self.append_event(
                request.run_id,
                EventType.RESERVATION_CREATED,
                {
                    "reservation_id": reservation.reservation_id,
                    "dispatch_key": request.dispatch_key,
                },
                timestamp=created,
            )
            await self.append_event(
                request.run_id,
                EventType.RESERVATION_HELD,
                {"reservation_id": reservation.reservation_id},
                timestamp=held,
            )
        return reservation

    async def reserve(self, request: ReservationRequest, **kwargs: Any) -> Reservation:
        """Compatibility alias for the explicit reservation operation."""
        return await self.reserve_reservation(request, **kwargs)

    async def get_reservation(self, reservation_id: str) -> Reservation:
        """Read one reservation fact."""
        row = await self._fetch_one(
            "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
        )
        if row is None:
            raise MissingEntityError(
                f"reservation {reservation_id} is not persisted",
                code=ErrorCode.ILLEGAL_STATE_TRANSITION,
            )
        return self._row_to_reservation(row)

    async def list_reservations(
        self,
        run_id: str | None = None,
        *,
        statuses: Iterable[ReservationStatus] | None = None,
    ) -> tuple[Reservation, ...]:
        """Read one run's reservations in creation order."""
        values = None if statuses is None else [status.value for status in statuses]
        sql = "SELECT * FROM reservations"
        params: list[Any] = []
        if run_id is not None:
            sql += " WHERE run_id = ?"
            params.append(run_id)
        if values is not None:
            if not values:
                return ()
            placeholders = ", ".join("?" for _ in values)
            sql += " AND" if run_id is not None else " WHERE"
            sql += f" status IN ({placeholders})"
            params.extend(values)
        sql += " ORDER BY created_at, reservation_id"
        rows = await self._fetch_all(sql, params)
        return tuple(self._row_to_reservation(row) for row in rows)

    async def count_held_reservations(self, capacity_key: str) -> int:
        """Count persisted active dispatches for one capacity key."""
        value = await self._fetch_one(
            """
            SELECT COUNT(*) AS count FROM reservations
             WHERE capacity_key = ? AND status = 'HELD'
            """,
            (capacity_key,),
        )
        if value is None:
            raise InternalError("sqlite did not return a reservation count")
        return int(value["count"])

    async def settle_reservation(
        self,
        reservation_id: str,
        *,
        measured_usage: Usage | None,
        completed_at: datetime | None = None,
    ) -> Reservation:
        """Atomically settle a HELD reservation, with same-fact idempotency."""
        completed = utc_now() if completed_at is None else completed_at
        async with self.transaction() as connection:
            current = await self._get_reservation_in_transaction(connection, reservation_id)
            if current.status is ReservationStatus.SETTLED:
                if (
                    current.completed_at == completed
                    and current.measured_usage == measured_usage
                ):
                    return current
                raise StateError(
                    "settled reservation conflicts with the existing terminal fact",
                    code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                )
            if current.status is not ReservationStatus.HELD:
                raise StateError(
                    f"cannot settle reservation in {current.status.value} state",
                    code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                )
            params = self._reservation_usage_params(
                reservation_id,
                completed_at=completed,
                measured_usage=measured_usage,
            )
            cursor = await connection.execute(
                """
                UPDATE reservations
                   SET status = 'SETTLED', completed_at = :completed_at,
                       measured_input_tokens = :measured_input_tokens,
                       measured_output_tokens = :measured_output_tokens,
                       measured_strong_model_tokens = :measured_strong_model_tokens,
                       measured_elapsed_ms = :measured_elapsed_ms
                 WHERE reservation_id = :reservation_id AND status = 'HELD'
                """,
                params,
            )
            if cursor.rowcount != 1:
                raise StateError(
                    "reservation settlement lost its HELD state",
                    code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                )
            await self.append_event(
                current.request.run_id,
                EventType.RESERVATION_SETTLED,
                {"reservation_id": reservation_id},
                timestamp=completed,
            )
            return await self._get_reservation_in_transaction(connection, reservation_id)

    async def release_reservation(
        self,
        reservation_id: str,
        *,
        expired: bool = False,
        completed_at: datetime | None = None,
    ) -> Reservation:
        """Atomically release or expire a HELD reservation."""
        completed = utc_now() if completed_at is None else completed_at
        target = ReservationStatus.EXPIRED if expired else ReservationStatus.RELEASED
        async with self.transaction() as connection:
            current = await self._get_reservation_in_transaction(connection, reservation_id)
            if current.status is target and current.completed_at == completed:
                return current
            if current.status is not ReservationStatus.HELD:
                raise StateError(
                    f"cannot release reservation in {current.status.value} state",
                    code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                )
            cursor = await connection.execute(
                """
                UPDATE reservations
                   SET status = :status, completed_at = :completed_at
                 WHERE reservation_id = :reservation_id AND status = 'HELD'
                """,
                {
                    "status": target.value,
                    "completed_at": _require_iso(completed),
                    "reservation_id": reservation_id,
                },
            )
            if cursor.rowcount != 1:
                raise StateError(
                    "reservation release lost its HELD state",
                    code=ErrorCode.ILLEGAL_STATE_TRANSITION,
                )
            await self.append_event(
                current.request.run_id,
                EventType.RESERVATION_EXPIRED if expired else EventType.RESERVATION_RELEASED,
                {"reservation_id": reservation_id},
                timestamp=completed,
            )
            return await self._get_reservation_in_transaction(connection, reservation_id)

    @staticmethod
    def _reservation_insert_params(reservation: Reservation) -> dict[str, Any]:
        request = reservation.request
        return {
            "reservation_id": reservation.reservation_id,
            "run_id": request.run_id,
            "work_unit_id": request.work_unit_id,
            "dispatch_key": request.dispatch_key,
            "attempt_units": request.attempt_units,
            "estimated_input_tokens": request.estimated_input_tokens,
            "estimated_output_tokens": request.estimated_output_tokens,
            "token_upper_bound": request.token_upper_bound,
            "strong_token_upper_bound": request.strong_token_upper_bound,
            "capacity_key": request.capacity_key,
            "status": reservation.status.value,
            "created_at": _require_iso(reservation.created_at),
            "held_at": _iso(reservation.held_at),
        }

    @staticmethod
    def _reservation_usage_params(
        reservation_id: str,
        *,
        completed_at: datetime,
        measured_usage: Usage | None,
    ) -> dict[str, Any]:
        return {
            "reservation_id": reservation_id,
            "completed_at": _require_iso(completed_at),
            "measured_input_tokens": (
                None if measured_usage is None else measured_usage.input_tokens
            ),
            "measured_output_tokens": (
                None if measured_usage is None else measured_usage.output_tokens
            ),
            "measured_strong_model_tokens": (
                None
                if measured_usage is None
                else measured_usage.strong_model_tokens
            ),
            "measured_elapsed_ms": (
                None if measured_usage is None else measured_usage.elapsed_ms
            ),
        }

    async def _get_reservation_in_transaction(
        self, connection: aiosqlite.Connection, reservation_id: str
    ) -> Reservation:
        async with connection.execute(
            "SELECT * FROM reservations WHERE reservation_id = ?", (reservation_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise MissingEntityError(
                f"reservation {reservation_id} is not persisted",
                code=ErrorCode.ILLEGAL_STATE_TRANSITION,
            )
        return self._row_to_reservation(row)

    @staticmethod
    def _row_to_reservation(row: aiosqlite.Row) -> Reservation:
        from prp_runtime.control.reservations import Reservation, ReservationRequest

        measured = None
        if row["measured_input_tokens"] is not None:
            measured = Usage(
                input_tokens=int(row["measured_input_tokens"]),
                output_tokens=int(row["measured_output_tokens"] or 0),
                strong_model_tokens=int(row["measured_strong_model_tokens"] or 0),
                elapsed_ms=int(row["measured_elapsed_ms"] or 0),
            )
        return Reservation(
            reservation_id=str(row["reservation_id"]),
            request=ReservationRequest(
                run_id=str(row["run_id"]),
                work_unit_id=str(row["work_unit_id"]),
                dispatch_key=str(row["dispatch_key"]),
                attempt_units=int(row["attempt_units"]),
                estimated_input_tokens=_optional_int(row["estimated_input_tokens"]),
                estimated_output_tokens=_optional_int(row["estimated_output_tokens"]),
                token_upper_bound=_optional_int(row["token_upper_bound"]),
                strong_token_upper_bound=_optional_int(
                    row["strong_token_upper_bound"]
                ),
                capacity_key=_optional_str(row["capacity_key"]),
            ),
            status=ReservationStatus(str(row["status"])),
            created_at=_require_datetime(row["created_at"]),
            held_at=_parse_datetime(row["held_at"]),
            completed_at=_parse_datetime(row["completed_at"]),
            measured_usage=measured,
        )

    @staticmethod
    def _row_to_evidence(row: aiosqlite.Row) -> Evidence:
        return Evidence(
            evidence_id=str(row["evidence_id"]),
            run_id=str(row["run_id"]),
            work_unit_id=str(row["work_unit_id"]),
            artifact_id=str(row["artifact_id"]),
            kind=EvidenceKind(str(row["kind"])),
            rule=_optional_str(row["rule"]),
            result=VerificationResult(str(row["result"])),
            detail=str(row["detail"]),
            created_at=_require_datetime(row["created_at"]),
        )

    @staticmethod
    def _row_to_artifact(row: aiosqlite.Row) -> Artifact:
        return Artifact(
            artifact_id=str(row["artifact_id"]),
            run_id=str(row["run_id"]),
            work_unit_id=str(row["work_unit_id"]),
            attempt_id=str(row["attempt_id"]),
            name=str(row["name"]),
            kind=ArtifactKind(str(row["kind"])),
            content=str(row["content"]),
            created_at=_require_datetime(row["created_at"]),
        )

    # --- events -----------------------------------------------------------------

    async def last_sequence(self, run_id: str) -> int | None:
        """The highest sequence recorded for a run, or ``None`` when empty."""
        row = await self._fetch_one(
            "SELECT MAX(sequence) AS last FROM events WHERE run_id = ?", (run_id,)
        )
        if row is None:
            return None
        return _optional_int(row["last"])

    async def append_event(
        self,
        run_id: str,
        event_type: EventType,
        payload: Mapping[str, JsonValue] | None = None,
        *,
        timestamp: datetime | None = None,
    ) -> RunEvent:
        """Append one event with the next sequence number for its run.

        The payload is validated by ``RunEvent`` before it is written. A lost race
        for a sequence number is retried a bounded number of times.
        """
        body = dict(payload or {})
        # Validate type, payload and timestamp before writing. The sequence used
        # here is only a probe: the authoritative number is assigned by the
        # database inside the insert, so concurrent appends cannot collide.
        draft_fields: dict[str, Any] = {
            "run_id": run_id,
            "sequence": next_sequence(await self.last_sequence(run_id)),
            "event_type": event_type,
            "payload": body,
        }
        if timestamp is not None:
            draft_fields["timestamp"] = timestamp
        draft = RunEvent(**draft_fields)
        params = {
            "run_id": draft.run_id,
            "event_type": draft.event_type.value,
            # ``allow_nan=False`` refuses NaN and Infinity at the write, so the
            # ledger cannot hold a token that standard JSON has no way to express
            # and the read below can never meet one.
            "payload_json": json.dumps(draft.payload, sort_keys=True, allow_nan=False),
            "timestamp": _require_iso(draft.timestamp),
        }
        last_error: sqlite3.IntegrityError | None = None
        for _ in range(_MAX_SEQUENCE_RETRIES):
            try:
                async with self.transaction() as connection:
                    cursor = await connection.execute(
                        """
                        INSERT INTO events (run_id, sequence, event_type, payload_json,
                                            timestamp)
                        SELECT :run_id, COALESCE(MAX(sequence), 0) + 1, :event_type,
                               :payload_json, :timestamp
                          FROM events WHERE run_id = :run_id
                        """,
                        params,
                    )
                    rowid = cursor.lastrowid
                    if rowid is None:
                        raise InternalError("sqlite did not report the inserted event row")
                    row = await self._fetch_one(
                        "SELECT sequence FROM events WHERE rowid = ?", (rowid,)
                    )
                    if row is None:
                        raise InternalError("the inserted event row disappeared")
                    assigned = int(row["sequence"])
                    if self._event_bus is not None:
                        self._pending_event_hints.append((run_id, assigned))
            except sqlite3.IntegrityError as error:
                message = str(error)
                if "FOREIGN KEY" in message:
                    raise DanglingReferenceError(
                        f"run {run_id} is not persisted", code=ErrorCode.RUN_NOT_FOUND
                    ) from error
                if "UNIQUE" not in message and "PRIMARY KEY" not in message:
                    raise _translate_integrity_error(error, "event") from error
                last_error = error
                continue
            event = draft.model_copy(update={"sequence": assigned})
            return event
        raise SequenceConflictError(
            f"could not allocate an event sequence for run {run_id} after "
            f"{_MAX_SEQUENCE_RETRIES} attempts",
            code=ErrorCode.EVENT_SEQUENCE_INVALID,
        ) from last_error

    async def list_events(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> tuple[RunEvent, ...]:
        """Read a run's ledger in sequence order, optionally after a cursor.

        A cursor beyond the last recorded sequence returns no events rather than
        an error, so a reconnecting subscriber can always resume.
        """
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if after_sequence is not None and after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        cursor_value = 0 if after_sequence is None else after_sequence
        sql = (
            "SELECT * FROM events WHERE run_id = ? AND sequence > ? ORDER BY sequence"
        )
        params: list[Any] = [run_id, cursor_value]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = await self._fetch_all(sql, params)
        return tuple(
            RunEvent(
                run_id=str(row["run_id"]),
                sequence=int(row["sequence"]),
                event_type=EventType(str(row["event_type"])),
                payload=strict_json_loads(str(row["payload_json"])),
                timestamp=_require_datetime(row["timestamp"]),
            )
            for row in rows
        )
