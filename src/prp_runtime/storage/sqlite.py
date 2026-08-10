"""SQLite store: schema, connection lifecycle and the current operation set.

There is exactly one schema and no migration path. A database created by a
different schema version is rejected with an instruction to delete the
development database.

Every mutating method joins the caller's transaction when one is open and
commits on its own when it is the outermost call, so a state change and its
events are never half committed. All JSON columns pass through the domain
models on the way in and on the way out.
"""

import json
import sqlite3
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Self, SupportsInt

import aiosqlite
from pydantic import JsonValue

from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    ModelRole,
    ResourceAccess,
    RunStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import ErrorCode, InternalError, StateError
from prp_runtime.domain.events import EventType, RunEvent, next_sequence
from prp_runtime.domain.models import (
    Artifact,
    ArtifactKind,
    Attempt,
    ErrorCategory,
    ErrorInfo,
    Evidence,
    EvidenceKind,
    NativeRunRequest,
    OutputRequirement,
    Run,
    Usage,
    VerificationResult,
    WorkUnit,
)
from prp_runtime.domain.values import ModelRef, ResourceClaim
from prp_runtime.json_support import strict_json_loads

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

SCHEMA_VERSION = 3

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_DEFAULT_BUSY_TIMEOUT_MS = 5_000

_MAX_SEQUENCE_RETRIES = 8


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


def _error_info(category: object, message: object) -> ErrorInfo | None:
    if category is None:
        return None
    if message is None:
        raise InternalError("stored error has a category but no message")
    return ErrorInfo(category=ErrorCategory(str(category)), message=str(message))


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
    ) -> None:
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        self._database_path = database_path
        self._busy_timeout_ms = busy_timeout_ms
        self._connection: aiosqlite.Connection | None = None
        self._transaction_depth = 0

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
        self._transaction_depth += 1
        try:
            yield connection
        except BaseException:
            self._transaction_depth -= 1
            if self._transaction_depth == 0:
                await connection.rollback()
            raise
        self._transaction_depth -= 1
        if self._transaction_depth == 0:
            await connection.commit()

    @property
    def in_transaction(self) -> bool:
        return self._transaction_depth > 0

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

    async def _fetch_one(self, sql: str, params: Sequence[Any]) -> aiosqlite.Row | None:
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
                              request_json, usage_input_tokens, usage_output_tokens,
                              usage_strong_model_tokens, usage_elapsed_ms,
                              error_category, error_message,
                              created_at, started_at, completed_at)
            VALUES (:run_id, :status, :routing_policy, :strategy, :graph_version,
                    :request_json, :usage_input_tokens, :usage_output_tokens,
                    :usage_strong_model_tokens, :usage_elapsed_ms,
                    :error_category, :error_message,
                    :created_at, :started_at, :completed_at)
            """,
            {
                "run_id": run.run_id,
                "status": run.status.value,
                "routing_policy": run.request.routing_policy.value,
                "strategy": None if run.strategy is None else run.strategy.value,
                "graph_version": run.graph_version,
                "request_json": run.request.model_dump_json(),
                "usage_input_tokens": run.usage.input_tokens,
                "usage_output_tokens": run.usage.output_tokens,
                "usage_strong_model_tokens": run.usage.strong_model_tokens,
                "usage_elapsed_ms": run.usage.elapsed_ms,
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
        """Add measured usage to a run atomically and return the new aggregate."""
        changed = await self._write(
            """
            UPDATE runs
               SET usage_input_tokens = usage_input_tokens + :input_tokens,
                   usage_output_tokens = usage_output_tokens + :output_tokens,
                   usage_strong_model_tokens =
                       usage_strong_model_tokens + :strong_model_tokens,
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
            raise MissingEntityError(f"run {run_id} is not persisted", code=ErrorCode.RUN_NOT_FOUND)
        return await self.get_run_usage(run_id)

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
            usage=Usage() if usage is None else usage,
            error=_error_info(row["error_category"], row["error_message"]),
            created_at=_require_datetime(row["created_at"]),
            started_at=_parse_datetime(row["started_at"]),
            completed_at=_parse_datetime(row["completed_at"]),
        )

    # --- work units -------------------------------------------------------------

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
            INSERT INTO work_units (work_unit_id, run_id, graph_version, name, instruction,
                                    acceptance_criteria, output_json, status, created_at)
            VALUES (:work_unit_id, :run_id, :graph_version, :name, :instruction,
                    :acceptance_criteria, :output_json, :status, :created_at)
            """,
            {
                "work_unit_id": work_unit.work_unit_id,
                "run_id": work_unit.run_id,
                "graph_version": work_unit.graph_version,
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
                                  error_category, error_message,
                                  created_at, started_at, completed_at)
            VALUES (:attempt_id, :run_id, :work_unit_id, :attempt_index, :role,
                    :provider, :model, :status, :provider_request_id,
                    :usage_input_tokens, :usage_output_tokens,
                    :usage_strong_model_tokens, :usage_elapsed_ms,
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
            return draft.model_copy(update={"sequence": assigned})
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
