"""Targeted tests for the SQLite schema and connection lifecycle."""

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from prp_runtime.domain.errors import InternalError
from prp_runtime.storage.sqlite import (
    SCHEMA_PATH,
    SCHEMA_VERSION,
    IncompatibleSchemaError,
    SqliteStore,
)

EXPECTED_TABLES = {
    "runs",
    "work_units",
    "work_unit_dependencies",
    "work_unit_resource_claims",
    "attempts",
    "artifacts",
    "evidence",
    "events",
}

EXPECTED_INDEXES = {
    "idx_runs_status",
    "idx_work_units_run",
    "idx_work_unit_dependencies_depends_on",
    "idx_work_unit_resource_claims_resource",
    "idx_attempts_run_status",
    "idx_artifacts_work_unit",
    "idx_artifacts_attempt",
    "idx_evidence_artifact",
    "idx_evidence_work_unit",
    "idx_events_type",
}


async def _names(store: SqliteStore, kind: str) -> set[str]:
    query = "SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE 'sqlite_%'"
    async with store.connection.execute(query, (kind,)) as cursor:
        return {str(row["name"]) for row in await cursor.fetchall()}


async def _scalar(store: SqliteStore, sql: str) -> object:
    async with store.connection.execute(sql) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return row[0]


async def _columns(store: SqliteStore, table: str) -> set[str]:
    async with store.connection.execute(f"PRAGMA table_info({table})") as cursor:
        return {str(row["name"]) for row in await cursor.fetchall()}


async def _insert_run(store: SqliteStore, run_id: str = "run_1") -> None:
    await store.connection.execute(
        """
        INSERT INTO runs (run_id, status, routing_policy, graph_version, request_json,
                          created_at)
        VALUES (?, 'PENDING', 'AUTO', 1, '{}', '2026-08-10T12:00:00+00:00')
        """,
        (run_id,),
    )


async def _insert_work_unit(
    store: SqliteStore, work_unit_id: str = "wu_1", run_id: str = "run_1"
) -> None:
    await store.connection.execute(
        """
        INSERT INTO work_units (work_unit_id, run_id, graph_version, name, instruction,
                                output_json, status, created_at)
        VALUES (?, ?, 1, 'unit', 'do it', '{}', 'PENDING', '2026-08-10T12:00:00+00:00')
        """,
        (work_unit_id, run_id),
    )


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "prp-test.db"


# --- lifecycle ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_creates_the_database_and_stamps_the_version(database_path: Path) -> None:
    store = SqliteStore(database_path)
    assert store.is_open is False
    assert not database_path.exists()
    async with store:
        assert store.is_open is True
        assert database_path.exists()
        assert await _scalar(store, "PRAGMA user_version") == SCHEMA_VERSION
    assert store.is_open is False


@pytest.mark.asyncio
async def test_open_is_idempotent(database_path: Path) -> None:
    store = SqliteStore(database_path)
    await store.open()
    first = store.connection
    await store.open()
    assert store.connection is first
    assert await _names(store, "table") == EXPECTED_TABLES
    await store.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_releases_the_connection(database_path: Path) -> None:
    store = SqliteStore(database_path)
    await store.open()
    await store.close()
    await store.close()
    assert store.is_open is False
    with pytest.raises(InternalError, match="not open"):
        _ = store.connection


@pytest.mark.asyncio
async def test_reopening_an_existing_database_keeps_the_data(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await store.connection.commit()
    async with SqliteStore(database_path) as store:
        assert await _scalar(store, "SELECT COUNT(*) FROM runs") == 1
        assert await _names(store, "table") == EXPECTED_TABLES


@pytest.mark.asyncio
async def test_transaction_commits_on_success_and_rolls_back_on_error(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        async with store.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO runs (run_id, status, routing_policy, graph_version,
                                  request_json, created_at)
                VALUES ('run_ok', 'PENDING', 'AUTO', 1, '{}', '2026-08-10T12:00:00+00:00')
                """
            )
        assert await _scalar(store, "SELECT COUNT(*) FROM runs") == 1

        with pytest.raises(RuntimeError):
            async with store.transaction() as connection:
                await connection.execute(
                    """
                    INSERT INTO runs (run_id, status, routing_policy, graph_version,
                                      request_json, created_at)
                    VALUES ('run_bad', 'PENDING', 'AUTO', 1, '{}',
                            '2026-08-10T12:00:00+00:00')
                    """
                )
                raise RuntimeError("boom")
        assert await _scalar(store, "SELECT COUNT(*) FROM runs") == 1


def test_store_rejects_a_non_positive_busy_timeout(database_path: Path) -> None:
    with pytest.raises(ValueError):
        SqliteStore(database_path, busy_timeout_ms=0)


# --- pragmas --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pragmas_are_applied(database_path: Path) -> None:
    async with SqliteStore(database_path, busy_timeout_ms=1234) as store:
        assert await _scalar(store, "PRAGMA foreign_keys") == 1
        assert str(await _scalar(store, "PRAGMA journal_mode")).lower() == "wal"
        assert await _scalar(store, "PRAGMA busy_timeout") == 1234


# --- schema shape ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_covers_every_persisted_entity(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        assert await _names(store, "table") == EXPECTED_TABLES
        assert EXPECTED_INDEXES <= await _names(store, "index")


@pytest.mark.asyncio
async def test_usage_columns_exist_on_runs_and_attempts(database_path: Path) -> None:
    usage_columns = {
        "usage_input_tokens",
        "usage_output_tokens",
        "usage_strong_model_tokens",
        "usage_elapsed_ms",
    }
    async with SqliteStore(database_path) as store:
        for table in ("runs", "attempts"):
            async with store.connection.execute(f"PRAGMA table_info({table})") as cursor:
                columns = {str(row["name"]) for row in await cursor.fetchall()}
            assert usage_columns <= columns


@pytest.mark.asyncio
async def test_foreign_keys_are_enforced(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_work_unit(store, run_id="run_missing")
        await _insert_run(store)
        await _insert_work_unit(store)
        assert await _scalar(store, "SELECT COUNT(*) FROM work_units") == 1


@pytest.mark.asyncio
async def test_deleting_a_run_cascades_to_its_children(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        await store.connection.execute(
            """
            INSERT INTO events (run_id, sequence, event_type, payload_json, timestamp)
            VALUES ('run_1', 1, 'RUN_CREATED', '{}', '2026-08-10T12:00:00+00:00')
            """
        )
        await store.connection.execute("DELETE FROM runs WHERE run_id = 'run_1'")
        assert await _scalar(store, "SELECT COUNT(*) FROM work_units") == 0
        assert await _scalar(store, "SELECT COUNT(*) FROM events") == 0


@pytest.mark.asyncio
async def test_event_sequence_is_unique_per_run_and_positive(database_path: Path) -> None:
    insert = (
        "INSERT INTO events (run_id, sequence, event_type, payload_json, timestamp) "
        "VALUES (?, ?, 'RUN_STARTED', '{}', '2026-08-10T12:00:00+00:00')"
    )
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_run(store, run_id="run_2")
        await store.connection.execute(insert, ("run_1", 1))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("run_1", 1))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("run_1", 0))
        await store.connection.execute(insert, ("run_2", 1))
        assert await _scalar(store, "SELECT COUNT(*) FROM events") == 2


@pytest.mark.asyncio
async def test_attempt_index_is_unique_per_work_unit(database_path: Path) -> None:
    insert = (
        "INSERT INTO attempts (attempt_id, run_id, work_unit_id, attempt_index, role, "
        "provider, model, status, created_at) VALUES (?, 'run_1', 'wu_1', ?, 'WORKER', "
        "'openai_compatible', 'weak', 'PENDING', '2026-08-10T12:00:00+00:00')"
    )
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        await store.connection.execute(insert, ("att_1", 1))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("att_2", 1))
        await store.connection.execute(insert, ("att_2", 2))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("att_3", 0))


@pytest.mark.asyncio
async def test_dependency_edge_cannot_be_self_referential(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                "INSERT INTO work_unit_dependencies (work_unit_id, depends_on_id) "
                "VALUES ('wu_1', 'wu_1')"
            )


async def _insert_artifact(store: SqliteStore) -> None:
    """Seed the attempt and artifact an evidence row has to reference."""
    await store.connection.execute(
        "INSERT INTO attempts (attempt_id, run_id, work_unit_id, attempt_index, role, "
        "provider, model, status, created_at) VALUES ('att_1', 'run_1', 'wu_1', 1, "
        "'WORKER', 'openai_compatible', 'weak', 'PENDING', '2026-08-10T12:00:00+00:00')"
    )
    await store.connection.execute(
        "INSERT INTO artifacts (artifact_id, run_id, work_unit_id, attempt_id, name, "
        "kind, content, created_at) VALUES ('art_1', 'run_1', 'wu_1', 'att_1', "
        "'answer', 'TEXT', 'text', '2026-08-10T12:00:00+00:00')"
    )


def _insert_evidence(columns: str, values: str) -> str:
    return (
        f"INSERT INTO evidence (evidence_id, run_id, work_unit_id, artifact_id, {columns}, "
        f"detail, created_at) VALUES ('ev_1', 'run_1', 'wu_1', 'art_1', {values}, "
        "'a detail', '2026-08-10T12:00:00+00:00')"
    )


@pytest.mark.asyncio
async def test_the_evidence_columns_are_exactly_the_verdict_contract(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        columns = await _columns(store, "evidence")
    # Pinned as an exact set: the boolean is derived in the domain model, and
    # storing it again would let a row claim a verdict its ``result`` does not
    # support. An exact comparison catches it coming back under any name.
    assert columns == {
        "evidence_id",
        "run_id",
        "work_unit_id",
        "artifact_id",
        "kind",
        "rule",
        "result",
        "detail",
        "created_at",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("columns", "values"),
    [
        # A verdict is mandatory: there is no row that means "never decided".
        ("kind, rule", "'DETERMINISTIC_CHECK', 'VALID_JSON'"),
        ("kind, rule, result", "'DETERMINISTIC_CHECK', 'VALID_JSON', NULL"),
        # Only the three named verdicts exist.
        ("kind, rule, result", "'DETERMINISTIC_CHECK', 'VALID_JSON', 'LOOKS_FINE'"),
        ("kind, rule, result", "'DETERMINISTIC_CHECK', 'VALID_JSON', 'pass'"),
        ("kind, rule, result", "'DETERMINISTIC_CHECK', 'VALID_JSON', 1"),
        # A deterministic check has to name the rule it applied.
        ("kind, result", "'DETERMINISTIC_CHECK', 'PASS'"),
        ("kind, rule, result", "'DETERMINISTIC_CHECK', NULL, 'PASS'"),
        # A blank rule names nothing.
        ("kind, rule, result", "'DETERMINISTIC_CHECK', '   ', 'PASS'"),
        ("kind, rule, result", "'MODEL_REVIEW', '', 'PASS'"),
        # Only the two known kinds exist.
        ("kind, rule, result", "'VIBE_CHECK', 'VALID_JSON', 'PASS'"),
    ],
)
async def test_the_evidence_table_refuses_an_unprovable_row(
    database_path: Path, columns: str, values: str
) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        await _insert_artifact(store)
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(_insert_evidence(columns, values))


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["PASS", "FAIL", "INCONCLUSIVE"])
async def test_every_verdict_is_storable(database_path: Path, result: str) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        await _insert_artifact(store)
        await store.connection.execute(
            _insert_evidence(
                "kind, rule, result", f"'DETERMINISTIC_CHECK', 'VALID_JSON', '{result}'"
            )
        )
        stored = await _scalar(store, "SELECT result FROM evidence WHERE evidence_id = 'ev_1'")
        assert stored == result


@pytest.mark.asyncio
async def test_a_model_review_may_omit_the_rule(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        await _insert_artifact(store)
        await store.connection.execute(
            _insert_evidence("kind, result", "'MODEL_REVIEW', 'PASS'")
        )
        assert await _scalar(store, "SELECT COUNT(*) FROM evidence") == 1


# --- schema version guard -------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_version", [1, 2, SCHEMA_VERSION + 1])
async def test_an_incompatible_schema_version_is_rejected(
    database_path: Path, stored_version: int
) -> None:
    """Includes 2, the version this repair replaced.

    A database written before the repair has a ``passed`` column and a nullable
    ``result``. There is no migration, so opening it must refuse rather than read
    rows whose verdict cannot be trusted.
    """
    async with aiosqlite.connect(database_path) as connection:
        await connection.execute(f"PRAGMA user_version = {stored_version}")
        await connection.commit()
    store = SqliteStore(database_path)
    with pytest.raises(IncompatibleSchemaError) as excinfo:
        await store.open()
    assert "delete the development database" in str(excinfo.value)
    assert str(stored_version) in str(excinfo.value)
    assert store.is_open is False


@pytest.mark.asyncio
async def test_schema_file_is_idempotent_and_has_no_migration_code() -> None:
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    assert text.count("CREATE TABLE") == text.count("CREATE TABLE IF NOT EXISTS")
    assert text.count("CREATE INDEX") == text.count("CREATE INDEX IF NOT EXISTS")
    lowered = text.lower()
    assert "alter table" not in lowered
    assert "migration" not in lowered.replace("migration framework", "")
