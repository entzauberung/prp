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
    "agent_history",
    "artifacts",
    "evidence",
    "reservations",
    "events",
    "workspaces",
    "snapshots",
    "snapshot_files",
    "change_sets",
    "change_set_files",
    "tool_calls",
    "tool_results",
    "bridge_clients",
    "bridge_claims",
    "approvals",
    "leases",
    "sessions",
    "session_runs",
    "progressive_rounds",
    "merge_ledger",
}

EXPECTED_INDEXES = {
    "idx_runs_status",
    "idx_work_units_run",
    "idx_work_unit_dependencies_depends_on",
    "idx_work_unit_resource_claims_resource",
    "idx_attempts_run_status",
    "idx_agent_history_run",
    "idx_artifacts_work_unit",
    "idx_artifacts_attempt",
    "idx_evidence_artifact",
    "idx_evidence_work_unit",
    "idx_reservations_run_status",
    "idx_reservations_work_unit",
    "uq_reservations_active_dispatch",
    "idx_events_type",
    "idx_workspaces_owner",
    "idx_snapshots_workspace",
    "idx_snapshot_files_snapshot",
    "idx_change_sets_run",
    "idx_change_sets_workspace",
    "idx_change_set_files_path",
    "idx_tool_calls_run_status",
    "idx_tool_calls_work_unit",
    "idx_tool_calls_workspace",
    "idx_tool_results_status",
    "idx_bridge_clients_principal",
    "idx_bridge_clients_workspace",
    "idx_bridge_claims_call",
    "idx_bridge_claims_owner",
    "idx_bridge_claims_client",
    "uq_bridge_claims_active_call",
    "idx_approvals_owner",
    "idx_approvals_run",
    "idx_approvals_call",
    "idx_leases_owner",
    "idx_leases_run",
    "idx_progressive_rounds_run",
    "idx_merge_ledger_run",
    "idx_merge_ledger_snapshot",
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


async def _insert_workspace(
    store: SqliteStore,
    workspace_id: str = "ws_1",
    owner_id: str = "owner_1",
    alias: str = "project-main",
) -> None:
    await store.connection.execute(
        """
        INSERT INTO workspaces (
            workspace_id, owner_id, alias, source_type, server_alias, status, created_at
        ) VALUES (?, ?, ?, 'SERVER_ALIAS', 'repo-main', 'ACTIVE',
                  '2026-08-10T12:00:00+00:00')
        """,
        (workspace_id, owner_id, alias),
    )


async def _insert_snapshot(
    store: SqliteStore,
    snapshot_id: str = "snap_1",
    workspace_id: str = "ws_1",
    manifest_hash: str = "a" * 64,
) -> None:
    await store.connection.execute(
        """
        INSERT INTO snapshots (
            snapshot_id, workspace_id, status, manifest_hash, file_count, total_size,
            created_at
        ) VALUES (?, ?, 'CREATING', ?, 0, 0, '2026-08-10T12:00:00+00:00')
        """,
        (snapshot_id, workspace_id, manifest_hash),
    )


async def _insert_tool_call(
    store: SqliteStore,
    call_id: str = "tc_1",
    run_id: str = "run_1",
    work_unit_id: str = "wu_1",
    workspace_id: str = "ws_1",
    snapshot_id: str = "snap_1",
    idempotency_key: str = "request-1",
    status: str = "REQUESTED",
    started_at: str | None = None,
    completed_at: str | None = None,
) -> None:
    await store.connection.execute(
        """
        INSERT INTO tool_calls (
            call_id, run_id, work_unit_id, workspace_id, base_snapshot_id,
            idempotency_key, tool_name, effect, arguments_json, status,
            requested_at, started_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'read_file', 'READ', '{}', ?,
                  '2026-08-10T12:00:00+00:00', ?, ?)
        """,
        (
            call_id,
            run_id,
            work_unit_id,
            workspace_id,
            snapshot_id,
            idempotency_key,
            status,
            started_at,
            completed_at,
        ),
    )


async def _insert_bridge_session(
    store: SqliteStore,
    session_id: str = "ses_1",
    run_id: str = "run_1",
    workspace_id: str = "ws_1",
    owner_id: str = "owner_1",
) -> None:
    await store.connection.execute(
        """
        INSERT INTO sessions (
            session_id, principal_id, workspace_id, access_json,
            agent_options_json, status, created_at
        ) VALUES (?, ?, ?, '["READ"]', '{"execution_location":"BRIDGE"}',
                  'ACTIVE', '2026-08-10T12:00:00+00:00')
        """,
        (session_id, owner_id, workspace_id),
    )
    await store.connection.execute(
        """
        INSERT INTO session_runs (session_id, run_id, created_at)
        VALUES (?, ?, '2026-08-10T12:00:00+00:00')
        """,
        (session_id, run_id),
    )


async def _insert_bridge_client(
    store: SqliteStore,
    client_id: str = "cli_bridgeclient1",
    principal_id: str = "owner_1",
    workspace_id: str = "ws_1",
    tools_json: str = '["read_file"]',
    effects_json: str = '["READ"]',
    fingerprint: str = "a" * 64,
    status: str = "ACTIVE",
    created_at: str = "2026-08-10T12:00:00+00:00",
    last_seen_at: str | None = None,
    disabled_at: str | None = None,
) -> None:
    await store.connection.execute(
        """
        INSERT INTO bridge_clients (
            client_id, principal_id, workspace_id, tools_json, effects_json,
            max_argument_bytes, max_output_bytes, max_runtime_ms,
            capability_fingerprint, status, created_at, last_seen_at, disabled_at
        ) VALUES (?, ?, ?, ?, ?, 65536, 262144, NULL, ?, ?, ?, ?, ?)
        """,
        (
            client_id,
            principal_id,
            workspace_id,
            tools_json,
            effects_json,
            fingerprint,
            status,
            created_at,
            last_seen_at,
            disabled_at,
        ),
    )


async def _insert_bridge_claim(
    store: SqliteStore,
    claim_id: str = "claim_1",
    call_id: str = "tc_1",
    run_id: str = "run_1",
    session_id: str = "ses_1",
    workspace_id: str = "ws_1",
    snapshot_id: str = "snap_1",
    owner_id: str = "owner_1",
    client_id: str = "cli_bridgeclient1",
    idempotency_key: str = "claim-request-1",
    status: str = "ACTIVE",
    claimed_at: str = "2026-08-10T12:00:00+00:00",
    expires_at: str = "2026-08-10T12:05:00+00:00",
    closed_at: str | None = None,
) -> None:
    await store.connection.execute(
        """
        INSERT INTO bridge_claims (
            claim_id, call_id, run_id, session_id, workspace_id, snapshot_id,
            owner_id, client_id, idempotency_key, fingerprint, status, claimed_at,
            expires_at, closed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            claim_id,
            call_id,
            run_id,
            session_id,
            workspace_id,
            snapshot_id,
            owner_id,
            client_id,
            idempotency_key,
            "a" * 64,
            status,
            claimed_at,
            expires_at,
            closed_at,
        ),
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
async def test_workspace_snapshot_columns_and_owner_contract_are_persisted(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        assert await _columns(store, "workspaces") == {
            "workspace_id",
            "owner_id",
            "alias",
            "source_type",
            "server_alias",
            "bridge_grant",
            "status",
            "created_at",
            "closed_at",
        }
        assert await _columns(store, "snapshots") == {
            "snapshot_id",
            "workspace_id",
            "status",
            "manifest_hash",
            "file_count",
            "total_size",
            "created_at",
            "completed_at",
        }
        assert await _columns(store, "snapshot_files") == {
            "snapshot_id",
            "path",
            "sha256",
            "size",
            "entry_type",
        }
        assert await _columns(store, "change_sets") == {
            "change_set_id",
            "run_id",
            "tool_call_id",
            "workspace_id",
            "base_snapshot_id",
            "new_snapshot_id",
            "patch_text",
            "patch_sha256",
            "created_at",
        }
        assert await _columns(store, "change_set_files") == {
            "change_set_id",
            "path",
            "action",
            "before_sha256",
            "before_size",
            "after_sha256",
            "after_size",
        }


@pytest.mark.asyncio
async def test_tool_tables_have_the_closed_call_and_result_contract(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        assert await _columns(store, "tool_calls") == {
            "call_id",
            "run_id",
            "work_unit_id",
            "workspace_id",
            "base_snapshot_id",
            "idempotency_key",
            "tool_name",
            "effect",
            "arguments_json",
            "status",
            "requested_at",
            "started_at",
            "completed_at",
        }
        assert await _columns(store, "tool_results") == {
            "call_id",
            "status",
            "result_json",
            "output",
            "truncated",
            "changed_paths_json",
            "exit_code",
            "error_category",
            "error_message",
            "completed_at",
        }
        assert await _columns(store, "bridge_clients") == {
            "client_id",
            "principal_id",
            "workspace_id",
            "tools_json",
            "effects_json",
            "max_argument_bytes",
            "max_output_bytes",
            "max_runtime_ms",
            "capability_fingerprint",
            "status",
            "created_at",
            "last_seen_at",
            "disabled_at",
        }
        assert await _columns(store, "bridge_claims") == {
            "claim_id",
            "call_id",
            "run_id",
            "session_id",
            "workspace_id",
            "snapshot_id",
            "owner_id",
            "client_id",
            "idempotency_key",
            "fingerprint",
            "status",
            "claimed_at",
            "expires_at",
            "closed_at",
        }
        assert "claimant_id" not in await _columns(store, "bridge_claims")


@pytest.mark.asyncio
async def test_bridge_claim_scope_lease_and_terminal_constraints_are_enforced(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        await _insert_workspace(store)
        await _insert_snapshot(store)
        await _insert_tool_call(store)
        await _insert_bridge_session(store)
        await _insert_bridge_client(store)
        await store.connection.commit()

        await _insert_bridge_claim(store)
        await store.connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_bridge_claim(store, claim_id="claim_2", idempotency_key="claim-2")
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_bridge_claim(store, claim_id="claim_3")

        await store.connection.execute(
            "UPDATE bridge_claims SET status = 'SETTLED', "
            "closed_at = '2026-08-10T12:01:00+00:00' WHERE claim_id = 'claim_1'"
        )
        await store.connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                "UPDATE bridge_claims SET status = 'ACTIVE', closed_at = NULL "
                "WHERE claim_id = 'claim_1'"
            )

        with pytest.raises(sqlite3.IntegrityError):
            await _insert_bridge_claim(
                store,
                claim_id="claim_4",
                call_id="tc_missing",
                idempotency_key="claim-4",
            )
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_bridge_claim(
                store,
                claim_id="claim_5",
                client_id="cli_missing",
                idempotency_key="claim-5",
            )
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_bridge_client(
                store,
                client_id="cli_emptytools",
                tools_json="",
                fingerprint="b" * 64,
            )


@pytest.mark.asyncio
async def test_tool_call_foreign_keys_and_idempotency_are_enforced(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        await _insert_workspace(store)
        await _insert_snapshot(store)
        await _insert_tool_call(store)
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_tool_call(store, call_id="tc_2")
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_tool_call(store, call_id="tc_2", idempotency_key="request-1")
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_tool_call(store, call_id="tc_3", workspace_id="ws_missing")
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_tool_call(store, call_id="tc_4", snapshot_id="snap_missing")
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_tool_call(store, call_id="tc_5", work_unit_id="wu_missing")


@pytest.mark.asyncio
async def test_tool_call_lifecycle_checks_reject_impossible_rows(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        await _insert_workspace(store)
        await _insert_snapshot(store)
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_tool_call(store, status="NOT_A_STATUS")
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_tool_call(store, status="RUNNING")
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_tool_call(
                store,
                status="REQUESTED",
                started_at="2026-08-10T12:00:01+00:00",
            )
        await _insert_tool_call(
            store,
            call_id="tc_running",
            idempotency_key="request-running",
            status="RUNNING",
            started_at="2026-08-10T12:00:01+00:00",
        )
        await _insert_tool_call(
            store,
            call_id="tc_done",
            idempotency_key="request-done",
            status="SUCCEEDED",
            completed_at="2026-08-10T12:00:02+00:00",
        )


@pytest.mark.asyncio
async def test_tool_result_is_one_to_one_terminal_and_bounded(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        await _insert_workspace(store)
        await _insert_snapshot(store)
        await _insert_tool_call(
            store,
            status="SUCCEEDED",
            completed_at="2026-08-10T12:00:02+00:00",
        )
        await store.connection.execute(
            """
            INSERT INTO tool_results (
                call_id, status, result_json, output, truncated,
                changed_paths_json, completed_at
            ) VALUES ('tc_1', 'SUCCEEDED', '{"content":"ok"}', 'ok', 0,
                      '["src/main.py"]', '2026-08-10T12:00:02+00:00')
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                "INSERT INTO tool_results (call_id, status, completed_at) "
                "VALUES ('tc_1', 'CANCELLED', '2026-08-10T12:00:03+00:00')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                "INSERT INTO tool_results (call_id, status, completed_at) "
                "VALUES ('tc_missing', 'SUCCEEDED', '2026-08-10T12:00:03+00:00')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                "INSERT INTO tool_results (call_id, status, completed_at) "
                "VALUES ('tc_bad', 'PENDING', '2026-08-10T12:00:03+00:00')"
            )


@pytest.mark.asyncio
async def test_tool_result_errors_and_output_limits_are_constrained(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        await _insert_workspace(store)
        await _insert_snapshot(store)
        await _insert_tool_call(
            store,
            status="FAILED",
            completed_at="2026-08-10T12:00:02+00:00",
        )
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                "INSERT INTO tool_results (call_id, status, completed_at) "
                "VALUES ('tc_1', 'FAILED', '2026-08-10T12:00:02+00:00')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                """
                INSERT INTO tool_results (
                    call_id, status, error_category, error_message, completed_at
                ) VALUES ('tc_1', 'FAILED', NULL, 'x',
                          '2026-08-10T12:00:02+00:00')
                """
            )
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                """
                INSERT INTO tool_results (
                    call_id, status, error_category, error_message, output, completed_at
                ) VALUES ('tc_1', 'FAILED', 'UNKNOWN', 'failed', :output,
                          '2026-08-10T12:00:02+00:00')
                """,
                {"output": "x" * 262145},
            )


@pytest.mark.asyncio
async def test_workspace_snapshot_constraints_preserve_owner_and_manifest_identity(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_workspace(store)
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_workspace(store, workspace_id="ws_2")
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                """
                INSERT INTO workspaces (
                    workspace_id, owner_id, alias, source_type, server_alias,
                    bridge_grant, status, created_at
                ) VALUES ('ws_3', 'owner_1', 'bridge', 'BRIDGE_GRANT',
                          'repo-main', 'grant_1', 'ACTIVE',
                          '2026-08-10T12:00:00+00:00')
                """
            )
        await _insert_snapshot(store)
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_snapshot(store, snapshot_id="snap_2")
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_snapshot(store, snapshot_id="snap_3", manifest_hash="B" * 64)
        with pytest.raises(sqlite3.IntegrityError):
            await _insert_snapshot(store, snapshot_id="snap_4", workspace_id="ws_missing")


@pytest.mark.asyncio
async def test_snapshot_files_reject_unsafe_paths_and_keep_audit_rows(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_workspace(store)
        await _insert_snapshot(store)
        insert = """
            INSERT INTO snapshot_files (snapshot_id, path, sha256, size, entry_type)
            VALUES ('snap_1', ?, ?, 3, 'FILE')
        """
        await store.connection.execute(insert, ("src/main.py", "a" * 64))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("../secret", "b" * 64))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("/etc/passwd", "b" * 64))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("README.md", "B" * 64))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("src/main.py", "c" * 64))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                "DELETE FROM workspaces WHERE workspace_id = 'ws_1'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute("DELETE FROM snapshots WHERE snapshot_id = 'snap_1'")
        assert await _scalar(store, "SELECT COUNT(*) FROM snapshot_files") == 1


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
async def test_reservation_schema_has_the_admission_and_settlement_columns(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        assert await _columns(store, "reservations") == {
            "reservation_id",
            "run_id",
            "work_unit_id",
            "dispatch_key",
            "attempt_units",
            "estimated_input_tokens",
            "estimated_output_tokens",
            "token_upper_bound",
            "strong_token_upper_bound",
            "capacity_key",
            "status",
            "created_at",
            "held_at",
            "completed_at",
            "measured_input_tokens",
            "measured_output_tokens",
            "measured_strong_model_tokens",
            "measured_elapsed_ms",
        }


@pytest.mark.asyncio
async def test_reservation_schema_rejects_duplicate_dispatch_and_invalid_lifecycle(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        await _insert_work_unit(store)
        insert = """
            INSERT INTO reservations (
                reservation_id, run_id, work_unit_id, dispatch_key, attempt_units,
                status, created_at
            ) VALUES (?, 'run_1', 'wu_1', ?, 1, ?, '2026-08-10T12:00:00+00:00')
        """
        await store.connection.execute(insert, ("res_1", "dispatch-1", "PENDING"))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("res_2", "dispatch-1", "PENDING"))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("res_3", "dispatch-2", "UNKNOWN"))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(insert, ("res_4", "   ", "PENDING"))
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                """
                INSERT INTO reservations (
                    reservation_id, run_id, work_unit_id, dispatch_key, attempt_units,
                    status, created_at, held_at, completed_at
                ) VALUES ('res_5', 'run_1', 'wu_1', 'dispatch-3', 1, 'SETTLED',
                          '2026-08-10T12:00:00+00:00', NULL,
                          '2026-08-10T12:00:01+00:00')
                """
            )


@pytest.mark.asyncio
async def test_reservation_schema_enforces_run_and_work_unit_foreign_keys(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                """
                INSERT INTO reservations (
                    reservation_id, run_id, work_unit_id, dispatch_key, attempt_units,
                    status, created_at
                ) VALUES ('res_missing', 'run_1', 'wu_missing', 'dispatch-1', 1,
                          'PENDING', '2026-08-10T12:00:00+00:00')
                """
            )


@pytest.mark.asyncio
async def test_runs_store_final_work_unit_id_and_work_units_have_same_graph_key(
    database_path: Path,
) -> None:
    async with SqliteStore(database_path) as store:
        assert {"final_work_unit_id"} <= await _columns(store, "runs")
        assert {"work_unit_id", "run_id", "graph_version"} <= await _columns(
            store, "work_units"
        )
        assert {
            "lineage_key",
            "dependency_fingerprint",
            "content_fingerprint",
        } <= await _columns(store, "work_units")


@pytest.mark.asyncio
async def test_work_unit_schema_rejects_malformed_fingerprints(database_path: Path) -> None:
    async with SqliteStore(database_path) as store:
        await _insert_run(store)
        with pytest.raises(sqlite3.IntegrityError):
            await store.connection.execute(
                """
                INSERT INTO work_units (
                    work_unit_id, run_id, graph_version, lineage_key,
                    dependency_fingerprint, content_fingerprint, name, instruction,
                    output_json, status, created_at
                ) VALUES ('wu_bad', 'run_1', 1, 'node', 'bad', :content,
                          'unit', 'do it', '{}', 'PENDING', '2026-08-10T12:00:00+00:00')
                """,
                {"content": "0" * 64},
            )


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
    assert "claimant_id" not in text
