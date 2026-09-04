"""Targeted SQLite tests for durable Bridge client registration."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from prp_runtime.domain.models import (
    BridgeClientStatus,
    RegisteredBridgeClient,
    new_client_id,
)
from prp_runtime.domain.values import new_principal_id, new_workspace_id
from prp_runtime.storage.sqlite import (
    DuplicateEntityError,
    MissingEntityError,
    SCHEMA_VERSION,
    SqliteStore,
)
from prp_runtime.workspace.models import (
    Workspace,
    WorkspaceSource,
    WorkspaceSourceType,
)

T0 = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
FINGERPRINT = "a" * 64


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "bridge-clients.db"


@pytest_asyncio.fixture
async def store(database_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(database_path) as opened:
        yield opened


def make_workspace(owner_id: str, alias: str = "project-main") -> Workspace:
    return Workspace(
        workspace_id=new_workspace_id(),
        owner_id=owner_id,
        alias=alias,
        source=WorkspaceSource(
            source_type=WorkspaceSourceType.SERVER_ALIAS,
            server_alias="repo-main",
        ),
        created_at=T0,
    )


def make_client(
    *,
    principal_id: str,
    workspace_id: str,
    **overrides: object,
) -> RegisteredBridgeClient:
    data: dict[str, object] = {
        "client_id": new_client_id(),
        "principal_id": principal_id,
        "workspace_id": workspace_id,
        "capability_fingerprint": FINGERPRINT,
        "created_at": T0,
    }
    data.update(overrides)
    return RegisteredBridgeClient(**data)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_bridge_client_register_get_list_disable_are_owner_scoped(
    store: SqliteStore,
) -> None:
    owner = new_principal_id()
    other = new_principal_id()
    workspace = make_workspace(owner)
    foreign = make_workspace(other, alias="other")
    await store.create_workspace(workspace)
    await store.create_workspace(foreign)

    client = make_client(principal_id=owner, workspace_id=workspace.workspace_id)
    persisted = await store.register_bridge_client(client)
    assert persisted == client
    assert await store.get_bridge_client(client.client_id, principal_id=owner) == client
    assert await store.list_bridge_clients(principal_id=owner) == (client,)
    assert await store.list_bridge_clients(
        principal_id=owner, workspace_id=workspace.workspace_id
    ) == (client,)
    assert await store.list_bridge_clients(principal_id=other) == ()
    with pytest.raises(MissingEntityError):
        await store.get_bridge_client(client.client_id, principal_id=other)

    payload = persisted.model_dump(mode="json")
    assert "token" not in payload
    assert "workspace_root" not in payload
    assert "api_key" not in payload
    assert "strategy" not in payload

    disabled_at = T0 + timedelta(minutes=5)
    disabled = await store.disable_bridge_client(
        client.client_id, principal_id=owner, at=disabled_at
    )
    assert disabled.status is BridgeClientStatus.DISABLED
    assert disabled.disabled_at == disabled_at
    reopened = SqliteStore(store.database_path)
    async with reopened:
        restored = await reopened.get_bridge_client(client.client_id, principal_id=owner)
    assert restored == disabled
    with pytest.raises(MissingEntityError):
        await store.disable_bridge_client(
            client.client_id, principal_id=other, at=disabled_at
        )


@pytest.mark.asyncio
async def test_bridge_client_duplicate_and_cross_owner_registration_fail(
    store: SqliteStore,
) -> None:
    owner = new_principal_id()
    other = new_principal_id()
    workspace = make_workspace(owner)
    foreign = make_workspace(other, alias="other")
    await store.create_workspace(workspace)
    await store.create_workspace(foreign)

    client = make_client(principal_id=owner, workspace_id=workspace.workspace_id)
    assert await store.register_bridge_client(client) == client
    assert await store.register_bridge_client(client) == client
    with pytest.raises(DuplicateEntityError):
        await store.register_bridge_client(
            client.model_copy(update={"capability_fingerprint": "b" * 64})
        )
    with pytest.raises(MissingEntityError):
        await store.register_bridge_client(
            client.model_copy(
                update={
                    "client_id": new_client_id(),
                    "principal_id": other,
                    "workspace_id": workspace.workspace_id,
                }
            )
        )
    stolen = make_client(
        principal_id=other,
        workspace_id=foreign.workspace_id,
        client_id=client.client_id,
    )
    with pytest.raises(MissingEntityError):
        await store.register_bridge_client(stolen)


@pytest.mark.asyncio
async def test_schema_version_gate_and_existing_workspace_facts_remain_readable(
    store: SqliteStore, database_path: Path
) -> None:
    owner = new_principal_id()
    workspace = make_workspace(owner)
    await store.create_workspace(workspace)
    assert SCHEMA_VERSION == 12
    async with SqliteStore(database_path) as reopened:
        restored = await reopened.get_workspace(
            workspace.workspace_id, owner_id=owner
        )
    assert restored == workspace
    assert restored.source.server_alias == "repo-main"
