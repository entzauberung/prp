"""Targeted tests for reusable runtime composition ownership."""

from pathlib import Path

import pytest

from prp_runtime.domain.enums import ExecutionLocation, IsolationMode
from prp_runtime.domain.models import Usage
from prp_runtime.providers.base import FinishReason, ProviderRequest, ProviderResponse
from prp_runtime.runtime.composition import RuntimeComposition, open_runtime_composition
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore


class FakeAdapter:
    def __init__(self) -> None:
        self.close_calls = 0

    @property
    def name(self) -> str:
        return "composition-fake"

    async def aclose(self) -> None:
        self.close_calls += 1

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        del request
        return ProviderResponse(
            text="unused",
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )


@pytest.mark.asyncio
async def test_composition_open_close_without_asgi_and_without_leaking_paths(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "owned.db"
    settings = Settings(database_path=database_path)
    adapter = FakeAdapter()
    composition = RuntimeComposition(settings, adapters={"worker": adapter})
    assert not database_path.exists()
    assert "owned.db" not in repr(composition)

    await composition.open()
    try:
        assert composition.store is not None
        assert composition.store.is_open
        assert composition.controller is not None
        assert composition.supervisor is not None
        assert composition.supervisor.running
        facts = composition.public_facts()
        assert facts["store_open"] is True
        assert facts["controller_present"] is True
        assert facts["execution_location"] == ExecutionLocation.CLOUD.value
        assert facts["isolation_mode"] == IsolationMode.SANDBOXED.value
        assert str(database_path) not in repr(composition)
        assert str(database_path) not in str(facts)
    finally:
        await composition.close()
        await composition.close()

    assert adapter.close_calls == 0
    assert composition.store is not None
    assert composition.store.is_open is False
    assert composition.public_facts()["closed"] is True


@pytest.mark.asyncio
async def test_injected_store_and_adapters_are_not_closed(tmp_path: Path) -> None:
    database_path = tmp_path / "injected.db"
    settings = Settings(database_path=database_path)
    store = SqliteStore(database_path)
    await store.open()
    adapter = FakeAdapter()

    composition = await open_runtime_composition(
        settings,
        adapters={"worker": adapter},
        store=store,
    )
    try:
        assert composition.owns_store is False
        assert composition.owns_adapters is False
        assert composition.store is store
        assert store.is_open
    finally:
        await composition.close()

    assert store.is_open
    assert adapter.close_calls == 0
    await store.close()


@pytest.mark.asyncio
async def test_owned_adapters_close_with_composition(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "owned-adapters.db")
    composition = RuntimeComposition(settings)
    # Construction remains side-effect free even when adapters will be owned.
    assert composition.owns_adapters is True
    await composition.open()
    await composition.close()
    assert composition.public_facts()["closed"] is True


@pytest.mark.asyncio
async def test_composition_open_failure_closes_owned_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "fail-open.db"
    adapter = FakeAdapter()

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("supervisor start failed")

    monkeypatch.setattr("prp_runtime.runtime.supervisor.RunSupervisor.start", boom)
    composition = RuntimeComposition(
        Settings(database_path=database_path), adapters={"worker": adapter}
    )
    with pytest.raises(RuntimeError, match="supervisor start failed"):
        await composition.open()
    facts = composition.public_facts()
    assert facts["closed"] is True
    assert facts["opened"] is False
    assert composition.store is not None
    assert composition.store.is_open is False
    assert adapter.close_calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        await composition.open()


@pytest.mark.asyncio
async def test_bridge_composition_open_does_not_resolve_server_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("BRIDGE composition must not resolve a server root")

    monkeypatch.setattr(
        "prp_runtime.runtime.tooling.WorkspaceResolver.resolve", boom
    )
    composition = RuntimeComposition(
        Settings(database_path=tmp_path / "bridge.db"),
        adapters={"worker": FakeAdapter()},
        execution_location=ExecutionLocation.BRIDGE,
    )
    await composition.open()
    try:
        facts = composition.public_facts()
        assert facts["execution_location"] == ExecutionLocation.BRIDGE.value
        assert composition.tool_runtime_provider is not None
        assert composition.tool_runtime_provider._resolver is None
    finally:
        await composition.close()
