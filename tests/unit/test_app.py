"""Targeted tests for settings and the application factory."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import prp_runtime.app as app_module
from prp_runtime import __version__
from prp_runtime.app import build_adapters, create_app
from prp_runtime.domain.enums import ModelRole
from prp_runtime.providers.base import ModelProfile
from prp_runtime.settings import Settings
from prp_runtime.workspace.sandbox import SandboxCapabilities


class FakeAdapter:
    def __init__(self) -> None:
        self.close_calls = 0

    @property
    def name(self) -> str:
        return "fake"

    async def aclose(self) -> None:
        self.close_calls += 1


class FailingCloseAdapter(FakeAdapter):
    async def aclose(self) -> None:
        self.close_calls += 1
        raise RuntimeError("adapter close failed")


def _profile(alias: str, role: ModelRole) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        provider="openai_compatible",
        model=f"{alias}-model",
        role=role,
        base_url="https://models.internal/v1",
        context_window_tokens=16_000,
        max_output_tokens=2_000,
    )


def test_health_returns_version_only(tmp_path: Path) -> None:
    app = create_app(Settings(database_path=tmp_path / "health.db"))
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_settings_are_immutable_and_reject_unknown_fields() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        Settings(unexpected_option=1)
    with pytest.raises(ValidationError):
        settings.max_request_bytes = 10


@pytest.mark.parametrize("limit_field", ["max_request_bytes", "max_input_chars"])
@pytest.mark.parametrize("bad_value", [0, -1])
def test_settings_reject_non_positive_limits(limit_field: str, bad_value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(**{limit_field: bad_value})


def test_settings_from_env_reads_known_variables() -> None:
    settings = Settings.from_env(
        {
            "PRP_DATABASE_PATH": "/tmp/prp-test.db",
            "PRP_MAX_REQUEST_BYTES": "2048",
            "PRP_MAX_INPUT_CHARS": "512",
            "PRP_LOG_LEVEL": "DEBUG",
            "PRP_ALLOW_HOST_YOLO": "true",
            "UNRELATED_VAR": "ignored",
        }
    )
    assert settings.database_path == Path("/tmp/prp-test.db")
    assert settings.max_request_bytes == 2048
    assert settings.max_input_chars == 512
    assert settings.log_level == "DEBUG"
    assert settings.allow_host_yolo is True


def test_host_yolo_setting_is_closed_by_default() -> None:
    assert Settings().allow_host_yolo is False


def test_settings_from_env_rejects_unknown_prefixed_variable() -> None:
    with pytest.raises(ValueError, match="PRP_TYPO"):
        Settings.from_env({"PRP_TYPO": "1"})


def test_settings_from_env_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env({"PRP_MAX_REQUEST_BYTES": "not-a-number"})
    with pytest.raises(ValidationError):
        Settings.from_env({"PRP_LOG_LEVEL": "TRACE"})


def test_create_app_does_not_touch_the_database_path(tmp_path: Path) -> None:
    database_path = tmp_path / "should-not-exist.db"
    app = create_app(Settings(database_path=database_path))
    assert app.state.settings.database_path == database_path
    assert not database_path.exists()


def test_build_adapters_uses_every_configured_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        leader_profile=_profile("leader", ModelRole.PLANNER),
        worker_profile=_profile("worker", ModelRole.WORKER),
        cascade_profiles=(
            _profile("worker-medium", ModelRole.WORKER),
            _profile("worker-large", ModelRole.WORKER),
        ),
    )
    built_for: list[str] = []

    def fake_provider(profile: ModelProfile) -> object:
        built_for.append(profile.alias)
        return object()

    monkeypatch.setattr(app_module, "OpenAICompatibleProvider", fake_provider)

    adapters = build_adapters(settings)

    assert list(adapters) == ["leader", "worker", "worker-medium", "worker-large"]
    assert built_for == list(adapters)


def test_default_lifespan_builds_cascade_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        database_path=tmp_path / "app.db",
        worker_profile=_profile("worker", ModelRole.WORKER),
        cascade_profiles=(
            _profile("worker-medium", ModelRole.WORKER),
            _profile("worker-large", ModelRole.WORKER),
        ),
    )
    created: dict[str, object] = {}

    def fake_provider(profile: ModelProfile) -> FakeAdapter:
        adapter = FakeAdapter()
        created[profile.alias] = adapter
        return adapter

    monkeypatch.setattr(app_module, "OpenAICompatibleProvider", fake_provider)
    app = create_app(settings)

    with TestClient(app):
        assert app.state.adapters == created
        assert list(app.state.adapters) == ["worker", "worker-medium", "worker-large"]

    assert set(created) == {"worker", "worker-medium", "worker-large"}
    assert all(
        isinstance(adapter, FakeAdapter) and adapter.close_calls == 1
        for adapter in created.values()
    )


def test_ready_requires_local_store_controller_profiles_and_adapters(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            database_path=tmp_path / "ready.db",
            leader_profile=_profile("leader", ModelRole.PLANNER),
            worker_profile=_profile("worker", ModelRole.WORKER),
        ),
        adapters={"leader": FakeAdapter(), "worker": FakeAdapter()},
    )
    with TestClient(app) as client:
        response = client.get("/ready")
        capabilities = app.state.sandbox_capabilities
        assert isinstance(capabilities, SandboxCapabilities)
        expected_status = 200 if capabilities.ready else 503
        assert response.status_code == expected_status
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {
            "status": "ready" if capabilities.ready else "not_ready",
            "store_open": True,
            "controller_present": True,
            "profiles_configured": True,
            "adapters_ready": True,
            "sandbox_ready": capabilities.ready,
        }


def test_ready_fails_closed_when_active_probe_is_not_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capabilities = SandboxCapabilities(
        backend="bubblewrap",
        available=True,
        version="0.9.0",
        reason="bubblewrap active probe returned an error",
    )
    monkeypatch.setattr(app_module, "probe_bwrap", lambda: capabilities)
    app = create_app(
        Settings(
            database_path=tmp_path / "sandbox-not-ready.db",
            leader_profile=_profile("leader", ModelRole.PLANNER),
            worker_profile=_profile("worker", ModelRole.WORKER),
        ),
        adapters={"leader": FakeAdapter(), "worker": FakeAdapter()},
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["sandbox_ready"] is False


def test_ready_rejects_missing_required_profile_without_provider_call(tmp_path: Path) -> None:
    app = create_app(
        Settings(database_path=tmp_path / "not-ready.db"),
        adapters={},
    )
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["status"] == "not_ready"
        assert response.json()["profiles_configured"] is False


def test_injected_adapters_are_not_closed_by_lifespan(tmp_path: Path) -> None:
    injected = FakeAdapter()
    app = create_app(
        Settings(database_path=tmp_path / "injected.db"),
        adapters={"injected": injected},
    )
    with TestClient(app):
        pass
    assert injected.close_calls == 0


def test_owned_adapter_close_failure_still_closes_owned_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        database_path=tmp_path / "close-failure.db",
        leader_profile=_profile("leader", ModelRole.PLANNER),
        worker_profile=_profile("worker", ModelRole.WORKER),
    )
    created: list[FailingCloseAdapter] = []

    def fake_provider(profile: ModelProfile) -> FailingCloseAdapter:
        adapter = FailingCloseAdapter()
        created.append(adapter)
        return adapter

    monkeypatch.setattr(app_module, "OpenAICompatibleProvider", fake_provider)
    app = create_app(settings)
    with pytest.raises(RuntimeError, match="adapter close failed"):
        with TestClient(app):
            pass
    assert app.state.store is not None
    assert app.state.store.is_open is False
    assert all(adapter.close_calls == 1 for adapter in created)


def test_create_app_registers_all_binding_routes_without_starting_lifespan() -> None:
    app = create_app(Settings())
    paths = set(app.openapi()["paths"])
    assert {
        "/v1/runs",
        "/v1/responses",
        "/v1/chat/completions",
        "/v1/messages",
    } <= paths
    assert app.state.store is None
    assert app.state.controller is None
