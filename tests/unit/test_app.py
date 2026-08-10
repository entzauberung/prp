"""Targeted tests for settings and the application factory."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from prp_runtime import __version__
from prp_runtime.app import create_app
from prp_runtime.settings import Settings


def test_health_returns_version_only() -> None:
    app = create_app(Settings())
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
            "UNRELATED_VAR": "ignored",
        }
    )
    assert settings.database_path == Path("/tmp/prp-test.db")
    assert settings.max_request_bytes == 2048
    assert settings.max_input_chars == 512
    assert settings.log_level == "DEBUG"


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
