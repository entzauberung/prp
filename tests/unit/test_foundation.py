"""Foundation tests: package identity, metadata consistency and network isolation."""

import tomllib
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError

import prp_runtime
from prp_runtime.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_VERSION = "0.0.3"
EXPECTED_LICENSE = "Apache-2.0"


@pytest.fixture(autouse=True)
def disable_environment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep HTTPX tests deterministic when the host exports an unsupported proxy URL."""
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


def _pyproject() -> dict[str, object]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_package_exposes_expected_identity() -> None:
    assert prp_runtime.__version__ == EXPECTED_VERSION
    assert prp_runtime.LICENSE_EXPRESSION == EXPECTED_LICENSE
    assert prp_runtime.package_info() == {
        "name": "prp-runtime",
        "version": EXPECTED_VERSION,
        "license": EXPECTED_LICENSE,
    }


def test_project_metadata_matches_package() -> None:
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    assert project["name"] == "prp-runtime"
    assert project["version"] == EXPECTED_VERSION
    assert project["license"] == EXPECTED_LICENSE
    assert project["requires-python"] == ">=3.12"


def test_installed_distribution_matches_package() -> None:
    project = _pyproject()["project"]
    assert project["version"] == prp_runtime.__version__ == EXPECTED_VERSION


def test_license_files_exist() -> None:
    for name in ("LICENSE-APACHE", "NOTICE", "TRADEMARKS.md"):
        assert (PROJECT_ROOT / name).is_file()


def test_real_network_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="real network access is disabled"):
        httpx.get("http://192.0.2.1/blocked", timeout=1.0)


def test_unregistered_request_fails_under_respx(mocked_http: respx.MockRouter) -> None:
    mocked_http.get("http://declared.invalid/ok").respond(200, text="ok")
    assert httpx.get("http://declared.invalid/ok").text == "ok"
    with pytest.raises(AssertionError):
        httpx.get("http://undeclared.invalid/missing")


def test_server_workspace_roots_are_immutable_and_redacted() -> None:
    settings = Settings(
        workspace_roots={
            "repo-main": "/srv/repos/main",
            "scratch": "/var/lib/prp/scratch",
        }
    )

    assert settings.workspace_roots.aliases == ("repo-main", "scratch")
    assert settings.workspace_roots.root_for("repo-main") == "/srv/repos/main"
    assert settings.workspace_roots.model_dump() == {}
    assert "/srv/repos/main" not in repr(settings)
    assert "/srv/repos/main" not in settings.model_dump_json()
    with pytest.raises(ValidationError):
        settings.workspace_roots.entries = ()  # type: ignore[misc]


@pytest.mark.parametrize("raw", ["[]", '"/srv/repos/main"', "{not-json"])
def test_workspace_roots_environment_requires_a_strict_json_object(raw: str) -> None:
    with pytest.raises(ValidationError):
        Settings.from_env({"PRP_WORKSPACE_ROOTS": raw})
