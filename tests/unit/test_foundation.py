"""Foundation tests: package identity, metadata consistency and network isolation."""

import tomllib
from importlib import metadata
from pathlib import Path

import httpx
import pytest
import respx

import prp_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_VERSION = "0.0.1"
EXPECTED_LICENSE = "MIT OR Apache-2.0"


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
    assert metadata.version("prp-runtime") == EXPECTED_VERSION


def test_license_files_exist() -> None:
    for name in ("LICENSE-MIT", "LICENSE-APACHE", "NOTICE"):
        assert (PROJECT_ROOT / name).is_file()


def test_real_network_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="real network access is disabled"):
        httpx.get("http://192.0.2.1/blocked", timeout=1.0)


def test_unregistered_request_fails_under_respx(mocked_http: respx.MockRouter) -> None:
    mocked_http.get("http://declared.invalid/ok").respond(200, text="ok")
    assert httpx.get("http://declared.invalid/ok").text == "ok"
    with pytest.raises(AssertionError):
        httpx.get("http://undeclared.invalid/missing")
