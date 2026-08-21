"""Offline tests for external transport, temporary resources, and budget gates."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

_EXTERNAL_TESTS_DIR = Path(__file__).resolve().parents[3] / "external_tests"
if str(_EXTERNAL_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_TESTS_DIR))

from support import (  # noqa: E402
    BudgetCounter,
    ExternalGateError,
    create_external_http_client,
    temporary_external_resources,
    validate_external_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://models.example/v1",
        "https://models.example.evil.example/v1",
        "https://user:password@models.example/v1",
        "https://127.0.0.1/v1",
        "https://[::1]/v1",
    ],
)
def test_url_validation_rejects_confusion_http_userinfo_and_ip(url: str) -> None:
    with pytest.raises(ExternalGateError):
        validate_external_url(url, ["models.example"])


def test_url_validation_requires_exact_allowlisted_host() -> None:
    validate_external_url("https://models.example/v1", ["models.example"])
    with pytest.raises(ExternalGateError):
        validate_external_url("https://sub.models.example/v1", ["models.example"])


def test_url_validation_rejects_an_unselected_matrix_host() -> None:
    with pytest.raises(ExternalGateError):
        validate_external_url("https://api.deepseek.com/v1", ["open.bigmodel.cn"])


@pytest.mark.asyncio
async def test_client_disables_ambient_proxy_and_redirect_following() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request)

    async with create_external_http_client(
        ["models.example"], transport=httpx.MockTransport(handler)
    ) as client:
        assert client.follow_redirects is False
        assert getattr(client, "_trust_env") is False
        response = await client.get("https://models.example/v1")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_client_rejects_redirect_to_non_allowlisted_host() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "https://evil.example/next"},
            request=request,
        )

    async with create_external_http_client(
        ["models.example"], transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(ExternalGateError, match="redirect"):
            await client.get("https://models.example/v1")


def test_temporary_resources_are_isolated_and_cleanup_is_idempotent(tmp_path: Path) -> None:
    with temporary_external_resources(tmp_path) as resources:
        assert resources.database_path.is_file()
        assert resources.workspace_path.is_dir()
        assert not resources.root.is_relative_to(Path.cwd())
        root = resources.root
        resources.cleanup()
        resources.cleanup()
        assert not root.exists()


def test_budget_counts_failed_attempts_and_rejects_token_overrun() -> None:
    counter = BudgetCounter(
        max_provider_calls=3,
        max_attempts_per_alias=2,
        max_successful_calls_per_alias=2,
        max_output_tokens=256,
    )
    first = counter.reserve("TEST", 256)
    counter.settle(first, success=False)
    assert counter.attempts == 1
    with pytest.raises(ExternalGateError, match="token"):
        counter.reserve("TEST", 257)
    second = counter.reserve("TEST", 1)
    counter.settle(second, success=True, observed_output_tokens=1)
    assert counter.successful_calls("TEST") == 1
    with pytest.raises(ExternalGateError, match="attempt"):
        counter.reserve("TEST", 1)


def test_budget_reservation_is_atomic_under_race() -> None:
    counter = BudgetCounter(
        max_provider_calls=1,
        max_attempts_per_alias=1,
        max_successful_calls_per_alias=1,
        max_output_tokens=256,
    )

    def attempt() -> bool:
        try:
            reservation = counter.reserve("RACE", 1)
        except ExternalGateError:
            return False
        counter.settle(reservation, success=False)
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: attempt(), range(8)))
    assert sum(results) == 1
    assert counter.attempts == 1


def test_temp_fixture_does_not_serialize_secrets(tmp_path: Path) -> None:
    with temporary_external_resources(tmp_path) as resources:
        assert "api_key" not in json.dumps(
            {
                "database_path": str(resources.database_path),
                "workspace": str(resources.workspace_path),
            }
        )
