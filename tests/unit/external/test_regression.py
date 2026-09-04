"""Offline historical/subset regression floor for WO-011.

Live provider tests in ``external_tests/test_regression.py`` remain an
unsupported declared protocol for this blueprint command. This module only
proves the offline compatibility subset: bindings reject undeclared fields and
a fake-adapter native run still completes without network.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from prp_runtime.api.bindings import reject_unsupported_fields
from prp_runtime.app import create_app
from prp_runtime.domain.enums import ModelRole, RunStatus
from prp_runtime.domain.errors import ErrorCode, PrpError
from prp_runtime.domain.models import Usage
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings

WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="fake",
    model="regression-worker",
    role=ModelRole.WORKER,
    base_url="https://models.invalid/v1",
    context_window_tokens=16_000,
    max_output_tokens=2_000,
)


class FakeAdapter:
    name = "regression-fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            text="regression answer",
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )


def test_declared_protocol_subset_rejects_stream_tools_and_unknown_fields() -> None:
    reject_unsupported_fields(
        {"model": "gpt", "messages": []},
        allowed=frozenset({"model", "messages"}),
    )
    try:
        reject_unsupported_fields(
            {"model": "gpt", "stream": True},
            allowed=frozenset({"model"}),
        )
    except PrpError as error:
        assert error.detail.code is ErrorCode.UNSUPPORTED_STREAM_MODE
    else:
        raise AssertionError("stream mode must remain unsupported")
    try:
        reject_unsupported_fields(
            {"model": "gpt", "tools": []},
            allowed=frozenset({"model"}),
        )
    except PrpError as error:
        assert error.detail.code is ErrorCode.UNSUPPORTED_TOOLS
    else:
        raise AssertionError("tools must remain unsupported")
    try:
        reject_unsupported_fields(
            {"model": "gpt", "temperature_top": 1},
            allowed=frozenset({"model"}),
        )
    except PrpError as error:
        assert error.detail.code is ErrorCode.UNSUPPORTED_FIELD
    else:
        raise AssertionError("unknown fields must remain unsupported")


def test_offline_native_run_still_completes_without_live_provider(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "regression.db",
        worker_profile=WORKER_PROFILE,
        service_token=SecretStr("regression-secret"),
        service_principal="prn_operator",
    )
    app = create_app(settings, adapters={"worker": FakeAdapter()})
    with TestClient(app) as client:
        created = client.post("/v1/runs", json={"input": "hello"})
        assert created.status_code == 201
        run_id = created.json()["run_id"]
        events = client.get(f"/v1/runs/{run_id}/events")
        body = client.get(f"/v1/runs/{run_id}").json()
    assert events.status_code == 200
    assert body["status"] == RunStatus.SUCCEEDED.value
    assert "https://models.invalid" not in events.text
    assert "regression-secret" not in events.text


def test_live_external_regression_remains_gated_and_unimported() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "external_tests" / "test_regression.py"
    )
    text = source.read_text(encoding="utf-8")
    assert "@pytest.mark.live_regression" in text
    assert "live_profile_deepseek" in text
    assert "OpenAIResponsesProvider" in text
