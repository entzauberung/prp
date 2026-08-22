"""Low-volume Zhipu text smoke through the production ASGI composition.

This module is collected only by the explicit external-test command. Every
provider request is bounded before dispatch, and the persisted runtime records
are the source of the redacted ledger entry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from external_tests.result_ledger import LedgerEntry, LedgerStore
from external_tests.support import (
    BudgetCounter,
    ExternalConfig,
    ExternalGateError,
    ExternalProfile,
    create_external_http_client,
    validate_external_url,
)
from prp_runtime.app import create_app
from prp_runtime.domain.enums import AttemptStatus, ModelRole, RunStatus
from prp_runtime.providers.base import ModelProfile, ProviderProtocol
from prp_runtime.providers.openai_compatible import OpenAICompatibleProvider
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

ZHIPU_ALIASES = (
    "ZHIPU_GLM_46V",
    "ZHIPU_GLM_45_AIR",
    "ZHIPU_GLM_47_FLASH",
)
OUT_OF_SCOPE_MODELS = frozenset(
    {"GLM-4V-Flash", "CogView-3-Flash", "CogVideoX-Flash"}
)
MAX_OUTPUT_TOKENS = 128
TERMINAL_RUN_STATUSES = {status.value for status in RunStatus if status.is_terminal}


@pytest.fixture(scope="module")
def zhipu_budget() -> BudgetCounter:
    """Reserve every attempt before it can reach a real provider."""

    raw_limit = os.environ.get("PRP_LIVE_MAX_CALLS", "4")
    try:
        max_provider_calls = int(raw_limit)
    except ValueError as exc:
        raise ExternalGateError("PRP_LIVE_MAX_CALLS must be an integer") from exc
    return BudgetCounter(
        max_provider_calls=max_provider_calls,
        max_attempts_per_alias=1,
        max_successful_calls_per_alias=1,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def _profile_for_runtime(profile: ExternalProfile) -> ModelProfile:
    """Translate the external matrix record into a server-owned profile."""

    if profile.protocol not in ("OPENAI_COMPATIBLE_CHAT", "OPENAI_RESPONSES"):
        raise ExternalGateError(
            f"unexpected Zhipu outbound protocol for {profile.alias}"
        )
    return ModelProfile(
        alias=profile.alias.lower(),
        provider=profile.vendor.lower(),
        model=profile.model_id,
        role=ModelRole.WORKER,
        base_url=profile.base_url,
        api_key=profile.api_key,
        protocol=ProviderProtocol.OPENAI_CHAT,
        context_window_tokens=8_192,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        timeout_seconds=60.0,
    )


def _profile_by_alias(config: ExternalConfig, alias: str) -> ExternalProfile:
    matches = [profile for profile in config.profiles if profile.alias == alias]
    if len(matches) != 1:
        raise ExternalGateError(f"Zhipu model matrix must contain exactly one {alias}")
    return matches[0]


def _wait_for_terminal(client: TestClient, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body.get("status") in TERMINAL_RUN_STATUSES:
            return body
        time.sleep(0.02)
    raise AssertionError("runtime run did not reach a terminal state")


def _persisted_facts(
    database_path: Path, run_id: str
) -> tuple[object, object, object, object]:
    async def inspect() -> tuple[object, object, object, object]:
        async with SqliteStore(database_path) as store:
            units = await store.list_work_units(run_id)
            if len(units) != 1:
                raise AssertionError("live smoke must persist exactly one work unit")
            attempts = await store.list_run_attempts(run_id)
            artifacts = await store.list_artifacts(units[0].work_unit_id)
            evidence = await store.list_evidence(units[0].work_unit_id)
            return units[0], attempts, artifacts, evidence

    return asyncio.run(inspect())


def _result_path() -> Path:
    value = os.environ.get("PRP_LIVE_RESULT_FILE")
    if value is None or not value.strip():
        raise ExternalGateError("PRP_LIVE_RESULT_FILE is required for live evidence")
    return Path(value)


def _record_entry(
    *,
    profile: ExternalProfile,
    run_id: str,
    attempt: Any,
    artifact: Any | None,
    status: str,
    error_code: str | None,
) -> LedgerEntry:
    usage = attempt.usage
    content = artifact.content if artifact is not None else None
    return LedgerEntry(
        scenario_id=f"wo-006-st-001-{profile.alias.lower()}",
        alias=profile.alias,
        model_id=profile.model_id,
        protocol=profile.protocol,
        endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
        run_id=run_id,
        attempt_id=attempt.attempt_id,
        status=status,
        actual_or_simulated="ACTUAL",
        input_tokens=usage.input_tokens if usage is not None else None,
        output_tokens=usage.output_tokens if usage is not None else None,
        known_cost="unknown",
        latency_ms=usage.elapsed_ms if usage is not None else None,
        error_code=error_code,
        output_sha256=(
            hashlib.sha256(content.encode("utf-8")).hexdigest()
            if isinstance(content, str)
            else None
        ),
        recorded_at=(
            attempt.completed_at.isoformat()
            if attempt.completed_at is not None
            else "unknown"
        ),
    )


@pytest.mark.live_provider
@pytest.mark.parametrize("alias", ZHIPU_ALIASES)
def test_zhipu_text_smoke_persists_redacted_evidence(
    alias: str,
    external_config: ExternalConfig,
    temporary_resources: Any,
    zhipu_budget: BudgetCounter,
) -> None:
    profile = _profile_by_alias(external_config, alias)
    if profile.model_id in OUT_OF_SCOPE_MODELS:
        raise AssertionError("multimodal Zhipu models must not enter text smoke")
    validate_external_url(profile.base_url, external_config.allowed_hosts)
    runtime_profile = _profile_for_runtime(profile)
    reservation = zhipu_budget.reserve(alias, MAX_OUTPUT_TOKENS)
    external_client = create_external_http_client(external_config.allowed_hosts)
    adapter = OpenAICompatibleProvider(runtime_profile, client=external_client)
    run_id: str | None = None
    final: dict[str, Any] | None = None
    attempt: Any | None = None
    artifact: Any | None = None
    try:
        settings = Settings(
            database_path=temporary_resources.database_path,
            worker_profile=runtime_profile,
        )
        app = create_app(settings, adapters={runtime_profile.alias: adapter})
        with TestClient(app) as client:
            created = client.post(
                "/v1/runs",
                json={"input": "Reply with one short word: ready."},
            )
            if created.status_code != 201:
                raise AssertionError("production runtime rejected the live smoke request")
            run_id = created.json()["run_id"]
            final = _wait_for_terminal(client, run_id)

        _, attempts, artifacts, evidence = _persisted_facts(
            temporary_resources.database_path, run_id
        )
        if len(attempts) != 1:
            raise AssertionError("live smoke must persist one provider attempt")
        attempt = attempts[0]
        artifact = artifacts[0] if len(artifacts) == 1 else None
        succeeded = (
            final.get("status") == RunStatus.SUCCEEDED.value
            and attempt.status is AttemptStatus.SUCCEEDED
            and attempt.model.provider == profile.vendor
            and attempt.model.model == profile.model_id
            and artifact is not None
            and bool(evidence)
            and bool(artifact.content.strip())
        )
        error_code = None
        if not succeeded:
            final_error = final.get("error") if final is not None else None
            if isinstance(final_error, dict) and isinstance(final_error.get("code"), str):
                error_code = final_error["code"]
            elif attempt.error is not None:
                error_code = attempt.error.category.value
        entry = _record_entry(
            profile=profile,
            run_id=run_id,
            attempt=attempt,
            artifact=artifact,
            status="PASS" if succeeded else "BLOCKED",
            error_code=error_code,
        )
        LedgerStore(_result_path()).merge(
            [entry], secret_values=[profile.api_key.get_secret_value()]
        )
        zhipu_budget.settle(
            reservation,
            success=succeeded,
            observed_output_tokens=(
                attempt.usage.output_tokens if attempt.usage is not None else None
            ),
        )
        if not succeeded:
            raise AssertionError(f"{alias} live smoke did not produce a persisted success")
    finally:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(external_client.aclose())
            else:
                loop.run_until_complete(external_client.aclose())
        except Exception:
            pass


@pytest.mark.live_provider
def test_zhipu_multimodal_candidates_remain_out_of_scope(
    external_config: ExternalConfig,
) -> None:
    """The boundary is represented as scope metadata, never a provider call."""

    in_scope_models = {profile.model_id for profile in external_config.profiles}
    assert not in_scope_models.intersection(OUT_OF_SCOPE_MODELS)
    matrix_path = Path(__file__).with_name("model_matrix.example.json")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    out_of_scope = {
        item["model_id"]
        for item in matrix["out_of_scope"]
        if item["scope"] == "OUT_OF_SCOPE"
    }
    assert OUT_OF_SCOPE_MODELS <= out_of_scope
