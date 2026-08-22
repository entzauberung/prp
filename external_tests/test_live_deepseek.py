"""Low-volume DeepSeek three-protocol smoke through production ASGI composition.

This module tests OPENAI_CHAT, OPENAI_RESPONSES, and ANTHROPIC_MESSAGES.
Every provider request is bounded before dispatch, and persisted runtime records
are the source of the redacted ledger entry.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
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
from prp_runtime.providers.anthropic import AnthropicMessagesProvider
from prp_runtime.providers.base import ModelProfile, ProviderProtocol
from prp_runtime.providers.openai_compatible import OpenAICompatibleProvider
from prp_runtime.providers.openai_responses import OpenAIResponsesProvider
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

DEEPSEEK_ALIASES = (
    "DEEPSEEK_FLASH_CHAT",
    "DEEPSEEK_FLASH_RESPONSES",
    "DEEPSEEK_FLASH_ANTHROPIC",
)
MAX_OUTPUT_TOKENS = 128
TERMINAL_RUN_STATUSES = {status.value for status in RunStatus if status.is_terminal}


@pytest.fixture(scope="module")
def deepseek_budget() -> BudgetCounter:
    """Reserve every attempt before it can reach a real provider."""
    raw_limit = os.environ.get("PRP_LIVE_MAX_CALLS", "6")
    try:
        max_provider_calls = int(raw_limit)
    except ValueError as exc:
        raise ExternalGateError("PRP_LIVE_MAX_CALLS must be an integer") from exc
    return BudgetCounter(
        max_provider_calls=max_provider_calls,
        max_attempts_per_alias=2,
        max_successful_calls_per_alias=1,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def _profile_for_runtime(profile: ExternalProfile) -> ModelProfile:
    """Translate the external matrix record into a server-owned profile."""
    protocol_map = {
        "OPENAI_CHAT": ProviderProtocol.OPENAI_CHAT,
        "OPENAI_RESPONSES": ProviderProtocol.OPENAI_RESPONSES,
        "ANTHROPIC_MESSAGES": ProviderProtocol.ANTHROPIC_MESSAGES,
    }
    runtime_protocol = protocol_map.get(profile.protocol)
    if runtime_protocol is None:
        raise ExternalGateError(
            f"unexpected DeepSeek protocol {profile.protocol} for {profile.alias}"
        )

    kwargs: dict[str, Any] = {
        "alias": profile.alias.lower(),
        "provider": profile.vendor.lower(),
        "model": profile.model_id,
        "role": ModelRole.WORKER,
        "base_url": profile.base_url,
        "api_key": profile.api_key,
        "protocol": runtime_protocol,
        "context_window_tokens": 8_192,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "timeout_seconds": 60.0,
    }
    if runtime_protocol == ProviderProtocol.ANTHROPIC_MESSAGES:
        kwargs["anthropic_version"] = "2023-06-01"
    return ModelProfile(**kwargs)


def _profile_by_alias(config: ExternalConfig, alias: str) -> ExternalProfile:
    matches = [profile for profile in config.profiles if profile.alias == alias]
    if len(matches) != 1:
        raise ExternalGateError(f"DeepSeek matrix must contain exactly one {alias}")
    return matches[0]


def _create_adapter(profile: ModelProfile, http_client: Any) -> Any:
    """Create the appropriate adapter based on protocol."""
    if profile.protocol == ProviderProtocol.OPENAI_CHAT:
        return OpenAICompatibleProvider(profile, client=http_client)
    if profile.protocol == ProviderProtocol.OPENAI_RESPONSES:
        return OpenAIResponsesProvider(profile, client=http_client)
    if profile.protocol == ProviderProtocol.ANTHROPIC_MESSAGES:
        return AnthropicMessagesProvider(profile, client=http_client)
    raise ExternalGateError(f"unsupported protocol {profile.protocol}")


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
        scenario_id=_scenario_id(profile, attempt.attempt_id),
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


def _scenario_id(profile: ExternalProfile, attempt_id: str) -> str:
    """Keep bounded retry evidence without conflicting with the first record."""
    base = f"wo-001-st-003-{profile.alias.lower()}"
    existing = LedgerStore(_result_path()).read()
    if any(entry.scenario_id == base for entry in existing):
        return f"{base}-retry-{attempt_id}"
    return base


def _failure_classification(final: dict[str, Any] | None, attempt: Any) -> str:
    """Classify the redacted runtime failure for the campaign ledger."""
    message = ""
    if final is not None:
        final_error = final.get("error")
        if isinstance(final_error, dict) and isinstance(final_error.get("message"), str):
            message = final_error["message"]
    if not message and attempt.error is not None:
        message = attempt.error.message
    normalized = message.lower()
    status_match = re.search(r"returned status (\d{3})", normalized)
    if status_match:
        status = int(status_match.group(1))
        if status in (401, 403):
            return "UPSTREAM_AUTH_OR_PERMISSION"
        if status == 429 or status in (408, 504) or status >= 500:
            return "UPSTREAM_TRANSIENT"
        if 400 <= status < 500:
            return "UPSTREAM_UNSUPPORTED"
    if "timed out" in normalized or "unreachable" in normalized:
        return "UPSTREAM_TRANSIENT"
    if "unusable response" in normalized:
        return "UPSTREAM_UNSUPPORTED"
    if "profile" in normalized or "contract" in normalized:
        return "PRODUCT_DEFECT"
    return attempt.error.category.value if attempt.error is not None else "UNKNOWN"


@pytest.mark.live_provider
@pytest.mark.parametrize("alias", DEEPSEEK_ALIASES)
def test_deepseek_multi_protocol_smoke_persists_redacted_evidence(
    alias: str,
    external_config: ExternalConfig,
    temporary_resources: Any,
    deepseek_budget: BudgetCounter,
) -> None:
    profile = _profile_by_alias(external_config, alias)
    validate_external_url(profile.base_url, external_config.allowed_hosts)
    runtime_profile = _profile_for_runtime(profile)
    reservation = deepseek_budget.reserve(alias, MAX_OUTPUT_TOKENS)
    external_client = create_external_http_client(external_config.allowed_hosts)
    adapter = _create_adapter(runtime_profile, external_client)
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
            error_code = _failure_classification(final, attempt)
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
        deepseek_budget.settle(
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
