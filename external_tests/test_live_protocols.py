"""Real inbound protocol composition checks with prerequisite classification."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from external_tests.result_ledger import LedgerEntry, LedgerStore
from external_tests.support import (
    ExternalConfig,
    ExternalGateError,
    ExternalProfile,
    create_external_http_client,
    validate_external_url,
)
from external_tests.test_live_deepseek import _create_adapter, _profile_for_runtime
from prp_runtime.app import create_app
from prp_runtime.domain.enums import AttemptStatus, RunStatus
from prp_runtime.providers.base import ModelProfile
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

INGRESS_CASES = {
    "responses": {
        "path": "/v1/responses",
        "protocol": "OPENAI_RESPONSES",
    },
    "chat": {
        "path": "/v1/chat/completions",
        "protocol": "OPENAI_CHAT",
    },
    "messages": {
        "path": "/v1/messages",
        "protocol": "ANTHROPIC_MESSAGES",
    },
}
PROTOCOL_RESULT_FILE = Path(os.environ["PRP_LIVE_RESULT_FILE"])
TERMINAL_RUN_STATUSES = {status.value for status in RunStatus if status.is_terminal}


def _profile_by_alias(config: ExternalConfig, alias: str) -> ExternalProfile:
    """Retain the canonical exact-alias lookup used by lifecycle fixtures."""

    matches = [profile for profile in config.profiles if profile.alias == alias]
    if len(matches) != 1:
        raise ExternalGateError(f"protocol matrix must contain exactly one {alias}")
    return matches[0]


def _select_ingress_profile(
    config: ExternalConfig, protocol: str
) -> tuple[ExternalProfile, bool]:
    """Return the first configured candidate with actual outbound PASS evidence."""

    profiles = {profile.alias: profile for profile in config.profiles}
    raw_order = os.environ.get("PRP_EXTERNAL_CANDIDATE_ORDER", "")
    candidate_order = tuple(part.strip() for part in raw_order.split(",") if part.strip())
    if not candidate_order:
        candidate_order = tuple(
            profile.alias for profile in config.profiles if profile.protocol == protocol
        )
    candidates = tuple(
        profiles[alias]
        for alias in candidate_order
        if alias in profiles and profiles[alias].protocol == protocol
    )
    if not candidates:
        raise ExternalGateError(f"protocol matrix has no candidate for {protocol}")

    passed_aliases = {
        entry.alias
        for entry in LedgerStore(PROTOCOL_RESULT_FILE).read()
        if entry.protocol == protocol
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
    }
    for profile in candidates:
        if profile.alias in passed_aliases:
            return profile, True
    return candidates[0], False


def _scenario_id(case_name: str, profile: ExternalProfile, attempt_id: str) -> str:
    base = f"wo-009-st-001-ingress-{case_name}-{profile.alias.lower()}"
    existing_ids = {
        entry.scenario_id for entry in LedgerStore(PROTOCOL_RESULT_FILE).read()
    }
    if base not in existing_ids:
        return base
    return f"{base}-retry-{attempt_id}"


def _prerequisite_entry(case_name: str, profile: ExternalProfile, path: str) -> LedgerEntry:
    return LedgerEntry(
        scenario_id=_scenario_id(case_name, profile, "prerequisite"),
        alias=profile.alias,
        model_id=profile.model_id,
        protocol=f"{profile.protocol}->{path}",
        endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
        run_id="not-run",
        attempt_id="not-run",
        status="NOT_APPLICABLE",
        actual_or_simulated="SIMULATED",
        input_tokens=None,
        output_tokens=None,
        known_cost="unknown",
        latency_ms=None,
        error_code="PREREQUISITE_NO_PROVIDER_PASS",
        output_sha256=None,
        recorded_at="not-run",
    )


def _wait_for_terminal(client: TestClient, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        body = client.get(f"/v1/responses/{run_id}").json()
        if body.get("status") in TERMINAL_RUN_STATUSES | {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.02)
    raise AssertionError("protocol run did not reach a terminal state")


def _persisted_facts(database_path: Path, run_id: str) -> tuple[Any, Any, Any, Any, Any]:
    async def inspect() -> tuple[Any, Any, Any, Any, Any]:
        async with SqliteStore(database_path) as store:
            units = await store.list_work_units(run_id)
            if len(units) != 1:
                raise AssertionError("protocol smoke must persist one work unit")
            attempts = await store.list_run_attempts(run_id)
            artifacts = await store.list_artifacts(units[0].work_unit_id)
            evidence = await store.list_evidence(units[0].work_unit_id)
            events = await store.list_events(run_id)
            return units[0], attempts, artifacts, evidence, events

    return asyncio.run(inspect())


def _live_entry(
    case_name: str,
    path: str,
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
        scenario_id=_scenario_id(case_name, profile, attempt.attempt_id),
        alias=profile.alias,
        model_id=profile.model_id,
        protocol=f"{profile.protocol}->{path}",
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


def _request_for(case_name: str) -> dict[str, object]:
    if case_name == "responses":
        return {"input": "Reply with one short word: ready."}
    if case_name == "chat":
        return {"messages": [{"role": "user", "content": "Reply with one short word: ready."}]}
    return {"messages": [{"role": "user", "content": "Reply with one short word: ready."}]}


@pytest.mark.live_protocols
@pytest.mark.parametrize("case_name", tuple(INGRESS_CASES))
def test_ingress_protocol_composition(
    case_name: str,
    external_config: ExternalConfig,
    temporary_resources: Any,
) -> None:
    case = INGRESS_CASES[case_name]
    protocol = str(case["protocol"])
    profile, provider_passed = _select_ingress_profile(external_config, protocol)
    path = str(case["path"])
    if not provider_passed:
        LedgerStore(PROTOCOL_RESULT_FILE).merge([_prerequisite_entry(case_name, profile, path)])
        return

    validate_external_url(profile.base_url, external_config.allowed_hosts)
    runtime_profile: ModelProfile = _profile_for_runtime(profile)
    external_client = create_external_http_client(external_config.allowed_hosts)
    adapter = _create_adapter(runtime_profile, external_client)
    run_id: str | None = None
    try:
        settings = Settings(
            database_path=temporary_resources.database_path,
            worker_profile=runtime_profile,
        )
        app = create_app(settings, adapters={runtime_profile.alias: adapter})
        with TestClient(app) as client:
            created = client.post(path, json=_request_for(case_name))
            expected_status = 202 if case_name == "responses" else 200
            if created.status_code != expected_status:
                raise AssertionError(f"{path} returned unexpected status {created.status_code}")
            body = created.json()
            run_id = body["id"]
            final = (
                _wait_for_terminal(client, run_id)
                if case_name == "responses"
                else body
            )

        _, attempts, artifacts, evidence, events = _persisted_facts(
            temporary_resources.database_path, run_id
        )
        if len(attempts) != 1:
            raise AssertionError("protocol smoke must persist one provider attempt")
        attempt = attempts[0]
        artifact = artifacts[0] if len(artifacts) == 1 else None
        envelope_ok = (
            final.get("id") == run_id
            and final.get("status") in {"completed", "in_progress"}
            and (
                final.get("object") == "response"
                or final.get("object") == "chat.completion"
                or final.get("type") == "message"
            )
        )
        succeeded = (
            envelope_ok
            and final.get("status") == "completed"
            and attempt.status is AttemptStatus.SUCCEEDED
            and attempt.model.model == profile.model_id
            and artifact is not None
            and bool(artifact.content.strip())
            and bool(evidence)
            and bool(events)
        )
        error_code = None if succeeded else "PROTOCOL_COMPOSITION_FAILED"
        entry = _live_entry(
            case_name,
            path,
            profile,
            run_id,
            attempt,
            artifact,
            "PASS" if succeeded else "PRODUCT_DEFECT",
            error_code,
        )
        LedgerStore(PROTOCOL_RESULT_FILE).merge(
            [entry], secret_values=[profile.api_key.get_secret_value()]
        )
        if not succeeded:
            raise AssertionError(f"{case_name} ingress did not produce a persisted success")
    finally:
        try:
            asyncio.run(external_client.aclose())
        except Exception:
            pass
