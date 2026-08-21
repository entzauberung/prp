"""Bounded real DIRECT, CASCADE, and AUTO strategy checks."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from prp_runtime.app import create_app
from prp_runtime.domain.enums import AttemptStatus, ExecutionStrategy, ModelRole, RunStatus
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.providers.base import ModelProfile, ProviderAdapter, ProviderRequest, ProviderResponse
from prp_runtime.providers.factory import build_provider_adapter
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

from external_tests.capability_ledger import CapabilityStore
from external_tests.result_ledger import LedgerEntry, LedgerStore
from external_tests.support import ExternalConfig, ExternalGateError, ExternalProfile, validate_external_url
from external_tests.test_live_deepseek import _profile_for_runtime
from external_tests.test_live_protocols import _profile_by_alias

PROVIDER_RESULT_FILE = Path("/home/bruce/文档/prp测试日志/real-gap-closure/10-providers.jsonl")
CAPABILITY_RESULT_FILE = Path("/home/bruce/prp/ai/PROVIDER-CAPABILITIES.json")
ALIAS = "DEEPSEEK_FLASH_RESPONSES"
TERMINAL_RUN_STATUSES = {status.value for status in RunStatus if status.is_terminal}


class LocalFailureAdapter:
    """Inject one classified local failure without making an upstream call."""

    def __init__(self, code: ErrorCode) -> None:
        self.code = code
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "local-fault"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        raise ProviderError("bounded local strategy fault", code=self.code)

    async def aclose(self) -> None:
        return None


def _local_profile(alias: str) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        provider="local",
        model="local-fault",
        role=ModelRole.WORKER,
        base_url="https://local.invalid",
        context_window_tokens=8_192,
        max_output_tokens=128,
    )


def _result_path() -> Path:
    value = os.environ.get("PRP_LIVE_RESULT_FILE")
    if value is None or not value.strip():
        raise ExternalGateError("PRP_LIVE_RESULT_FILE is required for strategy evidence")
    return Path(value)


def _provider_passed() -> bool:
    return any(
        entry.alias == ALIAS and entry.status == "PASS"
        for entry in LedgerStore(PROVIDER_RESULT_FILE).read()
    )


def _structured_output_passed() -> bool:
    return any(
        entry.alias == ALIAS
        and entry.capability == "structured_output"
        and entry.status == "PASS"
        for entry in CapabilityStore(CAPABILITY_RESULT_FILE).read()
    )


def _planned_passed() -> bool:
    return any(
        entry.scenario_id == "wo-004-st-002-planned" and entry.status == "PASS"
        for entry in LedgerStore(_result_path()).read()
    )


def _prerequisite_entries(profile: ExternalProfile) -> tuple[LedgerEntry, ...]:
    return tuple(
        LedgerEntry(
            scenario_id=f"wo-004-st-001-{name}",
            alias=profile.alias,
            model_id=profile.model_id,
            protocol=f"{strategy}->PREREQUISITE",
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
        for name, strategy in (
            ("direct", "DIRECT"),
            ("cascade", "CASCADE"),
            ("auto", "AUTO"),
        )
    )


def _wait_for_terminal(client: TestClient, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body.get("status") in TERMINAL_RUN_STATUSES:
            return body
        time.sleep(0.02)
    raise AssertionError("strategy run did not reach a terminal state")


def _execute(client: TestClient, payload: dict[str, object]) -> dict[str, Any]:
    created = client.post("/v1/runs", json=payload)
    if created.status_code != 201:
        raise AssertionError(f"strategy run creation failed: {created.status_code}")
    return _wait_for_terminal(client, created.json()["run_id"])


async def _facts(database: Path, run_id: str) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    async with SqliteStore(database) as store:
        return (
            await store.get_run(run_id),
            await store.list_run_attempts(run_id),
            await store.list_events(run_id),
        )


def _event_types(events: tuple[Any, ...]) -> set[str]:
    return {event.event_type.value for event in events}


def _record(
    *,
    scenario: str,
    profile: ExternalProfile,
    run_id: str,
    attempt: Any,
    protocol: str,
) -> None:
    usage = attempt.usage
    LedgerStore(_result_path()).merge(
        [
            LedgerEntry(
                scenario_id=scenario,
                alias=profile.alias,
                model_id=profile.model_id,
                protocol=protocol,
                endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
                run_id=run_id,
                attempt_id=attempt.attempt_id,
                status="PASS",
                actual_or_simulated="ACTUAL",
                input_tokens=usage.input_tokens if usage is not None else None,
                output_tokens=usage.output_tokens if usage is not None else None,
                known_cost="unknown",
                latency_ms=usage.elapsed_ms if usage is not None else None,
                error_code=None,
                output_sha256=None,
                recorded_at=(
                    attempt.completed_at.isoformat()
                    if attempt.completed_at is not None
                    else "completed"
                ),
            )
        ],
        secret_values=[profile.api_key.get_secret_value()],
    )


def _record_planned(
    *,
    profile: ExternalProfile,
    run: Any,
    attempts: tuple[Any, ...],
    status: str,
    error_code: str | None,
) -> None:
    usage = run.usage
    first_attempt = attempts[0] if attempts else None
    LedgerStore(_result_path()).merge(
        [
            LedgerEntry(
                scenario_id="wo-004-st-002-planned",
                alias=profile.alias,
                model_id=profile.model_id,
                protocol="PLANNED->GRAPH_V2",
                endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
                run_id=run.run_id,
                attempt_id=(
                    first_attempt.attempt_id if first_attempt is not None else "not-recorded"
                ),
                status=status,
                actual_or_simulated="ACTUAL",
                input_tokens=usage.input_tokens if usage is not None else None,
                output_tokens=usage.output_tokens if usage is not None else None,
                known_cost="unknown",
                latency_ms=usage.elapsed_ms if usage is not None else None,
                error_code=error_code,
                output_sha256=None,
                recorded_at=(
                    run.completed_at.isoformat()
                    if run.completed_at is not None
                    else "completed"
                ),
            )
        ],
        secret_values=[profile.api_key.get_secret_value()],
    )


def _planned_failure_classification(run: Any, attempts: tuple[Any, ...]) -> tuple[str, str | None]:
    error = run.error
    if error is None and attempts:
        error = attempts[-1].error
    category = error.category.value if error is not None else None
    if category == "AUTH":
        return "UPSTREAM_AUTH_OR_PERMISSION", category
    if category in {"NETWORK", "TIMEOUT", "RATE_LIMIT"}:
        return "UPSTREAM_TRANSIENT", category
    if category == "PROVIDER_ERROR":
        return "UPSTREAM_UNSUPPORTED", category
    return "PRODUCT_DEFECT", category


@pytest.mark.live_strategy
def test_routing_real_direct_cascade_auto(
    external_config: ExternalConfig,
    temporary_resources: Any,
) -> None:
    profile = _profile_by_alias(external_config, ALIAS)
    if not _provider_passed():
        LedgerStore(_result_path()).merge(_prerequisite_entries(profile))
        pytest.skip("real strategy checks require a prior Responses provider PASS")
    validate_external_url(profile.base_url, external_config.allowed_hosts)
    runtime_profile = _profile_for_runtime(profile)

    direct_app = create_app(
        Settings(
            database_path=temporary_resources.database_path,
            worker_profile=runtime_profile,
        )
    )
    with TestClient(direct_app) as client:
        direct = _execute(
            client,
            {
                "input": "Reply with one short word: direct.",
                "routing_policy": "MANUAL",
                "strategy": "DIRECT",
            },
        )
        auto = _execute(client, {"input": "Reply with one short word: auto."})

    direct_run, direct_attempts, direct_events = asyncio.run(
        _facts(temporary_resources.database_path, direct["run_id"])
    )
    auto_run, auto_attempts, auto_events = asyncio.run(
        _facts(temporary_resources.database_path, auto["run_id"])
    )
    for run, attempts, events, expected, name in (
        (direct_run, direct_attempts, direct_events, ExecutionStrategy.DIRECT, "direct"),
        (auto_run, auto_attempts, auto_events, ExecutionStrategy.DIRECT, "auto"),
    ):
        assert run.status is RunStatus.SUCCEEDED
        assert run.strategy is expected
        assert len(attempts) == 1
        assert attempts[0].status is AttemptStatus.SUCCEEDED
        assert attempts[0].model == runtime_profile.model_ref
        assert "STRATEGY_SELECTED" in _event_types(events)
        _record(
            scenario=f"wo-004-st-001-{name}",
            profile=profile,
            run_id=run.run_id,
            attempt=attempts[0],
            protocol=f"{name.upper()}->{expected.value}",
        )

    local_retryable = LocalFailureAdapter(ErrorCode.PROVIDER_UNAVAILABLE)
    fallback_adapter: ProviderAdapter = build_provider_adapter(runtime_profile)
    local_profile = _local_profile("local-retryable")
    cascade_app = create_app(
        Settings(
            database_path=temporary_resources.database_path,
            worker_profile=local_profile,
            cascade_profiles=(runtime_profile,),
        ),
        adapters={local_profile.alias: local_retryable, runtime_profile.alias: fallback_adapter},
    )
    with TestClient(cascade_app) as client:
        try:
            cascade = _execute(
                client,
                {
                    "input": "Reply with one short word: fallback.",
                    "routing_policy": "MANUAL",
                    "strategy": "CASCADE",
                },
            )
        finally:
            assert client.portal is not None
            client.portal.call(fallback_adapter.aclose)

    cascade_run, cascade_attempts, cascade_events = asyncio.run(
        _facts(temporary_resources.database_path, cascade["run_id"])
    )
    assert cascade_run.status is RunStatus.SUCCEEDED
    assert cascade_run.strategy is ExecutionStrategy.CASCADE
    assert len(cascade_attempts) == 2
    assert cascade_attempts[0].status is AttemptStatus.FAILED
    assert cascade_attempts[0].error is not None
    assert cascade_attempts[0].error.category.value == "NETWORK"
    assert cascade_attempts[1].status is AttemptStatus.SUCCEEDED
    assert cascade_attempts[1].model == runtime_profile.model_ref
    assert local_retryable.requests and len(local_retryable.requests) == 1
    assert "STRATEGY_ESCALATED" in _event_types(cascade_events)
    _record(
        scenario="wo-004-st-001-cascade",
        profile=profile,
        run_id=cascade_run.run_id,
        attempt=cascade_attempts[1],
        protocol="CASCADE->LOCAL_RETRYABLE->REAL_FALLBACK",
    )


@pytest.mark.live_strategy
def test_routing_cascade_auth_failure_stops_without_fallback(
    external_config: ExternalConfig,
    temporary_resources: Any,
) -> None:
    profile = _profile_by_alias(external_config, ALIAS)
    if not _provider_passed():
        pytest.skip("auth stop check requires a prior Responses provider PASS")
    runtime_profile = _profile_for_runtime(profile)
    local_auth = LocalFailureAdapter(ErrorCode.PROVIDER_AUTH_FAILED)
    fallback_adapter: ProviderAdapter = build_provider_adapter(runtime_profile)
    local_profile = _local_profile("local-auth")
    app = create_app(
        Settings(
            database_path=temporary_resources.database_path.with_name("auth-stop.sqlite3"),
            worker_profile=local_profile,
            cascade_profiles=(runtime_profile,),
        ),
        adapters={local_profile.alias: local_auth, runtime_profile.alias: fallback_adapter},
    )
    with TestClient(app) as client:
        try:
            final = _execute(
                client,
                {
                    "input": "auth stop check",
                    "routing_policy": "MANUAL",
                    "strategy": "CASCADE",
                },
            )
        finally:
            assert client.portal is not None
            client.portal.call(fallback_adapter.aclose)

    assert final["status"] == RunStatus.FAILED.value
    attempts = asyncio.run(
        _facts(temporary_resources.database_path.with_name("auth-stop.sqlite3"), final["run_id"])
    )[1]
    assert len(attempts) == 1
    assert attempts[0].status is AttemptStatus.FAILED
    assert attempts[0].error is not None
    assert attempts[0].error.category.value == "AUTH"
    assert len(local_auth.requests) == 1


@pytest.mark.live_strategy
def test_planned_real_dag(
    external_config: ExternalConfig,
    temporary_resources: Any,
) -> None:
    profile = _profile_by_alias(external_config, ALIAS)
    if not _provider_passed() or not _structured_output_passed():
        LedgerStore(_result_path()).merge(
            [
                LedgerEntry(
                    scenario_id="wo-004-st-002-planned",
                    alias=profile.alias,
                    model_id=profile.model_id,
                    protocol="PLANNED->GRAPH_V2",
                    endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
                    run_id="not-run",
                    attempt_id="not-run",
                    status="NOT_APPLICABLE",
                    actual_or_simulated="SIMULATED",
                    input_tokens=None,
                    output_tokens=None,
                    known_cost="unknown",
                    latency_ms=None,
                    error_code="PREREQUISITE_NO_STRUCTURED_OUTPUT_PASS",
                    output_sha256=None,
                    recorded_at="not-run",
                )
            ]
        )
        pytest.skip("real planned check requires a structured-output Provider PASS")
    validate_external_url(profile.base_url, external_config.allowed_hosts)
    runtime_profile = _profile_for_runtime(profile)
    planner_profile = runtime_profile.model_copy(
        update={
            "alias": "planned-planner",
            "role": ModelRole.PLANNER,
            "supports_structured_output": True,
            "max_output_tokens": 256,
        }
    )
    worker_profile = runtime_profile.model_copy(
        update={
            "alias": "planned-worker",
            "role": ModelRole.WORKER,
            "supports_structured_output": True,
            "max_output_tokens": 256,
        }
    )
    database = temporary_resources.database_path.with_name("planned-real.sqlite3")
    app = create_app(
        Settings(
            database_path=database,
            leader_profile=planner_profile,
            worker_profile=worker_profile,
        )
    )
    with TestClient(app) as client:
        final = _execute(
            client,
            {
                "input": (
                    "Create exactly two bounded steps for this request. First identify "
                    "the key fact needed to answer. Then use that fact to produce the "
                    "final short answer. The second step must depend on the first."
                ),
                "routing_policy": "MANUAL",
                "strategy": "PLANNED",
                "budget": {"max_attempts": 3, "max_concurrency": 1},
            },
        )

    run, attempts, events = asyncio.run(_facts(database, final["run_id"]))
    if run.status is not RunStatus.SUCCEEDED:
        classification, error_code = _planned_failure_classification(run, attempts)
        _record_planned(
            profile=profile,
            run=run,
            attempts=attempts,
            status=classification,
            error_code=error_code,
        )
        assert run.status is RunStatus.FAILED
        return

    units = asyncio.run(_planned_units(database, run.run_id))
    assert run.strategy is ExecutionStrategy.PLANNED
    assert run.graph_version >= 2
    assert run.final_work_unit_id is not None
    assert len(units) == 2
    assert run.final_work_unit_id in {unit.work_unit_id for unit in units}
    assert len(attempts) == 3
    assert attempts[0].role is ModelRole.PLANNER
    assert attempts[0].status is AttemptStatus.SUCCEEDED
    assert all(attempt.role is ModelRole.WORKER for attempt in attempts[1:])
    assert all(attempt.status is AttemptStatus.SUCCEEDED for attempt in attempts)
    assert {event.event_type.value for event in events} >= {
        "PLAN_PROPOSED",
        "PLAN_COMMITTED",
        "RUN_SUCCEEDED",
    }
    _record_planned(
        profile=profile,
        run=run,
        attempts=attempts,
        status="PASS",
        error_code=None,
    )


async def _planned_units(database: Path, run_id: str) -> tuple[Any, ...]:
    async with SqliteStore(database) as store:
        return await store.list_work_units(run_id, graph_version=2)


@pytest.mark.live_strategy
def test_progressive_real_revision(
    external_config: ExternalConfig,
    temporary_resources: Any,
) -> None:
    profile = _profile_by_alias(external_config, ALIAS)
    if not _planned_passed():
        LedgerStore(_result_path()).merge(
            [
                LedgerEntry(
                    scenario_id="wo-004-st-003-progressive",
                    alias=profile.alias,
                    model_id=profile.model_id,
                    protocol="PROGRESSIVE->REVISION",
                    endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
                    run_id="not-run",
                    attempt_id="not-run",
                    status="NOT_APPLICABLE",
                    actual_or_simulated="SIMULATED",
                    input_tokens=None,
                    output_tokens=None,
                    known_cost="unknown",
                    latency_ms=None,
                    error_code="PREREQUISITE_NO_REAL_PLANNER_DAG",
                    output_sha256=None,
                    recorded_at="not-run",
                )
            ]
        )
        pytest.skip("real Progressive revision requires a verified real Planner DAG")

    validate_external_url(profile.base_url, external_config.allowed_hosts)
    raise AssertionError(
        "Progressive live execution requires a verified real Planner DAG and is not "
        "implemented by a fake revision path"
    )
