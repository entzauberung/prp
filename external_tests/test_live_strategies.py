"""Bounded real DIRECT, CASCADE, and AUTO strategy checks."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from external_tests.capability_ledger import CapabilityStore
from external_tests.result_ledger import LedgerEntry, LedgerStore
from external_tests.support import (
    ExternalConfig,
    ExternalGateError,
    ExternalProfile,
    validate_external_url,
)
from external_tests.test_live_deepseek import _profile_for_runtime
from external_tests.test_live_protocols import _profile_by_alias
from prp_runtime.app import create_app
from prp_runtime.domain.enums import AttemptStatus, ExecutionStrategy, ModelRole, RunStatus
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import Usage
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderAdapter,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.providers.factory import build_provider_adapter
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

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


_LOCAL_PROGRESSIVE_SCHEMA = (
    '{"type":"object","properties":{"ok":{"const":true}},'
    '"required":["ok"],"additionalProperties":false}'
)


class LocalProgressivePlannerAdapter:
    """Deterministic planner fixture for the production Progressive composition."""

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "local-progressive-planner"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            document: dict[str, object] = {
                "summary": "local progressive initial graph",
                "final_node": "initial",
                "nodes": [
                    {
                        "key": "initial",
                        "name": "Initial",
                        "instruction": "produce initial",
                        "output": {"kind": "JSON", "json_schema": _LOCAL_PROGRESSIVE_SCHEMA},
                    }
                ],
            }
        else:
            document = {
                "base_graph_version": 2,
                "reason": "VERIFICATION_FAILED",
                "summary": "local progressive corrected graph",
                "proposal": {
                    "summary": "local progressive corrected graph",
                    "final_node": "revised",
                    "nodes": [
                        {
                            "key": "revised",
                            "name": "Revised",
                            "instruction": "produce revised",
                            "output": {"kind": "JSON", "json_schema": _LOCAL_PROGRESSIVE_SCHEMA},
                        }
                    ],
                },
            }
        return ProviderResponse(
            text=json.dumps(document),
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )

    async def aclose(self) -> None:
        return None


class LocalProgressiveWorkerAdapter:
    """Return one verifier-failing artifact, then one passing artifact."""

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "local-progressive-worker"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        output = "{\"ok\":false}" if len(self.requests) == 1 else "{\"ok\":true}"
        return ProviderResponse(
            text=output,
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )

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


def _capability_path() -> Path:
    value = os.environ.get("PRP_LIVE_CAPABILITY_FILE")
    if value is None or not value.strip():
        raise ExternalGateError(
            "PRP_LIVE_CAPABILITY_FILE is required for strategy evidence"
        )
    return Path(value)


def _provider_passed() -> bool:
    provider_passed = any(
        entry.alias == ALIAS
        and entry.protocol == "OPENAI_RESPONSES"
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
        for entry in LedgerStore(_result_path()).read()
    )
    capability_passed = any(
        entry.alias == ALIAS
        and entry.protocol == "OPENAI_RESPONSES"
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
        for entry in CapabilityStore(_capability_path()).read()
    )
    return provider_passed or capability_passed


def _structured_output_passed() -> bool:
    return any(
        entry.alias == ALIAS
        and entry.protocol == "OPENAI_RESPONSES"
        and entry.capability == "structured_output"
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
        for entry in CapabilityStore(_capability_path()).read()
    )


def _planned_passed() -> bool:
    return any(
        entry.scenario_id.startswith("wo-004-st-002-planned")
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
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
    result_path = _result_path()
    scenario_id = scenario
    if any(entry.scenario_id == scenario for entry in LedgerStore(result_path).read()):
        scenario_id = f"{scenario}-retry-{attempt.attempt_id}"
    LedgerStore(_result_path()).merge(
        [
            LedgerEntry(
                scenario_id=scenario_id,
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
    scenario_id = "wo-004-st-002-planned"
    if any(entry.scenario_id == scenario_id for entry in LedgerStore(_result_path()).read()):
        scenario_id = f"{scenario_id}-retry-{run.run_id}"
    LedgerStore(_result_path()).merge(
        [
            LedgerEntry(
                scenario_id=scenario_id,
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


def _record_progressive(
    *,
    path: Path,
    run: Any,
    attempts: tuple[Any, ...],
    status: str,
    actual_or_simulated: str,
    error_code: str | None,
    endpoint_host: str,
) -> None:
    usage = run.usage
    first_attempt = attempts[0] if attempts else None
    scenario_id = "wo-003-st-003-progressive"
    if any(entry.scenario_id == scenario_id for entry in LedgerStore(path).read()):
        scenario_id = f"{scenario_id}-retry-{run.run_id}"
    LedgerStore(path).merge(
        [
            LedgerEntry(
                scenario_id=scenario_id,
                alias="LOCAL_PROGRESSIVE" if actual_or_simulated == "SIMULATED" else ALIAS,
                model_id=(
                    "local-progressive"
                    if actual_or_simulated == "SIMULATED"
                    else (attempts[0].model.model if attempts else "unknown")
                ),
                protocol="PROGRESSIVE->REVISION",
                endpoint_host=endpoint_host,
                run_id=run.run_id,
                attempt_id=(
                    first_attempt.attempt_id if first_attempt is not None else "not-recorded"
                ),
                status=status,
                actual_or_simulated=actual_or_simulated,
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
        ]
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


def _local_progressive_profile(alias: str, role: ModelRole) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        provider="local",
        model=f"{alias}-model",
        role=role,
        base_url="https://local.invalid",
        supports_structured_output=True,
        context_window_tokens=8_192,
        max_output_tokens=256,
    )


def test_progressive_local_revision(temporary_resources: Any) -> None:
    """Run the real production Progressive path with an explicitly simulated adapter."""
    database = temporary_resources.database_path.with_name("progressive-local.sqlite3")
    result_path = temporary_resources.database_path.with_name("progressive-local-results.jsonl")
    planner = LocalProgressivePlannerAdapter()
    worker = LocalProgressiveWorkerAdapter()
    planner_profile = _local_progressive_profile("local-progressive-planner", ModelRole.PLANNER)
    worker_profile = _local_progressive_profile("local-progressive-worker", ModelRole.WORKER)
    app = create_app(
        Settings(
            database_path=database,
            leader_profile=planner_profile,
            worker_profile=worker_profile,
        ),
        adapters={planner_profile.alias: planner, worker_profile.alias: worker},
    )

    with TestClient(app) as client:
        final = _execute(
            client,
            {
                "input": "produce a verified JSON result",
                "routing_policy": "MANUAL",
                "strategy": "PROGRESSIVE",
                "budget": {"max_plan_revisions": 1, "max_attempts": 4},
            },
        )

    async def inspect() -> tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        async with SqliteStore(database) as store:
            run = await store.get_run(final["run_id"])
            attempts = await store.list_run_attempts(run.run_id)
            rounds = await store.list_rounds(run.run_id)
            events = await store.list_events(run.run_id)
            return run, attempts, rounds, events

    run, attempts, rounds, events = asyncio.run(inspect())
    assert run.status is RunStatus.SUCCEEDED, run.error
    assert run.strategy is ExecutionStrategy.PROGRESSIVE
    assert run.graph_version == 3
    assert [round_fact.status.value for round_fact in rounds] == ["FAILED", "VERIFIED"]
    assert [round_fact.graph_version for round_fact in rounds] == [2, 3]
    assert not rounds[0].evidence_ids
    assert rounds[1].evidence_ids
    assert len(planner.requests) == 2
    assert len(worker.requests) == 2
    assert len(attempts) == 4
    assert EventType.PLAN_REVISED in {event.event_type for event in events}

    _record_progressive(
        path=result_path,
        run=run,
        attempts=attempts,
        status="PASS",
        actual_or_simulated="SIMULATED",
        error_code=None,
        endpoint_host="local.invalid",
    )
    entries = LedgerStore(result_path).read()
    assert len(entries) == 1
    assert entries[0].actual_or_simulated == "SIMULATED"
    assert entries[0].status == "PASS"


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
            "max_output_tokens": 1024,
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
    runtime_profile = _profile_for_runtime(profile)
    planner_profile = runtime_profile.model_copy(
        update={
            "alias": "progressive-planner",
            "role": ModelRole.PLANNER,
            "supports_structured_output": True,
            "max_output_tokens": 4096,
        }
    )
    worker_profile = runtime_profile.model_copy(
        update={
            "alias": "progressive-worker",
            "role": ModelRole.WORKER,
            "supports_structured_output": True,
            "max_output_tokens": 256,
        }
    )
    database = temporary_resources.database_path.with_name("progressive-real.sqlite3")
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
                    "Produce a short JSON answer. Use a bounded plan and verify the "
                    "final artifact before stopping."
                ),
                "routing_policy": "MANUAL",
                "strategy": "PROGRESSIVE",
                "budget": {"max_plan_revisions": 1, "max_attempts": 6},
            },
        )

    run, attempts, events = asyncio.run(_facts(database, final["run_id"]))
    if run.status is not RunStatus.SUCCEEDED:
        classification, error_code = _planned_failure_classification(run, attempts)
        _record_progressive(
            path=_result_path(),
            run=run,
            attempts=attempts,
            status=classification,
            actual_or_simulated="ACTUAL",
            error_code=error_code,
            endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
        )
        pytest.fail(f"real Progressive run failed: {classification}")

    async def inspect_rounds() -> tuple[Any, ...]:
        async with SqliteStore(database) as store:
            return await store.list_rounds(run.run_id)

    rounds = asyncio.run(inspect_rounds())
    assert run.strategy is ExecutionStrategy.PROGRESSIVE
    assert 1 <= len(rounds) <= 2
    assert all(round_fact.evidence_ids for round_fact in rounds)
    assert all(round_fact.graph_version >= 2 for round_fact in rounds)
    if len(rounds) == 2:
        assert rounds[1].graph_version > rounds[0].graph_version
        assert rounds[1].revision_of_round_id == rounds[0].round_id
    assert EventType.CONTROLLER_DECISION in {event.event_type for event in events}
    _record_progressive(
        path=_result_path(),
        run=run,
        attempts=attempts,
        status="PASS",
        actual_or_simulated="ACTUAL",
        error_code=None,
        endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
    )
