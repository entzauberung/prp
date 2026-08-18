"""Small local relative benchmarks; timings are evidence, never an SLA."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import pytest
import pytest_asyncio

from prp_runtime.api.anthropic_messages import _normalize as normalize_anthropic
from prp_runtime.api.openai_chat import _normalize as normalize_chat
from prp_runtime.api.openai_responses import _normalize as normalize_responses
from prp_runtime.control.controller import RunController
from prp_runtime.domain.enums import ExecutionStrategy, ModelRole, RoutingPolicy, RunStatus
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import Budget, NativeRunRequest, Usage
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "protocol"


@dataclass(frozen=True)
class StrategySample:
    strategy: ExecutionStrategy
    wall_time_ns: int
    provider_calls: int
    attempts: int
    events: int
    revisions: int


class BenchmarkAdapter:
    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "local-benchmark-fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if request.alias == "planner":
            text = json.dumps(
                {
                    "summary": "one fixed benchmark node",
                    "final_node": "answer",
                    "nodes": [
                        {
                            "key": "answer",
                            "name": "Answer",
                            "instruction": "produce the answer",
                        }
                    ],
                }
            )
        else:
            text = "benchmark answer"
        return ProviderResponse(
            text=text,
            usage=Usage(input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
        )


def _profile(alias: str, role: ModelRole) -> ModelProfile:
    return ModelProfile(
        alias=alias,
        provider="fake",
        model=f"{alias}-model",
        role=role,
        base_url="https://models.invalid/v1",
        supports_structured_output=True,
        context_window_tokens=16_000,
        max_output_tokens=2_000,
    )


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "benchmark.db") as opened:
        yield opened


@pytest.mark.asyncio
async def test_four_strategy_relative_samples_have_stable_fact_shapes(
    store: SqliteStore,
) -> None:
    settings = Settings(
        leader_profile=_profile("planner", ModelRole.PLANNER),
        worker_profile=_profile("worker", ModelRole.WORKER),
    )
    samples: list[StrategySample] = []
    for strategy in ExecutionStrategy:
        adapter = BenchmarkAdapter()
        controller = RunController(
            store,
            settings,
            {"planner": adapter, "worker": adapter},  # type: ignore[arg-type]
        )
        budget = (
            Budget(max_plan_revisions=1)
            if strategy is ExecutionStrategy.PROGRESSIVE
            else Budget()
        )
        run = await controller.create_run(
            NativeRunRequest(
                input="fixed benchmark objective",
                routing_policy=RoutingPolicy.MANUAL,
                strategy=strategy,
                budget=budget,
            )
        )
        started = perf_counter_ns()
        finished = await controller.execute(run.run_id)
        elapsed = perf_counter_ns() - started
        events = await store.list_events(run.run_id)
        samples.append(
            StrategySample(
                strategy=strategy,
                wall_time_ns=elapsed,
                provider_calls=len(adapter.requests),
                attempts=len(await store.list_run_attempts(run.run_id)),
                events=len(events),
                revisions=sum(
                    event.event_type is EventType.PLAN_REVISED for event in events
                ),
            )
        )
        assert finished.status is RunStatus.SUCCEEDED, (strategy, finished.error)

    assert [sample.strategy for sample in samples] == list(ExecutionStrategy)
    assert all(sample.wall_time_ns >= 0 for sample in samples)
    assert all(sample.provider_calls >= 1 for sample in samples)
    assert all(sample.attempts >= 1 for sample in samples)
    assert [sample.provider_calls for sample in samples] == [1, 1, 2, 2]
    assert [sample.attempts for sample in samples] == [1, 1, 2, 2]
    assert all(sample.events >= 1 for sample in samples)
    assert all(sample.revisions >= 0 for sample in samples)


@pytest.mark.parametrize(
    ("fixture", "normalizer"),
    [
        ("openai_responses_text.json", normalize_responses),
        ("openai_chat_text.json", normalize_chat),
        ("anthropic_messages_text.json", normalize_anthropic),
    ],
)
def test_three_binding_samples_have_fixed_normalized_shapes(
    fixture: str,
    normalizer: object,
) -> None:
    payload = json.loads((FIXTURE_ROOT / fixture).read_text(encoding="utf-8"))
    started = perf_counter_ns()
    result = normalizer(payload)  # type: ignore[operator]
    elapsed = perf_counter_ns() - started

    assert elapsed >= 0
    assert result.request is not None
    assert result.operation.value == "CREATE"
    assert result.request.routing_policy.value == "AUTO"
