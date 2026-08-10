"""Single-model worker.

A worker executes exactly one attempt for one work unit: it calls the provider,
records the attempt outcome, the produced artifact and the measured usage. It
never decides whether a run is finished and never writes run or work unit state.
"""

from datetime import datetime

from pydantic import ValidationError

from prp_runtime.domain.enums import AttemptStatus, ModelRole
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.events import EventType
from prp_runtime.domain.models import (
    Artifact,
    Attempt,
    DomainModel,
    ErrorCategory,
    ErrorInfo,
    Run,
    Usage,
    WorkUnit,
    new_artifact_id,
)
from prp_runtime.domain.transitions import assert_can_start_attempt, transition_attempt
from prp_runtime.domain.values import new_attempt_id, utc_now
from prp_runtime.providers.base import ModelProfile, ProviderAdapter, ProviderRequest
from prp_runtime.runtime.context import ANSWER_ARTIFACT_NAME, WorkerContext
from prp_runtime.storage.sqlite import SqliteStore

__all__ = ["Worker", "WorkerResult"]

_CATEGORY_BY_CODE: dict[ErrorCode, ErrorCategory] = {
    ErrorCode.PROVIDER_TIMEOUT: ErrorCategory.TIMEOUT,
    ErrorCode.PROVIDER_RATE_LIMITED: ErrorCategory.RATE_LIMIT,
    ErrorCode.PROVIDER_AUTH_FAILED: ErrorCategory.AUTH,
    ErrorCode.PROVIDER_UNAVAILABLE: ErrorCategory.NETWORK,
    ErrorCode.PROVIDER_INVALID_RESPONSE: ErrorCategory.PROVIDER_ERROR,
    ErrorCode.PROVIDER_NOT_CONFIGURED: ErrorCategory.PROVIDER_ERROR,
}


def _category_for(code: ErrorCode) -> ErrorCategory:
    """Classify a provider failure for the attempt record."""
    return _CATEGORY_BY_CODE.get(code, ErrorCategory.UNKNOWN)


def _completed_at(started_at: datetime) -> datetime:
    """Now, never before the recorded start, so a clock step cannot invalidate it."""
    now = utc_now()
    return started_at if now < started_at else now


class WorkerResult(DomainModel):
    """What one attempt produced."""

    attempt: Attempt
    artifact: Artifact | None = None
    error: ErrorInfo | None = None

    @property
    def succeeded(self) -> bool:
        return self.attempt.status is AttemptStatus.SUCCEEDED and self.artifact is not None


class Worker:
    """Runs one work unit against one configured model."""

    def __init__(
        self, store: SqliteStore, adapter: ProviderAdapter, profile: ModelProfile
    ) -> None:
        self._store = store
        self._adapter = adapter
        self._profile = profile

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    async def execute(
        self,
        *,
        run: Run,
        work_unit: WorkUnit,
        context: WorkerContext,
        attempt_index: int = 1,
        role: ModelRole = ModelRole.WORKER,
    ) -> WorkerResult:
        """Perform one attempt and persist its facts.

        Raises ``AttemptNotAllowedError`` when the run or the work unit forbids a
        new attempt, so a cancelled run can never produce another provider call.
        """
        assert_can_start_attempt(run.status, work_unit.status)
        request = ProviderRequest.for_profile(
            self._profile,
            input=context.render_input(),
            instructions=context.render_instructions(),
            json_schema=context.output.json_schema,
        )
        attempt = await self._start_attempt(run, work_unit, attempt_index, role)
        try:
            response = await self._adapter.complete(request)
        except ProviderError as error:
            return await self._record_failure(
                attempt, ErrorInfo(category=_category_for(error.code), message=str(error))
            )
        except BaseException:
            # Cancellation or a hard interruption: the upstream outcome cannot be
            # proven, so the attempt becomes UNKNOWN and the error is re-raised.
            await self._record_unconfirmed(attempt)
            raise

        try:
            artifact = Artifact(
                artifact_id=new_artifact_id(),
                run_id=run.run_id,
                work_unit_id=work_unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                name=ANSWER_ARTIFACT_NAME,
                kind=context.output.kind,
                content=response.text,
            )
        except ValidationError:
            return await self._record_failure(
                attempt,
                ErrorInfo(
                    category=ErrorCategory.PROVIDER_ERROR,
                    message=(
                        f"upstream {self._profile.alias} returned no usable "
                        f"{context.output.kind.value} result"
                    ),
                ),
            )
        return await self._record_success(
            attempt, artifact, response.usage, response.provider_request_id
        )

    async def _start_attempt(
        self, run: Run, work_unit: WorkUnit, attempt_index: int, role: ModelRole
    ) -> Attempt:
        started_at = utc_now()
        attempt = Attempt(
            attempt_id=new_attempt_id(),
            run_id=run.run_id,
            work_unit_id=work_unit.work_unit_id,
            attempt_index=attempt_index,
            role=role,
            model=self._profile.model_ref,
            status=transition_attempt(AttemptStatus.PENDING, AttemptStatus.RUNNING),
            created_at=started_at,
            started_at=started_at,
        )
        async with self._store.transaction():
            await self._store.create_attempt(attempt)
            await self._store.append_event(
                attempt.run_id,
                EventType.ATTEMPT_STARTED,
                {
                    "work_unit_id": attempt.work_unit_id,
                    "attempt_id": attempt.attempt_id,
                    "model": self._profile.model_ref.identifier,
                    "role": role.value,
                    "attempt_index": attempt_index,
                },
            )
        return attempt

    async def _record_success(
        self,
        attempt: Attempt,
        artifact: Artifact,
        usage: Usage | None,
        provider_request_id: str | None,
    ) -> WorkerResult:
        completed = self._close_attempt(
            attempt,
            AttemptStatus.SUCCEEDED,
            usage=usage,
            provider_request_id=provider_request_id,
        )
        async with self._store.transaction():
            await self._store.update_attempt(completed)
            await self._store.add_artifact(artifact)
            await self._store.append_event(
                attempt.run_id,
                EventType.ATTEMPT_SUCCEEDED,
                {"work_unit_id": attempt.work_unit_id, "attempt_id": attempt.attempt_id},
            )
            await self._store.append_event(
                attempt.run_id,
                EventType.ARTIFACT_PRODUCED,
                {
                    "work_unit_id": artifact.work_unit_id,
                    "artifact_id": artifact.artifact_id,
                    "name": artifact.name,
                    "kind": artifact.kind.value,
                },
            )
            if usage is not None:
                total = await self._store.add_run_usage(attempt.run_id, usage)
                await self._store.append_event(
                    attempt.run_id,
                    EventType.USAGE_UPDATED,
                    {"usage": total.model_dump(mode="json")},
                )
        return WorkerResult(attempt=completed, artifact=artifact)

    async def _record_failure(self, attempt: Attempt, error: ErrorInfo) -> WorkerResult:
        failed = self._close_attempt(attempt, AttemptStatus.FAILED, error=error)
        async with self._store.transaction():
            await self._store.update_attempt(failed)
            await self._store.append_event(
                attempt.run_id,
                EventType.ATTEMPT_FAILED,
                {
                    "work_unit_id": attempt.work_unit_id,
                    "attempt_id": attempt.attempt_id,
                    "error": error.model_dump(mode="json"),
                },
            )
        return WorkerResult(attempt=failed, error=error)

    async def _record_unconfirmed(self, attempt: Attempt) -> None:
        unknown = self._close_attempt(attempt, AttemptStatus.UNKNOWN)
        async with self._store.transaction():
            await self._store.update_attempt(unknown)
            await self._store.append_event(
                attempt.run_id,
                EventType.ATTEMPT_UNKNOWN,
                {
                    "work_unit_id": attempt.work_unit_id,
                    "attempt_id": attempt.attempt_id,
                    "reason": "the upstream outcome could not be confirmed",
                },
            )

    @staticmethod
    def _close_attempt(
        attempt: Attempt,
        status: AttemptStatus,
        *,
        usage: Usage | None = None,
        error: ErrorInfo | None = None,
        provider_request_id: str | None = None,
    ) -> Attempt:
        """Re-validate the attempt into its terminal shape via the state machine."""
        changes: dict[str, object] = {
            "status": transition_attempt(attempt.status, status),
            "completed_at": _completed_at(attempt.started_at or attempt.created_at),
        }
        if usage is not None:
            changes["usage"] = usage
        if error is not None:
            changes["error"] = error
        if provider_request_id is not None:
            changes["provider_request_id"] = provider_request_id
        return Attempt.model_validate(attempt.model_dump() | changes)
