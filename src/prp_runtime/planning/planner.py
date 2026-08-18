"""Stateless Planner provider call and strict proposal parsing."""

import json
from typing import Literal

from pydantic import ValidationError, model_validator

from prp_runtime.domain.enums import ModelRole, WorkUnitStatus
from prp_runtime.domain.errors import (
    DomainValidationError,
    ErrorCode,
    ProviderError,
)
from prp_runtime.domain.models import (
    DomainModel,
    ErrorCategory,
    ErrorInfo,
    NativeRunRequest,
    OutputRequirement,
    Usage,
    WorkUnit,
)
from prp_runtime.domain.values import ModelRef, RunId, new_work_unit_id
from prp_runtime.json_support import StrictJsonError
from prp_runtime.planning.models import (
    MAX_PLAN_NODES,
    PlanProposal,
    PlanRejection,
    PlanRevision,
    PlanRevisionReason,
)
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderAdapter,
    ProviderRequest,
    ProviderResponse,
)

__all__ = [
    "PLANNING_GRAPH_VERSION",
    "PLANNING_WORK_UNIT_NAME",
    "Planner",
    "PlannerCallResult",
    "new_planning_work_unit",
]

PLANNING_GRAPH_VERSION = 1
PLANNING_WORK_UNIT_NAME = "internal-planner-call"

_PLANNER_INSTRUCTIONS = (
    "Propose a bounded work graph for the objective. Return only strict JSON matching "
    "the supplied schema. Include public summaries, dependencies, resources, output "
    "requirements, acceptance criteria, and exactly one final_node identifying the "
    "user-facing output node. Do not include reasoning, chain-of-thought, code, "
    "commands, provider configuration, credentials, or persistent IDs."
)


class _PlannerInput(DomainModel):
    """The only Native request fields sent to the Planner."""

    objective: str
    user_instructions: str | None = None
    output: OutputRequirement
    max_nodes: int = MAX_PLAN_NODES


class _PlannerRevisionInput(DomainModel):
    """Public revision facts sent to the Planner without prior model text."""

    objective: str
    user_instructions: str | None = None
    output: OutputRequirement
    max_nodes: int = MAX_PLAN_NODES
    base_graph_version: int
    reason: PlanRevisionReason
    feedback: str | None = None


_PLAN_PROPOSAL_SCHEMA = json.dumps(
    PlanProposal.model_json_schema(),
    ensure_ascii=True,
    allow_nan=False,
    separators=(",", ":"),
)
_PLAN_REVISION_SCHEMA = json.dumps(
    PlanRevision.model_json_schema(),
    ensure_ascii=True,
    allow_nan=False,
    separators=(",", ":"),
)

_CATEGORY_BY_CODE: dict[ErrorCode, ErrorCategory] = {
    ErrorCode.PROVIDER_TIMEOUT: ErrorCategory.TIMEOUT,
    ErrorCode.PROVIDER_RATE_LIMITED: ErrorCategory.RATE_LIMIT,
    ErrorCode.PROVIDER_AUTH_FAILED: ErrorCategory.AUTH,
    ErrorCode.PROVIDER_UNAVAILABLE: ErrorCategory.NETWORK,
    ErrorCode.PROVIDER_INVALID_RESPONSE: ErrorCategory.PROVIDER_ERROR,
    ErrorCode.PROVIDER_NOT_CONFIGURED: ErrorCategory.PROVIDER_ERROR,
}


class PlannerCallResult(DomainModel):
    """Parsed Planner output plus the provider facts needed for accounting."""

    role: Literal[ModelRole.PLANNER] = ModelRole.PLANNER
    model: ModelRef
    proposal: PlanProposal | PlanRevision | None = None
    rejection: PlanRejection | None = None
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    provider_request_id: str | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _outcome_and_provider_facts_are_consistent(self) -> "PlannerCallResult":
        if (self.proposal is None) == (self.rejection is None):
            raise ValueError("a Planner call must contain one proposal or rejection")
        if self.proposal is not None:
            if self.finish_reason is not FinishReason.STOP or self.error is not None:
                raise ValueError("an accepted Planner proposal requires a clean STOP response")
        if self.finish_reason is None and (
            self.usage is not None or self.provider_request_id is not None
        ):
            raise ValueError("provider response facts require a finish reason")
        return self

    @property
    def succeeded(self) -> bool:
        return self.proposal is not None and self.error is None


def new_planning_work_unit(run_id: RunId) -> WorkUnit:
    """Create an internal FK anchor isolated from every compiled user graph."""
    return WorkUnit(
        work_unit_id=new_work_unit_id(),
        run_id=run_id,
        graph_version=PLANNING_GRAPH_VERSION,
        name=PLANNING_WORK_UNIT_NAME,
        instruction="Record one internal Planner provider call",
        status=WorkUnitStatus.RUNNING,
    )


def _provider_error(error: ProviderError) -> ErrorInfo:
    return ErrorInfo(
        category=_CATEGORY_BY_CODE.get(error.code, ErrorCategory.UNKNOWN),
        message=f"Planner provider call failed ({error.code.value})",
    )


def _planner_error(message: str) -> ErrorInfo:
    return ErrorInfo(category=ErrorCategory.PROVIDER_ERROR, message=message)


def _planner_usage(usage: Usage | None) -> Usage | None:
    """Attribute every measured Planner token to the strong-model budget."""
    if usage is None:
        return None
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        strong_model_tokens=usage.total_tokens,
        elapsed_ms=usage.elapsed_ms,
    )


class Planner:
    """Ask one configured PLANNER profile for a proposal without writing state."""

    def __init__(
        self,
        adapter: ProviderAdapter,
        profile: ModelProfile,
    ) -> None:
        self._adapter = adapter
        self._profile = profile

    @property
    def profile(self) -> ModelProfile:
        """The server-side profile used for this call."""
        return self._profile

    def build_request(self, request: NativeRunRequest) -> ProviderRequest:
        """Build the outbound request from a bounded public Native summary."""
        if self._profile.role is not ModelRole.PLANNER:
            raise ProviderError(
                f"model alias {self._profile.alias!r} is not a PLANNER profile",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        summary = _PlannerInput(
            objective=request.input,
            user_instructions=request.instructions,
            output=request.output,
        )
        return ProviderRequest.for_profile(
            self._profile,
            input=summary.model_dump_json(),
            instructions=_PLANNER_INSTRUCTIONS,
            json_schema=_PLAN_PROPOSAL_SCHEMA,
        )

    async def propose(
        self, request: NativeRunRequest
    ) -> PlanProposal | PlanRejection:
        """Return a valid proposal or a stable public rejection.

        Cancellation is intentionally not caught. An interrupted upstream call
        cannot be reported as a deterministic rejection.
        """
        result = await self.propose_call(request)
        if result.rejection is not None:
            return result.rejection
        assert isinstance(result.proposal, PlanProposal)
        return result.proposal

    async def propose_call(self, request: NativeRunRequest) -> PlannerCallResult:
        """Return a parsed proposal or rejection with measured provider facts."""
        try:
            provider_request = self.build_request(request)
            response = await self._adapter.complete(provider_request)
        except ProviderError as error:
            return self._rejected_call(
                PlanRejection(
                    summary="The Planner provider call failed",
                    reasons=(f"provider error: {error.code.value}",),
                ),
                error=_provider_error(error),
            )
        except DomainValidationError as error:
            return self._rejected_call(
                PlanRejection(
                    summary="The Planner request was rejected",
                    reasons=(f"request error: {error.code.value}",),
                ),
                error=ErrorInfo(
                    category=ErrorCategory.INVALID_REQUEST,
                    message=f"Planner request failed validation ({error.code.value})",
                ),
            )

        if response.finish_reason is not FinishReason.STOP:
            return self._rejected_call(
                PlanRejection(
                    summary="The Planner did not complete a proposal",
                    reasons=(f"provider finish reason: {response.finish_reason.value}",),
                ),
                response=response,
                error=_planner_error("Planner provider response did not finish normally"),
            )
        try:
            if response.text is None:
                raise ValueError("Planner response did not contain text")
            proposal = PlanProposal.from_json(response.text)
        except (StrictJsonError, ValidationError, ValueError):
            return self._rejected_call(
                PlanRejection(
                    summary="The Planner response was rejected",
                    reasons=("response is not a valid PlanProposal",),
                ),
                response=response,
                error=_planner_error("Planner provider response was invalid"),
            )
        return PlannerCallResult(
            model=self._profile.model_ref,
            proposal=proposal,
            usage=_planner_usage(response.usage),
            finish_reason=response.finish_reason,
            provider_request_id=response.provider_request_id,
        )

    def build_revision_request(
        self,
        request: NativeRunRequest,
        *,
        base_graph_version: int,
        reason: PlanRevisionReason,
        feedback: str | None = None,
    ) -> ProviderRequest:
        """Build a bounded revision request from public persisted facts."""
        if base_graph_version < 1:
            raise ValueError("base_graph_version must be at least 1")
        if self._profile.role is not ModelRole.PLANNER:
            raise ProviderError(
                f"model alias {self._profile.alias!r} is not a PLANNER profile",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        summary = _PlannerRevisionInput(
            objective=request.input,
            user_instructions=request.instructions,
            output=request.output,
            base_graph_version=base_graph_version,
            reason=reason,
            feedback=feedback,
        )
        return ProviderRequest.for_profile(
            self._profile,
            input=summary.model_dump_json(),
            instructions=(
                "Revise the bounded work graph using only the public failure facts. "
                "Return strict JSON matching the PlanRevision schema, including exactly "
                "one final_node identifying the user-facing output node. Do not include "
                "reasoning, chain-of-thought, credentials, or persisted IDs."
            ),
            json_schema=_PLAN_REVISION_SCHEMA,
        )

    async def revise(
        self,
        request: NativeRunRequest,
        *,
        base_graph_version: int,
        reason: PlanRevisionReason,
        feedback: str | None = None,
    ) -> PlanRevision | PlanRejection:
        """Return a strict public revision or a stable rejection."""
        result = await self.revise_call(
            request,
            base_graph_version=base_graph_version,
            reason=reason,
            feedback=feedback,
        )
        if result.rejection is not None:
            return result.rejection
        assert isinstance(result.proposal, PlanRevision)
        return result.proposal

    async def revise_call(
        self,
        request: NativeRunRequest,
        *,
        base_graph_version: int,
        reason: PlanRevisionReason,
        feedback: str | None = None,
    ) -> PlannerCallResult:
        """Return a parsed revision or rejection with measured provider facts."""
        try:
            provider_request = self.build_revision_request(
                request,
                base_graph_version=base_graph_version,
                reason=reason,
                feedback=feedback,
            )
            response = await self._adapter.complete(provider_request)
        except ProviderError as error:
            return self._rejected_call(
                PlanRejection(
                    summary="The Planner revision provider call failed",
                    reasons=(f"provider error: {error.code.value}",),
                ),
                error=_provider_error(error),
            )
        except DomainValidationError as error:
            return self._rejected_call(
                PlanRejection(
                    summary="The Planner revision request was rejected",
                    reasons=(f"request error: {error.code.value}",),
                ),
                error=ErrorInfo(
                    category=ErrorCategory.INVALID_REQUEST,
                    message=f"Planner revision request failed validation ({error.code.value})",
                ),
            )

        if response.finish_reason is not FinishReason.STOP:
            return self._rejected_call(
                PlanRejection(
                    summary="The Planner did not complete a revision",
                    reasons=(f"provider finish reason: {response.finish_reason.value}",),
                ),
                response=response,
                error=_planner_error("Planner provider response did not finish normally"),
            )
        try:
            if response.text is None:
                raise ValueError("Planner revision response did not contain text")
            proposal = PlanRevision.from_json(response.text)
        except (StrictJsonError, ValidationError, ValueError):
            return self._rejected_call(
                PlanRejection(
                    summary="The Planner revision response was rejected",
                    reasons=("response is not a valid PlanRevision",),
                ),
                response=response,
                error=_planner_error("Planner provider response was invalid"),
            )
        return PlannerCallResult(
            model=self._profile.model_ref,
            proposal=proposal,
            usage=_planner_usage(response.usage),
            finish_reason=response.finish_reason,
            provider_request_id=response.provider_request_id,
        )

    def _rejected_call(
        self,
        rejection: PlanRejection,
        *,
        error: ErrorInfo,
        response: ProviderResponse | None = None,
    ) -> PlannerCallResult:
        return PlannerCallResult(
            model=self._profile.model_ref,
            rejection=rejection,
            usage=None if response is None else _planner_usage(response.usage),
            finish_reason=None if response is None else response.finish_reason,
            provider_request_id=(
                None if response is None else response.provider_request_id
            ),
            error=error,
        )
