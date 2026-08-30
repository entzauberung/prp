"""Structured domain errors.

Every error carries a stable code, a family, a human message and a deterministic
``retryable`` flag. No error ever carries a stack trace, an exception object, a
credential or a full upstream request.
"""

from collections.abc import Mapping
from enum import StrEnum, unique
from types import MappingProxyType
from typing import ClassVar

from pydantic import model_validator

from prp_runtime.domain.models import DomainModel
from prp_runtime.domain.transitions import DomainTransitionError

__all__ = [
    "BudgetError",
    "DomainValidationError",
    "ERROR_FAMILIES",
    "ErrorCode",
    "ErrorDetail",
    "ErrorFamily",
    "InternalError",
    "PrpError",
    "ProtocolError",
    "ProviderError",
    "RETRYABLE_CODES",
    "StateError",
    "state_error_from_transition",
]


@unique
class ErrorFamily(StrEnum):
    """Coarse error family. Inbound bindings map families to status codes."""

    VALIDATION = "VALIDATION"
    STATE = "STATE"
    BUDGET = "BUDGET"
    PROVIDER = "PROVIDER"
    PROTOCOL = "PROTOCOL"
    INTERNAL = "INTERNAL"


@unique
class ErrorCode(StrEnum):
    """Stable error codes. Codes are never renamed or aliased."""

    # VALIDATION
    INVALID_REQUEST = "invalid_request"
    INPUT_TOO_LARGE = "input_too_large"
    INVALID_BUDGET = "invalid_budget"
    INVALID_OUTPUT_REQUIREMENT = "invalid_output_requirement"
    INVALID_AGENT_OPTIONS = "invalid_agent_options"

    # STATE
    RUN_NOT_FOUND = "run_not_found"
    WORK_UNIT_NOT_FOUND = "work_unit_not_found"
    ILLEGAL_STATE_TRANSITION = "illegal_state_transition"
    RUN_ALREADY_TERMINAL = "run_already_terminal"
    RUN_CANCELLED = "run_cancelled"
    EVENT_SEQUENCE_INVALID = "event_sequence_invalid"

    # BUDGET
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    ATTEMPT_BUDGET_EXCEEDED = "attempt_budget_exceeded"
    REVISION_BUDGET_EXCEEDED = "revision_budget_exceeded"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RESOURCE_BUDGET_EXCEEDED = "resource_budget_exceeded"

    # PROVIDER
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    PROVIDER_AUTH_FAILED = "provider_auth_failed"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_INVALID_RESPONSE = "provider_invalid_response"

    # PROTOCOL
    UNSUPPORTED_FIELD = "unsupported_field"
    UNSUPPORTED_MODALITY = "unsupported_modality"
    UNSUPPORTED_TOOLS = "unsupported_tools"
    UNSUPPORTED_STREAM_MODE = "unsupported_stream_mode"

    # INTERNAL
    INTERNAL_ERROR = "internal_error"


ERROR_FAMILIES: Mapping[ErrorCode, ErrorFamily] = MappingProxyType(
    {
        ErrorCode.INVALID_REQUEST: ErrorFamily.VALIDATION,
        ErrorCode.INPUT_TOO_LARGE: ErrorFamily.VALIDATION,
        ErrorCode.INVALID_BUDGET: ErrorFamily.VALIDATION,
        ErrorCode.INVALID_OUTPUT_REQUIREMENT: ErrorFamily.VALIDATION,
        ErrorCode.INVALID_AGENT_OPTIONS: ErrorFamily.VALIDATION,
        ErrorCode.RUN_NOT_FOUND: ErrorFamily.STATE,
        ErrorCode.WORK_UNIT_NOT_FOUND: ErrorFamily.STATE,
        ErrorCode.ILLEGAL_STATE_TRANSITION: ErrorFamily.STATE,
        ErrorCode.RUN_ALREADY_TERMINAL: ErrorFamily.STATE,
        ErrorCode.RUN_CANCELLED: ErrorFamily.STATE,
        ErrorCode.EVENT_SEQUENCE_INVALID: ErrorFamily.STATE,
        ErrorCode.TOKEN_BUDGET_EXCEEDED: ErrorFamily.BUDGET,
        ErrorCode.ATTEMPT_BUDGET_EXCEEDED: ErrorFamily.BUDGET,
        ErrorCode.REVISION_BUDGET_EXCEEDED: ErrorFamily.BUDGET,
        ErrorCode.DEADLINE_EXCEEDED: ErrorFamily.BUDGET,
        ErrorCode.RESOURCE_BUDGET_EXCEEDED: ErrorFamily.BUDGET,
        ErrorCode.PROVIDER_NOT_CONFIGURED: ErrorFamily.PROVIDER,
        ErrorCode.PROVIDER_AUTH_FAILED: ErrorFamily.PROVIDER,
        ErrorCode.PROVIDER_RATE_LIMITED: ErrorFamily.PROVIDER,
        ErrorCode.PROVIDER_TIMEOUT: ErrorFamily.PROVIDER,
        ErrorCode.PROVIDER_UNAVAILABLE: ErrorFamily.PROVIDER,
        ErrorCode.PROVIDER_INVALID_RESPONSE: ErrorFamily.PROVIDER,
        ErrorCode.UNSUPPORTED_FIELD: ErrorFamily.PROTOCOL,
        ErrorCode.UNSUPPORTED_MODALITY: ErrorFamily.PROTOCOL,
        ErrorCode.UNSUPPORTED_TOOLS: ErrorFamily.PROTOCOL,
        ErrorCode.UNSUPPORTED_STREAM_MODE: ErrorFamily.PROTOCOL,
        ErrorCode.INTERNAL_ERROR: ErrorFamily.INTERNAL,
    }
)

RETRYABLE_CODES: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.PROVIDER_RATE_LIMITED,
        ErrorCode.PROVIDER_TIMEOUT,
        ErrorCode.PROVIDER_UNAVAILABLE,
    }
)


class ErrorDetail(DomainModel):
    """The serialisable form of an error. Safe to return to a client."""

    code: ErrorCode
    family: ErrorFamily
    message: str
    retryable: bool
    field: str | None = None

    @model_validator(mode="after")
    def _code_family_and_retryability_agree(self) -> "ErrorDetail":
        if ERROR_FAMILIES[self.code] is not self.family:
            raise ValueError(f"code {self.code.value} does not belong to family {self.family}")
        if self.retryable is not (self.code in RETRYABLE_CODES):
            raise ValueError(f"retryable flag does not match code {self.code.value}")
        if not self.message.strip():
            raise ValueError("message must not be blank")
        return self

    @classmethod
    def for_code(
        cls, code: ErrorCode, message: str, *, field: str | None = None
    ) -> "ErrorDetail":
        """Build a detail with the family and retryability implied by the code."""
        return cls(
            code=code,
            family=ERROR_FAMILIES[code],
            message=message,
            retryable=code in RETRYABLE_CODES,
            field=field,
        )


class PrpError(Exception):
    """Base runtime error. Carries a serialisable detail, never a stack trace."""

    family: ClassVar[ErrorFamily] = ErrorFamily.INTERNAL
    default_code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        field: str | None = None,
    ) -> None:
        resolved = code if code is not None else self.default_code
        if ERROR_FAMILIES[resolved] is not self.family:
            raise ValueError(
                f"code {resolved.value} cannot be raised as {type(self).__name__}"
            )
        self.detail = ErrorDetail.for_code(resolved, message, field=field)
        super().__init__(message)

    @property
    def code(self) -> ErrorCode:
        return self.detail.code

    @property
    def retryable(self) -> bool:
        return self.detail.retryable


class DomainValidationError(PrpError):
    """The request or a domain object is not acceptable."""

    family: ClassVar[ErrorFamily] = ErrorFamily.VALIDATION
    default_code: ClassVar[ErrorCode] = ErrorCode.INVALID_REQUEST


class StateError(PrpError):
    """The requested action does not fit the current persisted state."""

    family: ClassVar[ErrorFamily] = ErrorFamily.STATE
    default_code: ClassVar[ErrorCode] = ErrorCode.ILLEGAL_STATE_TRANSITION


class BudgetError(PrpError):
    """A declared ceiling would be crossed."""

    family: ClassVar[ErrorFamily] = ErrorFamily.BUDGET
    default_code: ClassVar[ErrorCode] = ErrorCode.TOKEN_BUDGET_EXCEEDED


class ProviderError(PrpError):
    """An outbound model call failed."""

    family: ClassVar[ErrorFamily] = ErrorFamily.PROVIDER
    default_code: ClassVar[ErrorCode] = ErrorCode.PROVIDER_UNAVAILABLE


class ProtocolError(PrpError):
    """An inbound request used a field this runtime does not support."""

    family: ClassVar[ErrorFamily] = ErrorFamily.PROTOCOL
    default_code: ClassVar[ErrorCode] = ErrorCode.UNSUPPORTED_FIELD


class InternalError(PrpError):
    """An invariant inside the runtime was violated."""

    family: ClassVar[ErrorFamily] = ErrorFamily.INTERNAL
    default_code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_ERROR


def state_error_from_transition(error: DomainTransitionError) -> StateError:
    """Wrap a state machine violation as a client-safe state error."""
    return StateError(str(error), code=ErrorCode.ILLEGAL_STATE_TRANSITION)
