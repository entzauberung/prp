"""Pure shared helpers for external request normalization.

Protocol routers translate their wire-specific text shapes into a small mapping
and call these functions. This module performs no IO and never reads provider
configuration, credentials, or runtime state.
"""

from collections.abc import Mapping
from enum import StrEnum, unique
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from prp_runtime.api.errors import binding_error
from prp_runtime.domain.errors import ErrorCode
from prp_runtime.domain.models import (
    Budget,
    NativeRunRequest,
    OutputRequirement,
    RoutingIntent,
)
from prp_runtime.domain.values import RunId

__all__ = [
    "BindingNormalizationResult",
    "BindingOperation",
    "normalize_cancel",
    "normalize_query",
    "normalize_request",
    "reject_unsupported_fields",
]


@unique
class BindingOperation(StrEnum):
    """The state operation represented by a binding input."""

    CREATE = "CREATE"
    QUERY = "QUERY"
    CANCEL = "CANCEL"


class BindingNormalizationResult(BaseModel):
    """Normalized operation data shared by all inbound protocol routers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: BindingOperation
    request: NativeRunRequest | None = None
    run_id: RunId | None = None

    @property
    def native_request(self) -> NativeRunRequest | None:
        """Alias used by adapters that name the domain value explicitly."""
        return self.request


_REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "input",
        "instructions",
        "routing_policy",
        "strategy",
        "routing",
        "budget",
        "output",
    }
)
_BUDGET_FIELDS: Final[frozenset[str]] = frozenset(Budget.model_fields)
_OUTPUT_FIELDS: Final[frozenset[str]] = frozenset(OutputRequirement.model_fields)
_ROUTING_FIELDS: Final[frozenset[str]] = frozenset(RoutingIntent.model_fields)
_UNSUPPORTED_FIELDS: Final[dict[str, ErrorCode]] = {
    "stream": ErrorCode.UNSUPPORTED_STREAM_MODE,
    "stream_options": ErrorCode.UNSUPPORTED_STREAM_MODE,
    "tools": ErrorCode.UNSUPPORTED_TOOLS,
    "tool_choice": ErrorCode.UNSUPPORTED_TOOLS,
    "modalities": ErrorCode.UNSUPPORTED_MODALITY,
    "audio": ErrorCode.UNSUPPORTED_MODALITY,
    "images": ErrorCode.UNSUPPORTED_MODALITY,
    "image": ErrorCode.UNSUPPORTED_MODALITY,
}


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            f"{field} must be an object",
            field=field,
        )
    return value


def _reject_unknown(fields: set[object], *, allowed: frozenset[str]) -> None:
    for field in sorted(fields, key=str):
        if not isinstance(field, str):
            raise binding_error(
                ErrorCode.UNSUPPORTED_FIELD,
                "the binding contains an unsupported field",
                field="body",
            )
        code = _UNSUPPORTED_FIELDS.get(field, ErrorCode.UNSUPPORTED_FIELD)
        if field not in allowed:
            raise binding_error(
                code,
                "the binding field is not supported",
                field=field,
            )


def reject_unsupported_fields(
    payload: Mapping[str, object],
    *,
    allowed: frozenset[str],
) -> None:
    """Reject wire fields outside one protocol's declared supported subset."""
    if not isinstance(payload, Mapping):
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "the binding request must be an object",
            field="body",
        )
    _reject_unknown(set(payload), allowed=allowed)


def _budget(value: object) -> Budget:
    data = _object(value, field="budget")
    _reject_unknown(set(data), allowed=_BUDGET_FIELDS)
    try:
        return Budget.model_validate(data)
    except ValidationError as error:
        raise binding_error(
            ErrorCode.INVALID_BUDGET,
            "budget does not match the native budget contract",
            field="budget",
        ) from error


def _output(value: object) -> OutputRequirement:
    data = _object(value, field="output")
    _reject_unknown(set(data), allowed=_OUTPUT_FIELDS)
    try:
        return OutputRequirement.model_validate(data)
    except ValidationError as error:
        raise binding_error(
            ErrorCode.INVALID_OUTPUT_REQUIREMENT,
            "output does not match the native output contract",
            field="output",
        ) from error


def _routing(value: object) -> RoutingIntent:
    data = _object(value, field="routing")
    _reject_unknown(set(data), allowed=_ROUTING_FIELDS)
    try:
        return RoutingIntent.model_validate(data)
    except ValidationError as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "routing does not match the native routing intent contract",
            field="routing",
        ) from error


def normalize_request(payload: Mapping[str, object]) -> BindingNormalizationResult:
    """Normalize one canonical binding mapping into a NativeRunRequest."""
    if not isinstance(payload, Mapping):
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "the binding request must be an object",
            field="body",
        )
    _reject_unknown(set(payload), allowed=_REQUEST_FIELDS)
    values: dict[str, object] = dict(payload)
    if "budget" in values:
        values["budget"] = _budget(values["budget"])
    if "output" in values:
        values["output"] = _output(values["output"])
    if "routing" in values:
        values["routing"] = _routing(values["routing"])
    try:
        request = NativeRunRequest.model_validate(values)
    except ValidationError as error:
        field = None
        if error.errors():
            location = error.errors()[0].get("loc", ())
            if location:
                field = str(location[0])
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "the binding request does not match the native request contract",
            field=field,
        ) from error
    return BindingNormalizationResult(
        operation=BindingOperation.CREATE,
        request=request,
    )


def _normalize_control(operation: BindingOperation, run_id: str) -> BindingNormalizationResult:
    try:
        result = BindingNormalizationResult(
            operation=operation,
            run_id=run_id,
        )
    except ValidationError as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "the run identifier is invalid",
            field="run_id",
        ) from error
    return result


def normalize_query(run_id: str) -> BindingNormalizationResult:
    """Validate a query identifier without touching the store."""
    return _normalize_control(BindingOperation.QUERY, run_id)


def normalize_cancel(run_id: str) -> BindingNormalizationResult:
    """Validate a cancellation identifier without touching the store."""
    return _normalize_control(BindingOperation.CANCEL, run_id)
