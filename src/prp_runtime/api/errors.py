"""HTTP error mapping.

Every failure leaves the runtime as a stable error code with a redacted message.
A stack trace, an internal path, a provider URL or a credential is never part of
a response body.
"""

from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from prp_runtime.domain.errors import ErrorCode, ErrorDetail, ErrorFamily, PrpError

__all__ = [
    "STATUS_BY_CODE",
    "STATUS_BY_FAMILY",
    "ErrorResponse",
    "error_response",
    "install_error_handlers",
    "status_for",
]

#: Default mapping from error family to HTTP status.
STATUS_BY_FAMILY: Mapping[ErrorFamily, int] = {
    ErrorFamily.VALIDATION: 400,
    ErrorFamily.PROTOCOL: 400,
    ErrorFamily.STATE: 409,
    ErrorFamily.BUDGET: 409,
    ErrorFamily.PROVIDER: 502,
    ErrorFamily.INTERNAL: 500,
}

#: Codes whose HTTP status is more specific than their family default.
STATUS_BY_CODE: Mapping[ErrorCode, int] = {
    ErrorCode.RUN_NOT_FOUND: 404,
    ErrorCode.WORK_UNIT_NOT_FOUND: 404,
    ErrorCode.INPUT_TOO_LARGE: 413,
    ErrorCode.DEADLINE_EXCEEDED: 504,
    ErrorCode.PROVIDER_RATE_LIMITED: 429,
    ErrorCode.PROVIDER_TIMEOUT: 504,
    ErrorCode.PROVIDER_UNAVAILABLE: 503,
    ErrorCode.PROVIDER_NOT_CONFIGURED: 503,
}


class ErrorResponse(BaseModel):
    """The only error body shape this API returns."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail


def status_for(detail: ErrorDetail) -> int:
    """The HTTP status for one error detail."""
    specific = STATUS_BY_CODE.get(detail.code)
    if specific is not None:
        return specific
    return STATUS_BY_FAMILY[detail.family]


def error_response(detail: ErrorDetail) -> JSONResponse:
    """Render one error detail as a JSON response."""
    body = ErrorResponse(error=detail)
    return JSONResponse(
        status_code=status_for(detail),
        content=body.model_dump(mode="json"),
        headers={"Cache-Control": "no-store"},
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers that turn failures into stable error bodies."""

    @app.exception_handler(PrpError)
    async def _domain_error(request: Request, error: Exception) -> JSONResponse:
        assert isinstance(error, PrpError)
        return error_response(error.detail)

    @app.exception_handler(RequestValidationError)
    async def _request_validation(
        request: Request, error: Exception
    ) -> JSONResponse:
        assert isinstance(error, RequestValidationError)
        field = None
        errors = error.errors()
        if errors:
            location = [str(part) for part in errors[0].get("loc", ()) if part != "body"]
            field = ".".join(location) or None
        return error_response(
            ErrorDetail.for_code(
                ErrorCode.INVALID_REQUEST,
                "the request does not match the native run request contract",
                field=field,
            )
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, error: Exception) -> JSONResponse:
        # The message is deliberately not the exception text: it can contain
        # internal detail. Only the exception type is reported.
        return error_response(
            ErrorDetail.for_code(
                ErrorCode.INTERNAL_ERROR,
                f"the runtime failed to handle the request ({type(error).__name__})",
            )
        )
