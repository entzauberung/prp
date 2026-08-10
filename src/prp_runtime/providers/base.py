"""Outbound provider contract.

A provider adapter turns one normalised text request into one normalised
response. Endpoint and credential come from server-side configuration only: a
caller can never supply a base URL or an API key, which keeps request-driven
SSRF and credential injection impossible.

This layer declares no tool calling and no multimodal capability.
"""

from enum import StrEnum, unique
from typing import Annotated, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, SecretStr, StringConstraints, model_validator

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import DomainValidationError, ErrorCode
from prp_runtime.domain.models import DomainModel, Usage
from prp_runtime.domain.values import ModelRef

__all__ = [
    "FinishReason",
    "ModelProfile",
    "ProviderAdapter",
    "ProviderRequest",
    "ProviderResponse",
]

Alias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$"
    ),
]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Endpoint = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2048, pattern=r"^https?://"),
]


@unique
class FinishReason(StrEnum):
    """Why the model stopped producing text."""

    STOP = "STOP"
    LENGTH = "LENGTH"
    CONTENT_FILTER = "CONTENT_FILTER"
    OTHER = "OTHER"


class ModelProfile(DomainModel):
    """A server-side model configuration referenced by alias.

    The API key is held as a secret: it never appears in ``repr`` or in JSON
    output.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: Alias
    provider: Name
    model: Name
    role: ModelRole
    base_url: Endpoint
    api_key: SecretStr | None = None
    supports_structured_output: bool = False
    context_window_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    input_price_per_million_tokens: float = Field(default=0.0, ge=0.0)
    output_price_per_million_tokens: float = Field(default=0.0, ge=0.0)
    max_concurrency: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0.0)

    @model_validator(mode="after")
    def _output_fits_the_context_window(self) -> "ModelProfile":
        if self.max_output_tokens > self.context_window_tokens:
            raise ValueError("max_output_tokens cannot exceed context_window_tokens")
        return self

    @property
    def model_ref(self) -> ModelRef:
        """The domain reference recorded on every attempt using this profile."""
        return ModelRef(provider=self.provider, model=self.model)


class ProviderRequest(DomainModel):
    """One normalised outbound text request.

    There is deliberately no ``base_url`` and no ``api_key`` field: both belong to
    the adapter's server-side profile.
    """

    alias: Alias
    model: Name
    input: str = Field(min_length=1)
    instructions: str | None = None
    max_output_tokens: int = Field(gt=0)
    json_schema: str | None = None
    timeout_seconds: float = Field(gt=0.0)

    @classmethod
    def for_profile(
        cls,
        profile: ModelProfile,
        *,
        input: str,
        instructions: str | None = None,
        json_schema: str | None = None,
        max_output_tokens: int | None = None,
    ) -> "ProviderRequest":
        """Build a request for a profile, rejecting capabilities it does not have."""
        if json_schema is not None and not profile.supports_structured_output:
            raise DomainValidationError(
                f"model profile {profile.alias} does not support structured output",
                code=ErrorCode.INVALID_OUTPUT_REQUIREMENT,
                field="json_schema",
            )
        limit = profile.max_output_tokens if max_output_tokens is None else max_output_tokens
        if limit > profile.max_output_tokens:
            raise DomainValidationError(
                f"max_output_tokens {limit} exceeds the limit of profile {profile.alias}",
                code=ErrorCode.INVALID_REQUEST,
                field="max_output_tokens",
            )
        return cls(
            alias=profile.alias,
            model=profile.model,
            input=input,
            instructions=instructions,
            json_schema=json_schema,
            max_output_tokens=limit,
            timeout_seconds=profile.timeout_seconds,
        )


class ProviderResponse(DomainModel):
    """One normalised outbound text response.

    ``usage`` is ``None`` when the upstream did not report token counts. Zero is a
    measurement, so an unreported count is never recorded as zero.
    """

    text: str
    usage: Usage | None = None
    finish_reason: FinishReason
    provider_request_id: str | None = None


@runtime_checkable
class ProviderAdapter(Protocol):
    """The only outbound model interface.

    Cancellation boundary: ``complete`` must let ``asyncio.CancelledError``
    propagate and must never swallow it. The caller decides how to record the
    attempt, because a cancelled in-flight call cannot be proven to have been
    stopped upstream.

    Failures are reported as ``ProviderError`` with a classified code; a timeout
    is reported rather than retried inside the adapter.
    """

    @property
    def name(self) -> str:
        """The provider name recorded on attempts."""

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Produce one completion for one request."""
