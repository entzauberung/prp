"""Outbound provider contract.

A provider adapter turns one normalised text request into one normalised
response. Endpoint and credential come from server-side configuration only: a
caller can never supply a base URL or an API key, which keeps request-driven
SSRF and credential injection impossible.

Tool descriptors are protocol-neutral public metadata. They never contain
handlers, effect decisions, executables, credentials or workspace roots.
"""

import json
from decimal import Decimal
from enum import StrEnum, unique
from typing import Annotated, Protocol, runtime_checkable

from pydantic import ConfigDict, Field, SecretStr, StringConstraints, model_validator

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import DomainValidationError, ErrorCode
from prp_runtime.domain.models import (
    MAX_AGENT_HISTORY_ITEMS,
    MAX_PROVIDER_TOOL_COUNT,
    MAX_PROVIDER_TOOLS_BYTES,
    AgentHistoryItem,
    AgentToolCall,
    AgentTurn,
    AttemptCost,
    DomainModel,
    Money,
    NonBlankText,
    PromptText,
    ProviderToolDescriptor,
    Usage,
)
from prp_runtime.domain.values import ModelRef

__all__ = [
    "FinishReason",
    "AgentToolCall",
    "AgentTurn",
    "AttemptCost",
    "ModelProfile",
    "ProviderProtocol",
    "ProviderAdapter",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderToolDescriptor",
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
    TOOL_CALLS = "TOOL_CALLS"
    OTHER = "OTHER"


@unique
class ProviderProtocol(StrEnum):
    """Closed set of supported outbound provider protocols."""

    OPENAI_CHAT = "OPENAI_CHAT"
    OPENAI_RESPONSES = "OPENAI_RESPONSES"
    ANTHROPIC_MESSAGES = "ANTHROPIC_MESSAGES"


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
    protocol: ProviderProtocol = ProviderProtocol.OPENAI_CHAT
    anthropic_version: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=10,
            max_length=32,
            pattern=r"^\d{4}-\d{2}-\d{2}$",
        ),
    ] | None = None
    supports_structured_output: bool = False
    context_window_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    input_price_per_million_tokens: Money = Decimal("0")
    output_price_per_million_tokens: Money = Decimal("0")
    max_concurrency: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0.0)

    @model_validator(mode="after")
    def _output_fits_the_context_window(self) -> "ModelProfile":
        if self.max_output_tokens > self.context_window_tokens:
            raise ValueError("max_output_tokens cannot exceed context_window_tokens")
        if self.protocol is ProviderProtocol.ANTHROPIC_MESSAGES:
            if self.anthropic_version is None:
                raise ValueError("ANTHROPIC_MESSAGES requires anthropic_version")
        elif self.anthropic_version is not None:
            raise ValueError("anthropic_version is only valid for ANTHROPIC_MESSAGES")
        return self

    @property
    def model_ref(self) -> ModelRef:
        """The domain reference recorded on every attempt using this profile."""
        return ModelRef(provider=self.provider, model=self.model)

    def cost_for_usage(self, usage: Usage | None) -> AttemptCost | None:
        """Calculate exact cost, preserving unknown Provider usage."""
        return AttemptCost.from_usage(
            usage,
            input_price_per_million_tokens=self.input_price_per_million_tokens,
            output_price_per_million_tokens=self.output_price_per_million_tokens,
        )


class ProviderRequest(DomainModel):
    """One normalised outbound text request.

    There is deliberately no ``base_url`` and no ``api_key`` field: both belong to
    the adapter's server-side profile.
    """

    alias: Alias
    model: Name
    input: PromptText
    instructions: PromptText | None = None
    max_output_tokens: int = Field(gt=0)
    json_schema: str | None = None
    timeout_seconds: float = Field(gt=0.0)
    history: tuple[AgentHistoryItem, ...] = ()
    tools: tuple[ProviderToolDescriptor, ...] = ()

    @model_validator(mode="after")
    def _request_is_bounded(self) -> "ProviderRequest":
        if len(self.history) > MAX_AGENT_HISTORY_ITEMS:
            raise ValueError("provider history has too many items")
        known_calls: dict[str, AgentToolCall] = {}
        for item in self.history:
            if isinstance(item, AgentTurn):
                for call in item.tool_calls:
                    previous = known_calls.get(call.call_id)
                    if previous is not None and previous != call:
                        raise ValueError("provider history reuses a tool call id differently")
                    known_calls[call.call_id] = call
            elif item.call_id not in known_calls:
                raise ValueError("provider history contains an orphaned tool result")
        if len(self.tools) > MAX_PROVIDER_TOOL_COUNT:
            raise ValueError("provider request has too many tools")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("provider request tool names must be unique")
        try:
            encoded = json.dumps(
                [tool.model_dump(mode="json") for tool in self.tools],
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("provider tools must contain standard JSON") from error
        if len(encoded) > MAX_PROVIDER_TOOLS_BYTES:
            raise ValueError("provider request tools exceed the size limit")
        return self

    @classmethod
    def for_profile(
        cls,
        profile: ModelProfile,
        *,
        input: str,
        instructions: str | None = None,
        json_schema: str | None = None,
        max_output_tokens: int | None = None,
        history: tuple[AgentHistoryItem, ...] = (),
        tools: tuple[ProviderToolDescriptor, ...] = (),
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
            history=history,
            tools=tools,
        )


class ProviderResponse(DomainModel):
    """One normalised text or tool-call response.

    ``usage`` is ``None`` when the upstream did not report token counts. Zero is a
    measurement, so an unreported count is never recorded as zero.
    """

    text: NonBlankText | None = None
    tool_calls: tuple[AgentToolCall, ...] = ()
    usage: Usage | None = None
    finish_reason: FinishReason
    provider_request_id: str | None = None

    @model_validator(mode="after")
    def _response_has_one_turn_shape(self) -> "ProviderResponse":
        if (self.text is None) == (not self.tool_calls):
            raise ValueError("provider response must contain text or tool_calls, exclusively")
        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("provider response tool call ids must be unique")
        if self.tool_calls and self.finish_reason is not FinishReason.TOOL_CALLS:
            raise ValueError("provider tool calls require the TOOL_CALLS finish reason")
        if self.text is not None and self.finish_reason is FinishReason.TOOL_CALLS:
            raise ValueError("provider text cannot use the TOOL_CALLS finish reason")
        return self

    @property
    def turn(self) -> AgentTurn:
        """Expose the provider result as the protocol-neutral AgentTurn contract."""
        return AgentTurn(text=self.text, tool_calls=self.tool_calls)


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

    async def aclose(self) -> None:
        """Close resources owned by this adapter, exactly once semantically.

        Injected clients or other host-owned resources remain the host's
        responsibility; implementations must make this operation idempotent.
        """

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Produce one completion for one request."""
