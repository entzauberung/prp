"""OpenAI-compatible outbound adapter.

One text completion per call over the Chat Completions shape. No tool calling, no
images and no outbound streaming. The endpoint and credential come from the
server-side profile; the request never carries them.
"""

from time import perf_counter
from typing import Any

import httpx

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.models import Usage
from prp_runtime.json_support import strict_json_loads
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)

__all__ = ["MAX_ERROR_BODY_CHARS", "OpenAICompatibleProvider"]

MAX_ERROR_BODY_CHARS = 500

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "end_turn": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def _non_negative_int(value: object) -> int | None:
    """Read a token count, refusing booleans and negatives."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class OpenAICompatibleProvider:
    """Calls one configured OpenAI-compatible endpoint."""

    def __init__(
        self, profile: ModelProfile, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._profile = profile
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return self._profile.provider

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def endpoint(self) -> str:
        return f"{self._profile.base_url.rstrip('/')}/chat/completions"

    async def aclose(self) -> None:
        """Close the client this adapter owns. An injected client is left alone."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Perform one completion.

        ``asyncio.CancelledError`` is never caught here: the caller decides how to
        record an attempt whose upstream outcome cannot be confirmed.
        """
        if request.alias != self._profile.alias:
            raise ProviderError(
                f"request for alias {request.alias!r} was sent to profile "
                f"{self._profile.alias!r}",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        payload = self._build_payload(request)
        client = self._ensure_client()
        started = perf_counter()
        try:
            response = await client.post(
                self.endpoint,
                json=payload,
                headers=self._headers(),
                timeout=httpx.Timeout(request.timeout_seconds),
            )
        except httpx.TimeoutException as error:
            raise ProviderError(
                f"upstream {self._profile.alias} timed out after "
                f"{request.timeout_seconds} seconds",
                code=ErrorCode.PROVIDER_TIMEOUT,
            ) from error
        except httpx.HTTPError as error:
            raise ProviderError(
                f"upstream {self._profile.alias} is unreachable ({type(error).__name__})",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
            ) from error
        elapsed_ms = max(int((perf_counter() - started) * 1000), 0)
        if response.status_code >= 400:
            raise self._status_error(response)
        return self._parse(response, elapsed_ms)

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._profile.api_key is not None:
            headers["Authorization"] = f"Bearer {self._profile.api_key.get_secret_value()}"
        return headers

    def _build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.instructions is not None:
            messages.append({"role": "system", "content": request.instructions})
        messages.append({"role": "user", "content": request.input})
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        if request.json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    # Strict: a non-finite number here would be sent to the
                    # provider as a token standard JSON cannot express.
                    "schema": strict_json_loads(request.json_schema),
                },
            }
        return payload

    def _redact(self, text: str) -> str:
        collapsed = " ".join(text.split())[:MAX_ERROR_BODY_CHARS]
        if self._profile.api_key is None:
            return collapsed
        return collapsed.replace(self._profile.api_key.get_secret_value(), "[redacted]")

    def _status_error(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        if status in (401, 403):
            code = ErrorCode.PROVIDER_AUTH_FAILED
        elif status == 429:
            code = ErrorCode.PROVIDER_RATE_LIMITED
        elif status == 408:
            code = ErrorCode.PROVIDER_TIMEOUT
        elif status >= 500:
            code = ErrorCode.PROVIDER_UNAVAILABLE
        else:
            code = ErrorCode.PROVIDER_INVALID_RESPONSE
        return ProviderError(
            f"upstream {self._profile.alias} returned status {status}: "
            f"{self._redact(response.text)}",
            code=code,
        )

    def _invalid_response(self, reason: str) -> ProviderError:
        return ProviderError(
            f"upstream {self._profile.alias} returned an unusable response: {reason}",
            code=ErrorCode.PROVIDER_INVALID_RESPONSE,
        )

    def _parse(self, response: httpx.Response, elapsed_ms: int) -> ProviderResponse:
        try:
            document = response.json()
        except ValueError as error:
            raise self._invalid_response("the body is not JSON") from error
        if not isinstance(document, dict):
            raise self._invalid_response("the body is not a JSON object")
        choices = document.get("choices")
        if not isinstance(choices, list) or not choices:
            raise self._invalid_response("no choices were returned")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise self._invalid_response("the first choice is not an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise self._invalid_response("the first choice has no message object")
        content = message.get("content")
        if not isinstance(content, str):
            raise self._invalid_response("the message content is not text")
        request_id = document.get("id")
        return ProviderResponse(
            text=content,
            usage=self._parse_usage(document.get("usage"), elapsed_ms),
            finish_reason=_FINISH_REASONS.get(
                str(choice.get("finish_reason") or ""), FinishReason.OTHER
            ),
            provider_request_id=request_id if isinstance(request_id, str) else None,
        )

    def _parse_usage(self, raw: object, elapsed_ms: int) -> Usage | None:
        """Normalise reported token counts, or report usage as unavailable."""
        if not isinstance(raw, dict):
            return None
        input_tokens = _non_negative_int(raw.get("prompt_tokens"))
        output_tokens = _non_negative_int(raw.get("completion_tokens"))
        if input_tokens is None or output_tokens is None:
            return None
        # A planner profile is the expensive model, so all of its tokens count as
        # strong-model tokens regardless of the role of the individual attempt.
        strong = (
            input_tokens + output_tokens if self._profile.role is ModelRole.PLANNER else 0
        )
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            strong_model_tokens=strong,
            elapsed_ms=elapsed_ms,
        )
