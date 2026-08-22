"""Native Anthropic Messages outbound adapter for the v0.0.2 text subset."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import httpx

from prp_runtime.domain.enums import ModelRole, ToolCallStatus
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.models import AgentToolCall, AgentTurn, Usage
from prp_runtime.json_support import strict_json_loads
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderProtocol,
    ProviderRequest,
    ProviderResponse,
)

__all__ = ["MAX_ERROR_BODY_CHARS", "AnthropicMessagesProvider", "AnthropicProvider"]

MAX_RESPONSE_ID_CHARS = 256
MAX_ERROR_BODY_CHARS = 500


class AnthropicMessagesProvider:
    """Calls the native Anthropic Messages endpoint without streaming."""

    def __init__(
        self, profile: ModelProfile, *, client: httpx.AsyncClient | None = None
    ) -> None:
        if profile.protocol is not ProviderProtocol.ANTHROPIC_MESSAGES:
            raise ValueError("AnthropicMessagesProvider requires ANTHROPIC_MESSAGES")
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
        base_url = self._profile.base_url.rstrip("/")
        if base_url.endswith("/messages"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/messages"
        return f"{base_url}/v1/messages"

    async def aclose(self) -> None:
        """Close only a client owned by this adapter."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Perform one non-streaming Messages completion."""

        self._validate_request(request)
        client = self._ensure_client()
        started = perf_counter()
        try:
            response = await client.post(
                self.endpoint,
                json=self._build_payload(request),
                headers=self._headers(),
                timeout=httpx.Timeout(request.timeout_seconds),
            )
        except httpx.TimeoutException as error:
            raise ProviderError(
                f"upstream {self._profile.alias} timed out",
                code=ErrorCode.PROVIDER_TIMEOUT,
            ) from error
        except httpx.HTTPError as error:
            raise ProviderError(
                f"upstream {self._profile.alias} is unreachable",
                code=ErrorCode.PROVIDER_UNAVAILABLE,
            ) from error
        elapsed_ms = max(int((perf_counter() - started) * 1000), 0)
        if response.status_code >= 400:
            raise self._status_error(response)
        return self._parse(response, elapsed_ms)

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(trust_env=False)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "anthropic-version": self._profile.anthropic_version or "",
        }
        if self._profile.api_key is not None:
            headers["x-api-key"] = self._profile.api_key.get_secret_value()
        return headers

    def _validate_request(self, request: ProviderRequest) -> None:
        if request.alias != self._profile.alias or request.model != self._profile.model:
            raise ProviderError(
                "request does not match the configured Anthropic profile",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        if request.max_output_tokens > self._profile.max_output_tokens:
            raise ProviderError(
                "request exceeds the configured Anthropic profile",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )

    def _build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens,
            "messages": self._messages_payload(request),
            "stream": False,
        }
        if request.instructions is not None:
            payload["system"] = request.instructions
        if request.tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]
        return payload

    @staticmethod
    def _messages_payload(request: ProviderRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for item in request.history:
            if isinstance(item, AgentTurn):
                if item.text is not None:
                    messages.append({"role": "assistant", "content": item.text})
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": call.call_id,
                                    "name": call.tool_name,
                                    "input": call.arguments,
                                }
                                for call in item.tool_calls
                            ],
                        }
                    )
            else:
                output = item.output
                if not output and item.result is not None:
                    output = json.dumps(
                        item.result,
                        ensure_ascii=True,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": item.call_id,
                                "content": output,
                                "is_error": item.status is not ToolCallStatus.SUCCEEDED,
                            }
                        ],
                    }
                )
        messages.append({"role": "user", "content": request.input})
        return messages

    def _parse(self, response: httpx.Response, elapsed_ms: int) -> ProviderResponse:
        try:
            document = strict_json_loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise self._invalid_response("the body is not JSON") from error
        if not isinstance(document, dict):
            raise self._invalid_response("the body is not a JSON object")
        request_id = self._parse_request_id(document.get("id"))
        raw_content = document.get("content")
        if isinstance(raw_content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in raw_content
        ):
            if any(
                isinstance(block, dict) and block.get("type") == "text"
                for block in raw_content
            ):
                raise self._invalid_response("tool_use cannot be mixed with text content")
            tool_calls = self._parse_tool_calls(raw_content)
            return ProviderResponse(
                text=None,
                tool_calls=tool_calls,
                usage=self._parse_usage(document.get("usage"), elapsed_ms),
                finish_reason=FinishReason.TOOL_CALLS,
                provider_request_id=request_id,
            )
        text = self._parse_text(raw_content)
        try:
            return ProviderResponse(
                text=text,
                usage=self._parse_usage(document.get("usage"), elapsed_ms),
                finish_reason=self._finish_reason(document.get("stop_reason")),
                provider_request_id=request_id,
            )
        except ValueError as error:
            raise self._invalid_response("text response violates the native contract") from error

    def _parse_tool_calls(self, raw: list[object]) -> tuple[AgentToolCall, ...]:
        if not raw:
            raise self._invalid_response("tool_use content is empty")
        calls: list[AgentToolCall] = []
        seen_ids: set[str] = set()
        for block in raw:
            if isinstance(block, dict) and block.get("type") in {
                "thinking",
                "redacted_thinking",
            }:
                self._validate_thinking_companion(block)
                continue
            if not isinstance(block, dict) or set(block) - {"type", "id", "name", "input"}:
                raise self._invalid_response("tool_use block has an unsupported shape")
            if block.get("type") != "tool_use":
                raise self._invalid_response("tool_use cannot be mixed with other content")
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input")
            if not isinstance(call_id, str) or not call_id.strip():
                raise self._invalid_response("tool_use id is missing")
            if not isinstance(name, str) or not name.strip():
                raise self._invalid_response("tool_use name is missing")
            if not isinstance(arguments, dict):
                raise self._invalid_response("tool_use input must be a JSON object")
            try:
                call = AgentToolCall(call_id=call_id, tool_name=name, arguments=arguments)
            except ValueError as error:
                raise self._invalid_response("tool_use violates the native contract") from error
            if call.call_id in seen_ids:
                raise self._invalid_response("tool_use ids must be unique")
            seen_ids.add(call.call_id)
            calls.append(call)
        if not calls:
            raise self._invalid_response("tool_use content is empty")
        return tuple(calls)

    def _validate_thinking_companion(self, block: dict[str, Any]) -> None:
        if block.get("type") == "thinking":
            if set(block) != {"type", "thinking", "signature"}:
                raise self._invalid_response("thinking companion has an unsupported shape")
            if not isinstance(block.get("thinking"), str) or not isinstance(
                block.get("signature"), str
            ):
                raise self._invalid_response("thinking companion is malformed")
            return
        if set(block) != {"type", "data"} or not isinstance(block.get("data"), str):
            raise self._invalid_response("redacted thinking companion is malformed")

    def _parse_text(self, raw: object) -> str:
        if not isinstance(raw, list) or not raw:
            raise self._invalid_response("content must be a non-empty list")
        text_parts: list[str] = []
        for block in raw:
            if not isinstance(block, dict) or set(block) - {"type", "text"}:
                raise self._invalid_response("content block contains unsupported fields")
            if block.get("type") != "text":
                raise self._invalid_response("content contains an unsupported block type")
            text = block.get("text")
            if not isinstance(text, str) or not text.strip():
                raise self._invalid_response("content text block is empty")
            text_parts.append(text)
        return "".join(text_parts)

    def _parse_request_id(self, raw: object) -> str | None:
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_RESPONSE_ID_CHARS:
            raise self._invalid_response("response id is malformed")
        return raw

    def _finish_reason(self, raw: object) -> FinishReason:
        if raw in (None, "end_turn", "stop_sequence"):
            return FinishReason.STOP
        if raw == "max_tokens":
            return FinishReason.LENGTH
        if raw == "tool_use":
            return FinishReason.TOOL_CALLS
        raise self._invalid_response("stop_reason is unsupported")

    def _parse_usage(self, raw: object, elapsed_ms: int) -> Usage | None:
        if not isinstance(raw, dict):
            return None
        input_tokens = self._non_negative_int(raw.get("input_tokens"))
        output_tokens = self._non_negative_int(raw.get("output_tokens"))
        if input_tokens is None or output_tokens is None:
            return None
        strong = (
            input_tokens + output_tokens if self._profile.role is ModelRole.PLANNER else 0
        )
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            strong_model_tokens=strong,
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _non_negative_int(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

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
        elif status in (408, 504):
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


# Alias for backward compatibility
AnthropicProvider = AnthropicMessagesProvider
