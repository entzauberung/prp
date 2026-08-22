"""Native OpenAI Responses outbound adapter for the v0.0.2 text subset."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import httpx

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.models import AgentToolCall, AgentToolResult, AgentTurn, Usage
from prp_runtime.json_support import strict_json_loads
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderProtocol,
    ProviderRequest,
    ProviderResponse,
)

__all__ = ["MAX_ERROR_BODY_CHARS", "OpenAIResponsesProvider"]

MAX_RESPONSE_ID_CHARS = 256
MAX_ERROR_BODY_CHARS = 500


class OpenAIResponsesProvider:
    """Calls the native Responses endpoint without streaming or tool blocks."""

    def __init__(
        self, profile: ModelProfile, *, client: httpx.AsyncClient | None = None
    ) -> None:
        if profile.protocol is not ProviderProtocol.OPENAI_RESPONSES:
            raise ValueError("OpenAIResponsesProvider requires OPENAI_RESPONSES")
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
        return base_url if base_url.endswith("/responses") else f"{base_url}/responses"

    async def aclose(self) -> None:
        """Close only a client owned by this adapter."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Perform one non-streaming Responses completion."""

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
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._profile.api_key is not None:
            headers["Authorization"] = f"Bearer {self._profile.api_key.get_secret_value()}"
        return headers

    def _validate_request(self, request: ProviderRequest) -> None:
        if request.alias != self._profile.alias or request.model != self._profile.model:
            raise ProviderError(
                "request does not match the configured Responses profile",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        if request.max_output_tokens > self._profile.max_output_tokens:
            raise ProviderError(
                "request exceeds the configured Responses profile",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        if request.json_schema is not None and not self._profile.supports_structured_output:
            raise ProviderError(
                "structured output is not enabled for this Responses profile",
                code=ErrorCode.INVALID_REQUEST,
            )

    def _build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": self._input_payload(request),
            "max_output_tokens": request.max_output_tokens,
            "stream": False,
        }
        if request.instructions is not None:
            payload["instructions"] = request.instructions
        if request.json_schema is not None:
            try:
                schema = strict_json_loads(request.json_schema)
            except ValueError as error:
                raise ProviderError(
                    "structured output schema is not standard JSON",
                    code=ErrorCode.INVALID_REQUEST,
                ) from error
            if not isinstance(schema, dict):
                raise ProviderError(
                    "structured output schema must be a JSON object",
                    code=ErrorCode.INVALID_REQUEST,
                )
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "output",
                    "strict": True,
                    "schema": schema,
                }
            }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": True,
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = "auto"
        return payload

    def _input_payload(self, request: ProviderRequest) -> str | list[dict[str, Any]]:
        if not request.history:
            return request.input
        items: list[dict[str, Any]] = []
        for item in request.history:
            if isinstance(item, AgentTurn):
                if item.text is not None:
                    items.append({"role": "assistant", "content": item.text})
                else:
                    items.extend(self._function_call_item(call) for call in item.tool_calls)
            else:
                items.append(self._function_call_output_item(item))
        items.append({"role": "user", "content": request.input})
        return items

    @staticmethod
    def _function_call_item(call: AgentToolCall) -> dict[str, Any]:
        return {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.tool_name,
            "arguments": json.dumps(
                call.arguments,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    @staticmethod
    def _function_call_output_item(result: AgentToolResult) -> dict[str, Any]:
        output: object = result.output
        if not output and result.result is not None:
            output = json.dumps(
                result.result,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return {"type": "function_call_output", "call_id": result.call_id, "output": output}

    def _parse(self, response: httpx.Response, elapsed_ms: int) -> ProviderResponse:
        try:
            document = strict_json_loads(response.content.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise self._invalid_response("the body is not JSON") from error
        if not isinstance(document, dict):
            raise self._invalid_response("the body is not a JSON object")
        request_id = self._parse_request_id(document.get("id"))
        raw_output = document.get("output")
        if isinstance(raw_output, list) and any(
            isinstance(block, dict) and block.get("type") == "function_call"
            for block in raw_output
        ):
            if "output_text" in document:
                raise self._invalid_response("tool calls cannot be combined with output text")
            tool_calls = self._parse_tool_calls(raw_output)
            return ProviderResponse(
                text=None,
                tool_calls=tool_calls,
                usage=self._parse_usage(document.get("usage"), elapsed_ms),
                finish_reason=FinishReason.TOOL_CALLS,
                provider_request_id=request_id,
            )
        text = self._parse_text(document)
        try:
            return ProviderResponse(
                text=text,
                usage=self._parse_usage(document.get("usage"), elapsed_ms),
                finish_reason=self._finish_reason(document.get("status")),
                provider_request_id=request_id,
            )
        except ValueError as error:
            raise self._invalid_response("text response violates the native contract") from error

    def _parse_text(self, document: dict[str, Any]) -> str:
        marker = object()
        raw_output_text = document.get("output_text", marker)
        raw_output = document.get("output", marker)
        direct_text: str | None = None
        block_text: str | None = None
        if raw_output_text is not marker:
            if not isinstance(raw_output_text, str) or not raw_output_text.strip():
                raise self._invalid_response("output_text must be non-empty text")
            direct_text = raw_output_text
        if raw_output is not marker:
            block_text = self._parse_output_blocks(raw_output)
        if direct_text is None and block_text is None:
            raise self._invalid_response("response has no supported text output")
        if direct_text is not None and block_text is not None and direct_text != block_text:
            raise self._invalid_response("output_text disagrees with output message blocks")
        return direct_text if direct_text is not None else block_text  # type: ignore[return-value]

    def _parse_tool_calls(self, raw: list[object]) -> tuple[AgentToolCall, ...]:
        if not raw:
            raise self._invalid_response("function_call output is empty")
        calls: list[AgentToolCall] = []
        seen_ids: set[str] = set()
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                self._validate_reasoning_companion(block)
                continue
            if not isinstance(block, dict) or set(block) - {
                "type",
                "id",
                "call_id",
                "name",
                "arguments",
                "status",
            }:
                raise self._invalid_response("function_call block has an unsupported shape")
            if block.get("type") != "function_call":
                raise self._invalid_response("tool calls cannot be mixed with other output blocks")
            call_id = block.get("call_id")
            name = block.get("name")
            arguments_text = block.get("arguments")
            if not isinstance(call_id, str) or not call_id.strip():
                raise self._invalid_response("function_call call_id is missing")
            if not isinstance(name, str) or not name.strip():
                raise self._invalid_response("function_call name is missing")
            if not isinstance(arguments_text, str):
                raise self._invalid_response("function_call arguments must be JSON text")
            try:
                arguments = strict_json_loads(arguments_text)
            except ValueError as error:
                raise self._invalid_response(
                    "function_call arguments are not standard JSON"
                ) from error
            if not isinstance(arguments, dict):
                raise self._invalid_response("function_call arguments must be a JSON object")
            try:
                call = AgentToolCall(call_id=call_id, tool_name=name, arguments=arguments)
            except ValueError as error:
                raise self._invalid_response(
                    "function_call violates the native contract"
                ) from error
            if call.call_id in seen_ids:
                raise self._invalid_response("function_call ids must be unique")
            seen_ids.add(call.call_id)
            calls.append(call)
        if not calls:
            raise self._invalid_response("function_call output is empty")
        return tuple(calls)

    def _validate_reasoning_companion(self, block: dict[str, Any]) -> None:
        if set(block) - {
            "type",
            "id",
            "status",
            "summary",
            "content",
            "encrypted_content",
        }:
            raise self._invalid_response("reasoning companion has an unsupported shape")
        item_id = block.get("id")
        if item_id is not None and (
            not isinstance(item_id, str)
            or not item_id.strip()
            or len(item_id) > MAX_RESPONSE_ID_CHARS
        ):
            raise self._invalid_response("reasoning companion id is malformed")
        status = block.get("status")
        if status not in (None, "in_progress", "completed", "incomplete"):
            raise self._invalid_response("reasoning companion status is unsupported")
        self._validate_reasoning_text_blocks(
            block.get("summary"), expected_type="summary_text", field_name="summary"
        )
        self._validate_reasoning_text_blocks(
            block.get("content"), expected_type="reasoning_text", field_name="content"
        )
        encrypted = block.get("encrypted_content")
        if encrypted is not None and not isinstance(encrypted, str):
            raise self._invalid_response("reasoning companion encryption is malformed")

    def _validate_reasoning_text_blocks(
        self,
        raw: object,
        *,
        expected_type: str,
        field_name: str,
    ) -> None:
        if raw is None:
            return
        if not isinstance(raw, list):
            raise self._invalid_response(f"reasoning companion {field_name} is malformed")
        for item in raw:
            if (
                not isinstance(item, dict)
                or set(item) != {"type", "text"}
                or item.get("type") != expected_type
                or not isinstance(item.get("text"), str)
            ):
                raise self._invalid_response(
                    f"reasoning companion {field_name} is malformed"
                )

    def _parse_output_blocks(self, raw: object) -> str:
        if not isinstance(raw, list) or not raw:
            raise self._invalid_response("output must contain at least one message block")

        # Find the final answer message (may have multiple blocks for reasoning models)
        final_message = None
        for msg in raw:
            if not isinstance(msg, dict):
                raise self._invalid_response("output message block is not an object")
            if msg.get("phase") == "final_answer":
                final_message = msg
                break

        # Fall back to the last message if no explicit final_answer phase
        if final_message is None:
            final_message = raw[-1]

        message = final_message
        if not isinstance(message, dict):
            raise self._invalid_response("output message block is not an object")
        extra_fields = set(message) - {
            "type", "id", "role", "status", "content", "phase",
            "encrypted_content", "summary"  # DeepSeek extensions
        }
        if extra_fields:
            raise self._invalid_response(
                "output message block contains unsupported fields: "
                f"{extra_fields}"
            )
        msg_type = message.get("type")
        if msg_type is not None and msg_type not in ("message", "reasoning"):
            raise self._invalid_response(f"output contains an unsupported block type: {msg_type}")
        if "role" in message and message["role"] != "assistant":
            raise self._invalid_response("output message block is not from the assistant")
        content = message.get("content")
        if not isinstance(content, list) or not content:
            raise self._invalid_response("output message has no content blocks")
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or set(block) - {
                "type",
                "text",
                "annotations",
                "logprobs",
            }:
                raise self._invalid_response("output text block contains unsupported fields")
            block_type = block.get("type")
            if block_type not in ("output_text", "reasoning_text"):
                raise self._invalid_response(
                    "output contains a mixed or unsupported text block: "
                    f"{block_type}"
                )
            value = block.get("text")
            if not isinstance(value, str) or not value.strip():
                raise self._invalid_response("output text block is empty")
            text_parts.append(value)
        return "".join(text_parts)

    def _parse_request_id(self, raw: object) -> str | None:
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_RESPONSE_ID_CHARS:
            raise self._invalid_response("response id is malformed")
        return raw

    def _finish_reason(self, raw: object) -> FinishReason:
        if raw in (None, "completed"):
            return FinishReason.STOP
        if raw == "incomplete":
            return FinishReason.LENGTH
        if isinstance(raw, str) and raw == "failed":
            return FinishReason.OTHER
        raise self._invalid_response("response status is unsupported")

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
