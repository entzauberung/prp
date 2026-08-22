"""Targeted tests for the native Responses text and structured subset."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx

from prp_runtime.domain.enums import ModelRole, ToolCallStatus
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.models import AgentToolCall, AgentToolResult, AgentTurn
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderProtocol,
    ProviderRequest,
    ProviderToolDescriptor,
)
from prp_runtime.providers.openai_responses import OpenAIResponsesProvider

BASE_URL = "https://models.internal/v1"
ENDPOINT = f"{BASE_URL}/responses"
TEST_KEY = "test-key"


def profile(**overrides: object) -> ModelProfile:
    data: dict[str, object] = {
        "alias": "worker",
        "provider": "responses-provider",
        "model": "responses-model",
        "role": ModelRole.WORKER,
        "base_url": BASE_URL,
        "api_key": TEST_KEY,
        "protocol": ProviderProtocol.OPENAI_RESPONSES,
        "context_window_tokens": 32_000,
        "max_output_tokens": 4_000,
    }
    data.update(overrides)
    return ModelProfile(**data)  # type: ignore[arg-type]


def request_for(adapter: OpenAIResponsesProvider, **kwargs: object) -> ProviderRequest:
    return ProviderRequest.for_profile(adapter.profile, input="summarise", **kwargs)  # type: ignore[arg-type]


def text_body(text: str = "the answer", **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "resp-123",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }
    body.update(overrides)
    return body


@pytest_asyncio.fixture
async def provider() -> AsyncIterator[OpenAIResponsesProvider]:
    adapter = OpenAIResponsesProvider(profile())
    try:
        yield adapter
    finally:
        await adapter.aclose()


def test_request_endpoint_is_appended_once() -> None:
    assert OpenAIResponsesProvider(profile()).endpoint == ENDPOINT
    assert OpenAIResponsesProvider(profile(base_url=ENDPOINT)).endpoint == ENDPOINT


@pytest.mark.asyncio
async def test_request_payload_uses_native_responses_shape(
    provider: OpenAIResponsesProvider, mocked_http: respx.MockRouter
) -> None:
    route = mocked_http.post(ENDPOINT).respond(200, json=text_body())
    response = await provider.complete(
        request_for(provider, instructions="be terse", max_output_tokens=64)
    )

    assert response.text == "the answer"
    assert response.finish_reason is FinishReason.STOP
    assert response.provider_request_id == "resp-123"
    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "model": "responses-model",
        "input": "summarise",
        "instructions": "be terse",
        "max_output_tokens": 64,
        "stream": False,
    }
    assert route.calls.last.request.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert TEST_KEY not in route.calls.last.request.content.decode("utf-8")


@pytest.mark.asyncio
async def test_text_output_text_field_is_normalised(
    provider: OpenAIResponsesProvider, mocked_http: respx.MockRouter
) -> None:
    mocked_http.post(ENDPOINT).respond(
        200,
        json={"id": "resp-direct", "output_text": "direct answer", "status": "completed"},
    )
    response = await provider.complete(request_for(provider))
    assert response.text == "direct answer"
    assert response.usage is None


@pytest.mark.asyncio
async def test_structured_request_uses_strict_json_schema_format(
    provider: OpenAIResponsesProvider, mocked_http: respx.MockRouter
) -> None:
    provider._profile = profile(supports_structured_output=True)  # type: ignore[misc]
    route = mocked_http.post(ENDPOINT).respond(
        200, json={"id": "resp-json", "output_text": '{"answer":"ok"}'}
    )
    response = await provider.complete(
        request_for(
            provider,
            json_schema=json.dumps(
                {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                }
            ),
        )
    )
    assert response.text == '{"answer":"ok"}'
    sent = json.loads(route.calls.last.request.content)
    assert sent["text"] == {
        "format": {
            "type": "json_schema",
            "name": "output",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        }
    }


@pytest.mark.asyncio
async def test_tool_request_and_result_history_map_to_responses_items(
    provider: OpenAIResponsesProvider, mocked_http: respx.MockRouter
) -> None:
    route = mocked_http.post(ENDPOINT).respond(200, json=text_body())
    call = AgentToolCall(
        call_id="call-read", tool_name="read_file", arguments={"path": "README.md"}
    )
    tool = ProviderToolDescriptor(name="read_file", input_schema={"type": "object"})
    await provider.complete(
        request_for(
            provider,
            history=(
                AgentTurn(tool_calls=(call,)),
                AgentToolResult(
                    call_id="call-read",
                    status=ToolCallStatus.SUCCEEDED,
                    result={"ok": True},
                ),
            ),
            tools=(tool,),
        )
    )
    sent = json.loads(route.calls.last.request.content)
    assert sent["input"] == [
        {
            "type": "function_call",
            "call_id": "call-read",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        },
        {"type": "function_call_output", "call_id": "call-read", "output": '{"ok":true}'},
        {"role": "user", "content": "summarise"},
    ]
    assert sent["tools"] == [
        {
            "type": "function",
            "name": "read_file",
            "description": "",
            "parameters": {"type": "object"},
            "strict": True,
        }
    ]
    assert sent["tool_choice"] == "auto"


def test_tool_response_function_call_is_normalised() -> None:
    adapter = OpenAIResponsesProvider(profile())
    response = adapter._parse(
        httpx.Response(
            200,
            json={
                "id": "resp-tool",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc-item-1",
                        "call_id": "call-search",
                        "name": "search_text",
                        "arguments": '{"pattern":"Provider"}',
                    }
                ],
            },
        ),
        elapsed_ms=3,
    )
    assert response.text is None
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls[0].call_id == "call-search"
    assert response.tool_calls[0].arguments == {"pattern": "Provider"}


def test_tool_response_accepts_validated_reasoning_companion() -> None:
    adapter = OpenAIResponsesProvider(profile())
    response = adapter._parse(
        httpx.Response(
            200,
            json={
                "id": "resp-tool-reasoning",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs-1",
                        "status": "completed",
                        "summary": [
                            {"type": "summary_text", "text": "private summary"}
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-search",
                        "name": "search_text",
                        "arguments": '{"pattern":"Provider"}',
                    },
                ],
            },
        ),
        elapsed_ms=3,
    )

    assert response.text is None
    assert response.tool_calls[0].tool_name == "search_text"


@pytest.mark.parametrize(
    "companion",
    [
        {"type": "message", "role": "assistant", "content": []},
        {"type": "reasoning", "summary": [], "unexpected": "private"},
        {"type": "reasoning", "summary": "private"},
    ],
)
def test_tool_response_rejects_unvalidated_companions_without_echo(
    companion: dict[str, object],
) -> None:
    adapter = OpenAIResponsesProvider(profile())
    body = {
        "output": [
            companion,
            {
                "type": "function_call",
                "call_id": "call-search",
                "name": "search_text",
                "arguments": "{}",
            },
        ]
    }

    with pytest.raises(ProviderError) as excinfo:
        adapter._parse(httpx.Response(200, json=body), elapsed_ms=1)

    assert "private" not in str(excinfo.value)


def test_tool_response_duplicate_ids_and_non_object_arguments_fail_closed() -> None:
    adapter = OpenAIResponsesProvider(profile())
    duplicate = {
        "type": "function_call",
        "call_id": "call-same",
        "name": "search_text",
        "arguments": "{}",
    }
    with pytest.raises(ProviderError, match="unique"):
        adapter._parse(
            httpx.Response(
                200,
                json={"output": [duplicate, dict(duplicate)], "status": "completed"},
            ),
            elapsed_ms=1,
        )
    with pytest.raises(ProviderError, match="JSON object"):
        adapter._parse(
            httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-array",
                            "name": "search_text",
                            "arguments": "[]",
                        }
                    ]
                },
            ),
            elapsed_ms=1,
        )


def test_usage_normalises_valid_counts_and_keeps_invalid_counts_unknown() -> None:
    adapter = OpenAIResponsesProvider(profile())
    valid = adapter._parse(
        httpx.Response(
            200,
            json={**text_body(), "usage": {"input_tokens": 11, "output_tokens": 7}},
        ),
        elapsed_ms=12,
    )
    assert valid.usage is not None
    assert (valid.usage.input_tokens, valid.usage.output_tokens) == (11, 7)
    assert valid.usage.elapsed_ms == 12
    for usage in (
        {"input_tokens": True, "output_tokens": 7},
        {"input_tokens": 11, "output_tokens": -1},
    ):
        unknown = adapter._parse(
            httpx.Response(200, json={**text_body(), "usage": usage}), elapsed_ms=1
        )
        assert unknown.usage is None


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, ErrorCode.PROVIDER_AUTH_FAILED),
        (429, ErrorCode.PROVIDER_RATE_LIMITED),
        (500, ErrorCode.PROVIDER_UNAVAILABLE),
    ],
)
@pytest.mark.asyncio
async def test_error_statuses_are_classified_without_retry_or_secret_echo(
    provider: OpenAIResponsesProvider,
    mocked_http: respx.MockRouter,
    status: int,
    code: ErrorCode,
) -> None:
    route = mocked_http.post(ENDPOINT).respond(
        status,
        text=f"Bearer {TEST_KEY} private upstream body",
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(request_for(provider))
    assert excinfo.value.code is code
    assert route.call_count == 1
    assert TEST_KEY not in str(excinfo.value)
    assert len(str(excinfo.value)) < 600


@pytest.mark.asyncio
async def test_timeout_is_classified_without_internal_retry() -> None:
    async def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout")

    client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    adapter = OpenAIResponsesProvider(profile(), client=client)
    with pytest.raises(ProviderError) as excinfo:
        await adapter.complete(request_for(adapter))
    assert excinfo.value.code is ErrorCode.PROVIDER_TIMEOUT
    await client.aclose()


@pytest.mark.asyncio
async def test_cancelled_error_propagates_unchanged() -> None:
    async def cancelled(_: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    client = httpx.AsyncClient(transport=httpx.MockTransport(cancelled))
    adapter = OpenAIResponsesProvider(profile(), client=client)
    with pytest.raises(asyncio.CancelledError):
        await adapter.complete(request_for(adapter))
    await client.aclose()


class CountingClient(httpx.AsyncClient):
    def __init__(self) -> None:
        super().__init__(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        await super().aclose()


@pytest.mark.asyncio
async def test_owned_client_closes_exactly_once_and_injected_client_is_untouched() -> None:
    owned_client = CountingClient()
    owned = OpenAIResponsesProvider(profile(), client=owned_client)
    owned._owns_client = True
    await owned.aclose()
    await owned.aclose()
    assert owned_client.close_calls == 1

    injected_client = CountingClient()
    injected = OpenAIResponsesProvider(profile(), client=injected_client)
    await injected.aclose()
    assert injected_client.close_calls == 0
    await injected_client.aclose()


@pytest.mark.asyncio
async def test_profile_mismatch_fails_before_http_and_owned_client_disables_proxy() -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    adapter = OpenAIResponsesProvider(profile(), client=client)
    request = ProviderRequest(
        alias="other",
        model="responses-model",
        input="summarise",
        max_output_tokens=64,
        timeout_seconds=5.0,
    )
    with pytest.raises(ProviderError, match="configured Responses profile"):
        await adapter.complete(request)
    assert calls == 0
    assert getattr(client, "_trust_env") is False
    await client.aclose()


@pytest.mark.parametrize(
    "body",
    [
        {"id": "empty", "output": [], "status": "completed"},
        {
            "id": "mixed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "ok"},
                        {"type": "refusal", "refusal": "no"},
                    ],
                }
            ],
        },
        {
            "id": "unknown",
            "output": [{"type": "reasoning", "summary": []}],
        },
    ],
)
def test_text_unknown_empty_or_mixed_blocks_fail_closed(body: dict[str, Any]) -> None:
    adapter = OpenAIResponsesProvider(profile())
    with pytest.raises(ProviderError) as excinfo:
        adapter._parse(__import__("httpx").Response(200, json=body), elapsed_ms=1)
    assert excinfo.value.code is ErrorCode.PROVIDER_INVALID_RESPONSE
    assert "refusal" not in str(excinfo.value)
