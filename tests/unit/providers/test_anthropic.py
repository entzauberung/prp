"""Targeted tests for the native Anthropic Messages text contract."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.providers.anthropic import AnthropicMessagesProvider
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderProtocol,
    ProviderRequest,
)

BASE_URL = "https://models.internal/anthropic"
ENDPOINT = f"{BASE_URL}/v1/messages"
TEST_KEY = "test-key"
VERSION = "2023-06-01"


def profile(**overrides: object) -> ModelProfile:
    data: dict[str, object] = {
        "alias": "worker",
        "provider": "anthropic-provider",
        "model": "claude-model",
        "role": ModelRole.WORKER,
        "base_url": BASE_URL,
        "api_key": TEST_KEY,
        "protocol": ProviderProtocol.ANTHROPIC_MESSAGES,
        "anthropic_version": VERSION,
        "context_window_tokens": 32_000,
        "max_output_tokens": 4_000,
    }
    data.update(overrides)
    return ModelProfile(**data)  # type: ignore[arg-type]


def request_for(adapter: AnthropicMessagesProvider, **kwargs: object) -> ProviderRequest:
    return ProviderRequest.for_profile(adapter.profile, input="summarise", **kwargs)  # type: ignore[arg-type]


def message_body(text: str = "the answer", **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg-123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
    }
    body.update(overrides)
    return body


@pytest_asyncio.fixture
async def provider() -> AsyncIterator[AnthropicMessagesProvider]:
    adapter = AnthropicMessagesProvider(profile())
    try:
        yield adapter
    finally:
        await adapter.aclose()


def test_request_endpoint_is_appended_once() -> None:
    assert AnthropicMessagesProvider(profile()).endpoint == ENDPOINT
    assert AnthropicMessagesProvider(profile(base_url=f"{BASE_URL}/v1")).endpoint == ENDPOINT
    assert AnthropicMessagesProvider(profile(base_url=ENDPOINT)).endpoint == ENDPOINT


@pytest.mark.asyncio
async def test_request_payload_and_headers_use_native_anthropic_shape(
    provider: AnthropicMessagesProvider, mocked_http: respx.MockRouter
) -> None:
    route = mocked_http.post(ENDPOINT).respond(200, json=message_body())
    response = await provider.complete(
        request_for(provider, instructions="be terse", max_output_tokens=64)
    )

    assert response.text == "the answer"
    assert response.finish_reason is FinishReason.STOP
    assert response.provider_request_id == "msg-123"
    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "model": "claude-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "summarise"}],
        "stream": False,
        "system": "be terse",
    }
    headers = route.calls.last.request.headers
    assert headers["x-api-key"] == TEST_KEY
    assert headers["anthropic-version"] == VERSION
    assert "authorization" not in headers
    assert TEST_KEY not in route.calls.last.request.content.decode("utf-8")


@pytest.mark.asyncio
async def test_text_content_and_stop_reason_are_normalised(
    provider: AnthropicMessagesProvider, mocked_http: respx.MockRouter
) -> None:
    mocked_http.post(ENDPOINT).respond(
        200,
        json=message_body(
            text="limited answer",
            stop_reason="max_tokens",
        ),
    )
    response = await provider.complete(request_for(provider))
    assert response.text == "limited answer"
    assert response.finish_reason is FinishReason.LENGTH
    assert response.usage is None


@pytest.mark.asyncio
async def test_tool_request_and_result_history_map_to_anthropic_blocks(
    provider: AnthropicMessagesProvider, mocked_http: respx.MockRouter
) -> None:
    route = mocked_http.post(ENDPOINT).respond(200, json=message_body())
    call = {
        "kind": "turn",
        "text": None,
        "tool_calls": [
            {
                "kind": "tool_call",
                "call_id": "call-read",
                "tool_name": "read_file",
                "arguments": {"path": "README.md"},
            }
        ],
    }
    result = {
        "kind": "tool_result",
        "call_id": "call-read",
        "status": "SUCCEEDED",
        "result": {"ok": True},
    }
    tool = {"name": "read_file", "description": "Read a file", "input_schema": {"type": "object"}}
    request = request_for(provider, history=(call, result), tools=(tool,))
    await provider.complete(request)
    sent = json.loads(route.calls.last.request.content)
    assert sent["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-read",
                    "name": "read_file",
                    "input": {"path": "README.md"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-read",
                    "content": '{"ok":true}',
                    "is_error": False,
                }
            ],
        },
        {"role": "user", "content": "summarise"},
    ]
    assert sent["tools"] == [tool]


def test_tool_response_is_normalised_and_duplicate_or_invalid_input_rejected() -> None:
    adapter = AnthropicMessagesProvider(profile())
    response = adapter._parse(
        httpx.Response(
            200,
            json={
                "id": "msg-tool",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-search",
                        "name": "search_text",
                        "input": {"pattern": "Provider"},
                    }
                ],
                "stop_reason": "tool_use",
            },
        ),
        elapsed_ms=1,
    )
    assert response.text is None
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls[0].call_id == "call-search"
    with pytest.raises(ProviderError, match="unique"):
        adapter._parse(
            httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "tool_use", "id": "same", "name": "search_text", "input": {}},
                        {"type": "tool_use", "id": "same", "name": "search_text", "input": {}},
                    ]
                },
            ),
            elapsed_ms=1,
        )
    with pytest.raises(ProviderError, match="JSON object"):
        adapter._parse(
            httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "tool_use", "id": "array", "name": "search_text", "input": []}
                    ]
                },
            ),
            elapsed_ms=1,
        )


def test_tool_use_and_text_cannot_be_mixed() -> None:
    adapter = AnthropicMessagesProvider(profile())
    with pytest.raises(ProviderError, match="mixed"):
        adapter._parse(
            httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "partial"},
                        {"type": "tool_use", "id": "call", "name": "read_file", "input": {}},
                    ]
                },
            ),
            elapsed_ms=1,
        )


def test_usage_normalises_valid_counts_and_keeps_invalid_counts_unknown() -> None:
    adapter = AnthropicMessagesProvider(profile())
    valid = adapter._parse(
        httpx.Response(
            200,
            json={**message_body(), "usage": {"input_tokens": 11, "output_tokens": 7}},
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
            httpx.Response(200, json={**message_body(), "usage": usage}), elapsed_ms=1
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
    provider: AnthropicMessagesProvider,
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
async def test_timeout_and_cancel_are_normalised_without_swallowing_cancel() -> None:
    async def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout")

    timeout_client = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    timeout_adapter = AnthropicMessagesProvider(profile(), client=timeout_client)
    with pytest.raises(ProviderError) as excinfo:
        await timeout_adapter.complete(request_for(timeout_adapter))
    assert excinfo.value.code is ErrorCode.PROVIDER_TIMEOUT
    await timeout_client.aclose()

    async def cancelled(_: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    cancel_client = httpx.AsyncClient(transport=httpx.MockTransport(cancelled))
    cancel_adapter = AnthropicMessagesProvider(profile(), client=cancel_client)
    with pytest.raises(asyncio.CancelledError):
        await cancel_adapter.complete(request_for(cancel_adapter))
    await cancel_client.aclose()


class CountingClient(httpx.AsyncClient):
    def __init__(self) -> None:
        super().__init__(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        await super().aclose()


@pytest.mark.asyncio
async def test_owned_client_closes_once_and_injected_client_is_untouched() -> None:
    owned_client = CountingClient()
    owned = AnthropicMessagesProvider(profile(), client=owned_client)
    owned._owns_client = True
    await owned.aclose()
    await owned.aclose()
    assert owned_client.close_calls == 1

    injected_client = CountingClient()
    injected = AnthropicMessagesProvider(profile(), client=injected_client)
    await injected.aclose()
    assert injected_client.close_calls == 0
    await injected_client.aclose()


def test_explicit_base_and_version_are_host_independent() -> None:
    deepseek = AnthropicMessagesProvider(
        profile(base_url="https://deepseek.example/anthropic", anthropic_version="2023-06-01")
    )
    claude = AnthropicMessagesProvider(
        profile(base_url="https://claude.example/v1", anthropic_version="2023-06-01")
    )
    assert deepseek.endpoint == "https://deepseek.example/anthropic/v1/messages"
    assert claude.endpoint == "https://claude.example/v1/messages"
    assert deepseek._headers()["anthropic-version"] == "2023-06-01"
    assert claude._headers()["anthropic-version"] == "2023-06-01"


def test_owned_client_disables_ambient_proxy() -> None:
    adapter = AnthropicMessagesProvider(profile())
    client = adapter._ensure_client()
    assert getattr(client, "_trust_env") is False


def test_text_history_maps_to_ordered_anthropic_messages() -> None:
    adapter = AnthropicMessagesProvider(profile())
    payload = adapter._build_payload(
        request_for(adapter, instructions="system", history=({"kind": "turn", "text": "prior"},))
    )
    assert payload["system"] == "system"
    assert payload["messages"] == [
        {"role": "assistant", "content": "prior"},
        {"role": "user", "content": "summarise"},
    ]


@pytest.mark.parametrize(
    "body",
    [
        {"id": "empty", "content": [], "stop_reason": "end_turn"},
        {
            "id": "unknown",
            "content": [{"type": "image", "source": {"type": "base64"}}],
            "stop_reason": "end_turn",
        },
        {
            "id": "blank",
            "content": [{"type": "text", "text": " "}],
            "stop_reason": "end_turn",
        },
    ],
)
def test_text_unknown_empty_or_blank_content_fails_closed(body: dict[str, Any]) -> None:
    adapter = AnthropicMessagesProvider(profile())
    with pytest.raises(ProviderError) as excinfo:
        adapter._parse(httpx.Response(200, json=body), elapsed_ms=1)
    assert excinfo.value.code is ErrorCode.PROVIDER_INVALID_RESPONSE
    assert "image" not in str(excinfo.value)


def test_profile_mismatch_is_rejected_before_http() -> None:
    adapter = AnthropicMessagesProvider(profile())
    request = ProviderRequest(
        alias="other",
        model="claude-model",
        input="summarise",
        max_output_tokens=64,
        timeout_seconds=5.0,
    )
    with pytest.raises(ProviderError, match="configured Anthropic profile"):
        adapter._validate_request(request)


def test_profile_secret_is_not_serialized() -> None:
    rendered = profile().model_dump_json()
    assert TEST_KEY not in rendered
