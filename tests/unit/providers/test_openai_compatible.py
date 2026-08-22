"""Targeted tests for the OpenAI-compatible adapter. Every call is mocked."""

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
    ProviderRequest,
    ProviderToolDescriptor,
)
from prp_runtime.providers.openai_compatible import (
    MAX_ERROR_BODY_CHARS,
    OpenAICompatibleProvider,
)

SECRET = "sk-not-a-real-key-0123456789"
BASE_URL = "https://models.internal/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"


def profile(**overrides: object) -> ModelProfile:
    data: dict[str, object] = {
        "alias": "worker",
        "provider": "openai_compatible",
        "model": "weak-model",
        "role": ModelRole.WORKER,
        "base_url": BASE_URL,
        "api_key": SECRET,
        "context_window_tokens": 32_000,
        "max_output_tokens": 4_000,
        "timeout_seconds": 5.0,
    }
    data.update(overrides)
    return ModelProfile(**data)  # type: ignore[arg-type]


def completion_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "the answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    body.update(overrides)
    return body


@pytest_asyncio.fixture
async def provider() -> AsyncIterator[OpenAICompatibleProvider]:
    adapter = OpenAICompatibleProvider(profile())
    try:
        yield adapter
    finally:
        await adapter.aclose()


def request_for(adapter: OpenAICompatibleProvider, **kwargs: object) -> ProviderRequest:
    return ProviderRequest.for_profile(adapter.profile, input="summarise", **kwargs)  # type: ignore[arg-type]


# --- success path ---------------------------------------------------------------


def test_endpoint_is_derived_from_the_profile() -> None:
    assert OpenAICompatibleProvider(profile()).endpoint == ENDPOINT
    assert OpenAICompatibleProvider(profile(base_url=BASE_URL + "/")).endpoint == ENDPOINT
    assert OpenAICompatibleProvider(profile()).name == "openai_compatible"


@pytest.mark.asyncio
async def test_successful_completion_is_normalised(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    route = mocked_http.post(ENDPOINT).respond(200, json=completion_body())
    response = await provider.complete(
        request_for(provider, instructions="be terse", max_output_tokens=64)
    )

    assert response.text == "the answer"
    assert response.finish_reason is FinishReason.STOP
    assert response.provider_request_id == "chatcmpl-123"
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (11, 7)
    assert response.usage.strong_model_tokens == 0
    assert response.usage.elapsed_ms >= 0

    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "model": "weak-model",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "summarise"},
        ],
        "max_tokens": 64,
        "stream": False,
    }
    assert route.calls.last.request.headers["authorization"] == f"Bearer {SECRET}"


def test_reasoning_content_metadata_preserves_final_text() -> None:
    adapter = OpenAICompatibleProvider(profile())
    body = completion_body(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "the final answer",
                    "reasoning_content": "provider-private reasoning metadata",
                },
                "finish_reason": "stop",
            }
        ]
    )

    response = adapter._parse(httpx.Response(200, json=body), elapsed_ms=1)

    assert response.text == "the final answer"


def test_reasoning_content_metadata_must_be_text_or_null() -> None:
    adapter = OpenAICompatibleProvider(profile())
    body = completion_body(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "the final answer",
                    "reasoning_content": {"unexpected": "shape"},
                },
                "finish_reason": "stop",
            }
        ]
    )

    with pytest.raises(ProviderError, match="reasoning content is not text"):
        adapter._parse(httpx.Response(200, json=body), elapsed_ms=1)


@pytest.mark.asyncio
async def test_instructions_are_omitted_when_absent(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    route = mocked_http.post(ENDPOINT).respond(200, json=completion_body())
    await provider.complete(request_for(provider))
    sent = json.loads(route.calls.last.request.content)
    assert sent["messages"] == [{"role": "user", "content": "summarise"}]


def test_public_history_maps_to_ordered_openai_messages() -> None:
    adapter = OpenAICompatibleProvider(profile())
    call = AgentToolCall(
        call_id="call-read",
        tool_name="read_file",
        arguments={"path": "src/main.py", "offset": 0},
    )
    history = (
        AgentTurn(text="I will inspect the file."),
        AgentTurn(tool_calls=(call,)),
        AgentToolResult(
            call_id="call-read",
            status=ToolCallStatus.SUCCEEDED,
            result={"ok": True, "line": 1},
        ),
        AgentTurn(text="The file is ready."),
    )
    payload = adapter._build_payload(
        request_for(
            adapter,
            instructions="Be concise.",
            history=history,
        )
    )

    assert payload["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "assistant", "content": "I will inspect the file."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-read",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"offset":0,"path":"src/main.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-read",
            "content": '{"line":1,"ok":true}',
        },
        {"role": "assistant", "content": "The file is ready."},
        {"role": "user", "content": "summarise"},
    ]


def test_history_does_not_duplicate_the_current_user_input() -> None:
    adapter = OpenAICompatibleProvider(profile())
    payload = adapter._build_payload(
        request_for(adapter, history=(AgentTurn(text="summarise"),))
    )

    assert [message for message in payload["messages"] if message["role"] == "user"] == [
        {"role": "user", "content": "summarise"}
    ]


def test_function_tool_calls_are_normalised_in_order() -> None:
    adapter = OpenAICompatibleProvider(profile())
    body = completion_body(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        },
                        {
                            "id": "call-two",
                            "type": "function",
                            "function": {
                                "name": "search_text",
                                "arguments": '{"pattern":"Provider"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )

    response = adapter._parse(httpx.Response(200, json=body), elapsed_ms=12)

    assert response.text is None
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert [call.call_id for call in response.tool_calls] == ["call-one", "call-two"]
    assert response.tool_calls[0].arguments == {"path": "README.md"}
    assert response.tool_calls[1].arguments == {"pattern": "Provider"}


def test_tool_call_ids_are_unique_after_contract_normalisation() -> None:
    adapter = OpenAICompatibleProvider(profile())
    body = completion_body(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": " call-one ",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )

    with pytest.raises(ProviderError, match="unique"):
        adapter._parse(httpx.Response(200, json=body), elapsed_ms=1)


def test_non_standard_response_numbers_fail_closed() -> None:
    adapter = OpenAICompatibleProvider(profile())
    body = completion_body(usage={"prompt_tokens": "NaN", "completion_tokens": 1})
    response = httpx.Response(
        200,
        content=json.dumps(body).replace('"NaN"', "NaN").encode("utf-8"),
    )

    with pytest.raises(ProviderError) as excinfo:
        adapter._parse(response, elapsed_ms=1)

    assert excinfo.value.code is ErrorCode.PROVIDER_INVALID_RESPONSE


def test_blank_response_text_is_a_provider_contract_error() -> None:
    adapter = OpenAICompatibleProvider(profile())
    body = completion_body(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": " \n"},
                "finish_reason": "stop",
            }
        ]
    )

    with pytest.raises(ProviderError) as excinfo:
        adapter._parse(httpx.Response(200, json=body), elapsed_ms=1)

    assert excinfo.value.code is ErrorCode.PROVIDER_INVALID_RESPONSE


@pytest.mark.parametrize(
    "tool_calls",
    [
        [
            {
                "id": "duplicate",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            },
            {
                "id": "duplicate",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            },
        ],
        [
            {
                "id": "array-args",
                "type": "function",
                "function": {"name": "read_file", "arguments": "[]"},
            }
        ],
        [
            {
                "id": "invalid-json",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": f'{{"secret":"{SECRET}","path":"/srv/private"',
                },
            }
        ],
        [
            {
                "id": "wrong-type",
                "type": "custom",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
        [
            {
                "id": "oversized",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"value": "x" * 70_000}),
                },
            }
        ],
    ],
)
def test_malformed_tool_calls_fail_closed_without_echoing_upstream_body(
    tool_calls: list[dict[str, Any]],
) -> None:
    adapter = OpenAICompatibleProvider(profile())
    body = completion_body(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ]
    )

    with pytest.raises(ProviderError) as excinfo:
        adapter._parse(httpx.Response(200, json=body), elapsed_ms=1)

    assert excinfo.value.code is ErrorCode.PROVIDER_INVALID_RESPONSE
    assert excinfo.value.retryable is False
    rendered = str(excinfo.value)
    assert SECRET not in rendered
    assert "/srv/private" not in rendered
    assert "oversized" not in rendered


def test_tool_calls_cannot_be_combined_with_content() -> None:
    adapter = OpenAICompatibleProvider(profile())
    body = completion_body(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hidden text",
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )

    with pytest.raises(ProviderError) as excinfo:
        adapter._parse(httpx.Response(200, json=body), elapsed_ms=1)

    assert excinfo.value.code is ErrorCode.PROVIDER_INVALID_RESPONSE
    assert "hidden text" not in str(excinfo.value)


def test_non_empty_tool_catalog_maps_to_openai_function_tools() -> None:
    adapter = OpenAICompatibleProvider(profile())
    tool = ProviderToolDescriptor(
        name="read_file",
        description="Read one relative file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    payload = adapter._build_payload(request_for(adapter, tools=(tool,)))

    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read one relative file.",
                "parameters": tool.input_schema,
            },
        }
    ]
    assert payload["tool_choice"] == "auto"
    assert "handler" not in json.dumps(payload)
    assert "effect" not in json.dumps(payload)
    assert SECRET not in json.dumps(payload)


@pytest.mark.asyncio
async def test_no_authorization_header_without_a_configured_key(
    mocked_http: respx.MockRouter,
) -> None:
    adapter = OpenAICompatibleProvider(profile(api_key=None))
    route = mocked_http.post(ENDPOINT).respond(200, json=completion_body())
    try:
        await adapter.complete(request_for(adapter))
    finally:
        await adapter.aclose()
    assert "authorization" not in route.calls.last.request.headers


@pytest.mark.asyncio
async def test_structured_output_is_sent_only_as_a_response_format(
    mocked_http: respx.MockRouter,
) -> None:
    adapter = OpenAICompatibleProvider(
        profile(alias="leader", role=ModelRole.PLANNER, supports_structured_output=True)
    )
    route = mocked_http.post(ENDPOINT).respond(200, json=completion_body())
    try:
        await adapter.complete(
            request_for(adapter, json_schema='{"type": "object"}', max_output_tokens=32)
        )
    finally:
        await adapter.aclose()
    sent = json.loads(route.calls.last.request.content)
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "output", "strict": True, "schema": {"type": "object"}},
    }
    assert "tools" not in sent
    assert sent["stream"] is False


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
@pytest.mark.asyncio
async def test_non_standard_schema_number_is_rejected_before_any_http_call(
    mocked_http: respx.MockRouter, token: str
) -> None:
    """A non-finite number in the schema text is rejected before any HTTP call.

    ``ProviderRequest`` itself does not parse ``json_schema``, so this proves the
    provider's own strict parse in ``_build_payload`` runs before ``client.post``.
    """
    adapter = OpenAICompatibleProvider(
        profile(alias="leader", role=ModelRole.PLANNER, supports_structured_output=True)
    )
    schema = f'{{"type": "object", "minimum": {token}}}'
    try:
        with pytest.raises(ValueError):
            await adapter.complete(
                request_for(adapter, json_schema=schema, max_output_tokens=32)
            )
    finally:
        await adapter.aclose()
    assert not mocked_http.calls


@pytest.mark.asyncio
async def test_planner_profile_tokens_count_as_strong_model_tokens(
    mocked_http: respx.MockRouter,
) -> None:
    adapter = OpenAICompatibleProvider(profile(alias="leader", role=ModelRole.PLANNER))
    mocked_http.post(ENDPOINT).respond(200, json=completion_body())
    try:
        response = await adapter.complete(request_for(adapter))
    finally:
        await adapter.aclose()
    assert response.usage is not None
    assert response.usage.strong_model_tokens == 18
    assert response.usage.total_tokens == 18


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("max_tokens", FinishReason.LENGTH),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("something_new", FinishReason.OTHER),
        (None, FinishReason.OTHER),
    ],
)
@pytest.mark.asyncio
async def test_finish_reason_is_normalised(
    provider: OpenAICompatibleProvider,
    mocked_http: respx.MockRouter,
    reported: str | None,
    expected: FinishReason,
) -> None:
    body = completion_body()
    body["choices"][0]["finish_reason"] = reported
    mocked_http.post(ENDPOINT).respond(200, json=body)
    response = await provider.complete(request_for(provider))
    assert response.finish_reason is expected


# --- usage normalisation --------------------------------------------------------


@pytest.mark.parametrize(
    "usage_field",
    [
        None,
        "not-an-object",
        {"completion_tokens": 5},
        {"prompt_tokens": 5},
        {"prompt_tokens": -1, "completion_tokens": 5},
        {"prompt_tokens": True, "completion_tokens": 5},
        {"prompt_tokens": "5", "completion_tokens": 5},
    ],
)
@pytest.mark.asyncio
async def test_unreported_usage_is_not_guessed(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter, usage_field: object
) -> None:
    body = completion_body()
    if usage_field is None:
        body.pop("usage")
    else:
        body["usage"] = usage_field
    mocked_http.post(ENDPOINT).respond(200, json=body)
    response = await provider.complete(request_for(provider))
    assert response.usage is None


@pytest.mark.asyncio
async def test_zero_token_usage_is_kept_as_a_measurement(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    body = completion_body(usage={"prompt_tokens": 0, "completion_tokens": 0})
    mocked_http.post(ENDPOINT).respond(200, json=body)
    response = await provider.complete(request_for(provider))
    assert response.usage is not None
    assert response.usage.total_tokens == 0


# --- error mapping --------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (401, ErrorCode.PROVIDER_AUTH_FAILED, False),
        (403, ErrorCode.PROVIDER_AUTH_FAILED, False),
        (408, ErrorCode.PROVIDER_TIMEOUT, True),
        (504, ErrorCode.PROVIDER_TIMEOUT, True),
        (429, ErrorCode.PROVIDER_RATE_LIMITED, True),
        (400, ErrorCode.PROVIDER_INVALID_RESPONSE, False),
        (404, ErrorCode.PROVIDER_INVALID_RESPONSE, False),
        (500, ErrorCode.PROVIDER_UNAVAILABLE, True),
        (503, ErrorCode.PROVIDER_UNAVAILABLE, True),
    ],
)
@pytest.mark.asyncio
async def test_status_codes_map_to_domain_errors(
    provider: OpenAICompatibleProvider,
    mocked_http: respx.MockRouter,
    status: int,
    code: ErrorCode,
    retryable: bool,
) -> None:
    mocked_http.post(ENDPOINT).respond(status, text="upstream said no")
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(request_for(provider))
    assert excinfo.value.code is code
    assert excinfo.value.retryable is retryable
    assert str(status) in excinfo.value.detail.message


@pytest.mark.asyncio
async def test_error_body_is_truncated_and_the_key_is_redacted(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    leaky = f"invalid key {SECRET} " + "x" * 4_000
    mocked_http.post(ENDPOINT).respond(401, text=leaky)
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(request_for(provider))
    message = excinfo.value.detail.message
    assert SECRET not in message
    assert SECRET not in str(excinfo.value)
    assert "[redacted]" in message
    assert len(message) < MAX_ERROR_BODY_CHARS + 200
    assert "Authorization" not in message


@pytest.mark.asyncio
async def test_timeout_maps_to_a_retryable_timeout_error(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    mocked_http.post(ENDPOINT).mock(side_effect=httpx.ReadTimeout("too slow"))
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(request_for(provider))
    assert excinfo.value.code is ErrorCode.PROVIDER_TIMEOUT
    assert excinfo.value.retryable is True


@pytest.mark.asyncio
async def test_transport_failure_maps_to_unavailable(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    mocked_http.post(ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(request_for(provider))
    assert excinfo.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert excinfo.value.retryable is True


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"choices": "nope"},
        {"choices": ["text"]},
        {"choices": [{"index": 0}]},
        {"choices": [{"index": 0, "message": {"content": None}}]},
        {"choices": [{"index": 0, "message": {"tool_calls": []}}]},
        ["not", "an", "object"],
    ],
)
@pytest.mark.asyncio
async def test_unusable_response_structures_are_rejected(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter, body: object
) -> None:
    mocked_http.post(ENDPOINT).respond(200, json=body)
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(request_for(provider))
    assert excinfo.value.code is ErrorCode.PROVIDER_INVALID_RESPONSE
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_non_json_body_is_rejected(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    mocked_http.post(ENDPOINT).respond(200, text="<html>gateway</html>")
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(request_for(provider))
    assert excinfo.value.code is ErrorCode.PROVIDER_INVALID_RESPONSE


@pytest.mark.asyncio
async def test_request_for_another_alias_is_refused(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    foreign = ProviderRequest(
        alias="leader",
        model="strong-model",
        input="summarise",
        max_output_tokens=10,
        timeout_seconds=1.0,
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(foreign)
    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED
    assert not mocked_http.calls


@pytest.mark.asyncio
async def test_request_for_another_model_is_refused(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    foreign = ProviderRequest(
        alias="worker",
        model="attacker-model",
        input="summarise",
        max_output_tokens=10,
        timeout_seconds=1.0,
    )
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(foreign)
    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED
    assert not mocked_http.calls


# --- cancellation ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellation_propagates_untouched(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    async def cancelled_upstream(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    mocked_http.post(ENDPOINT).mock(side_effect=cancelled_upstream)
    with pytest.raises(asyncio.CancelledError):
        await provider.complete(request_for(provider))


@pytest.mark.asyncio
async def test_an_awaiting_call_can_be_cancelled_by_its_task(
    provider: OpenAICompatibleProvider, mocked_http: respx.MockRouter
) -> None:
    async def never_answers(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200, json=completion_body())

    mocked_http.post(ENDPOINT).mock(side_effect=never_answers)
    task = asyncio.create_task(provider.complete(request_for(provider)))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --- client lifecycle -----------------------------------------------------------


@pytest.mark.asyncio
async def test_an_injected_client_is_reused_and_not_closed(
    mocked_http: respx.MockRouter,
) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        adapter = OpenAICompatibleProvider(profile(), client=client)
        mocked_http.post(ENDPOINT).respond(200, json=completion_body())
        await adapter.complete(request_for(adapter))
        await adapter.aclose()
        assert client.is_closed is False
        mocked_http.post(ENDPOINT).respond(200, json=completion_body())
        await adapter.complete(request_for(adapter))


@pytest.mark.asyncio
async def test_an_owned_client_is_created_lazily_and_closed(
    mocked_http: respx.MockRouter,
) -> None:
    adapter = OpenAICompatibleProvider(profile())
    mocked_http.post(ENDPOINT).respond(200, json=completion_body())
    await adapter.complete(request_for(adapter))
    await adapter.aclose()
    await adapter.aclose()


@pytest.mark.asyncio
async def test_an_injected_client_survives_error_and_repeated_close(
    mocked_http: respx.MockRouter,
) -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        adapter = OpenAICompatibleProvider(profile(), client=client)
        mocked_http.post(ENDPOINT).respond(503, text="temporary upstream failure")
        with pytest.raises(ProviderError):
            await adapter.complete(request_for(adapter))
        await adapter.aclose()
        await adapter.aclose()
        assert client.is_closed is False
