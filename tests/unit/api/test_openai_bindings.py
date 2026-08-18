"""Unit coverage for the OpenAI Responses and Chat text subsets."""

import pytest

from prp_runtime.api.openai_chat import _normalize as normalize_chat
from prp_runtime.api.openai_responses import _normalize as normalize_responses
from prp_runtime.api.openai_responses import (
    native_tool_call_to_responses,
    native_tool_result_to_responses,
    responses_function_call_output_to_native,
    responses_function_call_to_native,
)
from prp_runtime.api.tool_bindings import (
    chat_messages_to_native_tool_turn,
    native_tool_turn_to_chat_messages,
)
from prp_runtime.domain.enums import ToolCallStatus
from prp_runtime.domain.errors import ErrorCode, PrpError
from prp_runtime.domain.models import AgentToolCall, AgentToolResult


def test_responses_text_and_text_items_share_native_request() -> None:
    direct = normalize_responses(
        {
            "input": "hello",
            "instructions": "be brief",
            "routing": {"requires_cascade": True},
        }
    )
    blocks = normalize_responses(
        {
            "input": [
                {"type": "input_text", "text": "hello"},
                {"type": "text", "text": "again"},
            ]
        }
    )

    assert direct.request is not None
    assert blocks.request is not None
    assert direct.request.input == "hello"
    assert direct.request.instructions == "be brief"
    assert direct.request.routing is not None
    assert direct.request.routing.requires_cascade is True
    assert blocks.request.input == "hello\nagain"


def test_chat_system_and_user_messages_map_to_native_fields() -> None:
    result = normalize_chat(
        {
            "messages": [
                {"role": "system", "content": "follow the policy"},
                {"role": "user", "content": "answer this"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "prior answer"}],
                },
            ],
            "routing": {"requires_plan": True, "desired_parallelism": 2},
        }
    )

    assert result.request is not None
    assert result.request.instructions == "follow the policy"
    assert result.request.input == "answer this\nprior answer"
    assert result.request.routing is not None
    assert result.request.routing.requires_plan is True
    assert result.request.routing.desired_parallelism == 2


@pytest.mark.parametrize(
    ("normalizer", "payload", "code"),
    [
        (
            normalize_responses,
            {"input": "hello", "stream": True},
            ErrorCode.UNSUPPORTED_STREAM_MODE,
        ),
        (
            normalize_chat,
            {"messages": [{"role": "user", "content": "hello"}], "tools": []},
            ErrorCode.UNSUPPORTED_TOOLS,
        ),
        (
            normalize_responses,
            {"input": "hello", "api_key": "secret"},
            ErrorCode.UNSUPPORTED_FIELD,
        ),
    ],
)
def test_openai_unsupported_fields_are_stable(
    normalizer: object, payload: dict[str, object], code: ErrorCode
) -> None:
    with pytest.raises(PrpError) as excinfo:
        normalizer(payload)  # type: ignore[operator]
    assert excinfo.value.code is code
    assert "secret" not in str(excinfo.value)


def test_chat_image_content_is_rejected_as_multimodal() -> None:
    with pytest.raises(PrpError) as excinfo:
        normalize_chat(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": "private"}],
                    }
                ]
            }
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_MODALITY
    assert "private" not in str(excinfo.value)


def test_chat_unknown_role_and_empty_messages_are_validation_errors() -> None:
    with pytest.raises(PrpError) as role_error:
        normalize_chat({"messages": [{"role": "tool", "content": "no"}]})
    with pytest.raises(PrpError) as empty_error:
        normalize_chat({"messages": []})
    assert role_error.value.code is ErrorCode.INVALID_REQUEST
    assert empty_error.value.code is ErrorCode.INVALID_REQUEST


def test_responses_function_call_round_trips_call_id_and_strict_arguments() -> None:
    call = responses_function_call_to_native(
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "read_file",
            "arguments": '{"path":"src/main.py"}',
        }
    )

    assert call == AgentToolCall(
        call_id="call_123",
        tool_name="read_file",
        arguments={"path": "src/main.py"},
    )
    assert native_tool_call_to_responses(call) == {
        "type": "function_call",
        "call_id": "call_123",
        "name": "read_file",
        "arguments": '{"path":"src/main.py"}',
    }


def test_responses_function_call_output_maps_to_native_result() -> None:
    result = responses_function_call_output_to_native(
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": {"content": "safe"},
        }
    )

    assert result == AgentToolResult(
        call_id="call_123",
        status=ToolCallStatus.SUCCEEDED,
        result={"content": "safe"},
    )
    assert native_tool_result_to_responses(result) == {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": '{"content":"safe"}',
    }


def test_responses_tool_blocks_reject_unknown_fields_and_non_json_arguments() -> None:
    with pytest.raises(PrpError) as unknown:
        responses_function_call_to_native(
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "read_file",
                "arguments": "{}",
                "secret": "do not echo",
            }
        )
    with pytest.raises(PrpError) as invalid_json:
        responses_function_call_to_native(
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "read_file",
                "arguments": "NaN",
            }
        )
    assert unknown.value.code is ErrorCode.UNSUPPORTED_FIELD
    assert invalid_json.value.code is ErrorCode.INVALID_REQUEST
    assert "do not echo" not in str(unknown.value)


def test_chat_tool_messages_round_trip_through_shared_turn() -> None:
    turn = chat_messages_to_native_tool_turn(
        (
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-chat",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-chat", "content": "safe"},
        )
    )
    assert turn is not None
    assert native_tool_turn_to_chat_messages(turn)[1]["tool_call_id"] == "call-chat"
