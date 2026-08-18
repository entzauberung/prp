"""Unit coverage for the Anthropic Messages text-only subset."""

import pytest

from prp_runtime.api.anthropic_messages import _normalize
from prp_runtime.api.tool_bindings import (
    anthropic_messages_to_native_tool_turn,
    native_tool_turn_to_anthropic_content,
)
from prp_runtime.domain.enums import ToolCallStatus
from prp_runtime.domain.errors import ErrorCode, PrpError


def test_system_and_messages_map_to_native_text() -> None:
    result = _normalize(
        {
            "system": [
                {"type": "text", "text": "follow policy"},
                {"type": "text", "text": "be brief"},
            ],
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "prior answer"}],
                },
            ],
            "budget": {"max_total_tokens": 50},
            "routing": {"requires_revision": True},
        }
    )

    assert result.request is not None
    assert result.request.instructions == "follow policy\nbe brief"
    assert result.request.input == "question\nprior answer"
    assert result.request.budget.max_total_tokens == 50
    assert result.request.routing is not None
    assert result.request.routing.requires_revision is True


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (
            {"messages": [{"role": "user", "content": "hello"}], "stream": True},
            ErrorCode.UNSUPPORTED_STREAM_MODE,
        ),
        (
            {"messages": [{"role": "user", "content": "hello"}], "tools": []},
            ErrorCode.UNSUPPORTED_TOOLS,
        ),
        (
            {"messages": [{"role": "user", "content": "hello"}], "api_key": "secret"},
            ErrorCode.UNSUPPORTED_FIELD,
        ),
    ],
)
def test_unsupported_fields_have_stable_codes(
    payload: dict[str, object], code: ErrorCode
) -> None:
    with pytest.raises(PrpError) as excinfo:
        _normalize(payload)
    assert excinfo.value.code is code
    assert "secret" not in str(excinfo.value)


def test_image_blocks_are_rejected_as_multimodal() -> None:
    with pytest.raises(PrpError) as excinfo:
        _normalize(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": "private"}],
                    }
                ]
            }
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_MODALITY
    assert "private" not in str(excinfo.value)


def test_missing_messages_and_unknown_role_are_invalid() -> None:
    with pytest.raises(PrpError) as missing:
        _normalize({})
    with pytest.raises(PrpError) as role:
        _normalize({"messages": [{"role": "system", "content": "no"}]})
    assert missing.value.code is ErrorCode.INVALID_REQUEST
    assert role.value.code is ErrorCode.INVALID_REQUEST


def test_anthropic_tool_blocks_round_trip_through_shared_turn() -> None:
    turn = anthropic_messages_to_native_tool_turn(
        (
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-anthropic",
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
                        "tool_use_id": "call-anthropic",
                        "content": "safe",
                    }
                ],
            },
        )
    )
    assert turn is not None
    assert turn.tool_results[0].status is ToolCallStatus.SUCCEEDED
    assert native_tool_turn_to_anthropic_content(turn)[1]["tool_use_id"] == "call-anthropic"
