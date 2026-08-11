"""Unit coverage for the OpenAI Responses and Chat text subsets."""

import pytest

from prp_runtime.api.openai_chat import _normalize as normalize_chat
from prp_runtime.api.openai_responses import _normalize as normalize_responses
from prp_runtime.domain.errors import ErrorCode, PrpError


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
