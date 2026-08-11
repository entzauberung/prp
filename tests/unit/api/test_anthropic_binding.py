"""Unit coverage for the Anthropic Messages text-only subset."""

import pytest

from prp_runtime.api.anthropic_messages import _normalize
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
