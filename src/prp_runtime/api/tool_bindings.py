"""Shared protocol-edge mapping for public native tool turns."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import NoReturn

from pydantic import Field, JsonValue, ValidationError, model_validator

from prp_runtime.api.errors import binding_error
from prp_runtime.domain.enums import ToolCallStatus
from prp_runtime.domain.errors import ErrorCode
from prp_runtime.domain.models import AgentToolCall, AgentToolResult, DomainModel
from prp_runtime.json_support import strict_json_loads

__all__ = [
    "NativeToolTurn",
    "anthropic_messages_to_native_tool_turn",
    "build_native_tool_turn",
    "chat_messages_to_native_tool_turn",
    "native_tool_turn_to_anthropic_content",
    "native_tool_turn_to_chat_messages",
    "validate_native_tool_turn",
]

ToolTurnItem = AgentToolCall | AgentToolResult


def _invalid_turn() -> NoReturn:
    raise binding_error(
        ErrorCode.INVALID_REQUEST,
        "tool turn order does not match the native call/result contract",
        field="tools",
    )


def validate_native_tool_turn(
    items: Iterable[ToolTurnItem],
) -> tuple[ToolTurnItem, ...]:
    """Validate one ordered public tool turn and return an immutable copy."""
    ordered = tuple(items)
    if not ordered:
        _invalid_turn()
    calls: set[str] = set()
    results: set[str] = set()
    for item in ordered:
        if isinstance(item, AgentToolCall):
            if item.call_id in calls or item.call_id in results:
                _invalid_turn()
            calls.add(item.call_id)
            continue
        if item.call_id not in calls or item.call_id in results:
            _invalid_turn()
        results.add(item.call_id)
    return ordered


class NativeToolTurn(DomainModel):
    """Protocol-neutral ordered tool facts for one public assistant turn."""

    items: tuple[ToolTurnItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _items_are_closed(self) -> NativeToolTurn:
        validate_native_tool_turn(self.items)
        return self

    @property
    def tool_calls(self) -> tuple[AgentToolCall, ...]:
        """Return calls in the order in which the protocol presented them."""
        return tuple(item for item in self.items if isinstance(item, AgentToolCall))

    @property
    def tool_results(self) -> tuple[AgentToolResult, ...]:
        """Return results in the order in which the protocol presented them."""
        return tuple(item for item in self.items if isinstance(item, AgentToolResult))


def build_native_tool_turn(items: Iterable[ToolTurnItem]) -> NativeToolTurn:
    """Build one checked shared intermediate from either protocol's edge data."""
    return NativeToolTurn(items=validate_native_tool_turn(items))


def _block_error(message: str) -> NoReturn:
    raise binding_error(ErrorCode.INVALID_REQUEST, message, field="messages")


def _reject_keys(value: Mapping[str, object], allowed: frozenset[str]) -> None:
    if set(value) - allowed:
        _block_error("tool message contains an unsupported field")


def _arguments(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, str):
        _block_error("tool call arguments must be a JSON string")
    try:
        parsed = strict_json_loads(value)
    except ValueError as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "tool call arguments must be valid standard JSON",
            field="messages",
        ) from error
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        _block_error("tool call arguments must be a JSON object")
    return parsed


def _output(value: object) -> tuple[dict[str, JsonValue] | None, str]:
    if isinstance(value, str):
        return None, value
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        _block_error("tool result content must be text or a JSON object")
    try:
        parsed = strict_json_loads(
            json.dumps(value, ensure_ascii=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "tool result content must be standard JSON",
            field="messages",
        ) from error
    if not isinstance(parsed, dict):
        _block_error("tool result content must be a JSON object")
    return parsed, ""


def _native_call(
    call_id: object, name: object, arguments: object
) -> AgentToolCall:
    if not isinstance(call_id, str) or not call_id.strip():
        _block_error("tool call requires an id")
    if not isinstance(name, str) or not name.strip():
        _block_error("tool call requires a function name")
    try:
        return AgentToolCall(
            call_id=call_id,
            tool_name=name,
            arguments=_arguments(arguments),
        )
    except ValidationError as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "tool call does not match the native contract",
            field="messages",
        ) from error


def _native_result(
    call_id: object,
    output: object,
    *,
    failed: bool = False,
) -> AgentToolResult:
    if not isinstance(call_id, str) or not call_id.strip():
        _block_error("tool result requires a call id")
    result, text = _output(output)
    try:
        return AgentToolResult(
            call_id=call_id,
            status=ToolCallStatus.REJECTED if failed else ToolCallStatus.SUCCEEDED,
            result=result,
            output=text,
        )
    except ValidationError as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "tool result does not match the native contract",
            field="messages",
        ) from error


def chat_messages_to_native_tool_turn(
    messages: Iterable[Mapping[str, object]],
) -> NativeToolTurn | None:
    """Parse OpenAI Chat tool messages into the shared ordered turn."""
    items: list[ToolTurnItem] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list):
                _block_error("Chat assistant tool_calls must be a list")
            for value in calls:
                if not isinstance(value, Mapping):
                    _block_error("Chat tool calls must be objects")
                _reject_keys(value, frozenset({"id", "type", "function"}))
                if value.get("type") != "function":
                    _block_error("Chat tool call type must be function")
                function = value.get("function")
                if not isinstance(function, Mapping):
                    _block_error("Chat tool call function must be an object")
                _reject_keys(function, frozenset({"name", "arguments"}))
                items.append(
                    _native_call(
                        value.get("id"),
                        function.get("name"),
                        function.get("arguments"),
                    )
                )
        elif role == "tool":
            _reject_keys(message, frozenset({"role", "tool_call_id", "content"}))
            items.append(
                _native_result(message.get("tool_call_id"), message.get("content"))
            )
    return None if not items else build_native_tool_turn(items)


def anthropic_messages_to_native_tool_turn(
    messages: Iterable[Mapping[str, object]],
) -> NativeToolTurn | None:
    """Parse Anthropic tool_use/tool_result blocks into the shared turn."""
    items: list[ToolTurnItem] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, Mapping):
                _block_error("Anthropic content blocks must be objects")
            block_type = block.get("type")
            if block_type == "tool_use":
                _reject_keys(block, frozenset({"type", "id", "name", "input"}))
                arguments = block.get("input")
                if not isinstance(arguments, Mapping):
                    _block_error("Anthropic tool_use input must be an object")
                items.append(
                    _native_call(
                        block.get("id"),
                        block.get("name"),
                        json.dumps(arguments, ensure_ascii=True, allow_nan=False),
                    )
                )
            elif block_type == "tool_result":
                _reject_keys(
                    block,
                    frozenset({"type", "tool_use_id", "content", "is_error"}),
                )
                is_error = block.get("is_error", False)
                if not isinstance(is_error, bool):
                    _block_error("Anthropic tool_result is_error must be a boolean")
                items.append(
                    _native_result(
                        block.get("tool_use_id"),
                        block.get("content"),
                        failed=is_error,
                    )
                )
    return None if not items else build_native_tool_turn(items)


def native_tool_turn_to_chat_messages(
    turn: NativeToolTurn,
) -> tuple[dict[str, object], ...]:
    """Project a shared turn to OpenAI Chat assistant/tool messages."""
    messages: list[dict[str, object]] = []
    for item in turn.items:
        if isinstance(item, AgentToolCall):
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.tool_name,
                                "arguments": json.dumps(
                                    item.arguments,
                                    ensure_ascii=True,
                                    allow_nan=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    ],
                }
            )
        else:
            output: object = item.output
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
                    "role": "tool",
                    "tool_call_id": item.call_id,
                    "content": output,
                }
            )
    return tuple(messages)


def native_tool_turn_to_anthropic_content(
    turn: NativeToolTurn,
) -> tuple[dict[str, object], ...]:
    """Project a shared turn to Anthropic tool_use/tool_result blocks."""
    content: list[dict[str, object]] = []
    for item in turn.items:
        if isinstance(item, AgentToolCall):
            content.append(
                {
                    "type": "tool_use",
                    "id": item.call_id,
                    "name": item.tool_name,
                    "input": item.arguments,
                }
            )
        else:
            output: object = item.output
            if not output and item.result is not None:
                output = item.result
            content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": item.call_id,
                    "content": output,
                    "is_error": item.status is ToolCallStatus.REJECTED,
                }
            )
    return tuple(content)
