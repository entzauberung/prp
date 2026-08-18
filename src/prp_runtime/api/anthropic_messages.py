"""Minimal Anthropic Messages inbound binding for non-streaming text."""

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Request

from prp_runtime.api.bindings import (
    BindingNormalizationResult,
    normalize_cancel,
    normalize_query,
    normalize_request,
    reject_unsupported_fields,
)
from prp_runtime.api.errors import binding_error
from prp_runtime.api.tool_bindings import anthropic_messages_to_native_tool_turn
from prp_runtime.control.routing import facts_from_request
from prp_runtime.domain.enums import RunStatus
from prp_runtime.domain.errors import ErrorCode
from prp_runtime.domain.models import NativeRunRequest, Run
from prp_runtime.runtime.assembler import RunResult, assemble_run_result

__all__ = ["create_router"]


_SUPPORTED_FIELDS = frozenset(
    {"system", "messages", "routing", "budget", "output"}
)


def _router_state(request: Request) -> tuple[Any, Any]:
    return request.app.state.controller, request.app.state.store


async def _json(request: Request) -> Mapping[str, object]:
    try:
        value = await request.json()
    except ValueError as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "the request body must be valid JSON",
            field="body",
        ) from error
    if not isinstance(value, Mapping):
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "the request body must be a JSON object",
            field="body",
        )
    return value


def _text(value: object, *, field: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "Anthropic content must be text or text blocks",
            field=field,
        )
    chunks: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "Anthropic content blocks must be objects",
                field=field,
            )
        if item.get("type") != "text":
            raise binding_error(
                ErrorCode.UNSUPPORTED_MODALITY,
                "only Anthropic text blocks are supported",
                field=field,
            )
        reject_unsupported_fields(item, allowed=frozenset({"type", "text"}))
        if not isinstance(item.get("text"), str):
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "Anthropic text blocks must contain text",
                field=field,
            )
        chunks.append(item["text"])
    return "\n".join(chunks)


def _normalize(payload: Mapping[str, object]) -> BindingNormalizationResult:
    reject_unsupported_fields(payload, allowed=_SUPPORTED_FIELDS)
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "Anthropic messages must be a list",
            field="messages",
        )
    anthropic_messages_to_native_tool_turn(
        tuple(message for message in messages if isinstance(message, Mapping))
    )
    input_parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "Anthropic messages must be objects",
                field="messages",
            )
        reject_unsupported_fields(
            message,
            allowed=frozenset({"role", "content"}),
        )
        if message.get("role") not in ("user", "assistant"):
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "Anthropic message role is not supported",
                field="messages.role",
            )
        input_parts.append(_text(message.get("content"), field="messages.content"))
    values: dict[str, object] = {"input": "\n".join(input_parts)}
    if "system" in payload:
        values["instructions"] = _text(payload["system"], field="system")
    for field in ("routing", "budget", "output"):
        if field in payload:
            values[field] = payload[field]
    return normalize_request(values)


def _status(status: RunStatus) -> str:
    if status is RunStatus.SUCCEEDED:
        return "completed"
    if status is RunStatus.FAILED:
        return "failed"
    if status is RunStatus.CANCELLED:
        return "cancelled"
    return "in_progress"


def _error(run: Run) -> dict[str, str] | None:
    if run.error is None:
        return None
    return {
        "type": "runtime_error",
        "code": run.error.category.value,
        "message": run.error.message,
    }


def _envelope(run: Run, result: RunResult) -> dict[str, object]:
    text = result.output_text
    content = [] if text is None else [{"type": "text", "text": text}]
    return {
        "id": run.run_id,
        "type": "message",
        "role": "assistant",
        "status": _status(run.status),
        "content": content,
        "stop_reason": "end_turn" if text is not None else None,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
        "error": _error(run),
    }


def create_router() -> APIRouter:
    """Build the Anthropic Messages router."""
    router = APIRouter(prefix="/v1/messages", tags=["anthropic-messages"])

    @router.post("")
    async def create_message(request: Request) -> dict[str, object]:
        normalized = _normalize(await _json(request))
        assert normalized.request is not None
        controller, store = _router_state(request)
        native_request = NativeRunRequest.model_validate(normalized.request.model_dump())
        created = await controller.create_run(native_request)
        finished = await controller.execute(
            created.run_id,
            routing_facts=facts_from_request(native_request),
        )
        return _envelope(finished, await assemble_run_result(store, created.run_id))

    @router.get("/{run_id}")
    async def get_message(request: Request, run_id: str) -> dict[str, object]:
        normalized = normalize_query(run_id)
        assert normalized.run_id is not None
        _, store = _router_state(request)
        run = await store.get_run(normalized.run_id)
        return _envelope(run, await assemble_run_result(store, normalized.run_id))

    @router.post("/{run_id}/cancel")
    async def cancel_message(request: Request, run_id: str) -> dict[str, object]:
        normalized = normalize_cancel(run_id)
        assert normalized.run_id is not None
        controller, store = _router_state(request)
        await controller.cancel(normalized.run_id)
        run = await store.get_run(normalized.run_id)
        return _envelope(run, await assemble_run_result(store, normalized.run_id))

    return router
