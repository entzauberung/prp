"""Minimal OpenAI Chat Completions inbound binding for text messages."""

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
from prp_runtime.control.routing import facts_from_request
from prp_runtime.domain.enums import RunStatus
from prp_runtime.domain.errors import ErrorCode
from prp_runtime.domain.models import NativeRunRequest, Run
from prp_runtime.runtime.assembler import RunResult, assemble_run_result

__all__ = ["create_router"]


_SUPPORTED_FIELDS = frozenset({"messages", "routing", "budget", "output"})


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


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "Chat message content must be text",
            field="messages.content",
        )
    chunks: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "Chat content items must be text objects",
                field="messages.content",
            )
        if item.get("type") not in ("text", "input_text"):
            raise binding_error(
                ErrorCode.UNSUPPORTED_MODALITY,
                "only text Chat content is supported",
                field="messages.content",
            )
        reject_unsupported_fields(
            item,
            allowed=frozenset({"type", "text"}),
        )
        if not isinstance(item.get("text"), str):
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "Chat text content must be a string",
                field="messages.content",
            )
        chunks.append(item["text"])
    return "\n".join(chunks)


def _normalize(payload: Mapping[str, object]) -> BindingNormalizationResult:
    reject_unsupported_fields(payload, allowed=_SUPPORTED_FIELDS)
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "Chat messages must be a list",
            field="messages",
        )
    instructions: list[str] = []
    input_parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "Chat messages must be objects",
                field="messages",
            )
        reject_unsupported_fields(
            message,
            allowed=frozenset({"role", "content"}),
        )
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "Chat message role is not supported",
                field="messages.role",
            )
        text = _message_text(message.get("content"))
        if role == "system":
            instructions.append(text)
        else:
            input_parts.append(text)
    values: dict[str, object] = {
        "input": "\n".join(input_parts),
    }
    if instructions:
        values["instructions"] = "\n".join(instructions)
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
    return {
        "id": run.run_id,
        "object": "chat.completion",
        "status": _status(run.status),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop" if text is not None else None,
            }
        ],
        "usage": {
            "prompt_tokens": result.usage.input_tokens,
            "completion_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "error": _error(run),
    }


def create_router() -> APIRouter:
    """Build the OpenAI Chat Completions router."""
    router = APIRouter(prefix="/v1/chat/completions", tags=["openai-chat"])

    @router.post("")
    async def create_completion(request: Request) -> dict[str, object]:
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
    async def get_completion(request: Request, run_id: str) -> dict[str, object]:
        normalized = normalize_query(run_id)
        assert normalized.run_id is not None
        _, store = _router_state(request)
        run = await store.get_run(normalized.run_id)
        return _envelope(run, await assemble_run_result(store, normalized.run_id))

    @router.post("/{run_id}/cancel")
    async def cancel_completion(request: Request, run_id: str) -> dict[str, object]:
        normalized = normalize_cancel(run_id)
        assert normalized.run_id is not None
        controller, store = _router_state(request)
        await controller.cancel(normalized.run_id)
        run = await store.get_run(normalized.run_id)
        return _envelope(run, await assemble_run_result(store, normalized.run_id))

    return router
