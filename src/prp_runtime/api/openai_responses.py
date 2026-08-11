"""Minimal OpenAI Responses inbound binding.

Only non-streaming text input is accepted. The router delegates execution and
state to the shared Controller/Store facts and returns an OpenAI-shaped view.
"""

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


_SUPPORTED_FIELDS = frozenset(
    {"input", "instructions", "routing", "budget", "output"}
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


def _text_input(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "Responses input must be text or a list of text items",
            field="input",
        )
    chunks: list[str] = []
    for item in value:
        if isinstance(item, str):
            chunks.append(item)
            continue
        if not isinstance(item, Mapping):
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "Responses input items must be text objects",
                field="input",
            )
        if item.get("type", "input_text") not in ("input_text", "text"):
            raise binding_error(
                ErrorCode.UNSUPPORTED_MODALITY,
                "only text input items are supported",
                field="input",
            )
        extra = set(item) - {"type", "text"}
        if extra:
            reject_unsupported_fields(item, allowed=frozenset({"type", "text"}))
        if not isinstance(item.get("text"), str):
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "Responses text items must contain text",
                field="input",
            )
        chunks.append(item["text"])
    return "\n".join(chunks)


def _normalize(payload: Mapping[str, object]) -> BindingNormalizationResult:
    reject_unsupported_fields(payload, allowed=_SUPPORTED_FIELDS)
    values = dict(payload)
    if "input" in values:
        values["input"] = _text_input(values["input"])
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
    output: list[dict[str, object]] = []
    if text is not None:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )
    return {
        "id": run.run_id,
        "object": "response",
        "status": _status(run.status),
        "output": output,
        "output_text": text,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "error": _error(run),
    }


async def _read_result(request: Request, run_id: str) -> dict[str, object]:
    _, store = _router_state(request)
    run = await store.get_run(run_id)
    return _envelope(run, await assemble_run_result(store, run_id))


def create_router() -> APIRouter:
    """Build the OpenAI Responses router without owning application state."""
    router = APIRouter(prefix="/v1/responses", tags=["openai-responses"])

    @router.post("")
    async def create_response(request: Request) -> dict[str, object]:
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
    async def get_response(request: Request, run_id: str) -> dict[str, object]:
        normalized = normalize_query(run_id)
        assert normalized.run_id is not None
        return await _read_result(request, normalized.run_id)

    @router.post("/{run_id}/cancel")
    async def cancel_response(request: Request, run_id: str) -> dict[str, object]:
        normalized = normalize_cancel(run_id)
        assert normalized.run_id is not None
        controller, store = _router_state(request)
        await controller.cancel(normalized.run_id)
        run = await store.get_run(normalized.run_id)
        return _envelope(run, await assemble_run_result(store, normalized.run_id))

    return router
