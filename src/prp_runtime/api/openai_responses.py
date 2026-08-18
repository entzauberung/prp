"""Minimal OpenAI Responses inbound binding.

Only non-streaming text input is accepted. The router delegates execution and
state to the shared Controller/Store facts and returns an OpenAI-shaped view.
"""

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import JsonValue, ValidationError

from prp_runtime.api.auth import require_configured_auth
from prp_runtime.api.bindings import (
    BindingNormalizationResult,
    normalize_cancel,
    normalize_query,
    normalize_request,
    reject_unsupported_fields,
)
from prp_runtime.api.errors import binding_error
from prp_runtime.api.native import RunEventEnvelope
from prp_runtime.domain.enums import RunStatus, ToolCallStatus
from prp_runtime.domain.errors import ErrorCode
from prp_runtime.domain.models import (
    AgentToolCall,
    AgentToolResult,
    NativeRunRequest,
    Run,
)
from prp_runtime.json_support import strict_json_loads
from prp_runtime.runtime.assembler import RunResult, assemble_run_result
from prp_runtime.runtime.event_bus import EventBus
from prp_runtime.runtime.supervisor import RunSupervisor
from prp_runtime.storage.sqlite import MissingEntityError
from prp_runtime.tools.models import ToolCall, ToolResult

__all__ = [
    "create_router",
    "native_tool_call_to_responses",
    "native_tool_result_to_responses",
    "responses_function_call_output_to_native",
    "responses_function_call_to_native",
]


_SUPPORTED_FIELDS = frozenset(
    {"input", "instructions", "routing", "budget", "output"}
)

_FUNCTION_CALL_FIELDS = frozenset({"type", "call_id", "name", "arguments"})
_FUNCTION_CALL_OUTPUT_FIELDS = frozenset({"type", "call_id", "output"})


def _invalid_tool_block(message: str, *, field: str) -> NoReturn:
    raise binding_error(ErrorCode.INVALID_REQUEST, message, field=field)


def responses_function_call_to_native(
    block: Mapping[str, object],
) -> AgentToolCall:
    """Map one Responses ``function_call`` item to the Agent call contract."""
    reject_unsupported_fields(block, allowed=_FUNCTION_CALL_FIELDS)
    if block.get("type") != "function_call":
        _invalid_tool_block(
            "Responses item must have type function_call", field="type"
        )
    call_id = block.get("call_id")
    name = block.get("name")
    arguments = block.get("arguments")
    if not isinstance(call_id, str) or not call_id.strip():
        _invalid_tool_block("function_call requires a call_id", field="call_id")
    if not isinstance(name, str) or not name.strip():
        _invalid_tool_block("function_call requires a function name", field="name")
    if not isinstance(arguments, str):
        _invalid_tool_block(
            "function_call arguments must be a JSON string", field="arguments"
        )
    try:
        parsed = strict_json_loads(arguments)
    except ValueError as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "function_call arguments must be valid standard JSON",
            field="arguments",
        ) from error
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) for key in parsed
    ):
        _invalid_tool_block(
            "function_call arguments must be a JSON object", field="arguments"
        )
    try:
        return AgentToolCall(
            call_id=call_id,
            tool_name=name,
            arguments=parsed,
        )
    except ValidationError as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "function_call does not match the native tool-call contract",
            field="input",
        ) from error


def responses_function_call_output_to_native(
    block: Mapping[str, object],
) -> AgentToolResult:
    """Map one Responses ``function_call_output`` item to a terminal result."""
    reject_unsupported_fields(block, allowed=_FUNCTION_CALL_OUTPUT_FIELDS)
    if block.get("type") != "function_call_output":
        _invalid_tool_block(
            "Responses item must have type function_call_output", field="type"
        )
    call_id = block.get("call_id")
    output = block.get("output")
    if not isinstance(call_id, str) or not call_id.strip():
        _invalid_tool_block(
            "function_call_output requires a call_id", field="call_id"
        )
    if isinstance(output, str):
        try:
            return AgentToolResult(
                call_id=call_id,
                status=ToolCallStatus.SUCCEEDED,
                output=output,
            )
        except ValidationError as error:
            raise binding_error(
                ErrorCode.INVALID_REQUEST,
                "function_call_output exceeds the native result limit",
                field="output",
            ) from error
    if not isinstance(output, Mapping) or not all(
        isinstance(key, str) for key in output
    ):
        _invalid_tool_block(
            "function_call_output output must be text or a JSON object",
            field="output",
        )
    try:
        structured = strict_json_loads(
            json.dumps(output, ensure_ascii=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "function_call_output output must be standard JSON",
            field="output",
        ) from error
    if not isinstance(structured, dict):
        _invalid_tool_block(
            "function_call_output output must be a JSON object", field="output"
        )
    try:
        return AgentToolResult(
            call_id=call_id,
            status=ToolCallStatus.SUCCEEDED,
            result=structured,
        )
    except ValidationError as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "function_call_output does not match the native result contract",
            field="output",
        ) from error


def native_tool_call_to_responses(
    call: AgentToolCall | ToolCall,
) -> dict[str, object]:
    """Map a native Agent call to a Responses output item without changing ids."""
    return {
        "type": "function_call",
        "call_id": call.call_id,
        "name": call.tool_name,
        "arguments": json.dumps(
            call.arguments,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def native_tool_result_to_responses(
    result: AgentToolResult | ToolResult,
) -> dict[str, object]:
    """Map a native Agent result to a Responses ``function_call_output`` item."""
    if result.output:
        output: JsonValue = result.output
    elif result.result is not None:
        output = json.dumps(
            result.result,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        output = ""
    return {
        "type": "function_call_output",
        "call_id": result.call_id,
        "output": output,
    }


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


def _status(status: RunStatus, *, awaiting_approval: bool = False) -> str:
    if status is RunStatus.PENDING:
        return "pending"
    if awaiting_approval:
        return "awaiting_approval"
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


def _envelope(
    run: Run,
    result: RunResult,
    *,
    tool_output: list[dict[str, object]],
    awaiting_approval: bool,
) -> dict[str, object]:
    text = result.output_text
    output = list(tool_output)
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
        "status": _status(run.status, awaiting_approval=awaiting_approval),
        "output": output,
        "output_text": text,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "error": _error(run),
    }


async def _tool_output(store: Any, run_id: str) -> list[dict[str, object]]:
    """Project persisted native ToolCall/Result facts into Responses output."""
    calls = await store.list_tool_calls(run_id)
    output: list[dict[str, object]] = []
    for call in calls:
        output.append(native_tool_call_to_responses(call))
        if not call.status.is_terminal:
            continue
        try:
            result = await store.get_tool_result(call.call_id)
        except MissingEntityError:
            continue
        output.append(native_tool_result_to_responses(result))
    return output


async def _response_envelope(request: Request, run: Run) -> dict[str, object]:
    _, store = _router_state(request)
    pending = await store.list_tool_calls(
        run.run_id, statuses=[ToolCallStatus.AWAITING_APPROVAL]
    )
    return _envelope(
        run,
        await assemble_run_result(store, run.run_id),
        tool_output=await _tool_output(store, run.run_id),
        awaiting_approval=bool(pending),
    )


async def _read_result(request: Request, run_id: str) -> dict[str, object]:
    _, store = _router_state(request)
    run = await store.get_run(run_id)
    return await _response_envelope(request, run)


def _event_bus(request: Request) -> EventBus:
    event_bus = getattr(request.app.state, "event_bus", None)
    if not isinstance(event_bus, EventBus):
        raise RuntimeError("the application was built without an event bus")
    return event_bus


def _supervisor(request: Request) -> RunSupervisor:
    supervisor = getattr(request.app.state, "supervisor", None)
    if not isinstance(supervisor, RunSupervisor):
        raise RuntimeError("the application was built without a run supervisor")
    return supervisor


def _event_cursor(last_event_id: str | None, after: int | None) -> int:
    raw = last_event_id if last_event_id is not None else after
    if raw is None:
        return 0
    try:
        cursor = int(raw)
    except (TypeError, ValueError) as error:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "event cursor must be an integer",
            field="Last-Event-ID" if last_event_id is not None else "after",
        ) from error
    if cursor < 0:
        raise binding_error(
            ErrorCode.INVALID_REQUEST,
            "event cursor must not be negative",
            field="after",
        )
    return cursor


def create_router() -> APIRouter:
    """Build the OpenAI Responses router without owning application state."""
    router = APIRouter(
        prefix="/v1/responses",
        tags=["openai-responses"],
        dependencies=[Depends(require_configured_auth)],
    )

    @router.post("", status_code=202)
    async def create_response(request: Request) -> dict[str, object]:
        normalized = _normalize(await _json(request))
        assert normalized.request is not None
        controller, _ = _router_state(request)
        native_request = NativeRunRequest.model_validate(normalized.request.model_dump())
        created = await controller.create_run(native_request)
        await _supervisor(request).enqueue(created.run_id)
        return await _response_envelope(request, created)

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
        await _supervisor(request).enqueue(normalized.run_id)
        return await _response_envelope(request, run)

    @router.get("/{run_id}/events")
    async def stream_events(
        request: Request,
        run_id: str,
        after: int | None = Query(default=None, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> Response:
        normalized = normalize_query(run_id)
        assert normalized.run_id is not None
        _, store = _router_state(request)
        await store.get_run(normalized.run_id)
        cursor = _event_cursor(last_event_id, after)
        subscription = await _event_bus(request).subscribe(normalized.run_id)

        async def body() -> AsyncIterator[bytes]:
            stream_cursor = cursor
            try:
                while True:
                    events = await store.list_events(
                        normalized.run_id,
                        after_sequence=stream_cursor,
                        limit=500,
                    )
                    for event in events:
                        envelope = RunEventEnvelope(
                            run_id=event.run_id,
                            sequence=event.sequence,
                            event_type=event.event_type.value,
                            timestamp=event.timestamp,
                            payload=dict(event.payload),
                        )
                        payload = {
                            "type": f"response.{event.event_type.value.lower()}",
                            "sequence": event.sequence,
                            "response": envelope.model_dump(mode="json"),
                        }
                        stream_cursor = event.sequence
                        yield (
                            f"id: {event.sequence}\n"
                            f"event: {payload['type']}\n"
                            f"data: {json.dumps(payload, sort_keys=True)}\n\n"
                        ).encode()
                    if len(events) == 500:
                        continue
                    current = await store.get_run(normalized.run_id)
                    if current.status.is_terminal:
                        return
                    hint = await subscription.get()
                    if hint is None:
                        return
            finally:
                await subscription.close()

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    return router
