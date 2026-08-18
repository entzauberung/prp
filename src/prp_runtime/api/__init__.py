"""Inbound HTTP bindings.

Every binding maps onto the native domain. A binding never owns run state and
never extends the domain with transport-specific fields.
"""

from prp_runtime.api.anthropic_messages import create_router as create_anthropic_router
from prp_runtime.api.bindings import (
    BindingNormalizationResult,
    BindingOperation,
    normalize_cancel,
    normalize_query,
    normalize_request,
)
from prp_runtime.api.errors import (
    STATUS_BY_CODE,
    STATUS_BY_FAMILY,
    ErrorResponse,
    error_response,
    install_error_handlers,
    status_for,
)
from prp_runtime.api.native import (
    EVENT_STREAM_MEDIA_TYPE,
    RunEnvelope,
    RunEventEnvelope,
)
from prp_runtime.api.native import create_router as create_native_router
from prp_runtime.api.native_agent import (
    ApprovalDecisionRequest,
    ApprovalView,
)
from prp_runtime.api.native_agent import create_router as create_native_agent_router
from prp_runtime.api.openai_chat import create_router as create_openai_chat_router
from prp_runtime.api.openai_responses import (
    create_router as create_openai_responses_router,
)

create_router = create_native_router

__all__ = [
    "EVENT_STREAM_MEDIA_TYPE",
    "STATUS_BY_CODE",
    "STATUS_BY_FAMILY",
    "BindingNormalizationResult",
    "BindingOperation",
    "ApprovalDecisionRequest",
    "ApprovalView",
    "ErrorResponse",
    "RunEnvelope",
    "RunEventEnvelope",
    "create_anthropic_router",
    "create_native_router",
    "create_native_agent_router",
    "create_openai_chat_router",
    "create_openai_responses_router",
    "create_router",
    "error_response",
    "install_error_handlers",
    "normalize_cancel",
    "normalize_query",
    "normalize_request",
    "status_for",
]
