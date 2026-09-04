"""Protocol-independent tool contracts."""

from typing import Any

from prp_runtime.tools.executor import (
    ExecutionContext,
    ToolExecutionError,
    ToolExecutionOutcome,
    ToolExecutor,
    ToolStore,
)
from prp_runtime.tools.models import (
    MAX_CHANGED_PATHS,
    MAX_TOOL_ARGUMENT_BYTES,
    MAX_TOOL_OUTPUT_BYTES,
    ToolCall,
    ToolResult,
)
from prp_runtime.tools.registry import ToolDefinition, ToolHandler, ToolRegistry
from prp_runtime.tools.search import (
    SearchMatch,
    SearchRequest,
    SearchResult,
    SearchUnavailableError,
    build_rg_argv,
    require_rg,
    resolve_search_root,
)

__all__ = [
    "MAX_CHANGED_PATHS",
    "MAX_TOOL_ARGUMENT_BYTES",
    "MAX_TOOL_OUTPUT_BYTES",
    "ToolCall",
    "ToolDefinition",
    "ExecutionContext",
    "ListFilesArguments",
    "ReadFileArguments",
    "SearchMatch",
    "SearchRequest",
    "SearchResult",
    "SearchUnavailableError",
    "ToolHandler",
    "ToolExecutionError",
    "ToolExecutionOutcome",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "ToolStore",
    "build_filesystem_registry",
    "build_rg_argv",
    "make_list_files_handler",
    "make_read_file_handler",
    "require_rg",
    "resolve_search_root",
]

_FILESYSTEM_EXPORTS = frozenset(
    {
        "ListFilesArguments",
        "ReadFileArguments",
        "build_filesystem_registry",
        "make_list_files_handler",
        "make_read_file_handler",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _FILESYSTEM_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from prp_runtime.tools.filesystem import (
        ListFilesArguments,
        ReadFileArguments,
        build_filesystem_registry,
        make_list_files_handler,
        make_read_file_handler,
    )

    globals()["ListFilesArguments"] = ListFilesArguments
    globals()["ReadFileArguments"] = ReadFileArguments
    globals()["build_filesystem_registry"] = build_filesystem_registry
    globals()["make_list_files_handler"] = make_list_files_handler
    globals()["make_read_file_handler"] = make_read_file_handler
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(__all__)
