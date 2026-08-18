"""Read-only filesystem tool definitions backed by WorkspaceBackend."""

from collections.abc import Mapping
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from prp_runtime.domain.enums import ToolEffect
from prp_runtime.tools.executor import ExecutionContext
from prp_runtime.tools.registry import ToolDefinition, ToolHandler, ToolRegistry
from prp_runtime.workspace.backend import WorkspaceBackend

__all__ = [
    "ListFilesArguments",
    "ReadFileArguments",
    "build_filesystem_registry",
    "make_list_files_handler",
    "make_read_file_handler",
]

RelativePathText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
]


class ListFilesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, gt=0, le=1_000)


class ReadFileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: RelativePathText
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=64 * 1024, gt=0, le=256 * 1024)


def make_list_files_handler(backend: WorkspaceBackend) -> ToolHandler:
    """Create a handler bound to one server-authorized backend."""

    async def handler(context: BaseModel) -> Mapping[str, object]:
        if not isinstance(context, ExecutionContext):
            raise TypeError("list_files requires an execution context")
        arguments = context.arguments
        if not isinstance(arguments, ListFilesArguments):
            raise TypeError("list_files received an invalid argument model")
        entries = backend.list_directory(arguments.path)
        page = entries[arguments.offset : arguments.offset + arguments.limit]
        return {
            "path": arguments.path,
            "entries": [entry.model_dump(mode="json") for entry in page],
            "offset": arguments.offset,
            "limit": arguments.limit,
            "truncated": arguments.offset + len(page) < len(entries),
        }

    return cast(ToolHandler, handler)


def make_read_file_handler(backend: WorkspaceBackend) -> ToolHandler:
    """Create a bounded text-read handler bound to one backend."""

    async def handler(context: BaseModel) -> Mapping[str, object]:
        if not isinstance(context, ExecutionContext):
            raise TypeError("read_file requires an execution context")
        arguments = context.arguments
        if not isinstance(arguments, ReadFileArguments):
            raise TypeError("read_file received an invalid argument model")
        result = backend.read_file(
            arguments.path,
            offset=arguments.offset,
            limit=arguments.limit,
        )
        if result.binary:
            return {
                "path": result.path,
                "unsupported": "binary",
                "binary": True,
                "bytes_read": result.bytes_read,
                "offset": result.offset,
                "limit": result.limit,
                "truncated": result.truncated,
            }
        return result.model_dump(mode="json")

    return cast(ToolHandler, handler)


def build_filesystem_registry(backend: WorkspaceBackend) -> ToolRegistry:
    """Build a frozen registry for the first two read-only filesystem tools."""
    return ToolRegistry(
        (
            ToolDefinition(
                name="list_files",
                description="List entries below an authorized relative directory.",
                effect=ToolEffect.READ,
                argument_model=ListFilesArguments,
                handler=make_list_files_handler(backend),
                max_output_bytes=128 * 1024,
            ),
            ToolDefinition(
                name="read_file",
                description="Read bounded text from one authorized relative file.",
                effect=ToolEffect.READ,
                argument_model=ReadFileArguments,
                handler=make_read_file_handler(backend),
                max_output_bytes=256 * 1024,
            ),
        )
    )
