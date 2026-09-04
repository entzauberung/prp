"""Read-only filesystem tool definitions backed by WorkspaceBackend."""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from prp_runtime.domain.enums import IsolationMode, ToolCallStatus, ToolEffect
from prp_runtime.domain.values import utc_now
from prp_runtime.tools.command import (
    DEFAULT_COMMAND_REGISTRY,
    CommandRunner,
    build_targeted_test_definition,
)
from prp_runtime.tools.diff import DeferredDiffRunner, build_diff_definitions
from prp_runtime.tools.executor import ExecutionContext
from prp_runtime.tools.patch import LocalPatchStore, PatchRequest, PatchRunner, PatchStaleError
from prp_runtime.tools.registry import ToolDefinition, ToolHandler, ToolRegistry
from prp_runtime.tools.search import SearchRunner, SearchUnavailableError, build_search_definition
from prp_runtime.workspace.backend import WorkspaceBackend
from prp_runtime.workspace.models import Snapshot, SnapshotStatus

__all__ = [
    "ListFilesArguments",
    "ReadFileArguments",
    "build_bridge_registry",
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


def build_bridge_registry(
    backend: WorkspaceBackend,
    *,
    workspace_root: Path,
) -> ToolRegistry:
    """Compose the closed local Bridge catalog from existing primitives."""
    definitions = list(build_filesystem_registry(backend).definitions)
    try:
        definitions.append(
            build_search_definition(
                SearchRunner(backend, workspace_cwd=workspace_root)
            )
        )
    except SearchUnavailableError:
        pass
    store = LocalPatchStore()
    diff_runner = DeferredDiffRunner(backend.snapshot_manifest(), backend.snapshot_manifest)

    async def apply_patch(context: BaseModel) -> object:
        if not isinstance(context, ExecutionContext):
            raise TypeError("apply_patch requires an execution context")
        snapshot_id = context.call.snapshot_id
        if snapshot_id is None:
            raise PatchStaleError("mutating Bridge claim requires a base snapshot")
        if not isinstance(context.arguments, PatchRequest):
            raise TypeError("apply_patch received an invalid argument model")
        live = backend.snapshot_manifest()
        backend.require_base_manifest(live)
        runner = PatchRunner(
            backend,
            store,
            owner_id="bridge-local",
            base_snapshot=Snapshot(
                snapshot_id=snapshot_id,
                workspace_id=context.workspace_id,
                status=SnapshotStatus.READY,
                created_at=utc_now(),
                completed_at=utc_now(),
                file_count=len(live.entries),
                total_size=live.total_size,
            ),
            base_manifest=live,
        )
        result = await runner.apply(context.call, context.arguments)
        existing = await store.list_change_sets(tool_call_id=context.call.call_id)
        if existing:
            diff_runner.bind(existing[0])
        from prp_runtime.tools.models import ToolResult

        return ToolResult.from_call(
            context.call,
            status=ToolCallStatus.SUCCEEDED,
            result=result.model_dump(mode="json"),
            changed_paths=result.changed_paths,
            completed_at=result.completed_at,
        )

    definitions.append(
        ToolDefinition(
            name="apply_patch",
            description="Apply one validated patch to the authorized workspace.",
            effect=ToolEffect.WRITE,
            argument_model=PatchRequest,
            handler=cast(ToolHandler, apply_patch),
        )
    )
    definitions.extend(build_diff_definitions(diff_runner))
    definitions.append(
        build_targeted_test_definition(
            CommandRunner(
                DEFAULT_COMMAND_REGISTRY,
                workspace_cwd=workspace_root,
                test_only=True,
                isolation_mode=IsolationMode.HOST,
            )
        )
    )
    return ToolRegistry(tuple(definitions))
