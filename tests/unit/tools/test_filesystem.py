"""Targeted tests for read-only filesystem tool handlers."""

from pathlib import Path

import pytest

from prp_runtime.domain.enums import ToolEffect
from prp_runtime.domain.values import (
    new_run_id,
    new_snapshot_id,
    new_tool_call_id,
    new_work_unit_id,
)
from prp_runtime.tools import (
    ExecutionContext,
    ToolCall,
    build_filesystem_registry,
)
from prp_runtime.tools.filesystem import build_bridge_registry
from prp_runtime.workspace import WorkspaceBackend


def make_call(tool_name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        call_id=new_tool_call_id(),
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        tool_name=tool_name,
        effect=ToolEffect.READ,
        arguments=arguments,
        snapshot_id=new_snapshot_id(),
        requested_at="2026-08-14T12:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_list_handler_is_sorted_and_paginates_without_absolute_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "c.py").write_text("c", encoding="utf-8")
    with WorkspaceBackend(tmp_path) as backend:
        registry = build_filesystem_registry(backend)
        definition = registry["list_files"]
        call = make_call("list_files", {"offset": 1, "limit": 1})
        arguments = definition.validate_arguments(call.arguments)
        result = await definition.handler(
            ExecutionContext(
                call=call,
                arguments=arguments,
                workspace_id="ws-test",
            )
        )

    assert result["offset"] == 1
    assert result["truncated"] is True
    assert [entry["path"] for entry in result["entries"]] == ["b.py"]
    assert str(tmp_path) not in str(result)


@pytest.mark.asyncio
async def test_read_handler_returns_bounded_text_and_binary_metadata(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("0123456789", encoding="utf-8")
    (tmp_path / "image.bin").write_bytes(b"\x00\xff\x01")
    with WorkspaceBackend(tmp_path) as backend:
        registry = build_filesystem_registry(backend)
        read = registry["read_file"]
        text_call = make_call("read_file", {"path": "note.txt", "offset": 2, "limit": 4})
        text_args = read.validate_arguments(text_call.arguments)
        text_result = await read.handler(
            ExecutionContext(call=text_call, arguments=text_args, workspace_id="ws-test")
        )
        binary_call = make_call("read_file", {"path": "image.bin"})
        binary_args = read.validate_arguments(binary_call.arguments)
        binary_result = await read.handler(
            ExecutionContext(call=binary_call, arguments=binary_args, workspace_id="ws-test")
        )

    assert text_result["content"] == "2345"
    assert text_result["truncated"] is True
    assert binary_result["unsupported"] == "binary"
    assert "content" not in binary_result


def test_filesystem_registry_declares_only_read_effects(tmp_path: Path) -> None:
    with WorkspaceBackend(tmp_path) as backend:
        registry = build_filesystem_registry(backend)
    assert registry.names == ("list_files", "read_file")
    assert all(definition.effect is ToolEffect.READ for definition in registry)


def test_bridge_registry_advertises_closed_local_tools(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("ok\n", encoding="utf-8")
    with WorkspaceBackend(tmp_path) as backend:
        registry = build_bridge_registry(backend, workspace_root=tmp_path)
    required = {
        "list_files",
        "read_file",
        "apply_patch",
        "get_diff",
        "get_status",
        "run_targeted_test",
    }
    assert required <= set(registry.names)
    assert "run_shell" not in registry.names
    assert registry["apply_patch"].effect is ToolEffect.WRITE
    assert registry["run_targeted_test"].effect is ToolEffect.COMMAND
