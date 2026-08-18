import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from prp_runtime.client.executor import BridgeDispatchError, BridgeExecutor
from prp_runtime.domain.enums import (
    BridgeClaimStatus,
    IsolationMode,
    ToolEffect,
)
from prp_runtime.policy.models import CommandClass
from prp_runtime.tools import ToolDefinition, ToolRegistry, build_filesystem_registry
from prp_runtime.tools.command import (
    CommandInvocation,
    CommandParameter,
    CommandRegistry,
    CommandRunner,
    CommandSpec,
    build_targeted_test_definition,
)
from prp_runtime.workspace import WorkspaceBackend

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)
COMMAND_FIXTURE = Path(__file__).parents[2] / "fixtures" / "command_fixture.py"


def _claim(
    *,
    tool_name: str = "read_file",
    effect: ToolEffect = ToolEffect.READ,
    arguments: dict[str, Any] | None = None,
    paths: tuple[str, ...] = ("**",),
    **overrides: Any,
) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "claim_id": "claim_bridge01",
        "call_id": "tc_bridge01",
        "run_id": "run_bridge01",
        "workspace_id": "ws_bridge01",
        "status": BridgeClaimStatus.ACTIVE.value,
        "tool_name": tool_name,
        "effect": effect.value,
        "arguments": {"path": "src/main.py"} if arguments is None else arguments,
        "scope": {"paths": list(paths)},
        "claimed_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(minutes=5)).isoformat(),
    }
    claim.update(overrides)
    return claim


def test_plan_binds_only_registered_local_capabilities(tmp_path: Path) -> None:
    with WorkspaceBackend(tmp_path) as backend:
        executor = BridgeExecutor(build_filesystem_registry(backend), tmp_path)
        plan = executor.plan(_claim(), now=NOW)

    assert plan.tool_name == "read_file"
    assert plan.workspace_root == tmp_path
    assert plan.workspace_id == "ws_bridge01"
    assert plan.definition.name == "read_file"


def test_executor_requires_an_existing_real_directory(tmp_path: Path) -> None:
    with WorkspaceBackend(tmp_path) as backend:
        registry = build_filesystem_registry(backend)
        with pytest.raises(BridgeDispatchError, match="must exist"):
            BridgeExecutor(registry, tmp_path / "missing")

        file_path = tmp_path / "workspace-file"
        file_path.write_text("not a directory", encoding="utf-8")
        with pytest.raises(BridgeDispatchError, match="must be a directory"):
            BridgeExecutor(registry, file_path)

        link_path = tmp_path / "workspace-link"
        link_path.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(BridgeDispatchError, match="must not be a symlink"):
            BridgeExecutor(registry, link_path)


def test_plan_scopes_unified_diff_headers(tmp_path: Path) -> None:
    class Arguments(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        unified_diff: str

    async def handler(context: Any) -> None:
        del context

    registry = ToolRegistry(
        (
            ToolDefinition(
                name="apply_patch",
                effect=ToolEffect.WRITE,
                argument_model=Arguments,
                handler=handler,
            ),
        )
    )
    executor = BridgeExecutor(registry, tmp_path)
    valid_diff = "--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-old\n+new\n"

    plan = executor.plan(
        _claim(
            tool_name="apply_patch",
            effect=ToolEffect.WRITE,
            arguments={"unified_diff": valid_diff},
            paths=("src/**",),
        ),
        now=NOW,
    )
    assert plan.tool_name == "apply_patch"

    with pytest.raises(BridgeDispatchError, match="outside claim scope"):
        executor.plan(
            _claim(
                tool_name="apply_patch",
                effect=ToolEffect.WRITE,
                arguments={
                    "unified_diff": valid_diff.replace("src/main.py", "tests/test_main.py")
                },
                paths=("src/**",),
            ),
            now=NOW,
        )

    with pytest.raises(BridgeDispatchError, match="missing file headers"):
        executor.plan(
            _claim(
                tool_name="apply_patch",
                effect=ToolEffect.WRITE,
                arguments={"unified_diff": "@@ -1 +1 @@\n-old\n+new\n"},
            ),
            now=NOW,
        )


@pytest.mark.asyncio
async def test_execute_reads_fixture_root_and_returns_bounded_relative_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "main.py").write_text("needle\n", encoding="utf-8")

    with WorkspaceBackend(tmp_path) as backend:
        executor = BridgeExecutor(build_filesystem_registry(backend), tmp_path)
        payload = await executor.execute(_claim(), now=NOW)

    assert payload["status"] == "SUCCEEDED"
    assert payload["result"]["path"] == "src/main.py"
    assert str(tmp_path) not in str(payload)
    assert "call_id" not in payload
    assert "completed_at" not in payload


@pytest.mark.asyncio
async def test_execute_fails_closed_for_traversal_and_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    with WorkspaceBackend(tmp_path) as backend:
        executor = BridgeExecutor(build_filesystem_registry(backend), tmp_path)
        with pytest.raises(BridgeDispatchError, match="relative"):
            executor.plan(
                _claim(arguments={"path": "../secret.txt"}),
                now=NOW,
            )
        failed = await executor.execute(
            _claim(arguments={"path": "escape/secret.txt"}),
            now=NOW,
        )

    assert failed["status"] == "FAILED"
    assert failed["error"]["category"] == "INTERNAL"
    assert "secret" not in str(failed)


@pytest.mark.asyncio
async def test_execute_redacts_root_and_bounds_handler_output(tmp_path: Path) -> None:
    class Arguments(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        path: str

    async def handler(context: BaseModel) -> str:
        del context
        return f"leaked={tmp_path}/secret.txt " + ("x" * 500)

    registry = ToolRegistry(
        (
            ToolDefinition(
                name="emit_output",
                effect=ToolEffect.READ,
                argument_model=Arguments,
                handler=handler,
                max_output_bytes=64,
            ),
        )
    )
    executor = BridgeExecutor(registry, tmp_path)

    payload = await executor.execute(
        _claim(tool_name="emit_output", arguments={"path": "note.txt"}),
        now=NOW,
    )

    assert payload["status"] == "SUCCEEDED"
    assert payload["truncated"] is True
    assert str(tmp_path) not in str(payload)
    assert len(payload["output"].encode("utf-8")) <= 64


@pytest.mark.asyncio
async def test_handler_exception_and_timeout_are_safe_failures(tmp_path: Path) -> None:
    class Arguments(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        path: str

    async def raises(context: BaseModel) -> None:
        del context
        raise RuntimeError("private detail")

    async def hangs(context: BaseModel) -> None:
        del context
        await __import__("asyncio").sleep(1)

    registry = ToolRegistry(
        (
            ToolDefinition(
                name="raises",
                effect=ToolEffect.READ,
                argument_model=Arguments,
                handler=raises,
            ),
            ToolDefinition(
                name="hangs",
                effect=ToolEffect.READ,
                argument_model=Arguments,
                handler=hangs,
            ),
        )
    )
    executor = BridgeExecutor(registry, tmp_path, timeout_seconds=0.01)

    failed = await executor.execute(
        _claim(tool_name="raises", arguments={"path": "note.txt"}),
        now=NOW,
    )
    timed_out = await executor.execute(
        _claim(tool_name="hangs", arguments={"path": "note.txt"}),
        now=NOW,
    )

    assert failed["status"] == "FAILED"
    assert failed["error"]["message"] == "local tool handler failed"
    assert timed_out["status"] == "FAILED"
    assert timed_out["error"]["category"] == "TIMEOUT"


@pytest.mark.asyncio
async def test_registered_command_is_shell_free_and_redacts_workspace_output(
    tmp_path: Path,
) -> None:
    specification = CommandSpec(
        name="fixture",
        executable=sys.executable,
        argv_template=(str(COMMAND_FIXTURE), "{mode}"),
        parameters=(CommandParameter(name="mode"),),
        command_class=CommandClass.TEST,
        timeout_seconds=2,
        max_output_bytes=1024,
    )
    runner = CommandRunner(
        CommandRegistry((specification,)),
        workspace_cwd=tmp_path,
        test_only=True,
        isolation_mode=IsolationMode.HOST,
    )
    registry = ToolRegistry((build_targeted_test_definition(runner),))
    executor = BridgeExecutor(registry, tmp_path)

    payload = await executor.execute(
        _claim(
            tool_name="run_targeted_test",
            effect=ToolEffect.COMMAND,
            arguments={"spec_name": "fixture", "parameters": {"mode": "success"}},
        ),
        now=NOW,
    )

    assert payload["status"] == "SUCCEEDED"
    assert str(tmp_path) not in str(payload)
    assert "shell" not in str(payload).lower()


@pytest.mark.asyncio
async def test_untrusted_command_token_cannot_create_a_side_effect(tmp_path: Path) -> None:
    specification = CommandSpec(
        name="fixture",
        executable=sys.executable,
        argv_template=(str(COMMAND_FIXTURE), "{mode}"),
        parameters=(CommandParameter(name="mode"),),
        command_class=CommandClass.TEST,
        timeout_seconds=2,
        max_output_bytes=1024,
    )
    runner = CommandRunner(
        CommandRegistry((specification,)),
        workspace_cwd=tmp_path,
        test_only=True,
        isolation_mode=IsolationMode.HOST,
    )
    executor = BridgeExecutor(
        ToolRegistry((build_targeted_test_definition(runner),)), tmp_path
    )
    marker = tmp_path / "command-side-effect"
    injected = f"success;touch {marker}"

    payload = await executor.execute(
        _claim(
            tool_name="run_targeted_test",
            effect=ToolEffect.COMMAND,
            arguments={"spec_name": "fixture", "parameters": {"mode": injected}},
        ),
        now=NOW,
    )

    assert payload["status"] == "FAILED"
    assert marker.exists() is False
    assert str(marker) not in str(payload)


def test_unapproved_effect_cannot_reach_a_registered_command(tmp_path: Path) -> None:
    async def handler(context: BaseModel) -> None:
        del context
        raise AssertionError("command handler must not be reached")

    registry = ToolRegistry(
        (
            ToolDefinition(
                name="run_targeted_test",
                effect=ToolEffect.COMMAND,
                argument_model=CommandInvocation,
                handler=handler,
            ),
        )
    )
    executor = BridgeExecutor(registry, tmp_path)

    with pytest.raises(BridgeDispatchError, match="effect does not match"):
        executor.plan(
            _claim(
                tool_name="run_targeted_test",
                effect=ToolEffect.READ,
                arguments={"spec_name": "fixture", "parameters": {"mode": "success"}},
            ),
            now=NOW,
        )


def test_plan_rejects_unknown_tool_and_effect_mismatch(tmp_path: Path) -> None:
    with WorkspaceBackend(tmp_path) as backend:
        executor = BridgeExecutor(build_filesystem_registry(backend), tmp_path)
        with pytest.raises(BridgeDispatchError, match="unknown local tool"):
            executor.plan(_claim(tool_name="run_shell"), now=NOW)
        with pytest.raises(BridgeDispatchError, match="effect does not match"):
            executor.plan(_claim(effect=ToolEffect.WRITE), now=NOW)
