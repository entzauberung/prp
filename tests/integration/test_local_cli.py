"""Server-free local CLI one-shot with a fake provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from prp_runtime.client import cli
from prp_runtime.client import local_cli
from prp_runtime.domain.enums import IsolationMode, ModelRole, RunStatus, ToolEffect
from prp_runtime.domain.errors import DomainValidationError, ErrorCode, ProviderError
from prp_runtime.domain.models import AgentToolCall, Usage
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.runtime.local import LocalRuntime
from prp_runtime.settings import Settings
from prp_runtime.workspace.isolation import ExecutionCopyMode

WORKER_PROFILE = ModelProfile(
    alias="worker",
    provider="openai_compatible",
    model="weak-model",
    role=ModelRole.WORKER,
    base_url="https://models.invalid/v1",
    context_window_tokens=32_000,
    max_output_tokens=4_000,
)


class FakeAdapter:
    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "local-cli-fake"

    async def aclose(self) -> None:
        return None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, ProviderResponse):
            return outcome
        return ProviderResponse(
            text="the local answer",
            usage=Usage(input_tokens=4, output_tokens=3, elapsed_ms=2),
            finish_reason=FinishReason.STOP,
        )


class PatchAskAdapter:
    def __init__(self, runtime_holder: list[LocalRuntime]) -> None:
        self.requests: list[ProviderRequest] = []
        self._runtime_holder = runtime_holder

    @property
    def name(self) -> str:
        return "local-cli-patch"

    async def aclose(self) -> None:
        return None

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            snapshot_id = self._runtime_holder[0].last_snapshot_id
            assert snapshot_id is not None
            return ProviderResponse(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-patch",
                        tool_name="apply_patch",
                        arguments={
                            "patch": {
                                "base_snapshot_id": snapshot_id,
                                "unified_diff": (
                                    "--- a/README.md\n"
                                    "+++ b/README.md\n"
                                    "@@ -1 +1 @@\n"
                                    "-hello\n"
                                    "+patched\n"
                                ),
                            }
                        },
                    ),
                ),
                usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        return ProviderResponse(
            text="patched the file",
            usage=Usage(input_tokens=1, output_tokens=1, elapsed_ms=1),
            finish_reason=FinishReason.STOP,
        )


def _args(prompt: str, workspace: Path, **updates: object) -> argparse.Namespace:
    values = {
        "prompt": prompt,
        "workspace": workspace,
        "agent_mode": "NORMAL",
        "isolation_mode": "HOST",
        "user_explicit": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "local-cli.db",
        worker_profile=WORKER_PROFILE,
    )


@pytest.mark.asyncio
async def test_local_cli_success_does_not_contact_http(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    adapter = FakeAdapter()
    code = await local_cli.dispatch_local_run(
        _args("summarise", workspace),
        settings=_settings(tmp_path),
        adapters={"worker": adapter},
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["status"] == RunStatus.SUCCEEDED.value
    assert payload["output_text"] == "the local answer"
    assert str(workspace) not in captured.out
    assert "secret" not in captured.out
    assert len(adapter.requests) == 1
    dumped = adapter.requests[0].model_dump_json()
    assert "http://" not in dumped
    assert "https://" not in dumped
    assert "api_key" not in dumped
    assert str(workspace) not in dumped


@pytest.mark.asyncio
async def test_local_cli_isolation_mode_is_honored_or_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    adapter = FakeAdapter()
    holder: list[LocalRuntime] = []
    original_build = local_cli._build_runtime

    def build(runtime_settings: Settings, *, adapters=None) -> LocalRuntime:
        runtime = original_build(runtime_settings, adapters=adapters)
        holder.append(runtime)
        return runtime

    local_cli._build_runtime = build  # type: ignore[method-assign]
    try:
        code = await local_cli.dispatch_local_run(
            _args("summarise", workspace, isolation_mode="HOST"),
            settings=_settings(tmp_path),
            adapters={"worker": adapter},
        )
        captured = capsys.readouterr()
        assert code == 0
        assert json.loads(captured.out)["status"] == RunStatus.SUCCEEDED.value
        assert holder[-1].last_copy_mode is ExecutionCopyMode.IN_PLACE
        assert holder[-1].defaults.isolation_mode is IsolationMode.HOST
        with pytest.raises(DomainValidationError) as raised:
            await local_cli.dispatch_local_run(
                _args("summarise", workspace, isolation_mode="SANDBOXED"),
                settings=_settings(tmp_path),
                adapters={"worker": adapter},
            )
        assert raised.value.code is ErrorCode.INVALID_AGENT_OPTIONS
        assert raised.value.detail.field == "execution_copy_mode"
        assert (workspace / "README.md").read_text(encoding="utf-8") == "hello\n"
        assert str(workspace) not in capsys.readouterr().out
    finally:
        local_cli._build_runtime = original_build  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_local_cli_failure_and_paused_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    failing = FakeAdapter(ProviderError("upstream timed out", code=ErrorCode.PROVIDER_TIMEOUT))
    failed = await local_cli.dispatch_local_run(
        _args("hello", workspace),
        settings=_settings(tmp_path / "fail"),
        adapters={"worker": failing},
    )
    failed_payload = json.loads(capsys.readouterr().out)
    assert failed == 1
    assert failed_payload["status"] == RunStatus.FAILED.value
    assert failed_payload["error"] is not None

    holder: list[LocalRuntime] = []
    ask = PatchAskAdapter(holder)
    original_build = local_cli._build_runtime

    def build(settings: Settings, *, adapters=None) -> LocalRuntime:
        runtime = original_build(settings, adapters=adapters)
        holder.append(runtime)
        return runtime

    local_cli._build_runtime = build  # type: ignore[method-assign]
    try:
        paused = await local_cli.dispatch_local_run(
            _args("patch", workspace),
            settings=_settings(tmp_path / "ask"),
            adapters={"worker": ask},
        )
    finally:
        local_cli._build_runtime = original_build  # type: ignore[method-assign]
    paused_payload = json.loads(capsys.readouterr().out)
    assert paused == 2
    assert paused_payload["status"] == RunStatus.RUNNING.value
    assert paused_payload["pending_approvals"][0]["tool_name"] == "apply_patch"
    assert paused_payload["pending_approvals"][0]["effect"] == ToolEffect.WRITE.value
    assert (workspace / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert str(workspace) not in paused_payload["pending_approvals"][0]["request_id"]


def test_missing_workspace_fails_before_model_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing"
    adapter = FakeAdapter()
    code = cli.main(
        ["local", "run", "hello", "--workspace", str(missing)]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert adapter.requests == []
    assert "unavailable" in captured.err
    assert str(missing) not in captured.err
    assert str(missing) not in captured.out


@pytest.mark.asyncio
async def test_local_cli_approve_deny_replay_and_wrong_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    settings = _settings(tmp_path)
    holder: list[LocalRuntime] = []
    adapter = PatchAskAdapter(holder)
    original_build = local_cli._build_runtime

    def build(runtime_settings: Settings, *, adapters=None) -> LocalRuntime:
        runtime = original_build(runtime_settings, adapters=adapters)
        holder.append(runtime)
        return runtime

    local_cli._build_runtime = build  # type: ignore[method-assign]
    try:
        paused = await local_cli.dispatch_local_run(
            _args("patch", workspace),
            settings=settings,
            adapters={"worker": adapter},
        )
        paused_payload = json.loads(capsys.readouterr().out)
        assert paused == 2
        assert paused_payload["status"] == RunStatus.RUNNING.value
        assert (workspace / "README.md").read_text(encoding="utf-8") == "hello\n"
        request_id = paused_payload["pending_approvals"][0]["request_id"]

        with pytest.raises(ValueError, match="not available"):
            await local_cli.dispatch_local_approve(
                argparse.Namespace(
                    request_id="apr_other",
                    workspace=workspace,
                    reason=None,
                ),
                settings=settings,
                adapters={"worker": adapter},
            )

        approved = await local_cli.dispatch_local_approve(
            argparse.Namespace(
                request_id=request_id,
                workspace=workspace,
                reason=None,
            ),
            settings=settings,
            adapters={"worker": adapter},
        )
        approved_payload = json.loads(capsys.readouterr().out)
        assert approved == 0
        assert approved_payload["status"] == RunStatus.SUCCEEDED.value
        assert approved_payload["run_id"] == paused_payload["run_id"]
        assert (workspace / "README.md").read_text(encoding="utf-8") == "patched\n"

        replayed = await local_cli.dispatch_local_approve(
            argparse.Namespace(
                request_id=request_id,
                workspace=workspace,
                reason=None,
            ),
            settings=settings,
            adapters={"worker": adapter},
        )
        replayed_payload = json.loads(capsys.readouterr().out)
        assert replayed == 0
        assert replayed_payload["status"] == RunStatus.SUCCEEDED.value
        assert (workspace / "README.md").read_text(encoding="utf-8") == "patched\n"
    finally:
        local_cli._build_runtime = original_build  # type: ignore[method-assign]

    deny_workspace = tmp_path / "deny"
    deny_workspace.mkdir()
    (deny_workspace / "README.md").write_text("hello\n", encoding="utf-8")
    deny_settings = Settings(
        database_path=tmp_path / "local-cli-deny.db",
        worker_profile=WORKER_PROFILE,
    )
    deny_holder: list[LocalRuntime] = []
    deny_adapter = PatchAskAdapter(deny_holder)

    def deny_build(runtime_settings: Settings, *, adapters=None) -> LocalRuntime:
        runtime = original_build(runtime_settings, adapters=adapters)
        deny_holder.append(runtime)
        return runtime

    local_cli._build_runtime = deny_build  # type: ignore[method-assign]
    try:
        paused = await local_cli.dispatch_local_run(
            _args("patch", deny_workspace),
            settings=deny_settings,
            adapters={"worker": deny_adapter},
        )
        paused_payload = json.loads(capsys.readouterr().out)
        assert paused == 2
        request_id = paused_payload["pending_approvals"][0]["request_id"]
        denied = await local_cli.dispatch_local_deny(
            argparse.Namespace(
                request_id=request_id,
                workspace=deny_workspace,
                reason="not wanted",
            ),
            settings=deny_settings,
            adapters={"worker": deny_adapter},
        )
        denied_payload = json.loads(capsys.readouterr().out)
        assert denied in {0, 1}
        assert denied_payload["status"] != RunStatus.RUNNING.value
        assert (deny_workspace / "README.md").read_text(encoding="utf-8") == "hello\n"
    finally:
        local_cli._build_runtime = original_build  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_local_cli_fresh_approve_rejects_wrong_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "README.md").write_text("other\n", encoding="utf-8")
    settings = _settings(tmp_path)
    holder: list[LocalRuntime] = []
    adapter = PatchAskAdapter(holder)
    original_build = local_cli._build_runtime

    def build(runtime_settings: Settings, *, adapters=None) -> LocalRuntime:
        runtime = original_build(runtime_settings, adapters=adapters)
        holder.append(runtime)
        return runtime

    local_cli._build_runtime = build  # type: ignore[method-assign]
    try:
        paused = await local_cli.dispatch_local_run(
            _args("patch", workspace),
            settings=settings,
            adapters={"worker": adapter},
        )
        paused_payload = json.loads(capsys.readouterr().out)
        assert paused == 2
        request_id = paused_payload["pending_approvals"][0]["request_id"]
        holder.clear()
        with pytest.raises(ValueError, match="not available"):
            await local_cli.dispatch_local_approve(
                argparse.Namespace(
                    request_id=request_id,
                    workspace=other,
                    reason=None,
                ),
                settings=settings,
                adapters={"worker": adapter},
            )
        assert (workspace / "README.md").read_text(encoding="utf-8") == "hello\n"
        approved = await local_cli.dispatch_local_approve(
            argparse.Namespace(
                request_id=request_id,
                workspace=workspace,
                reason=None,
            ),
            settings=settings,
            adapters={"worker": adapter},
        )
        approved_out = capsys.readouterr().out
        approved_payload = json.loads(approved_out)
        assert approved == 0
        assert approved_payload["status"] == RunStatus.SUCCEEDED.value
        assert approved_payload["run_id"] == paused_payload["run_id"]
        assert (workspace / "README.md").read_text(encoding="utf-8") == "patched\n"
        assert str(workspace) not in approved_out
        replayed = await local_cli.dispatch_local_approve(
            argparse.Namespace(
                request_id=request_id,
                workspace=workspace,
                reason=None,
            ),
            settings=settings,
            adapters={"worker": adapter},
        )
        replayed_out = capsys.readouterr().out
        replayed_payload = json.loads(replayed_out)
        assert replayed == 0
        assert replayed_payload["status"] == RunStatus.SUCCEEDED.value
        assert (workspace / "README.md").read_text(encoding="utf-8") == "patched\n"
        assert str(workspace) not in replayed_out
    finally:
        local_cli._build_runtime = original_build  # type: ignore[method-assign]

