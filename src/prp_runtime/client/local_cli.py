"""In-process local CLI dispatch, separate from the Bridge command path."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from prp_runtime.domain.enums import AgentMode, IsolationMode, RunStatus
from prp_runtime.providers.base import ProviderAdapter
from prp_runtime.runtime.local import LocalRuntime
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import MissingEntityError
from prp_runtime.workspace.local import canonicalize_local_root, resolve_local_workspace
from prp_runtime.workspace.resolver import WorkspaceResolveError

__all__ = ["dispatch_local_approve", "dispatch_local_deny", "dispatch_local_run"]

_SUCCESS = 0
_FAILURE = 1
_PAUSED = 2


def _workspace_from_args(args: argparse.Namespace) -> Path:
    raw = Path.cwd() if args.workspace is None else args.workspace
    return canonicalize_local_root(raw)


def _redact(text: str, workspace: Path) -> str:
    redacted = text.replace(str(workspace), "<workspace>")
    home = str(Path.home())
    if home:
        redacted = redacted.replace(home, "<home>")
    return redacted


def _build_runtime(
    settings: Settings,
    *,
    adapters: Mapping[str, ProviderAdapter] | None = None,
) -> LocalRuntime:
    return LocalRuntime(settings, adapters=adapters)


async def dispatch_local_run(
    args: argparse.Namespace,
    *,
    stdout: TextIO | None = None,
    settings: Settings | None = None,
    adapters: Mapping[str, ProviderAdapter] | None = None,
) -> int:
    """Run one prompt through LocalRuntime without Bridge, HTTP or tokens."""
    output = stdout or sys.stdout
    try:
        workspace = _workspace_from_args(args)
        handle = resolve_local_workspace(workspace)
    except WorkspaceResolveError as error:
        fallback = Path.cwd() if args.workspace is None else Path(args.workspace)
        raise ValueError(_redact(str(error), fallback)) from error
    handle.close()
    runtime_settings = settings or Settings.from_env()
    async with _build_runtime(runtime_settings, adapters=adapters) as runtime:
        result = await runtime.run(
            args.prompt,
            workspace=workspace,
            agent_mode=AgentMode(args.agent_mode),
            isolation_mode=IsolationMode(
                getattr(args, "isolation_mode", IsolationMode.HOST)
            ),
            user_explicit=bool(getattr(args, "user_explicit", False)),
        )
        pending = []
        if result.status is RunStatus.RUNNING:
            pending = [
                {
                    "request_id": approval.request_id,
                    "tool_name": approval.tool_name,
                    "effect": approval.effect.value,
                }
                for approval in await runtime.pending_approvals(run_id=result.run_id)
            ]
    error = None
    if result.error is not None:
        error = result.error.model_dump(mode="json")
        message = error.get("message")
        if isinstance(message, str):
            error["message"] = _redact(message, workspace)
    payload = {
        "run_id": result.run_id,
        "status": result.status.value,
        "output_text": result.output_text,
        "error": error,
    }
    if pending:
        payload["pending_approvals"] = pending
    dumped = json.dumps(payload)
    output.write(_redact(dumped, workspace) + "\n")
    if result.status is RunStatus.SUCCEEDED:
        return _SUCCESS
    if result.status is RunStatus.RUNNING:
        return _PAUSED
    return _FAILURE


async def dispatch_local_approve(
    args: argparse.Namespace,
    *,
    stdout: TextIO | None = None,
    settings: Settings | None = None,
    adapters: Mapping[str, ProviderAdapter] | None = None,
) -> int:
    """Continue one owner-scoped local ASK request with ALLOW."""
    return await _dispatch_local_decision(
        args,
        outcome="ALLOW",
        stdout=stdout,
        settings=settings,
        adapters=adapters,
    )


async def dispatch_local_deny(
    args: argparse.Namespace,
    *,
    stdout: TextIO | None = None,
    settings: Settings | None = None,
    adapters: Mapping[str, ProviderAdapter] | None = None,
) -> int:
    """Continue one owner-scoped local ASK request with DENY."""
    return await _dispatch_local_decision(
        args,
        outcome="DENY",
        stdout=stdout,
        settings=settings,
        adapters=adapters,
    )


async def _dispatch_local_decision(
    args: argparse.Namespace,
    *,
    outcome: str,
    stdout: TextIO | None,
    settings: Settings | None,
    adapters: Mapping[str, ProviderAdapter] | None,
) -> int:
    output = stdout or sys.stdout
    try:
        workspace = _workspace_from_args(args)
        handle = resolve_local_workspace(workspace)
    except WorkspaceResolveError as error:
        fallback = Path.cwd() if args.workspace is None else Path(args.workspace)
        raise ValueError(_redact(str(error), fallback)) from error
    handle.close()
    runtime_settings = settings or Settings.from_env()
    async with _build_runtime(runtime_settings, adapters=adapters) as runtime:
        workspace_id = runtime.bind_workspace(workspace)
        try:
            if outcome == "ALLOW":
                result = await runtime.approve(
                    args.request_id,
                    workspace_id=workspace_id,
                    reason=getattr(args, "reason", None),
                )
            else:
                result = await runtime.deny(
                    args.request_id,
                    workspace_id=workspace_id,
                    reason=args.reason,
                )
        except MissingEntityError as error:
            raise ValueError("approval request is not available") from error
        pending = []
        if result.status is RunStatus.RUNNING:
            pending = [
                {
                    "request_id": approval.request_id,
                    "tool_name": approval.tool_name,
                    "effect": approval.effect.value,
                }
                for approval in await runtime.pending_approvals(run_id=result.run_id)
            ]
    error = None
    if result.error is not None:
        error = result.error.model_dump(mode="json")
        message = error.get("message")
        if isinstance(message, str):
            error["message"] = _redact(message, workspace)
    payload = {
        "run_id": result.run_id,
        "status": result.status.value,
        "output_text": result.output_text,
        "error": error,
    }
    if pending:
        payload["pending_approvals"] = pending
    output.write(_redact(json.dumps(payload), workspace) + "\n")
    if result.status is RunStatus.SUCCEEDED:
        return _SUCCESS
    if result.status is RunStatus.RUNNING:
        return _PAUSED
    return _FAILURE
