"""Standard-library CLI for the model-free local Workspace Bridge."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prp_runtime.client.bridge import (
    Bridge,
    BridgeError,
    BridgeToolLoopLimits,
    BridgeToolLoopPhase,
)
from prp_runtime.client.executor import BridgeExecutor
from prp_runtime.tools.filesystem import build_bridge_registry
from prp_runtime.workspace import WorkspaceBackend

__all__ = ["build_parser", "main"]

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_STATE_PATH = Path.home() / ".prp" / "bridge-state.json"
_SECRET_WORDS = frozenset({"api_key", "apikey", "password", "secret", "token"})


class CliError(RuntimeError):
    """A safe, user-facing command error."""


def _choices(name: str) -> tuple[str, ...]:
    return {
        "agent": ("NORMAL", "AUTO", "PLAN", "YOLO"),
        "isolation": ("SANDBOXED", "HOST"),
        "access": ("READ", "WRITE"),
    }[name]


def _add_connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=os.environ.get("PRP_BASE_URL", _DEFAULT_BASE_URL))
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(os.environ.get("PRP_BRIDGE_STATE", str(_DEFAULT_STATE_PATH))),
        help="local resume state file; it never contains the bearer token",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="local path boundary only; the absolute path is never sent to PRP",
    )
    parser.add_argument(
        "--token-stdin",
        action="store_true",
        help="read the bearer token from stdin without echoing it",
    )


def _add_agent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-mode", choices=_choices("agent"), default="NORMAL")
    parser.add_argument("--isolation-mode", choices=_choices("isolation"), default="SANDBOXED")


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without reading environment secrets or starting I/O."""
    parser = argparse.ArgumentParser(
        prog="prp",
        description=(
            "Model-free PRP Workspace Bridge. The local workspace-root is a path "
            "boundary, not an operating-system sandbox."
        ),
    )
    _add_connection_options(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    bootstrap = commands.add_parser(
        "bootstrap",
        help="register a model-free client and handshake local capabilities",
    )
    bootstrap.add_argument("workspace_id")
    bootstrap.add_argument("--client-id", default=None)

    connect = commands.add_parser("connect", help="create an authorized Workspace session")
    connect.add_argument("workspace_id")
    connect.add_argument("--access", choices=_choices("access"), nargs="+", default=["READ"])
    _add_agent_options(connect)

    run = commands.add_parser("run", help="submit one prompt to the saved session")
    run.add_argument("input_text", help="prompt text; local absolute paths are rejected")
    _add_agent_options(run)

    watch = commands.add_parser("watch", help="watch the saved run and resume its event cursor")
    watch.add_argument("--max-reconnects", type=int, default=3)
    watch.add_argument("--max-calls", type=int, default=32)
    watch.add_argument("--max-events", type=int, default=256)
    watch.add_argument("--max-seconds", type=float, default=300.0)
    _add_agent_options(watch)

    resume = commands.add_parser("resume", help="resume watching the saved run")
    resume.add_argument("--max-reconnects", type=int, default=3)
    resume.add_argument("--max-calls", type=int, default=32)
    resume.add_argument("--max-events", type=int, default=256)
    resume.add_argument("--max-seconds", type=float, default=300.0)
    _add_agent_options(resume)

    approve = commands.add_parser("approve", help="approve one pending tool request")
    approve.add_argument("request_id")
    approve.add_argument("--reason", default=None)

    deny = commands.add_parser("deny", help="deny one pending tool request")
    deny.add_argument("request_id")
    deny.add_argument("--reason", required=True)

    local = commands.add_parser(
        "local",
        help="run one in-process local prompt without HTTP or Bridge",
    )
    local_commands = local.add_subparsers(dest="local_command", required=True)
    local_run = local_commands.add_parser(
        "run",
        help=(
            "execute one prompt in-process through LocalRuntime; HOST isolation "
            "is a path-boundary only, not an operating-system sandbox"
        ),
        description=(
            "Execute one prompt in-process through LocalRuntime. HOST isolation "
            "is a path-boundary only, not an operating-system sandbox."
        ),
    )
    local_run.add_argument("prompt", help="prompt text executed in-process")
    local_run.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="local workspace directory; defaults to the current directory",
    )
    local_run.add_argument("--agent-mode", choices=_choices("agent"), default="NORMAL")
    local_run.add_argument(
        "--isolation-mode",
        choices=_choices("isolation"),
        default="HOST",
        help="HOST is a path-boundary only; local execution stays in-process",
    )
    local_run.add_argument(
        "--user-explicit",
        action="store_true",
        help="record explicit user intent for HOST YOLO",
    )

    local_approve = local_commands.add_parser(
        "approve",
        help="approve one paused local tool request in-process",
        description="Approve one durable local ASK request without HTTP or Bridge.",
    )
    local_approve.add_argument("request_id")
    local_approve.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="local workspace directory; defaults to the current directory",
    )
    local_approve.add_argument("--reason", default=None)

    local_deny = local_commands.add_parser(
        "deny",
        help="deny one paused local tool request in-process",
        description="Deny one durable local ASK request without HTTP or Bridge.",
    )
    local_deny.add_argument("request_id")
    local_deny.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="local workspace directory; defaults to the current directory",
    )
    local_deny.add_argument("--reason", required=True)

    serve = commands.add_parser(
        "serve",
        help="serve the existing ASGI app on loopback for local programs",
        description=(
            "Optional loopback HTTP surface over create_app. Default host is "
            "127.0.0.1; a wider bind must be explicit. Local run does not use this command."
        ),
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _read_token(*, from_stdin: bool) -> str:
    if not from_stdin:
        token = os.environ.get("PRP_TOKEN")
        if token:
            return token.strip()
    if not sys.stdin.isatty():
        token = sys.stdin.readline().strip()
        if token:
            return token
        raise CliError("no bearer token supplied; use PRP_TOKEN or --token-stdin")
    try:
        token = getpass.getpass("PRP token: ")
    except (EOFError, KeyboardInterrupt) as error:
        raise CliError("no bearer token supplied") from error
    if not token.strip():
        raise CliError("no bearer token supplied")
    return token.strip()


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if str(key).lower() in _SECRET_WORDS else _redact(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _print_json(value: Any) -> None:
    print(json.dumps(_redact(value), ensure_ascii=True, sort_keys=True))


def _agent_options(
    args: argparse.Namespace, *, user_explicit: bool = False
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "agent_mode": args.agent_mode,
        "isolation_mode": args.isolation_mode,
        "execution_location": "BRIDGE",
        "user_explicit": user_explicit,
    }
    return options


def _confirm_host_yolo(args: argparse.Namespace) -> bool:
    if args.agent_mode != "YOLO" or args.isolation_mode != "HOST":
        return False
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise CliError("HOST YOLO requires an interactive confirmation")
    try:
        answer = input('Type "HOST YOLO" to confirm host execution: ')
    except (EOFError, KeyboardInterrupt) as error:
        raise CliError("HOST YOLO confirmation was not provided") from error
    if answer.strip() != "HOST YOLO":
        raise CliError("HOST YOLO confirmation was not provided")
    return True


def _bridge(args: argparse.Namespace, token: str) -> Bridge:
    return Bridge(
        args.base_url,
        token,
        state_path=args.state,
        workspace_root=args.workspace_root,
    )


def _local_executor(args: argparse.Namespace) -> tuple[BridgeExecutor, WorkspaceBackend]:
    if args.workspace_root is None:
        raise CliError("this command requires --workspace-root for local tool execution")
    root = args.workspace_root.expanduser().resolve()
    backend = WorkspaceBackend(root)
    try:
        registry = build_bridge_registry(backend, workspace_root=root)
        return BridgeExecutor(registry, root), backend
    except BaseException:
        backend.close()
        raise


async def _bootstrap(args: argparse.Namespace) -> None:
    token = _read_token(from_stdin=args.token_stdin)
    executor, backend = _local_executor(args)
    try:
        from prp_runtime.domain.models import new_client_id

        client_id = args.client_id or new_client_id()
        request = executor.handshake_request(client_id, workspace_id=args.workspace_id)
        payload = request.model_dump(mode="json")
        async with _bridge(args, token) as bridge:
            await bridge.register_client(
                {
                    "client_id": payload["client_id"],
                    "workspace_id": payload["workspace_id"],
                    "fingerprint": payload["fingerprint"],
                    "capabilities": payload["capabilities"],
                }
            )
            response = await bridge.handshake(payload)
    finally:
        backend.close()
    public = {
        key: response[key]
        for key in ("accepted", "client_id", "protocol_version")
        if key in response
    }
    _print_json(public)


async def _connect(args: argparse.Namespace) -> None:
    user_explicit = _confirm_host_yolo(args)
    token = _read_token(from_stdin=args.token_stdin)
    async with _bridge(args, token) as bridge:
        response = await bridge.create_session(
            args.workspace_id,
            access=args.access,
            agent_options=_agent_options(args, user_explicit=user_explicit),
        )
    _print_json(response)


async def _run(args: argparse.Namespace) -> None:
    user_explicit = _confirm_host_yolo(args)
    token = _read_token(from_stdin=args.token_stdin)
    async with _bridge(args, token) as bridge:
        response = await bridge.create_run(
            None,
            {
                "input": args.input_text,
                "agent_options": _agent_options(args, user_explicit=user_explicit),
            },
        )
    _print_json(response)


async def _watch(args: argparse.Namespace) -> None:
    _confirm_host_yolo(args)
    token = _read_token(from_stdin=args.token_stdin)
    async with _bridge(args, token) as bridge:
        executor, backend = _local_executor(args)
        if args.max_reconnects < 0 or args.max_calls <= 0 or args.max_events <= 0 or args.max_seconds <= 0:
            raise CliError("watch/resume reconnect and loop limits must be finite and non-negative")
        try:
            outcome = await bridge.run_tool_loop(
                executor,
                limits=BridgeToolLoopLimits(
                    max_calls=args.max_calls,
                    max_events=args.max_events,
                    max_reconnects=args.max_reconnects,
                    max_seconds=args.max_seconds,
                ),
            )
        finally:
            backend.close()
        if outcome.phase is BridgeToolLoopPhase.WAITING_APPROVAL:
            raise CliError(
                "tool call requires explicit approve/deny; "
                f"pending={','.join(outcome.pending_call_ids)}"
            )
        if outcome.phase is BridgeToolLoopPhase.WAITING:
            raise CliError(
                "bridge is waiting on client liveness or lease; "
                f"pending={','.join(outcome.pending_call_ids)}"
            )
        _print_json(await bridge.get_run())


async def _decide(args: argparse.Namespace, outcome: str) -> None:
    token = _read_token(from_stdin=args.token_stdin)
    async with _bridge(args, token) as bridge:
        session_id = bridge.state.session_id
        if not session_id:
            raise CliError("connect must be completed before deciding an approval")
        body: dict[str, Any] = {"outcome": outcome}
        if args.reason is not None:
            body["reason"] = args.reason
        response = await bridge._request(
            "POST",
            f"/v1/sessions/{session_id}/approvals/{args.request_id}/decision",
            body=body,
        )
    _print_json(response)


async def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "bootstrap":
        await _bootstrap(args)
        return 0
    if args.command == "connect":
        await _connect(args)
        return 0
    if args.command == "run":
        await _run(args)
        return 0
    if args.command in {"watch", "resume"}:
        await _watch(args)
        return 0
    if args.command == "approve":
        await _decide(args, "APPROVE")
        return 0
    if args.command == "deny":
        await _decide(args, "DENY")
        return 0
    if args.command == "serve":
        raise CliError("serve must run on the synchronous CLI boundary")
    if args.command == "local":
        from prp_runtime.client.local_cli import (
            dispatch_local_approve,
            dispatch_local_deny,
            dispatch_local_run,
        )

        if args.local_command == "run":
            return await dispatch_local_run(args)
        if args.local_command == "approve":
            return await dispatch_local_approve(args)
        if args.local_command == "deny":
            return await dispatch_local_deny(args)
        raise CliError(f"unknown local command: {args.local_command}")
    raise CliError(f"unknown command: {args.command}")


def _serve_sync(args: argparse.Namespace) -> int:
    """Hand serve to the installed runner outside any asyncio.run loop."""
    from prp_runtime.client.serve import MAX_SERVE_PORT, MIN_SERVE_PORT, serve_app

    if args.port < MIN_SERVE_PORT or args.port > MAX_SERVE_PORT:
        raise CliError("serve port must be between 1 and 65535")
    serve_app(host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return a shell status code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            return _serve_sync(args)
        return asyncio.run(_dispatch(args))
    except (BridgeError, CliError, OSError, ValueError) as error:
        print(f"prp: {error}", file=sys.stderr)
        return 1
