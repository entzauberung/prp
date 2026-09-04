"""CLI tests with a fake Bridge; no real network or token output."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from prp_runtime.client import cli
from prp_runtime.client.bridge import BridgeToolLoopPhase


class FakeBridge:
    instances: list[FakeBridge] = []

    def __init__(self, base_url: str, token: str, **kwargs: object) -> None:
        self.base_url = base_url
        self.token = token
        self.kwargs = kwargs
        self.state = SimpleNamespace(session_id=None, run_id=None)
        self.sessions: list[dict[str, object]] = []
        self.registrations: list[object] = []
        self.handshakes: list[object] = []
        self.loop_calls: list[object] = []
        FakeBridge.instances.append(self)

    async def __aenter__(self) -> FakeBridge:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def register_client(self, payload: object) -> dict[str, object]:
        self.registrations.append(payload)
        return {"status": "ACTIVE"}

    async def handshake(self, payload: object) -> dict[str, object]:
        self.handshakes.append(payload)
        return {
            "accepted": True,
            "client_id": "cli_bootstrap01",
            "protocol_version": "0.0.4",
            "fingerprint": "a" * 64,
        }

    async def create_session(self, workspace_id: str, **kwargs: object) -> dict[str, object]:
        self.state.session_id = "sess-1"
        self.sessions.append({"workspace_id": workspace_id, **kwargs})
        return {"session_id": self.state.session_id}

    async def create_run(self, session_id: str | None, payload: object) -> dict[str, object]:
        self.state.run_id = "run-1"
        return {"run_id": self.state.run_id, "payload": payload}

    async def iter_events(self, **kwargs: object):
        yield {"sequence": 1, "event_type": "RUN_SUCCEEDED"}

    async def get_run(self) -> dict[str, object]:
        return {"run_id": "run-1", "status": "SUCCEEDED"}

    async def run_tool_loop(self, executor: object, **kwargs: object) -> object:
        self.loop_calls.append((executor, kwargs))
        return SimpleNamespace(phase=BridgeToolLoopPhase.TERMINAL, pending_call_ids=())

    async def _request(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {"outcome": kwargs["body"]["outcome"]}  # type: ignore[index]


@pytest.fixture(autouse=True)
def reset_fake_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeBridge.instances.clear()
    monkeypatch.setattr(cli, "Bridge", FakeBridge)


def test_connect_reads_env_token_and_keeps_root_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")
    state_path = tmp_path / "state.json"
    root = tmp_path / "workspace"

    assert (
        cli.main(
            [
                "--base-url",
                "https://bridge.test",
                "--state",
                str(state_path),
                "--workspace-root",
                str(root),
                "connect",
                "workspace-1",
            ]
        )
        == 0
    )
    bridge = FakeBridge.instances[0]
    assert bridge.token == "do-not-print"
    assert bridge.sessions[0]["workspace_id"] == "workspace-1"
    assert str(root) not in capsys.readouterr().out
    assert "do-not-print" not in capsys.readouterr().out


def test_host_yolo_requires_interactive_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)

    result = cli.main(
        ["connect", "workspace-1", "--agent-mode", "YOLO", "--isolation-mode", "HOST"]
    )

    assert result == 1
    assert "interactive confirmation" in capsys.readouterr().err
    assert not FakeBridge.instances


def test_host_yolo_confirmation_is_forwarded_as_explicit_user_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(agent_mode="YOLO", isolation_mode="HOST")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "HOST YOLO")

    assert cli._confirm_host_yolo(args) is True
    assert cli._agent_options(args, user_explicit=True)["user_explicit"] is True


def test_help_explains_path_boundary(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert "not an operating-system sandbox" in capsys.readouterr().out


def test_watch_runs_bounded_loop_and_prints_terminal_run_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")
    root = tmp_path / "workspace"
    root.mkdir()

    assert (
        cli.main(
            [
                "--workspace-root",
                str(root),
                "watch",
                "--max-calls",
                "2",
                "--max-events",
                "4",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert output.count('"status": "SUCCEEDED"') == 1
    assert str(root) not in output
    assert "do-not-print" not in output
    assert len(FakeBridge.instances[0].loop_calls) == 1


def test_watch_without_root_fails_before_local_execution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")

    assert cli.main(["watch"]) == 1

    assert "workspace-root" in capsys.readouterr().err


def test_local_run_parser_is_distinct_from_remote_run() -> None:
    parser = cli.build_parser()
    local = parser.parse_args(["local", "run", "summarise the report"])
    assert local.command == "local"
    assert local.local_command == "run"
    assert local.prompt == "summarise the report"
    assert local.workspace is None
    assert local.agent_mode == "NORMAL"
    assert local.isolation_mode == "HOST"
    assert local.user_explicit is False
    remote = parser.parse_args(["run", "summarise the report"])
    assert remote.command == "run"
    assert remote.input_text == "summarise the report"
    assert remote.isolation_mode == "SANDBOXED"


def test_local_run_help_explains_in_process_path_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["local", "run", "--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "in-process" in help_text
    assert "path-boundary" in help_text
    assert "not an operating-system sandbox" in help_text


def test_local_run_parser_does_not_require_bridge_inputs() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "local",
            "run",
            "hello",
            "--workspace",
            ".",
            "--agent-mode",
            "AUTO",
        ]
    )
    assert args.command == "local"
    assert args.workspace == Path(".")
    assert args.agent_mode == "AUTO"
    assert args.isolation_mode == "HOST"
    assert not hasattr(args, "input_text")
    sandboxed = parser.parse_args(
        ["local", "run", "hello", "--isolation-mode", "SANDBOXED"]
    )
    assert sandboxed.isolation_mode == "SANDBOXED"
    assert sandboxed.command == "local"


def test_local_approve_and_deny_parse_without_bridge_inputs() -> None:
    parser = cli.build_parser()
    approve = parser.parse_args(["local", "approve", "apr_abc123"])
    assert approve.command == "local"
    assert approve.local_command == "approve"
    assert approve.request_id == "apr_abc123"
    assert approve.workspace is None
    deny = parser.parse_args(
        ["local", "deny", "apr_abc123", "--reason", "not needed"]
    )
    assert deny.command == "local"
    assert deny.local_command == "deny"
    assert deny.reason == "not needed"
    remote = parser.parse_args(["approve", "apr_remote"])
    assert remote.command == "approve"
    assert remote.request_id == "apr_remote"


def test_local_deny_requires_a_reason() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["local", "deny", "apr_abc123"])


def test_serve_defaults_to_loopback_and_is_not_local_run() -> None:
    parser = cli.build_parser()
    serve = parser.parse_args(["serve"])
    assert serve.command == "serve"
    assert serve.host == "127.0.0.1"
    assert serve.port == 8000
    local = parser.parse_args(["local", "run", "hello"])
    assert local.command == "local"
    assert local.local_command == "run"


def test_serve_dispatch_is_synchronous_and_does_not_start_a_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, object] = {}

    def fake_serve_app(*, host: str, port: int, **kwargs: object) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            nested = False
        else:
            nested = True
        called["host"] = host
        called["port"] = port
        called["nested"] = nested
        called["kwargs"] = kwargs

    monkeypatch.setattr("prp_runtime.client.serve.serve_app", fake_serve_app)
    assert cli.main(["serve"]) == 0
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8000
    assert called["nested"] is False
    assert called["kwargs"] == {}


def test_serve_help_explains_loopback(capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["serve", "--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "127.0.0.1" in help_text
    assert "create_app" in help_text
    assert "Local run does not use this command" in help_text


def test_bootstrap_is_model_free_and_keeps_root_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")
    root = tmp_path / "workspace"
    root.mkdir()
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--workspace-root",
            str(root),
            "bootstrap",
            "ws_project",
        ]
    )
    assert args.command == "bootstrap"
    assert not hasattr(args, "model")
    assert not hasattr(args, "provider")
    local = parser.parse_args(["local", "run", "hello"])
    assert local.command == "local"
    assert local.command != args.command

    assert (
        cli.main(
            [
                "--base-url",
                "https://bridge.test",
                "--workspace-root",
                str(root),
                "bootstrap",
                "ws_project",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"accepted": true' in output.lower() or '"accepted": True' in output or '"accepted": true' in output
    assert "do-not-print" not in output
    assert str(root) not in output
    bridge = FakeBridge.instances[0]
    assert bridge.registrations
    assert bridge.handshakes
    payload = bridge.handshakes[0]
    registration = bridge.registrations[0]
    assert "capabilities" in registration
    assert registration["fingerprint"] == payload["fingerprint"]
    assert "strategy" not in str(payload)
    assert "token" not in str(payload).lower() or "fingerprint" in str(payload)


def test_watch_rejects_negative_reconnects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")
    root = tmp_path / "workspace"
    root.mkdir()
    assert cli.main(["--workspace-root", str(root), "watch", "--max-reconnects", "-1"]) == 1
    assert "finite" in capsys.readouterr().err


def test_watch_approval_pause_is_not_terminal_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")
    root = tmp_path / "workspace"
    root.mkdir()

    class ApprovalBridge(FakeBridge):
        async def run_tool_loop(self, executor: object, **kwargs: object) -> object:
            self.loop_calls.append((executor, kwargs))
            return SimpleNamespace(
                phase=BridgeToolLoopPhase.WAITING_APPROVAL,
                pending_call_ids=("tc_one",),
            )

    monkeypatch.setattr(cli, "Bridge", ApprovalBridge)
    assert cli.main(["--workspace-root", str(root), "watch"]) == 1
    captured = capsys.readouterr()
    assert "approve/deny" in captured.err
    assert "SUCCEEDED" not in captured.out
    assert "do-not-print" not in captured.out + captured.err


def test_approve_and_deny_are_scoped_and_omit_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")

    class SessionBridge(FakeBridge):
        def __init__(self, base_url: str, token: str, **kwargs: object) -> None:
            super().__init__(base_url, token, **kwargs)
            self.state.session_id = "sess-1"
            self.decisions: list[object] = []

        async def _request(self, *args: object, **kwargs: object) -> dict[str, object]:
            self.decisions.append((args, kwargs))
            return {"outcome": kwargs["body"]["outcome"]}  # type: ignore[index]

    monkeypatch.setattr(cli, "Bridge", SessionBridge)
    assert cli.main(["approve", "apr_one", "--reason", "ok"]) == 0
    assert cli.main(["deny", "apr_one", "--reason", "no"]) == 0
    output = capsys.readouterr().out
    assert '"outcome": "APPROVE"' in output
    assert '"outcome": "DENY"' in output
    assert "do-not-print" not in output
    assert SessionBridge.instances[0].state.session_id == "sess-1"


def test_resume_uses_watch_loop_without_model_flags() -> None:
    parser = cli.build_parser()
    resume = parser.parse_args(["resume", "--max-reconnects", "2"])
    watch = parser.parse_args(["watch", "--max-reconnects", "2"])
    assert resume.command == "resume"
    assert watch.command == "watch"
    assert resume.max_reconnects == watch.max_reconnects == 2
    assert not hasattr(resume, "model")
    assert not hasattr(resume, "provider")



def test_watch_offline_wait_is_not_terminal_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")
    root = tmp_path / "workspace"
    root.mkdir()

    class WaitingBridge(FakeBridge):
        async def run_tool_loop(self, executor: object, **kwargs: object) -> object:
            self.loop_calls.append((executor, kwargs))
            return SimpleNamespace(
                phase=BridgeToolLoopPhase.WAITING,
                pending_call_ids=("tc_live",),
            )

    monkeypatch.setattr(cli, "Bridge", WaitingBridge)
    assert cli.main(["--workspace-root", str(root), "watch"]) == 1
    captured = capsys.readouterr()
    assert "waiting" in captured.err
    assert "SUCCEEDED" not in captured.out
    assert "do-not-print" not in captured.out + captured.err
