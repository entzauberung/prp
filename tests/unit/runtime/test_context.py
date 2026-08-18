"""R0 contract tests for DEV runtime context boundaries."""

import pytest

from prp_runtime.domain.enums import ExecutionLocation, IsolationMode
from prp_runtime.domain.values import new_principal_id, new_workspace_id
from prp_runtime.policy.models import DevExecutionMode, guard_dev_scope
from prp_runtime.runtime.context import DevExecutionContext


def make_scope():
    return guard_dev_scope(
        principal_id=new_principal_id(),
        workspace_id=new_workspace_id(),
        mode=DevExecutionMode.HOST,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.CLOUD,
    )


def test_temporary_host_root_is_internal_and_not_public_evidence() -> None:
    context = DevExecutionContext.from_temporary_root(make_scope(), "/tmp/prp-dev-root")
    assert context.internal_temporary_root.as_posix() == "/tmp/prp-dev-root"
    assert context.resolve_internal_path("src/main.py").as_posix() == (
        "/tmp/prp-dev-root/src/main.py"
    )
    assert "/tmp/prp-dev-root" not in context.model_dump_json()
    assert "temporary_root" not in context.model_dump()


def test_dev_context_rejects_unsafe_or_missing_internal_paths() -> None:
    context = DevExecutionContext.from_temporary_root(make_scope(), "/tmp/prp-dev-root")
    for path in ("../secret", "/etc/passwd", "C:\\secret", "src//main.py"):
        with pytest.raises(ValueError):
            context.resolve_internal_path(path)
    with pytest.raises(RuntimeError, match="temporary root"):
        DevExecutionContext(scope=make_scope()).internal_temporary_root


def test_dev_context_requires_an_absolute_temporary_root() -> None:
    with pytest.raises(ValueError, match="absolute"):
        DevExecutionContext.from_temporary_root(make_scope(), "relative-root")
