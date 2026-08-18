"""Thin runtime worker facade for one policy-controlled tool call."""

from prp_runtime.domain.enums import (
    AgentMode,
    ExecutionLocation,
    IsolationMode,
)
from prp_runtime.policy.models import CommandClass
from prp_runtime.settings import Settings
from prp_runtime.tools.executor import ToolExecutionOutcome, ToolExecutor
from prp_runtime.tools.models import ToolCall

__all__ = ["ToolWorker"]


class ToolWorker:
    """Execute one tool through the shared executor without owning run state."""

    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor

    async def execute(
        self,
        call: ToolCall,
        mode: AgentMode,
        *,
        workspace_id: str,
        idempotency_key: str,
        resolved_paths: tuple[str, ...] = (),
        command_class: CommandClass | None = None,
        isolation_mode: IsolationMode = IsolationMode.SANDBOXED,
        execution_location: ExecutionLocation = ExecutionLocation.CLOUD,
        user_explicit_host_yolo: bool = False,
        settings: Settings | None = None,
        approved: bool | None = None,
    ) -> ToolExecutionOutcome:
        """Delegate one call; handler code never receives the Controller."""
        return await self._executor.execute(
            call,
            mode,
            workspace_id=workspace_id,
            idempotency_key=idempotency_key,
            resolved_paths=resolved_paths,
            command_class=command_class,
            isolation_mode=isolation_mode,
            execution_location=execution_location,
            user_explicit_host_yolo=user_explicit_host_yolo,
            settings=settings,
            approved=approved,
        )
