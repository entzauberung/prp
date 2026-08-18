"""Immutable registry contracts for policy-controlled tool handlers."""

from collections.abc import Collection, Iterator, Mapping
from types import MappingProxyType
from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StringConstraints, field_validator

from prp_runtime.domain.enums import ToolEffect
from prp_runtime.domain.models import ProviderToolDescriptor
from prp_runtime.tools.models import MAX_TOOL_OUTPUT_BYTES

__all__ = [
    "ToolDefinition",
    "ToolHandler",
    "ToolRegistry",
]

ToolName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_.-]{0,63}$",
    ),
]


@runtime_checkable
class ToolHandler(Protocol):
    """The callable implementation injected into a tool definition.

    A handler receives already validated arguments. It has no effect field: the
    registry definition is the only trusted source for the operation's risk
    classification.
    """

    async def __call__(self, arguments: BaseModel) -> Any:
        """Execute validated arguments and return an executor-owned value."""


class ToolDefinition(BaseModel):
    """Closed metadata for one registered tool.

    ``handler`` is intentionally opaque to the model contract. It cannot
    change ``effect`` or the result ceiling, which remain immutable definition
    facts owned by the registry.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    name: ToolName
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2_048)] = ""
    effect: ToolEffect
    argument_model: type[BaseModel] = Field(
        validation_alias=AliasChoices("argument_model", "arguments_model", "args_model")
    )
    handler: ToolHandler
    max_output_bytes: int = Field(
        default=MAX_TOOL_OUTPUT_BYTES,
        gt=0,
        le=MAX_TOOL_OUTPUT_BYTES,
        validation_alias=AliasChoices("max_output_bytes", "result_limit"),
    )

    @field_validator("handler")
    @classmethod
    def _handler_must_be_callable(cls, value: ToolHandler) -> ToolHandler:
        if not callable(value):
            raise ValueError("tool handler must be callable")
        return value

    @field_validator("argument_model")
    @classmethod
    def _argument_model_must_be_pydantic(
        cls, value: type[BaseModel]
    ) -> type[BaseModel]:
        if not isinstance(value, type) or not issubclass(value, BaseModel):
            raise ValueError("argument_model must be a Pydantic BaseModel subclass")
        return value

    @property
    def arguments_model(self) -> type[BaseModel]:
        """Compatibility spelling for callers that use the plural term."""
        return self.argument_model

    @property
    def result_limit(self) -> int:
        """The maximum output size declared by this tool."""
        return self.max_output_bytes

    def validate_arguments(self, arguments: Mapping[str, object]) -> BaseModel:
        """Validate raw JSON-like arguments using the registered model."""
        return self.argument_model.model_validate(arguments)

    def to_provider_descriptor(self) -> ProviderToolDescriptor:
        """Project only public schema metadata for an outbound Provider."""
        return ProviderToolDescriptor(
            name=self.name,
            description=self.description,
            input_schema=self.argument_model.model_json_schema(mode="validation"),
        )


class ToolRegistry:
    """A name-unique, immutable set of tool definitions."""

    __slots__ = ("_definitions",)
    _definitions: Mapping[str, ToolDefinition]

    def __init__(self, definitions: Collection[ToolDefinition] = ()) -> None:
        by_name: dict[str, ToolDefinition] = {}
        for definition in definitions:
            if definition.name in by_name:
                raise ValueError(f"duplicate tool definition: {definition.name}")
            by_name[definition.name] = definition
        object.__setattr__(self, "_definitions", MappingProxyType(by_name))

    @classmethod
    def build(cls, definitions: Collection[ToolDefinition] = ()) -> "ToolRegistry":
        """Build and freeze a registry in one explicit operation."""
        return cls(definitions)

    @property
    def names(self) -> tuple[str, ...]:
        """Registered names in deterministic insertion order."""
        return tuple(self._definitions)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Registered definitions in deterministic insertion order."""
        return tuple(self._definitions.values())

    @property
    def provider_catalog(self) -> tuple[ProviderToolDescriptor, ...]:
        """Return a stable, non-executable catalog for Provider requests."""
        return tuple(
            definition.to_provider_descriptor()
            for definition in sorted(self._definitions.values(), key=lambda item: item.name)
        )

    @property
    def catalog(self) -> tuple[ProviderToolDescriptor, ...]:
        """Compatibility alias for the public Provider catalog."""
        return self.provider_catalog

    def get(self, name: str) -> ToolDefinition:
        """Return a definition or raise a stable unknown-tool error."""
        try:
            return self._definitions[name]
        except KeyError as error:
            raise KeyError(f"unknown tool: {name}") from error

    def __contains__(self, name: object) -> bool:
        return name in self._definitions

    def __getitem__(self, name: str) -> ToolDefinition:
        return self.get(name)

    def __iter__(self) -> Iterator[ToolDefinition]:
        return iter(self._definitions.values())
