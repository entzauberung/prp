"""Targeted tests for the immutable tool registry."""

import json
from collections.abc import Mapping

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from prp_runtime.domain.enums import ToolEffect
from prp_runtime.domain.models import ProviderToolDescriptor
from prp_runtime.tools import ToolDefinition, ToolRegistry


class ReadArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str


async def fake_read_handler(arguments: BaseModel) -> Mapping[str, object]:
    return {"path": arguments.model_dump()["path"]}


def read_definition(**overrides: object) -> ToolDefinition:
    values: dict[str, object] = {
        "name": "read_file",
        "effect": ToolEffect.READ,
        "argument_model": ReadArguments,
        "handler": fake_read_handler,
        "max_output_bytes": 4096,
    }
    values.update(overrides)
    if "result_limit" in overrides:
        values.pop("max_output_bytes")
    return ToolDefinition(**values)  # type: ignore[arg-type]


def test_definition_owns_effect_arguments_and_result_limit() -> None:
    definition = read_definition(result_limit=2048)
    assert definition.effect is ToolEffect.READ
    assert definition.argument_model is ReadArguments
    assert definition.arguments_model is ReadArguments
    assert definition.result_limit == 2048
    assert definition.validate_arguments({"path": "src/main.py"}).path == "src/main.py"
    with pytest.raises(ValidationError):
        definition.validate_arguments({"path": "src/main.py", "extra": True})


def test_handler_cannot_supply_or_change_definition_effect() -> None:
    class MisleadingHandler:
        effect = ToolEffect.COMMAND

        async def __call__(self, arguments: BaseModel) -> Mapping[str, object]:
            return {"ok": True}

    definition = read_definition(handler=MisleadingHandler())
    assert definition.effect is ToolEffect.READ


def test_registry_is_unique_immutable_and_deterministic() -> None:
    first = read_definition()
    second = read_definition(name="search_files")
    registry = ToolRegistry.build((first, second))

    assert registry.names == ("read_file", "search_files")
    assert registry.definitions == (first, second)
    assert registry["read_file"] is first
    assert "search_files" in registry
    assert tuple(registry) == (first, second)

    with pytest.raises(KeyError, match="unknown tool: write_file"):
        registry.get("write_file")
    with pytest.raises(AttributeError):
        registry.register(first)  # type: ignore[attr-defined]


def test_registry_provider_catalog_is_sorted_and_excludes_execution_metadata() -> None:
    first = read_definition(
        name="search_files",
        description="Search the authorized workspace.",
    )
    second = read_definition(
        name="read_file",
        description="Read one authorized file.",
    )
    catalog = ToolRegistry((first, second)).provider_catalog

    assert [descriptor.name for descriptor in catalog] == ["read_file", "search_files"]
    assert catalog[0].description == "Read one authorized file."
    assert catalog[0].input_schema == ReadArguments.model_json_schema(mode="validation")
    payload = json.dumps(
        [descriptor.model_dump(mode="json") for descriptor in catalog],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert "handler" not in payload
    assert "effect" not in payload
    assert "argument_model" not in payload


def test_provider_catalog_does_not_depend_on_handler_identity() -> None:
    async def another_handler(arguments: BaseModel) -> Mapping[str, object]:
        return {"different": True}

    first = read_definition(handler=fake_read_handler)
    second = read_definition(handler=another_handler)

    assert ToolRegistry((first,)).provider_catalog == ToolRegistry((second,)).catalog


def test_tool_definition_description_is_closed_and_bounded() -> None:
    with pytest.raises(ValidationError):
        read_definition(description="x" * 2_049)
    with pytest.raises(ValidationError):
        read_definition(unexpected="value")

    descriptor = read_definition().to_provider_descriptor()
    assert isinstance(descriptor, ProviderToolDescriptor)


def test_duplicate_and_invalid_definitions_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate tool definition: read_file"):
        ToolRegistry((read_definition(), read_definition()))
    with pytest.raises(ValidationError):
        read_definition(handler=object())
    with pytest.raises(ValidationError):
        read_definition(argument_model=dict)
    with pytest.raises(ValidationError):
        read_definition(max_output_bytes=0)
    with pytest.raises(ValidationError):
        read_definition(name="ReadFile")


def test_definition_is_frozen_after_build() -> None:
    definition = read_definition()
    with pytest.raises(ValidationError):
        definition.effect = ToolEffect.WRITE


def test_registry_rejects_unknown_tool_and_keeps_handler_private() -> None:
    registry = ToolRegistry((read_definition(),))
    with pytest.raises(KeyError, match="unknown tool: run_shell"):
        registry.get("run_shell")
    catalog = json.dumps(
        [item.model_dump(mode="json") for item in registry.provider_catalog],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert "handler" not in catalog
    assert "max_output_bytes" not in catalog

