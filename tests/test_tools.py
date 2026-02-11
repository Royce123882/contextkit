"""Tests for tool registry and tool outputs (Phase 3)."""

from __future__ import annotations

import json

import pytest

from contextkit.tools.tool_models import ToolDefinition, ToolOutput
from contextkit.tools.tool_registry import ToolRegistry


class TestToolDefinition:
    """Tests for the ToolDefinition model."""

    def test_create_with_defaults(self) -> None:
        tool = ToolDefinition(name="search", description="Search the web")
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert tool.parameters == {}
        assert tool.tags == []

    def test_create_with_all_fields(self) -> None:
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {}},
            tags=["web", "retrieval"],
        )
        assert tool.parameters == {"type": "object", "properties": {}}
        assert tool.tags == ["web", "retrieval"]


class TestToolOutput:
    """Tests for the ToolOutput model."""

    def test_create_basic(self) -> None:
        output = ToolOutput(tool_name="search", result="Found 5 results")
        assert output.tool_name == "search"
        assert output.result == "Found 5 results"
        assert output.call_id == ""
        assert output.latency_ms == 0.0

    def test_to_block(self) -> None:
        output = ToolOutput(
            tool_name="search",
            call_id="call_123",
            result="Results here",
            latency_ms=150.0,
        )
        block = output.to_block()
        assert block.type.value == "tool_outputs"
        assert block.content == "Results here"
        assert block.origin is not None
        assert block.origin.source == "tool_output"
        assert block.origin.details["tool_name"] == "search"
        assert block.origin.details["call_id"] == "call_123"
        assert block.origin.details["latency_ms"] == 150.0

    def test_to_block_custom_priority(self) -> None:
        output = ToolOutput(tool_name="search", result="data")
        block = output.to_block(priority=90)
        assert block.priority == 90

    def test_called_at_auto_set(self) -> None:
        output = ToolOutput(tool_name="search", result="data")
        assert output.called_at is not None


class TestToolRegistry:
    """Tests for the ToolRegistry."""

    def test_register_tool(self) -> None:
        registry = ToolRegistry()
        tool = registry.register(
            name="search",
            description="Search the web",
        )
        assert tool.name == "search"
        assert registry.tool_count == 1

    def test_register_with_parameters(self) -> None:
        registry = ToolRegistry()
        params = {"type": "object", "properties": {"q": {"type": "string"}}}
        tool = registry.register(
            name="search",
            description="Search",
            parameters=params,
        )
        assert tool.parameters == params

    def test_register_with_tags(self) -> None:
        registry = ToolRegistry()
        tool = registry.register(
            name="search",
            description="Search",
            tags=["web"],
        )
        assert tool.tags == ["web"]

    def test_get_tool(self) -> None:
        registry = ToolRegistry()
        registry.register("search", "Search the web")
        tool = registry.get_tool("search")
        assert tool.name == "search"

    def test_get_tool_not_found(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="No tool registered"):
            registry.get_tool("nonexistent")

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        registry.register("b_tool", "Tool B")
        registry.register("a_tool", "Tool A")
        assert registry.list_tools() == ["a_tool", "b_tool"]

    def test_select_by_task(self) -> None:
        registry = ToolRegistry()
        registry.register("web_search", "Search the web for information")
        registry.register("calculator", "Perform mathematical calculations")
        registry.register("file_reader", "Read files from disk")

        blocks = registry.select("I need to search the web")
        assert len(blocks) >= 1
        # web_search should rank highest
        content = json.loads(blocks[0].content)
        assert content["name"] == "web_search"

    def test_select_with_tags(self) -> None:
        registry = ToolRegistry()
        registry.register("web_search", "Search web", tags=["web"])
        registry.register("calc", "Calculate things", tags=["math"])
        blocks = registry.select("search", tags=["web"])
        assert len(blocks) == 1

    def test_select_max_tools(self) -> None:
        registry = ToolRegistry()
        for i in range(10):
            registry.register(f"tool_{i}", f"Tool number {i}")
        blocks = registry.select("tool", max_tools=3)
        assert len(blocks) == 3

    def test_select_returns_context_blocks(self) -> None:
        registry = ToolRegistry()
        registry.register("search", "Search the web")
        blocks = registry.select("search")
        assert blocks[0].type.value == "tool_definitions"
        assert blocks[0].origin is not None
        assert blocks[0].origin.source == "tool"

    def test_to_block_all_tools(self) -> None:
        registry = ToolRegistry()
        registry.register("a", "Tool A")
        registry.register("b", "Tool B")
        block = registry.to_block()
        assert block.type.value == "tool_definitions"
        assert block.origin is not None
        assert block.origin.details["tool_count"] == 2
        # Content should be a list of tool dicts
        assert isinstance(block.content, list)
        assert len(block.content) == 2

    def test_to_block_custom_priority(self) -> None:
        registry = ToolRegistry()
        registry.register("a", "Tool A")
        block = registry.to_block(priority=80)
        assert block.priority == 80

    def test_tool_count(self) -> None:
        registry = ToolRegistry()
        assert registry.tool_count == 0
        registry.register("a", "A")
        registry.register("b", "B")
        assert registry.tool_count == 2

    def test_capture_output(self) -> None:
        registry = ToolRegistry()
        registry.register("search", "Search")
        output = registry.capture_output(
            tool_name="search",
            result="5 results",
            call_id="call_1",
            latency_ms=100.0,
        )
        assert isinstance(output, ToolOutput)
        assert output.tool_name == "search"
        assert output.result == "5 results"
        assert output.call_id == "call_1"
        assert output.latency_ms == 100.0

    def test_register_overwrites(self) -> None:
        registry = ToolRegistry()
        registry.register("search", "Old description")
        registry.register("search", "New description")
        assert registry.tool_count == 1
        tool = registry.get_tool("search")
        assert tool.description == "New description"
