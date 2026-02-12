"""Tool registry for managing tool definitions as context blocks."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from contextkit.constants import PRIORITY_TOOL_DEFINITION
from contextkit.core import BlockType, ContextBlock
from contextkit.observe.provenance import Origin
from contextkit.tools.tool_models import ToolDefinition, ToolOutput
from contextkit.utils.text_similarity import word_overlap_score

logger = logging.getLogger("contextkit")


class ToolRegistry:
    """Manages tool definitions and provides dynamic selection.

    Tools are registered with schemas and can be selected based
    on task descriptions. Selected tools are converted to
    ContextBlocks with Origin auto-populated.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any] | None = None,
        tags: List[str] | None = None,
    ) -> ToolDefinition:
        """Register a tool definition.

        Args:
            name: Unique tool name.
            description: Human-readable description.
            parameters: JSON Schema for parameters.
            tags: Optional categorization tags.

        Returns:
            The created ToolDefinition.
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {},
            tags=tags or [],
        )
        self._tools[name] = tool
        logger.debug("Registered tool '%s'", name)
        return tool

    def get_tool(self, name: str) -> ToolDefinition:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            The matching ToolDefinition.

        Raises:
            KeyError: If tool not found.
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return sorted(self._tools.keys())

    def select(
        self,
        task_description: str,
        tags: List[str] | None = None,
        max_tools: int | None = None,
    ) -> List[ContextBlock]:
        """Select relevant tools based on a task description.

        Uses keyword matching and tag filtering to find tools
        relevant to the given task. Returns ContextBlocks ready
        for injection into a context window.

        Args:
            task_description: Description of the current task.
            tags: Optional tag filter (tools must have all tags).
            max_tools: Maximum number of tools to return.

        Returns:
            List of ContextBlocks containing tool definitions.
        """
        candidates = list(self._tools.values())

        # Filter by tags
        if tags:
            candidates = [t for t in candidates if all(tag in t.tags for tag in tags)]

        # Score by keyword overlap with task description
        scored: List[Tuple[float, ToolDefinition]] = []
        for tool in candidates:
            combined_text = f"{tool.name} {tool.description}"
            score = word_overlap_score(task_description, combined_text)
            scored.append((score, tool))

        scored.sort(key=lambda x: x[0], reverse=True)

        if max_tools is not None:
            scored = scored[:max_tools]

        logger.info(
            "Selecting tools for task (candidates: %d, max: %s)",
            len(scored),
            max_tools or "all",
        )
        blocks: List[ContextBlock] = []
        for score, tool in scored:
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            origin = Origin(
                source="tool",
                details={
                    "tool_name": tool.name,
                    "selection_reason": (f"keyword_score={score:.2f}"),
                },
            )
            block = ContextBlock(
                type=BlockType.TOOL_DEFINITIONS,
                content=json.dumps(tool_schema),
                priority=PRIORITY_TOOL_DEFINITION,
                name=f"tool_{tool.name}",
                origin=origin,
            )
            blocks.append(block)

        return blocks

    def to_block(self, priority: int = PRIORITY_TOOL_DEFINITION) -> ContextBlock:
        """Convert all tools to a single ContextBlock.

        Creates a block containing all tool definitions as a list.

        Args:
            priority: Block priority.

        Returns:
            A ContextBlock of type TOOL_DEFINITIONS.
        """
        tools_list = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]
        origin = Origin(
            source="tool",
            details={
                "tool_count": len(tools_list),
            },
        )
        return ContextBlock(
            type=BlockType.TOOL_DEFINITIONS,
            content=tools_list,
            priority=priority,
            name="tool_definitions",
            origin=origin,
        )

    async def aselect(
        self,
        task_description: str,
        tags: List[str] | None = None,
        max_tools: int | None = None,
    ) -> List[ContextBlock]:
        """Async version of :meth:`select`."""
        return self.select(task_description, tags=tags, max_tools=max_tools)

    async def ato_block(self, priority: int = PRIORITY_TOOL_DEFINITION) -> ContextBlock:
        """Async version of :meth:`to_block`."""
        return self.to_block(priority=priority)

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def capture_output(
        self,
        tool_name: str,
        result: str,
        call_id: str = "",
        latency_ms: float = 0.0,
    ) -> ToolOutput:
        """Capture the output of a tool call.

        Args:
            tool_name: Name of the tool that was called.
            result: The tool's output content.
            call_id: Unique identifier for this call.
            latency_ms: How long the call took.

        Returns:
            A ToolOutput instance.
        """
        return ToolOutput(
            tool_name=tool_name,
            call_id=call_id,
            result=result,
            latency_ms=latency_ms,
        )
