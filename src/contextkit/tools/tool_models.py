"""Tool data models: ToolDefinition and ToolOutput."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from contextkit.constants import PRIORITY_TOOL_OUTPUT
from contextkit.core import BlockType, ContextBlock
from contextkit.observe.provenance import Origin


class ToolDefinition(BaseModel):
    """A registered tool definition.

    Attributes:
        name: Tool name (unique identifier).
        description: Human-readable description.
        parameters: JSON Schema for the tool's parameters.
        tags: Categorization tags for selection.
    """

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class ToolOutput(BaseModel):
    """Captured output from a tool call.

    Attributes:
        tool_name: Name of the tool that was called.
        call_id: Unique identifier for this call.
        result: The tool's output content.
        latency_ms: How long the tool call took in milliseconds.
        called_at: When the tool was called.
    """

    tool_name: str
    call_id: str = ""
    result: str = ""
    latency_ms: float = 0.0
    called_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_block(self, priority: int = PRIORITY_TOOL_OUTPUT) -> ContextBlock:
        """Convert this output to a ContextBlock.

        Auto-populates Origin with tool name, call ID, and latency.

        Args:
            priority: Block priority.

        Returns:
            A ContextBlock of type TOOL_OUTPUTS.
        """
        origin = Origin(
            source="tool_output",
            details={
                "tool_name": self.tool_name,
                "call_id": self.call_id,
                "latency_ms": self.latency_ms,
            },
        )
        return ContextBlock(
            type=BlockType.TOOL_OUTPUTS,
            content=self.result,
            priority=priority,
            name=f"output_{self.tool_name}",
            origin=origin,
        )
