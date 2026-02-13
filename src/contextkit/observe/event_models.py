"""Data models for the context event system."""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field


class ContextEvent(enum.Enum):
    """Events emitted during context lifecycle."""

    BLOCK_ADDED = "block_added"
    BLOCK_REMOVED = "block_removed"
    BLOCK_MUTATED = "block_mutated"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    ASSEMBLY_COMPLETE = "assembly_complete"
    PIPELINE_STEP = "pipeline_step"
    PIPELINE_COMPLETE = "pipeline_complete"
    WINDOW_RENDERED = "window_rendered"
    CONTEXT_INSUFFICIENT = "context_insufficient"  # emitted by SufficiencyChecker


class EventData(BaseModel):
    """Base class for event payloads.

    Attributes:
        event: The context lifecycle event type.
        timestamp: When the event was emitted.
        details: Additional event-specific key-value data.
    """

    event: ContextEvent = Field(description="The context lifecycle event type.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event was emitted.",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event-specific key-value data.",
    )

    model_config = {"arbitrary_types_allowed": True}


class BlockEventData(EventData):
    """Event data for block-related events.

    Attributes:
        block_name: Display name of the affected block.
        block_type: The BlockType value of the affected block.
        token_count: Token count of the affected block.
    """

    block_name: str = Field(default="", description="Display name of the affected block.")
    block_type: str = Field(default="", description="The BlockType value of the affected block.")
    token_count: int = Field(default=0, description="Token count of the affected block.")


class BudgetEventData(EventData):
    """Event data for budget-related events.

    Attributes:
        percent: Current budget utilization percentage.
        threshold: The threshold that was crossed.
        tokens_used: Current total tokens used.
        tokens_max: Maximum token budget.
    """

    percent: float = Field(default=0.0, description="Current budget utilization percentage.")
    threshold: float = Field(default=0.0, description="The threshold that was crossed.")
    tokens_used: int = Field(default=0, description="Current total tokens used.")
    tokens_max: int = Field(default=0, description="Maximum token budget.")


class PipelineEventData(EventData):
    """Event data for pipeline-related events.

    Attributes:
        step_name: Name of the pipeline step.
        tokens_before: Total tokens before the step.
        tokens_after: Total tokens after the step.
    """

    step_name: str = Field(default="", description="Name of the pipeline step.")
    tokens_before: int = Field(default=0, description="Total tokens before the step.")
    tokens_after: int = Field(default=0, description="Total tokens after the step.")
