"""Event system for context lifecycle hooks.

Provides a decorator-based API for registering callbacks that fire
on context lifecycle events (block added, budget warning, etc.).
Integrates with any observability stack via simple function callbacks.
"""

from __future__ import annotations

import enum
import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field

logger = logging.getLogger("contextkit")


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


# Global handler registry: event type -> list of callbacks
_handlers: Dict[ContextEvent, List[Callable[..., Any]]] = defaultdict(list)
_handlers_lock = threading.Lock()


def on(event: ContextEvent) -> Callable[..., Any]:
    """Register a callback for a context event via decorator.

    Usage::

        @on(ContextEvent.BLOCK_ADDED)
        def log_addition(event_data):
            print(f"Added {event_data.block_name}")
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        with _handlers_lock:
            _handlers[event].append(func)
        return func

    return decorator


def register_handler(event: ContextEvent, handler: Callable[..., Any]) -> None:
    """Register a callback for a context event (non-decorator form).

    Args:
        event: The event type to listen for.
        handler: The callback function.
    """
    with _handlers_lock:
        _handlers[event].append(handler)


def emit(event_data: EventData) -> None:
    """Fire an event to all registered callbacks.

    Handler exceptions are logged and do not prevent other
    handlers from running.

    Args:
        event_data: The event payload to deliver.
    """
    with _handlers_lock:
        handlers = list(_handlers.get(event_data.event, []))
    for handler in handlers:
        try:
            handler(event_data)
        except Exception:
            logger.exception(
                "Event handler %s failed for %s",
                handler.__name__,
                event_data.event.value,
            )


def clear_handlers() -> None:
    """Remove all registered event handlers. Useful in tests."""
    with _handlers_lock:
        _handlers.clear()
