"""Event system for context lifecycle hooks.

Provides a decorator-based API for registering callbacks that fire
on context lifecycle events (block added, budget warning, etc.).
Integrates with any observability stack via simple function callbacks.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Dict, List

from contextkit.observe.event_models import (
    BlockEventData,
    BudgetEventData,
    ContextEvent,
    EventData,
    PipelineEventData,
)

logger = logging.getLogger("contextkit")

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
