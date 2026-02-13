"""OpenTelemetry metrics export for contextkit.

Provides an OTelExporter that registers event handlers to
automatically emit metrics whenever blocks are added, windows
are rendered, or pipelines complete.

Requires the ``opentelemetry-api`` optional dependency. If not
installed, the exporter raises a clear ImportError on construction.

Usage::

    from contextkit.observe.telemetry import OTelExporter
    exporter = OTelExporter()
    exporter.attach()
    # Metrics are now emitted automatically on context events
"""

from __future__ import annotations

import logging
from typing import Any

from contextkit.observe.event_models import ContextEvent, EventData
from contextkit.observe.events import register_handler

logger = logging.getLogger("contextkit")

try:
    from opentelemetry import metrics as otel_metrics

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


class OTelExporter:
    """Export context metrics to OpenTelemetry-compatible backends.

    Creates counters and histograms for tracking token usage,
    block operations, and pipeline performance. Attaches to
    contextkit's event system for automatic emission.

    Args:
        meter_name: Name for the OpenTelemetry meter.
            Defaults to ``"contextkit"``.

    Raises:
        ImportError: If ``opentelemetry-api`` is not installed.
    """

    def __init__(self, meter_name: str = "contextkit") -> None:
        if not _HAS_OTEL:
            raise ImportError(
                "OpenTelemetry support requires the 'opentelemetry-api' package. "
                "Install it with: pip install opentelemetry-api"
            )

        meter = otel_metrics.get_meter(meter_name)
        self._token_counter = meter.create_counter(
            name="contextkit.tokens.total",
            description="Total tokens added to context windows",
            unit="tokens",
        )
        self._block_counter = meter.create_counter(
            name="contextkit.blocks.added",
            description="Number of blocks added to context windows",
            unit="blocks",
        )
        self._pipeline_tokens_saved = meter.create_counter(
            name="contextkit.pipeline.tokens_saved",
            description="Tokens saved by pipeline optimization",
            unit="tokens",
        )
        self._render_counter = meter.create_counter(
            name="contextkit.renders.total",
            description="Number of window render operations",
            unit="renders",
        )
        self._is_attached = False

    @property
    def is_attached(self) -> bool:
        """Whether the exporter is currently attached to events."""
        return self._is_attached

    def attach(self) -> None:
        """Register event handlers to automatically export metrics.

        Listens for BLOCK_ADDED, WINDOW_RENDERED, and
        PIPELINE_COMPLETE events.
        """
        if self._is_attached:
            logger.warning("OTelExporter is already attached")
            return

        register_handler(ContextEvent.BLOCK_ADDED, self._on_block_added)
        register_handler(ContextEvent.WINDOW_RENDERED, self._on_rendered)
        register_handler(ContextEvent.PIPELINE_COMPLETE, self._on_pipeline_complete)
        self._is_attached = True
        logger.info("OTelExporter attached to context events")

    def _on_block_added(self, event_data: Any) -> None:
        """Handle BLOCK_ADDED events by incrementing counters.

        Args:
            event_data: The event data (BlockEventData or EventData).
        """
        token_count = getattr(event_data, "token_count", 0)
        self._block_counter.add(1)
        if token_count > 0:
            self._token_counter.add(token_count)

    def _on_rendered(self, event_data: Any) -> None:
        """Handle WINDOW_RENDERED events.

        Args:
            event_data: The event data.
        """
        self._render_counter.add(1)
        token_count = getattr(event_data, "token_count", 0)
        if token_count > 0:
            self._token_counter.add(
                token_count, {"operation": "render"}
            )

    def _on_pipeline_complete(self, event_data: Any) -> None:
        """Handle PIPELINE_COMPLETE events.

        Args:
            event_data: The event data (EventData with details).
        """
        if isinstance(event_data, EventData):
            tokens_saved = event_data.details.get("tokens_saved", 0)
            if tokens_saved > 0:
                self._pipeline_tokens_saved.add(tokens_saved)
