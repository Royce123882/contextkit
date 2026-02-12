"""Tests for the contextkit.logging module: colored output configuration,
level filtering, handler deduplication, and top-level package exports.
"""

from __future__ import annotations

import io
import logging

from contextkit.logging import configure_logging, enable_debug, silence


class TestLoggingConfiguration:
    """configure_logging() sets up the contextkit logger with colored output."""

    def test_sets_requested_log_level(self) -> None:
        """The logger level should match the requested level string."""
        configure_logging(level="DEBUG")
        ctx_logger = logging.getLogger("contextkit")
        assert ctx_logger.level == logging.DEBUG
        silence()

    def test_plain_text_output_without_color(self) -> None:
        """With force_color=False, output contains no ANSI escape codes."""
        output_buffer = io.StringIO()
        configure_logging(level="INFO", force_color=False, stream=output_buffer)
        ctx_logger = logging.getLogger("contextkit")
        ctx_logger.info("plain text message")
        output = output_buffer.getvalue()
        assert "contextkit" in output
        assert "plain text message" in output
        assert "\033[" not in output
        silence()

    def test_ansi_codes_present_with_color(self) -> None:
        """With force_color=True, output contains ANSI escape sequences."""
        output_buffer = io.StringIO()
        configure_logging(level="INFO", force_color=True, stream=output_buffer)
        ctx_logger = logging.getLogger("contextkit")
        ctx_logger.info("colored message")
        output = output_buffer.getvalue()
        assert "\033[" in output
        silence()

    def test_enable_debug_sets_debug_level(self) -> None:
        """enable_debug() is a shortcut for configure_logging(level='DEBUG')."""
        enable_debug()
        ctx_logger = logging.getLogger("contextkit")
        assert ctx_logger.level == logging.DEBUG
        silence()

    def test_silence_sets_critical_level(self) -> None:
        """silence() suppresses all log output below CRITICAL."""
        silence()
        ctx_logger = logging.getLogger("contextkit")
        assert ctx_logger.level == logging.CRITICAL

    def test_repeated_configure_does_not_duplicate_handlers(self) -> None:
        """Calling configure_logging multiple times keeps exactly one handler."""
        configure_logging(level="INFO")
        configure_logging(level="DEBUG")
        configure_logging(level="WARNING")
        ctx_logger = logging.getLogger("contextkit")
        managed_handlers = [
            handler for handler in ctx_logger.handlers
            if getattr(handler, "_contextkit_managed", False)
        ]
        assert len(managed_handlers) == 1
        silence()

    def test_level_filtering_suppresses_lower_messages(self) -> None:
        """Messages below the configured level should not appear in output."""
        output_buffer = io.StringIO()
        configure_logging(level="WARNING", force_color=False, stream=output_buffer)
        ctx_logger = logging.getLogger("contextkit")
        ctx_logger.debug("debug should not appear")
        ctx_logger.info("info should not appear")
        ctx_logger.warning("warning message")
        output = output_buffer.getvalue()
        assert "warning message" in output
        assert "should not appear" not in output
        silence()

    def test_logging_exported_from_top_level_package(self) -> None:
        """configure_logging, enable_debug, and silence should be importable
        directly from the contextkit package.
        """
        import contextkit
        assert hasattr(contextkit, "configure_logging")
        assert hasattr(contextkit, "enable_debug")
        assert hasattr(contextkit, "silence")
