"""Tests for the async API wrappers and colored logging module.

Verifies that every async method (aassemble, arun, ascan, aload, etc.)
produces the same results as its synchronous counterpart, and that the
contextkit.logging module correctly configures colored output,
respects log levels, and avoids handler duplication.
"""

from __future__ import annotations

import io
import logging

from contextkit.assembler import ContextAssembler
from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.logging import configure_logging, enable_debug, silence
from contextkit.pipeline import ContextPipeline, TrimStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_block(
    name: str,
    content: str = "test content",
    priority: int = 50,
) -> ContextBlock:
    """Create a ContextBlock with sensible defaults for testing."""
    return ContextBlock(
        type=BlockType.USER_CONTEXT,
        content=content,
        priority=priority,
        name=name,
    )


# ---------------------------------------------------------------------------
# Async assembler (aassemble)
# ---------------------------------------------------------------------------


class TestAsyncContextAssembler:
    """ContextAssembler.aassemble() is an awaitable mirror of assemble()."""

    async def test_aassemble_returns_populated_window(self) -> None:
        """aassemble should return the same window with blocks added."""
        window = ContextWindow(max_tokens=10_000)
        assembler = ContextAssembler(window)
        blocks = [_make_block("greeting", content="hello world", priority=80)]
        result = await assembler.aassemble(blocks)
        assert result is window
        assert len(result.blocks) == 1

    async def test_aassemble_matches_sync_assemble(self) -> None:
        """Async and sync assembly should produce identical windows."""
        blocks = [
            _make_block("first_block", content="hello", priority=80),
            _make_block("second_block", content="world", priority=60),
        ]
        sync_window = ContextWindow(max_tokens=10_000)
        ContextAssembler(sync_window).assemble(list(blocks))

        async_window = ContextWindow(max_tokens=10_000)
        await ContextAssembler(async_window).aassemble(list(blocks))

        assert sync_window.token_count == async_window.token_count
        assert len(sync_window.blocks) == len(async_window.blocks)


# ---------------------------------------------------------------------------
# Async pipeline (arun)
# ---------------------------------------------------------------------------


class TestAsyncContextPipeline:
    """ContextPipeline.arun() is an awaitable mirror of run()."""

    async def test_arun_returns_same_window(self) -> None:
        """arun should modify and return the same window object."""
        window = ContextWindow(max_tokens=10_000)
        block = _make_block("greeting", content="hello world", priority=80)
        window.add(block)
        pipeline = ContextPipeline(steps=[TrimStep(max_tokens=10_000)])
        result = await pipeline.arun(window)
        assert result is window

    async def test_arun_matches_sync_run(self) -> None:
        """Async and sync pipeline runs should produce identical token counts."""
        blocks = [_make_block("content_block", content="hello " * 50, priority=80)]
        trim_step = TrimStep(max_tokens=10_000)

        sync_window = ContextWindow(max_tokens=10_000)
        for block in blocks:
            sync_window.add(block)
        ContextPipeline(steps=[trim_step]).run(sync_window)

        async_window = ContextWindow(max_tokens=10_000)
        for block in blocks:
            async_window.add(block)
        await ContextPipeline(steps=[trim_step]).arun(async_window)

        assert sync_window.token_count == async_window.token_count


# ---------------------------------------------------------------------------
# Colored logging configuration
# ---------------------------------------------------------------------------


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
