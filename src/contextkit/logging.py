"""Colored, structured logging for contextkit.

Provides a pre-configured logger with colored output so users can
easily see what the SDK is doing.  Colours are auto-detected (TTY)
and can be forced on or off via ``configure_logging(force_color=...)``.

Usage::

    from contextkit.logging import logger, configure_logging

    configure_logging(level="DEBUG")   # show everything
    configure_logging(level="WARNING") # quieter

The logger name is ``"contextkit"`` so it integrates cleanly with
any existing ``logging`` setup a user may already have.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import TextIO

# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_COLORS = {
    "DEBUG": "\033[36m",      # cyan
    "INFO": "\033[32m",       # green
    "WARNING": "\033[33m",    # yellow
    "ERROR": "\033[31m",      # red
    "CRITICAL": "\033[1;31m", # bold red
}

_SYMBOLS = {
    "DEBUG": "\u2022",    # bullet
    "INFO": "\u2714",     # check mark
    "WARNING": "\u26A0",  # warning sign
    "ERROR": "\u2718",    # cross mark
    "CRITICAL": "\u2718", # cross mark
}

# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class _ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colours and symbols to log output."""

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        symbol = _SYMBOLS.get(level, "\u2022")
        msg = record.getMessage()

        if self._use_color:
            color = _COLORS.get(level, "")
            prefix = f"{color}{_BOLD}{symbol} contextkit{_RESET} {color}[{level}]{_RESET}"
            return f"{prefix} {_DIM}{msg}{_RESET}"

        return f"{symbol} contextkit [{level}] {msg}"


class _JsonFormatter(logging.Formatter):
    """Formatter that outputs structured JSON log lines.

    Each log record is emitted as a single JSON object with
    ``timestamp``, ``level``, ``logger``, and ``message`` fields.
    Suitable for log aggregation services (Datadog, Elastic, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string.
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

logger = logging.getLogger("contextkit")


def _is_tty(stream: TextIO) -> bool:
    """Check whether *stream* is connected to a terminal."""
    try:
        return hasattr(stream, "isatty") and stream.isatty()
    except Exception:
        return False


def configure_logging(
    level: str | int = "WARNING",
    force_color: bool | None = None,
    stream: TextIO | None = None,
    log_format: str = "text",
) -> None:
    """Configure the ``contextkit`` logger with colored or JSON output.

    Safe to call multiple times -- previous handlers added by this
    function are removed first.

    Args:
        level: Log level name (``"DEBUG"``, ``"INFO"``, …) or int.
        force_color: ``True`` forces colour on, ``False`` forces off.
            ``None`` (default) auto-detects based on the output stream.
        stream: Output stream.  Defaults to ``sys.stderr``.
        log_format: Output format -- ``"text"`` (default) for human-readable
            colored output, or ``"json"`` for structured JSON logs suitable
            for log aggregation systems.
    """
    if stream is None:
        stream = sys.stderr

    if force_color is None:
        use_color = _is_tty(stream) and os.environ.get("NO_COLOR") is None
    else:
        use_color = force_color

    # Remove any existing contextkit handlers we previously added
    for handler in list(logger.handlers):
        if getattr(handler, "_contextkit_managed", False):
            logger.removeHandler(handler)

    new_handler = logging.StreamHandler(stream)
    new_handler._contextkit_managed = True  # type: ignore[attr-defined]

    if log_format == "json":
        new_handler.setFormatter(_JsonFormatter())
    else:
        new_handler.setFormatter(_ColorFormatter(use_color=use_color))

    logger.addHandler(new_handler)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)
    logger.setLevel(level)


def enable_debug() -> None:
    """Shortcut to enable full debug logging."""
    configure_logging(level="DEBUG")


def silence() -> None:
    """Shortcut to silence all contextkit logging."""
    configure_logging(level="CRITICAL")
