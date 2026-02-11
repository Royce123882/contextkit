"""Tests for the budget warning system."""

from __future__ import annotations

import logging

from contextkit.events import (
    BudgetEventData,
    ContextEvent,
    clear_handlers,
    on,
)
from contextkit.observe.warnings import BudgetMonitor


class TestBudgetMonitor:
    """Tests for BudgetMonitor threshold checking."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_warning_fires_at_threshold(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.75])
        fired = monitor.check(
            token_count=80, max_tokens=100
        )
        assert 0.75 in fired

    def test_warning_does_not_fire_below_threshold(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.75])
        fired = monitor.check(
            token_count=50, max_tokens=100
        )
        assert len(fired) == 0

    def test_warning_does_not_refire_same_threshold(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.75])
        first = monitor.check(token_count=80, max_tokens=100)
        second = monitor.check(token_count=85, max_tokens=100)
        assert 0.75 in first
        assert 0.75 not in second

    def test_warning_resets_when_below_threshold(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.75])
        first = monitor.check(token_count=80, max_tokens=100)
        assert 0.75 in first
        # Drop below threshold
        monitor.check(token_count=50, max_tokens=100)
        # Should fire again
        third = monitor.check(token_count=80, max_tokens=100)
        assert 0.75 in third

    def test_multiple_thresholds(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.50, 0.75, 0.90])
        fired = monitor.check(token_count=80, max_tokens=100)
        assert 0.50 in fired
        assert 0.75 in fired
        assert 0.90 not in fired

    def test_multiple_thresholds_progressive(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.50, 0.75, 0.90])
        fired1 = monitor.check(token_count=55, max_tokens=100)
        assert 0.50 in fired1
        assert 0.75 not in fired1

        fired2 = monitor.check(token_count=80, max_tokens=100)
        assert 0.50 not in fired2  # Already fired
        assert 0.75 in fired2

        fired3 = monitor.check(token_count=95, max_tokens=100)
        assert 0.90 in fired3

    def test_emits_budget_warning_event(self) -> None:
        received: list[BudgetEventData] = []

        @on(ContextEvent.BUDGET_WARNING)
        def handler(event: BudgetEventData) -> None:
            received.append(event)

        monitor = BudgetMonitor(thresholds=[0.75])
        monitor.check(token_count=80, max_tokens=100)
        assert len(received) == 1
        assert received[0].percent == 80.0
        assert received[0].threshold == 75.0

    def test_log_message_contains_details(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.75])
        logger = logging.getLogger("contextkit")
        with _CaptureLogs(logger) as logs:
            monitor.check(
                token_count=80,
                max_tokens=100,
                largest_block_name="history",
                largest_block_tokens=60,
            )
        assert any("80%" in record.message for record in logs)
        assert any("history" in record.message for record in logs)

    def test_zero_max_tokens_returns_empty(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.75])
        fired = monitor.check(token_count=80, max_tokens=0)
        assert len(fired) == 0

    def test_reset_clears_fired(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.75])
        monitor.check(token_count=80, max_tokens=100)
        monitor.reset()
        fired = monitor.check(token_count=80, max_tokens=100)
        assert 0.75 in fired

    def test_thresholds_are_sorted(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.90, 0.50, 0.75])
        assert monitor.thresholds == [0.50, 0.75, 0.90]

    def test_largest_block_info_in_log(self) -> None:
        monitor = BudgetMonitor(thresholds=[0.75])
        logger = logging.getLogger("contextkit")
        with _CaptureLogs(logger) as logs:
            monitor.check(
                token_count=80,
                max_tokens=100,
                largest_block_name="conversation",
                largest_block_tokens=50,
            )
        log_messages = " ".join(r.message for r in logs)
        assert "conversation" in log_messages
        assert "50" in log_messages


class _LogCapture:
    """Helper to capture log records."""

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []

    def handle(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _CaptureLogs:
    """Context manager to capture log records from a logger."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._handler = logging.Handler()
        self._records: list[logging.LogRecord] = []

    def __enter__(self) -> list[logging.LogRecord]:
        self._handler.emit = lambda record: self._records.append(record)  # type: ignore[method-assign]
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.DEBUG)
        return self._records

    def __exit__(self, *args: object) -> None:
        self._logger.removeHandler(self._handler)
