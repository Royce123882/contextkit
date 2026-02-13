"""Budget warning system for context windows.

Monitors token usage against configurable thresholds and emits
warnings through both the event system and Python's logging module.
"""

from __future__ import annotations

import logging
from typing import List, Set

from contextkit.observe.event_models import BudgetEventData, ContextEvent
from contextkit.observe.events import emit

logger = logging.getLogger("contextkit")


class BudgetMonitor:
    """Monitors token usage and fires warnings at configurable thresholds.

    Thresholds are expressed as fractions (e.g. 0.75 means 75%).
    Each threshold fires at most once until usage drops below it,
    at which point it resets and can fire again.

    Attributes:
        thresholds: Sorted list of warning thresholds.
    """

    def __init__(self, thresholds: List[float]) -> None:
        self.thresholds = sorted(thresholds)
        self._fired: Set[float] = set()

    def check(
        self,
        token_count: int,
        max_tokens: int,
        largest_block_name: str | None = None,
        largest_block_tokens: int = 0,
    ) -> List[float]:
        """Check token usage against thresholds and emit warnings.

        Args:
            token_count: Current total tokens used.
            max_tokens: Maximum token budget.
            largest_block_name: Name of the largest block (for log message).
            largest_block_tokens: Token count of the largest block.

        Returns:
            List of thresholds that fired during this check.
        """
        if max_tokens <= 0:
            return []

        usage_fraction = token_count / max_tokens
        self._reset_thresholds_below(usage_fraction)

        fired_this_check: List[float] = []
        for threshold in self.thresholds:
            if usage_fraction >= threshold and threshold not in self._fired:
                self._fired.add(threshold)
                fired_this_check.append(threshold)
                self._fire_warning(
                    usage_fraction,
                    threshold,
                    token_count,
                    max_tokens,
                    largest_block_name,
                    largest_block_tokens,
                )

        return fired_this_check

    def _reset_thresholds_below(self, usage_fraction: float) -> None:
        """Reset thresholds that usage has dropped below."""
        thresholds_to_reset = {t for t in self._fired if usage_fraction < t}
        self._fired -= thresholds_to_reset

    @staticmethod
    def _format_warning_message(
        usage_fraction: float,
        token_count: int,
        max_tokens: int,
        largest_block_name: str | None,
        largest_block_tokens: int,
    ) -> str:
        """Format a human-readable budget warning message."""
        percent = usage_fraction * 100
        msg = f"budget at {percent:.0f}% ({token_count:,} / {max_tokens:,} tokens)"
        if largest_block_name:
            block_share = (
                (largest_block_tokens / max_tokens * 100) if max_tokens > 0 else 0
            )
            msg += (
                f" | largest block: {largest_block_name} "
                f"({largest_block_tokens:,} tokens, "
                f"{block_share:.0f}% of total)"
            )
        return msg

    @staticmethod
    def _fire_warning(
        usage_fraction: float,
        threshold: float,
        token_count: int,
        max_tokens: int,
        largest_block_name: str | None,
        largest_block_tokens: int,
    ) -> None:
        """Log a warning and emit a BUDGET_WARNING event."""
        msg = BudgetMonitor._format_warning_message(
            usage_fraction,
            token_count,
            max_tokens,
            largest_block_name,
            largest_block_tokens,
        )
        logger.warning("[contextkit] WARNING: %s", msg)

        percent = usage_fraction * 100
        threshold_percent = threshold * 100
        emit(
            BudgetEventData(
                event=ContextEvent.BUDGET_WARNING,
                percent=percent,
                threshold=threshold_percent,
                tokens_used=token_count,
                tokens_max=max_tokens,
            )
        )

    def reset(self) -> None:
        """Reset all fired thresholds."""
        self._fired.clear()
