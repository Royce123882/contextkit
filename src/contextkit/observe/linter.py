"""Context linting for common anti-patterns.

Static analysis tool that inspects a ContextWindow and reports
potential issues such as missing system prompts, high-priority
blocks lost in the middle, oversized blocks, redundant content,
and empty blocks.

Usage::

    from contextkit.observe.linter import ContextLinter
    linter = ContextLinter()
    warnings = linter.lint(window)
    for w in warnings:
        print(f"[{w.code}] {w.message}")
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from contextkit.core import ContextBlock, ContextWindow
from contextkit.core.block import BlockType
from contextkit.models import UnknownModelError, get_model
from contextkit.utils.text_similarity import word_overlap_similarity


class LintWarning(BaseModel):
    """A single warning from the context linter.

    Attributes:
        code: Machine-readable warning code (e.g. "no_system_prompt").
        message: Human-readable description with actionable advice.
        severity: Warning severity -- ``"info"``, ``"warning"``, or ``"error"``.
    """

    code: str = Field(description="Machine-readable warning code.")
    message: str = Field(description="Human-readable warning message.")
    severity: str = Field(
        default="warning",
        description="Severity level: 'info', 'warning', or 'error'.",
    )


class ContextLinter:
    """Detect common context assembly anti-patterns.

    Inspects a ContextWindow and returns a list of LintWarning
    objects for any detected issues. Each check is implemented as
    a separate method for clarity and extensibility.

    Args:
        high_priority_threshold: Priority at or above which blocks
            are considered "high priority" for lost-in-middle checks.
        redundancy_threshold: Word overlap above this value triggers
            redundancy warnings.
        oversized_ratio: Fraction of total budget above which a
            single block is considered oversized.
    """

    def __init__(
        self,
        high_priority_threshold: int = 80,
        redundancy_threshold: float = 0.8,
        oversized_ratio: float = 0.5,
    ) -> None:
        self._high_priority_threshold = high_priority_threshold
        self._redundancy_threshold = redundancy_threshold
        self._oversized_ratio = oversized_ratio

    def lint(self, window: ContextWindow) -> List[LintWarning]:
        """Run all lint checks on a ContextWindow.

        Args:
            window: The ContextWindow to inspect.

        Returns:
            List of LintWarning objects (empty if no issues found).
        """
        warnings: List[LintWarning] = []
        warnings.extend(self._check_missing_system_prompt(window))
        warnings.extend(self._check_lost_in_middle(window))
        warnings.extend(self._check_oversized_blocks(window))
        warnings.extend(self._check_redundant_blocks(window))
        warnings.extend(self._check_empty_blocks(window))
        warnings.extend(self._check_effective_window(window))
        return warnings

    def _check_missing_system_prompt(
        self,
        window: ContextWindow,
    ) -> List[LintWarning]:
        """Warn if no SYSTEM_PROMPT block is present.

        Args:
            window: The ContextWindow to check.

        Returns:
            A list with one warning if missing, empty otherwise.
        """
        has_system = any(
            block.type == BlockType.SYSTEM_PROMPT for block in window.blocks
        )
        if not has_system:
            return [
                LintWarning(
                    code="no_system_prompt",
                    message=(
                        "No SYSTEM_PROMPT block found. Most models perform "
                        "better with explicit system instructions. "
                        "Add one with ContextBlock.system('...')."
                    ),
                    severity="warning",
                )
            ]
        return []

    def _check_lost_in_middle(
        self,
        window: ContextWindow,
    ) -> List[LintWarning]:
        """Warn if high-priority blocks are in the middle third.

        Based on "Lost in the Middle" research: LLMs attend less
        to content in the middle of the context window.

        Args:
            window: The ContextWindow to check.

        Returns:
            A list of warnings for high-priority blocks in the middle.
        """
        blocks = window.blocks
        block_count = len(blocks)
        if block_count < 6:
            return []

        middle_start = block_count // 3
        middle_end = 2 * block_count // 3
        middle_blocks = blocks[middle_start:middle_end]

        high_priority_in_middle = [
            block for block in middle_blocks
            if block.priority >= self._high_priority_threshold
        ]
        if not high_priority_in_middle:
            return []

        block_names = [block.display_name for block in high_priority_in_middle]
        return [
            LintWarning(
                code="lost_in_middle",
                message=(
                    f"High-priority blocks {block_names} are in the middle "
                    f"third of the context. Models attend less to middle "
                    f"positions. Consider using ReorderStep('prefix_stable') "
                    f"or ReorderStep('important_edges')."
                ),
                severity="warning",
            )
        ]

    def _check_oversized_blocks(
        self,
        window: ContextWindow,
    ) -> List[LintWarning]:
        """Warn if any single block uses more than the oversized ratio of the budget.

        Args:
            window: The ContextWindow to check.

        Returns:
            A list of warnings for oversized blocks.
        """
        warnings: List[LintWarning] = []
        max_tokens = window.max_tokens
        if max_tokens <= 0:
            return warnings

        threshold_tokens = int(max_tokens * self._oversized_ratio)
        for block in window.blocks:
            if block.token_count > threshold_tokens:
                percentage = block.token_count / max_tokens * 100
                warnings.append(
                    LintWarning(
                        code="oversized_block",
                        message=(
                            f"Block '{block.display_name}' uses "
                            f"{block.token_count:,}/{max_tokens:,} tokens "
                            f"({percentage:.0f}% of budget). "
                            f"Consider splitting or compressing this block."
                        ),
                        severity="warning",
                    )
                )
        return warnings

    def _check_redundant_blocks(
        self,
        window: ContextWindow,
    ) -> List[LintWarning]:
        """Warn if blocks have high content overlap.

        Args:
            window: The ContextWindow to check.

        Returns:
            A list of warnings for redundant block pairs.
        """
        warnings: List[LintWarning] = []
        seen_blocks: List[ContextBlock] = []

        for block in window.blocks:
            if not isinstance(block.content, str):
                continue
            for seen in seen_blocks:
                if not isinstance(seen.content, str):
                    continue
                similarity = word_overlap_similarity(
                    block.content, seen.content
                )
                if similarity > self._redundancy_threshold:
                    warnings.append(
                        LintWarning(
                            code="redundant_blocks",
                            message=(
                                f"Blocks '{block.display_name}' and "
                                f"'{seen.display_name}' have "
                                f"{similarity:.0%} content overlap. "
                                f"Consider using DeduplicateStep to "
                                f"remove near-duplicates."
                            ),
                            severity="info",
                        )
                    )
                    break
            seen_blocks.append(block)

        return warnings

    def _check_effective_window(
        self,
        window: ContextWindow,
    ) -> List[LintWarning]:
        """Warn if token count exceeds the model's effective window size.

        Research basis: arXiv:2509.21361 -- significant gaps between
        advertised and effective context window sizes. Models often
        degrade well before hitting advertised limits.

        Args:
            window: The ContextWindow to check.

        Returns:
            A list with one warning if exceeding effective limits.
        """
        if window.model_name is None:
            return []

        try:
            spec = get_model(window.model_name)
        except UnknownModelError:
            return []

        effective_token_limit = spec.effective_max_tokens
        if effective_token_limit is None:
            return []

        if window.token_count > effective_token_limit:
            percentage = window.token_count / spec.max_context * 100
            return [
                LintWarning(
                    code="exceeds_effective_window",
                    message=(
                        f"Token count ({window.token_count:,}) exceeds the "
                        f"effective window ({effective_token_limit:,} tokens) for model "
                        f"'{window.model_name}'. The advertised limit is "
                        f"{spec.max_context:,} but quality degrades beyond "
                        f"{effective_token_limit:,}. Currently at {percentage:.0f}% of "
                        f"advertised capacity. Consider trimming context or "
                        f"using a pipeline."
                    ),
                    severity="warning",
                )
            ]
        return []

    def _check_empty_blocks(
        self,
        window: ContextWindow,
    ) -> List[LintWarning]:
        """Warn if any block has empty or whitespace-only content.

        Args:
            window: The ContextWindow to check.

        Returns:
            A list of warnings for empty blocks.
        """
        warnings: List[LintWarning] = []
        for block in window.blocks:
            is_empty = False
            if isinstance(block.content, str) and not block.content.strip():
                is_empty = True
            elif isinstance(block.content, list) and not block.content:
                is_empty = True

            if is_empty:
                warnings.append(
                    LintWarning(
                        code="empty_block",
                        message=(
                            f"Block '{block.display_name}' has empty content. "
                            f"Empty blocks waste budget without contributing "
                            f"information. Remove it or add content."
                        ),
                        severity="error",
                    )
                )
        return warnings
