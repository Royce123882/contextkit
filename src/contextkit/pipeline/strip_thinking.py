"""Strip chain-of-thought reasoning from prior turns.

Removes ``<thinking>``, ``<scratchpad>``, and similar reasoning
trace blocks from context. Reasoning content is valuable during
generation but becomes pure noise in subsequent turns.

Research basis: LOCA-bench (arXiv:2602.07962) -- stripping thinking
content from prior turns reduces token usage without degrading
model performance.
"""

from __future__ import annotations

import re
from typing import Any, List

from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep

# Default patterns matching common reasoning trace wrappers.
# Each pattern uses re.DOTALL so '.' matches newlines.
_DEFAULT_PATTERNS: List[str] = [
    r"<thinking>.*?</thinking>",
    r"<scratchpad>.*?</scratchpad>",
    r"<reasoning>.*?</reasoning>",
    r"<inner_monologue>.*?</inner_monologue>",
    r"<chain_of_thought>.*?</chain_of_thought>",
]


class StripThinkingStep(PipelineStep):
    """Strip chain-of-thought reasoning traces from block content.

    Scans each block's string content for reasoning patterns
    (e.g. ``<thinking>...</thinking>``) and removes them. Blocks
    whose content becomes empty after stripping are removed entirely.

    Args:
        patterns: List of regex patterns to strip. Each is applied
            with ``re.DOTALL`` so ``.`` matches newlines. Defaults to
            common wrappers (``<thinking>``, ``<scratchpad>``, etc.).
        strip_empty: If True (default), blocks that become empty after
            stripping are removed from the output.
    """

    def __init__(
        self,
        patterns: List[str] | None = None,
        strip_empty: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        raw_patterns = patterns if patterns is not None else _DEFAULT_PATTERNS
        self._compiled_patterns = [re.compile(p, re.DOTALL) for p in raw_patterns]
        self._strip_empty = strip_empty

    @property
    def name(self) -> str:
        """Return the step name."""
        return "StripThinkingStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Strip reasoning traces from all string-content blocks.

        Args:
            blocks: Input blocks to process.

        Returns:
            Blocks with reasoning traces removed. Blocks that become
            empty after stripping are excluded if ``strip_empty`` is True.
        """
        result: List[ContextBlock] = []

        for block in blocks:
            if not isinstance(block.content, str):
                result.append(block)
                continue

            cleaned_content = self._apply_patterns(block.content)

            if cleaned_content == block.content:
                # No change -- keep as-is
                result.append(block)
                continue

            tokens_before = block.token_count

            if self._strip_empty and not cleaned_content.strip():
                # Content is now empty -- remove the block
                block.mutations.append(
                    Mutation(
                        step=self.name,
                        action="removed",
                        detail="block empty after stripping reasoning traces",
                        tokens_before=tokens_before,
                        tokens_after=0,
                    )
                )
                continue

            # Update the block content in place
            block.content = cleaned_content
            block.mutations.append(
                Mutation(
                    step=self.name,
                    action="compressed",
                    detail="stripped reasoning traces",
                    tokens_before=tokens_before,
                    tokens_after=block.token_count,
                )
            )
            result.append(block)

        return result

    def _apply_patterns(self, text: str) -> str:
        """Apply all compiled patterns to strip matching content.

        Args:
            text: The string to clean.

        Returns:
            The text with all matching patterns removed.
        """
        for compiled_pattern in self._compiled_patterns:
            text = compiled_pattern.sub("", text)
        return text
