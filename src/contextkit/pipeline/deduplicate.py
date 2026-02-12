"""Deduplicate pipeline step.

Removes blocks with overlapping content using word-overlap similarity.
"""

from __future__ import annotations

from typing import Any, List

from contextkit.constants import DEFAULT_SIMILARITY_THRESHOLD
from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep
from contextkit.utils.text_similarity import word_overlap_similarity


class DeduplicateStep(PipelineStep):
    """Remove blocks with overlapping content.

    Uses word-overlap similarity to detect near-duplicates.
    When two blocks are similar, the one with lower priority
    is removed.

    Args:
        similarity_threshold: Overlap threshold (0.0-1.0).
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._similarity_threshold = similarity_threshold

    @property
    def name(self) -> str:
        """Return the step name."""
        return "DeduplicateStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Remove duplicate blocks based on content similarity."""
        sorted_blocks = sorted(blocks, key=lambda block: block.priority, reverse=True)
        result: List[ContextBlock] = []

        for block in sorted_blocks:
            if not isinstance(block.content, str):
                result.append(block)
                continue

            duplicate_of = self._find_duplicate(block, result)
            if duplicate_of is not None:
                self._record_duplicate(block, duplicate_of)
            else:
                result.append(block)

        return result

    def _find_duplicate(
        self, block: ContextBlock, existing_blocks: List[ContextBlock]
    ) -> ContextBlock | None:
        """Find an existing block that is a near-duplicate of the given block."""
        if not isinstance(block.content, str):
            return None

        for existing in existing_blocks:
            if not isinstance(existing.content, str):
                continue
            similarity = word_overlap_similarity(block.content, existing.content)
            if similarity >= self._similarity_threshold:
                return existing

        return None

    def _record_duplicate(
        self, block: ContextBlock, duplicate_of: ContextBlock
    ) -> None:
        """Record a mutation for a block removed as a duplicate."""
        if isinstance(block.content, str) and isinstance(duplicate_of.content, str):
            similarity_score = word_overlap_similarity(
                block.content, duplicate_of.content
            )
        else:
            # Non-string content is never deduped by _find_duplicate,
            # so this branch is only a safety fallback.
            similarity_score = 0.0

        block.mutations.append(
            Mutation(
                step=self.name,
                action="removed",
                detail=(
                    f'overlaps with "{duplicate_of.display_name}" '
                    f"({similarity_score:.0%} similarity)"
                ),
                tokens_before=block.token_count,
                tokens_after=0,
            )
        )
