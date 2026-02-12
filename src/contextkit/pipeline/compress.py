"""Token-level prompt compression pipeline step.

Prunes low-information tokens from block content to reduce token
count while preserving meaning.  Uses a pluggable scorer function.

Research basis: LLMLingua (Jiang et al., 2023) -- coarse-to-fine
token pruning using information scores achieves up to 20x
compression with minimal performance loss.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import List

from contextkit.constants import (
    DEFAULT_COMPACT_MIN_TOKENS,
    DEFAULT_COMPRESSION_RATIO,
)
from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep
from contextkit.utils.token_counting import count as count_tokens


def _idf_scorer(tokens: List[str]) -> List[float]:
    """Score tokens by inverse frequency within the document.

    High-frequency tokens (the, is, a) score low and are pruned
    first.  Rare tokens score high and are kept.

    Args:
        tokens: List of whitespace-split words.

    Returns:
        Per-token scores in the same order as *tokens*.
    """
    if not tokens:
        return []

    counts = Counter(tokens)
    total = len(tokens)
    return [1.0 / (1.0 + counts[t] / total) for t in tokens]


class CompressStep(PipelineStep):
    """Prune low-information tokens from block content.

    Each token is scored by a *scorer* function.  Tokens with the
    lowest scores are removed until the *compression_ratio* is met.
    Pruning removes tokens from the **middle** of the text outward
    to preserve the start/end structure (inspired by "Lost in the
    Middle" positional findings).

    Args:
        compression_ratio: Target ratio of tokens to keep (0.0-1.0).
            0.5 means keep roughly half the tokens.
        scorer: Custom ``(tokens) -> scores`` function.
            Defaults to IDF-based scoring.
        min_tokens: Only compress blocks with at least this many tokens.
    """

    def __init__(
        self,
        compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
        scorer: Callable[[List[str]], List[float]] | None = None,
        min_tokens: int = DEFAULT_COMPACT_MIN_TOKENS,
    ) -> None:
        self._compression_ratio = compression_ratio
        self._scorer = scorer or _idf_scorer
        self._min_tokens = min_tokens

    @property
    def name(self) -> str:
        """Return the step name."""
        return "CompressStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Compress each qualifying block via token pruning."""
        return [self._compress_block(block) for block in blocks]

    def _compress_block(self, block: ContextBlock) -> ContextBlock:
        """Compress a single block if it exceeds the minimum threshold."""
        if not isinstance(block.content, str):
            return block

        tokens_before = block.token_count
        if tokens_before < self._min_tokens:
            return block

        words = block.content.split()
        if not words:
            return block

        target_count = max(1, int(len(words) * self._compression_ratio))
        if target_count >= len(words):
            return block

        compressed_text = self._prune_tokens(words, target_count)
        tokens_after = count_tokens(compressed_text)

        if tokens_after >= tokens_before:
            return block

        new_block = block.model_copy(update={"content": compressed_text})
        achieved_ratio = tokens_after / tokens_before if tokens_before else 1.0
        new_block.mutations.append(
            Mutation(
                step=self.name,
                action="compressed",
                detail=(
                    f"{tokens_before:,} -> {tokens_after:,} tokens "
                    f"(ratio {achieved_ratio:.2f})"
                ),
                tokens_before=tokens_before,
                tokens_after=tokens_after,
            )
        )
        return new_block

    def _prune_tokens(self, words: List[str], target_count: int) -> str:
        """Remove lowest-scored tokens, preferring middle removal.

        Tokens are indexed by position.  A positional bonus is
        applied so that start/end tokens are slightly harder to
        remove, preserving context edges.
        """
        scores = self._scorer(words)
        total = len(words)

        # Apply positional bonus: edges get a small boost
        boosted: List[float] = []
        for idx, score in enumerate(scores):
            # U-shaped bonus: 0.0 at edges, up to 0.2 at centre
            if total > 1:
                normalised_pos = idx / (total - 1)
                position_penalty = 0.2 * (1.0 - abs(2.0 * normalised_pos - 1.0))
            else:
                position_penalty = 0.0
            boosted.append(score + (0.1 - position_penalty))

        # Select the top target_count tokens by boosted score
        indexed_scores = sorted(
            range(total), key=lambda i: boosted[i], reverse=True
        )
        kept_indices = sorted(indexed_scores[:target_count])

        return " ".join(words[i] for i in kept_indices)
