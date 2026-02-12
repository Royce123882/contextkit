"""Mask pipeline step.

Replaces older or low-priority block content with short
placeholders, preserving structural context while saving tokens.

Based on the observation-masking technique from JetBrains Research
("Cutting Through the Noise", 2025) which showed that replacing
older observations with placeholders matches LLM summarization
in both cost savings and task-solving ability.
"""

from __future__ import annotations

from typing import Any, List

from contextkit.utils.token_counting import count as count_tokens
from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep

_DEFAULT_PLACEHOLDER = "[details omitted for brevity]"


class MaskStep(PipelineStep):
    """Replace old block content with a short placeholder.

    Keeps the most recent *window* blocks intact and masks the
    rest.  Masked blocks remain in the context (so the model
    knows something was there) but consume far fewer tokens.

    Args:
        window: Number of most-recent blocks to keep unmasked.
        placeholder: Text to substitute for masked content.
        block_types: If set, only mask blocks of these types.
            Other types are always kept.
        min_tokens: Only mask blocks that are at least this many
            tokens (avoids masking blocks that are already tiny).
    """

    def __init__(
        self,
        window: int = 5,
        placeholder: str = _DEFAULT_PLACEHOLDER,
        block_types: List[str] | None = None,
        min_tokens: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._window = window
        self._placeholder = placeholder
        self._block_types = set(block_types) if block_types else None
        self._min_tokens = min_tokens

    @property
    def name(self) -> str:
        """Return the step name."""
        return "MaskStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Mask older blocks outside the recency window."""
        if len(blocks) <= self._window:
            return blocks

        # Identify maskable blocks (those matching type filter)
        maskable_indices: List[int] = []
        for idx, block in enumerate(blocks):
            if self._block_types and block.type.value not in self._block_types:
                continue
            maskable_indices.append(idx)

        # Keep the last `window` maskable blocks; mask the rest
        to_keep = set(maskable_indices[-self._window :])
        result: List[ContextBlock] = []

        for idx, block in enumerate(blocks):
            if idx in to_keep or idx not in maskable_indices:
                result.append(block)
                continue

            if not isinstance(block.content, str):
                result.append(block)
                continue

            if block.token_count < self._min_tokens:
                result.append(block)
                continue

            tokens_before = block.token_count
            masked = block.model_copy(update={"content": self._placeholder})
            tokens_after = count_tokens(self._placeholder)
            masked.mutations.append(
                Mutation(
                    step=self.name,
                    action="masked",
                    detail=(
                        f"replaced {tokens_before} tokens with placeholder "
                        f"({tokens_after} tokens)"
                    ),
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                )
            )
            result.append(masked)

        return result
