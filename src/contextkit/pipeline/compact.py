"""LLM-based compaction pipeline step.

Splits block content into numbered paragraphs, saves the original
to a pluggable ``CompactionStore``, and calls a user-provided LLM
to produce a concise summary with ``[N]`` reference pointers back
to the original sections.

Falls back to simple truncation when no LLM is provided, preserving
backwards compatibility with earlier versions of the SDK.

Research basis: Ravaut et al. (2023) -- summarization can cause
"information collapse" where key terms vanish; monitoring
keyword retention catches this before it reaches the model.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, List

from contextkit.compaction.store import CompactionStore, LocalCompactionStore
from contextkit.constants import (
    DEFAULT_COMPACT_MIN_TOKENS,
    DEFAULT_COMPACT_TARGET_RATIO,
)
from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep
from contextkit.utils.text_similarity import word_overlap_score
from contextkit.utils.token_counting import count as count_tokens

logger = logging.getLogger("contextkit")

_COMPACTION_PROMPT_TEMPLATE = """\
Summarize the following content concisely. Use [N] references to cite \
the original sections by their number. Preserve key technical terms, \
names, and numbers. Do not introduce information not in the original.

{numbered_content}"""


class CompactStep(PipelineStep):
    """LLM-based compaction with reference pointers to original content.

    Splits block content into numbered paragraphs, saves the original
    to a ``CompactionStore``, and calls an LLM to produce a concise
    summary using ``[N]`` references to cite original sections.

    When no LLM is provided, falls back to simple truncation
    (backwards compatible with the previous implementation).

    After compaction, checks keyword retention between the original
    and compacted content.  If retention drops below *max_info_loss*,
    a warning is logged and the original block is returned unmodified.

    Args:
        llm: Sync callable ``(prompt: str) -> str`` for LLM compaction.
        async_llm: Async callable for use with ``pipeline.arun()``.
            If only ``llm`` is provided, the async path wraps it.
        store: ``CompactionStore`` for saving originals.
            Defaults to ``LocalCompactionStore(".contextkit/compacted")``.
        compactor: Legacy sync compaction function. If provided and
            ``llm`` is None, used as the compaction function.
        target_ratio: Target compression ratio for the truncation
            fallback (0.0-1.0).
        min_tokens: Only compact blocks above this token count.
        max_info_loss: Maximum acceptable keyword loss (0.0-1.0).
            If ``1.0 - keyword_retention`` exceeds this value, the
            original block is returned unmodified to prevent
            information collapse.
        paragraph_separator: How to split content into sections.
    """

    def __init__(
        self,
        llm: Callable[[str], str] | None = None,
        async_llm: Callable[[str], Awaitable[str]] | None = None,
        store: CompactionStore | None = None,
        compactor: Callable[[str], str] | None = None,
        target_ratio: float = DEFAULT_COMPACT_TARGET_RATIO,
        min_tokens: int = DEFAULT_COMPACT_MIN_TOKENS,
        max_info_loss: float = 0.5,
        paragraph_separator: str = "\n\n",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._async_llm = async_llm
        self._store = store or LocalCompactionStore()
        self._compactor = compactor
        self._target_ratio = target_ratio
        self._min_tokens = min_tokens
        self._max_info_loss = max_info_loss
        self._paragraph_separator = paragraph_separator

    @property
    def name(self) -> str:
        """Return the step name."""
        return "CompactStep"

    # ------------------------------------------------------------------
    # Sync path
    # ------------------------------------------------------------------

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Compact long blocks via LLM or truncation fallback."""
        return [self._compact_block(block) for block in blocks]

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def async_process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Async compaction using ``async_llm`` when available."""
        if self._llm is None and self._async_llm is None:
            return self.process(blocks)
        return [await self._compact_block_async(block) for block in blocks]

    # ------------------------------------------------------------------
    # Per-block orchestration (sync)
    # ------------------------------------------------------------------

    def _compact_block(self, block: ContextBlock) -> ContextBlock:
        """Compact a single block synchronously."""
        if not isinstance(block.content, str):
            return block

        tokens_before = block.token_count
        if tokens_before < self._min_tokens:
            return block

        before_content = block.content

        if self._llm is not None:
            compacted, ref_uri = self._compact_with_llm_sync(block)
        elif self._compactor is not None:
            compacted = self._compactor(before_content)
            ref_uri = None
        else:
            compacted = self._default_truncator(before_content)
            ref_uri = None

        return self._finalize(block, before_content, compacted, ref_uri)

    # ------------------------------------------------------------------
    # Per-block orchestration (async)
    # ------------------------------------------------------------------

    async def _compact_block_async(self, block: ContextBlock) -> ContextBlock:
        """Compact a single block asynchronously."""
        if not isinstance(block.content, str):
            return block

        tokens_before = block.token_count
        if tokens_before < self._min_tokens:
            return block

        before_content = block.content

        if self._async_llm is not None:
            compacted, ref_uri = await self._compact_with_llm_async(block)
        elif self._llm is not None:
            compacted, ref_uri = await asyncio.to_thread(
                self._compact_with_llm_sync, block
            )
        elif self._compactor is not None:
            compacted = self._compactor(before_content)
            ref_uri = None
        else:
            compacted = self._default_truncator(before_content)
            ref_uri = None

        return self._finalize(block, before_content, compacted, ref_uri)

    # ------------------------------------------------------------------
    # LLM compaction
    # ------------------------------------------------------------------

    def _compact_with_llm_sync(
        self, block: ContextBlock
    ) -> tuple[str, str | None]:
        """Run LLM compaction synchronously, saving original to store."""
        assert self._llm is not None
        content = str(block.content)

        paragraphs = self._split_paragraphs(content)
        numbered_content = self._build_numbered_content(paragraphs)
        markdown = self._build_markdown(block, numbered_content)
        key = self._make_key(block)

        ref_uri: str | None = None
        if self._store:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Inside an existing event loop -- create a new thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=1
                ) as pool:
                    ref_uri = pool.submit(
                        asyncio.run, self._store.save(key, markdown)
                    ).result()
            else:
                ref_uri = asyncio.run(self._store.save(key, markdown))

        prompt = _COMPACTION_PROMPT_TEMPLATE.format(
            numbered_content=numbered_content
        )
        compacted = self._llm(prompt)
        return compacted, ref_uri

    async def _compact_with_llm_async(
        self, block: ContextBlock
    ) -> tuple[str, str | None]:
        """Run LLM compaction asynchronously, saving original to store."""
        content = str(block.content)

        paragraphs = self._split_paragraphs(content)
        numbered_content = self._build_numbered_content(paragraphs)
        markdown = self._build_markdown(block, numbered_content)
        key = self._make_key(block)

        ref_uri = await self._store.save(key, markdown) if self._store else None

        prompt = _COMPACTION_PROMPT_TEMPLATE.format(
            numbered_content=numbered_content
        )

        if self._async_llm is not None:
            compacted = await self._async_llm(prompt)
        else:
            assert self._llm is not None
            compacted = await asyncio.to_thread(self._llm, prompt)

        return compacted, ref_uri

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _finalize(
        self,
        block: ContextBlock,
        before_content: str,
        compacted: str,
        ref_uri: str | None,
    ) -> ContextBlock:
        """Apply collapse detection, create mutation, return new block."""
        tokens_before = count_tokens(before_content)
        tokens_after = count_tokens(compacted)

        if tokens_after >= tokens_before:
            return block

        keyword_retention = word_overlap_score(before_content, compacted)
        info_loss = 1.0 - keyword_retention

        if info_loss > self._max_info_loss:
            logger.warning(
                "Collapse detected in block '%s': keyword retention %.0f%%, "
                "info loss %.0f%% exceeds threshold %.0f%%. "
                "Returning original block unmodified.",
                block.display_name,
                keyword_retention * 100,
                info_loss * 100,
                self._max_info_loss * 100,
            )
            return block

        update: dict[str, Any] = {"content": compacted}
        if ref_uri is not None:
            metadata = dict(block.metadata) if block.metadata else {}
            metadata["compaction_ref"] = ref_uri
            update["metadata"] = metadata

        new_block = block.model_copy(update=update)

        detail = f"{tokens_before:,} -> {tokens_after:,} tokens"
        if ref_uri:
            detail += f" (ref: {ref_uri})"

        new_block.mutations.append(
            Mutation(
                step=self.name,
                action="compacted",
                detail=detail,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                before_content=before_content,
                after_content=compacted,
            )
        )
        return new_block

    def _split_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs using the configured separator."""
        parts = content.split(self._paragraph_separator)
        return [p.strip() for p in parts if p.strip()]

    def _build_numbered_content(self, paragraphs: List[str]) -> str:
        """Build numbered sections from paragraphs."""
        sections = []
        for idx, para in enumerate(paragraphs, 1):
            sections.append(f"## [{idx}]\n{para}")
        return "\n\n".join(sections)

    def _build_markdown(
        self, block: ContextBlock, numbered_content: str
    ) -> str:
        """Build the full markdown document to save to the store."""
        now = datetime.now(timezone.utc).isoformat()
        header = (
            f"# Compacted Context: {block.display_name}\n"
            f"# Block type: {block.type.value}\n"
            f"# Saved: {now}\n\n"
        )
        return header + numbered_content

    def _make_key(self, block: ContextBlock) -> str:
        """Generate a unique key for the compaction artifact."""
        content_hash = hashlib.sha256(
            str(block.content).encode()
        ).hexdigest()[:12]
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in block.display_name
        )
        return f"{safe_name}_{content_hash}"

    def _default_truncator(self, content: str) -> str:
        """Truncate content to the target ratio, breaking at sentence boundaries."""
        target_len = int(len(content) * self._target_ratio)
        if target_len >= len(content):
            return content

        truncated = content[:target_len]
        last_period = truncated.rfind(".")
        if last_period > target_len * 0.5:
            return truncated[: last_period + 1]
        return truncated
