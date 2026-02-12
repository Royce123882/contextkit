"""LLM-based compaction pipeline step.

Splits block content into numbered paragraphs, saves the original
to a pluggable ``CompactionStore``, and calls a user-provided LLM
to produce a concise summary with ``[N]`` reference pointers back
to the original sections.

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
from contextkit.constants import DEFAULT_COMPACT_MIN_TOKENS
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

    After compaction, checks keyword retention between the original
    and compacted content.  If retention drops below *max_info_loss*,
    a warning is logged and the original block is returned unmodified.

    Args:
        llm: Sync callable ``(prompt: str) -> str`` for LLM compaction.
        async_llm: Async callable for use with ``pipeline.arun()``.
            If only ``llm`` is provided, the async path wraps it.
        store: ``CompactionStore`` for saving originals.
            Defaults to ``LocalCompactionStore(".contextkit/compacted")``.
        min_tokens: Only compact blocks above this token count.
        max_info_loss: Maximum acceptable keyword loss (0.0-1.0).
            If ``1.0 - keyword_retention`` exceeds this value, the
            original block is returned unmodified to prevent
            information collapse.
        paragraph_separator: How to split content into sections.

    Raises:
        ValueError: If neither ``llm`` nor ``async_llm`` is provided.
    """

    def __init__(
        self,
        llm: Callable[[str], str] | None = None,
        async_llm: Callable[[str], Awaitable[str]] | None = None,
        store: CompactionStore | None = None,
        min_tokens: int = DEFAULT_COMPACT_MIN_TOKENS,
        max_info_loss: float = 0.5,
        paragraph_separator: str = "\n\n",
        **kwargs: Any,
    ) -> None:
        if llm is None and async_llm is None:
            raise ValueError(
                "CompactStep requires an LLM. "
                "Provide either llm= (sync) or async_llm= (async)."
            )
        super().__init__(**kwargs)
        self._llm = llm
        self._async_llm = async_llm
        self._store = store or LocalCompactionStore()
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
        """Compact long blocks via LLM summarization."""
        return [self._compact_block(block) for block in blocks]

    # ------------------------------------------------------------------
    # Async path
    # ------------------------------------------------------------------

    async def async_process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Async compaction using ``async_llm`` when available."""
        return [await self._compact_block_async(block) for block in blocks]

    # ------------------------------------------------------------------
    # Per-block orchestration (sync)
    # ------------------------------------------------------------------

    def _compact_block(self, block: ContextBlock) -> ContextBlock:
        """Compact a single block synchronously."""
        if not isinstance(block.content, str):
            return block

        if block.token_count < self._min_tokens:
            return block

        before_content = block.content
        compacted, ref_uri = self._compact_with_llm_sync(block)
        return self._finalize(block, before_content, compacted, ref_uri)

    # ------------------------------------------------------------------
    # Per-block orchestration (async)
    # ------------------------------------------------------------------

    async def _compact_block_async(self, block: ContextBlock) -> ContextBlock:
        """Compact a single block asynchronously."""
        if not isinstance(block.content, str):
            return block

        if block.token_count < self._min_tokens:
            return block

        before_content = block.content

        if self._async_llm is not None:
            compacted, ref_uri = await self._compact_with_llm_async(block)
        else:
            compacted, ref_uri = await asyncio.to_thread(
                self._compact_with_llm_sync, block
            )

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
