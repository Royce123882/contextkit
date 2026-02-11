"""Tests for explain mode."""

from __future__ import annotations

from contextkit.assembler import ContextAssembler
from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.events import clear_handlers
from contextkit.observe.explain import explain_block
from contextkit.observe.provenance import Mutation, Origin


class TestExplainBlock:
    """Tests for the explain_block function."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_explain_included_block(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        origin = Origin(
            source="rag",
            details={
                "query": "how does auth work",
                "retriever": "chroma/docs-index",
                "relevance_score": 0.87,
            },
        )
        window.add(
            ContextBlock(
                type=BlockType.RAG,
                content="Auth uses JWT...",
                priority=70,
                name="rag_auth",
                origin=origin,
            )
        )

        output = explain_block(window, "rag_auth")
        assert "INCLUDED" in output
        assert "rag_auth" in output
        assert "70" in output
        assert "rag" in output

    def test_explain_included_block_with_origin_details(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        origin = Origin(
            source="rag",
            details={
                "query": "auth flow",
                "retriever": "chroma",
                "relevance_score": 0.92,
            },
        )
        window.add(
            ContextBlock(
                type=BlockType.RAG,
                content="Auth details...",
                name="rag_block",
                origin=origin,
            )
        )

        output = explain_block(window, "rag_block")
        assert "auth flow" in output
        assert "chroma" in output
        assert "0.92" in output

    def test_explain_included_block_with_mutations(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        mutation = Mutation(
            step="CompactStep",
            action="compacted",
            detail="Summarized conversation",
            tokens_before=8000,
            tokens_after=4000,
        )
        window.add(
            ContextBlock(
                type=BlockType.SHORT_TERM_MEMORY,
                content="Compacted history",
                name="history",
                mutations=[mutation],
            )
        )

        output = explain_block(window, "history")
        assert "Mutations applied:" in output
        assert "CompactStep" in output
        assert "compacted" in output

    def test_explain_included_block_no_mutations(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="sys",
            )
        )

        output = explain_block(window, "sys")
        assert "No mutations applied" in output

    def test_explain_included_block_budget_impact(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="sys",
            )
        )

        output = explain_block(window, "sys")
        assert "Budget impact:" in output
        assert "100,000" in output

    def test_explain_excluded_block(self) -> None:
        window = ContextWindow(max_tokens=20)
        assembler = ContextAssembler(window)

        blocks = [
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="Short",
                priority=100,
                name="sys",
            ),
            ContextBlock(
                type=BlockType.RAG,
                content="a" * 500,
                priority=30,
                name="rag_big",
                origin=Origin(source="rag"),
            ),
        ]

        assembler.assemble(blocks)
        output = explain_block(window, "rag_big")
        assert "EXCLUDED" in output
        assert "budget_exceeded" in output

    def test_explain_excluded_block_shows_would_have_cost(self) -> None:
        window = ContextWindow(max_tokens=20)
        assembler = ContextAssembler(window)

        blocks = [
            ContextBlock(
                type=BlockType.RAG,
                content="a" * 500,
                priority=30,
                name="too_big",
            ),
        ]

        assembler.assemble(blocks)
        output = explain_block(window, "too_big")
        assert "Would have used" in output

    def test_explain_unknown_block(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        output = explain_block(window, "nonexistent")
        assert "No block with name" in output
        assert "nonexistent" in output

    def test_explain_after_assembly(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        assembler = ContextAssembler(window)

        blocks = [
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="System prompt",
                priority=100,
                name="sys",
                origin=Origin(source="prompt"),
            ),
        ]

        assembler.assemble(blocks)
        output = explain_block(window, "sys")
        assert "Added by: ContextAssembler" in output

    def test_window_explain_method(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="sys",
            )
        )
        output = window.explain("sys")
        assert "INCLUDED" in output

    def test_explain_block_with_no_origin(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="sys",
            )
        )
        output = explain_block(window, "sys")
        assert "INCLUDED" in output
        # Should not crash when origin is None
