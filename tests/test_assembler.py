"""Tests for the ContextAssembler."""

from __future__ import annotations

from contextkit.assembler import AssemblyReport, BlockDecision, ContextAssembler
from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.events import ContextEvent, clear_handlers, on
from contextkit.observe.event_models import EventData
from contextkit.observe.provenance import Origin


class TestBlockDecision:
    """Tests for the BlockDecision model."""

    def test_create_included_decision(self) -> None:
        decision = BlockDecision(
            block_name="sys",
            block_type="system_prompt",
            tokens=100,
            priority=100,
            origin_summary="prompt/v1",
        )
        assert decision.reason is None

    def test_create_excluded_decision(self) -> None:
        decision = BlockDecision(
            block_name="rag_old",
            block_type="rag",
            tokens=5000,
            priority=30,
            origin_summary="rag/chroma",
            reason="budget_exceeded",
        )
        assert decision.reason == "budget_exceeded"


class TestAssemblyReport:
    """Tests for the AssemblyReport model."""

    def test_create_report(self) -> None:
        report = AssemblyReport(
            included=[
                BlockDecision(
                    block_name="sys",
                    block_type="system_prompt",
                    tokens=100,
                    priority=100,
                    origin_summary="prompt",
                )
            ],
            excluded=[
                BlockDecision(
                    block_name="rag",
                    block_type="rag",
                    tokens=5000,
                    priority=30,
                    origin_summary="rag",
                    reason="budget_exceeded",
                )
            ],
            total_tokens=100,
            budget_remaining=900,
            cost_estimate=0.001,
        )
        assert len(report.included) == 1
        assert len(report.excluded) == 1
        assert report.total_tokens == 100


class TestContextAssembler:
    """Tests for the ContextAssembler."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_assemble_all_blocks_fit(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        assembler = ContextAssembler(window)

        blocks = [
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="System prompt",
                priority=100,
                name="sys",
            ),
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="User info",
                priority=80,
                name="user",
            ),
        ]

        result = assembler.assemble(blocks)
        assert len(result) == 2

        report = assembler.report
        assert report is not None
        assert len(report.included) == 2
        assert len(report.excluded) == 0

    def test_assemble_priority_ordering(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        assembler = ContextAssembler(window)

        blocks = [
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="Low priority",
                priority=30,
                name="low",
            ),
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="High priority",
                priority=100,
                name="high",
            ),
        ]

        result = assembler.assemble(blocks)
        # Assembler adds by priority, so high goes first
        assert result.blocks[0].display_name == "high"
        assert result.blocks[1].display_name == "low"

    def test_assemble_drops_low_priority_when_budget_exceeded(self) -> None:
        window = ContextWindow(max_tokens=50)
        assembler = ContextAssembler(window)

        blocks = [
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="Important system instruction",
                priority=100,
                name="sys",
            ),
            ContextBlock(
                type=BlockType.RAG,
                content="a" * 500,  # Will exceed budget
                priority=30,
                name="rag_big",
            ),
        ]

        assembler.assemble(blocks)
        report = assembler.report
        assert report is not None
        assert len(report.excluded) >= 1
        excluded_names = [d.block_name for d in report.excluded]
        assert "rag_big" in excluded_names

    def test_report_includes_origin_summaries(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        assembler = ContextAssembler(window)

        origin = Origin(
            source="rag",
            details={"retriever": "chroma", "relevance_score": 0.87},
        )
        blocks = [
            ContextBlock(
                type=BlockType.RAG,
                content="test",
                priority=70,
                name="rag_block",
                origin=origin,
            ),
        ]

        assembler.assemble(blocks)
        report = assembler.report
        assert report is not None
        assert "rag" in report.included[0].origin_summary

    def test_report_includes_unknown_for_no_origin(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        assembler = ContextAssembler(window)

        blocks = [
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="sys",
            ),
        ]

        assembler.assemble(blocks)
        report = assembler.report
        assert report is not None
        assert report.included[0].origin_summary == "unknown"

    def test_assembly_emits_event(self) -> None:
        received: list[EventData] = []

        @on(ContextEvent.ASSEMBLY_COMPLETE)
        def handler(event: EventData) -> None:
            received.append(event)

        window = ContextWindow(max_tokens=100_000)
        assembler = ContextAssembler(window)
        assembler.assemble(
            [
                ContextBlock(
                    type=BlockType.SYSTEM_PROMPT,
                    content="test",
                )
            ]
        )
        assert len(received) == 1
        assert received[0].details["included_count"] == 1

    def test_report_stored_on_window(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        assembler = ContextAssembler(window)
        assembler.assemble(
            [
                ContextBlock(
                    type=BlockType.SYSTEM_PROMPT,
                    content="test",
                )
            ]
        )
        assert window._assembly_report is not None
        assert isinstance(window._assembly_report, AssemblyReport)

    def test_report_cost_estimate(self) -> None:
        window = ContextWindow(model="claude-sonnet-4-5-20250929")
        assembler = ContextAssembler(window)
        assembler.assemble(
            [
                ContextBlock(
                    type=BlockType.SYSTEM_PROMPT,
                    content="Hello world",
                )
            ]
        )
        report = assembler.report
        assert report is not None
        assert report.cost_estimate >= 0

    def test_excluded_decision_has_reason(self) -> None:
        window = ContextWindow(max_tokens=10)
        assembler = ContextAssembler(window)
        assembler.assemble(
            [
                ContextBlock(
                    type=BlockType.RAG,
                    content="a" * 500,
                    name="too_big",
                )
            ]
        )
        report = assembler.report
        assert report is not None
        assert len(report.excluded) == 1
        assert report.excluded[0].reason == "budget_exceeded"

    def test_assemble_empty_list(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        assembler = ContextAssembler(window)
        result = assembler.assemble([])
        assert len(result) == 0
        report = assembler.report
        assert report is not None
        assert report.total_tokens == 0


# ---------------------------------------------------------------------------
# Async assembler
# ---------------------------------------------------------------------------


class TestAsyncContextAssembler:
    """ContextAssembler.aassemble() is an awaitable mirror of assemble()."""

    async def test_aassemble_returns_populated_window(self) -> None:
        """aassemble should return the same window with blocks added."""
        window = ContextWindow(max_tokens=10_000)
        assembler = ContextAssembler(window)
        blocks = [
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="hello world",
                priority=80,
                name="greeting",
            )
        ]
        result = await assembler.aassemble(blocks)
        assert result is window
        assert len(result.blocks) == 1

    async def test_aassemble_matches_sync_assemble(self) -> None:
        """Async and sync assembly should produce identical windows."""
        blocks = [
            ContextBlock(type=BlockType.USER_CONTEXT, content="hello", priority=80, name="first_block"),
            ContextBlock(type=BlockType.USER_CONTEXT, content="world", priority=60, name="second_block"),
        ]
        sync_window = ContextWindow(max_tokens=10_000)
        ContextAssembler(sync_window).assemble(list(blocks))

        async_window = ContextWindow(max_tokens=10_000)
        await ContextAssembler(async_window).aassemble(list(blocks))

        assert sync_window.token_count == async_window.token_count
        assert len(sync_window.blocks) == len(async_window.blocks)
