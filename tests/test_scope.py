"""Tests for multi-agent scope, timeline, and handoff (Phase 5)."""

from __future__ import annotations

import json
import os
import tempfile

from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.observe.context_timeline import ContextTimeline
from contextkit.observe.turn_snapshot import TurnSnapshot
from contextkit.scope import (
    ContextScope,
    HandoffPackage,
    Scratchpad,
    SharedMemory,
)


class TestTurnSnapshot:
    """Tests for the TurnSnapshot model."""

    def test_create_with_defaults(self) -> None:
        snap = TurnSnapshot(turn=1)
        assert snap.turn == 1
        assert snap.token_count == 0
        assert snap.block_names == []
        assert snap.blocks_added == []
        assert snap.blocks_removed == []
        assert snap.events == []
        assert snap.timestamp is not None

    def test_create_with_all_fields(self) -> None:
        snap = TurnSnapshot(
            turn=3,
            token_count=500,
            max_tokens=1000,
            block_count=5,
            block_names=["a", "b"],
            blocks_added=["b"],
            blocks_removed=["c"],
            budget_percent=50.0,
            events=["warning"],
        )
        assert snap.budget_percent == 50.0
        assert snap.events == ["warning"]


class TestContextTimeline:
    """Tests for the ContextTimeline."""

    def test_record_snapshot(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        snap = tl.record(
            token_count=100,
            block_names=["a", "b"],
        )
        assert snap.turn == 1
        assert snap.token_count == 100
        assert tl.current_turn == 1

    def test_record_multiple_turns(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        tl.record(100, ["a"])
        tl.record(200, ["a", "b"])
        tl.record(150, ["b"])
        assert tl.current_turn == 3
        assert len(tl.snapshots) == 3

    def test_record_computes_diffs(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        tl.record(100, ["a", "b"])
        snap = tl.record(150, ["b", "c"])
        assert "c" in snap.blocks_added
        assert "a" in snap.blocks_removed

    def test_record_first_turn_all_added(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        snap = tl.record(100, ["a", "b"])
        assert sorted(snap.blocks_added) == ["a", "b"]
        assert snap.blocks_removed == []

    def test_budget_percent(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        snap = tl.record(500, ["a"])
        assert snap.budget_percent == 50.0

    def test_budget_percent_zero_max(self) -> None:
        tl = ContextTimeline(max_tokens=0)
        snap = tl.record(100, ["a"])
        assert snap.budget_percent == 0.0

    def test_record_with_events(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        snap = tl.record(100, ["a"], events=["budget_warning"])
        assert snap.events == ["budget_warning"]

    def test_snapshot_at(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        tl.record(100, ["a"])
        tl.record(200, ["a", "b"])
        snap = tl.snapshot_at(1)
        assert snap is not None
        assert snap.turn == 1
        assert snap.token_count == 100

    def test_snapshot_at_not_found(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        tl.record(100, ["a"])
        assert tl.snapshot_at(99) is None

    def test_timeline_summary(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        tl.record(100, ["a"])
        tl.record(200, ["a", "b"])
        summary = tl.timeline_summary()
        assert "Turn" in summary
        assert "tokens" in summary

    def test_timeline_summary_with_changes(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        tl.record(100, ["a"])
        tl.record(200, ["a", "b"])
        summary = tl.timeline_summary()
        assert "+b" in summary

    def test_export_json(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        tl.record(100, ["a"])
        tl.record(200, ["a", "b"])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            tl.export(path)
            with open(path) as f:
                data = json.load(f)
            assert data["total_turns"] == 2
            assert len(data["snapshots"]) == 2
            assert data["max_tokens"] == 1000
        finally:
            os.unlink(path)

    def test_clear(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        tl.record(100, ["a"])
        tl.clear()
        assert tl.current_turn == 0
        assert len(tl.snapshots) == 0

    def test_snapshots_returns_copy(self) -> None:
        tl = ContextTimeline(max_tokens=1000)
        tl.record(100, ["a"])
        snaps = tl.snapshots
        snaps.clear()
        assert len(tl.snapshots) == 1


class TestScratchpad:
    """Tests for the Scratchpad."""

    def test_write_and_read(self) -> None:
        pad = Scratchpad()
        pad.write("plan", "Step 1: Research")
        assert pad.read("plan") == "Step 1: Research"

    def test_read_not_found(self) -> None:
        pad = Scratchpad()
        assert pad.read("nonexistent") is None

    def test_delete(self) -> None:
        pad = Scratchpad()
        pad.write("key", "value")
        assert pad.delete("key") is True
        assert pad.read("key") is None
        assert pad.delete("key") is False

    def test_list_keys(self) -> None:
        pad = Scratchpad()
        pad.write("b_key", "B")
        pad.write("a_key", "A")
        assert pad.list_keys() == ["a_key", "b_key"]

    def test_to_block(self) -> None:
        pad = Scratchpad()
        pad.write("plan", "Do research")
        pad.write("status", "In progress")
        block = pad.to_block()
        assert block.type.value == "scratchpad"
        assert "[plan]" in block.content
        assert "[status]" in block.content
        assert block.origin is not None
        assert block.origin.source == "scratchpad"
        assert block.origin.details["note_count"] == 2

    def test_to_block_empty(self) -> None:
        pad = Scratchpad()
        block = pad.to_block()
        assert block.content == "(empty)"

    def test_to_block_custom_priority(self) -> None:
        pad = Scratchpad()
        pad.write("key", "val")
        block = pad.to_block(priority=80)
        assert block.priority == 80

    def test_clear(self) -> None:
        pad = Scratchpad()
        pad.write("key", "val")
        pad.clear()
        assert pad.note_count == 0

    def test_note_count(self) -> None:
        pad = Scratchpad()
        assert pad.note_count == 0
        pad.write("a", "A")
        pad.write("b", "B")
        assert pad.note_count == 2

    def test_overwrite_existing_key(self) -> None:
        pad = Scratchpad()
        pad.write("key", "old")
        pad.write("key", "new")
        assert pad.read("key") == "new"
        assert pad.note_count == 1


class TestSharedMemory:
    """Tests for SharedMemory."""

    def _make_block(self, name: str) -> ContextBlock:
        return ContextBlock(
            type=BlockType.USER_CONTEXT,
            content=f"Shared: {name}",
            name=name,
        )

    def test_publish_and_read(self) -> None:
        sm = SharedMemory()
        block = self._make_block("data")
        sm.publish("data", block)
        result = sm.read("data")
        assert result is not None
        assert result.name == "data"

    def test_read_not_found(self) -> None:
        sm = SharedMemory()
        assert sm.read("nonexistent") is None

    def test_list_blocks(self) -> None:
        sm = SharedMemory()
        sm.publish("b", self._make_block("b"))
        sm.publish("a", self._make_block("a"))
        assert sm.list_blocks() == ["a", "b"]

    def test_remove(self) -> None:
        sm = SharedMemory()
        sm.publish("data", self._make_block("data"))
        assert sm.remove("data") is True
        assert sm.read("data") is None
        assert sm.remove("data") is False

    def test_get_all(self) -> None:
        sm = SharedMemory()
        sm.publish("a", self._make_block("a"))
        sm.publish("b", self._make_block("b"))
        all_blocks = sm.get_all()
        assert len(all_blocks) == 2

    def test_block_count(self) -> None:
        sm = SharedMemory()
        assert sm.block_count == 0
        sm.publish("a", self._make_block("a"))
        assert sm.block_count == 1


class TestHandoffPackage:
    """Tests for HandoffPackage."""

    def test_create_basic(self) -> None:
        hp = HandoffPackage(
            source_agent="agent_a",
            target_agent="agent_b",
        )
        assert hp.source_agent == "agent_a"
        assert hp.target_agent == "agent_b"
        assert hp.blocks == []
        assert hp.scratchpad is None
        assert hp.metadata == {}

    def test_create_with_blocks(self) -> None:
        block = ContextBlock(
            type=BlockType.USER_CONTEXT,
            content="test",
            name="test",
        )
        hp = HandoffPackage(
            source_agent="a",
            target_agent="b",
            blocks=[block],
        )
        assert hp.block_count == 1

    def test_create_with_scratchpad(self) -> None:
        pad = Scratchpad()
        pad.write("plan", "Do X")
        hp = HandoffPackage(
            source_agent="a",
            target_agent="b",
            scratchpad=pad,
        )
        assert hp.scratchpad is not None
        assert hp.scratchpad.read("plan") == "Do X"


class TestContextScope:
    """Tests for the ContextScope."""

    def test_create_basic(self) -> None:
        window = ContextWindow(max_tokens=10000)
        scope = ContextScope("agent_a", window)
        assert scope.agent_name == "agent_a"
        assert scope.window is window
        assert scope.scratchpad is not None
        assert scope.shared_memory is None
        assert scope.timeline is None

    def test_create_with_shared_memory(self) -> None:
        window = ContextWindow(max_tokens=10000)
        sm = SharedMemory()
        scope = ContextScope("agent_a", window, shared_memory=sm)
        assert scope.shared_memory is sm

    def test_create_with_timeline(self) -> None:
        window = ContextWindow(max_tokens=10000)
        scope = ContextScope("agent_a", window, track_history=True)
        assert scope.timeline is not None

    def test_record_turn(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="test",
                name="block_a",
            )
        )
        scope = ContextScope("agent_a", window, track_history=True)
        scope.record_turn()
        assert scope.timeline is not None
        assert scope.timeline.current_turn == 1

    def test_record_turn_no_timeline(self) -> None:
        window = ContextWindow(max_tokens=10000)
        scope = ContextScope("agent_a", window)
        # Should not raise even without timeline
        scope.record_turn()

    def test_record_turn_with_events(self) -> None:
        window = ContextWindow(max_tokens=10000)
        scope = ContextScope("agent_a", window, track_history=True)
        scope.record_turn(events=["block_added"])
        snap = scope.timeline.snapshot_at(1)
        assert snap is not None
        assert "block_added" in snap.events

    def test_import_shared(self) -> None:
        sm = SharedMemory()
        sm.publish(
            "shared_data",
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="Shared content",
                name="shared_data",
            ),
        )

        window = ContextWindow(max_tokens=10000)
        scope = ContextScope("agent_a", window, shared_memory=sm)
        imported = scope.import_shared()
        assert imported == 1
        assert len(window.blocks) == 1

    def test_import_shared_specific_blocks(self) -> None:
        sm = SharedMemory()
        sm.publish(
            "data_a",
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="A",
                name="data_a",
            ),
        )
        sm.publish(
            "data_b",
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="B",
                name="data_b",
            ),
        )

        window = ContextWindow(max_tokens=10000)
        scope = ContextScope("agent_a", window, shared_memory=sm)
        imported = scope.import_shared(block_names=["data_a"])
        assert imported == 1

    def test_import_shared_no_shared_memory(self) -> None:
        window = ContextWindow(max_tokens=10000)
        scope = ContextScope("agent_a", window)
        imported = scope.import_shared()
        assert imported == 0

    def test_handoff(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="Important data",
                name="data",
            )
        )
        scope = ContextScope("agent_a", window)
        scope.scratchpad.write("plan", "Step 1")

        package = scope.handoff("agent_b")
        assert package.source_agent == "agent_a"
        assert package.target_agent == "agent_b"
        assert package.block_count == 1
        assert package.scratchpad is not None

    def test_handoff_specific_blocks(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="Data A",
                name="block_a",
            )
        )
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="Data B",
                name="block_b",
            )
        )
        scope = ContextScope("agent_a", window)
        package = scope.handoff("agent_b", block_names=["block_a"])
        assert package.block_count == 1

    def test_handoff_without_scratchpad(self) -> None:
        window = ContextWindow(max_tokens=10000)
        scope = ContextScope("agent_a", window)
        package = scope.handoff("agent_b", include_scratchpad=False)
        assert package.scratchpad is None

    def test_handoff_with_metadata(self) -> None:
        window = ContextWindow(max_tokens=10000)
        scope = ContextScope("agent_a", window)
        package = scope.handoff(
            "agent_b",
            metadata={"reason": "task_complete"},
        )
        assert package.metadata == {"reason": "task_complete"}

    def test_receive_handoff(self) -> None:
        # Agent A creates handoff
        window_a = ContextWindow(max_tokens=10000)
        window_a.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="Transfer data",
                name="transfer",
            )
        )
        scope_a = ContextScope("agent_a", window_a)
        scope_a.scratchpad.write("note", "Important info")
        package = scope_a.handoff("agent_b")

        # Agent B receives it
        window_b = ContextWindow(max_tokens=10000)
        scope_b = ContextScope("agent_b", window_b)
        added = scope_b.receive_handoff(package)
        assert added == 1
        assert len(window_b.blocks) == 1
        # Scratchpad notes should be imported with prefix
        assert scope_b.scratchpad.read("from_agent_a_note") == "Important info"

    def test_receive_handoff_budget_exceeded(self) -> None:
        # Create a window with very small budget
        window_b = ContextWindow(max_tokens=1)
        scope_b = ContextScope("agent_b", window_b)

        block = ContextBlock(
            type=BlockType.USER_CONTEXT,
            content="This content is way too long to fit " * 100,
            name="big_block",
        )
        package = HandoffPackage(
            source_agent="agent_a",
            target_agent="agent_b",
            blocks=[block],
        )
        added = scope_b.receive_handoff(package)
        assert added == 0  # Should not crash, just skip

    def test_full_multi_agent_workflow(self) -> None:
        """Integration test for multi-agent context flow."""
        shared = SharedMemory()

        # Agent A sets up context
        window_a = ContextWindow(max_tokens=10000)
        scope_a = ContextScope(
            "researcher",
            window_a,
            shared_memory=shared,
            track_history=True,
        )

        # Agent A adds blocks and records turns
        window_a.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="Research findings about AI",
                name="findings",
            )
        )
        scope_a.scratchpad.write("status", "research_complete")
        scope_a.record_turn()

        # Agent A publishes to shared memory
        shared.publish(
            "research_results",
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="AI research summary",
                name="research_results",
            ),
        )

        # Agent B imports from shared memory
        window_b = ContextWindow(max_tokens=10000)
        scope_b = ContextScope(
            "writer",
            window_b,
            shared_memory=shared,
            track_history=True,
        )
        imported = scope_b.import_shared()
        assert imported == 1
        scope_b.record_turn()

        # Agent A hands off to Agent B directly
        package = scope_a.handoff("writer")
        added = scope_b.receive_handoff(package)
        assert added >= 1
        scope_b.record_turn()

        # Verify timelines
        assert scope_a.timeline.current_turn == 1
        assert scope_b.timeline.current_turn == 2
