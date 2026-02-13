"""Tests for the event system."""

from __future__ import annotations

from contextkit.observe.event_models import (
    BlockEventData,
    BudgetEventData,
    ContextEvent,
    EventData,
    PipelineEventData,
)
from contextkit.observe.events import (
    clear_handlers,
    emit,
    on,
    register_handler,
)


class TestContextEvent:
    """Tests for the ContextEvent enum."""

    def test_all_events_exist(self) -> None:
        expected = [
            "BLOCK_ADDED",
            "BLOCK_REMOVED",
            "BLOCK_MUTATED",
            "BUDGET_WARNING",
            "BUDGET_EXCEEDED",
            "ASSEMBLY_COMPLETE",
            "PIPELINE_STEP",
            "PIPELINE_COMPLETE",
            "WINDOW_RENDERED",
            "CONTEXT_INSUFFICIENT",
        ]
        actual = [e.name for e in ContextEvent]
        assert sorted(actual) == sorted(expected)

    def test_event_values_are_strings(self) -> None:
        for event in ContextEvent:
            assert isinstance(event.value, str)


class TestEventData:
    """Tests for event data models."""

    def test_create_base_event_data(self) -> None:
        data = EventData(event=ContextEvent.BLOCK_ADDED)
        assert data.event == ContextEvent.BLOCK_ADDED
        assert data.timestamp is not None
        assert data.details == {}

    def test_create_block_event_data(self) -> None:
        data = BlockEventData(
            event=ContextEvent.BLOCK_ADDED,
            block_name="sys",
            block_type="system_prompt",
            token_count=100,
        )
        assert data.block_name == "sys"
        assert data.token_count == 100

    def test_create_budget_event_data(self) -> None:
        data = BudgetEventData(
            event=ContextEvent.BUDGET_WARNING,
            percent=78.0,
            threshold=75.0,
            tokens_used=156_000,
            tokens_max=200_000,
        )
        assert data.percent == 78.0
        assert data.threshold == 75.0

    def test_create_pipeline_event_data(self) -> None:
        data = PipelineEventData(
            event=ContextEvent.PIPELINE_STEP,
            step_name="TrimStep",
            tokens_before=10_000,
            tokens_after=8_000,
        )
        assert data.step_name == "TrimStep"

    def test_event_data_with_details(self) -> None:
        data = EventData(
            event=ContextEvent.ASSEMBLY_COMPLETE,
            details={"count": 5, "total_tokens": 1000},
        )
        assert data.details["count"] == 5


class TestEventRegistration:
    """Tests for event handler registration and emission."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_register_with_decorator(self) -> None:
        received: list[EventData] = []

        @on(ContextEvent.BLOCK_ADDED)
        def handler(event: EventData) -> None:
            received.append(event)

        emit(
            BlockEventData(
                event=ContextEvent.BLOCK_ADDED,
                block_name="test",
            )
        )
        assert len(received) == 1
        assert received[0].event == ContextEvent.BLOCK_ADDED

    def test_register_with_function(self) -> None:
        received: list[EventData] = []

        def handler(event: EventData) -> None:
            received.append(event)

        register_handler(ContextEvent.BLOCK_REMOVED, handler)
        emit(
            BlockEventData(
                event=ContextEvent.BLOCK_REMOVED,
                block_name="test",
            )
        )
        assert len(received) == 1

    def test_multiple_handlers_for_same_event(self) -> None:
        received_a: list[EventData] = []
        received_b: list[EventData] = []

        @on(ContextEvent.BLOCK_ADDED)
        def handler_a(event: EventData) -> None:
            received_a.append(event)

        @on(ContextEvent.BLOCK_ADDED)
        def handler_b(event: EventData) -> None:
            received_b.append(event)

        emit(
            BlockEventData(
                event=ContextEvent.BLOCK_ADDED,
                block_name="test",
            )
        )
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_handler_receives_correct_data(self) -> None:
        received: list[BlockEventData] = []

        @on(ContextEvent.BLOCK_ADDED)
        def handler(event: BlockEventData) -> None:
            received.append(event)

        emit(
            BlockEventData(
                event=ContextEvent.BLOCK_ADDED,
                block_name="my_block",
                block_type="system_prompt",
                token_count=42,
            )
        )
        assert received[0].block_name == "my_block"
        assert received[0].token_count == 42

    def test_handler_only_fires_for_registered_event(self) -> None:
        received: list[EventData] = []

        @on(ContextEvent.BLOCK_ADDED)
        def handler(event: EventData) -> None:
            received.append(event)

        # Emit a different event
        emit(EventData(event=ContextEvent.BLOCK_REMOVED))
        assert len(received) == 0

    def test_clear_handlers_resets_all(self) -> None:
        received: list[EventData] = []

        @on(ContextEvent.BLOCK_ADDED)
        def handler(event: EventData) -> None:
            received.append(event)

        clear_handlers()

        emit(
            BlockEventData(
                event=ContextEvent.BLOCK_ADDED,
                block_name="test",
            )
        )
        assert len(received) == 0

    def test_emit_with_no_handlers(self) -> None:
        # Should not raise
        emit(EventData(event=ContextEvent.ASSEMBLY_COMPLETE))

    def test_budget_event_data(self) -> None:
        received: list[BudgetEventData] = []

        @on(ContextEvent.BUDGET_WARNING)
        def handler(event: BudgetEventData) -> None:
            received.append(event)

        emit(
            BudgetEventData(
                event=ContextEvent.BUDGET_WARNING,
                percent=90.0,
                threshold=90.0,
                tokens_used=180_000,
                tokens_max=200_000,
            )
        )
        assert received[0].percent == 90.0
        assert received[0].tokens_used == 180_000

    def test_decorator_preserves_function(self) -> None:
        @on(ContextEvent.BLOCK_ADDED)
        def my_handler(event: EventData) -> str:
            return "test"

        # The decorator should return the original function
        assert my_handler(EventData(event=ContextEvent.BLOCK_ADDED)) == "test"
