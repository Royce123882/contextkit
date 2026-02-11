"""Tests for context inspection, diff, and dump."""

from __future__ import annotations

import json
import os
import tempfile

from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.events import clear_handlers
from contextkit.observe.diff import diff_windows
from contextkit.observe.inspect import dump_window, inspect_window
from contextkit.observe.provenance import Mutation, Origin
from contextkit.observe.renderers import format_text_table, truncate_content


class TestTextRenderer:
    """Tests for the text table renderer."""

    def test_format_basic_table(self) -> None:
        headers = ["Name", "Value"]
        rows = [["foo", "1"], ["bar", "2"]]
        result = format_text_table(headers, rows)
        assert "Name" in result
        assert "foo" in result
        assert "bar" in result
        assert "|" in result

    def test_format_table_with_footer(self) -> None:
        headers = ["Name", "Count"]
        rows = [["a", "10"], ["b", "20"]]
        footer = ["Total", "30"]
        result = format_text_table(headers, rows, footer=footer)
        assert "Total" in result
        assert "30" in result

    def test_truncate_short_content(self) -> None:
        assert truncate_content("Hello", max_length=10) == "Hello"

    def test_truncate_long_content(self) -> None:
        result = truncate_content("A" * 50, max_length=10)
        assert len(result) == 10
        assert result.endswith("...")


class TestInspectWindow:
    """Tests for window inspection."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_summary_inspect_contains_all_blocks(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="System prompt",
                priority=100,
                name="sys",
            )
        )
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="User info",
                priority=80,
                name="user",
            )
        )

        output = inspect_window(window)
        assert "sys" in output
        assert "user" in output

    def test_summary_inspect_shows_token_counts(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="Hello world",
                name="sys",
            )
        )

        output = inspect_window(window)
        # Should have numbers
        assert any(c.isdigit() for c in output)

    def test_summary_inspect_shows_origin(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        origin = Origin(source="prompt", details={"template": "v1"})
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="sys",
                origin=origin,
            )
        )

        output = inspect_window(window)
        assert "prompt" in output

    def test_summary_inspect_shows_cost_with_model(self) -> None:
        window = ContextWindow(model="claude-sonnet-4-5-20250929")
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="sys",
            )
        )

        output = inspect_window(window)
        assert "$" in output
        assert "Cost" in output

    def test_summary_inspect_no_cost_without_model(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="sys",
            )
        )

        output = inspect_window(window)
        assert "Cost" not in output

    def test_summary_shows_budget_remaining(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
            )
        )
        output = inspect_window(window)
        assert "Budget remaining" in output

    def test_drill_down_block_not_found(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        output = inspect_window(window, block_name="missing")
        assert "No block with name" in output

    def test_drill_down_system_prompt(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="You are a helpful assistant.",
                name="sys",
                origin=Origin(source="prompt"),
            )
        )

        output = inspect_window(window, block_name="sys")
        assert "sys" in output
        assert "system_prompt" in output
        assert "Content:" in output

    def test_drill_down_short_term_memory(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SHORT_TERM_MEMORY,
                content=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                name="history",
            )
        )

        output = inspect_window(window, block_name="history")
        assert "Messages:" in output
        assert "user" in output
        assert "assistant" in output

    def test_drill_down_rag_blocks(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.RAG,
                content=[
                    {
                        "content": "JWT tokens are issued...",
                        "source": "docs/auth.md",
                        "relevance": 0.92,
                    },
                ],
                name="rag_chunks",
            )
        )

        output = inspect_window(window, block_name="rag_chunks")
        assert "Chunks:" in output
        assert "docs/auth.md" in output

    def test_drill_down_tool_definitions(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.TOOL_DEFINITIONS,
                content=[
                    {
                        "name": "search_docs",
                        "description": "Search documentation",
                        "parameters": {
                            "properties": {"query": {"type": "string"}}
                        },
                    },
                ],
                name="tools",
            )
        )

        output = inspect_window(window, block_name="tools")
        assert "Tools:" in output
        assert "search_docs" in output

    def test_drill_down_shows_mutations(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        mutation = Mutation(
            step="TrimStep",
            action="trimmed",
            detail="Removed old messages",
            tokens_before=8000,
            tokens_after=4000,
        )
        window.add(
            ContextBlock(
                type=BlockType.SHORT_TERM_MEMORY,
                content="Trimmed content",
                name="history",
                mutations=[mutation],
            )
        )

        output = inspect_window(window, block_name="history")
        assert "Mutations:" in output
        assert "TrimStep" in output

    def test_window_inspect_method(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="sys",
            )
        )
        output = window.inspect()
        assert "sys" in output

    def test_window_inspect_block_method(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test content",
                name="sys",
            )
        )
        output = window.inspect("sys")
        assert "system_prompt" in output


class TestDiffWindows:
    """Tests for window diff comparison."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_diff_added_blocks(self) -> None:
        window_a = ContextWindow(max_tokens=100_000)
        window_b = ContextWindow(max_tokens=100_000)
        window_b.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="new_block",
            )
        )

        output = diff_windows(window_a, window_b)
        assert "Added:" in output
        assert "new_block" in output

    def test_diff_removed_blocks(self) -> None:
        window_a = ContextWindow(max_tokens=100_000)
        window_a.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="old_block",
            )
        )
        window_b = ContextWindow(max_tokens=100_000)

        output = diff_windows(window_a, window_b)
        assert "Removed:" in output
        assert "old_block" in output

    def test_diff_changed_blocks(self) -> None:
        window_a = ContextWindow(max_tokens=100_000)
        window_a.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="short",
                name="block",
            )
        )
        window_b = ContextWindow(max_tokens=100_000)
        window_b.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="this is much longer content for testing",
                name="block",
            )
        )

        output = diff_windows(window_a, window_b)
        assert "Changed:" in output
        assert "block" in output

    def test_diff_shows_token_delta(self) -> None:
        window_a = ContextWindow(max_tokens=100_000)
        window_b = ContextWindow(max_tokens=100_000)
        window_b.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="new",
            )
        )

        output = diff_windows(window_a, window_b)
        assert "Tokens:" in output

    def test_diff_shows_cost_delta_with_model(self) -> None:
        window_a = ContextWindow(model="claude-sonnet-4-5-20250929")
        window_b = ContextWindow(model="claude-sonnet-4-5-20250929")
        window_b.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="new",
            )
        )

        output = diff_windows(window_a, window_b)
        assert "Cost:" in output
        assert "$" in output

    def test_diff_identical_windows(self) -> None:
        window_a = ContextWindow(max_tokens=100_000)
        window_a.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="same",
                name="block",
            )
        )
        window_b = ContextWindow(max_tokens=100_000)
        window_b.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="same",
                name="block",
            )
        )

        output = diff_windows(window_a, window_b)
        assert "Added:" not in output
        assert "Removed:" not in output
        assert "Changed:" not in output

    def test_window_diff_method(self) -> None:
        window_a = ContextWindow(max_tokens=100_000)
        window_b = ContextWindow(max_tokens=100_000)
        window_b.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
                name="new",
            )
        )
        output = window_a.diff(window_b)
        assert "new" in output


class TestDumpWindow:
    """Tests for window JSON dump."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_dump_creates_valid_json(self) -> None:
        window = ContextWindow(model="gpt-4o")
        origin = Origin(source="prompt", details={"template": "v1"})
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="System prompt",
                name="sys",
                priority=100,
                origin=origin,
            )
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            dump_window(window, path)
            with open(path) as f:
                data = json.load(f)
            assert data["model"] == "gpt-4o"
            assert len(data["blocks"]) == 1
            assert data["blocks"][0]["name"] == "sys"
            assert data["blocks"][0]["origin"]["source"] == "prompt"
        finally:
            os.unlink(path)

    def test_dump_includes_assembly_report(self) -> None:
        from contextkit.assembler import ContextAssembler

        window = ContextWindow(model="gpt-4o")
        assembler = ContextAssembler(window)
        assembler.assemble(
            [
                ContextBlock(
                    type=BlockType.SYSTEM_PROMPT,
                    content="test",
                    name="sys",
                )
            ]
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            dump_window(window, path)
            with open(path) as f:
                data = json.load(f)
            assert data["assembly_report"] is not None
        finally:
            os.unlink(path)

    def test_window_dump_method(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
            )
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            window.dump(path)
            with open(path) as f:
                data = json.load(f)
            assert "blocks" in data
        finally:
            os.unlink(path)
