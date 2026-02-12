"""Context window inspection.

Multi-level inspection from summary table to per-block drill-down.
Supports both text output (terminals) and HTML (Jupyter notebooks).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List

from contextkit.core.block import BlockType
from contextkit.observe.renderers import (
    format_text_table,
    truncate_content,
)
from contextkit.utils.token_counting import count as count_tokens

if TYPE_CHECKING:
    from contextkit.core import ContextBlock, ContextWindow


def _preview(content: object) -> str:
    """Return a truncated preview of content."""
    if isinstance(content, str):
        return truncate_content(content)
    return str(content)[:40]


def inspect_window(
    window: ContextWindow,
    block_name: str | None = None,
    format: str = "text",
) -> str:
    """Inspect a context window or drill into a specific block.

    Args:
        window: The ContextWindow to inspect.
        block_name: If provided, drill into this specific block.
        format: Output format ("text" or "html"). Defaults to "text".

    Returns:
        Formatted inspection output as a string.
    """
    if block_name is not None:
        return _inspect_block(window, block_name, format)
    return _inspect_summary(window, format)


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------


def _inspect_summary(window: ContextWindow, format: str) -> str:
    """Render a summary table of all blocks in the window."""
    has_model = bool(window.model_name)
    headers = _build_summary_headers(has_model)
    rows = _build_summary_rows(window, has_model)
    footer_row = _build_summary_footer(window, has_model)

    result = format_text_table(headers, rows, footer=footer_row)
    result += _format_budget_remaining(window)
    return result


def _build_summary_headers(has_model: bool) -> List[str]:
    """Build the header row for the summary table."""
    headers = ["Block", "Type", "Tokens", "Priority"]
    if has_model:
        headers.append("Cost")
    headers.extend(["Budget %", "Origin"])
    return headers


def _build_summary_rows(window: ContextWindow, has_model: bool) -> List[List[str]]:
    """Build data rows for the summary table."""
    rows: List[List[str]] = []
    for block in window.blocks:
        budget_percent = (
            f"{block.token_count / window.max_tokens * 100:.1f}%"
            if window.max_tokens > 0
            else "N/A"
        )
        origin_str = block.origin.summary() if block.origin else "-"

        row: List[str] = [
            block.display_name,
            block.type.value,
            f"{block.token_count:,}",
            str(block.priority),
        ]

        if has_model:
            cost = block.token_count * window.input_cost_per_mtok / 1_000_000
            row.append(f"${cost:.4f}")

        row.extend([budget_percent, origin_str])
        rows.append(row)

    return rows


def _build_summary_footer(window: ContextWindow, has_model: bool) -> List[str]:
    """Build the footer row with totals for the summary table."""
    footer_row: List[str] = [
        "Total",
        "",
        f"{window.token_count:,}",
        "",
    ]
    if has_model:
        footer_row.append(f"${window.cost_estimate:.4f}")

    budget_percent_total = (
        f"{window.token_count / window.max_tokens * 100:.1f}%"
        if window.max_tokens > 0
        else "N/A"
    )
    footer_row.extend([budget_percent_total, ""])
    return footer_row


def _format_budget_remaining(window: ContextWindow) -> str:
    """Format the budget-remaining line appended to the summary."""
    if window.max_tokens <= 0:
        return ""
    return (
        f"\nBudget remaining: {window.budget_remaining:,} tokens "
        f"({window.budget_remaining / window.max_tokens * 100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Block drill-down helpers
# ---------------------------------------------------------------------------


def _inspect_block(window: ContextWindow, block_name: str, format: str) -> str:
    """Drill into a specific block for detailed inspection."""
    block = window.get_block(block_name)
    if block is None:
        return f"No block with name '{block_name}' found."

    lines = _build_block_header(block)
    lines += _build_mutation_lines(block)
    lines += _build_type_specific_details(block)

    return "\n".join(lines)


def _build_block_header(block: ContextBlock) -> List[str]:
    """Build the header lines for a block inspection."""
    lines = [
        f"Block: {block.display_name}",
        f"Type: {block.type.value}",
        f"Tokens: {block.token_count:,}",
        f"Priority: {block.priority}",
    ]
    if block.origin:
        lines.append(f"Origin: {block.origin.summary()}")
    return lines


def _build_mutation_lines(block: ContextBlock) -> List[str]:
    """Build mutation history lines for a block."""
    if not block.mutations:
        return []

    lines: List[str] = ["Mutations:"]
    for mutation in block.mutations:
        lines.append(
            f"  [{mutation.step}] {mutation.action}: "
            f"{mutation.detail} "
            f"({mutation.tokens_before:,} -> "
            f"{mutation.tokens_after:,} tokens)"
        )
    return lines


def _build_type_specific_details(block: ContextBlock) -> List[str]:
    """Build type-specific drill-down details for a block."""
    if block.type == BlockType.SHORT_TERM_MEMORY and isinstance(block.content, list):
        return _inspect_messages_table(block.content)

    if block.type == BlockType.RAG and isinstance(block.content, list):
        return _inspect_chunks_table(block.content)

    if block.type == BlockType.TOOL_DEFINITIONS and isinstance(block.content, list):
        return _inspect_tools_table(block.content)

    if isinstance(block.content, str):
        return [
            "",
            "Content:",
            truncate_content(block.content, max_length=200),
        ]

    return []


def _inspect_messages_table(messages: List[Dict[str, Any]]) -> List[str]:
    """Build a table of conversation messages for SHORT_TERM_MEMORY."""
    headers = ["#", "Role", "Tokens", "Content"]
    rows = []
    for i, message in enumerate(messages, 1):
        content = message.get("content", "")
        message_tokens = count_tokens(content) if isinstance(content, str) else 0
        rows.append(
            [
                str(i),
                message.get("role", "unknown"),
                str(message_tokens),
                _preview(content),
            ]
        )
    return ["", "Messages:", format_text_table(headers, rows)]


def _inspect_chunks_table(chunks: List[Dict[str, Any]]) -> List[str]:
    """Build a table of RAG chunks for RAG blocks."""
    headers = ["#", "Source", "Tokens", "Relevance", "Content"]
    rows = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        chunk_tokens = count_tokens(content) if isinstance(content, str) else 0
        source = chunk.get("source", "-")
        relevance = chunk.get("relevance", "-")
        rows.append(
            [
                str(i),
                str(source),
                str(chunk_tokens),
                str(relevance),
                _preview(content),
            ]
        )
    return ["", "Chunks:", format_text_table(headers, rows)]


def _inspect_tools_table(tools: List[Dict[str, Any]]) -> List[str]:
    """Build a table of tool definitions for TOOL_DEFINITIONS blocks."""
    headers = ["#", "Name", "Description", "Params"]
    rows = []
    for i, tool in enumerate(tools, 1):
        tool_name = tool.get("name", "-")
        description = tool.get("description", "-")
        params = tool.get("parameters", {})
        param_count = (
            len(params.get("properties", {})) if isinstance(params, dict) else 0
        )
        rows.append([
            str(i), tool_name,
            truncate_content(description), str(param_count),
        ])
    return ["", "Tools:", format_text_table(headers, rows)]


def dump_window(window: ContextWindow, path: str) -> None:
    """Write a full JSON snapshot of the window to a file.

    Includes all blocks with content, provenance, mutations,
    metadata, assembly report, model info, and stats.

    Args:
        window: The ContextWindow to dump.
        path: The file path to write the JSON snapshot to.
    """
    data = window.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
