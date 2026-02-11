"""Context window inspection.

Multi-level inspection from summary table to per-block drill-down.
Supports both text output (terminals) and HTML (Jupyter notebooks).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from contextkit.observe.renderers import (
    format_text_table,
    truncate_content,
)

if TYPE_CHECKING:
    from contextkit.core import ContextWindow


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


def _inspect_summary(window: ContextWindow, format: str) -> str:
    """Render a summary table of all blocks in the window."""
    headers = ["Block", "Type", "Tokens", "Priority", "Budget %", "Origin"]

    if window.model_name:
        headers.insert(4, "Cost")

    rows: list[list[str]] = []
    for block in window.blocks:
        budget_pct = (
            f"{block.token_count / window.max_tokens * 100:.1f}%"
            if window.max_tokens > 0
            else "N/A"
        )
        origin_str = block.origin.summary() if block.origin else "-"

        row: list[str] = [
            block.display_name,
            block.type.value,
            f"{block.token_count:,}",
            str(block.priority),
        ]

        if window.model_name:
            cost = block.token_count * window._input_cost_per_mtok / 1_000_000
            row.append(f"${cost:.4f}")

        row.extend([budget_pct, origin_str])
        rows.append(row)

    # Footer with totals
    footer_row: list[str] = [
        "Total",
        "",
        f"{window.token_count:,}",
        "",
    ]
    if window.model_name:
        footer_row.append(f"${window.cost_estimate:.4f}")

    budget_pct_total = (
        f"{window.token_count / window.max_tokens * 100:.1f}%"
        if window.max_tokens > 0
        else "N/A"
    )
    footer_row.extend([budget_pct_total, ""])

    result = format_text_table(headers, rows, footer=footer_row)

    # Add budget remaining line
    result += (
        f"\nBudget remaining: {window.budget_remaining:,} tokens "
        f"({window.budget_remaining / window.max_tokens * 100:.1f}%)"
        if window.max_tokens > 0
        else ""
    )

    return result


def _inspect_block(window: ContextWindow, block_name: str, format: str) -> str:
    """Drill into a specific block for detailed inspection."""
    from contextkit.core import BlockType

    block = window.get_block(block_name)
    if block is None:
        return f"No block with name '{block_name}' found."

    lines: list[str] = [
        f"Block: {block.display_name}",
        f"Type: {block.type.value}",
        f"Tokens: {block.token_count:,}",
        f"Priority: {block.priority}",
    ]

    if block.origin:
        lines.append(f"Origin: {block.origin.summary()}")

    if block.mutations:
        lines.append("Mutations:")
        for mutation in block.mutations:
            lines.append(
                f"  [{mutation.step}] {mutation.action}: "
                f"{mutation.detail} "
                f"({mutation.tokens_before:,} -> "
                f"{mutation.tokens_after:,} tokens)"
            )

    # Type-specific drill-down
    if block.type == BlockType.SHORT_TERM_MEMORY and isinstance(block.content, list):
        lines.append("")
        lines.append("Messages:")
        headers = ["#", "Role", "Tokens", "Content"]
        rows = []
        for i, msg in enumerate(block.content, 1):
            content = msg.get("content", "")
            from contextkit._tokens import count

            msg_tokens = count(content) if isinstance(content, str) else 0
            rows.append(
                [
                    str(i),
                    msg.get("role", "unknown"),
                    str(msg_tokens),
                    _preview(content),
                ]
            )
        lines.append(format_text_table(headers, rows))

    elif block.type == BlockType.RAG and isinstance(block.content, list):
        lines.append("")
        lines.append("Chunks:")
        headers = ["#", "Source", "Tokens", "Relevance", "Content"]
        rows = []
        for i, chunk in enumerate(block.content, 1):
            content = chunk.get("content", "")
            from contextkit._tokens import count

            chunk_tokens = count(content) if isinstance(content, str) else 0
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
        lines.append(format_text_table(headers, rows))

    elif block.type == BlockType.TOOL_DEFINITIONS and isinstance(block.content, list):
        lines.append("")
        lines.append("Tools:")
        headers = ["#", "Name", "Description", "Params"]
        rows = []
        for i, tool in enumerate(block.content, 1):
            name = tool.get("name", "-")
            desc = tool.get("description", "-")
            params = tool.get("parameters", {})
            param_count = (
                len(params.get("properties", {})) if isinstance(params, dict) else 0
            )
            rows.append([str(i), name, truncate_content(desc), str(param_count)])
        lines.append(format_text_table(headers, rows))

    elif isinstance(block.content, str):
        lines.append("")
        lines.append("Content:")
        lines.append(truncate_content(block.content, max_length=200))

    return "\n".join(lines)


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
