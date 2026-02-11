"""Explain mode for context blocks.

Provides plain-language explanations for why a block was included
in or excluded from a context window. Pulls from the assembly report
and mutation log to give a complete answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextkit.assembler import BlockDecision
    from contextkit.core import ContextBlock, ContextWindow


def explain_block(window: ContextWindow, block_name: str) -> str:
    """Explain why a block is included or excluded from a window.

    Checks the window's blocks first, then the assembly report's
    excluded list. Returns a plain-language multi-line explanation.

    Args:
        window: The ContextWindow to query.
        block_name: The name of the block to explain.

    Returns:
        A multi-line plain-language explanation string.
    """
    # Check if block is in the window (included)
    block = window.get_block(block_name)
    if block is not None:
        return _explain_included(window, block)

    # Check if block is in the assembly report's excluded list
    from contextkit.assembler import AssemblyReport

    report = window._assembly_report
    if isinstance(report, AssemblyReport):
        for decision in report.excluded:
            if decision.block_name == block_name:
                return _explain_excluded(window, decision)

    return (
        f'No block with name "{block_name}" found '
        f"in this window or its assembly report."
    )


def _explain_included(
    window: ContextWindow, block: ContextBlock
) -> str:
    """Explain an included block."""
    lines: list[str] = []
    budget_pct = (
        f"{block.token_count / window.max_tokens * 100:.1f}%"
        if window.max_tokens > 0
        else "N/A"
    )

    lines.append(
        f'Block "{block.display_name}" is INCLUDED '
        f"({block.token_count:,} tokens, "
        f"priority {block.priority})"
    )

    # Origin info
    if block.origin:
        lines.append(f"- Origin: {block.origin.summary()}")
        if block.origin.query:
            lines.append(f'- Query: "{block.origin.query}"')
        if block.origin.retriever:
            lines.append(
                f"- Retriever: {block.origin.retriever}"
            )
        if block.origin.relevance_score is not None:
            lines.append(
                f"- Relevance: "
                f"{block.origin.relevance_score:.2f}"
            )

    # Assembly info
    from contextkit.assembler import AssemblyReport

    report = window._assembly_report
    if isinstance(report, AssemblyReport):
        lines.append("- Added by: ContextAssembler")

    # Mutations
    if block.mutations:
        lines.append("- Mutations applied:")
        for m in block.mutations:
            lines.append(
                f"  [{m.step}] {m.action}: {m.detail} "
                f"({m.tokens_before:,} -> "
                f"{m.tokens_after:,} tokens)"
            )
    else:
        lines.append("- No mutations applied")

    lines.append(
        f"- Budget impact: {budget_pct} "
        f"of {window.max_tokens:,} token window"
    )

    return "\n".join(lines)


def _explain_excluded(
    window: ContextWindow,
    decision: BlockDecision,
) -> str:
    """Explain an excluded block."""
    lines: list[str] = []

    lines.append(
        f'Block "{decision.block_name}" is EXCLUDED'
    )
    lines.append(f"- Origin: {decision.origin_summary}")
    lines.append(f"- Reason: {decision.reason}")
    if window.max_tokens > 0:
        pct = decision.tokens / window.max_tokens * 100
        lines.append(
            f"- Would have used {decision.tokens:,} tokens "
            f"({pct:.1f}% of budget)"
        )
    else:
        lines.append(
            f"- Would have used {decision.tokens:,} tokens"
        )
    lines.append(f"- Priority: {decision.priority}")

    return "\n".join(lines)
