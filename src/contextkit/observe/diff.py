"""Context window diff utility.

Compares two ContextWindows and reports blocks added, removed,
and changed between them, with token and cost deltas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextkit.core import ContextWindow


def diff_windows(
    window_a: ContextWindow,
    window_b: ContextWindow,
    format: str = "text",  # noqa: A002
) -> str:
    """Compare two context windows and show differences.

    Args:
        window_a: The "before" window.
        window_b: The "after" window.
        format: Output format ("text" or "html").

    Returns:
        A formatted diff showing added, removed, and changed blocks.
    """
    blocks_a = {b.display_name: b for b in window_a.blocks}
    blocks_b = {b.display_name: b for b in window_b.blocks}

    names_a = set(blocks_a.keys())
    names_b = set(blocks_b.keys())

    added = names_b - names_a
    removed = names_a - names_b
    common = names_a & names_b

    changed: list[tuple[str, int, int]] = []
    unchanged: list[str] = []

    for name in common:
        tokens_a = blocks_a[name].token_count
        tokens_b = blocks_b[name].token_count
        if tokens_a != tokens_b:
            changed.append((name, tokens_a, tokens_b))
        else:
            unchanged.append(name)

    lines: list[str] = []
    lines.append(
        f"Context diff: {len(window_a.blocks)} blocks -> "
        f"{len(window_b.blocks)} blocks"
    )
    lines.append("")

    if added:
        lines.append("Added:")
        for name in sorted(added):
            block = blocks_b[name]
            lines.append(
                f"  + {name} ({block.token_count:,} tokens, "
                f"priority {block.priority})"
            )
        lines.append("")

    if removed:
        lines.append("Removed:")
        for name in sorted(removed):
            block = blocks_a[name]
            lines.append(
                f"  - {name} ({block.token_count:,} tokens, "
                f"priority {block.priority})"
            )
        lines.append("")

    if changed:
        lines.append("Changed:")
        for name, tok_a, tok_b in sorted(changed):
            delta = tok_b - tok_a
            sign = "+" if delta > 0 else ""
            lines.append(
                f"  ~ {name}: {tok_a:,} -> {tok_b:,} tokens "
                f"({sign}{delta:,})"
            )
        lines.append("")

    # Summary
    token_delta = window_b.token_count - window_a.token_count
    cost_delta = window_b.cost_estimate - window_a.cost_estimate
    sign_tok = "+" if token_delta > 0 else ""
    sign_cost = "+" if cost_delta > 0 else ""

    lines.append("Summary:")
    lines.append(
        f"  Tokens: {window_a.token_count:,} -> "
        f"{window_b.token_count:,} ({sign_tok}{token_delta:,})"
    )
    if window_a.model_name or window_b.model_name:
        lines.append(
            f"  Cost: ${window_a.cost_estimate:.4f} -> "
            f"${window_b.cost_estimate:.4f} ({sign_cost}${cost_delta:.4f})"
        )

    return "\n".join(lines)
