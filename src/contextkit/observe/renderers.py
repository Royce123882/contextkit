"""Output renderers for inspection, diff, and timeline.

Provides text and HTML formatters for tables and reports.
Text is the default for terminal environments; HTML is used
in Jupyter notebooks.
"""

from __future__ import annotations

from typing import List


def detect_environment() -> str:
    """Detect whether we're running in Jupyter or a terminal.

    Returns:
        "html" if in Jupyter/IPython, "text" otherwise.
    """
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ in (
            "ZMQInteractiveShell",
            "TerminalInteractiveShell",
        ):
            return "html"
    except (ImportError, NameError):
        pass
    return "text"


def format_text_table(
    headers: List[str],
    rows: List[List[str]],
    footer: List[str] | None = None,
) -> str:
    """Render a simple ASCII table.

    Args:
        headers: Column header strings.
        rows: List of rows, each a list of cell strings.
        footer: Optional footer row.

    Returns:
        A formatted table string.
    """
    # Calculate column widths
    all_rows = [headers, *rows]
    if footer:
        all_rows.append(footer)

    col_widths = [
        max(len(str(row[i])) for row in all_rows if i < len(row))
        for i in range(len(headers))
    ]

    def format_row(cells: List[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            width = col_widths[i] if i < len(col_widths) else len(cell)
            parts.append(str(cell).ljust(width))
        return "| " + " | ".join(parts) + " |"

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [separator, format_row(headers), separator]
    for row in rows:
        lines.append(format_row(row))
    lines.append(separator)

    if footer:
        lines.append(format_row(footer))
        lines.append(separator)

    return "\n".join(lines)


def truncate_content(content: str, max_length: int = 40) -> str:
    """Truncate content for preview display.

    Args:
        content: The string to truncate.
        max_length: Maximum length before truncation.

    Returns:
        The truncated string with ellipsis if needed.
    """
    if len(content) <= max_length:
        return content
    return content[: max_length - 3] + "..."
