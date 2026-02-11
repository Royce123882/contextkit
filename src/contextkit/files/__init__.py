"""File context management for contextkit.

Provides smart file handling for context injection with lazy
loading, chunking strategies, and format-aware parsing.
"""

from contextkit.files.context import FileContext

__all__ = ["FileContext"]
