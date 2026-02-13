"""Local filesystem compaction store implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from contextkit.constants import DEFAULT_COMPACTION_STORAGE_DIR


class LocalCompactionStore:
    """Filesystem-backed compaction store.

    Saves markdown files to a configurable directory.
    Default: ``.contextkit/compacted/`` relative to cwd.

    Args:
        base_dir: Directory for storing compaction artifacts.
    """

    def __init__(self, base_dir: str = DEFAULT_COMPACTION_STORAGE_DIR) -> None:
        self._base_dir = Path(base_dir)

    async def save(
        self,
        key: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Save content to a local markdown file.

        Args:
            key: Unique identifier (used as filename stem).
            content: The numbered markdown content.
            metadata: Ignored for local storage.

        Returns:
            Absolute path to the saved file.
        """
        os.makedirs(self._base_dir, exist_ok=True)
        path = self._base_dir / f"{key}.md"
        path.write_text(content, encoding="utf-8")
        return str(path.resolve())

    async def load(self, key: str) -> str | None:
        """Load content from a local markdown file.

        Args:
            key: The artifact identifier.

        Returns:
            File contents, or None if the file does not exist.
        """
        path = self._base_dir / f"{key}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    async def delete(self, key: str) -> bool:
        """Delete a local markdown file.

        Args:
            key: The artifact identifier.

        Returns:
            True if the file was deleted, False if it did not exist.
        """
        path = self._base_dir / f"{key}.md"
        if not path.exists():
            return False
        path.unlink()
        return True

    async def list_keys(self) -> List[str]:
        """List all artifact keys in the store directory.

        Returns:
            A list of keys (filename stems without ``.md`` extension).
        """
        if not self._base_dir.exists():
            return []
        return [p.stem for p in sorted(self._base_dir.glob("*.md"))]
