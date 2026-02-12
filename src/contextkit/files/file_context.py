"""Smart file handling for context injection.

Includes path traversal protection to prevent loading files
outside the configured base directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

from contextkit.constants import DEFAULT_ENCODING, DEFAULT_TOP_K, PRIORITY_FILE_CONTEXT
from contextkit.core import BlockType, ContextBlock
from contextkit.exceptions import PathTraversalError
from contextkit.files.file_reference import FileReference
from contextkit.observe.provenance import Origin
from contextkit.utils.token_counting import count as count_tokens

logger = logging.getLogger("contextkit")


class FileContext:
    """Smart file handling for context injection.

    Provides directory indexing (metadata-only), lazy loading
    with chunking, automatic Origin population, and path traversal
    protection.

    Args:
        base_path: Root directory to scan for files.
        extensions: File extensions to include (e.g. [".py", ".md"]).
            If None, includes all files.
    """

    def __init__(
        self,
        base_path: str,
        extensions: List[str] | None = None,
    ) -> None:
        self._base_path = Path(base_path).resolve()
        self._extensions = extensions
        self._index: List[FileReference] = []

    @property
    def base_path(self) -> Path:
        """The root directory."""
        return self._base_path

    @property
    def index(self) -> List[FileReference]:
        """The indexed file references."""
        return list(self._index)

    def scan(self) -> List[FileReference]:
        """Scan the base directory and index all matching files.

        Stores metadata (path, size, extension) without loading
        file content. Call load() to actually read files.
        Uses ``followlinks=False`` to prevent symlink traversal attacks.

        Returns:
            List of FileReference objects found.
        """
        logger.info("Scanning '%s' for files", self._base_path)
        self._index = []

        if not self._base_path.exists():
            return []

        for root, _dirs, files in os.walk(self._base_path, followlinks=False):
            for filename in sorted(files):
                filepath = Path(root) / filename
                ext = filepath.suffix.lower()

                if self._extensions and ext not in self._extensions:
                    continue

                ref = FileReference(
                    path=str(filepath),
                    size_bytes=filepath.stat().st_size,
                    extension=ext,
                    name=filename,
                )
                self._index.append(ref)

        logger.info("Found %d files", len(self._index))
        return list(self._index)

    async def ascan(self) -> List[FileReference]:
        """Async version of :meth:`scan`."""
        return self.scan()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[FileReference]:
        """Search indexed files by name or path substring.

        Args:
            query: Search string to match against file paths/names.
            top_k: Maximum results to return.

        Returns:
            Matching FileReferences sorted by relevance.
        """
        query_lower = query.lower()
        scored: List[Tuple[float, FileReference]] = []

        for ref in self._index:
            name_lower = ref.name.lower()
            path_lower = ref.path.lower()
            score = 0.0

            if query_lower in name_lower:
                score += 2.0
            if query_lower in path_lower:
                score += 1.0
            # Bonus for exact name match
            name_no_ext = Path(ref.name).stem.lower()
            if query_lower == name_no_ext:
                score += 3.0

            if score > 0:
                scored.append((score, ref))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ref for _, ref in scored[:top_k]]

    def load(
        self,
        refs: List[FileReference],
        max_tokens: int | None = None,
        encoding: str = DEFAULT_ENCODING,
        priority: int = PRIORITY_FILE_CONTEXT,
    ) -> List[ContextBlock]:
        """Load files into ContextBlocks with path traversal protection.

        Reads file content and creates blocks with Origin
        auto-populated. Respects token budget by stopping when
        budget is exhausted.

        Args:
            refs: File references to load.
            max_tokens: Maximum total tokens across all blocks.
            encoding: Tiktoken encoding for token counting.
            priority: Block priority for created blocks.

        Returns:
            List of ContextBlocks with file content.

        Raises:
            PathTraversalError: If any file path is outside the base directory.
        """
        logger.info(
            "Loading %d files (budget: %s tokens)",
            len(refs),
            max_tokens or "unlimited",
        )
        blocks: List[ContextBlock] = []
        total_tokens = 0

        for chunk_index, ref in enumerate(refs):
            validated_path = self._validate_path(ref.path)

            try:
                content = validated_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.debug("Skipped file %s: %s", ref.path, exc)
                continue

            token_count = count_tokens(content, encoding)

            if max_tokens is not None:
                if total_tokens + token_count > max_tokens:
                    remaining = max_tokens - total_tokens
                    if remaining > 0:
                        content = _chunk_to_budget(content, remaining, encoding)
                        token_count = count_tokens(content, encoding)
                    else:
                        break

            origin = Origin(
                source="file",
                details={
                    "file_path": ref.path,
                    "chunk_index": chunk_index,
                    "extension": ref.extension,
                    "size_bytes": ref.size_bytes,
                },
            )

            block = ContextBlock(
                type=BlockType.FILES,
                content=content,
                priority=priority,
                name=f"file_{ref.name}",
                origin=origin,
            )
            blocks.append(block)
            total_tokens += token_count

            if max_tokens is not None and total_tokens >= max_tokens:
                break

        logger.info(
            "Loaded %d file blocks (%s tokens)", len(blocks), f"{total_tokens:,}"
        )
        return blocks

    async def aload(
        self,
        refs: List[FileReference],
        max_tokens: int | None = None,
        encoding: str = DEFAULT_ENCODING,
        priority: int = PRIORITY_FILE_CONTEXT,
    ) -> List[ContextBlock]:
        """Async version of :meth:`load`."""
        return self.load(
            refs, max_tokens=max_tokens, encoding=encoding, priority=priority
        )

    def load_single(
        self,
        path: str,
        priority: int = PRIORITY_FILE_CONTEXT,
    ) -> ContextBlock:
        """Load a single file into a ContextBlock with path validation.

        Args:
            path: File path to load.
            priority: Block priority.

        Returns:
            A ContextBlock with the file content.

        Raises:
            PathTraversalError: If the path is outside the base directory.
        """
        validated_path = self._validate_path(path)
        content = validated_path.read_text(encoding="utf-8")

        origin = Origin(
            source="file",
            details={
                "file_path": str(validated_path),
                "extension": validated_path.suffix.lower(),
                "size_bytes": validated_path.stat().st_size,
            },
        )

        return ContextBlock(
            type=BlockType.FILES,
            content=content,
            priority=priority,
            name=f"file_{validated_path.name}",
            origin=origin,
        )

    async def aload_single(
        self,
        path: str,
        priority: int = PRIORITY_FILE_CONTEXT,
    ) -> ContextBlock:
        """Async version of :meth:`load_single`."""
        return self.load_single(path, priority=priority)

    def _validate_path(self, path: str) -> Path:
        """Validate that a file path is within the allowed base directory.

        Resolves the path to its absolute canonical form and checks
        that it is relative to the base directory.

        Args:
            path: The file path to validate.

        Returns:
            The resolved Path object.

        Raises:
            PathTraversalError: If the path escapes the base directory.
        """
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self._base_path):
            raise PathTraversalError(
                attempted_path=str(path),
                allowed_directory=str(self._base_path),
            )
        return resolved


def _chunk_to_budget(
    content: str,
    max_tokens: int,
    encoding: str,
) -> str:
    """Truncate content to fit within a token budget.

    Uses binary search to find the longest prefix that fits.

    Args:
        content: The content string to chunk.
        max_tokens: Maximum allowed tokens.
        encoding: Tiktoken encoding name.

    Returns:
        The truncated content string.
    """
    if count_tokens(content, encoding) <= max_tokens:
        return content

    low, high = 0, len(content)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(content[:mid], encoding) <= max_tokens:
            low = mid
        else:
            high = mid - 1

    return content[:low]
