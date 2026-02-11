"""File reference data model."""

from __future__ import annotations

from pydantic import BaseModel


class FileReference(BaseModel):
    """A lightweight pointer to a file without loading its content.

    Attributes:
        path: Absolute or relative file path.
        size_bytes: File size in bytes.
        extension: File extension (e.g. ".py", ".md").
        name: File name without path.
    """

    path: str
    size_bytes: int = 0
    extension: str = ""
    name: str = ""
