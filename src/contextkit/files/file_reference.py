"""File reference data model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileReference(BaseModel):
    """A lightweight pointer to a file without loading its content.

    Attributes:
        path: Absolute or relative file path.
        size_bytes: File size in bytes.
        extension: File extension (e.g. ".py", ".md").
        name: File name without path.
    """

    path: str = Field(description="Absolute or relative file path.")
    size_bytes: int = Field(default=0, description="File size in bytes.")
    extension: str = Field(default="", description="File extension (e.g. '.py', '.md').")
    name: str = Field(default="", description="File name without path.")
