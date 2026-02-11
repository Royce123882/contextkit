"""Prompt version data model."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PromptVersion(BaseModel):
    """A versioned prompt template.

    Attributes:
        name: Template name.
        version: Version string (e.g. "1.0", "2.1").
        template: The template string with {{variable}} placeholders.
        created_at: When this version was created.
        description: Optional description of this version.
    """

    name: str
    version: str
    template: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
