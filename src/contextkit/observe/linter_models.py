"""Data models for the context linter."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LintWarning(BaseModel):
    """A single warning from the context linter.

    Attributes:
        code: Machine-readable warning code (e.g. "no_system_prompt").
        message: Human-readable description with actionable advice.
        severity: Warning severity -- ``"info"``, ``"warning"``, or ``"error"``.
    """

    code: str = Field(description="Machine-readable warning code.")
    message: str = Field(description="Human-readable warning message.")
    severity: str = Field(
        default="warning",
        description="Severity level: 'info', 'warning', or 'error'.",
    )
