"""Prompt template management.

Provides versioned prompt templates with variable interpolation,
prompt composition (base + overrides), and automatic Origin
population for traceability.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from contextkit.core import BlockType, ContextBlock
from contextkit.observe.provenance import Origin


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


class PromptManager:
    """Manages versioned prompt templates.

    Templates use simple {{variable}} interpolation. Supports
    versioning, composition (base + overrides), and automatic
    Origin population on rendered blocks.

    Example:
        pm = PromptManager()
        pm.register("assistant_v1", "You are {{role}}. Rules: {{rules}}")
        block = pm.render("assistant_v1", role="analyst", rules="Be concise")
    """

    def __init__(self) -> None:
        self._templates: dict[str, list[PromptVersion]] = {}

    def register(
        self,
        name: str,
        template: str,
        version: str | None = None,
        description: str = "",
    ) -> PromptVersion:
        """Register a prompt template.

        If no version is specified, auto-increments from the last
        version. First version defaults to "1".

        Args:
            name: Template name (unique identifier).
            template: Template string with {{variable}} placeholders.
            version: Optional version string.
            description: Optional description.

        Returns:
            The created PromptVersion.
        """
        if name not in self._templates:
            self._templates[name] = []

        versions = self._templates[name]
        if version is None:
            version = str(len(versions) + 1)

        prompt_ver = PromptVersion(
            name=name,
            version=version,
            template=template,
            description=description,
        )
        versions.append(prompt_ver)
        return prompt_ver

    def get_template(
        self,
        name: str,
        version: str | None = None,
    ) -> PromptVersion:
        """Get a template by name and optional version.

        Args:
            name: Template name.
            version: Specific version. If None, returns the latest.

        Returns:
            The matching PromptVersion.

        Raises:
            KeyError: If template name or version not found.
        """
        if name not in self._templates:
            raise KeyError(f"No template registered with name '{name}'")

        versions = self._templates[name]
        if not versions:
            raise KeyError(f"No versions for template '{name}'")

        if version is None:
            return versions[-1]

        for prompt_version in versions:
            if prompt_version.version == version:
                return prompt_version

        raise KeyError(f"Version '{version}' not found for template '{name}'")

    def render(
        self,
        name: str,
        version: str | None = None,
        priority: int = 100,
        **variables: Any,
    ) -> ContextBlock:
        """Render a template into a ContextBlock.

        Auto-populates Origin with template name, version, and
        variables used.

        Args:
            name: Template name.
            version: Specific version (None = latest).
            priority: Block priority.
            **variables: Template variables.

        Returns:
            A ContextBlock with rendered content and Origin.
        """
        resolved_template = self.get_template(name, version)
        content = _interpolate(resolved_template.template, variables)

        origin = Origin(
            source="prompt",
            details={
                "template": resolved_template.name,
                "version": resolved_template.version,
                "variables": list(variables.keys()),
            },
        )

        return ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content=content,
            priority=priority,
            name=name,
            origin=origin,
        )

    def compose(
        self,
        base_name: str,
        override_name: str,
        separator: str = "\n\n",
        priority: int = 100,
        **variables: Any,
    ) -> ContextBlock:
        """Compose two templates (base + override) into one block.

        The base template is rendered first, then the override is
        appended with the given separator.

        Args:
            base_name: Base template name.
            override_name: Override template name.
            separator: String between base and override content.
            priority: Block priority.
            **variables: Template variables for both templates.

        Returns:
            A ContextBlock with composed content.
        """
        base_template = self.get_template(base_name)
        override_template = self.get_template(override_name)

        base_content = _interpolate(base_template.template, variables)
        override_content = _interpolate(override_template.template, variables)
        combined = f"{base_content}{separator}{override_content}"

        origin = Origin(
            source="prompt",
            details={
                "template": f"{base_name}+{override_name}",
                "base_version": base_template.version,
                "override_version": override_template.version,
                "variables": list(variables.keys()),
            },
        )

        return ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content=combined,
            priority=priority,
            name=f"{base_name}+{override_name}",
            origin=origin,
        )

    def diff(self, name: str, version_a: str, version_b: str) -> str:
        """Compare two versions of a template.

        Args:
            name: Template name.
            version_a: First version.
            version_b: Second version.

        Returns:
            A string showing the differences.
        """
        template_a = self.get_template(name, version_a)
        template_b = self.get_template(name, version_b)

        lines: list[str] = []
        lines.append(f"Template '{name}' diff: v{version_a} -> v{version_b}")

        if template_a.template == template_b.template:
            lines.append("No changes in template content.")
        else:
            lines.append(f"--- v{version_a}")
            lines.append(f"+++ v{version_b}")

            lines_a = template_a.template.splitlines()
            lines_b = template_b.template.splitlines()

            for line in lines_a:
                if line not in lines_b:
                    lines.append(f"- {line}")
            for line in lines_b:
                if line not in lines_a:
                    lines.append(f"+ {line}")

        return "\n".join(lines)

    def list_templates(self) -> list[str]:
        """List all registered template names."""
        return sorted(self._templates.keys())

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a template.

        Args:
            name: Template name.

        Returns:
            List of version strings.
        """
        if name not in self._templates:
            raise KeyError(f"No template registered with name '{name}'")
        return [v.version for v in self._templates[name]]


_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def _interpolate(template: str, variables: dict[str, Any]) -> str:
    """Replace {{variable}} placeholders with values.

    Args:
        template: Template string with {{variable}} placeholders.
        variables: Variable name-to-value mapping.

    Returns:
        The interpolated string.
    """

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name in variables:
            return str(variables[var_name])
        return match.group(0)

    return _VARIABLE_PATTERN.sub(replacer, template)
