"""Prompt template management.

Supports ``{{variable}}`` interpolation with optional strict mode
that raises TemplateRenderError on missing variables.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from contextkit.constants import PRIORITY_SYSTEM_PROMPT
from contextkit.core import BlockType, ContextBlock
from contextkit.exceptions import TemplateRenderError
from contextkit.observe.provenance import Origin
from contextkit.prompts.prompt_version import PromptVersion

logger = logging.getLogger("contextkit")

_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


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
        self._templates: Dict[str, List[PromptVersion]] = {}

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
        logger.debug("Registered prompt template '%s' (v%s)", name, version)
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
        priority: int = PRIORITY_SYSTEM_PROMPT,
        strict: bool = False,
        **variables: Any,
    ) -> ContextBlock:
        """Render a template into a ContextBlock.

        Auto-populates Origin with template name, version, and
        variables used.

        Args:
            name: Template name.
            version: Specific version (None = latest).
            priority: Block priority.
            strict: If True, raise TemplateRenderError when variables
                are missing. If False (default), log a warning and
                leave placeholders intact.
            **variables: Template variables.

        Returns:
            A ContextBlock with rendered content and Origin.

        Raises:
            TemplateRenderError: If strict=True and variables are missing.
        """
        resolved_template = self.get_template(name, version)
        logger.info(
            "Rendering prompt '%s' (v%s) with %d variables",
            name,
            resolved_template.version,
            len(variables),
        )
        content = _interpolate(
            resolved_template.template, variables, strict=strict, template_name=name
        )

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
        priority: int = PRIORITY_SYSTEM_PROMPT,
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

    async def arender(
        self,
        name: str,
        version: str | None = None,
        priority: int = PRIORITY_SYSTEM_PROMPT,
        **variables: Any,
    ) -> ContextBlock:
        """Async version of :meth:`render`."""
        return self.render(name, version=version, priority=priority, **variables)

    async def acompose(
        self,
        base_name: str,
        override_name: str,
        separator: str = "\n\n",
        priority: int = PRIORITY_SYSTEM_PROMPT,
        **variables: Any,
    ) -> ContextBlock:
        """Async version of :meth:`compose`."""
        return self.compose(
            base_name,
            override_name,
            separator=separator,
            priority=priority,
            **variables,
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

        lines: List[str] = []
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

    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return sorted(self._templates.keys())

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a template.

        Args:
            name: Template name.

        Returns:
            List of version strings.
        """
        if name not in self._templates:
            raise KeyError(f"No template registered with name '{name}'")
        return [v.version for v in self._templates[name]]


def _interpolate(
    template: str,
    variables: Dict[str, Any],
    strict: bool = False,
    template_name: str = "",
) -> str:
    """Replace ``{{variable}}`` placeholders with values.

    Collects any missing variable names and either raises
    TemplateRenderError (strict mode) or logs a warning.

    Args:
        template: Template string with ``{{variable}}`` placeholders.
        variables: Variable name-to-value mapping.
        strict: If True, raise on missing variables.
        template_name: Template name for error messages.

    Returns:
        The interpolated string.

    Raises:
        TemplateRenderError: If strict is True and variables are missing.
    """
    missing_variables: List[str] = []

    def _replacer(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        if variable_name in variables:
            return str(variables[variable_name])
        missing_variables.append(variable_name)
        return match.group(0)

    result = _VARIABLE_PATTERN.sub(_replacer, template)

    if missing_variables and strict:
        raise TemplateRenderError(
            template_name=template_name or "(unknown)",
            missing_variables=missing_variables,
        )
    if missing_variables:
        logger.warning(
            "Unresolved template variables in '%s': %s",
            template_name,
            missing_variables,
        )

    return result
