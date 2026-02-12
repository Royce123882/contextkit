"""MCP context provider.

Exposes a ContextWindow's blocks as MCP-compatible resources
that can be listed and read by MCP clients. This enables
agent-to-agent context sharing via the Model Context Protocol.

Research basis: CA-MCP (arXiv:2601.11595) -- shared context
stores enable autonomous server coordination through MCP.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from contextkit.core import ContextWindow
from contextkit.exceptions import RecordNotFoundError

logger = logging.getLogger("contextkit")

_RESOURCE_URI_PREFIX = "context://"


class MCPContextProvider:
    """Expose a ContextWindow's blocks as MCP-compatible resources.

    Each block in the window becomes a resource that can be listed
    and read by URI. Resource URIs follow the pattern
    ``context://<block_name>``.

    Args:
        window: The ContextWindow whose blocks to expose.
    """

    def __init__(self, window: ContextWindow) -> None:
        self._window = window

    @property
    def window(self) -> ContextWindow:
        """The underlying ContextWindow."""
        return self._window

    def list_resources(self) -> List[Dict[str, Any]]:
        """Return MCP resource descriptors for each block.

        Each descriptor includes a URI, name, MIME type,
        description, and token count metadata.

        Returns:
            List of MCP resource descriptor dicts.
        """
        resources: List[Dict[str, Any]] = []
        for block in self._window.blocks:
            resource = {
                "uri": f"{_RESOURCE_URI_PREFIX}{block.display_name}",
                "name": block.display_name,
                "mimeType": "text/plain",
                "description": (
                    f"{block.type.value} block "
                    f"({block.token_count:,} tokens, "
                    f"priority {block.priority})"
                ),
                "metadata": {
                    "token_count": block.token_count,
                    "priority": block.priority,
                    "block_type": block.type.value,
                },
            }
            resources.append(resource)

        logger.debug(
            "Listed %d MCP resources from window", len(resources)
        )
        return resources

    def read_resource(self, uri: str) -> str:
        """Read a single block's content by URI.

        Args:
            uri: The resource URI (e.g. ``context://system_prompt``).

        Returns:
            The block's content as a string. If the content is a
            list, it is serialized to JSON.

        Raises:
            RecordNotFoundError: If no block matches the given URI.
        """
        block_name = uri.replace(_RESOURCE_URI_PREFIX, "")
        block = self._window.get_block(block_name)

        if block is None:
            raise RecordNotFoundError(
                f"No block with name '{block_name}' in context window"
            )

        if isinstance(block.content, str):
            return block.content
        return json.dumps(block.content, indent=2)

    def resource_metadata(self, uri: str) -> Dict[str, Any]:
        """Get metadata for a single resource by URI.

        Args:
            uri: The resource URI.

        Returns:
            Dict with block metadata (type, priority, tokens, origin).

        Raises:
            RecordNotFoundError: If no block matches the given URI.
        """
        block_name = uri.replace(_RESOURCE_URI_PREFIX, "")
        block = self._window.get_block(block_name)

        if block is None:
            raise RecordNotFoundError(
                f"No block with name '{block_name}' in context window"
            )

        metadata: Dict[str, Any] = {
            "block_type": block.type.value,
            "priority": block.priority,
            "token_count": block.token_count,
        }
        if block.origin is not None:
            metadata["origin"] = block.origin.summary()

        return metadata
