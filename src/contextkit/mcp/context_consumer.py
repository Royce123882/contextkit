"""MCP context consumer.

Pulls resources from MCP resource descriptors into ContextBlocks.
Works with resource lists returned by MCPContextProvider or any
MCP-compatible server.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from contextkit.constants import PRIORITY_DEFAULT
from contextkit.core import BlockType, ContextBlock
from contextkit.observe.provenance import Origin

logger = logging.getLogger("contextkit")


class MCPContextConsumer:
    """Convert MCP resources into ContextBlocks.

    Takes MCP resource descriptors (as returned by
    :meth:`MCPContextProvider.list_resources`) and converts
    them into ContextBlocks that can be added to a ContextWindow.
    """

    def resources_to_blocks(
        self,
        resources: List[Dict[str, Any]],
        contents: Dict[str, str],
        priority: int = PRIORITY_DEFAULT,
    ) -> List[ContextBlock]:
        """Convert MCP resources and their contents into ContextBlocks.

        Args:
            resources: List of MCP resource descriptors (must have
                "uri" and "name" keys).
            contents: Mapping of resource URI to content string,
                as returned by reading each resource.
            priority: Default block priority for created blocks.

        Returns:
            List of ContextBlocks created from the resources.
        """
        blocks: List[ContextBlock] = []

        for resource in resources:
            uri = resource.get("uri", "")
            name = resource.get("name", "mcp_resource")
            content = contents.get(uri)

            if content is None:
                logger.debug(
                    "Skipping MCP resource '%s': no content provided", uri
                )
                continue

            block_type_str = (
                resource.get("metadata", {}).get("block_type", "")
            )
            block_type = self._resolve_block_type(block_type_str)

            resource_priority = (
                resource.get("metadata", {}).get("priority", priority)
            )

            origin = Origin(
                source="mcp",
                details={
                    "uri": uri,
                    "resource_name": name,
                    "mime_type": resource.get("mimeType", "text/plain"),
                },
            )

            block = ContextBlock(
                type=block_type,
                content=content,
                priority=resource_priority,
                name=name,
                origin=origin,
            )
            blocks.append(block)

        logger.info(
            "Converted %d MCP resources to context blocks", len(blocks)
        )
        return blocks

    @staticmethod
    def _resolve_block_type(block_type_str: str) -> BlockType:
        """Resolve a string block type to a BlockType enum.

        Falls back to SYSTEM_METADATA if the type is unknown.

        Args:
            block_type_str: The block type value string.

        Returns:
            The matching BlockType enum member.
        """
        for member in BlockType:
            if member.value == block_type_str:
                return member
        return BlockType.SYSTEM_METADATA
