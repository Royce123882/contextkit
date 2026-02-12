"""Model Context Protocol (MCP) integration.

Provides context provider and consumer classes for exposing
and consuming ContextWindow blocks via the MCP resource protocol.
"""

from contextkit.mcp.context_consumer import MCPContextConsumer
from contextkit.mcp.context_provider import MCPContextProvider

__all__ = ["MCPContextConsumer", "MCPContextProvider"]
