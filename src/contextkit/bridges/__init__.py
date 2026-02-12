"""Integration bridges for popular LLM frameworks.

Provides adapter classes that allow contextkit components to be
used within third-party frameworks like LangChain.
"""

from contextkit.bridges.langchain_bridge import (
    ContextKitMemory,
    ContextKitRetriever,
)

__all__ = ["ContextKitMemory", "ContextKitRetriever"]
