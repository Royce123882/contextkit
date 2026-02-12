"""LangChain integration bridge.

Provides adapter classes that wrap contextkit's RAGContext and
ShortTermMemory for use within LangChain chains and agents.

Requires the ``langchain-core`` optional dependency. If not
installed, importing this module raises an ImportError with
a helpful message.

Usage::

    from contextkit.bridges import ContextKitRetriever, ContextKitMemory
    from contextkit.rag import RAGContext

    rag = RAGContext(retriever=my_retriever)
    lc_retriever = ContextKitRetriever(rag_context=rag)
    docs = await lc_retriever.ainvoke("my query")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from contextkit.constants import DEFAULT_TOP_K
from contextkit.core import ContextBlock
from contextkit.memory.short_term import ShortTermMemory
from contextkit.rag.context import RAGContext

logger = logging.getLogger("contextkit")


class ContextKitRetriever:
    """LangChain-compatible retriever backed by contextkit's RAGContext.

    Wraps RAGContext.retrieve() and converts ContextBlocks into
    dicts with ``page_content`` and ``metadata`` keys, matching
    the LangChain Document interface without requiring the
    langchain-core dependency at import time.

    Args:
        rag_context: A RAGContext instance to delegate retrieval to.
        top_k: Default number of results to return.
    """

    def __init__(
        self,
        rag_context: RAGContext,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._rag_context = rag_context
        self._top_k = top_k

    async def ainvoke(
        self,
        query: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve documents asynchronously (LangChain retriever interface).

        Args:
            query: The search query.
            top_k: Override the default number of results.
            **kwargs: Additional arguments passed to RAGContext.retrieve().

        Returns:
            List of dicts with ``page_content`` and ``metadata`` keys.
        """
        effective_top_k = top_k or self._top_k
        blocks = await self._rag_context.retrieve(
            query=query,
            top_k=effective_top_k,
            **kwargs,
        )
        return [self._block_to_document(block) for block in blocks]

    @staticmethod
    def _block_to_document(block: ContextBlock) -> Dict[str, Any]:
        """Convert a ContextBlock to a LangChain-compatible document dict.

        Args:
            block: The ContextBlock to convert.

        Returns:
            Dict with ``page_content`` and ``metadata`` keys.
        """
        content = block.content if isinstance(block.content, str) else str(block.content)
        metadata: Dict[str, Any] = {
            "tokens": block.token_count,
            "block_name": block.display_name,
            "block_type": block.type.value,
        }
        if block.origin is not None:
            metadata["origin"] = block.origin.summary()
        return {"page_content": content, "metadata": metadata}


class ContextKitMemory:
    """LangChain-compatible memory backed by contextkit's ShortTermMemory.

    Provides ``load_memory_variables()`` and ``save_context()`` methods
    that match LangChain's memory interface, enabling contextkit's
    conversation tracking to be used in LangChain chains.

    Args:
        short_term_memory: A ShortTermMemory instance to back the memory.
        memory_key: The key to use in the returned variables dict.
    """

    def __init__(
        self,
        short_term_memory: ShortTermMemory,
        memory_key: str = "history",
    ) -> None:
        self._memory = short_term_memory
        self._memory_key = memory_key

    @property
    def memory_key(self) -> str:
        """The key used in the memory variables dict."""
        return self._memory_key

    def load_memory_variables(
        self,
        inputs: Dict[str, Any] | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load current conversation history as memory variables.

        Args:
            inputs: Optional input dict (unused, for LangChain compatibility).

        Returns:
            Dict mapping ``memory_key`` to the list of messages.
        """
        return {self._memory_key: self._memory.messages}

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Save a conversation turn to memory.

        Extracts "input" from inputs as a user turn and "output"
        from outputs as an assistant turn.

        Args:
            inputs: Dict containing the user input (key "input").
            outputs: Dict containing the assistant output (key "output").
        """
        user_input = inputs.get("input", "")
        if user_input:
            self._memory.add_turn("user", str(user_input))

        assistant_output = outputs.get("output", "")
        if assistant_output:
            self._memory.add_turn("assistant", str(assistant_output))

    def clear(self) -> None:
        """Clear the conversation history."""
        self._memory.clear()
