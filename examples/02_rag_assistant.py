"""Example 2: RAG-Powered Knowledge Assistant.

Demonstrates retrieval-augmented generation (RAG) with contextkit:
indexing documents, retrieving relevant chunks, deduplication,
budget-aware retrieval, and provenance tracking.

Usage:
    python examples/02_rag_assistant.py
"""

import asyncio

from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.observe.inspect import inspect_window
from contextkit.prompts.manager import PromptManager
from contextkit.rag.backends import Chunk, InMemoryRetriever
from contextkit.rag.context import RAGContext


async def main() -> None:
    """Run a RAG-powered knowledge assistant example."""
    # ---------------------------------------------------------------
    # Step 1: Build a knowledge base with document chunks
    # ---------------------------------------------------------------
    retriever = InMemoryRetriever()

    # Simulate indexed documentation chunks
    documents = [
        Chunk(
            content="ContextWindow is the core container that holds all context "
            "blocks. It tracks token usage, enforces budgets, and "
            "emits events when blocks are added or removed.",
            source="docs/core.md",
            relevance_score=0.9,
            metadata={"section": "Core Concepts"},
        ),
        Chunk(
            content="ContextBlock represents a single piece of context with a "
            "type (system prompt, memory, RAG, etc.), priority, and "
            "content. Blocks can be strings or message lists.",
            source="docs/core.md",
            relevance_score=0.85,
            metadata={"section": "Core Concepts"},
        ),
        Chunk(
            content="The ContextAssembler composes blocks into a window. It "
            "sorts by priority and produces an AssemblyReport explaining "
            "why each block was included or excluded.",
            source="docs/assembler.md",
            relevance_score=0.8,
            metadata={"section": "Assembly"},
        ),
        Chunk(
            content="RAGContext wraps a RetrieverBackend to provide high-level "
            "retrieval with ranking, deduplication, and attribution. "
            "Each retrieved chunk gets an Origin for provenance tracking.",
            source="docs/rag.md",
            relevance_score=0.95,
            metadata={"section": "RAG"},
        ),
        Chunk(
            content="The pipeline system provides TrimStep, CompactStep, "
            "DeduplicateStep, ReorderStep, and FilterStep for "
            "post-assembly context optimization.",
            source="docs/pipeline.md",
            relevance_score=0.7,
            metadata={"section": "Pipeline"},
        ),
        Chunk(
            content="Budget warnings fire at configurable thresholds (e.g., "
            "70%, 90%) to alert you before hitting token limits. "
            "Events can be captured with subscribe().",
            source="docs/observe.md",
            relevance_score=0.6,
            metadata={"section": "Observability"},
        ),
    ]
    retriever.add_chunks(documents)
    print(f"Knowledge base loaded: {retriever.chunk_count} chunks")

    # ---------------------------------------------------------------
    # Step 2: Create a RAG context manager
    # ---------------------------------------------------------------
    rag = RAGContext(retriever=retriever, retriever_name="docs-index")

    # ---------------------------------------------------------------
    # Step 3: Retrieve relevant chunks for a user query
    # ---------------------------------------------------------------
    query = "How does the context window work?"

    # Basic retrieval: top 3 most relevant chunks
    rag_blocks = await rag.retrieve(
        query=query,
        top_k=3,
        min_relevance=0.1,
    )

    print(f"\nQuery: '{query}'")
    print(f"Retrieved {len(rag_blocks)} chunks:")
    for block in rag_blocks:
        origin = block.origin
        score = origin.details.get("relevance_score", 0) if origin else 0
        source = origin.details.get("source", "?") if origin else "?"
        print(f"  - [{source}] relevance={score:.2f} ({block.token_count} tokens)")

    # ---------------------------------------------------------------
    # Step 4: Budget-aware retrieval (stop when budget reached)
    # ---------------------------------------------------------------
    budget_blocks = await rag.retrieve(
        query="Tell me about all the features",
        top_k=10,
        max_tokens=200,  # Stop retrieving when we hit 200 tokens
    )
    total_rag_tokens = sum(b.token_count for b in budget_blocks)
    print(
        f"\nBudget-aware retrieval (max 200 tokens): "
        f"{len(budget_blocks)} chunks, {total_rag_tokens} tokens"
    )

    # ---------------------------------------------------------------
    # Step 5: Assemble everything into a context window
    # ---------------------------------------------------------------
    window = ContextWindow(max_tokens=4000)

    # Add a system prompt
    prompt_manager = PromptManager()
    prompt_manager.register(
        "rag_assistant",
        "You are a documentation assistant. Answer questions using "
        "ONLY the provided context. If the context doesn't contain "
        "the answer, say so.\n\nContext:\n{{context}}",
    )

    # Combine RAG content for the prompt
    rag_content = "\n\n".join(
        f"[Source: {b.origin.details.get('source', '?')}]\n{b.content}"
        for b in rag_blocks
        if b.origin
    )

    system_block = prompt_manager.render(
        "rag_assistant",
        context=rag_content,
    )
    window.add(system_block)

    # Add the user's question
    user_block = ContextBlock(
        type=BlockType.USER_CONTEXT,
        content=query,
        priority=90,
        name="user_query",
    )
    window.add(user_block)

    # Add the RAG blocks for provenance tracking
    for block in rag_blocks:
        window.add(block)

    # ---------------------------------------------------------------
    # Step 6: Inspect the assembled window
    # ---------------------------------------------------------------
    print("\n--- Context Window ---")
    print(inspect_window(window))

    print(f"\nTotal: {len(window.blocks)} blocks, " f"{window.token_count:,} tokens")

    # Show provenance for each RAG block
    print("\n--- RAG Provenance ---")
    for block in rag_blocks:
        if block.origin:
            print(
                f"  [{block.name}] source={block.origin.source}, "
                f"retriever={block.origin.details.get('retriever')}, "
                f"score={block.origin.details.get('relevance_score')}"
            )


if __name__ == "__main__":
    asyncio.run(main())
