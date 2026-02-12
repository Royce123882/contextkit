"""Research Report Agent -- end-to-end agentic use case.

Combines RAG knowledge retrieval, scratchpad working notes, full
context pipeline (deduplicate, compress, trim, reorder), sufficiency
and quality checks, and dual-adapter formatting into a research
agent that gathers sources and produces a structured report.

Usage:
    python examples/research_report_agent/main.py
"""

import asyncio
import json

from contextkit.adapters.anthropic_adapter import AnthropicAdapter
from contextkit.adapters.openai_adapter import OpenAIAdapter
from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.memory.short_term import ShortTermMemory
from contextkit.observe.inspect import inspect_window
from contextkit.observe.quality import QualityScorer
from contextkit.observe.sufficiency import SufficiencyChecker
from contextkit.pipeline import (
    CompactStep,
    ContextPipeline,
    DeduplicateStep,
    ReorderStep,
    TrimStep,
)
from contextkit.prompts.prompt_manager import PromptManager
from contextkit.rag.chunk import Chunk
from contextkit.rag.context import RAGContext
from contextkit.rag.in_memory_retriever import InMemoryRetriever
from contextkit.scope import ContextScope, Scratchpad
from contextkit.tools.tool_registry import ToolRegistry


# ---------------------------------------------------------------
# 1. Knowledge Base -- research papers and articles indexed for RAG
# ---------------------------------------------------------------

RESEARCH_ARTICLES = [
    Chunk(
        content="Retrieval-Augmented Generation (RAG) combines information "
        "retrieval with language generation. The retriever first identifies "
        "relevant documents, then the generator produces an answer grounded "
        "in those documents. This reduces hallucinations and enables "
        "knowledge-intensive tasks without fine-tuning.",
        source="papers/lewis2020-rag.pdf",
        relevance_score=0.95,
        metadata={"topic": "rag", "year": 2020},
    ),
    Chunk(
        content="Lost in the Middle (Liu et al., 2023) shows that LLMs "
        "struggle with information placed in the middle of long contexts. "
        "Performance follows a U-shaped curve: models attend well to the "
        "beginning and end but poorly to the center. This has implications "
        "for context window design and information ordering.",
        source="papers/liu2023-lost-in-middle.pdf",
        relevance_score=0.92,
        metadata={"topic": "context_engineering", "year": 2023},
    ),
    Chunk(
        content="Chain-of-Thought prompting (Wei et al., 2022) improves "
        "reasoning by having the model produce intermediate steps. "
        "Scratchpad-based approaches extend this by giving the model a "
        "persistent workspace for multi-step reasoning across turns.",
        source="papers/wei2022-chain-of-thought.pdf",
        relevance_score=0.88,
        metadata={"topic": "prompting", "year": 2022},
    ),
    Chunk(
        content="Tool-augmented language models (Schick et al., 2023) can "
        "call external APIs, databases, and calculators. This extends LLM "
        "capabilities beyond text generation to real-world actions. Key "
        "challenges include tool selection, parameter extraction, and "
        "error handling in multi-step tool chains.",
        source="papers/schick2023-toolformer.pdf",
        relevance_score=0.85,
        metadata={"topic": "tools", "year": 2023},
    ),
    Chunk(
        content="Context engineering is the systematic design of the "
        "information fed to LLMs. It goes beyond prompt engineering by "
        "managing RAG retrieval, memory systems, tool outputs, and "
        "conversation history as structured blocks with priorities. "
        "Effective context engineering optimizes for relevance, "
        "deduplication, and positional attention patterns.",
        source="articles/context-engineering-overview.md",
        relevance_score=0.97,
        metadata={"topic": "context_engineering", "year": 2024},
    ),
    Chunk(
        content="Multi-agent systems use context handoff protocols to "
        "transfer knowledge between specialized agents. Shared memory "
        "enables persistent cross-agent state, while scratchpads give "
        "each agent a private workspace for intermediate reasoning.",
        source="articles/multi-agent-patterns.md",
        relevance_score=0.80,
        metadata={"topic": "multi_agent", "year": 2024},
    ),
    Chunk(
        content="Token budget management is critical for production RAG "
        "systems. Strategies include priority-based trimming, content "
        "compression via extractive summarization, deduplication of "
        "overlapping chunks, and dynamic reordering to place critical "
        "information at high-attention positions (start and end).",
        source="articles/token-budget-strategies.md",
        relevance_score=0.90,
        metadata={"topic": "context_engineering", "year": 2024},
    ),
]


# ---------------------------------------------------------------
# 2. Tool Definitions -- search, summarize, cite
# ---------------------------------------------------------------


def register_tools(registry: ToolRegistry) -> None:
    """Register all research agent tools."""
    registry.register(
        name="search_papers",
        description="Search academic papers by topic and date range. "
        "Returns titles, abstracts, and citation counts.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for papers",
                },
                "year_from": {
                    "type": "integer",
                    "description": "Start year filter",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum papers to return (default 10)",
                },
            },
            "required": ["query"],
        },
        tags=["search", "papers"],
    )

    registry.register(
        name="extract_key_findings",
        description="Extract key findings and methodology from a paper. "
        "Returns a structured summary with claims and evidence.",
        parameters={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Paper identifier or DOI",
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific areas to focus extraction on",
                },
            },
            "required": ["paper_id"],
        },
        tags=["analysis", "papers"],
    )

    registry.register(
        name="generate_citation",
        description="Generate a formatted citation for a paper in the "
        "specified style (APA, MLA, Chicago).",
        parameters={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Paper identifier or DOI",
                },
                "style": {
                    "type": "string",
                    "enum": ["apa", "mla", "chicago"],
                },
            },
            "required": ["paper_id", "style"],
        },
        tags=["citation"],
    )


# ---------------------------------------------------------------
# 3. Simulated tool executions
# ---------------------------------------------------------------


def simulate_paper_search(query: str) -> str:
    """Simulate a paper search API call."""
    return json.dumps({
        "query": query,
        "results": [
            {
                "title": "Context Engineering: A Framework for LLM Applications",
                "authors": ["Chen, A.", "Kumar, R."],
                "year": 2024,
                "citations": 127,
                "abstract": "We propose a systematic framework for managing "
                "context in LLM applications...",
            },
            {
                "title": "Optimal Context Window Composition for RAG Systems",
                "authors": ["Park, S.", "Li, W."],
                "year": 2024,
                "citations": 84,
                "abstract": "This paper studies the optimal composition of "
                "context windows in RAG systems...",
            },
        ],
        "total_results": 2,
    })


def simulate_extract_findings(paper_id: str) -> str:
    """Simulate key findings extraction."""
    return json.dumps({
        "paper_id": paper_id,
        "key_findings": [
            "Priority-based context assembly improves response quality by 23%",
            "Deduplication reduces token usage by 15-30% without quality loss",
            "U-shaped attention reordering improves recall on middle content",
        ],
        "methodology": "Controlled experiments across 5 LLM providers",
        "limitations": "Only tested on English-language tasks",
    })


# ---------------------------------------------------------------
# 4. Main agent loop
# ---------------------------------------------------------------


async def run_agent() -> None:
    """Run the research report agent."""
    # -- Setup components --
    retriever = InMemoryRetriever()
    retriever.add_chunks(RESEARCH_ARTICLES)
    rag = RAGContext(retriever=retriever, retriever_name="research-kb")

    tool_registry = ToolRegistry()
    register_tools(tool_registry)

    memory_backend = InMemoryBackend()
    long_term = LongTermMemory(backend=memory_backend)

    conversation = ShortTermMemory(strategy="sliding_window", max_turns=8)

    prompt_mgr = PromptManager()
    prompt_mgr.register(
        "research_agent_v1",
        "You are a research assistant specializing in AI and LLM systems.\n\n"
        "Guidelines:\n"
        "- Search for and synthesize academic papers\n"
        "- Take structured notes in your scratchpad\n"
        "- Cite all claims with sources\n"
        "- Identify gaps in current research\n"
        "- Produce a well-organized report\n\n"
        "Research topic: {{topic}}\n"
        "Audience level: {{audience}}",
        description="Main research agent system prompt",
    )

    # -- Create a scoped agent with scratchpad --
    window = ContextWindow(max_tokens=4000)
    scope = ContextScope(
        agent_name="researcher",
        window=window,
        track_history=True,
    )

    # -- Store researcher profile in long-term memory --
    await long_term.store(
        "researcher_preferences",
        "Prefers APA citation style. Focus on empirical results and "
        "quantitative findings. Include methodology assessments.",
        tags=["preferences", "research"],
        importance=0.7,
    )
    await long_term.store(
        "prior_research",
        "Previously researched prompt engineering techniques (2 weeks ago). "
        "Key finding: chain-of-thought significantly improves reasoning "
        "on math and logic tasks.",
        tags=["history", "research"],
        importance=0.5,
    )

    # -- Simulate a multi-turn research session --
    print("=" * 60)
    print("RESEARCH REPORT AGENT")
    print("=" * 60)

    research_topic = "context engineering for LLM-based agents"

    conversation.add_turn(
        "user",
        f"I need a comprehensive report on {research_topic}. "
        "Cover the key techniques, recent advances, and best practices.",
    )
    conversation.add_turn(
        "assistant",
        "I'll research that topic. Let me start by searching for "
        "relevant papers and gathering background information.",
    )
    conversation.add_turn(
        "user",
        "Focus especially on RAG, memory systems, and token budget management.",
    )

    # -- Agent takes notes in the scratchpad --
    scratchpad = scope.scratchpad
    scratchpad.write("topic", research_topic)
    scratchpad.write("status", "gathering sources")
    scratchpad.write(
        "outline",
        "1. Introduction to context engineering\n"
        "2. RAG techniques and retrieval strategies\n"
        "3. Memory systems (short-term + long-term)\n"
        "4. Token budget management and pipeline optimization\n"
        "5. Quality assurance and attention patterns\n"
        "6. Gaps and future directions",
    )

    # -- Build the context window for the next turn --
    user_query = "context engineering RAG memory token budget best practices"

    # 1. System prompt with topic
    pref_blocks = await long_term.retrieve_as_blocks("research preferences", top_k=2)
    system_block = prompt_mgr.render(
        "research_agent_v1",
        topic=research_topic,
        audience="technical (ML engineers and researchers)",
    )
    scope.window.add(system_block)
    print(f"\n[System prompt]  {system_block.token_count} tokens")

    # 2. Conversation history
    history_block = conversation.to_block()
    scope.window.add(history_block)
    print(
        f"[Conversation]   {history_block.token_count} tokens "
        f"({conversation.turn_count} turns)"
    )

    # 3. Long-term memory (researcher prefs + prior research)
    for block in pref_blocks:
        scope.window.add(block)
    print(f"[Long-term mem]  {len(pref_blocks)} blocks retrieved")

    # 4. RAG -- retrieve relevant research articles
    rag_blocks = await rag.retrieve(
        query=user_query, top_k=5, min_relevance=0.5
    )
    for block in rag_blocks:
        scope.window.add(block)
    print(f"[RAG]            {len(rag_blocks)} chunks retrieved")

    # 5. Tool definitions
    tool_blocks = tool_registry.select(user_query, max_tools=3)
    for block in tool_blocks:
        scope.window.add(block)
    print(f"[Tools]          {len(tool_blocks)} tools selected")

    # 6. Simulated tool outputs
    search_result = simulate_paper_search("context engineering LLM")
    search_output = tool_registry.capture_output(
        tool_name="search_papers",
        result=search_result,
        latency_ms=320.5,
    )
    scope.window.add(search_output.to_block(priority=75))
    print(f"[Tool output]    search_papers ({search_output.latency_ms}ms)")

    findings_result = simulate_extract_findings("chen2024-context-eng")
    findings_output = tool_registry.capture_output(
        tool_name="extract_key_findings",
        result=findings_result,
        latency_ms=145.8,
    )
    scope.window.add(findings_output.to_block(priority=75))
    print(f"[Tool output]    extract_key_findings ({findings_output.latency_ms}ms)")

    # 7. Scratchpad notes
    scratchpad.write("status", "analyzing sources")
    scratchpad.write(
        "key_insight_1",
        "Priority-based assembly + dedup = 23% quality improvement "
        "with 15-30% fewer tokens",
    )
    scratchpad.write(
        "key_insight_2",
        "U-shaped attention reordering addresses 'lost in the middle' "
        "problem for information recall",
    )
    scope.window.add(scratchpad.to_block())
    print(f"[Scratchpad]     {scratchpad.note_count} notes")

    # Record turn snapshot
    scope.record_turn(events=["sources_gathered", "analysis_started"])

    # -- Pipeline: optimize context --
    print(f"\n[Pre-pipeline]   {scope.window.token_count:,} tokens "
          f"/ {scope.window.max_tokens:,} budget")

    pipeline = ContextPipeline(
        steps=[
            DeduplicateStep(similarity_threshold=0.6),
            CompactStep(target_ratio=0.7, min_tokens=30),
            TrimStep(max_tokens=scope.window.max_tokens, min_priority=20),
            ReorderStep(strategy="important_edges"),
        ]
    )
    pipeline.run(scope.window)

    report = pipeline.last_report
    print(f"\n[Pipeline]       {len(report.steps)} steps executed")
    for step_report in report.steps:
        print(
            f"  {step_report.step_name}: "
            f"{step_report.tokens_before:,} -> {step_report.tokens_after:,} "
            f"(saved {step_report.tokens_saved:,}, "
            f"removed {step_report.blocks_removed} blocks)"
        )
    print(f"  Total saved: {report.total_tokens_saved:,} tokens")

    # -- Sufficiency check --
    checker = SufficiencyChecker()
    sufficiency = checker.check(user_query, scope.window.blocks)
    print(
        f"\n[Sufficiency]    sufficient={sufficiency.sufficient}, "
        f"confidence={sufficiency.confidence:.2f}, "
        f"coverage={sufficiency.query_coverage:.0%}"
    )
    if sufficiency.suggestions:
        for s in sufficiency.suggestions:
            print(f"                 {s}")

    # -- Quality check --
    scorer = QualityScorer()
    quality = scorer.score(scope.window.blocks)
    print(f"[Quality]        score={quality.overall_score:.2f}")
    if quality.warnings:
        for w in quality.warnings:
            print(f"                 {w}")

    # -- Mutation history --
    print("\n[Mutations]")
    for block in scope.window.blocks:
        if block.mutations:
            print(f"  {block.display_name}:")
            for mutation in block.mutations:
                print(f"    [{mutation.step}] {mutation.action}: {mutation.detail}")

    # -- Timeline tracking --
    scope.record_turn(events=["pipeline_optimized", "ready_for_generation"])
    print("\n[Timeline]")
    if scope.timeline:
        for snapshot in scope.timeline.snapshots:
            print(
                f"  Turn {snapshot.turn}: "
                f"{snapshot.block_count} blocks, "
                f"{snapshot.token_count:,} tokens, "
                f"budget {snapshot.budget_percent:.0f}%"
            )

    # -- Inspect the final context --
    print("\n" + "=" * 60)
    print("ASSEMBLED CONTEXT")
    print("=" * 60)
    print(inspect_window(scope.window))

    print(
        f"\nTotal: {len(scope.window.blocks)} blocks, "
        f"{scope.window.token_count:,}/{scope.window.max_tokens:,} tokens"
    )

    # -- Format for both providers --
    anthropic_payload = AnthropicAdapter().format(scope.window)
    openai_payload = OpenAIAdapter().format(scope.window)

    print(
        f"\nAnthropic payload: system={len(anthropic_payload.get('system', ''))} chars, "
        f"messages={len(anthropic_payload['messages'])}"
    )
    print(
        f"OpenAI payload:    messages={len(openai_payload['messages'])}"
    )


if __name__ == "__main__":
    asyncio.run(run_agent())
