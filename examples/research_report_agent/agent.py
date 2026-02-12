"""Core agent logic for the research report agent.

Handles session management, context assembly with RAG retrieval,
scratchpad note-taking, pipeline optimization, and dual-provider
formatting using contextkit components.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from contextkit.adapters.anthropic_adapter import AnthropicAdapter
from contextkit.adapters.openai_adapter import OpenAIAdapter
from contextkit.core import ContextWindow
from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.memory.short_term import ShortTermMemory
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
from contextkit.scope import ContextScope
from contextkit.tools.tool_registry import ToolRegistry

# ---------------------------------------------------------------
# Knowledge base of research articles
# ---------------------------------------------------------------

RESEARCH_ARTICLES: List[Chunk] = [
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
# Simulated tool outputs
# ---------------------------------------------------------------


def _simulate_paper_search(query: str) -> str:
    """Simulate an academic paper search API call.

    Args:
        query: Search query string.

    Returns:
        JSON string of search results.
    """
    return json.dumps({
        "query": query,
        "results": [
            {
                "title": "Context Engineering: A Framework for LLM Applications",
                "authors": ["Chen, A.", "Kumar, R."],
                "year": 2024,
                "citations": 127,
            },
            {
                "title": "Optimal Context Window Composition for RAG Systems",
                "authors": ["Park, S.", "Li, W."],
                "year": 2024,
                "citations": 84,
            },
        ],
        "total_results": 2,
    })


def _simulate_findings_extraction(paper_id: str) -> str:
    """Simulate extracting key findings from a paper.

    Args:
        paper_id: Paper identifier.

    Returns:
        JSON string of extracted findings.
    """
    return json.dumps({
        "paper_id": paper_id,
        "key_findings": [
            "Priority-based context assembly improves response quality by 23%",
            "Deduplication reduces token usage by 15-30% without quality loss",
            "U-shaped attention reordering improves recall on middle content",
        ],
        "methodology": "Controlled experiments across 5 LLM providers",
    })


class ResearchSession:
    """An active research session with its own context scope.

    Each session tracks its conversation, scratchpad notes, and
    context window independently.

    Attributes:
        session_id: Unique session identifier.
        topic: Research topic for this session.
        audience: Target audience level.
    """

    def __init__(
        self,
        session_id: str,
        topic: str,
        audience: str,
        scope: ContextScope,
        conversation: ShortTermMemory,
    ) -> None:
        self.session_id = session_id
        self.topic = topic
        self.audience = audience
        self.scope = scope
        self.conversation = conversation


class ResearchAgent:
    """Research report agent that uses contextkit for context assembly.

    Manages research sessions with RAG retrieval, scratchpad notes,
    full pipeline optimization (dedup, compact, trim, reorder), and
    dual-provider formatting.
    """

    TOKEN_BUDGET = 4000

    def __init__(self) -> None:
        """Initialize the research agent with contextkit components."""
        self._rag_retriever = InMemoryRetriever()
        self._rag_context = RAGContext(
            retriever=self._rag_retriever, retriever_name="research-kb"
        )
        self._tool_registry = ToolRegistry()
        self._memory_backend = InMemoryBackend()
        self._long_term_memory = LongTermMemory(backend=self._memory_backend)
        self._prompt_manager = PromptManager()
        self._anthropic_adapter = AnthropicAdapter()
        self._openai_adapter = OpenAIAdapter()
        self._quality_scorer = QualityScorer()
        self._sufficiency_checker = SufficiencyChecker()
        self._sessions: Dict[str, ResearchSession] = {}

    async def initialize(self) -> None:
        """Load the knowledge base, register tools, and seed memory.

        Call once at application startup.
        """
        self._rag_retriever.add_chunks(RESEARCH_ARTICLES)
        self._register_tools()
        self._register_prompts()
        await self._seed_long_term_memory()

    def _register_tools(self) -> None:
        """Register research tools with their JSON schemas."""
        self._tool_registry.register(
            name="search_papers",
            description="Search academic papers by topic and date range. "
            "Returns titles, abstracts, and citation counts.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "year_from": {"type": "integer", "description": "Start year."},
                    "max_results": {"type": "integer", "description": "Max papers."},
                },
                "required": ["query"],
            },
            tags=["search", "papers"],
        )
        self._tool_registry.register(
            name="extract_key_findings",
            description="Extract key findings and methodology from a paper.",
            parameters={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper ID or DOI."},
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Focus areas for extraction.",
                    },
                },
                "required": ["paper_id"],
            },
            tags=["analysis", "papers"],
        )
        self._tool_registry.register(
            name="generate_citation",
            description="Generate a formatted citation in APA, MLA, or Chicago.",
            parameters={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper ID or DOI."},
                    "style": {
                        "type": "string",
                        "enum": ["apa", "mla", "chicago"],
                    },
                },
                "required": ["paper_id", "style"],
            },
            tags=["citation"],
        )

    def _register_prompts(self) -> None:
        """Register versioned prompt templates."""
        self._prompt_manager.register(
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

    async def _seed_long_term_memory(self) -> None:
        """Seed long-term memory with researcher profile data."""
        await self._long_term_memory.store(
            "researcher_preferences",
            "Prefers APA citation style. Focus on empirical results and "
            "quantitative findings. Include methodology assessments.",
            tags=["preferences", "research"],
            importance=0.7,
        )
        await self._long_term_memory.store(
            "prior_research",
            "Previously researched prompt engineering techniques (2 weeks ago). "
            "Key finding: chain-of-thought significantly improves reasoning "
            "on math and logic tasks.",
            tags=["history", "research"],
            importance=0.5,
        )

    async def create_session(
        self, topic: str, audience: str, max_sources: int
    ) -> Dict[str, Any]:
        """Create a new research session and assemble initial context.

        Sets up a scoped context window, retrieves relevant sources,
        initializes the scratchpad with a report outline, runs the
        optimization pipeline, and formats for the LLM API.

        Args:
            topic: Research topic to investigate.
            audience: Target audience level.
            max_sources: Maximum number of RAG sources to retrieve.

        Returns:
            Dict with session_id, scratchpad_notes, formatted_payload,
            and context_inspection.
        """
        session_id = str(uuid.uuid4())[:8]

        context_window = ContextWindow(max_tokens=self.TOKEN_BUDGET)
        scope = ContextScope(
            agent_name="researcher",
            window=context_window,
            track_history=True,
        )
        conversation = ShortTermMemory(strategy="sliding_window", max_turns=8)

        session = ResearchSession(
            session_id=session_id,
            topic=topic,
            audience=audience,
            scope=scope,
            conversation=conversation,
        )
        self._sessions[session_id] = session

        # Record the initial user query
        conversation.add_turn("user", f"Research topic: {topic}")

        # 1. System prompt
        system_prompt_block = self._prompt_manager.render(
            "research_agent_v1", topic=topic, audience=audience
        )
        scope.window.add(system_prompt_block)

        # 2. Conversation history
        conversation_block = conversation.to_block()
        scope.window.add(conversation_block)

        # 3. Long-term memory (researcher preferences)
        preference_blocks = await self._long_term_memory.retrieve_as_blocks(
            "research preferences", top_k=2
        )
        for block in preference_blocks:
            scope.window.add(block)

        # 4. RAG retrieval
        rag_blocks = await self._rag_context.retrieve(
            query=topic, top_k=max_sources, min_relevance=0.5
        )
        for block in rag_blocks:
            scope.window.add(block)

        # 5. Tool definitions
        tool_definition_blocks = self._tool_registry.select(topic, max_tools=3)
        for block in tool_definition_blocks:
            scope.window.add(block)

        # 6. Simulated tool outputs
        search_result = _simulate_paper_search(topic)
        search_output = self._tool_registry.capture_output(
            tool_name="search_papers", result=search_result, latency_ms=320.5
        )
        scope.window.add(search_output.to_block(priority=75))

        findings_result = _simulate_findings_extraction("chen2024-context-eng")
        findings_output = self._tool_registry.capture_output(
            tool_name="extract_key_findings",
            result=findings_result,
            latency_ms=145.8,
        )
        scope.window.add(findings_output.to_block(priority=75))

        # 7. Scratchpad with initial outline
        scope.scratchpad.write("topic", topic)
        scope.scratchpad.write("status", "gathering sources")
        scope.scratchpad.write(
            "outline",
            "1. Introduction to the topic\n"
            "2. Key techniques and methods\n"
            "3. Recent advances and empirical results\n"
            "4. Token budget and optimization strategies\n"
            "5. Gaps and future directions",
        )
        scope.window.add(scope.scratchpad.to_block())

        scope.record_turn(events=["session_created", "sources_gathered"])

        # 8. Pipeline optimization
        context_inspection = self._optimize_and_inspect(scope, topic)

        # 9. Format for Anthropic
        formatted_payload = self._anthropic_adapter.format(scope.window)

        scratchpad_notes = self._get_scratchpad_notes(scope)

        return {
            "session_id": session_id,
            "topic": topic,
            "audience": audience,
            "scratchpad_notes": scratchpad_notes,
            "formatted_payload": formatted_payload,
            "context_inspection": context_inspection,
        }

    async def query_session(
        self, session_id: str, query: str
    ) -> Dict[str, Any]:
        """Process a follow-up research query within a session.

        Adds the query to conversation history, retrieves additional
        RAG sources, updates scratchpad notes, and re-optimizes the
        context window.

        Args:
            session_id: Existing session identifier.
            query: The follow-up research question.

        Returns:
            Dict with query, scratchpad_notes, formatted_payload,
            and context_inspection.

        Raises:
            KeyError: If the session does not exist.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")

        scope = session.scope
        conversation = session.conversation

        # Add the new query to conversation
        conversation.add_turn("user", query)

        # Rebuild the context window for the new turn
        scope.window.replace_blocks([])

        # System prompt
        system_prompt_block = self._prompt_manager.render(
            "research_agent_v1",
            topic=session.topic,
            audience=session.audience,
        )
        scope.window.add(system_prompt_block)

        # Updated conversation history
        conversation_block = conversation.to_block()
        scope.window.add(conversation_block)

        # RAG retrieval for the new query
        rag_blocks = await self._rag_context.retrieve(
            query=query, top_k=5, min_relevance=0.5
        )
        for block in rag_blocks:
            scope.window.add(block)

        # Tool definitions
        tool_definition_blocks = self._tool_registry.select(query, max_tools=3)
        for block in tool_definition_blocks:
            scope.window.add(block)

        # Update scratchpad
        scope.scratchpad.write("status", "investigating follow-up")
        scope.scratchpad.write("latest_query", query)
        scope.window.add(scope.scratchpad.to_block())

        scope.record_turn(events=["follow_up_query"])

        # Simulated assistant response
        conversation.add_turn(
            "assistant", "Let me investigate that further."
        )

        # Pipeline optimization
        context_inspection = self._optimize_and_inspect(scope, query)

        # Format for Anthropic
        formatted_payload = self._anthropic_adapter.format(scope.window)

        scratchpad_notes = self._get_scratchpad_notes(scope)

        return {
            "session_id": session_id,
            "query": query,
            "scratchpad_notes": scratchpad_notes,
            "formatted_payload": formatted_payload,
            "context_inspection": context_inspection,
        }

    def get_session(self, session_id: str) -> ResearchSession | None:
        """Retrieve a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            The session, or None if not found.
        """
        return self._sessions.get(session_id)

    def _optimize_and_inspect(
        self, scope: ContextScope, query: str
    ) -> Dict[str, Any]:
        """Run the pipeline and collect inspection data.

        Executes deduplication, compaction, trimming, and reordering,
        then evaluates sufficiency and quality.

        Args:
            scope: The agent's context scope.
            query: The current query for sufficiency checking.

        Returns:
            Context inspection dict.
        """
        pipeline = ContextPipeline(
            steps=[
                DeduplicateStep(similarity_threshold=0.6),
                CompactStep(target_ratio=0.7, min_tokens=30),
                TrimStep(
                    max_tokens=scope.window.max_tokens, min_priority=20
                ),
                ReorderStep(strategy="important_edges"),
            ]
        )
        pipeline.run(scope.window)
        pipeline_report = pipeline.last_report

        sufficiency_result = self._sufficiency_checker.check(
            query, scope.window.blocks
        )
        quality_report = self._quality_scorer.score(scope.window.blocks)

        block_summaries = [
            {
                "name": block.display_name,
                "type": block.type.value,
                "tokens": block.token_count,
                "priority": block.priority,
            }
            for block in scope.window.blocks
        ]

        step_reports = [
            {
                "step_name": step.step_name,
                "tokens_before": step.tokens_before,
                "tokens_after": step.tokens_after,
                "tokens_saved": step.tokens_saved,
                "blocks_removed": step.blocks_removed,
            }
            for step in pipeline_report.steps
        ]

        return {
            "total_blocks": len(scope.window.blocks),
            "total_tokens": scope.window.token_count,
            "max_tokens": scope.window.max_tokens,
            "budget_used_percent": round(
                scope.window.token_count / scope.window.max_tokens * 100, 1
            ),
            "blocks": block_summaries,
            "pipeline_steps": step_reports,
            "total_tokens_saved": pipeline_report.total_tokens_saved,
            "sufficiency_confident": sufficiency_result.sufficient,
            "sufficiency_score": round(sufficiency_result.confidence, 2),
            "quality_score": round(quality_report.overall_score, 2),
        }

    @staticmethod
    def _get_scratchpad_notes(scope: ContextScope) -> List[Dict[str, str]]:
        """Extract scratchpad notes as a list of dicts.

        Args:
            scope: The agent's context scope.

        Returns:
            List of dicts with key and content.
        """
        return [
            {"key": key, "content": scope.scratchpad.read(key) or ""}
            for key in scope.scratchpad.list_keys()
        ]

    @property
    def knowledge_base_size(self) -> int:
        """Number of chunks in the knowledge base."""
        return self._rag_retriever.chunk_count

    @property
    def registered_tool_count(self) -> int:
        """Number of registered tools."""
        return self._tool_registry.tool_count

    @property
    def active_session_count(self) -> int:
        """Number of active research sessions."""
        return len(self._sessions)
