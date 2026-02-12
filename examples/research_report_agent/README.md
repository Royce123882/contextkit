# Research Report Agent

A research assistant agent that gathers academic sources, takes structured notes, optimizes context with a full pipeline, and formats output for multiple LLM providers.

## Architecture

```
Research Query
    |
    v
+------------------------------+
|   Context Assembly            |
|                               |
|  +-------------------------+  |
|  | System Prompt           |  |  <- PromptManager (topic + audience)
|  | (priority=100)          |  |
|  +-------------------------+  |
|  | Conversation History    |  |  <- ShortTermMemory (last 8 turns)
|  | (priority=85)           |  |
|  +-------------------------+  |
|  | Researcher Preferences  |  |  <- LongTermMemory (persistent)
|  | (priority=80)           |  |
|  +-------------------------+  |
|  | Research Articles       |  |  <- RAG retrieval from knowledge base
|  | (priority=70)           |  |
|  +-------------------------+  |
|  | Tool Definitions        |  |  <- search, extract, cite
|  | (priority=60)           |  |
|  +-------------------------+  |
|  | Tool Outputs            |  |  <- Paper search + findings extraction
|  | (priority=75)           |  |
|  +-------------------------+  |
|  | Scratchpad              |  |  <- Agent working notes and outline
|  | (priority=55)           |  |
|  +-------------------------+  |
|                               |
|  Pipeline: Dedup -> Compact   |
|         -> Trim -> Reorder    |  <- Full 4-step optimization
+------------------------------+
    |
    v
  Format for Anthropic / OpenAI
```

## Features Demonstrated

- **RAG**: In-memory knowledge base of research articles, queried by relevance
- **Scratchpad**: Agent working notes with topic outline, key insights, and status tracking
- **Pipeline**: Full 4-step optimization (deduplication, compaction, trimming, reordering)
- **Memory**: Short-term conversation + long-term researcher preferences and prior research
- **Tools**: Paper search, key findings extraction, and citation generation
- **Observability**: Sufficiency checking, quality scoring, mutation history, and timeline tracking
- **Adapters**: Dual-format output for both Anthropic and OpenAI APIs
- **Scoped context**: Agent-isolated context via ContextScope with timeline

## Running

```bash
python examples/research_report_agent/main.py
```
