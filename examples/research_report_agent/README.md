# Research Report Agent

A FastAPI-powered research assistant that uses contextkit for RAG retrieval, scratchpad working notes, full pipeline optimization (dedup, compact, trim, reorder), and dual-provider formatting.

## Setup

```bash
cd examples/research_report_agent

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running

```bash
# From the project root
uvicorn examples.research_report_agent.main:app --reload --port 8001
```

The API docs are available at `http://localhost:8001/docs`.

## API Endpoints

| Method | Path                                    | Description                          |
|--------|-----------------------------------------|--------------------------------------|
| POST   | `/research`                             | Create a new research session        |
| POST   | `/research/{session_id}/query`          | Submit a follow-up query             |
| GET    | `/research/{session_id}/scratchpad`     | View agent's working notes           |
| GET    | `/research/{session_id}/timeline`       | View context evolution over turns    |
| GET    | `/health`                               | Health check with component status   |

## Example Requests

```bash
# Start a research session
curl -X POST http://localhost:8001/research \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "context engineering for LLM-based agents",
    "audience": "technical",
    "max_sources": 5
  }'

# Follow up with a question (use session_id from the response above)
curl -X POST http://localhost:8001/research/abc12345/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best practices for token budget management?"}'

# View the scratchpad
curl http://localhost:8001/research/abc12345/scratchpad

# View the timeline
curl http://localhost:8001/research/abc12345/timeline
```

## Architecture

```
POST /research
    |
    v
ResearchAgent.create_session()
    |
    +-- PromptManager.render()        -> System prompt (topic + audience)
    +-- ShortTermMemory.to_block()    -> Conversation history (last 8 turns)
    +-- LongTermMemory.retrieve()     -> Researcher preferences + prior research
    +-- RAGContext.retrieve()          -> Academic articles from knowledge base
    +-- ToolRegistry.select()         -> Paper search, extraction, citation tools
    +-- ToolRegistry.capture_output() -> Simulated tool results
    +-- Scratchpad.to_block()         -> Agent's working notes + report outline
    |
    v
ContextPipeline.run()
    +-- DeduplicateStep               -> Remove overlapping content
    +-- CompactStep                   -> Compress verbose blocks
    +-- TrimStep                      -> Fit within 4000-token budget
    +-- ReorderStep                   -> U-shaped attention optimization
    |
    v
SufficiencyChecker + QualityScorer   -> Evaluate context quality
    |
    v
AnthropicAdapter.format()            -> Provider-ready payload
```

## Project Structure

```
research_report_agent/
├── main.py             # FastAPI app with routes
├── agent.py            # ResearchAgent class (context assembly logic)
├── models.py           # Pydantic request/response models
├── requirements.txt    # Dependencies
└── README.md
```
