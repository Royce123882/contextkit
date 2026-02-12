# Multi-Agent Code Review

A FastAPI-powered code review service where three specialized agents (Architect, Security Analyst, Code Reviewer) collaborate using contextkit for isolated context scopes, shared memory, structured handoffs, and pipeline optimization.

## Setup

```bash
cd examples/multi_agent_code_review

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
uvicorn examples.multi_agent_code_review.main:app --reload --port 8002
```

The API docs are available at `http://localhost:8002/docs`.

## API Endpoints

| Method | Path                  | Description                          |
|--------|-----------------------|--------------------------------------|
| POST   | `/review`             | Submit code for multi-agent review   |
| GET    | `/review/{review_id}` | Get review summary                   |
| GET    | `/health`             | Health check with component status   |

## Example Request

```bash
curl -X POST http://localhost:8002/review \
  -H "Content-Type: application/json" \
  -d '{
    "code": "@router.put(\"/users/{user_id}\")\nasync def update_profile(user_id: int, data: dict):\n    query = f\"UPDATE users SET name='"'"'{data[\"name\"]}'"'"' WHERE id={user_id}\"\n    await db.execute(query)\n    return {\"status\": \"updated\"}",
    "description": "PR #342: Add user profile update endpoint"
  }'
```

## Agent Workflow

```
POST /review
    |
    v
SharedMemory.publish("pr_code")    -> Code available to all agents
    |
    +--- Agent 1: Architect ---+
    |   ContextScope (isolated) |
    |   RAG: coding standards   |
    |   Scratchpad: design notes|
    |   -> Publish findings     |
    +---------------------------+
    |
    +--- Agent 2: Security ----+
    |   ContextScope (isolated) |
    |   Import: architect work  |
    |   Tools: static analysis  |
    |   Tools: test runner      |
    |   Scratchpad: vuln notes  |
    |   -> Publish findings     |
    +---------------------------+
    |
    +--- Agent 3: Synthesizer -+
    |   ContextScope (isolated) |
    |   Receive: both handoffs  |
    |   Import: original code   |
    |   Pipeline: dedup + trim  |
    |   -> Decision + fix plan  |
    +---------------------------+
    |
    v
  Consolidated ReviewResult
```

## Project Structure

```
multi_agent_code_review/
├── main.py             # FastAPI app with routes + initialization
├── agents.py           # Three agent functions (architect, security, synthesizer)
├── models.py           # Pydantic request/response models
├── requirements.txt    # Dependencies
└── README.md
```
