# Customer Support Agent

A FastAPI-powered customer support chatbot that uses contextkit for RAG-powered knowledge retrieval, tool usage (order lookup, refund processing), conversation memory, and pipeline optimization.

## Setup

```bash
cd examples/customer_support_agent

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
uvicorn examples.customer_support_agent.main:app --reload --port 8000
```

The API docs are available at `http://localhost:8000/docs`.

## API Endpoints

| Method | Path                          | Description                          |
|--------|-------------------------------|--------------------------------------|
| POST   | `/chat`                       | Send a customer message              |
| GET    | `/chat/history/{customer_id}` | Get conversation history             |
| GET    | `/orders/{order_id}`          | Look up an order                     |
| GET    | `/health`                     | Health check with component status   |

## Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust-001",
    "message": "I received order ORD-98765 but the headphones are damaged. Can I get a refund?"
  }'
```

## Architecture

```
POST /chat
    |
    v
SupportAgent.handle_message()
    |
    +-- PromptManager.render()        -> System prompt with customer profile
    +-- ShortTermMemory.to_block()    -> Conversation history (last 10 turns)
    +-- LongTermMemory.retrieve()     -> Customer profile + interaction history
    +-- RAGContext.retrieve()          -> Relevant support articles
    +-- ToolRegistry.select()         -> Applicable tools for the query
    +-- ToolRegistry.capture_output() -> Simulated order lookup results
    |
    v
ContextPipeline.run()
    +-- DeduplicateStep               -> Remove overlapping content
    +-- TrimStep                      -> Fit within 4000-token budget
    +-- ReorderStep                   -> Optimize for LLM attention
    |
    v
SufficiencyChecker + QualityScorer   -> Evaluate context quality
    |
    v
AnthropicAdapter.format()            -> Provider-ready payload
```

## Project Structure

```
customer_support_agent/
├── main.py             # FastAPI app with routes
├── agent.py            # SupportAgent class (context assembly logic)
├── models.py           # Pydantic request/response models
├── requirements.txt    # Dependencies
└── README.md
```
