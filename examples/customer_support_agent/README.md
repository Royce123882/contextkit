# Customer Support Agent

An end-to-end customer support agent that combines RAG-powered knowledge retrieval, tool usage for order lookups and refund processing, conversation memory, and escalation logic.

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────┐
│   Context Assembly       │
│                          │
│  ┌────────────────────┐  │
│  │ System Prompt      │  │  ← Versioned via PromptManager
│  │ (priority=100)     │  │
│  ├────────────────────┤  │
│  │ Conversation       │  │  ← ShortTermMemory (last 10 turns)
│  │ History            │  │
│  │ (priority=85)      │  │
│  ├────────────────────┤  │
│  │ Customer Profile   │  │  ← LongTermMemory (persistent)
│  │ (priority=80)      │  │
│  ├────────────────────┤  │
│  │ Knowledge Base     │  │  ← RAG retrieval from help docs
│  │ (priority=70)      │  │
│  ├────────────────────┤  │
│  │ Tool Definitions   │  │  ← Order lookup, refund, escalate
│  │ (priority=60)      │  │
│  ├────────────────────┤  │
│  │ Tool Outputs       │  │  ← Results from executed tools
│  │ (priority=75)      │  │
│  └────────────────────┘  │
│                          │
│  Pipeline: Dedup → Trim  │  ← Optimize to fit budget
└──────────────────────────┘
    │
    ▼
  Format for Anthropic / OpenAI
```

## Features Demonstrated

- **RAG**: In-memory knowledge base of support articles, queried for each user message
- **Tools**: Order lookup, refund processing, and human escalation with captured outputs
- **Memory**: Short-term conversation history + long-term customer profile
- **Pipeline**: Deduplication and trimming to stay within token budget
- **Observability**: Context inspection, sufficiency checking, and quality scoring
- **Adapters**: Format the final context for Anthropic or OpenAI

## Running

```bash
python examples/customer_support_agent/main.py
```
