# Multi-Agent Code Review

Three specialized agents collaborate on a code review using isolated context scopes, shared memory, structured handoffs, and pipeline optimization.

## Architecture

```
PR Code Diff (shared memory)
    |
    +-------------------+--------------------+
    |                   |                    |
    v                   v                    v
+------------+   +---------------+   +---------------+
| Architect  |   | Security      |   | Code Reviewer |
| Agent      |   | Analyst       |   | (Synthesizer) |
|            |   |               |   |               |
| - Design   |   | - SQL inject  |   | - Consolidate |
| - Types    |   | - Auth checks |   | - Prioritize  |
| - SRP      |   | - Data expose |   | - Decision    |
| - Errors   |   | - Rate limits |   | - Fix plan    |
+-----+------+   +------+--------+   +-------+-------+
      |                  |                    ^
      |   Publish to     |                    |
      |  Shared Memory   |          Receive   |
      +--------+---------+         Handoffs   |
               |                              |
               +------> Shared Memory --------+
```

## Agent Workflow

1. **Architect Agent**: Reviews code structure, API design, type safety, and error handling. Publishes architecture findings to shared memory.

2. **Security Analyst Agent**: Imports architect findings + code, runs static analysis and tests, identifies vulnerabilities (SQL injection, missing auth, data exposure). Publishes security findings.

3. **Code Reviewer Agent**: Receives handoffs from both agents, synthesizes all findings, prioritizes issues, and produces a final review decision with actionable fix plan.

## Features Demonstrated

- **Scoped contexts**: Each agent has an isolated ContextScope with its own window and scratchpad
- **Shared memory**: Cross-agent block exchange (PR code, findings published and imported)
- **Handoff protocol**: Structured context transfer with metadata between agents
- **Scratchpad**: Per-agent working notes transferred via handoffs
- **RAG**: Coding standards knowledge base queried per agent's focus area
- **Tools**: Static analysis and test runner with captured outputs
- **Pipeline**: Deduplication, trimming, and reordering on the final synthesized context
- **Observability**: Sufficiency checking, quality scoring, and per-agent timeline tracking
- **Long-term memory**: Project configuration and incident history

## Running

```bash
python examples/multi_agent_code_review/main.py
```
