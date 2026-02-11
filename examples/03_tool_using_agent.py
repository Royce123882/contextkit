"""Example 3: Tool-Using Agent with Dynamic Selection.

Demonstrates tool registration, dynamic selection based on user
queries, tool output capture, and context assembly with tool
definitions and outputs.

Usage:
    python examples/03_tool_using_agent.py
"""

import asyncio
import json

from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.observe.inspect import inspect_window
from contextkit.prompts.prompt_manager import PromptManager
from contextkit.tools.tool_registry import ToolRegistry


async def main() -> None:
    """Run a tool-using agent example."""
    # ---------------------------------------------------------------
    # Step 1: Register tools in the registry
    # ---------------------------------------------------------------
    registry = ToolRegistry()

    registry.register(
        name="get_weather",
        description="Get the current weather for a location. Returns "
        "temperature, conditions, and humidity.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            },
            "required": ["location"],
        },
        tags=["weather", "external-api"],
    )

    registry.register(
        name="search_web",
        description="Search the web for information. Returns a list "
        "of relevant results with titles and snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        tags=["search", "external-api"],
    )

    registry.register(
        name="calculate",
        description="Evaluate a mathematical expression. Supports "
        "basic arithmetic, exponents, and common functions.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate (e.g., '2 + 3 * 4')",
                },
            },
            "required": ["expression"],
        },
        tags=["math", "local"],
    )

    registry.register(
        name="read_file",
        description="Read the contents of a file from the local filesystem.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path",
                },
            },
            "required": ["path"],
        },
        tags=["filesystem", "local"],
    )

    print(f"Registered {registry.tool_count} tools")
    print(f"Tools: {registry.list_tools()}")

    # ---------------------------------------------------------------
    # Step 2: Dynamic tool selection based on user query
    # ---------------------------------------------------------------
    user_query = "What's the weather like in Tokyo today?"

    # Select relevant tools for this query (returns ContextBlocks)
    tool_blocks = registry.select(user_query, max_tools=2)
    print(f"\nQuery: '{user_query}'")
    print(f"Selected tools: {[b.name for b in tool_blocks]}")

    # ---------------------------------------------------------------
    # Step 3: Simulate tool execution and capture output
    # ---------------------------------------------------------------
    # Simulate calling get_weather
    weather_result = {
        "location": "Tokyo",
        "temperature": 22,
        "units": "celsius",
        "conditions": "Partly cloudy",
        "humidity": 65,
    }

    # Capture the tool output
    tool_output = registry.capture_output(
        tool_name="get_weather",
        result=json.dumps(weather_result),
        latency_ms=120.5,
    )
    print(f"\nTool output captured: {tool_output.tool_name}")
    print(f"  Duration: {tool_output.latency_ms}ms")

    # Convert tool output to a context block
    output_block = tool_output.to_block()

    # ---------------------------------------------------------------
    # Step 4: Assemble everything into a context window
    # ---------------------------------------------------------------
    window = ContextWindow(model="claude-sonnet-4-5-20250929")

    # System prompt
    prompt_mgr = PromptManager()
    prompt_mgr.register(
        "tool_agent",
        "You are an AI assistant with access to tools. "
        "Use the provided tool definitions to decide which tools "
        "to call. After receiving tool results, synthesize them "
        "into a helpful response.",
    )
    system_block = prompt_mgr.render("tool_agent")
    window.add(system_block)

    # Add tool definitions
    for block in tool_blocks:
        window.add(block)

    # Add user message
    user_block = ContextBlock(
        type=BlockType.USER_CONTEXT,
        content=user_query,
        priority=90,
        name="user_message",
    )
    window.add(user_block)

    # Add tool output
    window.add(output_block)

    # ---------------------------------------------------------------
    # Step 5: Inspect the assembled context
    # ---------------------------------------------------------------
    print("\n--- Context Window ---")
    print(inspect_window(window))

    print(f"\nTotal: {len(window.blocks)} blocks, {window.token_count:,} tokens")

    # ---------------------------------------------------------------
    # Step 6: Show tool output metadata
    # ---------------------------------------------------------------
    print("\nTool output details:")
    print(f"  Tool: {tool_output.tool_name}")
    print(f"  Result: {tool_output.result[:80]}...")
    print(f"  Called at: {tool_output.called_at.isoformat()}")
    print(f"  Latency: {tool_output.latency_ms}ms")


if __name__ == "__main__":
    asyncio.run(main())
