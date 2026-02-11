"""Example 1: Basic Chatbot with Context Engineering.

Demonstrates how to build a chatbot that manages system prompts,
conversation history, and budget-aware context assembly.

Usage:
    python examples/01_basic_chatbot.py
"""

from contextkit.adapters.anthropic_adapter import AnthropicAdapter
from contextkit.adapters.openai_adapter import OpenAIAdapter
from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.memory.short_term import ShortTermMemory
from contextkit.observe.inspect import inspect_window
from contextkit.prompts.manager import PromptManager


def main() -> None:
    """Run a basic chatbot example with context engineering."""
    # ---------------------------------------------------------------
    # Step 1: Create a context window with a token budget
    # ---------------------------------------------------------------
    window = ContextWindow(model="claude-sonnet-4-5-20250929")
    print(f"Model: {window.model_name}")
    print(f"Token budget: {window.max_tokens:,}")

    # ---------------------------------------------------------------
    # Step 2: Define system prompts using the PromptManager
    # ---------------------------------------------------------------
    prompt_manager = PromptManager()

    # Register a versioned system prompt template
    prompt_manager.register(
        "assistant_v1",
        "You are {{role}}, a helpful AI assistant.\n\nRules:\n- {{rules}}",
        description="Basic assistant prompt",
    )

    # Render the prompt with variables into a ContextBlock
    system_block = prompt_manager.render(
        "assistant_v1",
        role="Aria",
        rules="Be concise. Use examples when helpful.",
    )
    window.add(system_block)
    print(f"\nSystem prompt added ({system_block.token_count} tokens)")

    # ---------------------------------------------------------------
    # Step 3: Manage conversation history with ShortTermMemory
    # ---------------------------------------------------------------
    conversation = ShortTermMemory(
        strategy="sliding_window",
        max_turns=10,  # Keep last 10 messages
    )

    # Simulate a conversation
    conversation.add_turn("user", "What is context engineering?")
    conversation.add_turn(
        "assistant",
        "Context engineering is the practice of designing and managing "
        "the information (context) that flows into an LLM's prompt. "
        "It includes system prompts, conversation history, RAG chunks, "
        "tool outputs, and more.",
    )
    conversation.add_turn("user", "How does it differ from prompt engineering?")
    conversation.add_turn(
        "assistant",
        "Prompt engineering focuses on crafting individual prompts. "
        "Context engineering is broader -- it treats the entire context "
        "window as a system to design: what goes in, what priority each "
        "part has, and how to stay within token budgets.",
    )
    conversation.add_turn("user", "Can you give me an example?")

    # Convert conversation to a context block
    history_block = conversation.to_block()
    window.add(history_block)
    print(
        f"Conversation added ({history_block.token_count} tokens, "
        f"{conversation.turn_count} turns)"
    )

    # ---------------------------------------------------------------
    # Step 4: Add user-specific context
    # ---------------------------------------------------------------
    user_context = ContextBlock(
        type=BlockType.USER_CONTEXT,
        content="User timezone: PST. Preferred language: English. "
        "Experience level: Intermediate developer.",
        priority=70,
        name="user_preferences",
    )
    window.add(user_context)

    # ---------------------------------------------------------------
    # Step 5: Inspect the context window
    # ---------------------------------------------------------------
    print("\n--- Context Window Summary ---")
    print(inspect_window(window))

    # ---------------------------------------------------------------
    # Step 6: Format for different providers
    # ---------------------------------------------------------------
    print(f"\nTotal tokens: {window.token_count:,}")
    print(f"Budget used: {window.token_count / window.max_tokens * 100:.1f}%")

    # Format for Anthropic
    anthropic_payload = AnthropicAdapter().format(window)
    print(f"\nAnthropic payload keys: {list(anthropic_payload.keys())}")
    print(f"System prompt length: {len(anthropic_payload.get('system', ''))}")
    print(f"Messages count: {len(anthropic_payload['messages'])}")

    # Format for OpenAI
    openai_payload = OpenAIAdapter().format(window)
    print(f"\nOpenAI payload keys: {list(openai_payload.keys())}")
    print(f"Messages count: {len(openai_payload['messages'])}")


if __name__ == "__main__":
    main()
