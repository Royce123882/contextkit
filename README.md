# contextkit

A context engineering SDK for building reliable AI agents.

**contextkit** treats the context window as a build artifact. Typed blocks go in, an optimized context window comes out.

## Status

Early development. See [PRD.md](PRD.md) for the full product requirements and phased delivery plan.

## Install

```bash
pip install contextkit
```

## Quick Start

```python
from contextkit import ContextWindow, ContextBlock, BlockType

system = ContextBlock(
    type=BlockType.SYSTEM_PROMPT,
    content="You are a helpful assistant...",
    priority=100,
)

history = ContextBlock(
    type=BlockType.SHORT_TERM_MEMORY,
    content=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ],
    priority=80,
)

window = ContextWindow(max_tokens=128_000)
window.add(system)
window.add(history)

print(window.token_count)
print(window.budget_remaining)
print(window.render())
```

## License

MIT
