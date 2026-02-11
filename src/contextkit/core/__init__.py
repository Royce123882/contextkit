"""Core data structures for contextkit.

Contains the fundamental building blocks: BlockType enum, ContextBlock
data model, ContextWindow container, and BudgetExceeded exception.
"""

from contextkit.core.block import BlockType, BudgetExceededError, ContextBlock
from contextkit.core.context_window import ContextWindow

__all__ = [
    "BlockType",
    "BudgetExceededError",
    "ContextBlock",
    "ContextWindow",
]
