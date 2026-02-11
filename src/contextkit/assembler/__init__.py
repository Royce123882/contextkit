"""Context assembler for composing blocks into a window.

The assembler takes a list of ContextBlocks and composes them into a
ContextWindow with ordering and priority rules. It produces an
AssemblyReport that powers explain() for understanding why blocks
were included or excluded.
"""

from contextkit.assembler.context_assembler import ContextAssembler
from contextkit.assembler.report import AssemblyReport, BlockDecision

__all__ = [
    "AssemblyReport",
    "BlockDecision",
    "ContextAssembler",
]
