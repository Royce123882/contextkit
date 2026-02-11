"""Public token counting utilities.

Standalone module for token counting and budget checking,
usable without the full ContextWindow framework.
"""

from contextkit.utils.token_counting import count, fits_budget

__all__ = ["count", "fits_budget"]
