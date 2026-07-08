"""
layer3/tokens.py — token counting and budget constants.

Isolated from context_window.py so the budget numbers and the counting
mechanism can be swapped independently (e.g. if a different encoding is
needed later) without touching the deque/eviction logic.
"""

import os
import logging

import tiktoken

logger = logging.getLogger("insureassist.layer3")

# Overridable via .env for testing (e.g. to force epoch compaction to fire
# after a handful of turns instead of hundreds). Defaults are the real
# production values — leave LAYER3_TOKEN_BUDGET / LAYER3_OUTPUT_RESERVE
# unset (or delete them from .env) to run with the real budget again.
TOKEN_BUDGET = int(os.getenv("LAYER3_TOKEN_BUDGET", "3000"))
OUTPUT_RESERVE = int(os.getenv("LAYER3_OUTPUT_RESERVE", "2000"))
AVAILABLE_FOR_CONTEXT = TOKEN_BUDGET - OUTPUT_RESERVE

if TOKEN_BUDGET != 6000 or OUTPUT_RESERVE != 2000:
    logger.warning(
        f"Layer 3 token budget overridden via env: TOKEN_BUDGET={TOKEN_BUDGET}, "
        f"OUTPUT_RESERVE={OUTPUT_RESERVE} -> AVAILABLE_FOR_CONTEXT={AVAILABLE_FOR_CONTEXT}. "
        "This is meant for testing epoch compaction, not production use."
    )

try:
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    # tiktoken downloads its BPE file from openaipublic.blob.core.windows.net on
    # first use — some corporate networks / sandboxes block that domain. Rather
    # than hard-crash the whole app on import, fall back to a rough
    # chars-per-token approximation so budgeting still works, just less precisely.
    logger.warning(
        f"tiktoken cl100k_base encoding unavailable ({e}); falling back to an "
        "approximate chars/4 token counter. Fix network access to the tiktoken "
        "blob storage for accurate counts."
    )
    _encoder = None


def count_tokens(text: str) -> int:
    if _encoder is not None:
        return len(_encoder.encode(text))
    return max(1, len(text) // 4) if text else 0