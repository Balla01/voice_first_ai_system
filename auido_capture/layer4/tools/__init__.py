"""
layer4/tools — the tool registry the LLM router chooses among.

build_default_registry() wires the tools the demo ships with. Add a new
capability by writing its executor + Tool and registering it here.
"""

from .base import Tool, ToolCall, ToolResult, ExecutionContext, validate_args
from .registry import ToolRegistry
from .knowledge_base import SEARCH_KNOWLEDGE_BASE
from .refine import REFINE_LAST_ANSWER


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SEARCH_KNOWLEDGE_BASE)
    # refine_last_answer is intentionally NOT registered: refinement is a UI
    # button that calls Layer 5 directly, not something the LLM router decides.
    # REFINE_LAST_ANSWER stays importable for that direct-call path.
    return registry


__all__ = [
    "Tool",
    "ToolCall",
    "ToolResult",
    "ExecutionContext",
    "validate_args",
    "ToolRegistry",
    "SEARCH_KNOWLEDGE_BASE",
    "REFINE_LAST_ANSWER",
    "build_default_registry",
]
