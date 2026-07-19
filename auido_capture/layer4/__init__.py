from .models import TriggerAction, TriggerResult, IntentMatch
from .trigger_gate import TriggerGate
from .generation_controller import GenerationController
from .generation_manager import GenerationManager

# Tool-calling router (LAYER4_TRIGGER_MODE=router). The tiered TriggerGate above
# stays as the deterministic fallback / demo-safe default.
from .tool_router import ToolRouter, RouterDecision
from .router_llm import get_router_client, get_fallback_router_client
from .tool_executor import execute_tool_calls
from .tools import ToolRegistry, ToolCall, ToolResult, ExecutionContext, build_default_registry

__all__ = [
    "TriggerAction",
    "TriggerResult",
    "IntentMatch",
    "TriggerGate",
    "GenerationController",
    "GenerationManager",
    # router
    "ToolRouter",
    "RouterDecision",
    "get_router_client",
    "get_fallback_router_client",
    "execute_tool_calls",
    "ToolRegistry",
    "ToolCall",
    "ToolResult",
    "ExecutionContext",
    "build_default_registry",
]
