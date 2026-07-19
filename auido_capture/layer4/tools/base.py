"""
layer4/tools/base.py — core abstractions for the tool-calling router.

A Tool is the unit the LLM router chooses among: a name, a human/LLM-facing
description, a JSON-schema for its arguments, and an async executor. RAG is
just the first tool (search_knowledge_base); adding another tool is one
registry entry + one executor, nothing else in the router changes.

Kept dependency-free (same reasoning as layer3/models.py and layer4/models.py):
importable and unit-testable without pulling in httpx, an LLM SDK, or Layer 5.
The JSON-schema validator here is intentionally a minimal subset (object +
typed properties + required + minLength) — enough to catch the malformed /
hallucinated args an LLM can emit even with real function-calling, without
adding a `jsonschema` dependency the repo doesn't otherwise use.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Tuple


@dataclass
class ExecutionContext:
    """Everything a tool executor needs that isn't in its own args. Passed in
    at execution time rather than closed over, so tools stay stateless and
    testable."""
    session_id: str
    layer5_client: Any = None          # Layer5Client (duck-typed to avoid the import here)
    customer_id: Optional[str] = None


@dataclass
class ToolResult:
    ok: bool
    tool: str
    query: str = ""        # what was searched / the refine instruction — for the UI + audit
    answer: str = ""       # answer text to surface to the agent
    error: str = ""
    meta: dict = field(default_factory=dict)   # timing etc. from downstream


@dataclass
class ToolCall:
    """A single tool invocation the router decided on, post-validation."""
    name: str
    arguments: dict = field(default_factory=dict)
    raw_arguments: str = ""   # the raw string the model emitted, kept for the audit trail


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON schema (an "object" schema)
    executor: Callable[[dict, ExecutionContext], Awaitable[ToolResult]]

    def to_openai_schema(self) -> dict:
        """OpenAI-compatible tools[] entry — the shape Ollama Cloud's /v1
        endpoint (and OpenAI) expect."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict:
        """Anthropic tool-use entry — for the Haiku swap target."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def execute(self, args: dict, ctx: ExecutionContext) -> ToolResult:
        return await self.executor(args, ctx)


_JSON_PY_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_args(schema: dict, args: Any) -> Tuple[bool, str]:
    """
    Minimal JSON-schema validation. Returns (ok, error_message). Covers the
    subset our tool schemas actually use:
      - top-level must be an object
      - every `required` key present
      - declared property `type` matches
      - `minLength` on strings (checked against the trimmed value, so a
        whitespace-only query counts as empty)
    Anything not declared is left alone (extra keys allowed) — we only reject
    what's demonstrably wrong, not what's merely unfamiliar.
    """
    if not isinstance(args, dict):
        return False, f"arguments must be an object, got {type(args).__name__}"

    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            return False, f"missing required argument '{req}'"

    for key, value in args.items():
        spec = props.get(key)
        if not spec:
            continue
        expected = spec.get("type")
        py_type = _JSON_PY_TYPES.get(expected)
        if py_type and not isinstance(value, py_type):
            return False, f"argument '{key}' expected {expected}, got {type(value).__name__}"
        if expected == "string" and isinstance(value, str):
            min_len = spec.get("minLength")
            if min_len is not None and len(value.strip()) < min_len:
                return False, f"argument '{key}' is empty/too short (minLength={min_len})"

    return True, ""
