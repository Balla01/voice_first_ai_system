"""
layer4/tools/registry.py — the set of tools the router is allowed to call.

One ToolRegistry instance holds every tool. The router asks it for the
provider-shaped schemas to send to the LLM, and looks tools up by name when a
call comes back. Adding a capability = register(one Tool) here.
"""

from typing import List, Optional

from .base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def openai_schemas(self) -> List[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def anthropic_schemas(self) -> List[dict]:
        return [t.to_anthropic_schema() for t in self._tools.values()]
