"""
layer4/router_llm.py — provider-agnostic tool-calling client for the router.

Kept separate from layer3/epoch_compaction.py's LLMClient on purpose: that
protocol is text-in/text-out (create_message -> str) for summarization, while
the router needs *structured tool calls* back. Tool-calling request/response
shapes differ enough per provider that a dedicated protocol is honest rather
than bolting a second method onto the Layer 3 one.

Phase-0 finding drives the defaults: qwen3.5:cloud is paywalled (403) on the
current Ollama account and the general models need an upgrade, so the working
default is qwen3-coder-next:cloud (probed 200, ~90% tool accuracy, ~914ms p50).
Latency is a known follow-up, not a blocker — the whole point of this
abstraction is that LAYER4_ROUTER_PROVIDER=anthropic swaps in Haiku (native
tool-use, faster) with no router-logic changes when we're ready.

TLS note: on a corporate TLS-inspecting proxy the cert chain is self-signed.
Honor OLLAMA_CA_BUNDLE / REQUESTS_CA_BUNDLE / SSL_CERT_FILE for the real
corporate CA; LAYER4_ROUTER_INSECURE_SSL=1 is a dev-only bypass (do NOT ship).
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import httpx

logger = logging.getLogger("insureassist.layer4")

OLLAMA_BASE_URL = "https://ollama.com/v1"
DEFAULT_OLLAMA_MODEL = "qwen3-coder-next:cloud"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class RawToolCall:
    """A tool call as the model emitted it — arguments still a raw JSON string,
    NOT yet validated. The router parses + validates these into tools.ToolCall."""
    name: str
    arguments_raw: str


@dataclass
class RouterLLMResponse:
    tool_calls: List[RawToolCall] = field(default_factory=list)
    text: str = ""          # any assistant text alongside/instead of tool calls
    model: str = ""
    raw: dict = field(default_factory=dict)


class RouterLLMClient(Protocol):
    model: str
    async def create_with_tools(
        self, system: str, user_content: str, tools: list, temperature: float = 0.0
    ) -> RouterLLMResponse: ...
    async def close(self) -> None: ...


def _verify_setting():
    """Corporate proxy support. Explicit dev bypass wins over a stray
    SSL_CERT_FILE (e.g. anaconda's), otherwise verify against a provided CA
    bundle, otherwise default system verification."""
    if os.getenv("LAYER4_ROUTER_INSECURE_SSL", "").lower() in ("1", "true", "yes"):
        logger.warning("Router LLM: TLS verification DISABLED (LAYER4_ROUTER_INSECURE_SSL) — dev only")
        return False
    ca = os.getenv("OLLAMA_CA_BUNDLE") or os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    if ca and os.path.exists(ca):
        return ca
    return True


class OllamaToolClient:
    """Ollama Cloud via its OpenAI-compatible /v1/chat/completions endpoint,
    using the native `tools` parameter (real function-calling, not
    JSON-parse-and-hope)."""

    def __init__(self, api_key: str, model: str = DEFAULT_OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(45.0),
            verify=_verify_setting(),
        )

    async def create_with_tools(
        self, system: str, user_content: str, tools: list, temperature: float = 0.0
    ) -> RouterLLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
        }
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        calls = [
            RawToolCall(name=c["function"]["name"], arguments_raw=c["function"].get("arguments", "{}"))
            for c in (message.get("tool_calls") or [])
        ]
        return RouterLLMResponse(
            tool_calls=calls,
            text=message.get("content") or "",
            model=self.model,
            raw=data,
        )

    async def close(self) -> None:
        await self._client.aclose()


class AnthropicToolClient:
    """Anthropic native tool-use — the swap target (Haiku 4.5). Same
    RouterLLMResponse shape out, so the router doesn't know which provider it's
    talking to."""

    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL, max_tokens: int = 512):
        import anthropic  # lazy import — only needed if this provider is selected
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self._max_tokens = max_tokens

    async def create_with_tools(
        self, system: str, user_content: str, tools: list, temperature: float = 0.0
    ) -> RouterLLMResponse:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            tools=tools,
            tool_choice={"type": "auto"},
            temperature=temperature,
        )
        calls, text = [], ""
        for block in response.content:
            if block.type == "tool_use":
                calls.append(RawToolCall(name=block.name, arguments_raw=json.dumps(block.input)))
            elif block.type == "text":
                text += block.text
        return RouterLLMResponse(
            tool_calls=calls,
            text=text,
            model=self.model,
            raw={"stop_reason": response.stop_reason},
        )

    async def close(self) -> None:
        await self._client.close()


def get_router_client() -> RouterLLMClient:
    """
    LAYER4_ROUTER_PROVIDER (ollama | anthropic) + LAYER4_ROUTER_MODEL select
    the router's LLM. Independent of Layer 3's LLM_PROVIDER — the router and
    the summarizer can use different models. Defaults to Ollama /
    qwen3-coder-next:cloud (the working Phase-0 tag).
    """
    provider = os.getenv("LAYER4_ROUTER_PROVIDER", "ollama").lower()

    if provider == "ollama":
        key = os.getenv("OLLAMA_API_KEY", "")
        if not key:
            raise RuntimeError("LAYER4_ROUTER_PROVIDER=ollama but OLLAMA_API_KEY is not set")
        model = os.getenv("LAYER4_ROUTER_MODEL", DEFAULT_OLLAMA_MODEL)
        base_url = os.getenv("OLLAMA_ROUTER_BASE_URL", OLLAMA_BASE_URL)
        return OllamaToolClient(api_key=key, model=model, base_url=base_url)

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("LAYER4_ROUTER_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        model = os.getenv("LAYER4_ROUTER_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return AnthropicToolClient(api_key=key, model=model)

    raise RuntimeError(f"Unknown LAYER4_ROUTER_PROVIDER: {provider!r} (expected ollama|anthropic)")
