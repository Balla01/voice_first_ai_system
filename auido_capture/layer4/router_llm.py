"""
layer4/router_llm.py — provider-agnostic tool-calling client for the router.

Kept separate from layer3/epoch_compaction.py's LLMClient on purpose: that
protocol is text-in/text-out (create_message -> str) for summarization, while
the router needs *structured tool calls* back. Tool-calling request/response
shapes differ enough per provider that a dedicated protocol is honest rather
than bolting a second method onto the Layer 3 one.

Phase-0 finding drives the defaults: qwen3.5:cloud (and qwen3.5:397b) are
paywalled (403) on the current Ollama account. The router's original default,
qwen3-coder-next:cloud, has since been retired from Ollama Cloud's catalog
entirely (returns 410 Gone, confirmed via GET /v1/models). gemma4:cloud is the
current working free tag (confirmed 200 + correct tool_calls, ~900ms p50) and
matches the tag Layer 3's OllamaClientAdapter already uses — see
layer3/epoch_compaction.py. Re-check GET https://ollama.com/v1/models if this
breaks again; Ollama's free cloud catalog rotates.

On any primary-router failure (this call erroring, not just a timeout),
get_fallback_router_client() builds a second, independent LLM (Gemini by
default) so a single Ollama outage doesn't silently drop every trigger down to
the deterministic TriggerGate — see tool_router.py's fallback chain.
LAYER4_ROUTER_PROVIDER=anthropic remains available to swap the *primary* to
Haiku (native tool-use) with no router-logic changes.

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
DEFAULT_OLLAMA_MODEL = "gemma4:cloud"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"


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


# Gemini's google.genai.types.Schema (as of the pinned SDK 0.3.0) declares
# these length/count constraints as `str`, not `int` — a protobuf int64-as-
# string quirk — and rejects a plain JSON-Schema int with a pydantic
# ValidationError. minimum/maximum are the one exception: those stay float.
_GEMINI_STRINGIFIED_INT_KEYS = {"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"}


def _to_gemini_schema(js: dict) -> dict:
    """JSON-Schema (as used by the tool registry / OpenAI-shaped tools) uses
    lowercase types ("object", "string"); Gemini's Schema requires uppercase
    ("OBJECT", "STRING") and rejects lowercase with a pydantic ValidationError.
    Recurses into properties/items only — the other JSON-Schema keys
    (description, required, enum, ...) pass through unchanged."""
    if not isinstance(js, dict):
        return js
    out = {}
    for k, v in js.items():
        if k == "type" and isinstance(v, str):
            out[k] = v.upper()
        elif k == "properties" and isinstance(v, dict):
            out[k] = {pk: _to_gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _to_gemini_schema(v)
        elif k in _GEMINI_STRINGIFIED_INT_KEYS and isinstance(v, int):
            out[k] = str(v)
        else:
            out[k] = v
    return out


class GeminiToolClient:
    """Gemini native function-calling. Used as the *fallback* router (a
    provider/network path independent of Ollama) rather than the primary —
    see get_fallback_router_client(). Same RouterLLMResponse shape out, so
    ToolRouter doesn't know which provider it's talking to."""

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL):
        from google import genai  # pip install google-genai
        from google.genai import types
        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model = model

    async def create_with_tools(
        self, system: str, user_content: str, tools: list, temperature: float = 0.0
    ) -> RouterLLMResponse:
        declarations = [
            self._types.FunctionDeclaration(
                name=t["function"]["name"],
                description=t["function"].get("description", ""),
                parameters=_to_gemini_schema(t["function"].get("parameters", {})),
            )
            for t in tools
        ]
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=user_content,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                tools=[self._types.Tool(function_declarations=declarations)],
                temperature=temperature,
            ),
        )
        calls, text = [], ""
        parts = response.candidates[0].content.parts if response.candidates else []
        for part in parts:
            if part.function_call:
                calls.append(RawToolCall(name=part.function_call.name, arguments_raw=json.dumps(dict(part.function_call.args))))
            elif part.text:
                text += part.text
        return RouterLLMResponse(tool_calls=calls, text=text, model=self.model, raw={})

    async def close(self) -> None:
        pass   # genai.Client has no explicit close/session to release


def get_router_client() -> RouterLLMClient:
    """
    LAYER4_ROUTER_PROVIDER (ollama | anthropic) + LAYER4_ROUTER_MODEL select
    the router's LLM. Independent of Layer 3's LLM_PROVIDER — the router and
    the summarizer can use different models. Defaults to Ollama /
    gemma4:cloud (the current working free tag — see module docstring).
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


def get_fallback_router_client() -> Optional[RouterLLMClient]:
    """
    Second, independent LLM used only when the primary router call itself
    errors (network/outage/model retired) — see tool_router.py's fallback
    chain. Deliberately a different provider than the default primary
    (Ollama) so the two don't share a single point of failure.

    LAYER4_FALLBACK_PROVIDER (gemini | anthropic | ollama, default gemini) +
    LAYER4_FALLBACK_MODEL select it. Returns None (fallback LLM disabled,
    ToolRouter drops straight to the deterministic TriggerGate on primary
    failure) if the required key isn't set — this is a soft dependency, not
    a hard requirement.
    """
    provider = os.getenv("LAYER4_FALLBACK_PROVIDER", "gemini").lower()

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            logger.info("LAYER4_FALLBACK_PROVIDER=gemini but GEMINI_API_KEY is not set; fallback LLM disabled")
            return None
        model = os.getenv("LAYER4_FALLBACK_MODEL", DEFAULT_GEMINI_MODEL)
        return GeminiToolClient(api_key=key, model=model)

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            logger.info("LAYER4_FALLBACK_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set; fallback LLM disabled")
            return None
        model = os.getenv("LAYER4_FALLBACK_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return AnthropicToolClient(api_key=key, model=model)

    if provider == "ollama":
        key = os.getenv("OLLAMA_API_KEY", "")
        if not key:
            logger.info("LAYER4_FALLBACK_PROVIDER=ollama but OLLAMA_API_KEY is not set; fallback LLM disabled")
            return None
        model = os.getenv("LAYER4_FALLBACK_MODEL", DEFAULT_OLLAMA_MODEL)
        return OllamaToolClient(api_key=key, model=model)

    raise RuntimeError(f"Unknown LAYER4_FALLBACK_PROVIDER: {provider!r} (expected gemini|anthropic|ollama)")
