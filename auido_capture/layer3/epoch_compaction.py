"""
layer3/epoch_compaction.py — Step 3: Overflow -> Epoch Compaction.

Wraps the two Claude calls this layer needs:
  - compact_turns():     summarize the oldest 500 raw turns into one summary
  - compact_summaries(): meta-compact the oldest 2 summaries into one, when
                          the max-10-summaries cap would otherwise be exceeded

The Anthropic client is injected (constructor param) rather than imported
and instantiated inline, so tests can pass a fake client with no network
calls. Model choice: Haiku 4.5 — this is a small, frequent, cheap
summarization task, not a reasoning-heavy one, so the fastest/cheapest
current model is the right fit.
"""

import os
import logging
from typing import List, Protocol

from .models import Turn, EpochSummary

logger = logging.getLogger("insureassist.layer3")

MAX_SUMMARY_TOKENS = 300

# One place to change a default model per provider, without touching call sites.
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "ollama": "gemma4:cloud",   # confirmed working Ollama Cloud tag as of this writing
}


class LLMClient(Protocol):
    """Every provider adapter implements just this. EpochCompactor only ever
    talks to this shape — swapping providers means swapping which adapter
    gets constructed, nothing else in Layer 3 changes."""
    async def create_message(self, max_tokens: int, prompt: str) -> str: ...


class AnthropicClientAdapter:
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["anthropic"]):
        import anthropic  # lazy import — tests never need the package installed
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def create_message(self, max_tokens: int, prompt: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIClientAdapter:
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["openai"]):
        import openai  # pip install openai
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = model

    async def create_message(self, max_tokens: int, prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class GeminiClientAdapter:
    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["gemini"]):
        from google import genai  # pip install google-genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def create_message(self, max_tokens: int, prompt: str) -> str:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config={"max_output_tokens": max_tokens},
        )
        return response.text


class OllamaClientAdapter:
    """Points at Ollama Cloud (https://ollama.com) using an API key by
    default. To use a local Ollama server instead, pass
    host="http://localhost:11434" and no Authorization header is needed —
    local Ollama has no auth (see .env.example note)."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["ollama"], host: str = "https://ollama.com"):
        from ollama import AsyncClient  # pip install ollama
        self._client = AsyncClient(host=host, headers={"Authorization": f"Bearer {api_key}"})
        self._model = model

    async def create_message(self, max_tokens: int, prompt: str) -> str:
        # Reasoning/"thinking" models handle the think switch differently per
        # family — gpt-oss requires the string "low"/"medium"/"high" (a plain
        # boolean is ignored), while most other reasoning models (including
        # Gemma 4) accept a boolean. We don't need chain-of-thought for a
        # summarization task on ANY model, so this just picks whichever form
        # turns thinking off/minimal for the model actually configured,
        # rather than hardcoding one family's quirk.
        think_param = "low" if self._model.startswith("gpt-oss") else False

        chat_kwargs = dict(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": max_tokens + 300},  # headroom in case thinking can't be fully disabled
        )
        try:
            response = await self._client.chat(think=think_param, **chat_kwargs)
        except TypeError as e:
            # Older `ollama` python client versions don't accept the `think`
            # kwarg (AsyncClient.chat() got an unexpected keyword argument
            # 'think'). Retry without it rather than failing the whole call.
            if "think" not in str(e):
                raise
            response = await self._client.chat(**chat_kwargs)
        content = response.message.content
        if not content:
            # Defensive fallback: if content still comes back empty (e.g. the
            # thinking trace ran long, or this model ignored think= entirely),
            # surface the reasoning trace instead of silently storing an
            # empty summary.
            logger.warning(
                f"Ollama ({self._model}) response.message.content was empty; falling back to "
                "thinking trace. Consider raising num_predict further if this recurs."
            )
            content = (response.message.thinking or "").strip()
        return content


def get_llm_client() -> LLMClient:
    """
    Reads LLM_PROVIDER from the environment (anthropic | openai | gemini |
    ollama) and builds the matching adapter. THIS is the one switch — change
    LLM_PROVIDER in .env and restart, nothing else in the codebase changes.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        return AnthropicClientAdapter(api_key=key)

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAIClientAdapter(api_key=key)

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
        return GeminiClientAdapter(api_key=key)

    if provider == "ollama":
        key = os.getenv("OLLAMA_API_KEY", "")
        host = os.getenv("OLLAMA_HOST", "https://ollama.com")
        model = os.getenv("OLLAMA_MODEL", DEFAULT_MODELS["ollama"])
        if not key:
            raise RuntimeError("LLM_PROVIDER=ollama but OLLAMA_API_KEY is not set")
        return OllamaClientAdapter(api_key=key, model=model, host=host)

    raise RuntimeError(f"Unknown LLM_PROVIDER: {provider!r} (expected anthropic|openai|gemini|ollama)")


class EpochCompactor:
    def __init__(self, client: LLMClient):
        self._client = client

    async def compact_turns(self, turns: List[Turn]) -> str:
        conversation_text = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
        prompt = (
            "Summarize this segment of an insurance sales conversation concisely. "
            "Preserve concrete facts: names, ages, numbers, plan names, decisions made, "
            "and any concerns raised. Do not include commentary or headers, just the summary.\n\n"
            f"{conversation_text}"
        )
        return await self._client.create_message(MAX_SUMMARY_TOKENS, prompt)

    async def compact_summaries(self, summaries: List[EpochSummary]) -> str:
        combined_text = "\n".join(f"- {s.text}" for s in summaries)
        prompt = (
            "These are sequential summaries of earlier parts of the same insurance sales "
            "conversation, oldest first. Merge them into a single, shorter summary that "
            "preserves every concrete fact (names, ages, numbers, plan names, decisions, "
            "concerns). Do not include commentary or headers, just the merged summary.\n\n"
            f"{combined_text}"
        )
        return await self._client.create_message(MAX_SUMMARY_TOKENS, prompt)