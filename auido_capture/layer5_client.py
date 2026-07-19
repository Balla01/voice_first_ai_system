"""
layer5_client.py — thin client for the Layer 5 (RAG + Prompt Builder) API.

Contract (as given):
    POST {base_url}/query
    { "query": str, "session_id": str, "customer_id": str, "stream": bool }

Confirmed stream=false response shape:
    {
      "answer": str,
      "retrieval_time_s": float,
      "llm_time_s": float,
      "total_time_s": float
    }

query_stream() (streaming mode) is still handled defensively since its body
format wasn't confirmed the same way — see notes there.

customer_id: this pipeline has no real customer-identity source (by design
— no CRM integration, see project scope). session_id is passed as
customer_id too for now. Replace with a real value if/when one exists.
"""

import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import httpx

logger = logging.getLogger("insureassist.layer5")

# "answer" is the confirmed field; the rest stay as a fallback in case a
# different endpoint/version of the API uses different naming.
ANSWER_FIELD_CANDIDATES = ("answer", "response", "result", "text", "message", "output")


@dataclass
class Layer5Response:
    answer: str
    retrieval_time_s: Optional[float] = None
    llm_time_s: Optional[float] = None
    total_time_s: Optional[float] = None   # as reported BY THE API itself
    ask_ai_session_id: Optional[str] = None  # echoed/minted only when advanced_filter=True was sent
    # True when the query was flagged as an ambiguous reference ("correct this
    # suggestion" etc.) and needs the caller to pick a context_source before a
    # real answer is produced — answer is "" in this case. See
    # rag_pipeline/api.py's module docstring for the full contract.
    needs_clarification: bool = False
    clarification_options: Optional[list] = None
    raw: dict = field(default_factory=dict)


def _extract_answer_text(data: dict) -> str:
    for key in ANSWER_FIELD_CANDIDATES:
        if key in data and isinstance(data[key], str):
            return data[key]
    logger.warning(
        f"Layer5: response didn't match any known answer field "
        f"{ANSWER_FIELD_CANDIDATES}; raw response: {data}"
    )
    return str(data)


def _clean_chunk(chunk: str) -> Optional[str]:
    """
    Light, non-destructive cleanup applied per raw chunk as it arrives —
    NOT line-based, since real token streams often have no newlines at all
    between tokens (line-based parsing would buffer everything until the
    connection closes, defeating the point of streaming — confirmed by
    testing against a fake server that streams without newlines).
    Strips an SSE-style "data:" prefix if present; otherwise passes the
    chunk through untouched.
    """
    if chunk.startswith("data:"):
        chunk = chunk[len("data:"):].lstrip()
    if chunk.strip() == "[DONE]":
        return None
    return chunk if chunk else None


class Layer5Client:
    def __init__(self, base_url: str, timeout_s: float = 30.0):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def query(
        self,
        query: str,
        session_id: str,
        customer_id: str,
        advanced_filter: bool = False,
        ask_ai_session_id: Optional[str] = None,
        context_source: Optional[str] = None,
    ) -> Layer5Response:
        """
        Non-streaming call — returns the full answer plus the API's own
        internal timing breakdown (retrieval_time_s, llm_time_s, total_time_s).
        Use this when you want the timing numbers; use query_stream() for
        token-by-token delivery (which doesn't carry this metadata).

        advanced_filter / ask_ai_session_id: opt-in Ask-AI "thread" mode (see
        rag_pipeline/api.py). Omit ask_ai_session_id (leave None) to let the
        server mint a new thread id, returned on the response — pass it back
        in on the next call to continue that thread.

        context_source: resolves an ambiguous-reference clarification
        ("suggestion_card" | "current_thread") — only has an effect when the
        prior call for this same query came back with needs_clarification=True.
        Leave None on a normal call.
        """
        payload = {
            "query": query,
            "session_id": session_id,
            "customer_id": customer_id,
            "stream": False,
            "advanced_filter": advanced_filter,
            "ask_ai_session_id": ask_ai_session_id,
            "context_source": context_source,
        }
        logger.debug(f"Layer5 request: {payload}")

        response = await self._client.post("/query", json=payload)
        response.raise_for_status()
        data = response.json()

        return Layer5Response(
            answer=_extract_answer_text(data),
            retrieval_time_s=data.get("retrieval_time_s"),
            llm_time_s=data.get("llm_time_s"),
            total_time_s=data.get("total_time_s"),
            ask_ai_session_id=data.get("ask_ai_session_id"),
            needs_clarification=data.get("needs_clarification", False),
            clarification_options=data.get("clarification_options"),
            raw=data,
        )

    async def get_profile(self, session_id: str, customer_id: str) -> dict:
        """
        GET {base_url}/profile?session_id=...&customer_id=... — returns
        {session_id, customer_id, profile: {name, age, profession, location,
        policy_product, category}}. Mirrors query()'s raise_for_status/raise
        style — let the caller (the /profile route in main.py) turn failures
        into an HTTPException, same as /ask already does around query().
        """
        response = await self._client.get(
            "/profile", params={"session_id": session_id, "customer_id": customer_id}
        )
        response.raise_for_status()
        return response.json()

    async def end_session(self, session_id: str, customer_id: str) -> None:
        """POST {base_url}/session/{id}/end?customer_id=... — tells Layer 5 to
        summarize + close this session's RuntimeHistory. Swallows its own
        exceptions: ending a call in the UI must never fail because Layer 5 is
        slow/down, and the caller has no user-facing way to surface a failure
        here anyway — mirrors the fire-and-forget email-send pattern elsewhere
        in this codebase (rag_pipeline/api.py's _send_answer_email)."""
        try:
            response = await self._client.post(
                f"/session/{session_id}/end", params={"customer_id": customer_id}
            )
            response.raise_for_status()
        except Exception as e:
            logger.warning(f"Layer5 end_session failed for session={session_id}: {e}")

    async def list_ask_ai_sessions(self, customer_id: str) -> dict:
        """GET {base_url}/ask-ai/sessions?customer_id=... — returns
        {customer_id, sessions: [{ask_ai_session_id, turn_count, last_query,
        last_timestamp}, ...]}, most-recent-first. Raises on failure, same
        style as query()/get_profile() — the caller turns it into an
        HTTPException."""
        response = await self._client.get("/ask-ai/sessions", params={"customer_id": customer_id})
        response.raise_for_status()
        return response.json()

    async def get_ask_ai_thread(self, customer_id: str, ask_ai_session_id: str) -> dict:
        """GET {base_url}/ask-ai/thread?customer_id=...&ask_ai_session_id=... —
        returns {customer_id, ask_ai_session_id, turns: [{query, answer,
        timestamp}, ...]}, chronological. Raises on failure, same style as
        query()/get_profile()."""
        response = await self._client.get(
            "/ask-ai/thread", params={"customer_id": customer_id, "ask_ai_session_id": ask_ai_session_id}
        )
        response.raise_for_status()
        return response.json()

    async def query_stream(
        self, query: str, session_id: str, customer_id: str, stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Always an async generator, regardless of stream=True/False — a
        blocking (stream=False) call just yields exactly one chunk, so
        callers can consume both the same way.
        """
        payload = {"query": query, "session_id": session_id, "customer_id": customer_id, "stream": stream}
        logger.debug(f"Layer5 request: {payload}")

        if stream:
            async with self._client.stream("POST", "/query", json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    text = _clean_chunk(chunk)
                    if text:
                        logger.debug(f"Layer5 stream chunk: {text!r}")
                        yield text
        else:
            response = await self._client.post("/query", json=payload)
            response.raise_for_status()
            data = response.json()
            yield _extract_answer_text(data)
