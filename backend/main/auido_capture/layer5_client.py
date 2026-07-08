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

    async def query(self, query: str, session_id: str, customer_id: str) -> Layer5Response:
        """
        Non-streaming call — returns the full answer plus the API's own
        internal timing breakdown (retrieval_time_s, llm_time_s, total_time_s).
        Use this when you want the timing numbers; use query_stream() for
        token-by-token delivery (which doesn't carry this metadata).
        """
        payload = {"query": query, "session_id": session_id, "customer_id": customer_id, "stream": False}
        logger.debug(f"Layer5 request: {payload}")

        response = await self._client.post("/query", json=payload)
        response.raise_for_status()
        data = response.json()

        return Layer5Response(
            answer=_extract_answer_text(data),
            retrieval_time_s=data.get("retrieval_time_s"),
            llm_time_s=data.get("llm_time_s"),
            total_time_s=data.get("total_time_s"),
            raw=data,
        )

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
