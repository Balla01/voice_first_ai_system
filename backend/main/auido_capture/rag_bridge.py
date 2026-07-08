"""
rag_bridge.py — bridges Layer 2 (live transcripts) into Layer 3 (the RAG
pipeline at main/src/main.py).

Loads main/src/main.py under an explicit module name ("rag_main") via
importlib rather than a bare `import main`, because auido_capture's OWN
main.py is *also* named "main". Python caches imported modules in
sys.modules keyed by name — whichever "main" gets imported first would win
for every subsequent `import main` in the process, silently resolving to
the wrong file. Loading by explicit file path + explicit registered name
sidesteps that collision entirely.
"""

import asyncio
import sys
import importlib.util
from pathlib import Path
from typing import Awaitable, Callable

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    # main/src/main.py's own internal imports (history.history_pipeline,
    # constants, ...) expect main/src on sys.path, same as running it directly.
    sys.path.insert(0, str(_SRC_DIR))


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Loaded once per process. Exposes everything main.py defines or imports at
# module level: RuntimeHistory, _embed, rerank, parallel_search,
# build_context, call_llm.
rag_main = _load_module("rag_main", _SRC_DIR / "main.py")


class LiveRagWorker:
    """
    One instance per audio session. Wraps a RuntimeHistory scoped to that
    session, and runs flushed customer turns through the exact same
    retrieval/rerank/context/LLM pipeline main.py itself uses — reused
    unmodified, not reimplemented.
    """

    def __init__(
        self,
        session_id: str,
        customer_id: str,
        on_answer: Callable[[str, str], Awaitable[None]],
    ):
        """
        on_answer(query, answer): called after a customer turn is answered.
        The caller is responsible for getting it to the UI (e.g. over the
        session's WebSocket) — this class only produces the answer.
        """
        self.history = rag_main.RuntimeHistory(session_id=session_id, customer_id=customer_id)
        self.on_answer = on_answer
        # Serializes turns for this session so overlapping customer turns
        # (RAG calls run in a thread pool) can't race on RuntimeHistory's
        # shared, non-thread-safe state (e.g. its internal id counter).
        self._lock = asyncio.Lock()

    async def handle_agent_segment(self, text: str):
        """Agent speech: recorded as conversational memory only — never triggers a query."""
        text = text.strip()
        if not text:
            return
        async with self._lock:
            await asyncio.to_thread(self.history.add, "agent", text)

    async def handle_customer_turn(self, text: str):
        """A flushed, buffered customer turn — the actual RAG query."""
        text = text.strip()
        if not text:
            return

        def _blocking_pipeline():
            self.history.add("customer", text)

            query_vec = rag_main._embed([text])[0]
            recent_turns = self.history.get_recent_history(n=5)
            history_ranked, summary_ranked, docs_ranked = rag_main.parallel_search(query_vec, self.history)
            context = rag_main.build_context(recent_turns, history_ranked, summary_ranked, docs_ranked)
            answer = rag_main.call_llm(text, context)

            self.history.add("assistant", answer)
            return answer

        async with self._lock:
            answer = await asyncio.to_thread(_blocking_pipeline)

        await self.on_answer(text, answer)

    def close(self):
        self.history.end_session()
        self.history.close()
