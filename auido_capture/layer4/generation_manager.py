"""
layer4/generation_manager.py — concurrent, queued generation orchestration for
the router path (the successor to GenerationController's abort-always model).

Behaviour (design decision: out-of-order, query-tagged, no holdback):
  - SEPARATE questions run concurrently and each surfaces its answer as soon as
    it completes — no aborting, so no triggered question goes unanswered.
  - Concurrency is bounded (max_concurrent); overflow QUEUES rather than drops,
    via a per-session asyncio.Semaphore the runner acquires before executing.
  - CONTINUATIONS (the agent refining the question still being answered) abort
    the LATEST in-flight generation and reissue the enhanced query in its place.

GenerationController is kept for the tiers path (its latest-wins abort is right
there); this manager is only used in router mode.
"""

import asyncio
import logging
from typing import Callable, Coroutine, Dict, Optional, Tuple

logger = logging.getLogger("insureassist.layer4")

DEFAULT_MAX_CONCURRENT = 3


class GenerationManager:
    def __init__(self, max_concurrent: int = DEFAULT_MAX_CONCURRENT):
        self._max = max_concurrent
        self._sems: Dict[str, asyncio.Semaphore] = {}
        # session_id -> {gen_id: (query, task)}. "Active" = running OR queued
        # (waiting on the semaphore); both count so a continuation can abort a
        # not-yet-started generation too.
        self._active: Dict[str, Dict[int, Tuple[str, asyncio.Task]]] = {}
        self._counter = 0

    def _sem(self, session_id: str) -> asyncio.Semaphore:
        if session_id not in self._sems:
            self._sems[session_id] = asyncio.Semaphore(self._max)
        return self._sems[session_id]

    def has_active(self, session_id: str) -> bool:
        return bool(self._active.get(session_id))

    def latest_query(self, session_id: str) -> Optional[str]:
        """Query of the most recently submitted still-active generation — what a
        follow-up would be continuing."""
        active = self._active.get(session_id)
        if not active:
            return None
        return active[max(active)][0]

    def submit(self, session_id: str, query: str, coro_fn: Callable[[], Coroutine]) -> int:
        """Start a generation. Runs immediately if under the concurrency cap,
        otherwise waits (queued) on the semaphore. Returns its gen_id."""
        self._counter += 1
        gen_id = self._counter
        self._active.setdefault(session_id, {})

        async def _runner():
            try:
                async with self._sem(session_id):
                    await coro_fn()
            except asyncio.CancelledError:
                logger.debug(f"GenerationManager[{session_id}]: gen {gen_id} cancelled")
                raise
            except Exception as e:  # noqa: BLE001 — one generation failing mustn't kill others
                logger.error(f"GenerationManager[{session_id}]: gen {gen_id} errored: {e}")
            finally:
                self._active.get(session_id, {}).pop(gen_id, None)

        task = asyncio.create_task(_runner())
        self._active[session_id][gen_id] = (query, task)
        logger.debug(f"GenerationManager[{session_id}]: submitted gen {gen_id} "
                     f"(active={len(self._active[session_id])}, cap={self._max})")
        return gen_id

    def abort_latest(self, session_id: str) -> bool:
        """Cancel the most recent active generation (used when a continuation
        supersedes the question still being answered)."""
        active = self._active.get(session_id)
        if not active:
            return False
        latest_id = max(active)
        _, task = active.pop(latest_id)
        if not task.done():
            task.cancel()
            logger.debug(f"GenerationManager[{session_id}]: aborted latest gen {latest_id} (continuation)")
            return True
        return False

    def clear(self, session_id: str) -> None:
        active = self._active.pop(session_id, None)
        if active:
            for _, task in active.values():
                if not task.done():
                    task.cancel()
        self._sems.pop(session_id, None)
