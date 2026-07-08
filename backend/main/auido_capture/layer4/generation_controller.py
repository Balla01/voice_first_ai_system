"""
layer4/generation_controller.py — AbortController (Section 3.4 of the design doc).

If a new trigger fires while a previous LLM response is still streaming from
Layer 5, the in-flight asyncio.Task is cancelled immediately and a fresh
generation starts — so the agent never sees two answers racing each other
on screen.

Layer 5 doesn't exist yet, so start_generation() takes a coroutine factory
(coro_fn) that the caller supplies — this class only owns the cancel/replace
bookkeeping, not the actual generation logic.
"""

import asyncio
import logging
from typing import Callable, Coroutine, Dict

logger = logging.getLogger("insureassist.layer4")


class GenerationController:
    def __init__(self):
        self._active_tasks: Dict[str, asyncio.Task] = {}

    async def start_generation(self, session_id: str, coro_fn: Callable[[], Coroutine]) -> asyncio.Task:
        aborted = self._cancel_if_running(session_id)
        if aborted:
            logger.debug(f"GenerationController[{session_id}]: aborted in-flight task, starting fresh")
        else:
            logger.debug(f"GenerationController[{session_id}]: no in-flight task, starting new")

        task = asyncio.create_task(coro_fn())
        self._active_tasks[session_id] = task
        return task

    def _cancel_if_running(self, session_id: str) -> bool:
        existing = self._active_tasks.get(session_id)
        if existing and not existing.done():
            existing.cancel()
            return True
        return False

    def clear(self, session_id: str) -> None:
        self._active_tasks.pop(session_id, None)