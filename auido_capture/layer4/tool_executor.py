"""
layer4/tool_executor.py — runs the tool calls the router decided on.

Multiple tools fire concurrently (a turn can legitimately match more than one
intent). Each call's result is handed to `on_result` as it completes, so the
caller (main.py) can push it to the UI. Runs inside a GenerationController
task, so a newer trigger cancels this whole batch cleanly mid-flight.
"""

import asyncio
import logging
from typing import Awaitable, Callable, List

from .tools import ToolRegistry, ToolCall, ToolResult, ExecutionContext

logger = logging.getLogger("insureassist.layer4")


async def execute_tool_calls(
    calls: List[ToolCall],
    registry: ToolRegistry,
    ctx: ExecutionContext,
    on_result: Callable[[ToolResult], Awaitable[None]],
) -> None:
    async def _run_one(call: ToolCall) -> None:
        tool = registry.get(call.name)
        if tool is None:
            logger.warning(f"[{ctx.session_id}] executor: unknown tool {call.name!r}, skipping")
            return
        try:
            result = await tool.execute(call.arguments, ctx)
        except asyncio.CancelledError:
            logger.debug(f"[{ctx.session_id}] executor: {call.name} cancelled (newer trigger)")
            raise
        except Exception as e:  # noqa: BLE001 — one tool failing shouldn't kill the batch
            logger.error(f"[{ctx.session_id}] executor: {call.name} raised {e}")
            result = ToolResult(ok=False, tool=call.name, error=str(e))
        await on_result(result)

    await asyncio.gather(*(_run_one(c) for c in calls))
