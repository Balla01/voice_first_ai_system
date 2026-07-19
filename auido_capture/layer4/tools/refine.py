"""
layer4/tools/refine.py — the refinement tool (refine_last_answer).

The tool-based successor to layer4/refinement.py's regex check: instead of
matching a fixed phrase list, the router recognizes an agent's "make it
shorter / rephrase / add an example" as a request to edit the LAST answer and
calls this tool.

Like the old REFINE path in main.py, this currently forwards the instruction
to Layer 5 as a follow-up query on the same session_id, relying on Layer 5's
own session memory to interpret it as an edit of its previous answer (the
/query contract has no dedicated "refine" mode). Revisit if Layer 5 gains an
explicit refine signal.
"""

import logging

from .base import Tool, ExecutionContext, ToolResult

logger = logging.getLogger("insureassist.layer4")


async def _refine_last_answer(args: dict, ctx: ExecutionContext) -> ToolResult:
    instruction = args["instruction"].strip()
    logger.debug(f"[{ctx.session_id}] tool refine_last_answer -> Layer 5 instruction={instruction!r}")
    try:
        resp = await ctx.layer5_client.query(
            query=instruction,
            session_id=ctx.session_id,
            customer_id=ctx.customer_id or ctx.session_id,
        )
        return ToolResult(
            ok=True,
            tool="refine_last_answer",
            query=instruction,
            answer=resp.answer,
            meta={
                "retrieval_time_s": resp.retrieval_time_s,
                "llm_time_s": resp.llm_time_s,
                "total_time_s": resp.total_time_s,
                "refine": True,
            },
        )
    except Exception as e:
        logger.error(f"[{ctx.session_id}] refine_last_answer failed: {e}")
        return ToolResult(ok=False, tool="refine_last_answer", query=instruction, error=str(e))


REFINE_LAST_ANSWER = Tool(
    name="refine_last_answer",
    description=(
        "Edit the LAST answer already shown to the agent, in place. Call this "
        "only when the AGENT (not the customer) gives an instruction to modify "
        "the previous answer, e.g. 'make it shorter', 'rephrase that', 'add an "
        "example'. Do not call it for new customer questions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "minLength": 1,
                "description": "The edit instruction, e.g. 'make it shorter'.",
            }
        },
        "required": ["instruction"],
    },
    executor=_refine_last_answer,
)
