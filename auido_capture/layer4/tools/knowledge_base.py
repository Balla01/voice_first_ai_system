"""
layer4/tools/knowledge_base.py — the RAG tool (search_knowledge_base).

Wraps the existing Layer5Client.query() unchanged — the router's job is only
to DECIDE to search and to hand over a clean query string; Layer 5 still owns
retrieval + answer generation (router-only design, Layer 5 boundary intact).

The `query` arg is deliberately the LLM's *rewritten* search query, not the
raw spoken turn — messy ASR text ("uh what's, what's not covered like dental")
becomes a clean retrieval query, which is a free retrieval-quality win over
the old path that fed raw turn text straight to RAG.
"""

import logging

from .base import Tool, ExecutionContext, ToolResult

logger = logging.getLogger("insureassist.layer4")


async def _search_knowledge_base(args: dict, ctx: ExecutionContext) -> ToolResult:
    query = args["query"].strip()
    logger.debug(f"[{ctx.session_id}] tool search_knowledge_base -> Layer 5 query={query!r}")
    try:
        resp = await ctx.layer5_client.query(
            query=query,
            session_id=ctx.session_id,
            customer_id=ctx.customer_id or ctx.session_id,
        )
        return ToolResult(
            ok=True,
            tool="search_knowledge_base",
            query=query,
            answer=resp.answer,
            meta={
                "retrieval_time_s": resp.retrieval_time_s,
                "llm_time_s": resp.llm_time_s,
                "total_time_s": resp.total_time_s,
            },
        )
    except Exception as e:
        logger.error(f"[{ctx.session_id}] search_knowledge_base failed: {e}")
        return ToolResult(ok=False, tool="search_knowledge_base", query=query, error=str(e))


SEARCH_KNOWLEDGE_BASE = Tool(
    name="search_knowledge_base",
    description=(
        "Search the insurance knowledge base to answer a customer's question, "
        "objection, or concern. Call this when the customer asks about coverage, "
        "premiums, claims, exclusions, tax benefits, or buying/onboarding, or "
        "raises a price or competitor objection. Provide `query` as a clean, "
        "self-contained search query rewritten from the (possibly messy) spoken turn."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Clean retrieval query rewritten from the spoken turn.",
            },
            "is_followup": {
                "type": "boolean",
                "description": (
                    "Set true ONLY if a knowledge-base search is already in "
                    "progress (shown in context) and this turn continues or "
                    "refines THAT same question. Set false for a new, separate "
                    "question. Omit if no search is in progress."
                ),
            },
        },
        "required": ["query"],
    },
    executor=_search_knowledge_base,
)
