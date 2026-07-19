"""
layer4/tool_router.py — the LLM tool-calling replacement for the tiered
TriggerGate. Given the latest turn + Layer 3 context, the LLM decides which
tool(s) to call (if any). RAG is one tool among many, so multi-tool
integration is just "register more tools".

Pipeline per turn (cheap checks first, LLM last), serialized per session by a
lock so cooldown check-and-record is atomic (no concurrent double-fires):
    cooldown -> backchannel skip -> LLM router
    LLM route (real function-calling)
      -> validate each tool call's args against its JSON schema
      -> ONE retry re-prompting with the validation error on failure
      -> drop still-invalid calls (never send hallucinated args downstream)
    timeout / error -> fall back to the deterministic TriggerGate (reliability
      net: same fallback the LAYER4_TRIGGER_MODE flag flips to wholesale)

refine_last_answer is deliberately NOT a router tool — refinement ("make it
shorter" etc.) is an explicit UI-button action that calls Layer 5 directly,
bypassing the router. The router only ever decides search_knowledge_base.

Every decision is logged as a structured record (audit trail — feeds the L2
"Security, Compliance & Governance / auditability" criteria).
"""

import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .cooldown import CooldownTracker
from .pregate import is_backchannel
from .router_llm import RouterLLMClient
from .tools import ToolRegistry, ToolCall, validate_args

logger = logging.getLogger("insureassist.layer4")

DEFAULT_ROUTER_TIMEOUT_S = 8.0

SYSTEM_PROMPT = (
    "You are a real-time trigger router for a live insurance sales-call assistant. "
    "You are given the recent conversation context and the latest spoken turn "
    "(with its speaker label). Decide whether to call a tool to help the human "
    "sales agent, and if so, which one, with what arguments.\n\n"
    "Guidance:\n"
    "- Call search_knowledge_base when EITHER speaker (customer OR agent) asks a "
    "question, raises an objection, or voices a concern that needs information "
    "from the insurance knowledge base (coverage, premium, plan details, claims, "
    "exclusions, tax, buying). Rewrite the (possibly messy, code-mixed "
    "Hindi/English) turn into a clean English search query.\n"
    "- ERR ON THE SIDE OF ANSWERING: if the turn is a genuine question or "
    "request about insurance products, plans, premiums, claims, coverage, "
    "exclusions, tax, or policy details, call search_knowledge_base — even if "
    "it is phrased indirectly or is a classification/comparison question (e.g. "
    "'what type of plan is X, par or non-par?'). When a product question is "
    "plausible, fire.\n"
    "- Only skip (call no tool) for pure greetings, small talk, acknowledgements, "
    "filler, or narration that asks nothing (e.g. 'so today we're going to "
    "discuss a few things').\n"
    "Use the conversation context to resolve short follow-ups (e.g. 'and for "
    "kids?') into a complete query."
)


@dataclass
class RouterDecision:
    action: str                                   # "fire" | "no_trigger"
    tool_calls: List[ToolCall] = field(default_factory=list)
    reason: str = ""
    source: str = "router"                        # router | pregate | fallback
    model: str = ""
    latency_ms: float = 0.0
    retried: bool = False
    is_continuation: bool = False                 # true = refines the in-flight query (abort+reissue)


class ToolRouter:
    def __init__(
        self,
        session_id: str,
        llm_client: RouterLLMClient,
        registry: ToolRegistry,
        fallback_gate=None,
        timeout_s: float = DEFAULT_ROUTER_TIMEOUT_S,
    ):
        self.session_id = session_id
        self._llm = llm_client
        self._registry = registry
        self._fallback = fallback_gate            # a TriggerGate, used on router failure
        self._timeout_s = timeout_s
        self._cooldown = CooldownTracker()
        # Serializes routing per session so the cooldown check-and-record is
        # atomic — two turns arriving within one LLM round-trip can no longer
        # both pass the cooldown check and double-fire.
        self._route_lock = asyncio.Lock()

    # ---- public API ----

    async def route(
        self, speaker: str, text: str, context: str, now: float, in_flight_query: Optional[str] = None
    ) -> RouterDecision:
        """
        in_flight_query: the query of the generation currently being answered
        for this session, if any. When set, (1) the cooldown is skipped so a
        genuine follow-up arriving mid-answer isn't dropped, and (2) the LLM is
        told about it so it can mark this turn as a continuation vs a separate
        question.
        """
        text = text.strip()
        if not text:
            return self._log(RouterDecision("no_trigger", reason="empty turn", source="pregate"),
                             speaker, text)

        # Both speakers are eligible for LLM routing. (Agent/customer speech is
        # recorded in Layer 3 memory upstream regardless; this only decides
        # whether to also trigger a knowledge-base lookup.) The lock makes the
        # cooldown check + record atomic against concurrent turns.
        async with self._route_lock:
            # Cooldown only guards the IDLE case (nothing being answered). While a
            # generation is in flight we must let the turn reach the LLM so it can
            # be classified as a continuation or a new question — otherwise a
            # follow-up 1.5-3s later would be silently dropped here.
            if in_flight_query is None and self._cooldown.is_in_cooldown(now):
                return self._log(RouterDecision("no_trigger", reason="cooldown active", source="pregate"),
                                 speaker, text)

            if is_backchannel(text):
                return self._log(RouterDecision("no_trigger", reason="backchannel/filler", source="pregate"),
                                 speaker, text)

            decision = await self._route_via_llm(speaker, text, context, now, in_flight_query)
            return self._log(decision, speaker, text)

    # ---- internals ----

    async def _route_via_llm(
        self, speaker: str, text: str, context: str, now: float, in_flight_query: Optional[str]
    ) -> RouterDecision:
        user_content = self._build_user_content(speaker, text, context, in_flight_query)
        tools = self._registry.openai_schemas()  # OpenAI/Ollama shape; Anthropic client re-shapes internally
        t0 = time.perf_counter()

        try:
            resp = await asyncio.wait_for(
                self._llm.create_with_tools(SYSTEM_PROMPT, user_content, tools),
                timeout=self._timeout_s,
            )
        except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001 — any LLM/network failure -> fallback
            latency_ms = (time.perf_counter() - t0) * 1000
            return self._fallback_decision(speaker, text, now, reason=f"router error: {e}", latency_ms=latency_ms)

        latency_ms = (time.perf_counter() - t0) * 1000
        valid, retried = await self._parse_validate_retry(resp, user_content, tools)

        if valid:
            self._cooldown.record_trigger(now)
            # Continuation only meaningful when something is in flight; read the
            # model's is_followup flag off the search call.
            is_continuation = bool(
                in_flight_query is not None
                and any(c.arguments.get("is_followup") for c in valid)
            )
            return RouterDecision(
                "fire", valid, reason="llm tool selection",
                source="router", model=resp.model, latency_ms=latency_ms,
                retried=retried, is_continuation=is_continuation,
            )
        return RouterDecision(
            "no_trigger", reason="llm chose no tool", source="router",
            model=resp.model, latency_ms=latency_ms, retried=retried,
        )

    async def _parse_validate_retry(self, resp, user_content, tools):
        """Validate the model's tool calls; on any failure, re-prompt once with
        the errors and re-validate. Returns (valid_calls, retried?)."""
        valid, errors = self._collect_valid(resp.tool_calls)
        if not errors:
            return valid, False

        # One corrective retry.
        correction = (
            user_content
            + "\n\nYour previous tool call(s) had invalid arguments:\n"
            + "\n".join(f"- {name}: {err}" for name, err in errors)
            + "\nReturn corrected tool call(s) with valid arguments, or no tool call."
        )
        logger.debug(f"[{self.session_id}] router arg validation failed {errors}; retrying once")
        try:
            resp2 = await asyncio.wait_for(
                self._llm.create_with_tools(SYSTEM_PROMPT, correction, tools),
                timeout=self._timeout_s,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{self.session_id}] router retry failed: {e}; dropping invalid calls")
            return valid, True

        valid2, errors2 = self._collect_valid(resp2.tool_calls)
        if errors2:
            logger.warning(f"[{self.session_id}] router calls still invalid after retry {errors2}; dropped")
        # Merge first-pass valid calls with retry's valid calls, de-duped by name.
        seen = {c.name for c in valid}
        for c in valid2:
            if c.name not in seen:
                valid.append(c)
                seen.add(c.name)
        return valid, True

    def _collect_valid(self, raw_calls):
        valid: List[ToolCall] = []
        errors = []
        for rc in raw_calls:
            tool = self._registry.get(rc.name)
            if tool is None:
                errors.append((rc.name, "unknown tool (not in registry)"))
                continue
            try:
                args = json.loads(rc.arguments_raw) if isinstance(rc.arguments_raw, str) else rc.arguments_raw
            except json.JSONDecodeError:
                errors.append((rc.name, f"arguments not valid JSON: {rc.arguments_raw[:80]!r}"))
                continue
            ok, err = validate_args(tool.parameters, args)
            if not ok:
                errors.append((rc.name, err))
                continue
            valid.append(ToolCall(name=rc.name, arguments=args, raw_arguments=str(rc.arguments_raw)))
        return valid, errors

    def _fallback_decision(self, speaker, text, now, reason, latency_ms) -> RouterDecision:
        if self._fallback is None:
            logger.warning(f"[{self.session_id}] {reason}; no fallback gate -> no_trigger")
            return RouterDecision("no_trigger", reason=reason, source="fallback", latency_ms=latency_ms)

        logger.warning(f"[{self.session_id}] {reason}; falling back to deterministic TriggerGate")
        result = self._fallback.check(speaker=speaker, text=text, is_final=True, now=now)
        # Only FIRE maps to a search. Tiers' REFINE is ignored here — refine is a
        # UI-button action now, not something the router (or its fallback) fires.
        if result.action.value == "fire":
            self._cooldown.record_trigger(now)
            call = ToolCall(name="search_knowledge_base", arguments={"query": text})
            return RouterDecision("fire", [call], reason=f"fallback:{result.reason}", source="fallback",
                                  latency_ms=latency_ms)
        return RouterDecision("no_trigger", reason=f"fallback:{result.reason}", source="fallback",
                              latency_ms=latency_ms)

    def _build_user_content(self, speaker: str, text: str, context: str, in_flight_query: Optional[str]) -> str:
        ctx = context.strip() if context else "(no prior context)"
        parts = [f"Recent conversation context:\n{ctx}"]
        if in_flight_query:
            parts.append(
                "A knowledge-base search is CURRENTLY IN PROGRESS for this "
                f"question:\n  \"{in_flight_query}\"\n"
                "If the latest turn continues or refines THAT question, call "
                "search_knowledge_base with is_followup=true and a merged/"
                "enhanced query. If it is a NEW, separate question, call it with "
                "is_followup=false."
            )
        parts.append(f"Latest turn:\n[{speaker}] {text}")
        return "\n\n".join(parts)

    def _log(self, decision: RouterDecision, speaker: str, text: str) -> RouterDecision:
        """Structured, machine-parseable audit record of every routing decision."""
        record = {
            "event": "router_decision",
            "session_id": self.session_id,
            "speaker": speaker,
            "text": text,
            "action": decision.action,
            "source": decision.source,
            "reason": decision.reason,
            "model": decision.model,
            "latency_ms": round(decision.latency_ms, 1),
            "retried": decision.retried,
            "tools": [{"name": c.name, "args": c.arguments} for c in decision.tool_calls],
        }
        # Visible (INFO) when the LLM was actually consulted or we fired; trivial
        # pre-gate skips (cooldown/backchannel/empty) stay at DEBUG to avoid spam.
        visible = decision.action == "fire" or decision.source in ("router", "fallback")
        level = logging.INFO if visible else logging.DEBUG
        logger.log(level, f"router_decision {json.dumps(record, ensure_ascii=False)}")
        return decision
