# InsureAssist AI — Layer 4: Smart Trigger Gate

Decides WHEN a moment in the conversation is worth firing Layer 5 (RAG +
LLM), instead of reacting to every single sentence. Matches
`Layer3_4_Technical_Explainer.docx`, Section 3.

## File structure

```
backend/layer4/
  models.py                  TriggerAction (FIRE/NO_TRIGGER/REFINE), IntentMatch, TriggerResult
  cooldown.py                 3s minimum between fired triggers, per session
  refinement.py                'make it shorter' / 'rephrase' / 'add example' detection
  intent_tier3_heuristic.py   Tier 3: "assistant answered x2 + short follow-up", ~0ms
  trigger_gate.py              Orchestrator — the only module main.py imports for decisions
  generation_controller.py    AbortController — cancels in-flight Layer 5 calls (ready, unused until Layer 5 exists)
  __init__.py                  Exports: TriggerAction, TriggerResult, IntentMatch, TriggerGate,
                                GenerationController
```

Same modular pattern as Layer 3: each file has one job, `TriggerGate` wires
them together via constructor injection.

Note: Tier 1 (regex registry) and Tier 2 (MiniLM embedding classifier) that
used to sit here were removed — see the module docstring at the top of
`trigger_gate.py` for why. `TriggerGate` is now only the last-resort fallback
`ToolRouter` reaches for after both its primary and fallback LLM routers fail
(see `tool_router.py`), and consists solely of Tier 3.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

No new env vars needed — Layer 4 doesn't call any external API itself (Tier
3 is pure Python). It only reads from Layer 3's context and, if it decides to
fire, calls back into Layer 3 (`tracker.mark_important(turn)`).

```bash
uvicorn main:app --reload --port 8000
```

## What you'll see now

**Terminal** — unchanged conversation view, plus one new line whenever a
trigger actually fires:
```
[customer] What does this plan cover for diabetes?
  -> Layer 4 FIRE: ['policy_inquiry'] (would call Layer 5 here)
```

**`logs/app.log`** (new folder, auto-created on first run) — full DEBUG
detail for every gate decision: cooldown state, which regex matched, Tier
2/3 scores, why something didn't fire. This is the file to open when you
need to understand *why* a turn did or didn't trigger.

```bash
tail -f logs/app.log          # watch it live while testing
grep "TriggerGate decision" logs/app.log   # just the final decisions
```

## How it's wired into main.py

- Each session gets its own `TriggerGate` (cooldown and Tier 3's
  "assistant answered" counter are per-session state), created alongside its
  `SessionTracker` in `_get_or_create_router()`.
- `_handle_merged_segment()` calls `gate.check(...)` right after every turn
  is added to Layer 3. On `FIRE`, it calls `tracker.mark_important(turn)` —
  this is the two-way link between Layer 3 and Layer 4 described in the
  design doc (a turn only gets tagged IMPORTANT in the formatted context if
  Layer 4 actually fired on it).
- `GenerationController` is built but not called yet — there's no Layer 5
  to actually generate/stream an answer, so there's nothing to abort yet.
  Wire it in when Layer 5 exists: call
  `generation_controller.start_generation(session_id, coro_fn)` every time
  `gate.check()` returns `FIRE`, and it'll auto-cancel any still-running
  generation for that session.

## Logging convention (applies as later layers add debug logs too)

- `logger.info(...)` → shows on the terminal. Conversation lines and
  high-level decisions worth narrating live (FIRE/REFINE).
- `logger.debug(...)` → only ever goes to `logs/app.log`. Internal
  step-by-step detail.

This split is controlled entirely by `logging_config.py` (shared
infrastructure, not layer-specific) — nothing about it needs to change as
Layer 1/2/3 gain debug logging later too.

## What's NOT built yet

- "Images bypass cooldown" — still unclear what this refers to in an
  audio-only pipeline; treated as out of scope.