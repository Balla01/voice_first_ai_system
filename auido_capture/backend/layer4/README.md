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
  intent_tier1_regex.py       Tier 1: 6-intent business registry, multi-match, ~0ms
  intent_tier2_embedding.py   Tier 2: MiniLM cosine-similarity fallback, ~10-30ms
  intent_tier3_heuristic.py   Tier 3: "assistant answered x2 + short follow-up", ~0ms
  trigger_gate.py              Orchestrator — the only module main.py imports for decisions
  generation_controller.py    AbortController — cancels in-flight Layer 5 calls (ready, unused until Layer 5 exists)
  __init__.py                  Exports: TriggerAction, TriggerResult, IntentMatch, TriggerGate,
                                GenerationController, Tier2EmbeddingClassifier
```

Same modular pattern as Layer 3: each file has one job, `TriggerGate` wires
them together via constructor injection (you can pass a shared
`Tier2EmbeddingClassifier` instance across sessions, which `main.py` does).

## Setup

```bash
cd backend
pip install -r requirements.txt -r requirements-layer3.txt   # now includes sentence-transformers
```

No new env vars needed — Layer 4 doesn't call any external API itself (Tier
1 is regex, Tier 2 is a local model, Tier 3 is pure Python). It only reads
from Layer 3's context and, if it decides to fire, calls back into Layer 3
(`tracker.mark_important(turn)`).

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

## Known, expected warning: Tier 2 (MiniLM) probably won't load

```
WARNING ... Tier2: embedder unavailable (...); Tier 2 will report no match
on every call, falling through to Tier 3.
```

`sentence-transformers` downloads its model from `huggingface.co` on first
use — same category of problem as tiktoken's blob storage earlier, and
likely blocked on a locked-down corporate network. This is **not a crash**:
Tier 2 just always reports "no match," so every turn falls through to Tier
1 (regex) and Tier 3 (heuristic), which cover your actual demo intents fine.
Fix it only if you specifically need the embedding fallback for intents Tier
1's regex doesn't catch.

## How it's wired into main.py

- One shared `Tier2EmbeddingClassifier` is created once at app startup
  (loading the MiniLM model, or logging the warning above, only once — not
  per session).
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

- Layer 5 (RAG + Prompt Builder) — the actual thing that gets triggered.
  `main.py` currently just logs `(would call Layer 5 here)`.
- The full 8–10 regex patterns per business intent (only representative
  subsets are in `intent_tier1_regex.py` — pull the complete list in before
  the demo, per the design doc's open items).
- "Images bypass cooldown" — still unclear what this refers to in an
  audio-only pipeline; treated as out of scope.