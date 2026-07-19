# InsureAssist AI — Layer 3: Session Tracker

Modular implementation of the Context Window / Session Tracker, matching
`Layer3_4_Technical_Explainer.docx`.

## File structure

```
backend/layer3/
  models.py            Turn, EpochSummary — plain dataclasses, no dependencies
  dedup.py              Step 1: 500ms same-speaker-same-text dedup
  tokens.py             tiktoken counting + budget constants (6000 / 2000 / 4000)
  context_window.py     Step 2: in-memory deque + formatting (pure logic, no I/O)
  epoch_compaction.py   Step 3: async Claude calls (compact_turns, compact_summaries)
  persistence.py        Step 4: Postgres via asyncpg (schema, insert, update, reload)
  session_tracker.py     Orchestrator — the only module main.py needs to import
  __init__.py            Exports: Turn, EpochSummary, SessionTracker, EpochCompactor,
                          AnthropicClientAdapter, Persistence

backend/tests/
  fakes.py                In-memory FakePersistence + FakeCompactor (no real DB/API)
  test_dedup.py
  test_tokens.py
  test_context_window.py
  test_session_tracker.py
```

Each file has exactly one job, and every module except `session_tracker.py`
can be imported and unit tested with zero external dependencies (no
Postgres, no Anthropic API key, no network). `session_tracker.py` wires the
pieces together via constructor injection (`persistence` and `compactor` are
passed in, not created inside), which is what makes `tests/fakes.py`
possible — tests substitute in-memory fakes for both.

## Setup

```bash
cd backend
pip install -r requirements.txt -r requirements-layer3.txt
cp .env.example .env   # fill in DEEPGRAM_API_KEY, ANTHROPIC_API_KEY, DATABASE_URL
```

Make sure Postgres is running and `DATABASE_URL` points at a real database —
`Persistence.connect()` runs `CREATE TABLE IF NOT EXISTS` for `turns` and
`epoch_summaries` automatically on startup, so no separate migration step is
needed.

```bash
uvicorn main:app --reload --port 8000
```

## Running the tests

```bash
cd backend
PYTHONPATH=. python -m pytest tests/ -v
```

All 24 tests run fully offline in well under a second — no Postgres, no
Anthropic key, no network required, because `session_tracker.py` takes its
`persistence` and `compactor` as constructor arguments rather than
constructing them itself.

**Note on tiktoken:** it downloads its BPE encoding file from
`openaipublic.blob.core.windows.net` on first use. If that domain is blocked
(corporate firewall, sandboxed CI), `tokens.py` logs a warning and falls back
to an approximate chars/4 counter automatically — token *budgeting* still
works, just less precisely. If you hit this, either allow that domain or plug
in a locally-cached encoding file.

## How this wires into main.py

- One shared `Persistence` (Postgres pool) and one shared `EpochCompactor`
  (Anthropic client) are created once at app startup.
- Each session gets its own `SessionTracker`, created alongside its
  `AudioRouter` in `_get_or_create_router()`.
- `TranscriptMerger`'s `on_merged_final` callback (synchronous, called from
  inside Deepgram's async message handler) is bridged into Layer 3's async
  `SessionTracker.add_turn()` via `asyncio.create_task(...)` — see
  `_make_segment_handler()` / `_handle_merged_segment()` in `main.py`.
- `tracker.get_formatted_context()` is the hook point for Layer 4 (Smart
  Trigger Gate), which doesn't exist yet. When it does, it needs to call
  `tracker.mark_important(turn)` on any turn it fires a trigger on — that's
  the one piece of two-way coupling between Layer 3 and Layer 4 described in
  the design doc.

## What's NOT built yet

- Layer 4 (Smart Trigger Gate) — `get_formatted_context()` is called and
  logged at debug level in `main.py`, but nothing decides whether to fire a
  trigger yet.
- The full 8–10 regex patterns per business intent (Tier 1) — that's Layer 4
  scope, not Layer 3.