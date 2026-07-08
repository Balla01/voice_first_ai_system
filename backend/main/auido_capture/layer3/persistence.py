"""
layer3/persistence.py — Step 4: Persistence (PostgreSQL).

Every final turn and every epoch summary is written through immediately
(not batched), so a reconnect can rebuild full session state. Raw SQL via
asyncpg rather than an ORM, per the hackathon-speed tech stack decision.

asyncpg is imported lazily inside connect() so unit tests can exercise the
rest of the module (and everything importing it) without the package
installed or a real database running.
"""

from typing import List, Optional, Tuple

from .models import Turn, EpochSummary

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS turns (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    spoken_at DOUBLE PRECISION,
    is_important BOOLEAN NOT NULL DEFAULT FALSE,
    is_final BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS epoch_summaries (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    epoch_index INT NOT NULL,
    summary_text TEXT NOT NULL,
    covers_turn_count INT NOT NULL,
    superseded BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_epoch_session ON epoch_summaries(session_id, epoch_index);
"""


class Persistence:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool = None

    async def connect(self) -> None:
        import asyncpg  # lazy import, see module docstring
        self._pool = await asyncpg.create_pool(dsn=self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def insert_turn(self, turn: Turn) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO turns (session_id, speaker, text, spoken_at, is_important, is_final)
                VALUES ($1, $2, $3, $4, $5, TRUE)
                RETURNING id
                """,
                turn.session_id, turn.speaker, turn.text, turn.timestamp, turn.is_important,
            )
            return row["id"]

    async def update_turn_importance(self, turn: Turn) -> None:
        if turn.db_id is None:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE turns SET is_important = $1 WHERE id = $2",
                turn.is_important, turn.db_id,
            )

    async def insert_epoch_summary(self, summary: EpochSummary) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO epoch_summaries (session_id, epoch_index, summary_text, covers_turn_count)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                summary.session_id, summary.epoch_index, summary.text, summary.covers_turn_count,
            )
            return row["id"]

    async def mark_summaries_superseded(self, summaries: List[EpochSummary]) -> None:
        ids = [s.db_id for s in summaries if s.db_id is not None]
        if not ids:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE epoch_summaries SET superseded = TRUE WHERE id = ANY($1::bigint[])",
                ids,
            )

    async def load_session_history(self, session_id: str) -> Tuple[List[Turn], List[EpochSummary]]:
        """Reconnect support: rebuild in-memory state from what's in Postgres."""
        async with self._pool.acquire() as conn:
            turn_rows = await conn.fetch(
                "SELECT id, speaker, text, spoken_at, is_important FROM turns "
                "WHERE session_id = $1 ORDER BY created_at ASC",
                session_id,
            )
            summary_rows = await conn.fetch(
                "SELECT id, epoch_index, summary_text, covers_turn_count FROM epoch_summaries "
                "WHERE session_id = $1 AND superseded = FALSE ORDER BY epoch_index ASC",
                session_id,
            )

        turns = [
            Turn(
                session_id=session_id, speaker=r["speaker"], text=r["text"],
                timestamp=r["spoken_at"], is_important=r["is_important"], db_id=r["id"],
            )
            for r in turn_rows
        ]
        summaries = [
            EpochSummary(
                session_id=session_id, text=r["summary_text"], covers_turn_count=r["covers_turn_count"],
                epoch_index=r["epoch_index"], db_id=r["id"],
            )
            for r in summary_rows
        ]
        return turns, summaries