"""
layer3/persistence.py — Step 4: Persistence (SQLite).

Every final turn and every epoch summary is written through immediately
(not batched), so a reconnect can rebuild full session state. Raw SQL via
the stdlib ``sqlite3`` module rather than an ORM, per the hackathon-speed
tech stack decision.

Originally backed by PostgreSQL (Neon) via asyncpg; migrated to a local
SQLite file so development doesn't depend on reaching an external database.
``sqlite3`` is synchronous, so every DB call is run in a worker thread via
``asyncio.to_thread`` and serialized behind a lock — this keeps the public
API fully ``async`` and identical to the old asyncpg version, so nothing
outside this module changed.

The DSN is interpreted as a SQLite location:
    sqlite:///history.db          -> ./history.db (relative to CWD)
    sqlite:////abs/path/db.sqlite -> /abs/path/db.sqlite (absolute)
    /abs/path/db.sqlite           -> used as-is (bare path)
    history.db                    -> used as-is (bare relative path)
A leftover postgres:// URL falls back to ./layer3_history.db with a warning,
so an un-updated .env still boots instead of crashing.

sqlite3 is imported lazily inside connect() so unit tests can exercise the
rest of the module (and everything importing it) without touching the disk.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .models import Turn, EpochSummary

logger = logging.getLogger("insureassist.layer3.persistence")

# SQLite equivalents of the old Postgres schema:
#   BIGSERIAL PRIMARY KEY -> INTEGER PRIMARY KEY AUTOINCREMENT (rowid alias)
#   UUID                  -> TEXT   (SQLite is untyped; session_ids need not be UUIDs)
#   DOUBLE PRECISION      -> REAL
#   BOOLEAN               -> INTEGER (0/1)
#   TIMESTAMPTZ DEFAULT now() -> TEXT DEFAULT CURRENT_TIMESTAMP
# Insertion order is preserved by ordering on the monotonic id (see
# load_session_history) rather than created_at, which only has 1s resolution.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    spoken_at REAL,
    is_important INTEGER NOT NULL DEFAULT 0,
    is_final INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS epoch_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    epoch_index INTEGER NOT NULL,
    summary_text TEXT NOT NULL,
    covers_turn_count INTEGER NOT NULL,
    superseded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);
CREATE INDEX IF NOT EXISTS idx_epoch_session ON epoch_summaries(session_id, epoch_index);
"""


def _dsn_to_path(dsn: str) -> str:
    """Turn the DATABASE_URL into a SQLite file path (see module docstring)."""
    if not dsn:
        return "layer3_history.db"
    if dsn.startswith("sqlite:///"):
        # sqlite:////abs -> /abs (4 slashes); sqlite:///rel -> rel (3 slashes)
        rest = dsn[len("sqlite://"):]          # keeps one leading slash
        return rest[1:] if not rest.startswith("//") else rest
    if dsn.startswith(("postgres://", "postgresql://")):
        logger.warning(
            "DATABASE_URL is a Postgres URL but this build uses SQLite; "
            "falling back to ./layer3_history.db. Set DATABASE_URL=sqlite:///layer3_history.db to silence this."
        )
        return "layer3_history.db"
    return dsn  # treat anything else as a bare file path


class Persistence:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._path = _dsn_to_path(dsn)
        self._conn = None
        # SQLite serializes writes anyway; this lock also keeps our shared
        # connection (check_same_thread=False) safe across to_thread workers.
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        import sqlite3  # lazy import, see module docstring

        path = Path(self._path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        print("SQLite DB =", path.resolve())

        def _open():
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")      # better concurrent read/write
            conn.execute("PRAGMA busy_timeout=5000;")     # wait up to 5s on a lock instead of erroring
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            return conn

        self._conn = await asyncio.to_thread(_open)

    async def close(self) -> None:
        if self._conn:
            conn = self._conn
            self._conn = None
            await asyncio.to_thread(conn.close)

    async def insert_turn(self, turn: Turn) -> int:
        def _op():
            cur = self._conn.execute(
                """
                INSERT INTO turns (session_id, speaker, text, spoken_at, is_important, is_final)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (turn.session_id, turn.speaker, turn.text, turn.timestamp, int(turn.is_important)),
            )
            self._conn.commit()
            return cur.lastrowid

        async with self._lock:
            return await asyncio.to_thread(_op)

    async def update_turn_importance(self, turn: Turn) -> None:
        if turn.db_id is None:
            return

        def _op():
            self._conn.execute(
                "UPDATE turns SET is_important = ? WHERE id = ?",
                (int(turn.is_important), turn.db_id),
            )
            self._conn.commit()

        async with self._lock:
            await asyncio.to_thread(_op)

    async def insert_epoch_summary(self, summary: EpochSummary) -> int:
        def _op():
            cur = self._conn.execute(
                """
                INSERT INTO epoch_summaries (session_id, epoch_index, summary_text, covers_turn_count)
                VALUES (?, ?, ?, ?)
                """,
                (summary.session_id, summary.epoch_index, summary.text, summary.covers_turn_count),
            )
            self._conn.commit()
            return cur.lastrowid

        async with self._lock:
            return await asyncio.to_thread(_op)

    async def mark_summaries_superseded(self, summaries: List[EpochSummary]) -> None:
        ids = [s.db_id for s in summaries if s.db_id is not None]
        if not ids:
            return

        def _op():
            placeholders = ",".join("?" for _ in ids)
            self._conn.execute(
                f"UPDATE epoch_summaries SET superseded = 1 WHERE id IN ({placeholders})",
                ids,
            )
            self._conn.commit()

        async with self._lock:
            await asyncio.to_thread(_op)

    async def load_session_history(self, session_id: str) -> Tuple[List[Turn], List[EpochSummary]]:
        """Reconnect support: rebuild in-memory state from what's in SQLite."""
        def _op():
            turn_rows = self._conn.execute(
                "SELECT id, speaker, text, spoken_at, is_important FROM turns "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            summary_rows = self._conn.execute(
                "SELECT id, epoch_index, summary_text, covers_turn_count FROM epoch_summaries "
                "WHERE session_id = ? AND superseded = 0 ORDER BY epoch_index ASC",
                (session_id,),
            ).fetchall()
            return turn_rows, summary_rows

        async with self._lock:
            turn_rows, summary_rows = await asyncio.to_thread(_op)

        turns = [
            Turn(
                session_id=session_id, speaker=r[1], text=r[2],
                timestamp=r[3], is_important=bool(r[4]), db_id=r[0],
            )
            for r in turn_rows
        ]
        summaries = [
            EpochSummary(
                session_id=session_id, text=r[2], covers_turn_count=r[3],
                epoch_index=r[1], db_id=r[0],
            )
            for r in summary_rows
        ]
        return turns, summaries
