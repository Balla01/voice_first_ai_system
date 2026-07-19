"""
agent_store.py — sales_agents: a minimal SQLite table for agent login.

Deliberately basic (per project scope: static login page, single demo
agent) — no session tokens/expiry, just a username/password check against a
stored hash. Passwords are hashed+salted with stdlib hashlib.pbkdf2_hmac (no
new dependency) — never stored in plaintext, even for a demo.

A fresh sqlite3 connection is opened and closed per call rather than shared
across requests — logins are low-frequency, and this sidesteps sqlite3's
not-thread-safe-by-default connection semantics entirely rather than adding
locking for it.

Seeds one default agent on first run (see ensure_seeded()): mukul.vyas / 123456.
This becomes agent_id for every Ask-AI thread that agent creates — see
rag_pipeline/api.py's module docstring.
"""

import hashlib
import secrets
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "agents.db"

DEFAULT_USERNAME = "mukul.vyas"
DEFAULT_PASSWORD = "123456"

PBKDF2_ITERATIONS = 100_000

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sales_agents (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    ).hex()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(SCHEMA_SQL)
    return conn


def create_agent(username: str, password: str, conn: sqlite3.Connection = None) -> None:
    """Insert (or overwrite) one agent with a freshly salted password hash."""
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    owns_conn = conn is None
    if owns_conn:
        conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO sales_agents (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, password_hash, salt),
        )
        conn.commit()
    finally:
        if owns_conn:
            conn.close()


def agent_exists(username: str) -> bool:
    conn = _connect()
    try:
        row = conn.execute("SELECT 1 FROM sales_agents WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()
    return row is not None


def register_agent(username: str, password: str) -> bool:
    """Creates a new agent iff the username isn't already taken. Returns True
    if created, False if it already existed — unlike create_agent(), never
    overwrites an existing account's password."""
    if agent_exists(username):
        return False
    create_agent(username, password)
    return True


def verify_agent(username: str, password: str) -> bool:
    """True if username exists and password matches its stored hash."""
    if not username or not password:
        return False
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT password_hash, salt FROM sales_agents WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return False
    password_hash, salt = row
    return secrets.compare_digest(_hash_password(password, salt), password_hash)


def ensure_seeded() -> None:
    """Inserts the default demo agent if it doesn't already exist. Safe to
    call on every startup — idempotent, never overwrites an existing row (so
    a password changed later via create_agent() survives a restart)."""
    conn = _connect()
    try:
        existing = conn.execute(
            "SELECT 1 FROM sales_agents WHERE username = ?", (DEFAULT_USERNAME,)
        ).fetchone()
        if existing is None:
            create_agent(DEFAULT_USERNAME, DEFAULT_PASSWORD, conn=conn)
    finally:
        conn.close()
