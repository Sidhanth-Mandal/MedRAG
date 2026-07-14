"""
api/db.py
──────────
Persistent store backed by Supabase / PostgreSQL.

Note: bcrypt is called directly (not via passlib) to avoid the
      "error reading bcrypt version" warning from passlib + bcrypt>=4.
"""

from __future__ import annotations

import json
import os
from typing import Any

import bcrypt as _bcrypt
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ── Password hashing (direct bcrypt — avoids passlib version warning) ─────────

def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── Connection pool (module-level singleton) ──────────────────────────────────

_conn: psycopg2.extensions.connection | None = None


def _get_conn() -> psycopg2.extensions.connection:
    """Return the open connection, reconnecting if needed."""
    global _conn
    if _conn is None or _conn.closed:
        db_url = os.environ.get("SUPABASE_DB_URL", "")
        if not db_url:
            raise RuntimeError(
                "SUPABASE_DB_URL is not set — cannot persist chat history."
            )
        _conn = psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)
        _conn.autocommit = True
    return _conn


# ── Schema bootstrap ──────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL      PRIMARY KEY,
    username      TEXT        UNIQUE NOT NULL,
    password_hash TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id                    BIGSERIAL    PRIMARY KEY,
    session_id            TEXT         NOT NULL,
    user_id               INTEGER      REFERENCES users(id) ON DELETE CASCADE,
    role                  TEXT         NOT NULL,
    content               TEXT         NOT NULL,
    citations             JSONB        NOT NULL DEFAULT '[]',
    route                 TEXT,
    is_refused            BOOLEAN      DEFAULT FALSE,
    has_contradiction     BOOLEAN      DEFAULT FALSE,
    contradiction_details TEXT,
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages (session_id, created_at);
"""

# Migration: add user_id to existing tables, then add its index
# ALTER TABLE ... ADD COLUMN IF NOT EXISTS is safe to run repeatedly

_MIGRATE_SQL = """
ALTER TABLE chat_messages
    ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_chat_messages_user
    ON chat_messages (user_id, created_at DESC);
"""


def ensure_schema() -> None:
    """Create tables and run migrations if needed."""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
            cur.execute(_MIGRATE_SQL)
        print("[OK] DB schema ready (users + chat_messages)")
    except Exception as exc:
        print(f"[WARN] Could not ensure schema: {exc}")


# ── User management ───────────────────────────────────────────────────────────

def create_user(username: str, password: str) -> dict[str, Any] | None:
    """
    Create a new user. Returns the user dict on success, None if username taken.
    """
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, password_hash)
                VALUES (%s, %s)
                RETURNING id, username, created_at
                """,
                (username.strip().lower(), hash_password(password)),
            )
            return dict(cur.fetchone())
    except psycopg2.errors.UniqueViolation:
        return None
    except Exception as exc:
        print(f"[WARN] create_user failed: {exc}")
        return None


def get_user_by_username(username: str) -> dict[str, Any] | None:
    """Return the user row (including password_hash) or None."""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username, password_hash FROM users WHERE username = %s",
                (username.strip().lower(),),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as exc:
        print(f"[WARN] get_user_by_username failed: {exc}")
        return None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """Return the user row by ID or None."""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, username FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as exc:
        print(f"[WARN] get_user_by_id failed: {exc}")
        return None


# ── Write ─────────────────────────────────────────────────────────────────────

def save_message(
    session_id: str,
    role: str,
    content: str,
    user_id: int | None = None,
    citations: list[dict] | None = None,
    route: str | None = None,
    is_refused: bool = False,
    has_contradiction: bool = False,
    contradiction_details: str | None = None,
) -> None:
    """Persist a single chat turn to Postgres."""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages
                    (session_id, user_id, role, content, citations, route,
                     is_refused, has_contradiction, contradiction_details)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    user_id,
                    role,
                    content,
                    json.dumps(citations or []),
                    route,
                    is_refused,
                    has_contradiction,
                    contradiction_details,
                ),
            )
    except Exception as exc:
        print(f"[WARN] Could not save message to DB: {exc}")


# ── Read ──────────────────────────────────────────────────────────────────────

def get_history(session_id: str) -> list[dict[str, Any]]:
    """
    Return all messages for a session ordered by creation time.
    """
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, citations, route,
                       is_refused, has_contradiction, contradiction_details
                FROM   chat_messages
                WHERE  session_id = %s
                ORDER  BY created_at ASC
                """,
                (session_id,),
            )
            rows = cur.fetchall()
        result = []
        for row in rows:
            row = dict(row)
            if isinstance(row.get("citations"), str):
                row["citations"] = json.loads(row["citations"])
            result.append(row)
        return result
    except Exception as exc:
        print(f"[WARN] Could not fetch history from DB: {exc}")
        return []


def list_sessions(user_id: int | None = None) -> list[dict[str, Any]]:
    """
    Return a summary row per session:
      { session_id, message_count, preview }
    Filtered by user_id when provided.
    """
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            if user_id is not None:
                cur.execute(
                    """
                    SELECT
                        session_id,
                        COUNT(*) FILTER (WHERE role = 'human') AS message_count,
                        (
                            SELECT LEFT(content, 80)
                            FROM   chat_messages c2
                            WHERE  c2.session_id = cm.session_id
                              AND  c2.role = 'human'
                            ORDER  BY c2.created_at ASC
                            LIMIT  1
                        ) AS preview
                    FROM  chat_messages cm
                    WHERE cm.user_id = %s
                    GROUP BY session_id
                    ORDER BY MAX(created_at) DESC
                    """,
                    (user_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        session_id,
                        COUNT(*) FILTER (WHERE role = 'human') AS message_count,
                        (
                            SELECT LEFT(content, 80)
                            FROM   chat_messages c2
                            WHERE  c2.session_id = cm.session_id
                              AND  c2.role = 'human'
                            ORDER  BY c2.created_at ASC
                            LIMIT  1
                        ) AS preview
                    FROM  chat_messages cm
                    GROUP BY session_id
                    ORDER BY MAX(created_at) DESC
                    """
                )
            return [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        print(f"[WARN] Could not list sessions from DB: {exc}")
        return []


def delete_session(session_id: str) -> None:
    """Hard-delete all messages for a session."""
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chat_messages WHERE session_id = %s",
                (session_id,),
            )
    except Exception as exc:
        print(f"[WARN] Could not delete session from DB: {exc}")
