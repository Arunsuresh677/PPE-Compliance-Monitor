"""
src/database/repository.py — Thread-safe SQLite persistence for violation events.

SQLite is opened in WAL mode so the WebRTC callback thread (writer) and the
Streamlit main thread (reader) can access it concurrently without blocking each
other. A threading.Lock serialises writes so two writer threads never interleave.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.tracking.violation_tracker import ViolationEvent

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS violation_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session         TEXT    NOT NULL,
    track_id        INTEGER NOT NULL,
    violation_class TEXT    NOT NULL,
    start_time      REAL    NOT NULL,
    end_time        REAL,
    duration_secs   REAL,
    frame_count     INTEGER DEFAULT 1
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_violation_events_session
    ON violation_events (session);
"""


class ViolationRepository:
    """Thin persistence layer over a single SQLite file."""

    def __init__(self, db_path: str | None = None) -> None:
        self._path = db_path or settings.db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def init(self) -> None:
        """Open the database and create schema if it doesn't exist."""
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._conn = sqlite3.connect(
                self._path,
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_INDEX)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ── Writes ────────────────────────────────────────────────────────────

    def save_violation(self, event: ViolationEvent, session: str) -> None:
        """Persist a closed ViolationEvent."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO violation_events
                    (session, track_id, violation_class,
                     start_time, end_time, duration_secs, frame_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session,
                    event.track_id,
                    event.violation_class,
                    event.start_time,
                    event.end_time,
                    event.duration_secs,
                    event.frame_count,
                ),
            )
            self._conn.commit()

    def save_violations(self, events: list[ViolationEvent], session: str) -> None:
        """Batch-persist multiple closed events in a single transaction."""
        if not events:
            return
        rows = [
            (
                session,
                ev.track_id,
                ev.violation_class,
                ev.start_time,
                ev.end_time,
                ev.duration_secs,
                ev.frame_count,
            )
            for ev in events
        ]
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO violation_events
                    (session, track_id, violation_class,
                     start_time, end_time, duration_secs, frame_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()

    # ── Reads ─────────────────────────────────────────────────────────────

    def get_violations(
        self,
        session: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id, session, track_id, violation_class,
                       start_time, end_time, duration_secs, frame_count
                FROM violation_events
                WHERE session = ?
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (session, limit),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_session_summary(self, session: str) -> dict[str, Any]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT
                    COUNT(*)                                        AS total_events,
                    COUNT(DISTINCT track_id)                        AS distinct_violators,
                    ROUND(SUM(COALESCE(duration_secs, 0)), 1)       AS total_violation_secs,
                    violation_class,
                    COUNT(*)                                        AS class_count
                FROM violation_events
                WHERE session = ?
                GROUP BY violation_class
                """,
                (session,),
            )
            rows = cur.fetchall()

        if not rows:
            return {
                "total_events"        : 0,
                "distinct_violators"  : 0,
                "total_violation_secs": 0.0,
                "by_class"            : {},
            }

        total_events = sum(r[4] for r in rows)
        distinct     = rows[0][1]
        total_secs   = sum(r[2] or 0 for r in rows)
        by_class     = {r[3]: r[4] for r in rows}

        return {
            "total_events"        : total_events,
            "distinct_violators"  : distinct,
            "total_violation_secs": round(total_secs, 1),
            "by_class"            : by_class,
        }

    def list_sessions(self) -> list[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT DISTINCT session FROM violation_events ORDER BY session"
            )
            return [r[0] for r in cur.fetchall()]
