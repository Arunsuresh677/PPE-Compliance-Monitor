"""
database.py — SQLite violation event log

Single shared connection in WAL mode, protected by a threading.Lock so it is
safe to call from both the main thread and WebRTC worker threads.

Schema
──────
violation_events
    id              AUTOINCREMENT primary key
    session         TEXT   — ISO timestamp of the detection session
    track_id        INT    — ByteTrack worker ID
    violation_class TEXT   — e.g. "NO-Hardhat"
    start_time      TEXT   — ISO datetime when violation began
    end_time        TEXT   — ISO datetime when it ended (NULL = still open)
    duration_secs   REAL   — wall-clock seconds the violation was active
    frame_count     INT    — YOLO frames the violation was detected across
"""

import logging
import sqlite3
import threading
from datetime import datetime
from typing import Optional

from tracker import ViolationEvent

log = logging.getLogger(__name__)

_DB_PATH = "ppe_violations.db"
_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()


def init_db(path: str = _DB_PATH) -> None:
    """Open the shared connection and create tables. Call once at startup."""
    global _conn, _DB_PATH
    _DB_PATH = path
    _conn = sqlite3.connect(path, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA synchronous=NORMAL")
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS violation_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session         TEXT    NOT NULL,
            track_id        INTEGER NOT NULL,
            violation_class TEXT    NOT NULL,
            start_time      TEXT    NOT NULL,
            end_time        TEXT,
            duration_secs   REAL    NOT NULL,
            frame_count     INTEGER NOT NULL
        )
    """)
    _conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session ON violation_events(session)"
    )
    _conn.commit()
    log.info("Violation DB initialised at %s", path)


def save_violation(event: ViolationEvent, session: str) -> None:
    """Persist a closed ViolationEvent. No-op if DB is not initialised."""
    if _conn is None or event.end_time is None:
        return
    with _lock:
        _conn.execute("""
            INSERT INTO violation_events
                (session, track_id, violation_class,
                 start_time, end_time, duration_secs, frame_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session,
            event.track_id,
            event.violation_class,
            _fmt(event.start_time),
            _fmt(event.end_time),
            event.duration_secs,
            event.frame_count,
        ))
        _conn.commit()


def get_violations(session: Optional[str] = None, limit: int = 200) -> list[dict]:
    """Fetch recent violation events, optionally filtered by session."""
    if _conn is None:
        return []
    with _lock:
        if session:
            cur = _conn.execute(
                "SELECT * FROM violation_events WHERE session=? ORDER BY id DESC LIMIT ?",
                (session, limit),
            )
        else:
            cur = _conn.execute(
                "SELECT * FROM violation_events ORDER BY id DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in cur.fetchall()]


def get_session_summary(session: str) -> dict:
    """Per-class and aggregate stats for one session."""
    if _conn is None:
        return {}
    with _lock:
        cur = _conn.execute("""
            SELECT
                violation_class,
                COUNT(*)         AS events,
                COUNT(DISTINCT track_id) AS workers,
                SUM(duration_secs)       AS total_secs,
                AVG(duration_secs)       AS avg_secs,
                MAX(duration_secs)       AS max_secs
            FROM violation_events
            WHERE session = ?
            GROUP BY violation_class
        """, (session,))
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return {"total_events": 0, "distinct_violators": 0, "by_class": {}}

    total_events   = sum(r["events"] for r in rows)
    distinct       = set()
    for r in rows:
        distinct.add(r["workers"])   # approximate — exact needs a sub-query
    by_class = {
        r["violation_class"]: {
            "events"    : r["events"],
            "total_secs": round(r["total_secs"] or 0, 1),
            "avg_secs"  : round(r["avg_secs"]   or 0, 1),
            "max_secs"  : round(r["max_secs"]   or 0, 1),
        }
        for r in rows
    }
    return {
        "total_events"     : total_events,
        "distinct_violators": rows[0]["workers"],
        "by_class"         : by_class,
    }


def close_db() -> None:
    global _conn
    with _lock:
        if _conn:
            _conn.close()
            _conn = None
    log.info("Violation DB closed.")


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
