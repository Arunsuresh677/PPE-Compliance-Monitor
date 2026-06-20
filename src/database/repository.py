"""
src/database/repository.py — Thread-safe PostgreSQL persistence for violation events.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import psycopg2
import psycopg2.extras

from src.config.settings import settings
from src.tracking.violation_tracker import ViolationEvent

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS violation_events (
    id              BIGSERIAL PRIMARY KEY,
    session         TEXT      NOT NULL,
    camera_id       TEXT      NOT NULL DEFAULT 'default',
    track_id        INTEGER   NOT NULL,
    violation_class TEXT      NOT NULL,
    start_time      DOUBLE PRECISION NOT NULL,
    end_time        DOUBLE PRECISION,
    duration_secs   DOUBLE PRECISION,
    frame_count     INTEGER   DEFAULT 1,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ve_session    ON violation_events (session);",
    "CREATE INDEX IF NOT EXISTS idx_ve_camera     ON violation_events (camera_id);",
    "CREATE INDEX IF NOT EXISTS idx_ve_created_at ON violation_events (created_at DESC);",
]


class ViolationRepository:
    """Thin persistence layer over PostgreSQL."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or settings.database_url
        self._lock = threading.Lock()
        self._conn: psycopg2.extensions.connection | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def init(self) -> None:
        """Open connection and create schema if it doesn't exist."""
        with self._lock:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = False
            with self._conn.cursor() as cur:
                cur.execute(_CREATE_TABLE)
                for idx in _CREATE_INDEXES:
                    cur.execute(idx)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def _ensure_connected(self) -> None:
        """Reconnect if the connection was dropped."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = False

    # ── Writes ─────────────────────────────────────────────────────────────

    def save_violation(self, event: ViolationEvent, session: str, camera_id: str = "default") -> None:
        """Persist a closed ViolationEvent."""
        with self._lock:
            self._ensure_connected()
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO violation_events
                        (session, camera_id, track_id, violation_class,
                         start_time, end_time, duration_secs, frame_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session,
                        camera_id,
                        event.track_id,
                        event.violation_class,
                        event.start_time,
                        event.end_time,
                        event.duration_secs,
                        event.frame_count,
                    ),
                )
            self._conn.commit()

    def save_violations(self, events: list[ViolationEvent], session: str, camera_id: str = "default") -> None:
        """Batch-persist multiple closed events in a single transaction."""
        if not events:
            return
        rows = [
            (
                session,
                camera_id,
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
            self._ensure_connected()
            with self._conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO violation_events
                        (session, camera_id, track_id, violation_class,
                         start_time, end_time, duration_secs, frame_count)
                    VALUES %s
                    """,
                    rows,
                )
            self._conn.commit()

    # ── Reads ──────────────────────────────────────────────────────────────

    def get_violations(
        self,
        session: str,
        limit: int = 500,
        camera_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_connected()
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if camera_id:
                    cur.execute(
                        """
                        SELECT id, session, camera_id, track_id, violation_class,
                               start_time, end_time, duration_secs, frame_count, created_at
                        FROM violation_events
                        WHERE session = %s AND camera_id = %s
                        ORDER BY start_time DESC
                        LIMIT %s
                        """,
                        (session, camera_id, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, session, camera_id, track_id, violation_class,
                               start_time, end_time, duration_secs, frame_count, created_at
                        FROM violation_events
                        WHERE session = %s
                        ORDER BY start_time DESC
                        LIMIT %s
                        """,
                        (session, limit),
                    )
                return [dict(row) for row in cur.fetchall()]

    def get_session_summary(self, session: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_connected()
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*)                                        AS total_events,
                        COUNT(DISTINCT track_id)                        AS distinct_violators,
                        ROUND(SUM(COALESCE(duration_secs, 0))::numeric, 1) AS total_violation_secs,
                        COUNT(DISTINCT camera_id)                       AS active_cameras
                    FROM violation_events
                    WHERE session = %s
                    """,
                    (session,),
                )
                overall = cur.fetchone()

                cur.execute(
                    """
                    SELECT violation_class, COUNT(*) AS class_count
                    FROM violation_events
                    WHERE session = %s
                    GROUP BY violation_class
                    ORDER BY class_count DESC
                    """,
                    (session,),
                )
                by_class = {r[0]: r[1] for r in cur.fetchall()}

        if not overall or overall[0] == 0:
            return {
                "total_events"        : 0,
                "distinct_violators"  : 0,
                "total_violation_secs": 0.0,
                "active_cameras"      : 0,
                "by_class"            : {},
            }

        return {
            "total_events"        : overall[0],
            "distinct_violators"  : overall[1],
            "total_violation_secs": float(overall[2] or 0.0),
            "active_cameras"      : overall[3],
            "by_class"            : by_class,
        }

    def list_sessions(self) -> list[str]:
        with self._lock:
            self._ensure_connected()
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT session FROM violation_events ORDER BY session"
                )
                return [r[0] for r in cur.fetchall()]

    def list_cameras(self, session: str) -> list[str]:
        with self._lock:
            self._ensure_connected()
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT camera_id FROM violation_events WHERE session = %s ORDER BY camera_id",
                    (session,),
                )
                return [r[0] for r in cur.fetchall()]
