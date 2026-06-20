"""
src/tracking/violation_tracker.py — Per-ID PPE violation state machine.

Converts per-frame YOLO+ByteTrack detections into discrete violation events:
  - one ViolationEvent per (track_id, violation_class) incident
  - tracks start time, duration, and frame count
  - closes events when a worker becomes compliant or leaves the frame
  - tolerates brief occlusion with a configurable stale timeout
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


VIOLATION_CLASSES: frozenset[str] = frozenset({"NO-Hardhat", "NO-Mask", "NO-Safety Vest"})
COMPLIANT_CLASSES: frozenset[str] = frozenset({"Hardhat", "Mask", "Safety Vest"})

_DEFAULT_STALE_TIMEOUT = 3.0


@dataclass
class ViolationEvent:
    track_id:        int
    violation_class: str
    start_time:      float
    end_time:        Optional[float] = field(default=None)
    frame_count:     int             = field(default=1)

    @property
    def duration_secs(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return round(end - self.start_time, 1)

    @property
    def is_closed(self) -> bool:
        return self.end_time is not None

    def to_dict(self) -> dict:
        return {
            "track_id"       : self.track_id,
            "violation_class": self.violation_class,
            "start_time"     : self.start_time,
            "end_time"       : self.end_time,
            "duration_secs"  : self.duration_secs,
            "frame_count"    : self.frame_count,
        }


class ViolationTracker:
    """
    Stateful per-track-ID violation tracker.

    State machine per track ID:
        no open event  + violation detected  →  open ViolationEvent
        open event     + same violation      →  extend (increment frame_count)
        open event     + worker compliant    →  close event immediately
        open event     + not seen > timeout  →  close event (left frame)

    Call update() once per frame with the list of (track_id, class_name)
    detections. Call flush_closed() to drain events ready for persistence.
    """

    def __init__(self, stale_timeout: float = _DEFAULT_STALE_TIMEOUT) -> None:
        self._stale_timeout = stale_timeout
        # track_id → {violation_class → ViolationEvent}
        self._open: dict[int, dict[str, ViolationEvent]] = {}
        # track_id → last-seen epoch seconds
        self._last_seen: dict[int, float] = {}
        # events closed since last flush_closed() call
        self._closed_buffer: list[ViolationEvent] = []

    # ── Public API ────────────────────────────────────────────────────────

    def update(
        self,
        detections: list[tuple[int, str]],
    ) -> list[ViolationEvent]:
        """
        Process one frame of (track_id, class_name) detections.
        Returns currently open ViolationEvents for live display.
        """
        now = time.time()
        seen: set[int] = set()

        for track_id, cls_name in detections:
            seen.add(track_id)
            self._last_seen[track_id] = now

            if cls_name in VIOLATION_CLASSES:
                events = self._open.setdefault(track_id, {})
                if cls_name not in events:
                    events[cls_name] = ViolationEvent(
                        track_id=track_id,
                        violation_class=cls_name,
                        start_time=now,
                    )
                else:
                    events[cls_name].frame_count += 1

            elif cls_name in COMPLIANT_CLASSES:
                self._close_id(track_id, now)

        stale = [
            tid for tid, last in self._last_seen.items()
            if tid not in seen and (now - last) > self._stale_timeout
        ]
        for tid in stale:
            self._close_id(tid, now)
            del self._last_seen[tid]

        return self.active_events

    def flush_closed(self) -> list[ViolationEvent]:
        """Return and clear all events closed since the last call."""
        closed, self._closed_buffer = self._closed_buffer, []
        return closed

    def close_all(self) -> list[ViolationEvent]:
        """Close every open event — call at stream end or session reset."""
        now = time.time()
        for tid in list(self._open):
            self._close_id(tid, now)
        self._last_seen.clear()
        return self.flush_closed()

    def reset(self) -> None:
        """Clear all state — call on shift/session reset."""
        self._open.clear()
        self._last_seen.clear()
        self._closed_buffer.clear()

    @property
    def active_events(self) -> list[ViolationEvent]:
        return [ev for events in self._open.values() for ev in events.values()]

    def summary(self) -> dict:
        all_events = self.active_events + self._closed_buffer
        by_class: dict[str, int] = {}
        distinct: set[int] = set()
        total_secs = 0.0
        for ev in all_events:
            by_class[ev.violation_class] = by_class.get(ev.violation_class, 0) + 1
            distinct.add(ev.track_id)
            total_secs += ev.duration_secs
        return {
            "total_events"        : len(all_events),
            "distinct_violators"  : len(distinct),
            "total_violation_secs": round(total_secs, 1),
            "by_class"            : by_class,
        }

    # ── Private ───────────────────────────────────────────────────────────

    def _close_id(self, track_id: int, end_time: float) -> None:
        if track_id in self._open:
            for ev in self._open[track_id].values():
                ev.end_time = end_time
                self._closed_buffer.append(ev)
            del self._open[track_id]
