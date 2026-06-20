"""
tracker.py — Per-ID PPE violation state machine

Converts per-frame YOLO+ByteTrack detections into discrete violation events:
  - one ViolationEvent per (track_id, violation_class) incident
  - tracks start time, duration, and frame count
  - closes events when a worker becomes compliant or leaves the frame
  - tolerates brief occlusion with a configurable stale timeout

Usage:
    tracker = ViolationTracker()
    active = tracker.update([(7, "NO-Hardhat"), (12, "Person")])
    closed = tracker.flush_closed()  # events ready to persist
"""

import time
from dataclasses import dataclass, field
from typing import Optional


VIOLATION_CLASSES = frozenset({"NO-Hardhat", "NO-Mask", "NO-Safety Vest"})
COMPLIANT_CLASSES = frozenset({"Hardhat", "Mask", "Safety Vest"})

# Seconds a track ID can be absent before its open events are closed.
# Short enough to catch workers leaving frame; long enough to survive
# brief occlusion (another person walking in front).
_STALE_TIMEOUT_SECS = 3.0


@dataclass
class ViolationEvent:
    track_id:        int
    violation_class: str
    start_time:      float          # epoch seconds
    end_time:        Optional[float] = None
    frame_count:     int            = 1

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

    Call update() once per frame with the list of (track_id, class_name)
    detections from the YOLO tracker output.

    State machine per track ID:
        no open event  + violation detected  →  open event
        open event     + same violation      →  extend (increment frame_count)
        open event     + now compliant       →  close event
        open event     + not seen > timeout  →  close event (left frame)
    """

    def __init__(self, stale_timeout: float = _STALE_TIMEOUT_SECS):
        self._stale_timeout = stale_timeout
        # track_id → {violation_class → ViolationEvent}
        self._open: dict[int, dict[str, ViolationEvent]] = {}
        # track_id → last-seen epoch seconds
        self._last_seen: dict[int, float] = {}
        # events closed this session — flushed to DB by the caller
        self._closed_buffer: list[ViolationEvent] = []

    # ── Public API ────────────────────────────────────────────────────────

    def update(
        self,
        detections: list[tuple[int, str]],
    ) -> list[ViolationEvent]:
        """
        Process one frame of detections.

        detections: list of (track_id, class_name) pairs for every detected box.
        Returns the list of currently open ViolationEvents (for live display).
        """
        now = time.time()
        seen_ids: set[int] = set()

        for track_id, cls_name in detections:
            seen_ids.add(track_id)
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
                # Worker corrected their PPE — close any open violations
                self._close_id(track_id, now)

        # Close events for IDs absent longer than the stale timeout
        stale_ids = [
            tid for tid, last in self._last_seen.items()
            if tid not in seen_ids and (now - last) > self._stale_timeout
        ]
        for tid in stale_ids:
            self._close_id(tid, now)
            del self._last_seen[tid]

        return self.active_events

    def flush_closed(self) -> list[ViolationEvent]:
        """
        Return and clear events closed since the last flush.
        Call this each frame to drain events that are ready to be persisted.
        """
        closed, self._closed_buffer = self._closed_buffer, []
        return closed

    def close_all(self) -> list[ViolationEvent]:
        """Close every open event — call on stream end or session reset."""
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
        return [
            ev
            for events in self._open.values()
            for ev in events.values()
        ]

    def summary(self) -> dict:
        """Aggregate stats across all events (open + closed this session)."""
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
