"""Tests for ViolationTracker state machine."""

from __future__ import annotations

import time

import pytest

from src.tracking.violation_tracker import ViolationEvent, ViolationTracker


@pytest.fixture()
def tracker() -> ViolationTracker:
    return ViolationTracker(stale_timeout=0.1)


def test_opens_event_on_first_violation(tracker: ViolationTracker) -> None:
    active = tracker.update([(1, "NO-Hardhat")])
    assert len(active) == 1
    assert active[0].track_id == 1
    assert active[0].violation_class == "NO-Hardhat"


def test_increments_frame_count(tracker: ViolationTracker) -> None:
    tracker.update([(1, "NO-Hardhat")])
    tracker.update([(1, "NO-Hardhat")])
    active = tracker.update([(1, "NO-Hardhat")])
    assert active[0].frame_count == 3


def test_closes_on_compliance(tracker: ViolationTracker) -> None:
    tracker.update([(1, "NO-Hardhat")])
    tracker.update([(1, "Hardhat")])   # compliant — must close
    active = tracker.active_events
    assert len(active) == 0
    closed = tracker.flush_closed()
    assert len(closed) == 1
    assert closed[0].is_closed


def test_closes_stale_after_timeout(tracker: ViolationTracker) -> None:
    tracker.update([(1, "NO-Hardhat")])
    time.sleep(0.15)          # wait past stale_timeout=0.1
    tracker.update([])        # empty frame triggers stale check
    assert tracker.active_events == []
    assert len(tracker.flush_closed()) == 1


def test_multiple_ids_tracked_independently(tracker: ViolationTracker) -> None:
    tracker.update([(1, "NO-Hardhat"), (2, "NO-Mask")])
    active = tracker.active_events
    ids = {e.track_id for e in active}
    assert ids == {1, 2}


def test_close_all_drains_open_events(tracker: ViolationTracker) -> None:
    tracker.update([(1, "NO-Hardhat"), (2, "NO-Safety Vest")])
    closed = tracker.close_all()
    assert len(closed) == 2
    assert tracker.active_events == []


def test_reset_clears_all_state(tracker: ViolationTracker) -> None:
    tracker.update([(1, "NO-Hardhat")])
    tracker.reset()
    assert tracker.active_events == []
    assert tracker.flush_closed() == []


def test_summary_counts_by_class(tracker: ViolationTracker) -> None:
    tracker.update([(1, "NO-Hardhat"), (2, "NO-Mask")])
    tracker.close_all()
    s = tracker.summary()
    assert s["total_events"] == 2
    assert s["distinct_violators"] == 2
    assert "NO-Hardhat" in s["by_class"]
    assert "NO-Mask" in s["by_class"]


def test_event_duration_is_positive(tracker: ViolationTracker) -> None:
    tracker.update([(1, "NO-Hardhat")])
    time.sleep(0.05)
    active = tracker.active_events
    assert active[0].duration_secs > 0


def test_to_dict_has_all_keys(tracker: ViolationTracker) -> None:
    tracker.update([(1, "NO-Hardhat")])
    ev = tracker.active_events[0]
    d = ev.to_dict()
    for key in ("track_id", "violation_class", "start_time", "end_time",
                "duration_secs", "frame_count"):
        assert key in d
