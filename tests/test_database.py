"""Tests for ViolationRepository."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.database.repository import ViolationRepository
from src.tracking.violation_tracker import ViolationEvent


def _make_event(track_id: int = 1, cls: str = "NO-Hardhat", duration: float = 5.0) -> ViolationEvent:
    now = time.time()
    return ViolationEvent(
        track_id=track_id,
        violation_class=cls,
        start_time=now - duration,
        end_time=now,
        frame_count=30,
    )


def test_save_and_retrieve(tmp_db: ViolationRepository) -> None:
    ev = _make_event()
    tmp_db.save_violation(ev, session="s1")
    rows = tmp_db.get_violations("s1")
    assert len(rows) == 1
    assert rows[0]["violation_class"] == "NO-Hardhat"
    assert rows[0]["track_id"] == 1


def test_session_filtering(tmp_db: ViolationRepository) -> None:
    tmp_db.save_violation(_make_event(), session="s1")
    tmp_db.save_violation(_make_event(), session="s2")
    assert len(tmp_db.get_violations("s1")) == 1
    assert len(tmp_db.get_violations("s2")) == 1


def test_limit_respected(tmp_db: ViolationRepository) -> None:
    for i in range(10):
        tmp_db.save_violation(_make_event(track_id=i), session="s1")
    assert len(tmp_db.get_violations("s1", limit=5)) == 5


def test_summary_aggregation(tmp_db: ViolationRepository) -> None:
    tmp_db.save_violation(_make_event(1, "NO-Hardhat", 4.0), session="s1")
    tmp_db.save_violation(_make_event(2, "NO-Mask",    3.0), session="s1")
    summary = tmp_db.get_session_summary("s1")
    assert summary["total_events"] == 2
    assert summary["distinct_violators"] == 2
    assert summary["total_violation_secs"] > 0
    assert "NO-Hardhat" in summary["by_class"]


def test_empty_session_summary(tmp_db: ViolationRepository) -> None:
    summary = tmp_db.get_session_summary("nonexistent")
    assert summary["total_events"] == 0
    assert summary["by_class"] == {}


def test_batch_save(tmp_db: ViolationRepository) -> None:
    events = [_make_event(i) for i in range(5)]
    tmp_db.save_violations(events, session="batch")
    assert len(tmp_db.get_violations("batch")) == 5


def test_list_sessions(tmp_db: ViolationRepository) -> None:
    tmp_db.save_violation(_make_event(), session="alpha")
    tmp_db.save_violation(_make_event(), session="beta")
    sessions = tmp_db.list_sessions()
    assert "alpha" in sessions
    assert "beta" in sessions
