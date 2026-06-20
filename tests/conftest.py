"""Shared pytest fixtures."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.database.repository import ViolationRepository
from src.tracking.violation_tracker import ViolationEvent


@pytest.fixture()
def tmp_db(tmp_path: Path) -> ViolationRepository:
    """Fresh in-memory (temp-file) SQLite repository for each test."""
    repo = ViolationRepository(db_path=str(tmp_path / "test.db"))
    repo.init()
    yield repo
    repo.close()


@pytest.fixture()
def closed_event() -> ViolationEvent:
    now = time.time()
    ev = ViolationEvent(
        track_id=1,
        violation_class="NO-Hardhat",
        start_time=now - 5.0,
        end_time=now,
        frame_count=30,
    )
    return ev


@pytest.fixture()
def mock_yolo_results():
    """Minimal YOLO result stub — one NO-Hardhat box with track_id=7."""
    box = MagicMock()
    box.conf = [0.91]
    box.cls  = [0]         # index into names dict
    box.xyxy = [[10, 20, 100, 120]]
    box.id   = MagicMock()
    box.id.int.return_value.tolist.return_value = [7]

    result = MagicMock()
    result.boxes = [box]
    result.names = {0: "NO-Hardhat"}
    return [result]
