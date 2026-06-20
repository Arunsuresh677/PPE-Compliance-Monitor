"""Tests for draw_detections."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.detection.draw import draw_detections


def _make_result(class_name: str, conf: float = 0.91, track_id: int | None = 7):
    box = MagicMock()
    box.conf = [conf]
    box.cls  = [0]
    box.xyxy = [[10, 20, 200, 300]]
    if track_id is not None:
        box.id = MagicMock()
        box.id.int.return_value.tolist.return_value = [track_id]
    else:
        box.id = None

    r = MagicMock()
    r.boxes = [box]
    r.names = {0: class_name}
    return [r]


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_violation_class_appears_in_output() -> None:
    frame, total, violations, compliant, raw = draw_detections(
        _blank_frame(), _make_result("NO-Hardhat"), conf_thresh=0.5
    )
    assert total == 1
    assert "NO-Hardhat" in violations
    assert compliant == []
    assert raw == [(7, "NO-Hardhat")]


def test_compliant_class_appears_in_output() -> None:
    _, total, violations, compliant, raw = draw_detections(
        _blank_frame(), _make_result("Hardhat"), conf_thresh=0.5
    )
    assert "Hardhat" in compliant
    assert violations == []


def test_low_confidence_box_skipped() -> None:
    _, total, _, _, _ = draw_detections(
        _blank_frame(), _make_result("NO-Hardhat", conf=0.2), conf_thresh=0.5
    )
    assert total == 0


def test_no_track_id_renders_without_error() -> None:
    _, total, _, _, raw = draw_detections(
        _blank_frame(), _make_result("Person", track_id=None), conf_thresh=0.5
    )
    assert total == 1
    assert raw[0][0] is None


def test_multiple_boxes() -> None:
    r1 = _make_result("NO-Hardhat", track_id=1)
    r2 = _make_result("Mask", track_id=2)
    combined = r1 + r2
    _, total, violations, compliant, _ = draw_detections(
        _blank_frame(), combined, conf_thresh=0.5
    )
    assert total == 2
    assert "NO-Hardhat" in violations
    assert "Mask" in compliant
