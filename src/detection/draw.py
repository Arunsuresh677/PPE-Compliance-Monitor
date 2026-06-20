"""
src/detection/draw.py — Annotate frames with bounding boxes and status banners.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.tracking.violation_tracker import COMPLIANT_CLASSES, VIOLATION_CLASSES

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "Hardhat"       : (0, 220, 90),
    "Mask"          : (0, 200, 255),
    "Safety Vest"   : (0, 180, 80),
    "NO-Hardhat"    : (220, 30,  30),
    "NO-Mask"       : (200, 50,  200),
    "NO-Safety Vest": (220, 80,  0),
    "Person"        : (100, 180, 255),
    "Safety Cone"   : (255, 200, 0),
    "machinery"     : (160, 160, 160),
    "vehicle"       : (120, 120, 220),
}


def draw_detections(
    frame: np.ndarray,
    results,
    conf_thresh: float,
) -> tuple[np.ndarray, int, list[str], list[str], list[tuple[int | None, str]]]:
    """
    Annotate *frame* in-place with bounding boxes, track IDs, confidence
    labels, and a status banner.

    Returns:
        annotated_frame   — the same array, modified in-place
        total             — number of boxes drawn
        violations        — class names of violation boxes
        compliant         — class names of compliant boxes
        raw_detections    — [(track_id_or_None, class_name), …]
    """
    violations: list[str]                  = []
    compliant:  list[str]                  = []
    raw:        list[tuple[int | None, str]] = []
    total = 0

    for r in results:
        if r.boxes is None:
            continue
        track_ids: list[int | None] = (
            r.boxes.id.int().tolist()
            if r.boxes.id is not None
            else [None] * len(r.boxes)
        )
        for box, tid in zip(r.boxes, track_ids):
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            cls_id = int(box.cls[0])
            name   = r.names[cls_id]
            color  = CLASS_COLORS.get(name, (200, 200, 200))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            total += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            id_prefix = f"#{tid} " if tid is not None else ""
            label     = f"{id_prefix}{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

            raw.append((tid, name))
            if name in VIOLATION_CLASSES:
                violations.append(name)
            elif name in COMPLIANT_CLASSES:
                compliant.append(name)

    _draw_status_banner(frame, violations, compliant)
    return frame, total, violations, compliant, raw


def _draw_status_banner(
    frame: np.ndarray,
    violations: list[str],
    compliant: list[str],
) -> None:
    h, w = frame.shape[:2]
    if violations:
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 180), -1)
        cv2.putText(
            frame,
            f"  VIOLATION: {', '.join(set(violations))}",
            (6, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2,
        )
    elif compliant:
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 140, 40), -1)
        cv2.putText(
            frame, "  ALL PPE COMPLIANT",
            (6, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2,
        )
