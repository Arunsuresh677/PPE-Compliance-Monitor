"""
src/camera/stream.py — Background thread that reads one camera source,
runs YOLOv8 + ByteTrack inference, and persists violations to PostgreSQL.

Supports:
  - Webcam index  (e.g. source=0)
  - RTSP URL      (e.g. source="rtsp://192.168.1.10:554/stream")
  - HTTP MJPEG    (e.g. source="http://192.168.1.10:8080/video")
  - Local file    (e.g. source="site_footage.mp4")
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.database.repository import ViolationRepository
from src.detection.draw import draw_detections
from src.tracking.violation_tracker import ViolationTracker
from src.metrics.prometheus import (
    INFERENCE_LATENCY, FPS_GAUGE, VIOLATIONS_COUNTER,
    ACTIVE_WORKERS, CAMERA_UP, EVENTS_SAVED, DB_WRITE_LATENCY,
)

log = logging.getLogger(__name__)

_RECONNECT_DELAY = 5.0   # seconds between reconnect attempts
_MAX_RETRIES     = 10


class CameraStream:
    """
    One camera = one background thread.

    Thread-safe: latest_frame and stats are accessed under self.lock.
    """

    def __init__(
        self,
        camera_id: str,
        source:    str | int,
        session:   str,
        repo:      ViolationRepository,
        model_path: str = "best.pt",
        conf:      float = 0.5,
    ) -> None:
        self.camera_id = camera_id
        self.source    = source
        self.session   = session

        self._repo    = repo
        self._conf    = conf
        self._model   = YOLO(model_path)
        self._tracker = ViolationTracker()

        self.lock:          threading.Lock    = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._stats: dict[str, Any] = {
            "fps": 0.0, "total": 0, "violations": 0,
            "compliant": 0, "active_events": 0, "distinct_violators": 0,
            "status": "stopped",
        }

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.camera_id}", daemon=True
        )
        self._thread.start()
        log.info("[%s] stream started — source=%s", self.camera_id, self.source)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        with self.lock:
            self._stats["status"] = "stopped"
        log.info("[%s] stream stopped", self.camera_id)

    @property
    def latest_frame(self) -> np.ndarray | None:
        with self.lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def stats(self) -> dict[str, Any]:
        with self.lock:
            return dict(self._stats)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_conf(self, conf: float) -> None:
        with self.lock:
            self._conf = conf

    # ── Internal ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        retries = 0
        while not self._stop_event.is_set() and retries < _MAX_RETRIES:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                retries += 1
                log.warning("[%s] cannot open source (attempt %d/%d)", self.camera_id, retries, _MAX_RETRIES)
                with self.lock:
                    self._stats["status"] = f"reconnecting ({retries}/{_MAX_RETRIES})"
                self._stop_event.wait(_RECONNECT_DELAY)
                continue

            retries = 0
            with self.lock:
                self._stats["status"] = "live"
            CAMERA_UP.labels(camera_id=self.camera_id).set(1)
            log.info("[%s] connected", self.camera_id)

            last_ts = time.monotonic()

            try:
                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        log.warning("[%s] lost stream", self.camera_id)
                        break

                    now     = time.monotonic()
                    elapsed = now - last_ts
                    fps     = 1.0 / elapsed if elapsed > 0 else 0.0
                    last_ts = now

                    with self.lock:
                        conf = self._conf

                    t_infer = time.monotonic()
                    results = self._model.track(
                        frame,
                        conf=conf,
                        tracker="bytetrack.yaml",
                        persist=True,
                        verbose=False,
                    )
                    INFERENCE_LATENCY.labels(camera_id=self.camera_id).observe(
                        time.monotonic() - t_infer
                    )

                    out_frame, total, viols, comp, raw = draw_detections(frame, results, conf)

                    for v in viols:
                        VIOLATIONS_COUNTER.labels(camera_id=self.camera_id, violation_class=v).inc()

                    tracked = [(tid, cls) for tid, cls in raw if tid is not None]
                    active  = self._tracker.update(tracked)
                    closed  = self._tracker.flush_closed()
                    if closed:
                        t_db = time.monotonic()
                        self._repo.save_violations(closed, self.session, camera_id=self.camera_id)
                        DB_WRITE_LATENCY.observe(time.monotonic() - t_db)
                        EVENTS_SAVED.labels(camera_id=self.camera_id).inc(len(closed))

                    distinct = len({ev.track_id for ev in active})
                    ACTIVE_WORKERS.labels(camera_id=self.camera_id).set(distinct)

                    cv2.putText(
                        out_frame,
                        f"{self.camera_id}  {fps:.1f} FPS",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (88, 166, 255), 2,
                    )

                    FPS_GAUGE.labels(camera_id=self.camera_id).set(fps)

                    with self.lock:
                        self._latest_frame = out_frame
                        self._stats = {
                            "fps"               : round(fps, 1),
                            "total"             : total,
                            "violations"        : len(viols),
                            "compliant"         : len(comp),
                            "active_events"     : len(active),
                            "distinct_violators": distinct,
                            "status"            : "live",
                        }
            finally:
                cap.release()

            CAMERA_UP.labels(camera_id=self.camera_id).set(0)
            if not self._stop_event.is_set():
                retries += 1
                self._stop_event.wait(_RECONNECT_DELAY)

        # Flush remaining open events
        for ev in self._tracker.close_all():
            self._repo.save_violation(ev, self.session, camera_id=self.camera_id)

        with self.lock:
            self._stats["status"] = "stopped"
