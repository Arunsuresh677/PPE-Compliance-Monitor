"""
src/camera/manager.py — Manages a fleet of CameraStream instances.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.camera.stream import CameraStream
from src.database.repository import ViolationRepository

log = logging.getLogger(__name__)


class CameraManager:
    """Add, remove, start, and stop multiple camera streams."""

    def __init__(self, session: str, repo: ViolationRepository, model_path: str = "best.pt") -> None:
        self._session    = session
        self._repo       = repo
        self._model_path = model_path
        self._cameras: dict[str, CameraStream] = {}

    # ── Camera lifecycle ───────────────────────────────────────────────────

    def add_camera(self, camera_id: str, source: str | int, conf: float = 0.5) -> None:
        if camera_id in self._cameras:
            log.warning("Camera %s already exists — ignoring", camera_id)
            return
        cam = CameraStream(
            camera_id=camera_id,
            source=source,
            session=self._session,
            repo=self._repo,
            model_path=self._model_path,
            conf=conf,
        )
        self._cameras[camera_id] = cam
        cam.start()
        log.info("Added camera: %s → %s", camera_id, source)

    def remove_camera(self, camera_id: str) -> None:
        cam = self._cameras.pop(camera_id, None)
        if cam:
            cam.stop()
            log.info("Removed camera: %s", camera_id)

    def stop_all(self) -> None:
        for cam in self._cameras.values():
            cam.stop()
        self._cameras.clear()

    # ── Data access ────────────────────────────────────────────────────────

    def get_frames(self) -> dict[str, np.ndarray | None]:
        return {cid: cam.latest_frame for cid, cam in self._cameras.items()}

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        return {cid: cam.stats for cid, cam in self._cameras.items()}

    def set_conf(self, conf: float) -> None:
        for cam in self._cameras.values():
            cam.set_conf(conf)

    @property
    def camera_ids(self) -> list[str]:
        return list(self._cameras.keys())

    @property
    def count(self) -> int:
        return len(self._cameras)
