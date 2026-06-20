"""
src/alerts/manager.py — Deduplicates and throttles Slack alerts.

Alert rules:
  - Violation open > PPE_ALERT_VIOLATION_SECS → fire once per (camera, worker, class)
  - Camera down → fire once; fire recovery when it comes back
  - No re-alert for the same violation until it closes and reopens
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.alerts.slack import SlackAlerter
from src.config.settings import settings
from src.tracking.violation_tracker import ViolationEvent

log = logging.getLogger(__name__)


class AlertManager:
    """
    Checks active ViolationEvents and camera status against alert thresholds.
    Call check_violations() once per frame, check_camera_status() on status change.
    """

    def __init__(self) -> None:
        self._slack = SlackAlerter()
        self._threshold = settings.alert_violation_secs
        # Keys already alerted: (camera_id, track_id, violation_class)
        self._alerted_violations: set[tuple[str, int, str]] = set()
        # Camera down state: camera_id → True if down alert already sent
        self._camera_down: dict[str, bool] = {}

    def check_violations(
        self,
        camera_id: str,
        active_events: list[ViolationEvent],
    ) -> None:
        """Fire alerts for violations that have exceeded the threshold."""
        active_keys: set[tuple[str, int, str]] = set()

        for ev in active_events:
            key = (camera_id, ev.track_id, ev.violation_class)
            active_keys.add(key)

            if key not in self._alerted_violations:
                if ev.duration_secs >= self._threshold:
                    log.warning(
                        "[alert] violation threshold exceeded: camera=%s worker=#%d class=%s duration=%.1fs",
                        camera_id, ev.track_id, ev.violation_class, ev.duration_secs,
                    )
                    self._slack.send_violation_alert(
                        camera_id=camera_id,
                        track_id=ev.track_id,
                        violation_class=ev.violation_class,
                        duration_secs=ev.duration_secs,
                    )
                    self._alerted_violations.add(key)

        # Clear alerted state for violations that are no longer active
        stale = {k for k in self._alerted_violations if k[0] == camera_id and k not in active_keys}
        self._alerted_violations -= stale

    def check_camera_status(self, camera_id: str, is_live: bool) -> None:
        """Fire down/recovery alerts based on camera health transitions."""
        was_down = self._camera_down.get(camera_id, False)

        if not is_live and not was_down:
            self._camera_down[camera_id] = True
            self._slack.send_camera_down_alert(camera_id)

        elif is_live and was_down:
            self._camera_down[camera_id] = False
            self._slack.send_camera_recovered_alert(camera_id)
