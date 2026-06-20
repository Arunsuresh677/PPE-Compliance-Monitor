"""
src/alerts/slack.py — Slack webhook alert sender for PPE violations and camera outages.

Set PPE_SLACK_WEBHOOK_URL in .env to enable. If unset, alerts are logged only.
"""

from __future__ import annotations

import logging
import time

import requests

from src.config.settings import settings

log = logging.getLogger(__name__)

_VIOLATION_COLORS = {
    "NO-Hardhat":     "#E53E3E",
    "NO-Mask":        "#DD6B20",
    "NO-Safety Vest": "#D69E2E",
}
_DEFAULT_COLOR = "#E53E3E"


class SlackAlerter:
    """Sends formatted Slack messages via incoming webhook."""

    def __init__(self) -> None:
        self._webhook = settings.slack_webhook_url
        self._enabled = bool(self._webhook)
        if not self._enabled:
            log.warning("PPE_SLACK_WEBHOOK_URL not set — Slack alerts disabled")

    def send_violation_alert(
        self,
        camera_id:       str,
        track_id:        int,
        violation_class: str,
        duration_secs:   float,
    ) -> None:
        color = _VIOLATION_COLORS.get(violation_class, _DEFAULT_COLOR)
        payload = {
            "attachments": [{
                "color": color,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"⚠️ PPE Violation — {violation_class}",
                        },
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Camera:*\n`{camera_id}`"},
                            {"type": "mrkdwn", "text": f"*Worker ID:*\n#{track_id}"},
                            {"type": "mrkdwn", "text": f"*Violation:*\n{violation_class}"},
                            {"type": "mrkdwn", "text": f"*Duration:*\n{duration_secs:.1f}s"},
                        ],
                    },
                    {
                        "type": "context",
                        "elements": [{
                            "type": "mrkdwn",
                            "text": f"PPE Monitor · {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        }],
                    },
                ],
            }]
        }
        self._post(payload, f"violation alert camera={camera_id} worker=#{track_id}")

    def send_camera_down_alert(self, camera_id: str, reason: str = "stream lost") -> None:
        payload = {
            "attachments": [{
                "color": "#718096",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "🔴 Camera Offline"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Camera:*\n`{camera_id}`"},
                            {"type": "mrkdwn", "text": f"*Reason:*\n{reason}"},
                        ],
                    },
                    {
                        "type": "context",
                        "elements": [{
                            "type": "mrkdwn",
                            "text": f"PPE Monitor · {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        }],
                    },
                ],
            }]
        }
        self._post(payload, f"camera down alert camera={camera_id}")

    def send_camera_recovered_alert(self, camera_id: str) -> None:
        payload = {
            "attachments": [{
                "color": "#38A169",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "🟢 Camera Recovered"},
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Camera `{camera_id}` is back online."},
                    },
                ],
            }]
        }
        self._post(payload, f"camera recovered alert camera={camera_id}")

    def _post(self, payload: dict, description: str) -> None:
        if not self._enabled:
            log.info("[slack] (disabled) would send: %s", description)
            return
        try:
            resp = requests.post(self._webhook, json=payload, timeout=5)
            if resp.status_code != 200:
                log.error("[slack] webhook returned %d: %s", resp.status_code, resp.text)
            else:
                log.info("[slack] sent: %s", description)
        except requests.RequestException as exc:
            log.error("[slack] failed to send %s: %s", description, exc)
