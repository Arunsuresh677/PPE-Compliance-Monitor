"""
src/metrics/prometheus.py — Prometheus metrics registry for PPE Monitor.

Metrics exposed:
  ppe_inference_duration_seconds  — YOLO inference latency per camera
  ppe_fps                         — current FPS per camera
  ppe_violations_total            — violation detections per camera/class
  ppe_active_workers              — workers currently tracked per camera
  ppe_camera_up                   — 1=live, 0=down per camera
  ppe_events_saved_total          — violation events persisted to DB per camera
  ppe_db_write_duration_seconds   — PostgreSQL write latency
"""

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, REGISTRY

# ── Inference ──────────────────────────────────────────────────────────────────
INFERENCE_LATENCY = Histogram(
    "ppe_inference_duration_seconds",
    "YOLOv8 inference latency per frame",
    labelnames=["camera_id"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
)

FPS_GAUGE = Gauge(
    "ppe_fps",
    "Current frames per second per camera",
    labelnames=["camera_id"],
)

# ── Detections ─────────────────────────────────────────────────────────────────
VIOLATIONS_COUNTER = Counter(
    "ppe_violations_total",
    "Total violation detections per camera and class",
    labelnames=["camera_id", "violation_class"],
)

ACTIVE_WORKERS = Gauge(
    "ppe_active_workers",
    "Number of workers currently being tracked per camera",
    labelnames=["camera_id"],
)

# ── Camera health ──────────────────────────────────────────────────────────────
CAMERA_UP = Gauge(
    "ppe_camera_up",
    "Camera stream health: 1=live, 0=down",
    labelnames=["camera_id"],
)

# ── Persistence ────────────────────────────────────────────────────────────────
EVENTS_SAVED = Counter(
    "ppe_events_saved_total",
    "Total violation events persisted to PostgreSQL per camera",
    labelnames=["camera_id"],
)

DB_WRITE_LATENCY = Histogram(
    "ppe_db_write_duration_seconds",
    "PostgreSQL write latency for violation events",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
