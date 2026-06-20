# System Architecture

## Overview

PPE Compliance Monitor is a real-time computer-vision system that identifies missing
personal protective equipment on construction sites. It runs at the edge (a single
GPU workstation or Raspberry Pi 5) and requires no cloud connectivity after the
initial model download.

```
┌───────────────────────────────────────────────────────────────────┐
│                         Input Sources                              │
│    Webcam (WebRTC)    RTSP Feed    Video File    Still Image      │
└──────────────────────────────┬────────────────────────────────────┘
                               │ raw frames
                               ▼
┌─────────────────────── YOLOv8s ───────────────────────────────────┐
│  Detects 10 classes: Hardhat, NO-Hardhat, Mask, NO-Mask,         │
│  Safety Vest, NO-Safety Vest, Person, Safety Cone,               │
│  machinery, vehicle                                               │
│  640×640 inference  ·  ~15 ms/frame on RTX 3060                  │
└──────────────────────────────┬────────────────────────────────────┘
                               │ boxes + class ids + confidence
                               ▼
┌─────────────────────── ByteTrack ─────────────────────────────────┐
│  Assigns persistent integer track IDs across frames              │
│  Handles occlusion, re-entry, and multi-person scenes            │
│  Built into Ultralytics — zero extra dependencies                 │
└──────────────────────────────┬────────────────────────────────────┘
                               │ (track_id, class) per detection
                               ▼
┌─────────────────── ViolationTracker ──────────────────────────────┐
│  State machine: one ViolationEvent per (track_id, class)         │
│  Prevents per-frame duplicate events                              │
│  3-second stale-timeout handles brief occlusions                 │
│  Compliant detection immediately closes open violation            │
└──────────────────────────────┬────────────────────────────────────┘
                               │ closed ViolationEvents
                               ▼
┌─────────────────────── SQLite (WAL) ──────────────────────────────┐
│  Thread-safe: Lock serialises writers, WAL allows concurrent     │
│  readers (Streamlit main thread ↔ WebRTC callback thread)        │
│  violation_events table: session, track_id, class,               │
│  start_time, end_time, duration_secs, frame_count               │
└──────────────────┬────────────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 Streamlit UI            FastAPI REST
 (live dashboard)        (external consumers)
```

---

## Design Decisions

### Why YOLOv8s over YOLOv8n?

| Model   | mAP@0.5 | Latency (GPU) | Latency (CPU) |
|---------|---------|---------------|---------------|
| YOLOv8n | 0.71    | 6 ms          | 90 ms         |
| YOLOv8s | 0.856   | 15 ms         | 180 ms        |
| YOLOv8m | 0.88    | 28 ms         | 380 ms        |

The `s` (small) model delivers an acceptable 85.6% mAP at 15 ms per frame on a
mid-range GPU — fast enough for 30 FPS streams. The nano model's 71% mAP
produces too many missed hardhat detections on workers at distance. The medium
model yields only a 3% mAP gain at 2× the compute cost.

### Why ByteTrack over DeepSORT?

- **DeepSORT** requires a separate re-ID network (ResNet-50 or similar) to extract
  appearance features. That adds ~30 ms/frame and another model checkpoint to
  manage.
- **ByteTrack** uses only motion (Kalman filter + IoU matching). It achieves
  comparable MOTA on the MOT17 benchmark (77.8 vs 75.2) with no extra model.
- **Ultralytics ships ByteTrack natively** — enabling it is a single parameter
  change: `model.track(tracker="bytetrack.yaml", persist=True)`. No third-party
  package needed.

### Why SQLite over PostgreSQL?

- Single-node deployment: one SQLite file is simpler to back up and replicate
  than a running Postgres instance.
- WAL mode + a threading.Lock gives sufficient concurrent-access safety for the
  two-thread pattern (Streamlit + WebRTC callback).
- If multi-site centralisation becomes a requirement, migrating from SQLite to
  Postgres requires only swapping the `repository.py` implementation behind the
  same interface — the rest of the codebase is unaffected.

### Why Pydantic BaseSettings?

- All tuneable parameters (thresholds, paths, ports) live in one typed class.
- Overriding for production requires no code changes: set a `PPE_*` env var or
  `.env` file and the application picks it up at startup.
- IDE auto-complete works correctly because every setting is a typed field.

---

## Threading Model

```
Main thread (Streamlit)
  └─ reads SQLite (WAL reader — no lock needed)
  └─ renders dashboard / charts

WebRTC callback thread (per session)
  └─ PPEProcessor.recv()
       ├─ YOLO.track()  ← each PPEProcessor owns its own YOLO instance
       ├─ ViolationTracker.update()
       └─ repo.save_violations()  ← Lock acquired here
```

Each `PPEProcessor` owns its own YOLO model instance so `model.track(persist=True)`
state (ByteTrack's Kalman filter and track ID counters) is isolated per webcam
session. Sharing a single YOLO instance across sessions would cause track IDs to
collide between users.

---

## Scaling to 100 Cameras

The current architecture runs comfortably on 1–4 simultaneous RTSP streams
with a single mid-range GPU. To scale to 100 cameras:

1. **Inference farm**: deploy N GPU workers (each running `detect.py` against a
   partition of RTSP streams). Results are written to a central DB.
2. **Central database**: swap `ViolationRepository` to PostgreSQL or TimescaleDB.
   The existing interface (`save_violations`, `get_session_summary`) needs no
   change in callers.
3. **Message bus** (optional): add a Redis Streams or Kafka topic between workers
   and the DB to absorb write bursts during shift-start.
4. **Dashboard**: point the Streamlit app at the central DB. No other code
   changes are required because all queries go through `ViolationRepository`.

Estimated capacity per RTX 4090: ~12 × 1080p RTSP streams at 15 FPS with
YOLOv8s (batch inference).
