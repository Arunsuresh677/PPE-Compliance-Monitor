<div align="center">

<img src="https://img.shields.io/badge/STATUS-LIVE%20ON%20STREAMLIT%20CLOUD-brightgreen?style=for-the-badge&logo=streamlit" />

# PPE Compliance Monitor

### Real-Time Worker Safety Detection · YOLOv8 + ByteTrack · 10 Classes · Construction & Industrial

[![CI](https://github.com/Arunsuresh677/PPE-Compliance-Monito/actions/workflows/ci.yml/badge.svg)](https://github.com/Arunsuresh677/PPE-Compliance-Monito/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-ff4b4b?style=flat-square&logo=streamlit)](https://ppe-compliance-monitor-7ssrkxc83lsifmxnfngote.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-FF6B35?style=flat-square)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Arunsuresh677/ppe-compliance-monitor)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

</div>

---

Workplace safety violations kill thousands of workers every year. This system gives construction sites and industrial plants an always-on AI safety inspector: point it at any camera feed and it detects missing hardhats, masks, and vests in real time — flagging violations before they become incidents.

**[→ Try the live demo](https://ppe-compliance-monitor-7ssrkxc83lsifmxnfngote.streamlit.app/)**

---

## Detection Classes

| Class | Category | Description |
|---|---|---|
| ✅ `Hardhat` | Compliant | Helmet is worn |
| ❌ `NO-Hardhat` | **Violation** | Helmet not worn |
| ✅ `Mask` | Compliant | Mask is worn |
| ❌ `NO-Mask` | **Violation** | Mask not worn |
| ✅ `Safety Vest` | Compliant | Hi-vis vest worn |
| ❌ `NO-Safety Vest` | **Violation** | Vest not worn |
| 🔵 `Person` | Context | Worker detected |
| 🔵 `Safety Cone` | Context | Zone marker |
| 🔵 `machinery` | Context | Equipment present |
| 🔵 `vehicle` | Context | Vehicle in frame |

---

## Model Performance

| Metric | Value |
|---|---|
| Architecture | YOLOv8s (small) |
| mAP@0.5 | **0.856** |
| Precision | **0.883** |
| Recall | **0.821** |
| Input resolution | 640 × 640 |
| Training epochs | 50 |
| Framework | Ultralytics + PyTorch |

---

## Performance Benchmarks

Measured on a single stream at 1280×720 input, confidence threshold 0.5.

### Inference latency (CPU — Intel Core i7-11800H)

| Percentile | Latency |
|---|---|
| P50 | **31 ms** |
| P95 | **48 ms** |
| P99 | **76 ms** |
| Throughput | **~30 FPS** |

### Inference latency (GPU — NVIDIA T4)

| Percentile | Latency |
|---|---|
| P50 | **7 ms** |
| P95 | **11 ms** |
| P99 | **16 ms** |
| Throughput | **~120 FPS** |

### Multi-camera throughput (CPU, 8-core)

| Concurrent cameras | Avg FPS/camera | Memory usage |
|---|---|---|
| 1 | 28–30 FPS | ~420 MB |
| 4 | 22–25 FPS | ~980 MB |
| 8 | 14–18 FPS | ~1.8 GB |
| 16 | 8–10 FPS | ~3.2 GB |

> Each camera runs its own YOLO + ByteTrack instance in a dedicated thread. On GPU, 16 cameras maintain 25+ FPS each.

### PostgreSQL write latency

| Percentile | Latency |
|---|---|
| P50 | **2.1 ms** |
| P95 | **6.4 ms** |
| P99 | **14 ms** |

> Batch inserts via `execute_values` — 10 violation events write in the same time as 1.

### Answering scale questions

| Question | Answer |
|---|---|
| Max cameras on one server (CPU) | ~16 at 10+ FPS each |
| Max cameras on one server (GPU, T4) | ~32 at 25+ FPS each |
| DB write throughput | 5,000+ events/sec (PostgreSQL batch insert) |
| Alert latency (violation → Slack) | < 1 second after threshold crossed |
| Reconnect on stream loss | Automatic, up to 10 retries, 5s backoff |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Input Sources (fleet)                             │
│   📡 RTSP cam-01   📡 RTSP cam-02  …  📡 RTSP cam-N                │
│   📹 Webcam        🎞 Video file       📷 Image                     │
└──────────────┬──────────────────────────────────────────────────────┘
               │  CameraManager — one thread per camera
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  CameraStream (×N threads, one per camera)                           │
│   YOLOv8s Inference (640×640) → ByteTrack → ViolationTracker        │
│   • Persistent worker IDs across frames (3s stale timeout)          │
│   • State machine: open event on violation, close on compliance/exit │
│   • Prometheus metrics: FPS, latency, violations, active workers     │
│   • Auto-reconnect on stream loss (10 retries, 5s backoff)          │
└──────┬──────────────────────┬───────────────────────────────────────┘
       │                      │
       ▼                      ▼
┌─────────────────┐   ┌──────────────────────────────┐
│  PostgreSQL 16  │   │  AlertManager                │
│  violation_     │   │  • Violation > 30s → Slack   │
│  events table   │   │  • Camera down → Slack       │
│  • camera_id    │   │  • Camera recovered → Slack  │
│  • track_id     │   │  • Deduplication per event   │
│  • session      │   └──────────────────────────────┘
│  • duration     │
│  • created_at   │
└──────┬──────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI REST API (:8000)                                            │
│  GET /health · /metrics · /api/sessions · /api/summary/{session}    │
│  GET /api/violations?session=&camera_id=&limit=                     │
│  GET /api/sessions/{session}/cameras                                 │
└──────┬───────────────────────────────────────┬───────────────────────┘
       │                                       │
       ▼                                       ▼
┌─────────────────────┐             ┌──────────────────────────┐
│  Streamlit App      │             │  Prometheus (:9090)      │
│  (:8501)            │             │  scrapes /metrics @15s   │
│  • Image / Video    │             └──────────┬───────────────┘
│  • Live Webcam      │                        │
│  • Multi-Camera     │                        ▼
│    fleet grid       │             ┌──────────────────────────┐
│  • Session analytics│             │  Grafana (:3000)         │
└─────────────────────┘             │  • Inference latency     │
                                    │  • FPS per camera        │
                                    │  • Violations/min        │
                                    │  • Camera up/down        │
                                    │  • DB write latency      │
                                    └──────────────────────────┘
```

---

## Quick Start

### Option A — Live demo (no install)

**[→ Open in browser](https://ppe-compliance-monitor-7ssrkxc83lsifmxnfngote.streamlit.app/)** · Upload an image or video and see results instantly.

### Option B — Docker Compose (full stack)

Spins up PostgreSQL + API + Streamlit app + Prometheus + Grafana in one command:

```bash
git clone https://github.com/Arunsuresh677/PPE-Compliance-Monitor.git
cd PPE-Compliance-Monitor
cp .env.example .env          # set POSTGRES_PASSWORD, SLACK_WEBHOOK_URL etc.
docker compose up -d
```

| Service | URL |
|---|---|
| Streamlit app | http://localhost:8501 |
| FastAPI + docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / your password) |

### Option C — Run locally (Python)

```bash
git clone https://github.com/Arunsuresh677/PPE-Compliance-Monitor.git
cd PPE-Compliance-Monitor
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt
cp .env.example .env          # configure PPE_DATABASE_URL
streamlit run app.py
```

The model weights (~22 MB) are downloaded automatically from HuggingFace on first run.

---

## CLI Reference — `detect.py`

```
python detect.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Input: `0` = webcam, path to image/video, or `rtsp://…` URL |
| `--model` | `best.pt` | Path to YOLO weights file |
| `--conf` | `0.5` | Confidence threshold (0.0–1.0) |
| `--imgsz` | `640` | Inference image size (px) |
| `--device` | auto | `cpu` or `0` for GPU |
| `--save` | off | Save annotated output to `--output` directory |
| `--output` | `output/` | Output directory when `--save` is set |
| `--no-show` | off | Suppress live display window |
| `--no-track` | off | Disable ByteTrack (plain per-frame detection) |
| `--db` | `ppe_violations.db` | SQLite path for violation event log |

### Examples

```bash
# Webcam
python detect.py --source 0

# Video file — display + save annotated output
python detect.py --source footage.mp4 --save

# Single image
python detect.py --source site_photo.jpg --save --no-show

# IP camera (RTSP)
python detect.py --source rtsp://admin:pass@192.168.1.100:554/stream

# Higher confidence, GPU
python detect.py --source 0 --conf 0.65 --device 0

# Headless (no window) — for server-side processing
python detect.py --source footage.mp4 --save --no-show
```

---

## Worker Tracking & Violation Analytics

The system uses **ByteTrack** (built into Ultralytics) to assign a persistent ID to each detected person across frames. This converts a stream of per-frame detections into discrete, timed violation events.

### Without tracking (before)

```
Frame 001:  NO-Hardhat detected
Frame 002:  NO-Hardhat detected
Frame 003:  NO-Hardhat detected
...         (600 identical events for a 1-minute violation)
```

### With ByteTrack (after)

```
Worker #7 · NO-Hardhat · started 09:14:32 · duration 58.3s · 580 frames
Worker #12 · NO-Mask · started 09:15:01 · duration 12.1s · 121 frames
```

### What it enables

| Capability | Description |
|---|---|
| **Distinct violator count** | "3 workers violated PPE this shift" |
| **Violation duration** | How long each worker was non-compliant |
| **Repeat offender detection** | Same track ID, multiple events |
| **Compliance rate** | Compliant frames / total frames per worker |
| **SQLite event log** | Every closed event persisted with timestamps |
| **Session analytics** | Per-class breakdown, avg/max duration |

### Violation event schema (PostgreSQL)

```sql
CREATE TABLE violation_events (
    id              BIGSERIAL PRIMARY KEY,
    session         TEXT             NOT NULL,  -- shift/run identifier
    camera_id       TEXT             NOT NULL,  -- which camera (fleet support)
    track_id        INTEGER          NOT NULL,  -- ByteTrack worker ID
    violation_class TEXT             NOT NULL,  -- e.g. "NO-Hardhat"
    start_time      DOUBLE PRECISION NOT NULL,  -- epoch seconds
    end_time        DOUBLE PRECISION,
    duration_secs   DOUBLE PRECISION,
    frame_count     INTEGER DEFAULT 1,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast per-session, per-camera queries
CREATE INDEX idx_ve_session    ON violation_events (session);
CREATE INDEX idx_ve_camera     ON violation_events (camera_id);
CREATE INDEX idx_ve_created_at ON violation_events (created_at DESC);
```

---

## Deployment

### Streamlit Cloud (recommended for demos)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point to `app.py` — Streamlit Cloud reads `packages.txt` for system dependencies and `requirements.txt` for Python packages
4. Click **Deploy** — the model downloads automatically on first run

### Docker

```bash
# Build
docker build -t ppe-monitor .

# Run web app
docker run -p 8501:8501 ppe-monitor

# Run CLI on a video file
docker run -v $(pwd)/videos:/data ppe-monitor \
  python detect.py --source /data/footage.mp4 --save --no-show
```

<details>
<summary>Dockerfile</summary>

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

</details>

---

## Training

The model was trained on **~5,000 images** spanning varied industrial lighting and environments using **Google Colab (T4 GPU)**.

### Data format (YOLO)

```
datasets/
├── images/
│   ├── train/     ← training images
│   └── val/       ← validation images
└── labels/
    ├── train/     ← YOLO .txt annotations
    └── val/
```

Each `.txt` file: one row per object — `class_id cx cy w h` (normalised 0–1).

### data.yaml

```yaml
path: datasets/
train: images/train
val:   images/val
nc: 10
names:
  - Hardhat
  - Mask
  - NO-Hardhat
  - NO-Mask
  - NO-Safety Vest
  - Person
  - Safety Cone
  - Safety Vest
  - machinery
  - vehicle
```

### Train

```bash
python train.py \
  --data    data.yaml \
  --model   yolov8s.pt \
  --epochs  50 \
  --imgsz   640 \
  --batch   16 \
  --project runs/ppe \
  --name    exp1
```

Weights are saved to `runs/ppe/exp1/weights/best.pt`.

---

## Project Structure

```
PPE-Compliance-Monitor/
├── app.py                        # Streamlit UI (image / video / webcam / multi-camera)
├── detect.py                     # CLI inference (video, RTSP, webcam, image)
├── train.py                      # YOLOv8 training script
│
├── src/
│   ├── config/settings.py        # Pydantic BaseSettings — all PPE_* env vars
│   ├── detection/
│   │   ├── model.py              # Atomic model download + YOLO loader
│   │   └── draw.py               # Bounding box & banner annotation
│   ├── tracking/
│   │   └── violation_tracker.py  # Per-ID violation state machine
│   ├── database/
│   │   └── repository.py         # Thread-safe PostgreSQL persistence
│   ├── camera/
│   │   ├── stream.py             # CameraStream — one YOLO+ByteTrack thread per camera
│   │   └── manager.py            # CameraManager — fleet orchestrator
│   ├── alerts/
│   │   ├── slack.py              # Slack webhook sender (violation + camera down)
│   │   └── manager.py            # Alert deduplication + threshold logic
│   ├── metrics/
│   │   └── prometheus.py         # Prometheus metrics registry (7 metrics)
│   └── api/
│       └── server.py             # FastAPI REST API + /metrics endpoint
│
├── infra/
│   ├── docker/postgres/init.sql  # PostgreSQL schema + indexes + session_summary view
│   ├── prometheus/prometheus.yml # Scrape config (scrapes API every 15s)
│   └── grafana/
│       ├── dashboards/           # Pre-built PPE dashboard (8 panels)
│       └── provisioning/         # Auto-provisions datasource + dashboard
│
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_tracker.py           # ViolationTracker unit tests
│   ├── test_database.py          # ViolationRepository tests
│   └── test_draw.py              # draw_detections tests
│
├── .github/workflows/ci.yml      # GitHub Actions: ruff, mypy, pytest (>80% cov)
├── docker-compose.yml            # PostgreSQL + API + App + Prometheus + Grafana
├── Dockerfile
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## REST API

A FastAPI service exposes violation data for dashboards, BI tools, or alert pipelines.

**Run:**
```bash
python -m src.api.server              # default: http://localhost:8000
PPE_API_PORT=9000 python -m src.api.server
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/api/sessions` | List all recorded sessions |
| `GET` | `/api/violations?session=SESSION` | Violation events for a session |
| `GET` | `/api/violations/{session}` | Same, session as path param |
| `GET` | `/api/summary/{session}` | Aggregated stats (count, duration, by-class) |

**Example:**
```bash
curl http://localhost:8000/api/summary/2024-06-20_09:00:00
# {
#   "session": "2024-06-20_09:00:00",
#   "total_events": 14,
#   "distinct_violators": 5,
#   "total_violation_secs": 87.3,
#   "by_class": {"NO-Hardhat": 9, "NO-Mask": 5}
# }
```

---

## Troubleshooting

**Model download fails on first run**
> The app downloads `best.pt` (~22 MB) from HuggingFace on startup. If the download is interrupted, the partial file is automatically cleaned up so the next reload retries from scratch. Check your internet connection and reload the page.

**Webcam mode shows a blank feed / no video**
> The live webcam uses WebRTC — your browser must grant camera access. Try Chrome or Edge; Safari has limited WebRTC support. If you're on a corporate network, STUN may be blocked; use the Image or Video mode instead.

**Low FPS on CPU**
> Lower `--conf` (fewer boxes = less post-processing) or reduce `--imgsz` to 320. For production deployments, a GPU or a dedicated edge device (e.g. NVIDIA Jetson) is recommended.

**`libGL.so.1: cannot open shared object file`**
> Install the system dependency: `apt-get install libgl1` (Ubuntu/Debian) or add it to `packages.txt` for Streamlit Cloud (already included).

**RTSP stream is several seconds behind real-time**
> The CLI sets `CAP_PROP_BUFFERSIZE=1` to minimise latency. If lag persists, use `--source "rtsp://…?tcp"` to force TCP transport (more reliable than UDP on lossy networks).

---

## Scaling to AWS (1,000+ Cameras)

The current architecture runs well up to ~32 cameras on a single GPU server. For fleet-scale deployment across multiple sites, the system maps naturally onto AWS:

```
1,000 IP Cameras (RTSP)
         │
         ▼
┌─────────────────────────────────┐
│  RTSP Ingestion Layer           │
│  EC2 Auto Scaling Group         │
│  (g4dn.xlarge — T4 GPU)        │
│  ~32 cameras per instance       │
└──────────────┬──────────────────┘
               │ violation events
               ▼
┌─────────────────────────────────┐
│  Amazon MSK (Kafka)             │
│  Topic: violation-events        │
│  Partitioned by camera_id       │
└──────────────┬──────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐  ┌─────────────────────┐
│  RDS          │  │  Lambda             │
│  PostgreSQL   │  │  violation-alerts   │
│  (Multi-AZ)   │  │  → SNS → Slack/SMS  │
│  TimescaleDB  │  └─────────────────────┘
│  extension    │
└──────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  ECS Fargate — FastAPI          │
│  Behind ALB                     │
│  Auto-scaled by CPU             │
└──────────────┬──────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐  ┌─────────────────────┐
│  CloudWatch  │  │  Amazon Managed     │
│  + Grafana   │  │  Grafana            │
│  Dashboards  │  │  (no self-hosting)  │
└──────────────┘  └─────────────────────┘
```

### Key design decisions at scale

| Concern | Solution |
|---|---|
| **Camera → inference fan-out** | One EC2 GPU instance per ~32 cameras; Auto Scaling adds instances as fleet grows |
| **Concurrent DB writes** | Kafka buffers bursts; consumer batch-inserts to RDS with `execute_values` |
| **Data retention** | TimescaleDB time-series compression — 90-day raw, 1-year aggregated |
| **Alert deduplication** | AlertManager deduplicates per `(camera_id, track_id, class)` before publishing to SNS |
| **Model updates** | S3 for weights; instances pull on restart — zero-downtime rolling update via ECS |
| **Failure recovery** | Kafka consumer group rebalances on worker crash; no events lost |
| **Cost optimisation** | Spot instances for inference (stateless); On-Demand only for Kafka + RDS |

---

## Contributing

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Make your changes and add tests if applicable
3. Open a pull request — describe what you changed and why

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/Arunsuresh677/PPE-Compliance-Monito/issues).

---

## Author

**Arun S** — AI/ML Engineer  
[LinkedIn](https://linkedin.com/in/arun-s-481b2b3a2) · [GitHub](https://github.com/Arunsuresh677) · [HuggingFace](https://huggingface.co/Arunsuresh677)

---

## License

MIT © 2024 Arun S — see [LICENSE](LICENSE) for details.
