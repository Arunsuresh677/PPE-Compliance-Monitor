<div align="center">

<img src="https://img.shields.io/badge/STATUS-LIVE%20IN%20PRODUCTION-brightgreen?style=for-the-badge&logo=streamlit" />

# PPE Compliance Monitor

### Real-Time Worker Safety Detection · YOLOv8 + ByteTrack · 10 Classes · Construction & Industrial

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
| Inference speed | ~30 FPS on CPU |
| Training epochs | 50 |
| Framework | Ultralytics + PyTorch |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Sources                            │
│   📷 Image file   🎞 Video file   📹 Webcam   📡 RTSP stream   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      YOLOv8s Inference                          │
│   • 640×640 input  • conf threshold  • 10-class detection      │
│   • Weights hosted on HuggingFace, auto-downloaded first run   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ByteTrack (per-frame tracking)                │
│   • Assigns persistent ID to each detected worker              │
│   • Survives brief occlusion (3s stale timeout)                │
│   • Enables: duration, distinct violators, repeat offenders    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ViolationTracker (tracker.py)                      │
│   • State machine: one ViolationEvent per (worker, class)      │
│   • Opens on first violation frame, closes on compliance/exit  │
│   • Emits closed events to SQLite                              │
└──────────┬────────────────────────────────────────┬────────────┘
           │                                        │
           ▼                                        ▼
┌─────────────────────┐                 ┌───────────────────────┐
│  Streamlit Web App  │                 │   CLI (detect.py)     │
│  • Image upload     │                 │  • RTSP livestream    │
│  • Video + tracking │                 │  • Video + tracking   │
│  • Live webcam      │                 │  • Session summary    │
│    (WebRTC)         │                 │  • Save annotated MP4 │
│  • Violation log    │                 └───────────────────────┘
│  • Session analytics│
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  SQLite (database.py)│
│  violation_events   │
│  • track_id         │
│  • violation_class  │
│  • start / end time │
│  • duration_secs    │
│  • frame_count      │
└─────────────────────┘
```

---

## Quick Start

### Option A — Live demo (no install)

**[→ Open in browser](https://ppe-compliance-monitor-7ssrkxc83lsifmxnfngote.streamlit.app/)** · Upload an image or video and see results instantly.

### Option B — Run locally

```bash
# 1. Clone
git clone https://github.com/Arunsuresh677/PPE-Compliance-Monito.git
cd PPE-Compliance-Monito

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch web app
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

### Violation event schema

```sql
CREATE TABLE violation_events (
    id              INTEGER PRIMARY KEY,
    session         TEXT,     -- ISO timestamp of detection run
    track_id        INTEGER,  -- ByteTrack worker ID
    violation_class TEXT,     -- e.g. "NO-Hardhat"
    start_time      TEXT,     -- when violation began
    end_time        TEXT,     -- when it ended
    duration_secs   REAL,     -- wall-clock seconds
    frame_count     INTEGER   -- YOLO frames active
);
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

The model was trained on a public PPE/helmet detection dataset using **Google Colab (T4 GPU)**.

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
PPE-Compliance-Monito/
├── app.py            # Streamlit web app (image / video / live webcam + tracking)
├── detect.py         # CLI inference (webcam, video, image, RTSP + ByteTrack)
├── tracker.py        # ViolationTracker state machine — per-ID event lifecycle
├── database.py       # SQLite violation log (WAL mode, thread-safe)
├── train.py          # Training script (YOLOv8, YOLO data format)
├── requirements.txt  # Python dependencies
├── packages.txt      # System packages (for Streamlit Cloud)
└── README.md
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
