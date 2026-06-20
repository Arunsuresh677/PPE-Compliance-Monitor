import logging
import os
import tempfile
import threading
import time
import urllib.request
from datetime import datetime

import av
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from src.database.repository import ViolationRepository
from src.tracking.violation_tracker import ViolationTracker, VIOLATION_CLASSES, COMPLIANT_CLASSES
from src.detection.model import load_model as _load_model_impl
from src.detection.draw import CLASS_COLORS, draw_detections
from src.camera.manager import CameraManager
from src.config.settings import settings

log = logging.getLogger(__name__)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPE Compliance Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

* { box-sizing: border-box; }
body, .stApp { background-color: #0d1117 !important; color: #e6edf3; font-family: 'Rajdhani', sans-serif; }
h1, h2, h3, h4 { font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: 0.5px; }

section[data-testid="stSidebar"] { background: #010409 !important; border-right: 1px solid #21262d; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

div[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 14px 18px;
}
div[data-testid="stMetric"] label { color: #8b949e !important; font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
div[data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.8rem; font-weight: 700; }

.alert-ok {
    background: linear-gradient(90deg, #0d4429 0%, #0a3320 100%);
    border-left: 4px solid #3fb950; border-radius: 8px;
    padding: 14px 18px; color: #3fb950;
    font-weight: 700; font-size: 1.05rem; margin: 10px 0;
    font-family: 'Share Tech Mono', monospace;
}
.alert-viol {
    background: linear-gradient(90deg, #3d0000 0%, #2d0000 100%);
    border-left: 4px solid #f85149; border-radius: 8px;
    padding: 14px 18px; color: #ff7b72;
    font-weight: 700; font-size: 1.05rem; margin: 10px 0;
    font-family: 'Share Tech Mono', monospace;
}
.class-tag {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.78rem; font-family: 'Share Tech Mono', monospace;
    margin: 2px; font-weight: 600; letter-spacing: 0.5px;
}
.tag-ok   { background: #0d4429; color: #3fb950; border: 1px solid #238636; }
.tag-viol { background: #3d0000; color: #ff7b72; border: 1px solid #da3633; }
.tag-info { background: #0c2d6b; color: #79c0ff; border: 1px solid #1f6feb; }

.fps-badge {
    display: inline-block; background: #161b22; border: 1px solid #30363d;
    border-radius: 6px; padding: 4px 12px;
    font-family: 'Share Tech Mono', monospace; color: #58a6ff; font-size: 0.85rem;
}
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: #fff; border: none; border-radius: 8px;
    font-family: 'Rajdhani', sans-serif; font-weight: 700;
    font-size: 1rem; padding: 0.5rem 1.5rem;
}
div[data-testid="stFileUploader"] {
    background: #161b22; border: 2px dashed #30363d; border-radius: 10px; padding: 8px;
}
.detection-log {
    background: #010409; border: 1px solid #21262d; border-radius: 8px;
    padding: 10px 14px; max-height: 220px; overflow-y: auto;
    font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #8b949e;
}
.log-entry  { padding: 2px 0; border-bottom: 1px solid #21262d; }
.log-viol   { color: #ff7b72; }
.log-ok     { color: #3fb950; }
.log-event  { color: #e3b341; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_URL  = "https://huggingface.co/Arunsuresh677/ppe-compliance-monitor/resolve/main/best.pt"
MODEL_PATH = "best.pt"
DB_PATH    = "ppe_violations.db"

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})


# ─── Shared resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model() -> YOLO:
    """
    Download (atomically) and load YOLO weights.
    A failed/interrupted download never leaves a corrupt best.pt on disk.
    """
    if not os.path.exists(MODEL_PATH):
        tmp = MODEL_PATH + ".tmp"
        try:
            log.info("Downloading model weights from HuggingFace…")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            os.replace(tmp, MODEL_PATH)
        except Exception as exc:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError(f"Model download failed: {exc}") from exc

    try:
        return YOLO(MODEL_PATH)
    except Exception as exc:
        log.error("Corrupt weights — deleting %s: %s", MODEL_PATH, exc)
        os.remove(MODEL_PATH)
        raise RuntimeError(
            "Model file was corrupt and has been deleted. Reload the page to re-download."
        ) from exc


@st.cache_resource(show_spinner=False)
def get_db() -> ViolationRepository:
    """Initialise and return the shared ViolationRepository (once per process)."""
    repo = ViolationRepository()
    repo.init()
    return repo


# ─── WebRTC Processor ─────────────────────────────────────────────────────────
class PPEProcessor(VideoProcessorBase):
    """
    Per-connection YOLOv8 + ByteTrack processor for streamlit-webrtc.

    Each instance owns its own YOLO model (not the cached singleton) so that
    model.track(persist=True) maintains independent tracker state per session.
    All attributes shared with the Streamlit main thread are accessed under lock.
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model weights not found — reload the page to trigger download."
            )
        # Own YOLO instance — tracker state must NOT be shared between sessions
        self._model   = YOLO(MODEL_PATH)
        self._tracker = ViolationTracker()
        self._session = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.lock     = threading.Lock()
        self._conf    = 0.5
        self._fps     = 0.0
        self._last_ts = time.monotonic()
        self._stats   = {
            "total": 0, "violations": 0, "compliant": 0,
            "active_events": 0, "distinct_violators": 0,
        }

    @property
    def conf(self) -> float:
        with self.lock:
            return self._conf

    @conf.setter
    def conf(self, value: float) -> None:
        with self.lock:
            self._conf = float(value)

    @property
    def fps(self) -> float:
        with self.lock:
            return self._fps

    @property
    def stats(self) -> dict:
        with self.lock:
            return dict(self._stats)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        now = time.monotonic()
        with self.lock:
            elapsed       = now - self._last_ts
            self._fps     = 1.0 / elapsed if elapsed > 0 else 0.0
            self._last_ts = now
            conf          = self._conf

        results = self._model.track(
            img,
            conf=conf,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        img, total, viols, comp, raw = draw_detections(img, results, conf)

        # Update tracker; persist closed events to SQLite
        tracked = [(tid, cls) for tid, cls in raw if tid is not None]
        active  = self._tracker.update(tracked)
        closed = self._tracker.flush_closed()
        if closed:
            get_db().save_violations(closed, self._session)

        distinct = len({ev.track_id for ev in active})

        with self.lock:
            self._stats = {
                "total"             : total,
                "violations"        : len(viols),
                "compliant"         : len(comp),
                "active_events"     : len(active),
                "distinct_violators": distinct,
            }

        cv2.putText(img, f"FPS: {self._fps:.1f}",
                    (img.shape[1] - 110, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (88, 166, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦺 PPE Monitor")
    st.caption("YOLOv8 + ByteTrack · 10 Classes · Real-time")
    st.markdown("---")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 1.00, 0.50, 0.05)
    st.markdown("---")
    mode = st.radio("Input Mode", ["📷 Image", "🎞️ Video", "📹 Live Webcam", "📡 Multi-Camera"])
    st.markdown("---")
    st.markdown("**Detection Classes**")
    for cls in CLASS_COLORS:
        if cls in VIOLATION_CLASSES:
            st.markdown(f'<span class="class-tag tag-viol">❌ {cls}</span>', unsafe_allow_html=True)
        elif cls in COMPLIANT_CLASSES:
            st.markdown(f'<span class="class-tag tag-ok">✅ {cls}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="class-tag tag-info">🔵 {cls}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Built by Arun S**  \nAI/ML Engineer  \n[LinkedIn](https://linkedin.com/in/arun-suresh-481b2b3a2) · [GitHub](https://github.com/Arunsuresh677)")

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🦺 PPE Compliance Monitor")
st.markdown("Real-time safety detection · YOLOv8 + ByteTrack · 10 Classes · Construction & Industrial")
st.markdown("---")

# Initialise shared resources
repo = get_db()
try:
    with st.spinner("Loading model…"):
        model = load_model()
except RuntimeError as e:
    st.error(f"⚠️ Model load failed: {e}")
    st.stop()


# ─── IMAGE MODE ───────────────────────────────────────────────────────────────
if mode == "📷 Image":
    st.info("Image mode uses single-frame detection (no tracking — ByteTrack needs video frames to assign IDs).")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Could not decode the uploaded image. Please try a different file.")
        else:
            results = model(frame, conf=conf_thresh, verbose=False)
            out_frame, total, viols, comp, _ = draw_detections(
                frame.copy(), results, conf_thresh
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.markdown("**Detections**")
                st.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("#### Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Detections", total)
            c2.metric("✅ Compliant",     len(comp))
            c3.metric("❌ Violations",    len(viols))

            if viols:
                st.markdown(
                    f'<div class="alert-viol">⚠ VIOLATION DETECTED &nbsp;·&nbsp; {", ".join(set(viols))}</div>',
                    unsafe_allow_html=True,
                )
            elif comp:
                st.markdown(
                    '<div class="alert-ok">✅ ALL PPE COMPLIANT — No violations detected</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No PPE classes detected. Try lowering the confidence threshold.")


# ─── VIDEO MODE ───────────────────────────────────────────────────────────────
elif mode == "🎞️ Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded:
        session = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        tracker = ViolationTracker()

        tfile    = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_path = tfile.name
        try:
            tfile.write(uploaded.read())
            tfile.close()   # close before VideoCapture (required on Windows)

            cap = cv2.VideoCapture(tmp_path)
            try:
                if not cap.isOpened():
                    st.error("Could not open the video. Please try a different file.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25
                    skip         = max(1, int(fps_src // 10))

                    stframe  = st.empty()
                    progress = st.progress(0)

                    c1, c2, c3, c4 = st.columns(4)
                    m_total    = c1.empty()
                    m_comp     = c2.empty()
                    m_viol     = c3.empty()
                    m_events   = c4.empty()

                    log_box  = st.empty()
                    stop_btn = st.button("⏹ Stop")

                    log_entries: list[str] = []
                    _LOG_CAP = 500
                    frame_idx = 0
                    closed_events: list[dict] = []

                    while cap.isOpened() and not stop_btn:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_idx += 1
                        if frame_idx % skip != 0:
                            continue

                        results = model.track(
                            frame,
                            conf=conf_thresh,
                            tracker="bytetrack.yaml",
                            persist=True,
                            verbose=False,
                        )
                        out_frame, total, viols, comp, raw = draw_detections(
                            frame.copy(), results, conf_thresh
                        )

                        # Tracker update
                        tracked = [(tid, cls) for tid, cls in raw if tid is not None]
                        active  = tracker.update(tracked)
                        for ev in tracker.flush_closed():
                            repo.save_violation(ev, session)
                            closed_events.append(ev.to_dict())
                            log_entries.append(
                                f'<div class="log-entry log-event">'
                                f'[{frame_idx:05d}] EVENT CLOSED  '
                                f'Worker #{ev.track_id}  {ev.violation_class}  '
                                f'{ev.duration_secs:.1f}s</div>'
                            )

                        stframe.image(
                            cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                        )
                        progress.progress(min(frame_idx / max(total_frames, 1), 1.0))
                        m_total.metric("Detections",    total)
                        m_comp.metric("✅ Compliant",   len(comp))
                        m_viol.metric("❌ Violations",  len(viols))
                        m_events.metric("Events logged", len(closed_events))

                        if viols:
                            entry = f'<div class="log-entry log-viol">[{frame_idx:05d}] ⚠ {", ".join(set(viols))}</div>'
                        elif comp:
                            entry = f'<div class="log-entry log-ok">[{frame_idx:05d}] ✓ Compliant</div>'
                        else:
                            entry = f'<div class="log-entry">[{frame_idx:05d}] — No detection</div>'

                        if len(log_entries) >= _LOG_CAP:
                            log_entries = log_entries[-(_LOG_CAP // 2):]
                        log_entries.append(entry)
                        log_box.markdown(
                            f'<div class="detection-log">{"".join(log_entries[-25:])}</div>',
                            unsafe_allow_html=True,
                        )

                    # Flush remaining open events at end of video
                    for ev in tracker.close_all():
                        repo.save_violation(ev, session)
                        closed_events.append(ev.to_dict())

                    st.success(f"✅ Done — {frame_idx} frames · {len(closed_events)} violation events logged")

                    # ── Session analytics ──────────────────────────────────
                    if closed_events:
                        st.markdown("#### Violation Event Log")
                        summary = repo.get_session_summary(session)
                        sa, sb, sc = st.columns(3)
                        sa.metric("Total Events",       summary.get("total_events", 0))
                        sb.metric("Distinct Violators", summary.get("distinct_violators", 0))
                        sc.metric("Violation Types",    len(summary.get("by_class", {})))

                        rows = repo.get_violations(session=session)
                        st.dataframe(
                            rows,
                            column_order=["track_id", "violation_class", "start_time",
                                          "end_time", "duration_secs", "frame_count"],
                            use_container_width=True,
                        )

            finally:
                cap.release()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ─── WEBCAM MODE ──────────────────────────────────────────────────────────────
elif mode == "📹 Live Webcam":
    st.markdown("### 📹 Live Webcam — Real-time PPE Detection + ByteTrack")
    st.info(
        "Click **START** → allow camera access → live PPE detection with worker tracking. "
        "Each worker gets a persistent ID (#N) and violations are logged by duration."
    )

    ctx = webrtc_streamer(
        key="ppe-webcam",
        video_processor_factory=PPEProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    )

    if ctx.video_processor:
        ctx.video_processor.conf = conf_thresh

        col1, col2 = st.columns([2, 1])

        with col1:
            stat_cols  = st.columns(4)
            stat_total = stat_cols[0].empty()
            stat_comp  = stat_cols[1].empty()
            stat_viol  = stat_cols[2].empty()
            stat_ev    = stat_cols[3].empty()
            fps_disp   = st.empty()

        with col2:
            st.markdown("**Live Violation Events (DB)**")
            events_box = st.empty()

        while ctx.state.playing:
            s   = ctx.video_processor.stats
            fps = ctx.video_processor.fps

            stat_total.metric("Detections",         s["total"])
            stat_comp.metric("✅ Compliant",        s["compliant"])
            stat_viol.metric("❌ Violations",       s["violations"])
            stat_ev.metric("Active events",         s["active_events"])
            fps_disp.markdown(
                f'<span class="fps-badge">⚡ {fps:.1f} FPS  '
                f'· {s["distinct_violators"]} active violator(s)</span>',
                unsafe_allow_html=True,
            )

            # Pull latest events from DB for live display
            recent = repo.get_violations(session=ctx.video_processor._session if ctx.video_processor else "", limit=10)
            if recent:
                rows_html = "".join(
                    f'<div class="log-entry log-event">'
                    f'#{r["track_id"]}  {r["violation_class"]}  '
                    f'{r["duration_secs"]:.1f}s</div>'
                    for r in recent
                )
                events_box.markdown(
                    f'<div class="detection-log">{rows_html}</div>',
                    unsafe_allow_html=True,
                )

            time.sleep(0.3)

        # ── Post-session analytics ─────────────────────────────────────────
        if ctx.video_processor:
            session = ctx.video_processor._session
            summary = repo.get_session_summary(session)
            if summary.get("total_events", 0) > 0:
                st.markdown("#### Session Summary")
                sa, sb, sc = st.columns(3)
                sa.metric("Total Events",       summary["total_events"])
                sb.metric("Distinct Violators", summary["distinct_violators"])
                sc.metric("Violation Types",    len(summary.get("by_class", {})))

                rows = repo.get_violations(session=session)
                st.dataframe(
                    rows,
                    column_order=["track_id", "violation_class", "start_time",
                                  "end_time", "duration_secs", "frame_count"],
                    use_container_width=True,
                )


# ─── MULTI-CAMERA MODE ────────────────────────────────────────────────────────
elif mode == "📡 Multi-Camera":
    st.markdown("### 📡 Multi-Camera Fleet Monitor")
    st.info(
        "Monitor multiple RTSP cameras simultaneously. "
        "Each camera runs its own YOLOv8 + ByteTrack instance. "
        "All violations are saved to PostgreSQL with per-camera tagging."
    )

    # ── Session init ──────────────────────────────────────────────────────
    if "mc_session" not in st.session_state:
        st.session_state.mc_session = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if "mc_manager" not in st.session_state:
        st.session_state.mc_manager = None
    if "mc_cameras" not in st.session_state:
        st.session_state.mc_cameras = []   # list of (camera_id, source)

    session = st.session_state.mc_session

    # ── Add camera form ───────────────────────────────────────────────────
    with st.expander("➕ Add Camera", expanded=len(st.session_state.mc_cameras) == 0):
        col_a, col_b, col_c = st.columns([1, 3, 1])
        with col_a:
            cam_id = st.text_input("Camera ID", placeholder="cam-01")
        with col_b:
            cam_src = st.text_input(
                "Source",
                placeholder="rtsp://192.168.1.10:554/stream  or  0  (webcam index)",
            )
        with col_c:
            st.markdown("<br>", unsafe_allow_html=True)
            add_btn = st.button("Add")

        if add_btn and cam_id and cam_src:
            source = int(cam_src) if cam_src.isdigit() else cam_src
            existing_ids = [c[0] for c in st.session_state.mc_cameras]
            if cam_id in existing_ids:
                st.warning(f"Camera '{cam_id}' already added.")
            else:
                st.session_state.mc_cameras.append((cam_id, source))
                st.success(f"Camera '{cam_id}' added.")
                st.rerun()

    # ── Camera list + controls ────────────────────────────────────────────
    if st.session_state.mc_cameras:
        st.markdown(f"**{len(st.session_state.mc_cameras)} camera(s) configured**")

        remove_cols = st.columns(len(st.session_state.mc_cameras))
        to_remove = None
        for i, (cid, src) in enumerate(st.session_state.mc_cameras):
            with remove_cols[i]:
                st.caption(f"`{cid}` — `{src}`")
                if st.button(f"✕ Remove {cid}", key=f"rm_{cid}"):
                    to_remove = cid
        if to_remove:
            st.session_state.mc_cameras = [c for c in st.session_state.mc_cameras if c[0] != to_remove]
            if st.session_state.mc_manager:
                st.session_state.mc_manager.remove_camera(to_remove)
            st.rerun()

        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            start_btn = st.button("▶ Start All Cameras", type="primary",
                                  disabled=st.session_state.mc_manager is not None)
        with ctrl_col2:
            stop_btn = st.button("⏹ Stop All Cameras",
                                 disabled=st.session_state.mc_manager is None)

        if start_btn:
            manager = CameraManager(session=session, repo=repo, model_path=MODEL_PATH)
            for cid, src in st.session_state.mc_cameras:
                manager.add_camera(cid, src, conf=conf_thresh)
            st.session_state.mc_manager = manager
            st.rerun()

        if stop_btn and st.session_state.mc_manager:
            st.session_state.mc_manager.stop_all()
            st.session_state.mc_manager = None
            st.rerun()

        # ── Live grid ─────────────────────────────────────────────────────
        if st.session_state.mc_manager:
            manager: CameraManager = st.session_state.mc_manager
            manager.set_conf(conf_thresh)

            st.markdown("#### Live Feeds")
            cam_ids = manager.camera_ids
            n_cols  = min(2, len(cam_ids))
            cols    = st.columns(n_cols)
            placeholders = {cid: cols[i % n_cols].empty() for i, cid in enumerate(cam_ids)}
            stat_boxes   = {cid: cols[i % n_cols].empty() for i, cid in enumerate(cam_ids)}

            st.markdown("---")
            st.markdown("#### Fleet Summary")
            fleet_cols    = st.columns(4)
            m_cameras     = fleet_cols[0].empty()
            m_total_viols = fleet_cols[1].empty()
            m_total_workers = fleet_cols[2].empty()
            m_total_fps   = fleet_cols[3].empty()

            refresh_count = 0
            while st.session_state.mc_manager is not None:
                frames = manager.get_frames()
                stats  = manager.get_all_stats()

                total_viols   = 0
                total_workers = 0
                total_fps     = 0.0

                for cid in cam_ids:
                    frame = frames.get(cid)
                    s     = stats.get(cid, {})

                    if frame is not None:
                        placeholders[cid].image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                        )
                    else:
                        placeholders[cid].info(f"⏳ {cid} — {s.get('status', 'connecting...')}")

                    stat_boxes[cid].markdown(
                        f"**{cid}** · {s.get('fps', 0):.1f} FPS · "
                        f"❌ {s.get('violations', 0)} violations · "
                        f"👷 {s.get('distinct_violators', 0)} workers · "
                        f"_{s.get('status', '...')}_"
                    )

                    total_viols   += s.get("violations", 0)
                    total_workers += s.get("distinct_violators", 0)
                    total_fps     += s.get("fps", 0.0)

                m_cameras.metric("Active Cameras",    manager.count)
                m_total_viols.metric("❌ Violations",  total_viols)
                m_total_workers.metric("👷 Workers",   total_workers)
                m_total_fps.metric("Avg FPS/Camera",  f"{total_fps / max(manager.count, 1):.1f}")

                refresh_count += 1
                time.sleep(0.5)

    else:
        st.info("Add at least one camera above to get started.")
