import logging
import os
import tempfile
import threading
import time
import urllib.request
import urllib.error

import av
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
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
    padding: 10px 14px; max-height: 180px; overflow-y: auto;
    font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #8b949e;
}
.log-entry { padding: 2px 0; border-bottom: 1px solid #21262d; }
.log-viol  { color: #ff7b72; }
.log-ok    { color: #3fb950; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_URL  = "https://huggingface.co/Arunsuresh677/ppe-compliance-monitor/resolve/main/best.pt"
MODEL_PATH = "best.pt"

CLASS_COLORS = {
    "Hardhat":        (0, 220, 90),
    "Mask":           (0, 200, 255),
    "Safety Vest":    (0, 180, 80),
    "NO-Hardhat":     (220, 30,  30),
    "NO-Mask":        (200, 50,  200),
    "NO-Safety Vest": (220, 80,  0),
    "Person":         (100, 180, 255),
    "Safety Cone":    (255, 200, 0),
    "machinery":      (160, 160, 160),
    "vehicle":        (120, 120, 220),
}
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
COMPLIANT_CLASSES = {"Hardhat", "Mask", "Safety Vest"}

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# ─── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model() -> YOLO:
    """
    Download model weights from HuggingFace (first run only) and load YOLO.

    Uses an atomic write pattern: download to a temp file beside the target,
    then rename on success. A failed or interrupted download never leaves a
    corrupt best.pt on disk, so the next call always retries cleanly.
    """
    if not os.path.exists(MODEL_PATH):
        tmp_path = MODEL_PATH + ".tmp"
        try:
            log.info("Downloading model weights from HuggingFace…")
            urllib.request.urlretrieve(MODEL_URL, tmp_path)
            os.replace(tmp_path, MODEL_PATH)   # atomic on POSIX; near-atomic on Windows
            log.info("Model downloaded to %s", MODEL_PATH)
        except Exception as exc:
            # Clean up any partial download so the next run retries.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(
                f"Failed to download model weights from {MODEL_URL}: {exc}"
            ) from exc

    try:
        return YOLO(MODEL_PATH)
    except Exception as exc:
        # Corrupt weights file — remove so the next run re-downloads.
        log.error("YOLO failed to load %s — deleting corrupt file: %s", MODEL_PATH, exc)
        os.remove(MODEL_PATH)
        raise RuntimeError(
            f"Model file {MODEL_PATH} was corrupt and has been deleted. "
            "Reload the page to re-download."
        ) from exc


# ─── Draw detections ──────────────────────────────────────────────────────────
def draw_detections(
    frame: np.ndarray,
    results,
    conf_thresh: float = 0.5,
) -> tuple[np.ndarray, int, list[str], list[str]]:
    """
    Overlay bounding boxes, labels, and a status banner on *frame* (in-place).

    Returns (annotated_frame, total_detections, violation_names, compliant_names).
    The model is already called with the same conf_thresh, so boxes below the
    threshold will not appear in results — the per-box check here is a
    belt-and-suspenders guard for callers that pass a looser model threshold.
    """
    violations: list[str] = []
    compliant:  list[str] = []
    total = 0

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            cls_id = int(box.cls[0])
            name   = r.names[cls_id]
            color  = CLASS_COLORS.get(name, (200, 200, 200))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            total += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)

            if name in VIOLATION_CLASSES:
                violations.append(name)
            elif name in COMPLIANT_CLASSES:
                compliant.append(name)

    h, w = frame.shape[:2]
    if violations:
        cv2.rectangle(frame, (0, 0), (w, 40), (180, 0, 0), -1)
        cv2.putText(frame, f"  VIOLATION: {', '.join(set(violations))}", (6, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    elif compliant:
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 140, 40), -1)
        cv2.putText(frame, "  ALL PPE COMPLIANT", (6, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

    return frame, total, violations, compliant


# ─── WebRTC Processor ─────────────────────────────────────────────────────────
class PPEProcessor(VideoProcessorBase):
    """
    Per-connection YOLOv8 inference processor for streamlit-webrtc.

    All attributes shared between the recv() thread and the Streamlit main
    thread are read and written under self.lock to prevent data races.
    """

    def __init__(self):
        self._model   = load_model()   # uses cached singleton — no extra load
        self.lock     = threading.Lock()
        self._conf    = 0.5
        self._fps     = 0.0
        self._last_ts = time.monotonic()
        self._stats   = {"total": 0, "violations": 0, "compliant": 0}

    # ── Thread-safe conf property ──────────────────────────────────────────
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

    # ── WebRTC callback (runs in its own thread) ───────────────────────────
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        now = time.monotonic()
        with self.lock:
            elapsed       = now - self._last_ts
            self._fps     = 1.0 / elapsed if elapsed > 0 else 0.0
            self._last_ts = now
            conf          = self._conf

        results = self._model(img, conf=conf, verbose=False)
        img, total, viols, comp = draw_detections(img, results, conf)

        with self.lock:
            self._stats = {
                "total"      : total,
                "violations" : len(viols),
                "compliant"  : len(comp),
            }

        cv2.putText(img, f"FPS: {self._fps:.1f}",
                    (img.shape[1] - 110, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (88, 166, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦺 PPE Monitor")
    st.caption("YOLOv8 · 10 Classes · Real-time")
    st.markdown("---")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 1.00, 0.50, 0.05)
    st.markdown("---")
    mode = st.radio("Input Mode", ["📷 Image", "🎞️ Video", "📹 Live Webcam"])
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
    st.markdown("**Built by Arun S**  \nAI/ML Engineer  \n[LinkedIn](https://linkedin.com/in/arun-s-481b2b3a2) · [GitHub](https://github.com/Arunsuresh677)")

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🦺 PPE Compliance Monitor")
st.markdown("Real-time safety detection · YOLOv8 · 10 Classes · Construction & Industrial")
st.markdown("---")

try:
    with st.spinner("Loading model…"):
        model = load_model()
except RuntimeError as e:
    st.error(f"⚠️ Model load failed: {e}")
    st.stop()

# ─── IMAGE MODE ───────────────────────────────────────────────────────────────
if mode == "📷 Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Could not decode the uploaded image. Please try a different file.")
        else:
            results    = model(frame, conf=conf_thresh, verbose=False)
            out_frame, total, viols, comp = draw_detections(frame.copy(), results, conf_thresh)

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
        # Write to a named temp file so OpenCV can open it by path.
        # delete=False because we close the handle before VideoCapture opens it
        # (required on Windows — a file open by one handle cannot be opened by
        # another). We clean up explicitly in the finally block.
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_path = tfile.name
        try:
            tfile.write(uploaded.read())
            tfile.close()   # must close before VideoCapture on Windows

            cap = cv2.VideoCapture(tmp_path)
            try:
                if not cap.isOpened():
                    st.error("Could not open the uploaded video. Please try a different file.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25
                    skip         = max(1, int(fps_src // 10))  # process ~10 fps

                    stframe  = st.empty()
                    progress = st.progress(0)
                    c1, c2, c3 = st.columns(3)
                    m_total = c1.empty(); m_comp = c2.empty(); m_viol = c3.empty()
                    log_box  = st.empty()
                    stop_btn = st.button("⏹ Stop")

                    # Cap log_entries at 500 to avoid unbounded memory growth on
                    # long videos (only the last 20 are rendered at a time anyway).
                    log_entries: list[str] = []
                    _LOG_CAP = 500
                    frame_idx = 0

                    while cap.isOpened() and not stop_btn:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_idx += 1
                        if frame_idx % skip != 0:
                            continue

                        results = model(frame, conf=conf_thresh, verbose=False)
                        out_frame, total, viols, comp = draw_detections(
                            frame.copy(), results, conf_thresh
                        )

                        stframe.image(
                            cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                        )
                        progress.progress(min(frame_idx / max(total_frames, 1), 1.0))
                        m_total.metric("Detections",   total)
                        m_comp.metric("✅ Compliant",  len(comp))
                        m_viol.metric("❌ Violations", len(viols))

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
                            f'<div class="detection-log">{"".join(log_entries[-20:])}</div>',
                            unsafe_allow_html=True,
                        )

                    st.success(f"✅ Done — processed {frame_idx} frames")
            finally:
                cap.release()
        finally:
            # Always delete the temp file, even if an exception occurs.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

# ─── WEBCAM MODE ──────────────────────────────────────────────────────────────
elif mode == "📹 Live Webcam":
    st.markdown("### 📹 Live Webcam — Real-time PPE Detection")
    st.info("Click **START** → allow camera access → live PPE detection runs on every frame.")

    ctx = webrtc_streamer(
        key="ppe-webcam",
        video_processor_factory=PPEProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    )

    if ctx.video_processor:
        # conf setter acquires the lock internally — safe to call from main thread
        ctx.video_processor.conf = conf_thresh

        stat_cols  = st.columns(3)
        stat_total = stat_cols[0].empty()
        stat_comp  = stat_cols[1].empty()
        stat_viol  = stat_cols[2].empty()
        fps_disp   = st.empty()

        while ctx.state.playing:
            s   = ctx.video_processor.stats   # returns a copy under lock
            fps = ctx.video_processor.fps
            stat_total.metric("Detections",   s["total"])
            stat_comp.metric("✅ Compliant",  s["compliant"])
            stat_viol.metric("❌ Violations", s["violations"])
            fps_disp.markdown(
                f'<span class="fps-badge">⚡ {fps:.1f} FPS</span>',
                unsafe_allow_html=True,
            )
            time.sleep(0.3)
