import os
import time
import urllib.request
import tempfile
import threading
import av
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return YOLO(MODEL_PATH)

# ─── Draw detections ──────────────────────────────────────────────────────────
def draw_detections(frame, results, conf_thresh=0.5):
    violations, compliant = [], []
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
    def __init__(self):
        self.model    = load_model()
        self.conf     = 0.5
        self.lock     = threading.Lock()
        self.fps      = 0.0
        self._last_ts = time.time()
        self.stats    = {"total": 0, "violations": 0, "compliant": 0}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        now = time.time()
        with self.lock:
            self.fps      = 1.0 / max(now - self._last_ts, 1e-6)
            self._last_ts = now

        results = self.model(img, conf=self.conf, verbose=False)
        img, total, viols, comp = draw_detections(img, results, self.conf)

        with self.lock:
            self.stats = {"total": total, "violations": len(viols), "compliant": len(comp)}

        cv2.putText(img, f"FPS: {self.fps:.1f}", (img.shape[1] - 110, img.shape[0] - 10),
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

with st.spinner("Loading model…"):
    model = load_model()

# ─── IMAGE MODE ───────────────────────────────────────────────────────────────
if mode == "📷 Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
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
            st.markdown(f'<div class="alert-viol">⚠ VIOLATION DETECTED &nbsp;·&nbsp; {", ".join(set(viols))}</div>',
                        unsafe_allow_html=True)
        elif comp:
            st.markdown('<div class="alert-ok">✅ ALL PPE COMPLIANT — No violations detected</div>',
                        unsafe_allow_html=True)
        else:
            st.info("No PPE classes detected. Try lowering the confidence threshold.")

# ─── VIDEO MODE ───────────────────────────────────────────────────────────────
elif mode == "🎞️ Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()

        cap          = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25
        skip         = max(1, int(fps_src // 10))  # ~10 processed fps

        stframe  = st.empty()
        progress = st.progress(0)
        c1, c2, c3 = st.columns(3)
        m_total = c1.empty(); m_comp = c2.empty(); m_viol = c3.empty()
        log_box = st.empty()
        stop_btn = st.button("⏹ Stop")

        log_entries = []
        frame_idx   = 0

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % skip != 0:
                continue

            results = model(frame, conf=conf_thresh, verbose=False)
            out_frame, total, viols, comp = draw_detections(frame.copy(), results, conf_thresh)

            stframe.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
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
            log_entries.append(entry)
            log_box.markdown(
                f'<div class="detection-log">{"".join(log_entries[-20:])}</div>',
                unsafe_allow_html=True
            )

        cap.release()
        st.success(f"✅ Done — processed {frame_idx} frames")

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
        ctx.video_processor.conf = conf_thresh

        stat_cols  = st.columns(3)
        stat_total = stat_cols[0].empty()
        stat_comp  = stat_cols[1].empty()
        stat_viol  = stat_cols[2].empty()
        fps_disp   = st.empty()

        while ctx.state.playing:
            with ctx.video_processor.lock:
                s   = ctx.video_processor.stats
                fps = ctx.video_processor.fps
            stat_total.metric("Detections",   s["total"])
            stat_comp.metric("✅ Compliant",  s["compliant"])
            stat_viol.metric("❌ Violations", s["violations"])
            fps_disp.markdown(
                f'<span class="fps-badge">⚡ {fps:.1f} FPS</span>',
                unsafe_allow_html=True
            )
            time.sleep(0.3)