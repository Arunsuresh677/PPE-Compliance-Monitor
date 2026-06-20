"""
PPE Compliance Monitor - Inference / Detection Script
Author: Arun S | Nithil Innovations

Detects PPE compliance in real time using YOLOv8 + ByteTrack.
Each detected person receives a persistent track ID across frames so the
system logs one ViolationEvent per incident (not per frame), enabling:
  - distinct violator counts
  - violation duration per worker
  - per-session analytics stored in SQLite

Classes: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
         Safety Cone, Safety Vest, machinery, vehicle
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

import database
from tracker import ViolationTracker, VIOLATION_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

CLASS_COLORS = {
    "hardhat"       : (0, 200, 0),
    "mask"          : (0, 180, 0),
    "safety vest"   : (0, 160, 0),
    "no-hardhat"    : (0, 0, 220),
    "no-mask"       : (0, 0, 200),
    "no-safety vest": (0, 0, 180),
    "person"        : (200, 200, 0),
    "safety cone"   : (0, 165, 255),
    "machinery"     : (180, 100, 0),
    "vehicle"       : (150, 80, 0),
}

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "rtsp://", "rtsps://"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPE Compliance Monitor — YOLOv8 + ByteTrack inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source",   type=str,   default="0",        help="0=webcam, video.mp4, image.jpg, rtsp://...")
    parser.add_argument("--model",    type=str,   default="best.pt",  help="Path to YOLO weights")
    parser.add_argument("--conf",     type=float, default=0.5,        help="Confidence threshold")
    parser.add_argument("--imgsz",    type=int,   default=640,        help="Inference image size (px)")
    parser.add_argument("--device",   type=str,   default="",         help="Device: cpu or 0 (GPU)")
    parser.add_argument("--save",     action="store_true",            help="Save annotated output")
    parser.add_argument("--output",   type=str,   default="output/",  help="Output directory")
    parser.add_argument("--no-show",  action="store_true",            help="Suppress display window")
    parser.add_argument("--no-track", action="store_true",            help="Disable ByteTrack (use plain detection)")
    parser.add_argument("--db",       type=str,   default="ppe_violations.db", help="SQLite DB path")
    return parser.parse_args()


def draw_detections(frame, results, conf_threshold: float) -> tuple:
    """
    Annotate frame with bounding boxes, track IDs, labels, and status banner.

    Returns (annotated_frame, detections) where detections is a list of
    (track_id_or_None, class_name) for every box drawn.
    """
    detections: list[tuple] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        track_ids = (
            boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        )

        for box, track_id in zip(boxes, track_ids):
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id   = int(box.cls[0])
            cls_name = result.names[cls_id]
            cls_key  = cls_name.lower()
            color    = CLASS_COLORS.get(cls_key, (200, 200, 200))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label: show track ID when available
            id_prefix = f"#{track_id} " if track_id is not None else ""
            label = f"{id_prefix}{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            detections.append((track_id, cls_name))

    # Status banner
    violations = [cls for _, cls in detections if cls in VIOLATION_CLASSES]
    if violations:
        unique = list(set(violations))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 180), -1)
        cv2.putText(frame, f"VIOLATION: {', '.join(unique)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 140, 0), -1)
        cv2.putText(frame, "PPE COMPLIANT", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame, detections


def run(args: argparse.Namespace) -> None:
    log.info("=" * 60)
    log.info("PPE Compliance Monitor — YOLOv8%s Inference",
             " + ByteTrack" if not args.no_track else "")
    log.info("=" * 60)
    log.info("Model   : %s", args.model)
    log.info("Source  : %s", args.source)
    log.info("Conf    : %.2f | Device : %s", args.conf, args.device or "auto")
    log.info("Tracking: %s", "disabled" if args.no_track else "ByteTrack")

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    # Session ID used to group all events from this run in the DB
    session = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    database.init_db(args.db)

    model  = YOLO(args.model)
    use_tracking = not args.no_track
    tracker = ViolationTracker() if use_tracking else None

    source: str | int = int(args.source) if args.source.isdigit() else args.source
    is_local_image = (
        isinstance(source, str)
        and not _is_url(source)
        and Path(source).suffix.lower() in _IMAGE_SUFFIXES
    )

    if args.save:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── Image mode ────────────────────────────────────────────────────────
    if is_local_image:
        frame = cv2.imread(str(source))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {source}")

        results = model.predict(
            source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False
        )
        frame, detections = draw_detections(frame, results, args.conf)

        violations = [cls for _, cls in detections if cls in VIOLATION_CLASSES]
        if violations:
            log.warning("VIOLATION: %s", ", ".join(set(violations)))
        else:
            log.info("Result: PPE compliant — no violations")

        if not args.no_show:
            cv2.imshow("PPE Compliance Monitor", frame)
            cv2.waitKey(0)
        if args.save:
            out_path = Path(args.output) / Path(str(source)).name
            cv2.imwrite(str(out_path), frame)
            log.info("Saved: %s", out_path)
        cv2.destroyAllWindows()
        database.close_db()
        return

    # ── Video / Webcam / RTSP mode ────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # Minimise buffer so RTSP detections reflect current reality, not
    # frames buffered several seconds ago.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        out_path = Path(args.output) / "output.mp4"
        writer   = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h)
        )
        log.info("Saving to: %s", out_path)

    frame_count = 0
    fps_display = 0.0
    t_start     = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("Stream ended.")
                break

            # ── Inference (with or without ByteTrack) ─────────────────
            if use_tracking:
                results = model.track(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                )
            else:
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )

            frame, detections = draw_detections(frame, results, args.conf)

            # ── Update tracker, persist closed events ──────────────────
            if tracker is not None:
                # Only feed detections that have a track ID
                tracked = [(tid, cls) for tid, cls in detections if tid is not None]
                active  = tracker.update(tracked)
                for ev in tracker.flush_closed():
                    database.save_violation(ev, session)
                    log.info(
                        "VIOLATION CLOSED  #%d %s  %.1fs  (%d frames)",
                        ev.track_id, ev.violation_class,
                        ev.duration_secs, ev.frame_count,
                    )
                # Overlay active violation count
                if active:
                    cv2.putText(
                        frame,
                        f"Active violations: {len(active)}",
                        (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    )

            # ── FPS overlay ───────────────────────────────────────────
            frame_count += 1
            elapsed = time.monotonic() - t_start
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                t_start     = time.monotonic()

            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if writer:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow("PPE Compliance Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Stopped by user.")
                    break

    finally:
        # Close any open tracker events before exiting
        if tracker is not None:
            for ev in tracker.close_all():
                database.save_violation(ev, session)

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # ── Session summary ────────────────────────────────────────────
        if tracker is not None:
            s = database.get_session_summary(session)
            log.info("=" * 60)
            log.info("SESSION SUMMARY  [%s]", session)
            log.info("  Total violation events : %d", s.get("total_events", 0))
            log.info("  Distinct violators     : %d", s.get("distinct_violators", 0))
            for cls, stats in s.get("by_class", {}).items():
                log.info(
                    "  %-20s %d events  avg %.1fs  max %.1fs",
                    cls, stats["events"], stats["avg_secs"], stats["max_secs"],
                )
            log.info("=" * 60)

        database.close_db()
        log.info("Detection complete.")


if __name__ == "__main__":
    run(parse_args())
