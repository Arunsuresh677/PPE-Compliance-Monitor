"""
PPE Compliance Monitor - Inference / Detection Script
Author: Arun S | Nithil Innovations
Description: Detect PPE compliance in real-time from webcam, video, image, or RTSP stream
Classes: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
         Safety Cone, Safety Vest, machinery, vehicle
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Class colors — Green for compliant, Red for violations, Yellow for context
CLASS_COLORS = {
    "hardhat":        (0, 200, 0),
    "mask":           (0, 180, 0),
    "safety vest":    (0, 160, 0),
    "no-hardhat":     (0, 0, 220),
    "no-mask":        (0, 0, 200),
    "no-safety vest": (0, 0, 180),
    "person":         (200, 200, 0),
    "safety cone":    (0, 165, 255),
    "machinery":      (180, 100, 0),
    "vehicle":        (150, 80, 0),
}

VIOLATION_CLASSES = {"no-hardhat", "no-mask", "no-safety vest"}

# File-path suffixes that OpenCV's imread can read directly.
# HTTP/HTTPS/RTSP URLs are excluded intentionally — imread doesn't support them
# and Path().suffix would incorrectly classify e.g. "http://host/cam.jpg" as
# an image source.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "rtsp://", "rtsps://"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPE Compliance Monitor - YOLOv8 Inference")
    parser.add_argument("--source",  type=str,   default="0",       help="Source: 0=webcam, video.mp4, image.jpg, rtsp://...")
    parser.add_argument("--model",   type=str,   default="best.pt", help="Path to trained model weights")
    parser.add_argument("--conf",    type=float, default=0.5,       help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--imgsz",   type=int,   default=640,       help="Inference image size")
    parser.add_argument("--device",  type=str,   default="",        help="Device: cpu or 0 (GPU)")
    parser.add_argument("--save",    action="store_true",           help="Save output video/images")
    parser.add_argument("--output",  type=str,   default="output/", help="Output directory")
    parser.add_argument("--no-show", action="store_true",           help="Do not display live window")
    return parser.parse_args()


def draw_detections(frame, results, conf_threshold: float):
    """Draw bounding boxes, labels, and an alert banner on *frame* (in-place).

    Returns the annotated frame and a list of detected violation class names.
    The model is already called with the same conf_threshold, so the per-box
    check is a belt-and-suspenders guard against callers using a looser threshold.
    """
    violations: list[str] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id   = int(box.cls[0])
            cls_name = result.names[cls_id]
            cls_key  = cls_name.lower()
            color    = CLASS_COLORS.get(cls_key, (200, 200, 200))

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if cls_key in VIOLATION_CLASSES:
                violations.append(cls_name)

    if violations:
        unique     = list(set(violations))
        alert_text = f"VIOLATION: {', '.join(unique)}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 180), -1)
        cv2.putText(frame, alert_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 140, 0), -1)
        cv2.putText(frame, "PPE COMPLIANT", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame, violations


def run(args: argparse.Namespace) -> None:
    log.info("=" * 55)
    log.info("PPE Compliance Monitor — YOLOv8 Inference")
    log.info("=" * 55)
    log.info("Model  : %s", args.model)
    log.info("Source : %s", args.source)
    log.info("Conf   : %.2f", args.conf)
    log.info("Press Q to quit")

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = YOLO(args.model)

    # Resolve numeric webcam indices; leave all other strings (paths, URLs) as-is.
    source: str | int = int(args.source) if args.source.isdigit() else args.source

    # An HTTP/HTTPS URL whose path ends in an image extension must NOT be treated
    # as a local file — cv2.imread cannot fetch URLs and returns None, which
    # causes a crash downstream. Only local file paths go through image mode.
    is_local_image = (
        isinstance(source, str)
        and not _is_url(source)
        and Path(source).suffix.lower() in _IMAGE_SUFFIXES
    )

    if args.save:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── Image mode ────────────────────────────────────────────────────────────
    if is_local_image:
        frame = cv2.imread(str(source))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {source}")

        results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        frame, violations = draw_detections(frame, results, args.conf)

        if violations:
            log.warning("VIOLATION detected: %s", ", ".join(set(violations)))
        else:
            log.info("Result: PPE compliant")

        if not args.no_show:
            cv2.imshow("PPE Compliance Monitor", frame)
            cv2.waitKey(0)

        if args.save:
            out_path = Path(args.output) / Path(str(source)).name
            cv2.imwrite(str(out_path), frame)
            log.info("Saved: %s", out_path)

        cv2.destroyAllWindows()
        return

    # ── Video / Webcam / RTSP mode ────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # Reduce buffering for RTSP/webcam so detections reflect current reality,
    # not frames that are several seconds old.
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
        log.info("Saving output to: %s", out_path)

    frame_count  = 0
    fps_display  = 0.0
    t_start      = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("Stream ended.")
                break

            results = model.predict(
                source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False
            )
            frame, violations = draw_detections(frame, results, args.conf)

            if violations:
                log.warning("VIOLATION: %s", ", ".join(set(violations)))

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
        # Always release resources — even if an exception occurs mid-loop.
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        log.info("Detection complete.")


if __name__ == "__main__":
    run(parse_args())
