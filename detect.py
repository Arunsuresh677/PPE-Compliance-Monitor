"""
PPE Compliance Monitor - Inference / Detection Script
Author: Arun S | Nithil Innovations
Description: Detect PPE compliance in real-time from webcam, video, image, or RTSP stream
Classes: Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person,
         Safety Cone, Safety Vest, machinery, vehicle
"""

import argparse
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# Class colors — Green for compliant, Red for violations, Yellow for context
CLASS_COLORS = {
    "hardhat":        (0, 200, 0),      # Green
    "mask":           (0, 180, 0),      # Green
    "safety vest":    (0, 160, 0),      # Green
    "no-hardhat":     (0, 0, 220),      # Red
    "no-mask":        (0, 0, 200),      # Red
    "no-safety vest": (0, 0, 180),      # Red
    "person":         (200, 200, 0),    # Yellow
    "safety cone":    (0, 165, 255),    # Orange
    "machinery":      (180, 100, 0),    # Brown
    "vehicle":        (150, 80, 0),     # Dark Brown
}

# Classes that trigger safety alert
VIOLATION_CLASSES = {"no-hardhat", "no-mask", "no-safety vest"}


def parse_args():
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


def draw_detections(frame, results, conf_threshold):
    """Draw bounding boxes, labels, and alerts on frame."""
    violations = []

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

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if cls_key in VIOLATION_CLASSES:
                violations.append(cls_name)

    # Alert banner for violations
    if violations:
        unique = list(set(violations))
        alert_text = f"⚠ VIOLATION: {', '.join(unique)}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 180), -1)
        cv2.putText(frame, alert_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 140, 0), -1)
        cv2.putText(frame, "✓ PPE COMPLIANT", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame


def run(args):
    print("\n" + "=" * 55)
    print("   🦺 PPE Compliance Monitor — YOLOv8 Inference")
    print("=" * 55)
    print(f"   Model   : {args.model}")
    print(f"   Source  : {args.source}")
    print(f"   Conf    : {args.conf}")
    print("   Press Q to quit")
    print("=" * 55 + "\n")

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model  = YOLO(args.model)
    source = int(args.source) if args.source.isdigit() else args.source
    is_image = isinstance(source, str) and Path(source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]

    if args.save:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # Image mode
    if is_image:
        results = model.predict(source=source, conf=args.conf, imgsz=args.imgsz, verbose=False)
        frame   = cv2.imread(source)
        frame   = draw_detections(frame, results, args.conf)
        if not args.no_show:
            cv2.imshow("PPE Compliance Monitor", frame)
            cv2.waitKey(0)
        if args.save:
            out_path = Path(args.output) / Path(source).name
            cv2.imwrite(str(out_path), frame)
            print(f"✅ Saved: {out_path}")
        cv2.destroyAllWindows()
        return

    # Video / Webcam / RTSP mode
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        out_path = Path(args.output) / "output.mp4"
        writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
        print(f"💾 Saving output to: {out_path}")

    frame_count = 0
    fps_display = 0
    t_start     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        results     = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        frame       = draw_detections(frame, results, args.conf)
        frame_count += 1
        elapsed     = time.time() - t_start

        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t_start     = time.time()

        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if writer:
            writer.write(frame)

        if not args.no_show:
            cv2.imshow("PPE Compliance Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n🛑 Stopped by user.")
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("✅ Detection complete.")


if __name__ == "__main__":
    args = parse_args()
    run(args)
