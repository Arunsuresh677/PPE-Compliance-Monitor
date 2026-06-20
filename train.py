"""
PPE Compliance Monitor — Training Script
Author: Arun S | Nithil Innovations

Fine-tunes YOLOv8 on a PPE dataset in YOLO annotation format.
Run with --help to see all options.

Typical usage:
    python train.py --data data.yaml --epochs 50 --batch 16

For Google Colab (T4 GPU):
    python train.py --data data.yaml --epochs 50 --batch 16 --device 0
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for PPE compliance detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml (YOLO dataset config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base model weights: yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt",
    )
    parser.add_argument("--epochs",  type=int,   default=50,    help="Number of training epochs")
    parser.add_argument("--imgsz",   type=int,   default=640,   help="Input image size (px)")
    parser.add_argument("--batch",   type=int,   default=16,    help="Batch size (-1 = auto)")
    parser.add_argument("--device",  type=str,   default="",    help="Device: '' = auto, 'cpu', '0' = GPU 0")
    parser.add_argument("--project", type=str,   default="runs/ppe", help="Output directory")
    parser.add_argument("--name",    type=str,   default="exp",  help="Run name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint in --project/--name",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_path}\n"
            "Create a data.yaml file pointing to your images/labels directories."
        )

    log.info("=" * 60)
    log.info("PPE Compliance Monitor — YOLOv8 Training")
    log.info("=" * 60)
    log.info("Base model : %s", args.model)
    log.info("Data       : %s", args.data)
    log.info("Epochs     : %d", args.epochs)
    log.info("Image size : %d", args.imgsz)
    log.info("Batch      : %d", args.batch)
    log.info("Device     : %s", args.device or "auto")
    log.info("Output     : %s/%s", args.project, args.name)
    log.info("=" * 60)

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        resume=args.resume,
        save=True,
        plots=True,
        verbose=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    log.info("Training complete.")
    log.info("Best weights : %s", best_weights)
    log.info("mAP@0.5      : %.3f", results.results_dict.get("metrics/mAP50(B)", 0))
    log.info("Precision    : %.3f", results.results_dict.get("metrics/precision(B)", 0))
    log.info("Recall       : %.3f", results.results_dict.get("metrics/recall(B)", 0))

    log.info("")
    log.info("To run inference with the trained model:")
    log.info("  python detect.py --model %s --source 0", best_weights)


if __name__ == "__main__":
    train(parse_args())
