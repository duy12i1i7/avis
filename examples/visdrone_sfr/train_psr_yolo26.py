from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the routed YOLO26 VisDrone prototype.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26-sfr-visdrone.yaml"),
        help="Path to the custom model YAML.",
    )
    parser.add_argument("--data", default="VisDrone.yaml", help="Dataset YAML. Defaults to Ultralytics VisDrone config.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs" / "visdrone"))
    parser.add_argument("--name", default="yolo26_sfr_visdrone")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM for faster training.")
    parser.add_argument("--resume", default=None, help="Resume from a previous checkpoint if provided.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.resume or args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        cache=args.cache,
        pretrained=False,
        close_mosaic=20,
        degrees=0.0,
        copy_paste=0.0,
        save_json=False,
    )


if __name__ == "__main__":
    main()
