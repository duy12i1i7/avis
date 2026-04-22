from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the routed YOLO26 VisDrone prototype.")
    parser.add_argument("--model", required=True, help="Checkpoint or model YAML.")
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default="val")
    parser.add_argument("--project", default=str(ROOT / "runs" / "visdrone"))
    parser.add_argument("--name", default="yolo26_sfr_val")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the validation output directory if it exists.")
    parser.add_argument("--save-json", action="store_true", help="Export predictions.json for tiny-human AP analysis.")
    parser.add_argument("--plots", dest="plots", action="store_true", help="Save validation plots.")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="Disable validation plots.")
    parser.set_defaults(plots=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        split=args.split,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        save_json=args.save_json,
        plots=args.plots,
    )
    print(metrics)


if __name__ == "__main__":
    main()
