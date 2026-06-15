from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-platform SFR full family suite runner.")
    parser.add_argument("--stage", default="train", choices=("train", "eval", "all"))
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--dataset-tag", default="visdrone")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(ROOT / "runs" / "sfr_full"))
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--tiny-eval", default="auto", choices=("auto", "always", "never"))
    parser.add_argument("--plots", dest="plots", action="store_true")
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(plots=True, amp=None)
    return parser.parse_args()


def build_runs(dataset_tag: str) -> list[tuple[str, str, str]]:
    return [
        (f"yolo26n_sfrfull_{dataset_tag}", "ultralytics/cfg/models/26/yolo26n-sfrfull-visdrone.yaml", "yolo26n.pt"),
        (f"yolo11n_sfrfull_{dataset_tag}", "ultralytics/cfg/models/11/yolo11n-sfrfull-visdrone.yaml", "yolo11n.pt"),
        (f"yolo12n_sfrfull_{dataset_tag}", "ultralytics/cfg/models/12/yolo12n-sfrfull-visdrone.yaml", "yolo12n.pt"),
        (f"yolov8n_sfrfull_{dataset_tag}", "ultralytics/cfg/models/v8/yolov8n-sfrfull-visdrone.yaml", "yolov8n.pt"),
        (f"yolov10n_sfrfull_{dataset_tag}", "ultralytics/cfg/models/v10/yolov10n-sfrfull-visdrone.yaml", "yolov10n.pt"),
    ]


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    for name, model, weights in build_runs(args.dataset_tag):
        cmd = [
            sys.executable,
            str(ROOT / "examples" / "visdrone_sfr" / "run_sfr_full_rebuild.py"),
            "--stage",
            args.stage,
            "--model",
            model,
            "--weights",
            weights,
            "--data",
            str(args.data),
            "--name",
            name,
            "--project",
            str(args.project),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--workers",
            str(args.workers),
            "--device",
            str(args.device),
            "--optimizer",
            str(args.optimizer),
            "--tiny-eval",
            args.tiny_eval,
        ]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.lr0 is not None:
            cmd += ["--lr0", str(args.lr0)]
        if args.plots:
            cmd.append("--plots")
        else:
            cmd.append("--no-plots")
        if args.amp is True:
            cmd.append("--amp")
        elif args.amp is False:
            cmd.append("--no-amp")
        print()
        print(f"=== SFR FULL {name} ===")
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
