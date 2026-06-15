from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-platform SFR full dataset suite runner.")
    parser.add_argument("--stage", default="train", choices=("train", "eval", "all"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--project-root", default=str(ROOT / "runs" / "sfr_full"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--visdrone-data", default="VisDrone.yaml")
    parser.add_argument("--visdrone-imgsz", type=int, default=None)
    parser.add_argument("--visdrone-batch", type=int, default=None)
    parser.add_argument("--tinyperson-data", default="TinyPerson.yaml")
    parser.add_argument("--tinyperson-imgsz", type=int, default=None)
    parser.add_argument("--tinyperson-batch", type=int, default=None)
    parser.add_argument("--tiny-eval", default="auto", choices=("auto", "always", "never"))
    parser.add_argument("--plots", dest="plots", action="store_true")
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(plots=True, amp=None)
    return parser.parse_args()


def run_dataset(args: argparse.Namespace, tag: str, data: str, imgsz: int, batch: int) -> None:
    if not data:
        return
    cmd = [
        sys.executable,
        str(ROOT / "examples" / "visdrone_sfr" / "run_sfrfull_family_suite.py"),
        "--stage",
        args.stage,
        "--data",
        data,
        "--dataset-tag",
        tag,
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--project",
        str(Path(args.project_root).expanduser().resolve() / tag),
        "--optimizer",
        str(args.optimizer),
        "--tiny-eval",
        args.tiny_eval,
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.plots:
        cmd.append("--plots")
    else:
        cmd.append("--no-plots")
    if args.amp is True:
        cmd.append("--amp")
    elif args.amp is False:
        cmd.append("--no-amp")
    print()
    print(f"########## SFR FULL DATASET {tag} ##########")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    run_dataset(
        args,
        "visdrone",
        args.visdrone_data,
        args.visdrone_imgsz or args.imgsz,
        args.visdrone_batch or args.batch,
    )
    run_dataset(
        args,
        "tinyperson",
        args.tinyperson_data,
        args.tinyperson_imgsz or args.imgsz,
        args.tinyperson_batch or args.batch,
    )


if __name__ == "__main__":
    main()
