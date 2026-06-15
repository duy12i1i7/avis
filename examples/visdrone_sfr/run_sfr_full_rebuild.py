from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DONE_MARKER_NAME = ".train_complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-platform SFR full train/eval runner.")
    parser.add_argument("--stage", default="train", choices=("train", "eval", "all"))
    parser.add_argument("--model", default="ultralytics/cfg/models/26/yolo26n-sfrfull-visdrone.yaml")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--name", default="yolo26n_sfrfull_visdrone")
    parser.add_argument("--project", default=str(ROOT / "runs" / "sfr_full"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
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


def count_epochs(results_path: Path) -> int:
    if not results_path.exists():
        return 0
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    except OSError:
        return 0


def resolve_run_dir(project: Path, base_name: str) -> tuple[Path, int, bool, bool, bool]:
    candidates: list[Path] = []
    for run_dir in project.glob(f"{base_name}*"):
        if not run_dir.is_dir():
            continue
        suffix = run_dir.name[len(base_name) :]
        if suffix and not suffix.isdigit():
            continue
        candidates.append(run_dir)
    if not candidates:
        target = project / base_name
        return target, 0, False, False, False

    def sort_key(run_dir: Path) -> tuple[int, int, int, int, str]:
        return (
            count_epochs(run_dir / "results.csv"),
            int((run_dir / DONE_MARKER_NAME).exists()),
            int((run_dir / "weights" / "last.pt").exists()),
            int((run_dir / "weights" / "best.pt").exists()),
            run_dir.name,
        )

    target = max(candidates, key=sort_key)
    return (
        target,
        count_epochs(target / "results.csv"),
        (target / "weights" / "last.pt").exists(),
        (target / "weights" / "best.pt").exists(),
        (target / DONE_MARKER_NAME).exists(),
    )


def checkpoint_is_finite(ckpt: Path) -> bool:
    if not ckpt.exists():
        return False
    try:
        import torch
        from ultralytics.utils.patches import torch_load

        payload = torch_load(ckpt, map_location="cpu")
        model = payload.get("ema") or payload.get("model")
        if model is None or not hasattr(model, "state_dict"):
            return False
        state_dict = model.state_dict()
        return all(torch.isfinite(v).all() for v in state_dict.values() if isinstance(v, torch.Tensor))
    except Exception:
        return False


def should_run_tiny_eval(data_path: str, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    data = YAML.load(check_yaml(data_path))
    names = data["names"]
    name_lookup = list(names.values()) if isinstance(names, dict) else list(names)
    targets = {"pedestrian", "people", "person"}
    return any(name in targets for name in name_lookup)


def run_subprocess(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def base_train_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "examples" / "visdrone_sfr" / "train_sfr_module_bench.py"),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--patience",
        str(args.patience),
        "--optimizer",
        str(args.optimizer),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.lr0 is not None:
        cmd += ["--lr0", str(args.lr0)]
    if args.amp is True:
        cmd.append("--amp")
    elif args.amp is False:
        cmd.append("--no-amp")
    return cmd


def run_train(args: argparse.Namespace) -> None:
    project = Path(args.project).expanduser().resolve()
    run_dir, completed_epochs, has_last, has_best, is_done = resolve_run_dir(project, args.name)
    weights_dir = run_dir / "weights"
    last_ckpt = weights_dir / "last.pt"
    last_finite_ckpt = weights_dir / "last_finite.pt"
    best_ckpt = weights_dir / "best.pt"
    done_marker = run_dir / DONE_MARKER_NAME

    if is_done:
        print()
        print(f"=== SKIP {args.name} (complete marker found in {run_dir.name}) ===")
        return

    if completed_epochs >= args.epochs and (has_best or has_last):
        run_dir.mkdir(parents=True, exist_ok=True)
        done_marker.touch()
        print()
        print(f"=== SKIP {args.name} (completed {completed_epochs}/{args.epochs} epochs in {run_dir.name}) ===")
        return

    resume_ckpt: Path | None = None
    if last_finite_ckpt.exists() and checkpoint_is_finite(last_finite_ckpt):
        resume_ckpt = last_finite_ckpt
    elif has_last:
        if checkpoint_is_finite(last_ckpt):
            resume_ckpt = last_ckpt
        else:
            print(f"=== WARN {args.name}: last.pt is non-finite, preserving as last.nan.pt ===")
            weights_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(last_ckpt), str(weights_dir / "last.nan.pt"))
            has_last = False

    if resume_ckpt is None and has_best and checkpoint_is_finite(best_ckpt):
        resume_ckpt = best_ckpt

    if resume_ckpt is None and (has_last or has_best or completed_epochs > 0):
        quarantine_run_dir = run_dir.with_name(f"{run_dir.name}_corrupt_nan_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(
            f"=== WARN {args.name}: no finite checkpoint available, moving {run_dir.name} "
            f"to {quarantine_run_dir.name} and restarting fresh ==="
        )
        shutil.move(str(run_dir), str(quarantine_run_dir))
        completed_epochs = 0
        run_dir = project / args.name
        done_marker = run_dir / DONE_MARKER_NAME

    print()
    if resume_ckpt is not None:
        print(f"=== RESUME {args.name} from {run_dir.name} using {resume_ckpt.name} ({completed_epochs}/{args.epochs} epochs logged) ===")
        cmd = base_train_cmd(args) + ["--resume", str(resume_ckpt)]
    else:
        print(f"=== TRAIN {args.name} ===")
        cmd = base_train_cmd(args) + [
            "--model",
            str(args.model),
            "--weights",
            str(args.weights),
            "--data",
            str(args.data),
            "--project",
            str(project),
            "--name",
            str(args.name),
        ]
    run_subprocess(cmd)
    run_dir.mkdir(parents=True, exist_ok=True)
    done_marker.touch()


def resolve_eval_dir(run_dir: Path) -> Path | None:
    base = f"{run_dir.name}_val"
    candidates: list[Path] = []
    for child in run_dir.parent.glob(f"{base}*"):
        if not child.is_dir():
            continue
        suffix = child.name[len(base) :]
        if suffix and not suffix.isdigit():
            continue
        candidates.append(child)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (int(path.name[len(base) :] or "0"), path.stat().st_mtime_ns))


def run_eval(args: argparse.Namespace) -> None:
    project = Path(args.project).expanduser().resolve()
    run_dir, _, _, _, _ = resolve_run_dir(project, args.name)
    ckpt = run_dir / "weights" / "best.pt"
    if not ckpt.exists():
        print(f"Skipping {args.name}: missing checkpoint {ckpt}", file=sys.stderr)
        return

    val_name = f"{args.name}_val"
    val_dir = project / val_name
    json_path = val_dir / "predictions.json"
    tiny_path = val_dir / "tiny_human_metrics.json"

    print()
    print(f"=== EVAL {args.name} from {run_dir.name} ===")
    val_cmd = [
        sys.executable,
        str(ROOT / "examples" / "visdrone_sfr" / "val_psr_yolo26.py"),
        "--model",
        str(ckpt),
        "--data",
        str(args.data),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
        "--project",
        str(project),
        "--name",
        val_name,
        "--exist-ok",
        "--save-json",
    ]
    val_cmd.append("--plots" if args.plots else "--no-plots")
    run_subprocess(val_cmd)

    if should_run_tiny_eval(str(args.data), args.tiny_eval):
        if json_path.exists() and json_path.stat().st_size > 0:
            tiny_cmd = [
                sys.executable,
                str(ROOT / "examples" / "visdrone_sfr" / "tiny_human_eval.py"),
                "--pred-json",
                str(json_path),
                "--data",
                str(args.data),
                "--save",
                str(tiny_path),
            ]
            try:
                run_subprocess(tiny_cmd)
            except subprocess.CalledProcessError as exc:
                print(f"Skipping tiny-human eval for {args.name}: {exc}", file=sys.stderr)
        else:
            print(f"Skipping tiny-human eval for {args.name}: missing or empty {json_path}", file=sys.stderr)
    else:
        print(f"Skipping tiny-human eval for {args.name}: dataset has no pedestrian/people/person classes.")


def main() -> None:
    os.chdir(ROOT)
    args = parse_args()
    if args.stage in {"train", "all"}:
        run_train(args)
    if args.stage in {"eval", "all"}:
        run_eval(args)


if __name__ == "__main__":
    main()
