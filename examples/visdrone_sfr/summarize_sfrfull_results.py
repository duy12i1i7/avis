from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = ROOT / "runs" / "sfr_full"
VAL_DIR_RE = re.compile(r".*_val\d*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize rebuilt SFR full experiment results.")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT), help="Root directory containing SFR full runs.")
    parser.add_argument("--output-csv", default=None, help="Optional path to save the summary as CSV.")
    parser.add_argument("--output-md", default=None, help="Optional path to save the summary as Markdown.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    parsed[key] = None
                    continue
                value = value.strip()
                if value == "":
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def detect_dataset(run_dir: Path, args_data: Any) -> str:
    text = str(args_data or "").lower()
    if "tinyperson" in text:
        return "TinyPerson"
    if "visdrone" in text:
        return "VisDrone"
    parent = run_dir.parent.name.lower()
    if parent == "tinyperson":
        return "TinyPerson"
    if parent == "visdrone":
        return "VisDrone"
    suffix = run_dir.name.rsplit("_", 1)[-1].lower()
    if suffix == "tinyperson":
        return "TinyPerson"
    if suffix == "visdrone":
        return "VisDrone"
    if args_data:
        return Path(str(args_data)).stem
    return parent or "unknown"


def format_float(value: Any, digits: int = 5) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{float(value):.{digits}f}"


def format_int(value: Any) -> str:
    if value is None:
        return ""
    return str(int(value))


def format_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def stop_status(last_epoch: int | None, target_epochs: int | None) -> str:
    if last_epoch is None:
        return ""
    if target_epochs is not None and last_epoch >= target_epochs:
        return str(target_epochs)
    return f"ES@{last_epoch}"


def resolve_eval_dir(run_dir: Path) -> Path | None:
    parent = run_dir.parent
    base = f"{run_dir.name}_val"
    candidates = []
    for child in parent.glob(f"{base}*"):
        if not child.is_dir():
            continue
        suffix = child.name[len(base) :]
        if suffix and not suffix.isdigit():
            continue
        candidates.append(child)
    if not candidates:
        return None
    return max(candidates, key=lambda p: (int(p.name[len(base) :] or "0"), p.stat().st_mtime_ns))


def load_tiny_metrics(eval_dir: Path | None) -> dict[str, Any]:
    if eval_dir is None:
        return {}
    path = eval_dir / "tiny_human_metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    return data if isinstance(data, dict) else {}


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    args_path = run_dir / "args.yaml"
    results_path = run_dir / "results.csv"
    if not args_path.exists() or not results_path.exists():
        return None
    if VAL_DIR_RE.fullmatch(run_dir.name):
        return None

    args = load_yaml(args_path)
    rows = load_results(results_path)
    if not rows:
        return None

    metric_key = "metrics/mAP50-95(B)"
    best_row = max(rows, key=lambda row: float(row.get(metric_key) or float("-inf")))
    last_row = rows[-1]

    target_epochs = args.get("epochs")
    target_epochs = int(target_epochs) if isinstance(target_epochs, (int, float)) else None
    last_epoch = last_row.get("epoch")
    last_epoch = int(last_epoch) if isinstance(last_epoch, (int, float)) else None
    best_epoch = best_row.get("epoch")
    best_epoch = int(best_epoch) if isinstance(best_epoch, (int, float)) else None

    eval_dir = resolve_eval_dir(run_dir)
    tiny_metrics = load_tiny_metrics(eval_dir)

    return {
        "dataset": detect_dataset(run_dir, args.get("data")),
        "run": run_dir.name,
        "stop": stop_status(last_epoch, target_epochs),
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "precision": best_row.get("metrics/precision(B)"),
        "recall": best_row.get("metrics/recall(B)"),
        "map50": best_row.get("metrics/mAP50(B)"),
        "map50_95": best_row.get("metrics/mAP50-95(B)"),
        "train_router_loss": best_row.get("train/router_loss"),
        "val_router_loss": best_row.get("val/router_loss"),
        "tiny_human_ap50_95": tiny_metrics.get("tiny_human_ap50_95"),
        "tiny_human_ap50": tiny_metrics.get("tiny_human_ap50"),
        "optimizer": args.get("optimizer"),
        "batch": args.get("batch"),
        "imgsz": args.get("imgsz"),
        "device": args.get("device"),
        "workers": args.get("workers"),
        "amp": args.get("amp"),
        "model_cfg": Path(str(args.get("model", ""))).name if args.get("model") else "",
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir) if eval_dir else "",
    }


def collect_runs(project_root: Path) -> list[dict[str, Any]]:
    rows = []
    for results_path in project_root.rglob("results.csv"):
        run_dir = results_path.parent
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: (row["dataset"], row["run"]))
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Dataset",
        "Run",
        "Stop",
        "BestEp",
        "P",
        "R",
        "mAP50",
        "mAP50-95",
        "TrainRouter",
        "ValRouter",
        "TinyAP50-95",
        "TinyAP50",
        "Batch",
        "Imgsz",
        "Opt",
        "AMP",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [
            row["dataset"],
            f"`{row['run']}`",
            row["stop"],
            format_int(row["best_epoch"]),
            format_float(row["precision"]),
            format_float(row["recall"]),
            format_float(row["map50"]),
            format_float(row["map50_95"]),
            format_float(row["train_router_loss"]),
            format_float(row["val_router_loss"]),
            format_float(row["tiny_human_ap50_95"]),
            format_float(row["tiny_human_ap50"]),
            format_int(row["batch"]),
            format_int(row["imgsz"]),
            format_text(row["optimizer"]),
            format_text(row["amp"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "run",
        "stop",
        "best_epoch",
        "last_epoch",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "train_router_loss",
        "val_router_loss",
        "tiny_human_ap50_95",
        "tiny_human_ap50",
        "optimizer",
        "batch",
        "imgsz",
        "device",
        "workers",
        "amp",
        "model_cfg",
        "run_dir",
        "eval_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    rows = collect_runs(project_root)
    if not rows:
        raise SystemExit(f"No SFR full runs found under {project_root}")

    table = markdown_table(rows)
    print(table)

    if args.output_csv:
        write_csv(rows, Path(args.output_csv).expanduser().resolve())
    if args.output_md:
        path = Path(args.output_md).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
