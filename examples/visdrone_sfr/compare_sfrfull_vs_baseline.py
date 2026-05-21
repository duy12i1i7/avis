from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from summarize_sfrfull_results import collect_runs, format_float, format_int


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_ROOT = ROOT / "runs" / "sfr_suite"
DEFAULT_SFRFULL_ROOT = ROOT / "runs" / "sfr_full"
FAMILIES = ("yolo26n", "yolo11n", "yolo12n", "yolov8n", "yolov10n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rebuilt SFR full runs against baseline runs.")
    parser.add_argument(
        "--baseline-root",
        default=str(DEFAULT_BASELINE_ROOT),
        help="Root directory containing baseline/base runs, usually runs/sfr_suite.",
    )
    parser.add_argument(
        "--sfrfull-root",
        default=str(DEFAULT_SFRFULL_ROOT),
        help="Root directory containing rebuilt SFR full runs, usually runs/sfr_full.",
    )
    parser.add_argument("--output-csv", default=None, help="Optional path to save comparison as CSV.")
    parser.add_argument("--output-md", default=None, help="Optional path to save comparison as Markdown.")
    return parser.parse_args()


def detect_family(run_name: str) -> str | None:
    lower = run_name.lower()
    for family in FAMILIES:
        if lower.startswith(family):
            return family
    return None


def detect_variant(run_name: str) -> str | None:
    lower = run_name.lower()
    if "_base_" in lower:
        return "base"
    if "_sfrfull_" in lower:
        return "sfrfull"
    return None


def normalize_dataset(name: Any) -> str:
    text = str(name or "")
    lower = text.lower()
    if lower == "visdrone":
        return "VisDrone"
    if lower == "tinyperson":
        return "TinyPerson"
    return text


def choose_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            int(row.get("last_epoch") or 0),
            1 if row.get("eval_dir") else 0,
            1 if row.get("tiny_human_ap50_95") is not None else 0,
            float(row.get("map50_95") or float("-inf")),
            row.get("run", ""),
        ),
    )


def index_runs(rows: list[dict[str, Any]], expected_variant: str) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        family = detect_family(str(row.get("run", "")))
        variant = detect_variant(str(row.get("run", "")))
        if family is None or variant != expected_variant:
            continue
        key = (normalize_dataset(row.get("dataset")), family)
        grouped.setdefault(key, []).append(row)
    return {key: choose_best_candidate(values) for key, values in grouped.items()}


def delta(current: Any, baseline: Any) -> float | None:
    if current is None or baseline is None:
        return None
    return float(current) - float(baseline)


def build_rows(
    baseline_index: dict[tuple[str, str], dict[str, Any]],
    sfrfull_index: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted(set(baseline_index) | set(sfrfull_index), key=lambda item: (item[0], item[1]))
    rows = []
    for dataset, family in keys:
        base = baseline_index.get((dataset, family))
        full = sfrfull_index.get((dataset, family))
        status = "ok"
        if base is None:
            status = "missing_base"
        elif full is None:
            status = "missing_sfrfull"
        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "status": status,
                "base_run": base.get("run", "") if base else "",
                "sfrfull_run": full.get("run", "") if full else "",
                "base_stop": base.get("stop", "") if base else "",
                "sfrfull_stop": full.get("stop", "") if full else "",
                "base_best_epoch": base.get("best_epoch") if base else None,
                "sfrfull_best_epoch": full.get("best_epoch") if full else None,
                "base_precision": base.get("precision") if base else None,
                "sfrfull_precision": full.get("precision") if full else None,
                "delta_precision": delta(full.get("precision") if full else None, base.get("precision") if base else None),
                "base_recall": base.get("recall") if base else None,
                "sfrfull_recall": full.get("recall") if full else None,
                "delta_recall": delta(full.get("recall") if full else None, base.get("recall") if base else None),
                "base_map50": base.get("map50") if base else None,
                "sfrfull_map50": full.get("map50") if full else None,
                "delta_map50": delta(full.get("map50") if full else None, base.get("map50") if base else None),
                "base_map50_95": base.get("map50_95") if base else None,
                "sfrfull_map50_95": full.get("map50_95") if full else None,
                "delta_map50_95": delta(full.get("map50_95") if full else None, base.get("map50_95") if base else None),
                "base_tiny_ap50_95": base.get("tiny_human_ap50_95") if base else None,
                "sfrfull_tiny_ap50_95": full.get("tiny_human_ap50_95") if full else None,
                "delta_tiny_ap50_95": delta(
                    full.get("tiny_human_ap50_95") if full else None,
                    base.get("tiny_human_ap50_95") if base else None,
                ),
                "base_tiny_ap50": base.get("tiny_human_ap50") if base else None,
                "sfrfull_tiny_ap50": full.get("tiny_human_ap50") if full else None,
                "delta_tiny_ap50": delta(
                    full.get("tiny_human_ap50") if full else None,
                    base.get("tiny_human_ap50") if base else None,
                ),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Dataset",
        "Family",
        "Status",
        "BaseRun",
        "SFRFullRun",
        "BaseStop",
        "FullStop",
        "BaseBest",
        "FullBest",
        "BaseP",
        "FullP",
        "ΔP",
        "BaseR",
        "FullR",
        "ΔR",
        "Base50-95",
        "Full50-95",
        "Δ50-95",
        "BaseTiny50-95",
        "FullTiny50-95",
        "ΔTiny50-95",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [
            row["dataset"],
            f"`{row['family']}`",
            row["status"],
            f"`{row['base_run']}`" if row["base_run"] else "",
            f"`{row['sfrfull_run']}`" if row["sfrfull_run"] else "",
            row["base_stop"],
            row["sfrfull_stop"],
            format_int(row["base_best_epoch"]),
            format_int(row["sfrfull_best_epoch"]),
            format_float(row["base_precision"]),
            format_float(row["sfrfull_precision"]),
            format_float(row["delta_precision"]),
            format_float(row["base_recall"]),
            format_float(row["sfrfull_recall"]),
            format_float(row["delta_recall"]),
            format_float(row["base_map50_95"]),
            format_float(row["sfrfull_map50_95"]),
            format_float(row["delta_map50_95"]),
            format_float(row["base_tiny_ap50_95"]),
            format_float(row["sfrfull_tiny_ap50_95"]),
            format_float(row["delta_tiny_ap50_95"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "family",
        "status",
        "base_run",
        "sfrfull_run",
        "base_stop",
        "sfrfull_stop",
        "base_best_epoch",
        "sfrfull_best_epoch",
        "base_precision",
        "sfrfull_precision",
        "delta_precision",
        "base_recall",
        "sfrfull_recall",
        "delta_recall",
        "base_map50",
        "sfrfull_map50",
        "delta_map50",
        "base_map50_95",
        "sfrfull_map50_95",
        "delta_map50_95",
        "base_tiny_ap50_95",
        "sfrfull_tiny_ap50_95",
        "delta_tiny_ap50_95",
        "base_tiny_ap50",
        "sfrfull_tiny_ap50",
        "delta_tiny_ap50",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baseline_rows = collect_runs(Path(args.baseline_root).expanduser().resolve())
    sfrfull_rows = collect_runs(Path(args.sfrfull_root).expanduser().resolve())

    baseline_index = index_runs(baseline_rows, expected_variant="base")
    sfrfull_index = index_runs(sfrfull_rows, expected_variant="sfrfull")
    rows = build_rows(baseline_index, sfrfull_index)
    if not rows:
        raise SystemExit("No comparable runs found.")

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
