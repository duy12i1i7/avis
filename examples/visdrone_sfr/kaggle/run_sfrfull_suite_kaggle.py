#!/usr/bin/env python3
"""Kaggle kernel entrypoint for the SFR full suite."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("job-config.json")


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    print("+", shlex.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def maybe_run(cmd: list[str], *, cwd: Path | None = None) -> int:
    try:
        return run(cmd, cwd=cwd, check=False)
    except Exception:
        return 1


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text())
    repo_url = config["repo_url"]
    git_ref = config["git_ref"]
    workdir = Path(config.get("workdir", "/kaggle/working/avis"))
    project_root_rel = Path(config.get("project_root_rel", "runs/sfr_full"))
    baseline_root = config.get("baseline_root", "")
    run_args = config["run_args"]

    print("Kaggle SFR full config:", json.dumps(config, indent=2), flush=True)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", ""), flush=True)
    maybe_run(["nvidia-smi", "-L"])

    if workdir.exists():
        shutil.rmtree(workdir)
    run(["git", "clone", repo_url, str(workdir)])
    run(["git", "checkout", git_ref], cwd=workdir)

    project_root = workdir / project_root_rel
    train_cmd = [
        "bash",
        "run_sfr_full_suite.sh",
        "--stage",
        "train",
        "--device",
        str(run_args["device"]),
        "--epochs",
        str(run_args["epochs"]),
        "--batch",
        str(run_args["batch"]),
        "--imgsz",
        str(run_args["imgsz"]),
        "--workers",
        str(run_args["workers"]),
    ]
    train_rc = run(train_cmd, cwd=workdir, check=False)

    summary_cmd = [
        "bash",
        "summarize_sfr_full_results.sh",
        "--project-root",
        str(project_root_rel),
        "--output-csv",
        str(project_root_rel / "sfrfull_summary.csv"),
        "--output-md",
        str(project_root_rel / "sfrfull_summary.md"),
    ]
    maybe_run(summary_cmd, cwd=workdir)

    if baseline_root and Path(baseline_root).exists():
        compare_cmd = [
            "bash",
            "compare_sfr_full_vs_baseline.sh",
            "--baseline-root",
            baseline_root,
            "--sfrfull-root",
            str(project_root_rel),
            "--output-csv",
            str(project_root_rel / "sfrfull_vs_base.csv"),
            "--output-md",
            str(project_root_rel / "sfrfull_vs_base.md"),
        ]
        maybe_run(compare_cmd, cwd=workdir)
    else:
        print(f"Skipping baseline comparison. Missing baseline root: {baseline_root}", flush=True)

    manifest_path = Path("/kaggle/working/sfrfull_job_manifest.json")
    manifest_path.write_text(json.dumps(config, indent=2))

    if project_root.exists():
        maybe_run(
            [
                "tar",
                "-czf",
                "/kaggle/working/sfr_full_artifacts.tgz",
                "-C",
                str(workdir),
                str(project_root_rel),
            ]
        )

    return train_rc


if __name__ == "__main__":
    raise SystemExit(main())
