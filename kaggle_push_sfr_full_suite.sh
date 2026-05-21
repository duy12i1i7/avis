#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_DIR="${ROOT}/.kaggle/sfr_full_suite_kernel"
KERNEL_ID=""
TITLE="SFR Full Suite"
REPO_URL=""
GIT_REF="$(git -C "${ROOT}" rev-parse HEAD)"
DEVICE="0,1"
EPOCHS="300"
BATCH="8"
IMGSZ="960"
WORKERS="4"
ACCELERATOR="NvidiaTeslaT4"
TIMEOUT=""
PRIVATE="true"
ENABLE_INTERNET="true"
PREPARE_ONLY="0"
declare -a DATASET_SOURCES=()
BASELINE_ROOT=""

usage() {
  cat <<'EOF'
Usage:
  bash kaggle_push_sfr_full_suite.sh \
    --kernel-id <username/kernel-slug> \
    [--title "SFR Full Suite"] \
    [--repo-url https://github.com/user/repo.git] \
    [--git-ref <commit>] \
    [--device 0,1] \
    [--epochs 300] \
    [--batch 8] \
    [--imgsz 960] \
    [--workers 4] \
    [--dataset-source <username/dataset-slug>] \
    [--baseline-root /kaggle/input/<dataset-slug>/runs/sfr_suite]

This generates a Kaggle kernel folder locally, pushes it with the Kaggle CLI,
and starts the run on Kaggle.
EOF
}

git_remote_to_https() {
  local remote="$1"
  if [[ "${remote}" =~ ^https?:// ]]; then
    printf '%s\n' "${remote}"
    return 0
  fi
  if [[ "${remote}" =~ ^git@github.com:(.+)$ ]]; then
    printf 'https://github.com/%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "${remote}" =~ ^ssh://git@github.com/(.+)$ ]]; then
    printf 'https://github.com/%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2 ;;
    --title) TITLE="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --git-ref) GIT_REF="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --kernel-dir) KERNEL_DIR="$2"; shift 2 ;;
    --dataset-source) DATASET_SOURCES+=("$2"); shift 2 ;;
    --baseline-root) BASELINE_ROOT="$2"; shift 2 ;;
    --accelerator) ACCELERATOR="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --prepare-only) PREPARE_ONLY="1"; shift ;;
    --public) PRIVATE="false"; shift ;;
    --private) PRIVATE="true"; shift ;;
    --internet) ENABLE_INTERNET="true"; shift ;;
    --no-internet) ENABLE_INTERNET="false"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${KERNEL_ID}" ]]; then
  echo "--kernel-id is required" >&2
  exit 1
fi

if [[ "${PREPARE_ONLY}" != "1" ]] && ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: python3 -m pip install kaggle" >&2
  exit 1
fi

if [[ -z "${REPO_URL}" ]]; then
  REMOTE_URL="$(git -C "${ROOT}" config --get remote.origin.url || true)"
  REPO_URL="$(git_remote_to_https "${REMOTE_URL:-}" || true)"
fi

if [[ -z "${REPO_URL}" ]]; then
  echo "Could not infer repo URL. Pass --repo-url explicitly." >&2
  exit 1
fi

if [[ "${#DATASET_SOURCES[@]}" -gt 0 && -z "${BASELINE_ROOT}" ]]; then
  LAST_SOURCE="${DATASET_SOURCES[0]}"
  LAST_SLUG="${LAST_SOURCE##*/}"
  BASELINE_ROOT="/kaggle/input/${LAST_SLUG}/runs/sfr_suite"
fi

mkdir -p "${KERNEL_DIR}"
cp "${ROOT}/examples/visdrone_sfr/kaggle/run_sfrfull_suite_kaggle.py" "${KERNEL_DIR}/run_sfrfull_suite_kaggle.py"

DATASET_SOURCES_JSON="$(printf '%s\n' "${DATASET_SOURCES[@]}" | python3 -c 'import json,sys; print(json.dumps([x.strip() for x in sys.stdin.read().splitlines() if x.strip()]))')"

KERNEL_DIR="${KERNEL_DIR}" \
KERNEL_ID="${KERNEL_ID}" \
TITLE="${TITLE}" \
PRIVATE="${PRIVATE}" \
ENABLE_INTERNET="${ENABLE_INTERNET}" \
REPO_URL="${REPO_URL}" \
GIT_REF="${GIT_REF}" \
DEVICE="${DEVICE}" \
EPOCHS="${EPOCHS}" \
BATCH="${BATCH}" \
IMGSZ="${IMGSZ}" \
WORKERS="${WORKERS}" \
BASELINE_ROOT="${BASELINE_ROOT}" \
DATASET_SOURCES_JSON="${DATASET_SOURCES_JSON}" \
python3 - <<'PY'
import json
import os
from pathlib import Path

kernel_dir = Path(os.environ["KERNEL_DIR"])
dataset_sources = json.loads(os.environ["DATASET_SOURCES_JSON"])

job_config = {
    "repo_url": os.environ["REPO_URL"],
    "git_ref": os.environ["GIT_REF"],
    "workdir": "/kaggle/working/avis",
    "project_root_rel": "runs/sfr_full",
    "baseline_root": os.environ["BASELINE_ROOT"],
    "run_args": {
        "device": os.environ["DEVICE"],
        "epochs": int(os.environ["EPOCHS"]),
        "batch": int(os.environ["BATCH"]),
        "imgsz": int(os.environ["IMGSZ"]),
        "workers": int(os.environ["WORKERS"]),
    },
}

metadata = {
    "id": os.environ["KERNEL_ID"],
    "title": os.environ["TITLE"],
    "code_file": "run_sfrfull_suite_kaggle.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": os.environ["PRIVATE"],
    "enable_gpu": "true",
    "enable_internet": os.environ["ENABLE_INTERNET"],
    "dataset_sources": dataset_sources,
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}

(kernel_dir / "job-config.json").write_text(json.dumps(job_config, indent=2))
(kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
PY

echo "Prepared Kaggle kernel in ${KERNEL_DIR}"
echo "Kernel id: ${KERNEL_ID}"
echo "Repo URL: ${REPO_URL}"
echo "Git ref: ${GIT_REF}"
if [[ "${DEVICE}" == "0,1" ]]; then
  echo "Note: Kaggle CLI exposes GPU enablement and accelerator selection, but not a documented GPU-count field."
  echo "The kernel script will request device 0,1 inside Kaggle, but actual dual-T4 availability still depends on Kaggle runtime support."
fi

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "Prepare-only mode: skipping kaggle kernels push"
  exit 0
fi

PUSH_CMD=(kaggle kernels push -p "${KERNEL_DIR}" --accelerator "${ACCELERATOR}")
if [[ -n "${TIMEOUT}" ]]; then
  PUSH_CMD+=(--timeout "${TIMEOUT}")
fi
"${PUSH_CMD[@]}"

echo
echo "Started Kaggle kernel: ${KERNEL_ID}"
echo "Watch with:"
echo "  bash kaggle_watch_kernel.sh --kernel-id ${KERNEL_ID}"
echo "Download outputs with:"
echo "  bash kaggle_pull_kernel_output.sh --kernel-id ${KERNEL_ID}"
