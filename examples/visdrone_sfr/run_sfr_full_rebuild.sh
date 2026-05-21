#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/examples/visdrone_sfr/kaggle_resume_utils.sh"

STAGE="train"
MODEL="ultralytics/cfg/models/26/yolo26n-sfrfull-visdrone.yaml"
WEIGHTS="yolo26n.pt"
DATA="VisDrone.yaml"
RUN_NAME="yolo26n_sfrfull_visdrone"
PROJECT="${ROOT}/runs/sfr_full"
IMGSZ="960"
BATCH="8"
EPOCHS="300"
PATIENCE="80"
WORKERS="4"
DEVICE="0"
OPTIMIZER="auto"
SEED=""
LR0=""
AMP_MODE="default"
DONE_MARKER_NAME=".train_complete"
TINY_EVAL_MODE="auto"
PLOTS="1"

usage() {
  cat <<'EOF'
Usage:
  bash examples/visdrone_sfr/run_sfr_full_rebuild.sh \
    --stage train \
    --device 0 \
    --epochs 300 \
    --batch 8 \
    --imgsz 960

This runner is intentionally scoped to the rebuilt YOLO26n + VisDrone SFR full experiment.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --name) RUN_NAME="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --lr0) LR0="$2"; shift 2 ;;
    --tiny-eval) TINY_EVAL_MODE="$2"; shift 2 ;;
    --plots) PLOTS="1"; shift ;;
    --no-plots) PLOTS="0"; shift ;;
    --amp) AMP_MODE="on"; shift ;;
    --no-amp) AMP_MODE="off"; shift ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${STAGE}" != "train" && "${STAGE}" != "eval" && "${STAGE}" != "all" ]]; then
  echo "--stage must be one of: train, eval, all" >&2
  exit 1
fi

if [[ "${TINY_EVAL_MODE}" != "auto" && "${TINY_EVAL_MODE}" != "always" && "${TINY_EVAL_MODE}" != "never" ]]; then
  echo "--tiny-eval must be one of: auto, always, never" >&2
  exit 1
fi

resolve_run_dir() {
  local project="$1"
  local base_name="$2"
  python3 - "${project}" "${base_name}" "${DONE_MARKER_NAME}" <<'PY'
from pathlib import Path
import sys

project = Path(sys.argv[1])
base_name = sys.argv[2]
done_marker_name = sys.argv[3]

def count_epochs(run_dir: Path) -> int:
    path = run_dir / "results.csv"
    if not path.exists():
        return 0
    try:
        lines = path.read_text().strip().splitlines()
    except OSError:
        return 0
    return max(len(lines) - 1, 0)

candidates = []
for run_dir in project.glob(f"{base_name}*"):
    if not run_dir.is_dir():
        continue
    suffix = run_dir.name[len(base_name):]
    if suffix and not suffix.isdigit():
        continue
    candidates.append(run_dir)

if not candidates:
    target = project / base_name
    print(f"{target}|0|0|0|0")
    raise SystemExit

def sort_key(run_dir: Path):
    return (
        count_epochs(run_dir),
        int((run_dir / done_marker_name).exists()),
        int((run_dir / "weights" / "last.pt").exists()),
        int((run_dir / "weights" / "best.pt").exists()),
        run_dir.name,
    )

target = max(candidates, key=sort_key)
print(
    f"{target}|{count_epochs(target)}|"
    f"{int((target / 'weights' / 'last.pt').exists())}|"
    f"{int((target / 'weights' / 'best.pt').exists())}|"
    f"{int((target / done_marker_name).exists())}"
)
PY
}

checkpoint_is_finite() {
  local ckpt="$1"
  python3 - "${ckpt}" <<'PY'
from pathlib import Path
import sys

import torch

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit

try:
    ckpt = torch.load(path, map_location="cpu")
    model = ckpt.get("ema") or ckpt.get("model")
    if model is None or not hasattr(model, "state_dict"):
        print(0)
        raise SystemExit
    state_dict = model.state_dict()
    finite = all(torch.isfinite(v).all() for v in state_dict.values() if isinstance(v, torch.Tensor))
    print(int(finite))
except Exception:
    print(0)
PY
}

should_run_tiny_eval() {
  local data_path="$1"
  local mode="$2"
  if [[ "${mode}" == "always" ]]; then
    printf '1\n'
    return 0
  fi
  if [[ "${mode}" == "never" ]]; then
    printf '0\n'
    return 0
  fi
  python3 - "${data_path}" <<'PY'
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml
import sys

data = YAML.load(check_yaml(sys.argv[1]))
names = data["names"]
name_lookup = list(names.values()) if isinstance(names, dict) else list(names)
targets = {"pedestrian", "people", "person"}
print(int(any(name in targets for name in name_lookup)))
PY
}

run_train() {
  local resolved
  local run_dir
  local completed_epochs
  local has_last
  local has_best
  local is_done
  local last_ckpt
  local last_finite_ckpt
  local best_ckpt
  local done_marker
  local resume_ckpt=""
  local quarantine_run_dir=""
  local -a train_extra=()
  if [[ -n "${SEED}" ]]; then
    train_extra+=(--seed "${SEED}")
  fi
  if [[ -n "${LR0}" ]]; then
    train_extra+=(--lr0 "${LR0}")
  fi
  if [[ "${AMP_MODE}" == "on" ]]; then
    train_extra+=(--amp)
  elif [[ "${AMP_MODE}" == "off" ]]; then
    train_extra+=(--no-amp)
  fi

  resolved="$(resolve_run_dir "${PROJECT}" "${RUN_NAME}")"
  IFS="|" read -r run_dir completed_epochs has_last has_best is_done <<<"${resolved}"
  last_ckpt="${run_dir}/weights/last.pt"
  last_finite_ckpt="${run_dir}/weights/last_finite.pt"
  best_ckpt="${run_dir}/weights/best.pt"
  done_marker="${run_dir}/${DONE_MARKER_NAME}"

  if [[ "${is_done}" == "1" ]]; then
    echo
    echo "=== SKIP ${RUN_NAME} (complete marker found in $(basename "${run_dir}")) ==="
    return 0
  fi

  if [[ "${completed_epochs}" -ge "${EPOCHS}" ]] && [[ "${has_best}" == "1" || "${has_last}" == "1" ]]; then
    mkdir -p "${run_dir}"
    touch "${done_marker}"
    echo
    echo "=== SKIP ${RUN_NAME} (completed ${completed_epochs}/${EPOCHS} epochs in $(basename "${run_dir}")) ==="
    return 0
  fi

  echo
  if [[ -f "${last_finite_ckpt}" && "$(checkpoint_is_finite "${last_finite_ckpt}")" == "1" ]]; then
    resume_ckpt="${last_finite_ckpt}"
  elif [[ "${has_last}" == "1" ]]; then
    if [[ "$(checkpoint_is_finite "${last_ckpt}")" == "1" ]]; then
      resume_ckpt="${last_ckpt}"
    else
      echo "=== WARN ${RUN_NAME}: last.pt is non-finite, preserving as last.nan.pt ==="
      mv -f "${last_ckpt}" "${run_dir}/weights/last.nan.pt"
      has_last="0"
    fi
  fi

  if [[ -z "${resume_ckpt}" && "${has_best}" == "1" && "$(checkpoint_is_finite "${best_ckpt}")" == "1" ]]; then
    resume_ckpt="${best_ckpt}"
  fi

  if [[ -z "${resume_ckpt}" && ( "${has_last}" == "1" || "${has_best}" == "1" || "${completed_epochs}" -gt 0 ) ]]; then
    quarantine_run_dir="${run_dir}_corrupt_nan_$(date +%Y%m%d_%H%M%S)"
    echo "=== WARN ${RUN_NAME}: no finite checkpoint available, moving $(basename "${run_dir}") to $(basename "${quarantine_run_dir}") and restarting fresh ==="
    mv "${run_dir}" "${quarantine_run_dir}"
    completed_epochs="0"
  fi

  if [[ -n "${resume_ckpt}" ]]; then
    echo "=== RESUME ${RUN_NAME} from $(basename "${run_dir}") using $(basename "${resume_ckpt}") (${completed_epochs}/${EPOCHS} epochs logged) ==="
    python3 examples/visdrone_sfr/train_sfr_module_bench.py \
      --resume "${resume_ckpt}" \
      --epochs "${EPOCHS}" \
      --imgsz "${IMGSZ}" \
      --batch "${BATCH}" \
      --patience "${PATIENCE}" \
      --optimizer "${OPTIMIZER}" \
      --workers "${WORKERS}" \
      --device "${DEVICE}" \
      "${train_extra[@]}"
  else
    run_dir="${PROJECT}/${RUN_NAME}"
    done_marker="${run_dir}/${DONE_MARKER_NAME}"
    echo "=== TRAIN ${RUN_NAME} ==="
    python3 examples/visdrone_sfr/train_sfr_module_bench.py \
      --model "${MODEL}" \
      --weights "${WEIGHTS}" \
      --data "${DATA}" \
      --imgsz "${IMGSZ}" \
      --batch "${BATCH}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --optimizer "${OPTIMIZER}" \
      --workers "${WORKERS}" \
      --device "${DEVICE}" \
      --project "${PROJECT}" \
      --name "${RUN_NAME}" \
      "${train_extra[@]}"
  fi

  mkdir -p "${run_dir}"
  touch "${done_marker}"
  kaggle_snapshot_tree "${ROOT}" "runs/sfr_full" "sfr_full"
}

run_eval() {
  local resolved
  local run_dir
  local completed_epochs
  local has_last
  local has_best
  local is_done
  local ckpt
  local val_name="${RUN_NAME}_val"
  local val_dir="${PROJECT}/${val_name}"
  local json_path="${val_dir}/predictions.json"
  local tiny_path="${val_dir}/tiny_human_metrics.json"
  local -a plot_args=()

  resolved="$(resolve_run_dir "${PROJECT}" "${RUN_NAME}")"
  IFS="|" read -r run_dir completed_epochs has_last has_best is_done <<<"${resolved}"
  ckpt="${run_dir}/weights/best.pt"

  if [[ ! -f "${ckpt}" ]]; then
    echo "Skipping ${RUN_NAME}: missing checkpoint ${ckpt}" >&2
    return 0
  fi

  if [[ "${PLOTS}" == "1" ]]; then
    plot_args+=(--plots)
  else
    plot_args+=(--no-plots)
  fi

  echo
  echo "=== EVAL ${RUN_NAME} from $(basename "${run_dir}") ==="
  python3 examples/visdrone_sfr/val_psr_yolo26.py \
    --model "${ckpt}" \
    --data "${DATA}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --device "${DEVICE}" \
    --workers "${WORKERS}" \
    --project "${PROJECT}" \
    --name "${val_name}" \
    --exist-ok \
    --save-json \
    "${plot_args[@]}"

  if [[ "$(should_run_tiny_eval "${DATA}" "${TINY_EVAL_MODE}")" == "1" ]]; then
    if [[ -s "${json_path}" ]]; then
      python3 examples/visdrone_sfr/tiny_human_eval.py \
        --pred-json "${json_path}" \
        --data "${DATA}" \
        --save "${tiny_path}"
    else
      echo "Skipping tiny-human eval for ${RUN_NAME}: missing or empty ${json_path}" >&2
    fi
  else
    echo "Skipping tiny-human eval for ${RUN_NAME}: dataset has no pedestrian/people/person classes."
  fi

  kaggle_snapshot_tree "${ROOT}" "runs/sfr_full" "sfr_full"
}

if [[ "${STAGE}" == "train" || "${STAGE}" == "all" ]]; then
  run_train
fi

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
  run_eval
fi
