#!/bin/bash
set -euo pipefail

# Full SFR experiment matrix for VisDrone:
# - YOLO26n baseline + host-module ablations
# - YOLO11n/YOLOv8n/YOLOv10n/YOLO12n baseline + SFRC2f transfer
#
# Examples:
# bash examples/visdrone_sfr/run_sfr_full_matrix.sh
# bash examples/visdrone_sfr/run_sfr_full_matrix.sh --stage train --device 0 --epochs 300
# bash examples/visdrone_sfr/run_sfr_full_matrix.sh --stage eval --batch 8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

STAGE="all"
DATA="VisDrone.yaml"
DATASET_TAG="visdrone"
IMGSZ="960"
BATCH="16"
EPOCHS="300"
PATIENCE="80"
WORKERS="8"
DEVICE="0"
PROJECT="${ROOT}/runs/visdrone"
OPTIMIZER="auto"
SEED=""
DONE_MARKER_NAME=".train_complete"
TINY_EVAL_MODE="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --dataset-tag) DATASET_TAG="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --tiny-eval) TINY_EVAL_MODE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
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

TRAIN_EXTRA=()
if [[ -n "${SEED}" ]]; then
  TRAIN_EXTRA+=(--seed "${SEED}")
fi

RUNS=(
  "yolo26n_base_${DATASET_TAG}|ultralytics/cfg/models/26/yolo26.yaml|yolo26n.pt"
  "yolo26n_sfrc2f_${DATASET_TAG}|ultralytics/cfg/models/26/yolo26n-sfrc2f-visdrone.yaml|auto"
  "yolo26n_sfrc3k_${DATASET_TAG}|ultralytics/cfg/models/26/yolo26n-sfrc3k-visdrone.yaml|auto"
  "yolo26n_sfrc3k2_${DATASET_TAG}|ultralytics/cfg/models/26/yolo26n-sfrc3k2-visdrone.yaml|auto"
  "yolo11n_base_${DATASET_TAG}|ultralytics/cfg/models/11/yolo11.yaml|yolo11n.pt"
  "yolo11n_sfrc2f_${DATASET_TAG}|ultralytics/cfg/models/11/yolo11n-sfrc2f-visdrone.yaml|auto"
  "yolov8n_base_${DATASET_TAG}|ultralytics/cfg/models/v8/yolov8.yaml|yolov8n.pt"
  "yolov8n_sfrc2f_${DATASET_TAG}|ultralytics/cfg/models/v8/yolov8n-sfrc2f-visdrone.yaml|auto"
  "yolov10n_base_${DATASET_TAG}|ultralytics/cfg/models/v10/yolov10n.yaml|yolov10n.pt"
  "yolov10n_sfrc2f_${DATASET_TAG}|ultralytics/cfg/models/v10/yolov10n-sfrc2f-visdrone.yaml|auto"
  "yolo12n_base_${DATASET_TAG}|ultralytics/cfg/models/12/yolo12.yaml|yolo12n.pt"
  "yolo12n_sfrc2f_${DATASET_TAG}|ultralytics/cfg/models/12/yolo12n-sfrc2f-visdrone.yaml|auto"
)

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

should_run_tiny_eval() {
  local data_path="$1"
  local mode="$2"
  if [[ "${mode}" == "always" ]]; then
    return 0
  fi
  if [[ "${mode}" == "never" ]]; then
    return 1
  fi
  python3 - "${data_path}" <<'PY'
from pathlib import Path
import sys

from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

data = YAML.load(check_yaml(sys.argv[1]))
names = data["names"]
name_lookup = list(names.values()) if isinstance(names, dict) else list(names)
targets = {"pedestrian", "people", "person"}
print(int(any(name in targets for name in name_lookup)))
PY
}

run_train() {
  local name="$1"
  local model="$2"
  local weights="$3"
  local resolved
  local run_dir
  local completed_epochs
  local has_last
  local has_best
  local is_done
  local last_ckpt
  local best_ckpt
  local done_marker

  resolved="$(resolve_run_dir "${PROJECT}" "${name}")"
  IFS="|" read -r run_dir completed_epochs has_last has_best is_done <<<"${resolved}"
  last_ckpt="${run_dir}/weights/last.pt"
  best_ckpt="${run_dir}/weights/best.pt"
  done_marker="${run_dir}/${DONE_MARKER_NAME}"

  if [[ "${is_done}" == "1" ]]; then
    echo
    echo "=== SKIP ${name} (complete marker found in $(basename "${run_dir}")) ==="
    return 0
  fi

  if [[ "${completed_epochs}" -ge "${EPOCHS}" ]] && [[ "${has_best}" == "1" || "${has_last}" == "1" ]]; then
    mkdir -p "${run_dir}"
    touch "${done_marker}"
    echo
    echo "=== SKIP ${name} (completed ${completed_epochs}/${EPOCHS} epochs in $(basename "${run_dir}")) ==="
    return 0
  fi

  echo
  if [[ "${has_last}" == "1" ]]; then
    echo "=== RESUME ${name} from $(basename "${run_dir}") (${completed_epochs}/${EPOCHS} epochs logged) ==="
    python3 examples/visdrone_sfr/train_sfr_module_bench.py \
      --resume "${last_ckpt}" \
      --imgsz "${IMGSZ}" \
      --batch "${BATCH}" \
      --patience "${PATIENCE}" \
      --workers "${WORKERS}" \
      --device "${DEVICE}" \
      "${TRAIN_EXTRA[@]}"
  else
    run_dir="${PROJECT}/${name}"
    done_marker="${run_dir}/${DONE_MARKER_NAME}"
    echo "=== TRAIN ${name} ==="
    python3 examples/visdrone_sfr/train_sfr_module_bench.py \
      --model "${model}" \
      --weights "${weights}" \
      --data "${DATA}" \
      --imgsz "${IMGSZ}" \
      --batch "${BATCH}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --optimizer "${OPTIMIZER}" \
      --workers "${WORKERS}" \
      --device "${DEVICE}" \
      --project "${PROJECT}" \
      --name "${name}" \
      "${TRAIN_EXTRA[@]}"
  fi

  mkdir -p "${run_dir}"
  touch "${done_marker}"
}

run_eval() {
  local name="$1"
  local resolved
  local run_dir
  local completed_epochs
  local has_last
  local has_best
  local is_done
  local ckpt
  local val_name="${name}_val"
  local json_path="${PROJECT}/${val_name}/predictions.json"
  local tiny_path="${PROJECT}/${val_name}/tiny_human_metrics.json"

  resolved="$(resolve_run_dir "${PROJECT}" "${name}")"
  IFS="|" read -r run_dir completed_epochs has_last has_best is_done <<<"${resolved}"
  ckpt="${run_dir}/weights/best.pt"

  if [[ ! -f "${ckpt}" ]]; then
    echo "Skipping ${name}: missing checkpoint ${ckpt}" >&2
    return 0
  fi

  echo
  echo "=== EVAL ${name} from $(basename "${run_dir}") ==="
  python3 examples/visdrone_sfr/val_psr_yolo26.py \
    --model "${ckpt}" \
    --data "${DATA}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --device "${DEVICE}" \
    --project "${PROJECT}" \
    --name "${val_name}" \
    --save-json

  if [[ "$(should_run_tiny_eval "${DATA}" "${TINY_EVAL_MODE}")" == "1" ]]; then
    python3 examples/visdrone_sfr/tiny_human_eval.py \
      --pred-json "${json_path}" \
      --data "${DATA}" \
      --save "${tiny_path}"
  else
    echo "Skipping tiny-human eval for ${name}: dataset has no pedestrian/people/person classes."
  fi
}

for spec in "${RUNS[@]}"; do
  IFS="|" read -r name model weights <<<"${spec}"

  if [[ "${STAGE}" == "train" || "${STAGE}" == "all" ]]; then
    run_train "${name}" "${model}" "${weights}"
  fi

  if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
    run_eval "${name}"
  fi
done
