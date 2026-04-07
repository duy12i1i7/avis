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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "${STAGE}" != "train" && "${STAGE}" != "eval" && "${STAGE}" != "all" ]]; then
  echo "--stage must be one of: train, eval, all" >&2
  exit 1
fi

TRAIN_EXTRA=()
if [[ -n "${SEED}" ]]; then
  TRAIN_EXTRA+=(--seed "${SEED}")
fi

RUNS=(
  "yolo26n_base_visdrone|ultralytics/cfg/models/26/yolo26.yaml|yolo26n.pt"
  "yolo26n_sfrc2f_visdrone|ultralytics/cfg/models/26/yolo26n-sfrc2f-visdrone.yaml|auto"
  "yolo26n_sfrc3k_visdrone|ultralytics/cfg/models/26/yolo26n-sfrc3k-visdrone.yaml|auto"
  "yolo26n_sfrc3k2_visdrone|ultralytics/cfg/models/26/yolo26n-sfrc3k2-visdrone.yaml|auto"
  "yolo11n_base_visdrone|ultralytics/cfg/models/11/yolo11.yaml|yolo11n.pt"
  "yolo11n_sfrc2f_visdrone|ultralytics/cfg/models/11/yolo11n-sfrc2f-visdrone.yaml|auto"
  "yolov8n_base_visdrone|ultralytics/cfg/models/v8/yolov8.yaml|yolov8n.pt"
  "yolov8n_sfrc2f_visdrone|ultralytics/cfg/models/v8/yolov8n-sfrc2f-visdrone.yaml|auto"
  "yolov10n_base_visdrone|ultralytics/cfg/models/v10/yolov10n.yaml|yolov10n.pt"
  "yolov10n_sfrc2f_visdrone|ultralytics/cfg/models/v10/yolov10n-sfrc2f-visdrone.yaml|auto"
  "yolo12n_base_visdrone|ultralytics/cfg/models/12/yolo12.yaml|yolo12n.pt"
  "yolo12n_sfrc2f_visdrone|ultralytics/cfg/models/12/yolo12n-sfrc2f-visdrone.yaml|auto"
)

run_train() {
  local name="$1"
  local model="$2"
  local weights="$3"
  local run_dir="${PROJECT}/${name}"
  local last_ckpt="${run_dir}/weights/last.pt"
  local best_ckpt="${run_dir}/weights/best.pt"
  local results_csv="${run_dir}/results.csv"
  local done_marker="${run_dir}/${DONE_MARKER_NAME}"
  local completed_epochs="0"

  if [[ -f "${results_csv}" ]]; then
    completed_epochs="$(python3 - "${results_csv}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
try:
    lines = path.read_text().strip().splitlines()
except FileNotFoundError:
    print(0)
    raise SystemExit

print(max(len(lines) - 1, 0))
PY
)"
  fi

  if [[ -f "${done_marker}" ]]; then
    echo
    echo "=== SKIP ${name} (complete marker found) ==="
    return 0
  fi

  if [[ "${completed_epochs}" -ge "${EPOCHS}" ]] && [[ -f "${best_ckpt}" || -f "${last_ckpt}" ]]; then
    mkdir -p "${run_dir}"
    touch "${done_marker}"
    echo
    echo "=== SKIP ${name} (completed ${completed_epochs}/${EPOCHS} epochs) ==="
    return 0
  fi

  echo
  if [[ -f "${last_ckpt}" ]]; then
    echo "=== RESUME ${name} (${completed_epochs}/${EPOCHS} epochs logged) ==="
    python3 examples/visdrone_sfr/train_sfr_module_bench.py \
      --resume "${last_ckpt}" \
      --imgsz "${IMGSZ}" \
      --batch "${BATCH}" \
      --patience "${PATIENCE}" \
      --workers "${WORKERS}" \
      --device "${DEVICE}" \
      "${TRAIN_EXTRA[@]}"
  else
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
  local ckpt="${PROJECT}/${name}/weights/best.pt"
  local val_name="${name}_val"
  local json_path="${PROJECT}/${val_name}/predictions.json"
  local tiny_path="${PROJECT}/${val_name}/tiny_human_metrics.json"

  if [[ ! -f "${ckpt}" ]]; then
    echo "Skipping ${name}: missing checkpoint ${ckpt}" >&2
    return 0
  fi

  echo
  echo "=== EVAL ${name} ==="
  python3 examples/visdrone_sfr/val_psr_yolo26.py \
    --model "${ckpt}" \
    --data "${DATA}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --device "${DEVICE}" \
    --project "${PROJECT}" \
    --name "${val_name}" \
    --save-json

  python3 examples/visdrone_sfr/tiny_human_eval.py \
    --pred-json "${json_path}" \
    --data "${DATA}" \
    --save "${tiny_path}"
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
