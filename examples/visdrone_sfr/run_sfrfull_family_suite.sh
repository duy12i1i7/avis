#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

STAGE="train"
DATA="VisDrone.yaml"
DATASET_TAG="visdrone"
IMGSZ="960"
BATCH="8"
EPOCHS="300"
PATIENCE="80"
WORKERS="4"
DEVICE="0"
PROJECT="${ROOT}/runs/sfr_full"
OPTIMIZER="auto"
SEED=""
LR0=""
TINY_EVAL_MODE="auto"
PLOTS="1"
AMP_MODE="default"

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
    --lr0) LR0="$2"; shift 2 ;;
    --tiny-eval) TINY_EVAL_MODE="$2"; shift 2 ;;
    --plots) PLOTS="1"; shift ;;
    --no-plots) PLOTS="0"; shift ;;
    --amp) AMP_MODE="on"; shift ;;
    --no-amp) AMP_MODE="off"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

RUNS=(
  "yolo26n_sfrfull_${DATASET_TAG}|ultralytics/cfg/models/26/yolo26n-sfrfull-visdrone.yaml|yolo26n.pt"
  "yolo11n_sfrfull_${DATASET_TAG}|ultralytics/cfg/models/11/yolo11n-sfrfull-visdrone.yaml|yolo11n.pt"
  "yolo12n_sfrfull_${DATASET_TAG}|ultralytics/cfg/models/12/yolo12n-sfrfull-visdrone.yaml|yolo12n.pt"
  "yolov8n_sfrfull_${DATASET_TAG}|ultralytics/cfg/models/v8/yolov8n-sfrfull-visdrone.yaml|yolov8n.pt"
  "yolov10n_sfrfull_${DATASET_TAG}|ultralytics/cfg/models/v10/yolov10n-sfrfull-visdrone.yaml|yolov10n.pt"
)

for spec in "${RUNS[@]}"; do
  IFS="|" read -r name model weights <<<"${spec}"
  cmd=(
    bash examples/visdrone_sfr/run_sfr_full_rebuild.sh
    --stage "${STAGE}"
    --model "${model}"
    --weights "${weights}"
    --data "${DATA}"
    --name "${name}"
    --project "${PROJECT}"
    --imgsz "${IMGSZ}"
    --batch "${BATCH}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --workers "${WORKERS}"
    --device "${DEVICE}"
    --optimizer "${OPTIMIZER}"
    --tiny-eval "${TINY_EVAL_MODE}"
  )
  if [[ -n "${SEED}" ]]; then
    cmd+=(--seed "${SEED}")
  fi
  if [[ -n "${LR0}" ]]; then
    cmd+=(--lr0 "${LR0}")
  fi
  if [[ "${PLOTS}" == "1" ]]; then
    cmd+=(--plots)
  else
    cmd+=(--no-plots)
  fi
  if [[ "${AMP_MODE}" == "on" ]]; then
    cmd+=(--amp)
  elif [[ "${AMP_MODE}" == "off" ]]; then
    cmd+=(--no-amp)
  fi
  echo
  echo "=== SFR FULL ${name} ==="
  "${cmd[@]}"
done
