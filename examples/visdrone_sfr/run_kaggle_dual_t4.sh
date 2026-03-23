#!/bin/bash
set -euo pipefail

# Example:
# bash examples/visdrone_sfr/run_kaggle_dual_t4.sh \
#   --data VisDrone.yaml \
#   --imgsz 960 \
#   --batch 16

DATA="VisDrone.yaml"
IMGSZ="960"
BATCH="16"
EPOCHS="500"
WORKERS="4"
DEVICE="0,1"
PROJECT="/kaggle/working/runs/visdrone"
NAME="yolo26_spd_visdrone_attack_kaggle"
MODEL=""
WEIGHTS="auto"
TEACHER_MODEL=""
TEACHER_WEIGHTS=""
RESUME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --teacher-model) TEACHER_MODEL="$2"; shift 2 ;;
    --teacher-weights) TEACHER_WEIGHTS="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

CMD=(
  python3 examples/visdrone_sfr/train_psr_yolo26.py
  --data "${DATA}"
  --imgsz "${IMGSZ}"
  --batch "${BATCH}"
  --epochs "${EPOCHS}"
  --workers "${WORKERS}"
  --device "${DEVICE}"
  --project "${PROJECT}"
  --name "${NAME}"
  --weights "${WEIGHTS}"
)

if [[ -n "${MODEL}" ]]; then
  CMD+=(--model "${MODEL}")
fi

if [[ -n "${TEACHER_MODEL}" ]]; then
  CMD+=(--teacher-model "${TEACHER_MODEL}")
fi

if [[ -n "${TEACHER_WEIGHTS}" ]]; then
  CMD+=(--teacher-weights "${TEACHER_WEIGHTS}")
fi

if [[ -n "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi

"${CMD[@]}"
