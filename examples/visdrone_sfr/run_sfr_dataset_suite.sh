#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

STAGE="all"
DEVICE="0"
PROJECT_ROOT="${ROOT}/runs/sfr_suite"
EPOCHS="300"
PATIENCE="80"
WORKERS="8"
OPTIMIZER="auto"
SEED=""
VISDRONE_DATA="VisDrone.yaml"
TINYPERSON_DATA=""
COMMON_IMGSZ="960"
COMMON_BATCH="16"
VISDRONE_IMGSZ=""
VISDRONE_BATCH=""
TINYPERSON_IMGSZ=""
TINYPERSON_BATCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --imgsz) COMMON_IMGSZ="$2"; shift 2 ;;
    --batch) COMMON_BATCH="$2"; shift 2 ;;
    --visdrone-data) VISDRONE_DATA="$2"; shift 2 ;;
    --visdrone-imgsz) VISDRONE_IMGSZ="$2"; shift 2 ;;
    --visdrone-batch) VISDRONE_BATCH="$2"; shift 2 ;;
    --tinyperson-data) TINYPERSON_DATA="$2"; shift 2 ;;
    --tinyperson-imgsz) TINYPERSON_IMGSZ="$2"; shift 2 ;;
    --tinyperson-batch) TINYPERSON_BATCH="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

run_dataset() {
  local tag="$1"
  local data="$2"
  local imgsz="$3"
  local batch="$4"
  local cmd=(
    bash examples/visdrone_sfr/run_sfr_full_matrix.sh
    --stage "${STAGE}"
    --data "${data}"
    --dataset-tag "${tag}"
    --imgsz "${imgsz}"
    --batch "${batch}"
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --workers "${WORKERS}"
    --device "${DEVICE}"
    --project "${PROJECT_ROOT}/${tag}"
    --optimizer "${OPTIMIZER}"
    --tiny-eval auto
  )
  if [[ -z "${data}" ]]; then
    return 0
  fi
  if [[ -n "${SEED}" ]]; then
    cmd+=(--seed "${SEED}")
  fi
  echo
  echo "########## DATASET ${tag} ##########"
  "${cmd[@]}"
}

run_dataset "visdrone" "${VISDRONE_DATA}" "${VISDRONE_IMGSZ:-${COMMON_IMGSZ}}" "${VISDRONE_BATCH:-${COMMON_BATCH}}"
run_dataset "tinyperson" "${TINYPERSON_DATA}" "${TINYPERSON_IMGSZ:-${COMMON_IMGSZ}}" "${TINYPERSON_BATCH:-${COMMON_BATCH}}"
