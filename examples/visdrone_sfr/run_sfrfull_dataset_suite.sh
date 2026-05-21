#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

STAGE="train"
DEVICE="0"
PROJECT_ROOT="${ROOT}/runs/sfr_full"
EPOCHS="300"
PATIENCE="80"
WORKERS="4"
OPTIMIZER="auto"
SEED=""
COMMON_IMGSZ="960"
COMMON_BATCH="8"
VISDRONE_DATA="VisDrone.yaml"
TINYPERSON_DATA="TinyPerson.yaml"
VISDRONE_IMGSZ=""
VISDRONE_BATCH=""
TINYPERSON_IMGSZ=""
TINYPERSON_BATCH=""
TINY_EVAL_MODE="auto"
PLOTS="1"
AMP_MODE="default"

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
    --tiny-eval) TINY_EVAL_MODE="$2"; shift 2 ;;
    --plots) PLOTS="1"; shift ;;
    --no-plots) PLOTS="0"; shift ;;
    --amp) AMP_MODE="on"; shift ;;
    --no-amp) AMP_MODE="off"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

run_dataset() {
  local tag="$1"
  local data="$2"
  local imgsz="$3"
  local batch="$4"
  local cmd=(
    bash examples/visdrone_sfr/run_sfrfull_family_suite.sh
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
    --tiny-eval "${TINY_EVAL_MODE}"
  )
  if [[ -z "${data}" ]]; then
    return 0
  fi
  if [[ -n "${SEED}" ]]; then
    cmd+=(--seed "${SEED}")
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
  echo "########## SFR FULL DATASET ${tag} ##########"
  "${cmd[@]}"
}

run_dataset "visdrone" "${VISDRONE_DATA}" "${VISDRONE_IMGSZ:-${COMMON_IMGSZ}}" "${VISDRONE_BATCH:-${COMMON_BATCH}}"
run_dataset "tinyperson" "${TINYPERSON_DATA}" "${TINYPERSON_IMGSZ:-${COMMON_IMGSZ}}" "${TINYPERSON_BATCH:-${COMMON_BATCH}}"
