#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT}/.venv"

STAGE="train"
DEVICE="0"
PROJECT_ROOT="${ROOT}/runs/sfr_suite"
EPOCHS="300"
PATIENCE="80"
WORKERS="4"
OPTIMIZER="auto"
SEED=""
BATCH="8"
IMGSZ="960"
VISDRONE_DATA="VisDrone.yaml"
AITODV2_DATA="AI-TODv2.yaml"
AITODV2_OUTPUT=""
AITODV2_YAML=""
AITODV2_TRAIN_IMAGES=""
AITODV2_TRAIN_JSON=""
AITODV2_VAL_IMAGES=""
AITODV2_VAL_JSON=""
AITODV2_TEST_IMAGES=""
AITODV2_TEST_JSON=""
TINYPERSON_DATA="TinyPerson.yaml"
TINYPERSON_OUTPUT=""
TINYPERSON_YAML=""
TINYPERSON_TRAIN_IMAGES=""
TINYPERSON_TRAIN_JSON=""
TINYPERSON_VAL_IMAGES=""
TINYPERSON_VAL_JSON=""
TINYPERSON_TEST_IMAGES=""
TINYPERSON_TEST_JSON=""

usage() {
  cat <<'EOF'
Usage:
  bash run_sfr_multidataset.sh \
    --stage train \
    --device 0 \
    --epochs 300 \
    --visdrone-data VisDrone.yaml \
    --aitodv2-data AI-TODv2.yaml \
    --tinyperson-data TinyPerson.yaml

Notes:
  - TinyPerson.yaml can auto-download and auto-convert the official release.
  - AI-TODv2.yaml auto-converts from a prepared raw root. Set:
      export AITODV2_RAW_ROOT=/path/to/aitodv2_raw

Or prepare AI-TOD-v2 / TinyPerson from raw COCO-style files:
  bash run_sfr_multidataset.sh \
    --aitodv2-output /data/aitodv2_yolo \
    --aitodv2-train-images /data/AI-TOD-v2/train/images \
    --aitodv2-train-json /data/AI-TOD-v2/train.json \
    --aitodv2-val-images /data/AI-TOD-v2/val/images \
    --aitodv2-val-json /data/AI-TOD-v2/val.json \
    --tinyperson-output /data/tinyperson_yolo \
    --tinyperson-train-images /data/TinyPerson/train/images \
    --tinyperson-train-json /data/TinyPerson/train.json \
    --tinyperson-val-images /data/TinyPerson/val/images \
    --tinyperson-val-json /data/TinyPerson/val.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --optimizer) OPTIMIZER="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --visdrone-data) VISDRONE_DATA="$2"; shift 2 ;;

    --aitodv2-data) AITODV2_DATA="$2"; shift 2 ;;
    --aitodv2-output) AITODV2_OUTPUT="$2"; shift 2 ;;
    --aitodv2-yaml) AITODV2_YAML="$2"; shift 2 ;;
    --aitodv2-train-images) AITODV2_TRAIN_IMAGES="$2"; shift 2 ;;
    --aitodv2-train-json) AITODV2_TRAIN_JSON="$2"; shift 2 ;;
    --aitodv2-val-images) AITODV2_VAL_IMAGES="$2"; shift 2 ;;
    --aitodv2-val-json) AITODV2_VAL_JSON="$2"; shift 2 ;;
    --aitodv2-test-images) AITODV2_TEST_IMAGES="$2"; shift 2 ;;
    --aitodv2-test-json) AITODV2_TEST_JSON="$2"; shift 2 ;;

    --tinyperson-data) TINYPERSON_DATA="$2"; shift 2 ;;
    --tinyperson-output) TINYPERSON_OUTPUT="$2"; shift 2 ;;
    --tinyperson-yaml) TINYPERSON_YAML="$2"; shift 2 ;;
    --tinyperson-train-images) TINYPERSON_TRAIN_IMAGES="$2"; shift 2 ;;
    --tinyperson-train-json) TINYPERSON_TRAIN_JSON="$2"; shift 2 ;;
    --tinyperson-val-images) TINYPERSON_VAL_IMAGES="$2"; shift 2 ;;
    --tinyperson-val-json) TINYPERSON_VAL_JSON="$2"; shift 2 ;;
    --tinyperson-test-images) TINYPERSON_TEST_IMAGES="$2"; shift 2 ;;
    --tinyperson-test-json) TINYPERSON_TEST_JSON="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

prepare_dataset_if_needed() {
  local name="$1"
  local output="$2"
  local yaml_out="$3"
  local train_images="$4"
  local train_json="$5"
  local val_images="$6"
  local val_json="$7"
  local test_images="$8"
  local test_json="$9"

  if [[ -z "${output}" && -z "${train_images}" && -z "${train_json}" && -z "${val_images}" && -z "${val_json}" ]]; then
    return 0
  fi

  if [[ -z "${output}" || -z "${train_images}" || -z "${train_json}" || -z "${val_images}" || -z "${val_json}" ]]; then
    echo "Incomplete raw dataset spec for ${name}. Need output + train-images + train-json + val-images + val-json." >&2
    exit 1
  fi

  local cmd=(
    python examples/visdrone_sfr/prepare_coco_detection_dataset.py
    --name "${name}"
    --output "${output}"
    --train-images "${train_images}"
    --train-json "${train_json}"
    --val-images "${val_images}"
    --val-json "${val_json}"
  )

  if [[ -n "${yaml_out}" ]]; then
    cmd+=(--yaml-path "${yaml_out}")
  fi
  if [[ -n "${test_images}" || -n "${test_json}" ]]; then
    if [[ -z "${test_images}" || -z "${test_json}" ]]; then
      echo "Both --${name}-test-images and --${name}-test-json are required together." >&2
      exit 1
    fi
    cmd+=(--test-images "${test_images}" --test-json "${test_json}")
  fi

  echo
  echo "=== PREPARE ${name} ==="
  "${cmd[@]}"
}

resolve_yaml_path() {
  local name="$1"
  local explicit_yaml="$2"
  local output="$3"
  local explicit_data="$4"

  if [[ -n "${output}" ]]; then
    python - "${name}" "${output}" <<'PY'
from pathlib import Path
import sys

name = sys.argv[1]
output = Path(sys.argv[2]).expanduser().resolve()
print(output.parent / f"{name}.yaml")
PY
    return 0
  fi
  if [[ -n "${explicit_yaml}" ]]; then
    printf '%s\n' "${explicit_yaml}"
    return 0
  fi
  if [[ -n "${explicit_data}" ]]; then
    printf '%s\n' "${explicit_data}"
    return 0
  fi
  printf '%s\n' ""
}

cd "${ROOT}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install pycocotools

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"gpu[{i}]:", torch.cuda.get_device_name(i))
PY

nvidia-smi -L || true

prepare_dataset_if_needed "aitodv2" "${AITODV2_OUTPUT}" "${AITODV2_YAML}" "${AITODV2_TRAIN_IMAGES}" "${AITODV2_TRAIN_JSON}" "${AITODV2_VAL_IMAGES}" "${AITODV2_VAL_JSON}" "${AITODV2_TEST_IMAGES}" "${AITODV2_TEST_JSON}"
prepare_dataset_if_needed "tinyperson" "${TINYPERSON_OUTPUT}" "${TINYPERSON_YAML}" "${TINYPERSON_TRAIN_IMAGES}" "${TINYPERSON_TRAIN_JSON}" "${TINYPERSON_VAL_IMAGES}" "${TINYPERSON_VAL_JSON}" "${TINYPERSON_TEST_IMAGES}" "${TINYPERSON_TEST_JSON}"

AITODV2_DATA="$(resolve_yaml_path "aitodv2" "${AITODV2_YAML}" "${AITODV2_OUTPUT}" "${AITODV2_DATA}")"
TINYPERSON_DATA="$(resolve_yaml_path "tinyperson" "${TINYPERSON_YAML}" "${TINYPERSON_OUTPUT}" "${TINYPERSON_DATA}")"

SUITE_CMD=(
  bash examples/visdrone_sfr/run_sfr_dataset_suite.sh
  --stage "${STAGE}"
  --device "${DEVICE}"
  --project-root "${PROJECT_ROOT}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --workers "${WORKERS}"
  --optimizer "${OPTIMIZER}"
  --batch "${BATCH}"
  --imgsz "${IMGSZ}"
  --visdrone-data "${VISDRONE_DATA}"
  --aitodv2-data "${AITODV2_DATA}"
  --tinyperson-data "${TINYPERSON_DATA}"
)

if [[ -n "${SEED}" ]]; then
  SUITE_CMD+=(--seed "${SEED}")
fi

echo
echo "=== RUN MULTI-DATASET SUITE ==="
printf ' %q' "${SUITE_CMD[@]}"
echo
"${SUITE_CMD[@]}"
