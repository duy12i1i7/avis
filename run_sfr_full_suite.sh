#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT}/.venv"
source "${ROOT}/examples/visdrone_sfr/kaggle_resume_utils.sh"

unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE PYTHONEXECUTABLE
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

usage() {
  cat <<'EOF'
Usage:
  bash run_sfr_full_suite.sh \
    --stage train \
    --device 0 \
    --epochs 300 \
    --batch 8 \
    --imgsz 960

Canonical one-shot runner for the rebuilt SFR full family suite across VisDrone and TinyPerson.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "${ROOT}"

kaggle_restore_tree "${ROOT}" "${ROOT}/runs/sfr_full" "sfr_full"
kaggle_restore_tree "${ROOT}" "${ROOT}/runs/sfr_suite" "sfr_suite"

if [[ -d "${VENV_DIR}" && ! -x "${VENV_DIR}/bin/python" ]]; then
  rm -rf "${VENV_DIR}"
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip "setuptools<82" wheel
python -m pip install -e .
python -m pip install pycocotools typeguard

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

bash "${ROOT}/examples/visdrone_sfr/run_sfrfull_dataset_suite.sh" "$@"
kaggle_snapshot_tree "${ROOT}" "runs/sfr_full" "sfr_full"
