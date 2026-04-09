#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/duy12i1i7/avis.git"
REPO_DIR="${HOME}/avis"
BRANCH="main"

usage() {
  cat <<'EOF'
Usage:
  bash bootstrap_sfr_multidataset.sh [--repo-url URL] [--repo-dir DIR] [--branch BRANCH] -- [suite args...]

Examples:
  bash bootstrap_sfr_multidataset.sh -- \
    --device 0 \
    --epochs 300 \
    --visdrone-data VisDrone.yaml \
    --aitodv2-data /data/aitodv2.yaml \
    --tinyperson-data /data/tinyperson.yaml

  bash bootstrap_sfr_multidataset.sh --repo-dir /workspace/avis -- \
    --device 0 \
    --epochs 300 \
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

FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --)
      shift
      FORWARD_ARGS=("$@")
      break
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git fetch origin
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}"
git log -1 --oneline

INNER_SCRIPT="${REPO_DIR}/examples/visdrone_sfr/setup_and_run_multidataset.sh"
if [[ ! -f "${INNER_SCRIPT}" ]]; then
  echo "Missing ${INNER_SCRIPT}" >&2
  exit 1
fi

bash "${INNER_SCRIPT}" "${FORWARD_ARGS[@]}"
