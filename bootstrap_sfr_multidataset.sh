#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/duy12i1i7/avis.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
BRANCH="main"

usage() {
  cat <<'EOF'
Usage:
  bash bootstrap_sfr_multidataset.sh [bootstrap options] [run_sfr_multidataset args...]

Bootstrap options:
  --repo-url URL     Git repository to clone if repo-dir does not exist
  --repo-dir DIR     Target repository directory
  --branch BRANCH    Branch to checkout and pull

Examples:
  bash bootstrap_sfr_multidataset.sh \
    --repo-dir /workspace/avis \
    -- \
    --stage train \
    --device 0 \
    --epochs 300 \
    --visdrone-data VisDrone.yaml \
    --tinyperson-data /data/tinyperson.yaml

  bash bootstrap_sfr_multidataset.sh --repo-dir /workspace/avis \
    --stage train \
    --device 0 \
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
      FORWARD_ARGS+=("$@")
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

exec bash "${REPO_DIR}/run_sfr_multidataset.sh" "${FORWARD_ARGS[@]}"
