#!/usr/bin/env bash
set -euo pipefail

KERNEL_ID=""
OUTDIR=""

usage() {
  cat <<'EOF'
Usage:
  bash kaggle_pull_kernel_output.sh \
    --kernel-id <username/kernel-slug> \
    [--outdir ./kaggle-output/<kernel-slug>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${KERNEL_ID}" ]]; then
  echo "--kernel-id is required" >&2
  exit 1
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: python3 -m pip install kaggle" >&2
  exit 1
fi

if [[ -z "${OUTDIR}" ]]; then
  OUTDIR="./kaggle-output/${KERNEL_ID##*/}"
fi

mkdir -p "${OUTDIR}"
kaggle kernels output "${KERNEL_ID}" -p "${OUTDIR}"
echo "Downloaded outputs to ${OUTDIR}"
