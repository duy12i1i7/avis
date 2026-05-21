#!/usr/bin/env bash
set -euo pipefail

KERNEL_ID=""
INTERVAL="60"

usage() {
  cat <<'EOF'
Usage:
  bash kaggle_watch_kernel.sh --kernel-id <username/kernel-slug> [--interval 60]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
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

while true; do
  OUTPUT="$(kaggle kernels status "${KERNEL_ID}" 2>&1 || true)"
  printf '%s\n' "${OUTPUT}"
  if grep -Eiq 'complete|completed|success' <<<"${OUTPUT}"; then
    exit 0
  fi
  if grep -Eiq 'error|failed|cancelled|canceled' <<<"${OUTPUT}"; then
    exit 1
  fi
  sleep "${INTERVAL}"
done
