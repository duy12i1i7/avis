#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE PYTHONEXECUTABLE
export PYTHONNOUSERSITE=1
for candidate in "${ROOT}/.venv/bin/python" "${ROOT}/.eval-venv/bin/python"; do
  if [[ -x "${candidate}" ]]; then
    exec "${candidate}" "${ROOT}/examples/visdrone_sfr/compare_sfrfull_vs_baseline.py" "$@"
  fi
done

exec python3 "${ROOT}/examples/visdrone_sfr/compare_sfrfull_vs_baseline.py" "$@"
