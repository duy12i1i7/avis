#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/bootstrap_sfr_multidataset.sh" --repo-dir "${ROOT}" "$@"
