#!/usr/bin/env bash
set -euo pipefail

is_kaggle_runtime() {
  [[ -d /kaggle/input && -d /kaggle/working ]]
}

dir_has_contents() {
  local dir="$1"
  [[ -d "${dir}" ]] && find "${dir}" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

find_kaggle_restore_source() {
  local logical_name="$1"
  python3 - "${logical_name}" <<'PY'
from pathlib import Path
import sys

logical = sys.argv[1]
root = Path("/kaggle/input")
if not root.exists():
    raise SystemExit

dir_candidates = []
archive_candidates = []

for dataset_dir in root.iterdir():
    if not dataset_dir.is_dir():
        continue
    for rel in (Path("runs") / logical, Path(logical)):
        candidate = dataset_dir / rel
        if candidate.is_dir():
            dir_candidates.append(candidate)
    for name in (
        f"{logical}_resume.tgz",
        f"{logical}_resume.tar.gz",
        f"{logical}_artifacts.tgz",
        f"{logical}_artifacts.tar.gz",
        f"{logical}.tgz",
        f"{logical}.tar.gz",
    ):
        candidate = dataset_dir / name
        if candidate.is_file():
            archive_candidates.append(candidate)

def sort_key(path: Path):
    stat = path.stat()
    return (stat.st_mtime, stat.st_size, str(path))

if dir_candidates:
    print(max(dir_candidates, key=sort_key))
elif archive_candidates:
    print(max(archive_candidates, key=sort_key))
PY
}

kaggle_restore_tree() {
  local repo_root="$1"
  local target_dir="$2"
  local logical_name="$3"
  local source=""

  is_kaggle_runtime || return 0

  if dir_has_contents "${target_dir}"; then
    return 0
  fi

  source="$(find_kaggle_restore_source "${logical_name}" || true)"
  if [[ -z "${source}" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "${target_dir}")"
  if [[ -d "${source}" ]]; then
    echo "=== KAGGLE RESTORE ${logical_name}: ${source} -> ${target_dir} ==="
    rm -rf "${target_dir}"
    mkdir -p "${target_dir}"
    cp -a "${source}/." "${target_dir}/"
    return 0
  fi

  if [[ -f "${source}" ]]; then
    echo "=== KAGGLE RESTORE ${logical_name}: unpack ${source} into ${repo_root} ==="
    tar -xzf "${source}" -C "${repo_root}"
    return 0
  fi
}

kaggle_snapshot_tree() {
  local repo_root="$1"
  local rel_path="$2"
  local logical_name="$3"
  local abs_path="${repo_root}/${rel_path}"
  local out_dir="/kaggle/working/resume_state"
  local tar_path="/kaggle/working/${logical_name}_resume.tgz"

  is_kaggle_runtime || return 0
  [[ -d "${abs_path}" ]] || return 0

  mkdir -p "${out_dir}"
  rm -rf "${out_dir}/${logical_name}"
  mkdir -p "${out_dir}/${logical_name}"
  cp -a "${abs_path}/." "${out_dir}/${logical_name}/"
  tar -czf "${tar_path}" -C "${repo_root}" "${rel_path}"
  echo "=== KAGGLE SNAPSHOT ${logical_name}: ${tar_path} ==="
}
