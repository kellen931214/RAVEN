#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-}"
python_executable="${PYTHON_EXECUTABLE:-}"
forward=()

while (($#)); do
  case "$1" in
    --repo-root)
      repo_root="$2"
      forward+=("$1" "$2")
      shift 2
      ;;
    --python-executable)
      python_executable="$2"
      forward+=("$1" "$2")
      shift 2
      ;;
    *)
      forward+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$repo_root" ]]; then
  echo "ERROR: --repo-root or REPO_ROOT is required" >&2
  exit 64
fi
if [[ -z "$python_executable" ]]; then
  echo "ERROR: --python-executable or PYTHON_EXECUTABLE is required" >&2
  exit 64
fi
if [[ ! -x "$python_executable" ]]; then
  echo "ERROR: Python executable is not executable: $python_executable" >&2
  exit 66
fi
if [[ ! -f "$repo_root/raven_repro/scripts/color_alignment_waiter.py" ]]; then
  echo "ERROR: waiter implementation missing under repo root: $repo_root" >&2
  exit 66
fi

exec "$python_executable" -u "$repo_root/raven_repro/scripts/color_alignment_waiter.py" "${forward[@]}"
