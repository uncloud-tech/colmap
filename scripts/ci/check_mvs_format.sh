#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Check changed files, not unrelated formatting debt in the experimental base.
set -euo pipefail
cd "$(dirname "$0")/../.."
base=${1:-HEAD^}
if ! git rev-parse --verify --quiet "$base^{commit}" >/dev/null; then
  echo "Invalid format base: $base" >&2
  exit 1
fi
[[ $(ruff --version) == 'ruff 0.15.20' ]]
[[ $(clang-format --version) == *'version 22.1.5'* ]]
cpp=()
python=()
files=$(mktemp "${RUNNER_TEMP:-${TMPDIR:-$PWD}}/mvs-format.XXXXXX")
trap 'rm -f -- "$files"' EXIT
if git diff --name-only -z --diff-filter=ACMR "$base" HEAD > "$files"; then
  :
else
  status=$?
  echo "Cannot discover files to format" >&2
  exit "$status"
fi
# Git validates the base above. A file list preserves whitespace and cannot
# become shell source. Only existing files are selected (deletions excluded).
while IFS= read -r -d '' file; do
  case "$file" in
    *.py) python+=("$file") ;;
    src/colmap/*|src/glomap/*|src/pycolmap/*|benchmark/*)
      case "$file" in
        *.cc|*.h|*.hpp|*.cpp|*.cu) cpp+=("$file") ;;
      esac ;;
  esac
done < "$files"
printf 'Format base: %s; C++ files: %s; Python files: %s\n' \
  "$base" "${#cpp[@]}" "${#python[@]}"
if ((${#cpp[@]})); then
  clang-format --dry-run --Werror "${cpp[@]}"
fi
if ((${#python[@]})); then
  ruff format --check --config ruff.toml "${python[@]}"
  ruff check --config ruff.toml "${python[@]}"
fi
