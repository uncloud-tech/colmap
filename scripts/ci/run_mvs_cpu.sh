#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail
root=$(cd "${1:-$(dirname "$0")/../..}" && pwd)
cd "$root"
contracts=(
  optimized_dispatch_test optimized_photo_forecast_test
  optimized_photo_binding_test optimized_photo_dispatch_test
  optimized_photo_delivery_test optimized_photo_options_test
  optimized_photo_paths_test optimized_photo_prepared_test
  optimized_photo_concurrency_test fusion_dependencies_test
  owned_graph_test owned_graph_coordinator_test native_run_deadline_test
)
# Fail closed before generating or compiling if the intended coverage disappears.
for name in "${contracts[@]}"; do
  if [[ ! -f "src/colmap/mvs/$name.cc" ]] ||
     ! grep -Fq "$name" src/colmap/mvs/CMakeLists.txt; then
    echo "Missing registered CPU contract: $name" >&2
    exit 1
  fi
done
work=$(mktemp -d "${RUNNER_TEMP:-$root}/mvs-cpu.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
export TMPDIR="$work"
include="$work/include/colmap/mvs/generated"
python3 -I scripts/resident/generate_photo_phase_fingerprint.py \
  --repo-root "$root" --output "$include/photo_phase_fingerprint.h"
python3 -I scripts/resident/generate_resident_allocation_fingerprint.py \
  --repo-root "$root" --output "$include/resident_allocation_fingerprint.h" \
  --depfile "$work/resident.d"
git rev-parse HEAD
"${CXX:-c++}" --version
for name in "${contracts[@]}"; do
  echo "CPU contract: $name"
  # Existing aggregate fixtures intentionally omit trailing defaulted fields.
  "${CXX:-c++}" -std=c++17 -O2 -UNDEBUG -Wall -Wextra -Werror -pedantic \
    -Wno-missing-field-initializers \
    -pthread -Isrc -I"$work/include" "src/colmap/mvs/$name.cc" -o "$work/$name"
  timeout 30s "$work/$name"
done
for test in test_photo_phase_fingerprint test_generated_header_install test_windows_checkout; do
  echo "Resident regression: $test (install plumbing is not a full SDK build)"
  python3 -I "scripts/resident/$test.py"
done
python3 -I scripts/ci/test_lean_mvs_ci.py
