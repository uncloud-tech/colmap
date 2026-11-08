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
echo "CPU contract: owned_strip_storage_test"
"${CXX:-c++}" -std=c++17 -O2 -UNDEBUG -Wall -Wextra -Werror -pedantic \
  -Isrc src/colmap/mvs/owned_strip_storage_test.cc \
  src/colmap/mvs/owned_strip_storage.cc -DSTRIP_STORAGE_WRAP_IO \
  -Wl,--wrap=pread,--wrap=pwrite -o "$work/owned_strip_storage_test"
timeout 30s "$work/owned_strip_storage_test"
echo "CPU contract: owned_stereo_admission_test"
"${CXX:-c++}" -std=c++17 -O2 -UNDEBUG -Wall -Wextra -Werror -pedantic \
  -Isrc src/colmap/mvs/owned_stereo_admission_test.cc \
  src/colmap/mvs/owned_stereo_admission.cc \
  src/colmap/mvs/owned_strip_storage.cc -o "$work/owned_stereo_admission_test"
timeout 30s "$work/owned_stereo_admission_test"
echo "CPU contract: single_device_host_plan_test"
# Exercise the actual native adapter and JSON producer without SDK dependencies.
# Only unrelated Image validation/driver entry sections are discarded here;
# the CMake target links the complete native library in SDK builds.
"${CXX:-c++}" -std=c++17 -O2 -UNDEBUG -Wall -Wextra -Werror -pedantic \
  -ffunction-sections -fdata-sections -Isrc \
  src/colmap/mvs/single_device_host_plan_test.cc \
  src/colmap/mvs/multi_gpu_fixed_cohort.cc src/colmap/mvs/multi_gpu_phase_plan.cc \
  src/colmap/mvs/owned_stereo_admission.cc src/colmap/mvs/owned_strip_storage.cc \
  -Wl,--gc-sections -o "$work/single_device_host_plan_test"
timeout 30s "$work/single_device_host_plan_test"
echo "CPU contract: matching_cache_budget_test"
"${CXX:-c++}" -std=c++17 -O2 -UNDEBUG -Wall -Wextra -Werror -pedantic \
  -Isrc src/colmap/controllers/matching_cache_budget_test.cc \
  -o "$work/matching_cache_budget_test"
timeout 30s "$work/matching_cache_budget_test"
for test in test_blackwell_build_selection test_photo_phase_fingerprint test_generated_header_install test_windows_checkout; do
  echo "Resident regression: $test (install plumbing is not a full SDK build)"
  python3 -I "scripts/resident/$test.py"
done
python3 -I scripts/ci/test_lean_mvs_ci.py
