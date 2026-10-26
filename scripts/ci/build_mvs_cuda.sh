#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Native compilation on a CPU runner; deliberately never invokes GPU executables.
set -euo pipefail
cd "$(dirname "$0")/../.."
git rev-parse HEAD
git status --short
cmake --version
c++ --version
nvcc --version
cmake -S . -B build-mvs-ci -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES='86;89' \
  -DCUDA_ENABLED=ON -DMVS_ENABLED=ON -DTESTS_ENABLED=ON \
  -DGUI_ENABLED=OFF -DONNX_ENABLED=OFF -DOPENGL_ENABLED=OFF
# Explicit CUDA library and .cu test targets make a CPU fallback a hard error.
targets=(
  colmap_main colmap_mvs_cuda
  colmap_mvs_gpu_mat_test colmap_mvs_owned_prng_test
  colmap_mvs_owned_input_bank_cuda_test
  colmap_mvs_owned_distinct_source_cache_trace_test
  colmap_mvs_owned_distinct_source_cache_test
  colmap_mvs_owned_host_lookahead_test colmap_mvs_owned_normal_lookahead_test
)
# S22 is optional: never invent a target absent from this revision's CMake graph.
if [[ -f src/colmap/mvs/owned_normal_rotation_test.cu ||
      -f src/colmap/mvs/owned_normal_rotation_test.cc ]] &&
   grep -Eq 'NAME[[:space:]]+owned_normal_rotation_test([[:space:]]|$)' src/colmap/mvs/CMakeLists.txt; then
  targets+=(colmap_mvs_owned_normal_rotation_test)
fi
printf 'Native compile/link target: %s\n' "${targets[@]}"
cmake --build build-mvs-ci --parallel 8 --target "${targets[@]}"
# Record the built CLI bytes without trying to initialize CUDA on a CPU runner.
sha256sum build-mvs-ci/src/colmap/exe/colmap \
  build-mvs-ci/src/colmap/mvs/libcolmap_mvs_cuda.a
for target in "${targets[@]:2}"; do
  sha256sum "build-mvs-ci/src/colmap/mvs/${target#colmap_mvs_}"
done
