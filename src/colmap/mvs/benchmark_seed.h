// SPDX-License-Identifier: BSD-3-Clause

// Benchmark-only deterministic seeding for patch-match stereo.
//
// The stock seeding derives each pixel's cuRAND state from the launch geometry
// (blockIdx/blockDim/threadIdx). That makes a pixel's random stream depend on the
// kernel launch configuration, which changes with the sweep tile and, on the
// experimental auto path, with the device's SM count. Reproducibility across
// machines therefore cannot be guaranteed with the stock behaviour.
//
// This helper derives the seed from the PIXEL COORDINATE (and an explicit user
// seed) instead, so the random stream of a pixel is a pure function of
// (seed, row, col) independent of tile, launch geometry, device and MPS state.
//
// Scope: benchmark reproducibility only. Production runs must leave
// PatchMatchOptions::random_seed at its default (-1), which preserves the
// historical seeding exactly.
#pragma once

#include <cstdint>

namespace colmap {
namespace mvs {
#if defined(__CUDACC__) || defined(__HIPCC__)
#define COLMAP_BENCHMARK_SEED_HD __host__ __device__
#else
#define COLMAP_BENCHMARK_SEED_HD
#endif

namespace benchmark_seed {

// splitmix64 finalizer: cheap, well-mixed, and identical on host and device.
COLMAP_BENCHMARK_SEED_HD inline uint64_t Mix(const uint64_t value) {
  uint64_t x = value + 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Seed for one pixel. `row`/`col` are the coordinates in the reference image
// (after the current 90-degree rotation, as the caller indexes the state map).
COLMAP_BENCHMARK_SEED_HD inline uint64_t PixelSeed(const int64_t seed,
                                                  const int64_t row,
                                                  const int64_t col) {
  const uint64_t coordinate =
      (static_cast<uint64_t>(static_cast<uint32_t>(col)) << 32) ^
      static_cast<uint64_t>(static_cast<uint32_t>(row));
  return Mix(Mix(static_cast<uint64_t>(seed)) ^ coordinate);
}

}  // namespace benchmark_seed
}  // namespace mvs
}  // namespace colmap

#undef COLMAP_BENCHMARK_SEED_HD
