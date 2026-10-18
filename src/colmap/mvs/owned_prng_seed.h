// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define COLMAP_OWNED_PRNG_HD __host__ __device__
#else
#define COLMAP_OWNED_PRNG_HD
#endif

namespace colmap {
namespace mvs {
namespace owned {
namespace prng_seed {

// Coordinate seeding belongs to the explicitly selected owned backend.
// The ordinary backend retains its historical launch-index initialization.
COLMAP_OWNED_PRNG_HD inline uint64_t Mix(const uint64_t value) {
  uint64_t x = value + 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Coordinates are those of the current (possibly rotated) reference map.
COLMAP_OWNED_PRNG_HD inline uint64_t PixelSeed(const int64_t seed,
                                             const int64_t row,
                                             const int64_t col) {
  const uint64_t coordinate =
      (static_cast<uint64_t>(static_cast<uint32_t>(col)) << 32) ^
      static_cast<uint64_t>(static_cast<uint32_t>(row));
  return Mix(Mix(static_cast<uint64_t>(seed)) ^ coordinate);
}

}  // namespace prng_seed
}  // namespace owned
}  // namespace mvs
}  // namespace colmap

#undef COLMAP_OWNED_PRNG_HD
