// SPDX-License-Identifier: BSD-3-Clause

// Runtime sweep geometry helpers. No CUDA runtime dependency.
#pragma once

#include <algorithm>
#include <cstdint>

namespace colmap {
namespace mvs {
namespace sweep_tile {

constexpr int kMaxTile = 32;
constexpr int kMinTile = 8;
constexpr int kSharedColumns = 3 * kMaxTile;
constexpr int kBlocksPerSm = 2;

constexpr bool IsValidOption(const int tile) {
  return tile == 0 || tile == 8 || tile == 16 || tile == 32;
}

// 0 requests experimental auto selection. Fixed tiles must cover the halo.
// Unknown SM count falls back to 32. Returns 0 for invalid input.
inline int Select(const int width,
                  const int height,
                  const int num_sms,
                  const int radius,
                  const int requested) {
  if (width <= 0 || height <= 0 || radius < 1 || radius > kMaxTile ||
      !IsValidOption(requested)) {
    return 0;
  }
  if (requested != 0) {
    return requested >= radius ? requested : 0;
  }
  int tile = kMaxTile;
  if (num_sms > 0) {
    // Optimistic, orientation-invariant coverage prior, NOT a calibrated cost
    // model. max(width,height) counts columns in the better-covered orientation;
    // the serial row walk follows the OTHER dimension. K=2 is unvalidated.
    const int columns = std::max(width, height);
    const int64_t target = int64_t{kBlocksPerSm} * num_sms;
    while (tile > kMinTile && 1 + (columns - 1) / tile < target) {
      tile /= 2;
    }
  }
  while (tile < radius) {
    tile *= 2;
  }
  return tile;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
#define COLMAP_SWEEP_HD __host__ __device__
#else
#define COLMAP_SWEEP_HD
#endif

// Each specialized kernel owns a tightly packed, compile-time-sized layout.
// Production producers and consumers use this same type for all row strides.
template <int kTile>
struct RefLayout {
  static_assert(kTile == 8 || kTile == 16 || kTile == 32,
                "Unsupported sweep tile");
  static constexpr int kNumColumns = 3 * kTile;

  COLMAP_SWEEP_HD static constexpr int Index(const int row, const int column) {
    return row * kNumColumns + column;
  }

  COLMAP_SWEEP_HD static constexpr int PatchColumn(const int radius,
                                                  const int thread) {
    return kTile - radius + thread;
  }
};

// Legacy padded layout retained for the host regression/mutation tests only.
COLMAP_SWEEP_HD constexpr int RefIndex(const int row, const int column) {
  return row * kSharedColumns + column;
}

COLMAP_SWEEP_HD constexpr int PatchColumn(const int tile,
                                         const int radius,
                                         const int thread) {
  return tile - radius + thread;
}

#undef COLMAP_SWEEP_HD

}  // namespace sweep_tile
}  // namespace mvs
}  // namespace colmap
