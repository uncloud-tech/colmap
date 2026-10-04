// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/compact_prng.h"
#include "colmap/mvs/gpu_mat.h"

#include <memory>

namespace colmap {
namespace mvs {

using GpuMatPRNGRowView = PrngRowView<curandState, GpuMatView<curandState>>;

// Benchmark seed-0 geometric phase only. Stores the unique perimeter states;
// unlike a full map, this must never be used for photometric initialization.
class GpuMatPRNGEdges {
 public:
  GpuMatPRNGEdges(size_t width, size_t height);
  GpuMatPRNGRowView RowView(int orientation) const;

 private:
  PrngPerimeter perimeter_;
  std::unique_ptr<GpuMat<curandState>> states_;
};

class GpuMatPRNG : public GpuMat<curandState> {
 public:
  // `random_seed < 0` keeps the historical launch-geometry-derived seeding.
  // `random_seed >= 0` seeds each pixel from (random_seed, row, col) so the
  // random stream is independent of launch geometry, sweep tile and device.
  // Benchmark reproducibility only; production must keep the default (-1).
  GpuMatPRNG(const int width, const int height, const int random_seed = -1);

 private:
  void InitRandomState();
};

}  // namespace mvs
}  // namespace colmap
