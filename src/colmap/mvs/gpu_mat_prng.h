// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/gpu_mat.h"

namespace colmap {
namespace mvs {

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
