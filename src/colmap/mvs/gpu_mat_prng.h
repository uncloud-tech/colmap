// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/gpu_mat.h"

namespace colmap {
namespace mvs {

class GpuMatPRNG : public GpuMat<curandState> {
 public:
  GpuMatPRNG(const int width, const int height);
  static std::unique_ptr<GpuMat<curandState>> CreateCompact(
      int width, int height);

 private:
  void InitRandomState();
};

}  // namespace mvs
}  // namespace colmap
