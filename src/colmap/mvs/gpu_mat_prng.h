// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/gpu_mat.h"

namespace colmap {
namespace mvs {

class GpuMatPRNG : public GpuMat<curandState> {
 public:
  GpuMatPRNG(const int width, const int height);
#if !defined(COLMAP_HIP_ENABLED)
  GpuMatPRNG(int width, int height, cudaStream_t stream);

  // Allocate without initializing states; call Reinitialize before use.
  struct UninitializedTag {};
  GpuMatPRNG(int width, int height, UninitializedTag);

  void Reinitialize(cudaStream_t stream);
#endif

 private:
  void InitRandomState();
};

}  // namespace mvs
}  // namespace colmap
