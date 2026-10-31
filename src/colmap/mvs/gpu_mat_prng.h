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

  // Retire the legacy pitched initialization map after its last use on stream.
  // The caller must drain stream before staging further allocations, including
  // on exceptions. Do not call during stream capture or use this map afterward.
  // Ownership transfers only after a successful free submission; otherwise the
  // ordinary destructor still owns the pointer.
  void ReleaseAfterInitialization(cudaStream_t stream) {
    if (!array_ptr_) return;
#if CUDART_VERSION >= 11020
    int device = 0;
    int supported = 0;
    CUDA_SAFE_CALL(cudaGetDevice(&device));
    CUDA_SAFE_CALL(cudaDeviceGetAttribute(
        &supported, cudaDevAttrMemoryPoolsSupported, device));
    if (supported) {
      CUDA_SAFE_CALL(cudaFreeAsync(array_ptr_, stream));
      array_ptr_ = nullptr;
      return;
    }
#endif
    // Keep targets without stream-ordered allocator support functional. Unlike
    // the async branch, ordinary cudaFree needs last-use completion first.
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
    CUDA_SAFE_CALL(cudaFree(array_ptr_));
    array_ptr_ = nullptr;
  }
#endif

 private:
  void InitRandomState();
};

}  // namespace mvs
}  // namespace colmap
