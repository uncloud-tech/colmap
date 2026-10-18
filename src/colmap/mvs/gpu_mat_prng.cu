// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/gpu_mat_prng.h"
#include "colmap/mvs/stock_prng_state.h"

#include <type_traits>

namespace colmap {
namespace mvs {

#if defined(COLMAP_CUDA_ENABLED) && defined(COLMAP_MVS_ENABLED) && \
    !defined(COLMAP_HIP_ENABLED)
StockPrngStateLayout GetStockPrngStateLayout() {
  static_assert(std::is_base_of<GpuMat<curandState>, GpuMatPRNG>::value);
  return {sizeof(curandState), alignof(curandState)};
}
#endif

namespace {

__global__ void InitRandomStateKernel(GpuMatView<curandState> output) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  const size_t uniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
  const size_t id = uniqueBlockIndex * blockDim.y * blockDim.x +
                    threadIdx.y * blockDim.x + threadIdx.x;

  // Each thread gets same seed, a different sequence number, no offset.
  if (col < output.GetWidth() && row < output.GetHeight()) {
    curand_init(id, 0, 0, &output.GetRef(row, col));
  }
}

}  // namespace

GpuMatPRNG::GpuMatPRNG(const int width, const int height)
    : GpuMat(width, height) {
  InitRandomStateKernel<<<gridSize_, blockSize_>>>(View());
}

#if !defined(COLMAP_HIP_ENABLED)
GpuMatPRNG::GpuMatPRNG(const int width,
                     const int height,
                     const cudaStream_t stream)
    : GpuMat(width, height) {
  Reinitialize(stream);
}

GpuMatPRNG::GpuMatPRNG(const int width,
                     const int height,
                     UninitializedTag)
    : GpuMat(width, height) {}

void GpuMatPRNG::Reinitialize(const cudaStream_t stream) {
  owned::ScopedStreamCompletion completion(stream);
  InitRandomStateKernel<<<gridSize_, blockSize_, 0, stream>>>(View());
  completion.Wait();
}
#endif  // !defined(COLMAP_HIP_ENABLED)

}  // namespace mvs
}  // namespace colmap
