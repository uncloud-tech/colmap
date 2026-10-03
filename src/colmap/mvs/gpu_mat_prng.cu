// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/gpu_mat_prng.h"

#include "colmap/mvs/benchmark_seed.h"

namespace colmap {
namespace mvs {
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

// Benchmark-only: pixel-coordinate seeding, independent of the launch geometry.
__global__ void InitRandomStateFromSeedKernel(GpuMatView<curandState> output,
                                              const int64_t random_seed) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < output.GetWidth() && row < output.GetHeight()) {
    curand_init(benchmark_seed::PixelSeed(
                    random_seed, static_cast<int64_t>(row), static_cast<int64_t>(col)),
                0, 0, &output.GetRef(row, col));
  }
}

}  // namespace

GpuMatPRNG::GpuMatPRNG(const int width, const int height, const int random_seed)
    : GpuMat(width, height) {
  if (random_seed < 0) {
    InitRandomStateKernel<<<gridSize_, blockSize_>>>(View());
  } else {
    InitRandomStateFromSeedKernel<<<gridSize_, blockSize_>>>(View(), random_seed);
  }
}

}  // namespace mvs
}  // namespace colmap
