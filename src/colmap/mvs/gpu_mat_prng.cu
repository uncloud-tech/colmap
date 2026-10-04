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

// Seed at ORIGINAL coordinates exactly as the full seed-0 map does. Never
// reseed after a sweep: adjacent orientations share updated corner states.
__global__ void InitEdgeRandomStateKernel(GpuMatView<curandState> output,
                                          PrngPerimeter perimeter) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < perimeter.Size()) {
    const auto p = perimeter.Coordinate(index);
    curand_init(benchmark_seed::PixelSeed(0, p.row, p.col),
                0, 0, &output.GetRef(0, index));
  }
}

}  // namespace

GpuMatPRNGEdges::GpuMatPRNGEdges(size_t width, size_t height)
    : perimeter_{width, height} {
  THROW_CHECK_GT(width, 0);
  THROW_CHECK_GT(height, 0);
  states_ = GpuMat<curandState>::CreateCompact(perimeter_.Size(), 1);
  InitEdgeRandomStateKernel<<<(perimeter_.Size() + 255) / 256, 256>>>(
      states_->View(), perimeter_);
}

GpuMatPRNGRowView GpuMatPRNGEdges::RowView(int orientation) const {
  return {{nullptr, 0, 0, 0, 0}, states_->GetPtr(), perimeter_, orientation};
}

GpuMatPRNG::GpuMatPRNG(const int width, const int height, const int random_seed)
    : GpuMat(width, height) {
  if (random_seed < 0) {
    InitRandomStateKernel<<<gridSize_, blockSize_>>>(View());
  } else {
    InitRandomStateFromSeedKernel<<<gridSize_, blockSize_>>>(View(), random_seed);
  }
}

std::unique_ptr<GpuMat<curandState>> GpuMatPRNG::CreateCompact(
    const int width, const int height, const int random_seed) {
  auto result = GpuMat<curandState>::CreateCompact(width, height);
  // Identical launch geometry/seed/coordinates to the ordinary constructor.
  const dim3 block(kBlockDimX, kBlockDimY, 1);
  const dim3 grid((width - 1) / kBlockDimX + 1,
                  (height - 1) / kBlockDimY + 1, 1);
  if (random_seed < 0) {
    InitRandomStateKernel<<<grid, block>>>(result->View());
  } else {
    InitRandomStateFromSeedKernel<<<grid, block>>>(result->View(), random_seed);
  }
  return result;
}

}  // namespace mvs
}  // namespace colmap
