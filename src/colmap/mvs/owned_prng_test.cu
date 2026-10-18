// SPDX-License-Identifier: BSD-3-Clause

#if defined(__CUDACC__) || defined(__HIPCC__)
#define BOOST_PP_VARIADICS 0
#endif

#include "colmap/mvs/owned_prng.h"
#include "colmap/mvs/owned_prng_seed.h"

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

// Independent legacy launch-index oracle; seeded cases also exercise current
// coordinate mapping against a separate launch, not opaque state padding.
__global__ void ReferenceSeedKernel(GpuMatView<curandState> output, int seed) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t block = blockIdx.y * gridDim.x + blockIdx.x;
  const size_t id = block * blockDim.y * blockDim.x +
                    threadIdx.y * blockDim.x + threadIdx.x;
  if (col < output.GetWidth() && row < output.GetHeight()) {
    curand_init(seed < 0 ? id : owned::prng_seed::PixelSeed(seed, row, col),
                0, 0, &output.GetRef(row, col));
  }
}

class ReferenceSeedMap : public GpuMatPRNG {
 public:
  using GpuMatPRNG::GpuMatPRNG;
  void Reference(int seed) {
    ReferenceSeedKernel<<<gridSize_, blockSize_>>>(View(), seed);
  }
  void SetGeometry(unsigned int x, unsigned int y) {
    blockSize_ = dim3(x, y);
    gridSize_ = dim3((GetWidth() + x - 1) / x, (GetHeight() + y - 1) / y);
  }
};

std::vector<float> Draws(GpuMatPRNG& states) {
  GpuMat<float> draws(states.GetWidth(), states.GetHeight());
  draws.FillWithRandomNumbers(0.0f, 1.0f, states);
  std::vector<float> result(states.GetWidth() * states.GetHeight());
  draws.CopyToHost(result.data(), states.GetWidth() * sizeof(float));
  return result;
}

TEST(OwnedPrng, LegacyZeroAndNonzeroSeeds) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  for (int seed : {-1, 0, 7, 2147483647}) {
    auto actual = owned::MakeOwnedPrng(37, 19, stream, seed);
    ReferenceSeedMap reference(37, 19);
    reference.Reference(seed);
    EXPECT_EQ(Draws(*actual), Draws(reference));
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(OwnedPrng, StreamGeometryAndRetainedRotationReinitialize) {
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  for (int seed : {-1, 0, 7}) {
    ReferenceSeedMap actual(37, 19);
    ReferenceSeedMap alternate(19, 37, GpuMatPRNG::UninitializedTag{});
    for (int turn = 0; turn < 4; ++turn) {
      ReferenceSeedMap& target = turn % 2 == 0 ? alternate : actual;
      target.SetGeometry(turn % 2 == 0 ? 8 : 16, 4);
      owned::ReinitializeOwnedPrng(target, stream, seed);
      ReferenceSeedMap reference(target.GetWidth(), target.GetHeight());
      if (seed < 0) reference.SetGeometry(turn % 2 == 0 ? 8 : 16, 4);
      reference.Reference(seed);
      EXPECT_EQ(Draws(target), Draws(reference));
    }
  }
  for (int seed : {-1, 7}) {
    auto even = owned::MakeOwnedPrng(37, 19, stream, seed);
    auto odd = std::make_unique<GpuMatPRNG>(
        19, 37, GpuMatPRNG::UninitializedTag{});
    auto reference = owned::MakeOwnedPrng(37, 19, stream, seed);
    GpuMatPRNG* current = even.get();
    GpuMatPRNG* alternate = odd.get();
    for (int turn = 0; turn < 4; ++turn) {
      owned::ReinitializeOwnedPrng(*alternate, stream, seed);
      current->Rotate(alternate, stream);
      std::swap(current, alternate);
      auto rotated = owned::MakeOwnedPrng(
          reference->GetHeight(), reference->GetWidth(), stream, seed);
      reference->Rotate(rotated.get());
      reference.swap(rotated);
      EXPECT_EQ(Draws(*current), Draws(*reference));
    }
  }
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
