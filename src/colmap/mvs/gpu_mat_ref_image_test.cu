// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/gpu_mat_ref_image.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

namespace colmap::mvs {
namespace {

// Exact float32 encodings, uploaded before either kernel runs. In particular,
// sigma_color is not a double literal rounded by the host compiler.
// Columns: sigma_spatial, sigma_color, row, col, center, reference.
constexpr std::array<std::array<uint32_t, 6>, 8> kInputs = {{
    {0x40a00000, 0x3e4ccccd, 0, 0x40000000, 0, 0x3e088889},  // sample 62
    {0x40a00000, 0x3e4ccccd, 0x3f800000, 0x3f800000, 0, 0x3e5cdcdd},  // sample
                                                                      // 72
    {0x40a00000, 0x3e4ccccd, 0, 0, 0, 0},  // unit weight
    {0x40a00000, 0x3e4ccccd, 0, 0xc0000000, 0x3e088889, 0},
    {0x40a00000, 0x3e4ccccd, 0, 0, 0x80000000, 0},  // signed zero
    {0x40a00000, 0x3e4ccccd, 0, 0, 0, 0x00000001},  // FTZ input
    {0x40a00000, 0x3e4ccccd, 0, 0, 0, 0x00800000},  // squared underflow
    {0x40a00000, 0x3e4ccccd, 0, 0, 0, 0x41000000},  // exp underflow
}};

constexpr int kWindowRadius = 5;
constexpr int kWindowSize = 2 * kWindowRadius + 1;
constexpr int kPatchArea = kWindowSize * kWindowSize;
constexpr size_t kNumWeights = kInputs.size() * kPatchArea;

// As in ComputeInitialCost/Sweep's NCC window, each thread computes an entire
// 11x11 patch. Unrolling exposes constant spatial offsets to Compute, while
// sigmas and image colors still come from device memory. A scalar call with
// runtime offsets can compile identically after reverting the production fix
// and therefore does not reproduce the production contraction regression.
// Keep the actual helper and its compiler context separate from the oracle.
__global__ void ComputeWeights(GpuMatView<uint32_t> inputs,
                               GpuMatView<uint32_t> weights) {
  const size_t i = threadIdx.x;
  if (i >= inputs.height / kPatchArea) return;
  const size_t base = i * kPatchArea;
  const BilateralWeightComputer computer(__uint_as_float(inputs.Get(base, 0)),
                                         __uint_as_float(inputs.Get(base, 1)));
  const float center = __uint_as_float(inputs.Get(base, 4));
#pragma unroll
  for (int row = -kWindowRadius; row <= kWindowRadius; ++row) {
#pragma unroll
    for (int col = -kWindowRadius; col <= kWindowRadius; ++col) {
      const int sample =
          (row + kWindowRadius) * kWindowSize + col + kWindowRadius;
      const float reference = __uint_as_float(inputs.Get(base + sample, 5));
      const float weight = computer.Compute(row, col, center, reference);
      weights.Set(base + sample, 0, __float_as_uint(weight));
    }
  }
}

// Independent instruction-level oracle. Volatile PTX fixes each rounding/FTZ
// boundary without relying on the frontend's choice of expression contraction.
// RCP and EX2 deliberately execute on the same GPU as Compute; CPU exp/double
// and universal golden approximations are not valid bitwise oracles here.
__device__ float Add(float a, float b) {
  float result;
  asm volatile("add.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ float Mul(float a, float b) {
  float result;
  asm volatile("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ float Fma(float a, float b, float c) {
  float result;
  asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;"
               : "=f"(result)
               : "f"(a), "f"(b), "f"(c));
  return result;
}

__device__ float Reciprocal(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ float Exp(float value) {
  // CUDA fast-math exp(float): RN/FTZ multiply by float32 log2(e), then EX2.
  const float argument = Mul(value, __uint_as_float(0x3fb8aa3b));
  float result;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(argument));
  return result;
}

__global__ void ReferenceWeights(GpuMatView<uint32_t> inputs,
                                 GpuMatView<uint32_t> results) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= inputs.height) return;
  const float sigma_s = __uint_as_float(inputs.Get(i, 0));
  const float sigma_c = __uint_as_float(inputs.Get(i, 1));
  const float spatial = Reciprocal(Mul(Mul(2.0f, sigma_s), sigma_s));
  const float color = Reciprocal(Mul(Mul(2.0f, sigma_c), sigma_c));
  const float row = __uint_as_float(inputs.Get(i, 2));
  const float col = __uint_as_float(inputs.Get(i, 3));
  // All row/col squares and their sums in this fixture are exact.
  const float distance = Add(Mul(row, row), Mul(col, col));
  const float difference = Add(__uint_as_float(inputs.Get(i, 4)),
                               -__uint_as_float(inputs.Get(i, 5)));
  const float square = Mul(difference, difference);
  const float intended = Fma(-distance, spatial, -Mul(square, color));
  // Deliberately round the spatial product and fuse the color product instead.
  // An ordinary legacy C++ expression can choose the intended order, so it is
  // not a reliable negative control across frontend architectures.
  const float opposite = Fma(-square, color, Mul(-distance, spatial));
  results.Set(i, 0, __float_as_uint(Exp(intended)));
  results.Set(i, 1, __float_as_uint(Exp(opposite)));
  results.Set(i, 2, __float_as_uint(intended));
  results.Set(i, 3, __float_as_uint(opposite));
#if defined(__CUDA_ARCH__)
  results.Set(i, 4, __CUDA_ARCH__);
#endif
}

struct Results {
  std::array<uint32_t, kNumWeights> actual;
  std::array<std::array<uint32_t, 5>, kNumWeights> reference;
};

Results RunWeights() {
  // Repeat each runtime color pair across all spatial offsets, including the
  // exact A17 sample 62 (0, 2) and sample 72 (1, 1) encodings above. The oracle
  // receives offsets as data; only the production call sees unrolled constants.
  std::array<std::array<uint32_t, 6>, kNumWeights> host_inputs;
  for (size_t i = 0; i < kInputs.size(); ++i) {
    for (int sample = 0; sample < kPatchArea; ++sample) {
      auto& input = host_inputs[i * kPatchArea + sample];
      input = kInputs[i];
      const float row = sample / kWindowSize - kWindowRadius;
      const float col = sample % kWindowSize - kWindowRadius;
      std::memcpy(&input[2], &row, sizeof(row));
      std::memcpy(&input[3], &col, sizeof(col));
    }
  }
  GpuMat<uint32_t> inputs(6, kNumWeights);
  GpuMat<uint32_t> actual(1, kNumWeights);
  GpuMat<uint32_t> reference(5, kNumWeights);
  inputs.CopyToDevice(host_inputs.front().data(), sizeof(host_inputs.front()));
  ComputeWeights<<<1, 32>>>(inputs.View(), actual.View());
  CUDA_SAFE_CALL(cudaGetLastError());
  ReferenceWeights<<<(kNumWeights + 127) / 128, 128>>>(inputs.View(),
                                                       reference.View());
  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  Results results;
  actual.CopyToHost(results.actual.data(), sizeof(uint32_t));
  reference.CopyToHost(results.reference.front().data(),
                       sizeof(results.reference.front()));
  return results;
}

TEST(BilateralWeightComputer, MatchesExplicitDeviceRounding) {
  const auto results = RunWeights();
  RecordProperty("cuda_frontend_arch", results.reference[0][4]);
  for (size_t i = 0; i < kNumWeights; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(results.actual[i], results.reference[i][0]);
  }
  EXPECT_EQ(results.actual[2 * kPatchArea + 60], 0x3f800000u);
  EXPECT_EQ(results.actual[7 * kPatchArea + 60], 0u);
}

TEST(BilateralWeightComputer, Sm120Cuda128HistoricalContractionControl) {
  // These literals describe a qualified configuration, NOT all GPU approximate
  // math. The portable test above compares against the same-device chain.
  int device;
  CUDA_SAFE_CALL(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device));
  if (properties.major != 12 || properties.minor != 0 ||
      std::string(properties.name).find("RTX 5090") == std::string::npos ||
      __CUDACC_VER_MAJOR__ != 12 || __CUDACC_VER_MINOR__ != 8 ||
      __CUDACC_VER_BUILD__ != 93) {
    GTEST_SKIP() << "Historical literals require RTX 5090 and nvcc 12.8.93";
  }
  const auto results = RunWeights();
  if (results.reference[0][4] < 1000) {
    GTEST_SKIP()
        << "Contraction regression requires compute_100/120 frontend; "
           "configure native Blackwell rather than compute_90 fallback";
  }
  const uint32_t intended[] = {0x3f3d3a7d, 0x3f0981a2};
  const uint32_t opposite[] = {0x3f3d3a7e, 0x3f0981a3};
  for (size_t i = 0; i < 2; ++i) {
    SCOPED_TRACE(i == 0 ? "sample 62" : "sample 72");
    const size_t index = i * kPatchArea + (i == 0 ? 62 : 72);
    EXPECT_EQ(results.reference[index][0], intended[i]);
    EXPECT_EQ(results.reference[index][1], opposite[i]);
    EXPECT_NE(results.reference[index][2], results.reference[index][3]);
    EXPECT_EQ(results.actual[index], intended[i]);
  }
}

}  // namespace
}  // namespace colmap::mvs
