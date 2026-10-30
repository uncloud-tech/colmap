// SPDX-License-Identifier: BSD-3-Clause

// As in the cache trace test, include the actual kernels with internal linkage
// and exclude solver definitions to avoid duplicate symbols in colmap_mvs_cuda.
#define COLMAP_OWNED_CACHE_TRACE_TEST
#include "colmap/mvs/patch_match_cuda_owned.cu"
#undef COLMAP_OWNED_CACHE_TRACE_TEST

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {

// Frozen pre-fusion basis pass; spatial rotation uses the independent generic
// CudaRotate kernel, once per plane, just as the previous solver did.
__global__ void OldNormalBasis(GpuMatView<float> normal) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < normal.GetWidth() && y < normal.GetHeight()) {
    const float nx = normal.Get(y, x, 0);
    const float ny = normal.Get(y, x, 1);
    normal.Set(y, x, 0, ny);
    normal.Set(y, x, 1, -nx);
  }
}

// Deliberately use unequal padded pitches, not cudaMallocPitch's potentially
// identical rounded pitches. This is test storage only, not a product
// allocator.
struct PitchedNormal {
  PitchedNormal(size_t w, size_t h, size_t padding)
      : width(w), height(h), pitch((w + padding) * sizeof(float)) {
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ptr), Bytes()));
  }
  ~PitchedNormal() { (void)cudaFree(ptr); }
  size_t Bytes() const { return pitch * height * 3; }
  GpuMatView<float> View() const { return {ptr, pitch, width, height, 3}; }
  float* ptr = nullptr;
  size_t width, height, pitch;
};

void CheckRotation(size_t width, size_t height, bool wait) {
  OwnedCudaStream stream;
  OwnedRotationPair<PitchedNormal> fused, reference;
  for (unsigned p = 0; p < 2; ++p) {
    const size_t w = p ? height : width;
    const size_t h = p ? width : height;
    fused.Set(p, std::make_unique<PitchedNormal>(w, h, p ? 23 : 11));
    reference.Set(p, std::make_unique<PitchedNormal>(w, h, p ? 31 : 17));
  }
  std::vector<float> original(width * height * 3);
  // Include both zero signs, infinities, subnormals and signed quiet/signaling
  // NaN payloads. Use memcpy so the host never performs floating-point math on
  // these patterns. Most pixels are coordinate/plane-specific finite values.
  const uint32_t patterns[] = {0x00000000u,
                               0x80000000u,
                               0x7f800000u,
                               0xff800000u,
                               0x00000001u,
                               0x80000001u,
                               0x7fc12345u,
                               0xffc54321u,
                               0x7f812345u,
                               0xff854321u};
  for (size_t s = 0; s < 3; ++s) {
    for (size_t i = 0; i < width * height; ++i) {
      // Match the all-zero invalid normal produced by filtering. In particular
      // do not canonicalize the signed zero created by the subsequent negation.
      const size_t index = s * width * height + i;
      const uint32_t bits =
          i % 7 == 0
              ? 0u
              : (i % 3 == 0 ? patterns[(i / 3 + s) % 10]
                            : 0x3f000000u + static_cast<uint32_t>(index));
      std::memcpy(&original[index], &bits, sizeof(bits));
    }
  }
  std::vector<float> initial[2][2];
  std::vector<float> result[4], expected[4], before[4], after[4];
  PitchedNormal roundtrip(width, height, 19);
  std::vector<float> roundtrip_expected(roundtrip.Bytes() / sizeof(float));
  ScopedStreamCompletion completion(stream.Get());
  for (unsigned arm = 0; arm < 2; ++arm) {
    auto& pair = arm ? reference : fused;
    for (unsigned p = 0; p < 2; ++p) {
      auto* map = pair.At(p);
      auto& host = initial[arm][p];
      host.assign(map->Bytes() / sizeof(float), 12345.0f);
      if (p == 0) {
        for (size_t s = 0; s < 3; ++s) {
          for (size_t y = 0; y < height; ++y) {
            std::memcpy(host.data() + (s * height + y) * map->pitch / 4,
                        original.data() + (s * height + y) * width,
                        width * sizeof(float));
          }
        }
      }
      CUDA_SAFE_CALL(cudaMemcpyAsync(map->ptr,
                                     host.data(),
                                     map->Bytes(),
                                     cudaMemcpyHostToDevice,
                                     stream.Get()));
    }
  }
  // Four spatial rotations restore coordinates, but legacy floating-point
  // negation need not restore raw NaN payloads or subnormals. Apply just the
  // four legacy basis changes to an independent copy in the original layout.
  CUDA_SAFE_CALL(cudaMemcpy2DAsync(roundtrip.ptr,
                                   roundtrip.pitch,
                                   reference.Current()->ptr,
                                   reference.Current()->pitch,
                                   width * sizeof(float),
                                   height * 3,
                                   cudaMemcpyDeviceToDevice,
                                   stream.Get()));
  const dim3 roundtrip_block(32, 16);
  const dim3 roundtrip_grid((width + 31) / 32, (height + 15) / 16);
  for (unsigned turn = 0; turn < 4; ++turn) {
    OldNormalBasis<<<roundtrip_grid, roundtrip_block, 0, stream.Get()>>>(
        roundtrip.View());
  }
  CUDA_SAFE_CALL(cudaMemcpyAsync(roundtrip_expected.data(),
                                 roundtrip.ptr,
                                 roundtrip.Bytes(),
                                 cudaMemcpyDeviceToHost,
                                 stream.Get()));
  CUDA_CHECK();
  // All host buffers and device owners outlive the enclosing stream drain,
  // including on exceptions. No per-rotation synchronization in chained mode.
  for (unsigned turn = 0; turn < 4; ++turn) {
    auto* input = fused.Current();
    auto* output = fused.Alternate();
    const dim3 block(32, 16);
    const dim3 grid((input->width + 31) / 32, (input->height + 15) / 16);
    before[turn].resize(input->Bytes() / 4);
    after[turn].resize(input->Bytes() / 4);
    result[turn].resize(output->Bytes() / 4);
    expected[turn].resize(reference.Alternate()->Bytes() / 4);
    CUDA_SAFE_CALL(cudaMemcpyAsync(before[turn].data(),
                                   input->ptr,
                                   input->Bytes(),
                                   cudaMemcpyDeviceToHost,
                                   stream.Get()));
    // Exercise exactly the production launch geometry, including partial tiles.
    LaunchNormalMapRotation(input->View(), output->View(), stream.Get());
    CUDA_SAFE_CALL(cudaMemcpyAsync(after[turn].data(),
                                   input->ptr,
                                   input->Bytes(),
                                   cudaMemcpyDeviceToHost,
                                   stream.Get()));
    OldNormalBasis<<<grid, block, 0, stream.Get()>>>(reference->View());
    auto src = reference->View();
    auto dst = reference.Alternate()->View();
    for (size_t s = 0; s < 3; ++s) {
      CudaRotate(src.ptr + s * src.pitch / 4 * src.height,
                 dst.ptr + s * dst.pitch / 4 * dst.height,
                 src.width,
                 src.height,
                 src.pitch,
                 dst.pitch,
                 stream.Get());
    }
    CUDA_SAFE_CALL(cudaMemcpyAsync(result[turn].data(),
                                   output->ptr,
                                   output->Bytes(),
                                   cudaMemcpyDeviceToHost,
                                   stream.Get()));
    CUDA_SAFE_CALL(cudaMemcpyAsync(expected[turn].data(),
                                   dst.ptr,
                                   reference.Alternate()->Bytes(),
                                   cudaMemcpyDeviceToHost,
                                   stream.Get()));
    CUDA_CHECK();
    fused.Flip();
    reference.Flip();
    EXPECT_EQ(fused.CurrentParity(), (turn + 1) % 2);
    if (wait) SynchronizeStream(stream.Get());
  }
  completion.Wait();
  for (unsigned turn = 0; turn < 4; ++turn) {
    SCOPED_TRACE(turn);
    EXPECT_EQ(std::memcmp(before[turn].data(),
                          after[turn].data(),
                          before[turn].size() * sizeof(float)),
              0);
    const unsigned p = (turn + 1) % 2;
    auto* actual = fused.At(p);
    auto* oracle = reference.At(p);
    for (size_t s = 0; s < 3; ++s) {
      for (size_t y = 0; y < actual->height; ++y) {
        const float* row =
            result[turn].data() + (s * actual->height + y) * actual->pitch / 4;
        const float* expected_row =
            expected[turn].data() +
            (s * oracle->height + y) * oracle->pitch / 4;
        for (size_t x = 0; x < actual->width; ++x) {
          uint32_t actual_bits, expected_bits;
          std::memcpy(&actual_bits, row + x, sizeof(actual_bits));
          std::memcpy(&expected_bits, expected_row + x, sizeof(expected_bits));
          ASSERT_EQ(actual_bits, expected_bits)
              << "plane=" << s << " row=" << y << " col=" << x << " actual=0x"
              << std::hex << actual_bits << " expected=0x" << expected_bits;
        }
        for (size_t x = actual->width; x < actual->pitch / 4; ++x) {
          EXPECT_EQ(row[x], 12345.0f);
        }
        if (turn == 3) {
          EXPECT_EQ(std::memcmp(
                        row,
                        roundtrip_expected.data() +
                            (s * height + y) * roundtrip.pitch / sizeof(float),
                        width * sizeof(float)),
                    0);
        }
      }
    }
  }
}

// Exercise the real legacy cudaMallocPitch -> compact -> cudaFreeAsync seam.
// Compare subsequent draws rather than opaque curandState padding. The control
// uses the previous wait-then-destruct release path. Run with compute-sanitizer
// as well as normally; Nsight API tracing separately verifies release calls.
TEST(OwnedPrngRelease, CompactedDrawsMatchSynchronousRelease) {
  OwnedCudaStream stream;
  for (const auto shape :
       {std::pair<int, int>{37, 19}, {19, 37}, {1, 65}, {65, 1}, {1, 1}}) {
    for (int seed : {-1, 0, 7}) {
      SCOPED_TRACE(::testing::Message()
                   << shape.first << "x" << shape.second << " seed=" << seed);
      const PrngBoundaryLayout layout(shape.first, shape.second);
      auto actual =
          MakeOwnedPrng(shape.first, shape.second, stream.Get(), seed);
      auto control =
          MakeOwnedPrng(shape.first, shape.second, stream.Get(), seed);
      const size_t original_pitch = actual->GetPitch();
      EXPECT_GE(original_pitch, shape.first * sizeof(curandState));
      // Advance full-image states before compaction, as initialization does.
      GpuMat<float> init_draws(shape.first, shape.second);
      init_draws.FillWithRandomNumbers(0.0f, 1.0f, *actual, stream.Get());
      init_draws.FillWithRandomNumbers(0.0f, 1.0f, *control, stream.Get());
      GpuMat<curandState> compact(layout.Count(), 1);
      GpuMat<curandState> reference(layout.Count(), 1);
      const dim3 block(32, 16);
      const dim3 grid((shape.first + 31) / 32, (shape.second + 15) / 16);
      {
        ScopedStreamCompletion completion(stream.Get());
        CompactOwnedPrngBoundary<<<grid, block, 0, stream.Get()>>>(
            actual->View(), compact.GetPtr(), layout);
        actual->ReleaseAfterInitialization(stream.Get());
        EXPECT_EQ(actual->GetPtr(), nullptr);
        EXPECT_EQ(actual->GetPitch(), original_pitch);
        completion.Wait();
      }
      actual.reset();  // Must not free the transferred pointer again.
      {
        ScopedStreamCompletion completion(stream.Get());
        CompactOwnedPrngBoundary<<<grid, block, 0, stream.Get()>>>(
            control->View(), reference.GetPtr(), layout);
        completion.Wait();
      }
      control.reset();
      GpuMat<float> draws(layout.Count(), 1), expected(layout.Count(), 1);
      std::vector<float> observed(layout.Count()), oracle(layout.Count());
      for (int draw = 0; draw < 16; ++draw) {
        draws.FillWithRandomNumbers(0.0f, 1.0f, compact, stream.Get());
        expected.FillWithRandomNumbers(0.0f, 1.0f, reference, stream.Get());
        draws.CopyToHost(
            observed.data(), observed.size() * sizeof(float), stream.Get());
        expected.CopyToHost(
            oracle.data(), oracle.size() * sizeof(float), stream.Get());
        EXPECT_EQ(observed, oracle);
      }
    }
  }
}

TEST(OwnedNormalRotation, PitchedTwoPassBitwiseReference) {
  const std::pair<size_t, size_t> shapes[] = {{65, 97},
                                              {97, 65},
                                              {31, 33},
                                              {33, 31},
                                              {32, 32},
                                              {64, 64},
                                              {35, 19},
                                              {3, 7},
                                              {1, 65},
                                              {65, 1},
                                              {1, 9},
                                              {9, 1},
                                              {1, 1}};
  for (const auto& shape : shapes) {
    for (bool wait : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << shape.first << "x" << shape.second << " wait=" << wait);
      CheckRotation(shape.first, shape.second, wait);
    }
  }
}

}  // namespace
}  // namespace colmap::mvs::owned
