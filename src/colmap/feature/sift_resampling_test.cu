// SPDX-License-Identifier: BSD-3-Clause
// Original kernel copyright (c) 2007 University of North Carolina at Chapel
// Hill. All Rights Reserved. Permission to use, copy, modify and distribute
// this software and its documentation for educational, research and non-profit
// purposes, without fee, and without a written agreement is hereby granted,
// provided that the above copyright notice and the following paragraph appear
// in all copies. The University of North Carolina at Chapel Hill make no
// representations about the suitability of this software for any purpose. It is
// provided 'as is' without express or implied warranty.

// Run: ctest --test-dir <build> -R feature/sift_resampling_test
// --output-on-failure Passing this probe alone does NOT enable or qualify
// aligned -1 streaming. Test-only original resampling oracle. No streamed
// kernel/index helper is used.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
using std::vector;
// clang-format off: legacy bases must precede derived classes.
#define SIFTGPU_CUDA_ENABLED
#include "thirdparty/SiftGPU/GlobalUtil.h"
#include "thirdparty/SiftGPU/GLTexImage.h"
#include "thirdparty/SiftGPU/CuTexImage.h"
#include "thirdparty/SiftGPU/ProgramCU.h"
#include "thirdparty/SiftGPU/StreamedPlane.h"
#include "thirdparty/SiftGPU/SiftGPU.h"
#include "thirdparty/SiftGPU/SiftPyramid.h"
#include "thirdparty/SiftGPU/PyramidCU.h"
// clang-format on

namespace colmap {
namespace {
size_t CountNonfinite(const std::vector<float>& values) {
  size_t count = 0;
  for (float value : values) count += !std::isfinite(value);
  return count;
}

TEST(SiftOriginalResampling, NonfiniteCountsPreserveBits) {
  const std::vector<float> values = {0.0f,
                                     -0.0f,
                                     std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity()};
  const auto original = values;
  EXPECT_EQ(CountNonfinite(values), 3);
  EXPECT_EQ(CountNonfinite({0.0f, -0.0f, 1.0f}), 0);
  EXPECT_EQ(std::memcmp(values.data(), original.data(), values.size() * 4), 0);
}

class ProbeImage : public CuTexImage {
 public:
  float* Data() { return static_cast<float*>(_cuData); }
};
// Frozen from d798b6f23^ ProgramCU.cu, retaining arithmetic and index+1.
// Tile multiplication replaces __mul24 only for these tiny positive fixtures.
template <int LOG_SCALE>
__global__ void OriginalUpsampleKernel(cudaTextureObject_t texData,
                                       float* d_result,
                                       int width) {
  const int SCALE = (1 << LOG_SCALE), SCALE_MASK = (SCALE - 1);
  const float INV_SCALE = 1.0f / (float(SCALE));
  int col = (blockIdx.x * 128) + threadIdx.x;
  if (col >= width) return;

  int row = blockIdx.y >> LOG_SCALE;
  int index = row * width + col;
  int dst_row = blockIdx.y;
  int dst_idx = (width * dst_row + col) * SCALE;
  int helper = blockIdx.y & SCALE_MASK;
  if (helper) {
    float v11 = tex1Dfetch<float>(texData, index);
    float v12 = tex1Dfetch<float>(texData, index + 1);
    index += width;
    float v21 = tex1Dfetch<float>(texData, index);
    float v22 = tex1Dfetch<float>(texData, index + 1);
    float w1 = INV_SCALE * helper, w2 = 1.0 - w1;
    float v1 = (v21 * w1 + w2 * v11);
    float v2 = (v22 * w1 + w2 * v12);
    d_result[dst_idx] = v1;
#pragma unroll
    for (int i = 1; i < SCALE; ++i) {
      const float r2 = i * INV_SCALE;
      const float r1 = 1.0f - r2;
      d_result[dst_idx + i] = v1 * r1 + v2 * r2;
    }
  } else {
    float v1 = tex1Dfetch<float>(texData, index);
    float v2 = tex1Dfetch<float>(texData, index + 1);
    d_result[dst_idx] = v1;
#pragma unroll
    for (int i = 1; i < SCALE; ++i) {
      const float r2 = i * INV_SCALE;
      const float r1 = 1.0f - r2;
      d_result[dst_idx + i] = v1 * r1 + v2 * r2;
    }
  }
}

// CUDA runtime texture-object documentation says addressMode is ignored for
// cudaResourceTypeLinear. Do not infer an out-of-resource value from Clamp:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html
// The original bound extent is CuTexImage::_numBytes, not active width*height.
// This tiny probe records actual raw fetch bits, without assigning semantics.
__global__ void ObserveLinearFetches(cudaTextureObject_t texture,
                                     float* output,
                                     int width,
                                     int height) {
  const int indexes[] = {width * height - 1,
                         width * height,
                         width * height + width - 1,
                         width * height + width};
  output[threadIdx.x] = tex1Dfetch<float>(texture, indexes[threadIdx.x]);
}

void OriginalSample(ProbeImage& destination, CuTexImage& source) {
  cudaTextureDesc descriptor{};
  descriptor.readMode = cudaReadModeElementType;
  descriptor.filterMode = cudaFilterModePoint;
  for (auto& mode : descriptor.addressMode) mode = cudaAddressModeClamp;
  auto texture = source.BindTexture(descriptor, cudaCreateChannelDesc<float>());
  ASSERT_NE(texture.handle, 0);
  ProbeImage observation;
  ASSERT_TRUE(observation.InitTexture(4, 1));
  ObserveLinearFetches<<<1, 4>>>(texture.handle,
                                 observation.Data(),
                                 source.GetImgWidth(),
                                 source.GetImgHeight());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  uint32_t observed[4];
  ASSERT_EQ(observation.CopyToHost(observed), cudaSuccess);
  std::cout << "SIFT_LINEAR_FETCH active="
            << source.GetImgWidth() * source.GetImgHeight()
            << " bound_bytes=" << source.GetDataSize()
            << " bits=" << observed[0] << ',' << observed[1] << ','
            << observed[2] << ',' << observed[3] << '\n';
  OriginalUpsampleKernel<1>
      <<<dim3((source.GetImgWidth() + 127) / 128, source.GetImgHeight() * 2),
         128>>>(texture.handle, destination.Data(), source.GetImgWidth());
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(SiftOriginalResampling, AlignedAllocationEdgesTailsAndReuse) {
  int devices = 0;
  ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
  if (!devices) GTEST_SKIP() << "No CUDA device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  // Fresh capacity versus retained, explicitly initialized tail capacity.
  // Out-of-resource results are OBSERVATIONS, never assumed clamp or zero.
  for (bool retained : {false, true}) {
    ProbeImage source, original, current;
    for (auto shape : {std::pair<int, int>{132, 9},
                       std::pair<int, int>{128, 3},
                       std::pair<int, int>{132, 9}}) {
      const int w = shape.first, h = shape.second;
      SCOPED_TRACE(::testing::Message()
                   << w << 'x' << h << " retained=" << retained);
      if (!retained) {
        // Each iteration below also has a fresh exact-capacity control.
        ProbeImage fresh;
        ASSERT_TRUE(fresh.InitTexture(w, h));
        std::vector<float> values(w * h);
        for (int i = 0; i < w * h; ++i) values[i] = float(i + 1) / 4096.0f;
        ASSERT_EQ(fresh.CopyFromHost(values.data()), cudaSuccess);
        ASSERT_TRUE(original.InitTexture(w * 2, h * 2));
        ASSERT_TRUE(current.InitTexture(w * 2, h * 2));
        OriginalSample(original, fresh);
        ProgramCU::SampleImageU(&current, &fresh, 1);
      } else {
        ASSERT_TRUE(source.InitTexture(w, h + 2));
        std::vector<float> values(w * (h + 2));
        for (size_t i = 0; i < values.size(); ++i)
          values[i] = float(i + 1) / 4096.0f;
        ASSERT_EQ(source.CopyFromHost(values.data()), cudaSuccess);
        source.SetImageSize(w, h);
        ASSERT_TRUE(original.InitTexture(w * 2, h * 2));
        ASSERT_TRUE(current.InitTexture(w * 2, h * 2));
        OriginalSample(original, source);
        ProgramCU::SampleImageU(&current, &source, 1);
      }
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::vector<float> expected(w * h * 4), actual(expected.size());
      ASSERT_EQ(original.CopyToHost(expected.data()), cudaSuccess);
      ASSERT_EQ(current.CopyToHost(actual.data()), cudaSuccess);
      const auto expected_nonfinite = CountNonfinite(expected);
      const auto actual_nonfinite = CountNonfinite(actual);
      std::cout << "SIFT_U_NUMERICAL w=" << w << " h=" << h
                << " retained=" << retained
                << " expected_nonfinite=" << expected_nonfinite
                << " actual_nonfinite=" << actual_nonfinite << '\n';
      // Nonfatal checks preserve all raw-bit observations below. Nonfinite
      // results fail numerical qualification; they do not define or repair
      // unqualified out-of-resource fetch semantics.
      EXPECT_EQ(expected_nonfinite, 0);
      EXPECT_EQ(actual_nonfinite, 0);
      EXPECT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * 4),
                0);
      // Known in-allocation flattened right edge, NOT horizontal clamping.
      const int edge = (2 * w) + 2 * (w - 1) + 1;
      const float literal =
          (float(w) * 0.5f + float(2 * w) * 0.5f) * 0.5f +
          (float(w + 1) * 0.5f + float(2 * w + 1) * 0.5f) * 0.5f;
      EXPECT_EQ(actual[edge], literal / 4096.0f);
      uint32_t bits[3];
      const int positions[] = {
          2 * w * (2 * h - 2) + 2 * w - 1, 2 * w * (2 * h - 1), 4 * w * h - 1};
      for (int i = 0; i < 3; ++i)
        std::memcpy(bits + i, actual.data() + positions[i], 4);
      std::cout << "SIFT_ORIGINAL_U_EDGE w=" << w << " h=" << h
                << " retained=" << retained << " tail_bits=" << bits[0] << ','
                << bits[1] << ',' << bits[2] << '\n';
    }
  }
}

#if !defined(_WIN32)
TEST(SiftOriginalResampling, StreamedAlignedOriginalBytesAndPublishedExtent) {
  int devices = 0;
  ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
  if (!devices) GTEST_SKIP() << "No CUDA device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  char pattern[] = "./sift-resampling-XXXXXX";
  const char* directory = mkdtemp(pattern);
  ASSERT_NE(directory, nullptr);
  struct RemoveDirectory {
    const char* path;
    ~RemoveDirectory() { rmdir(path); }
  } cleanup{directory};
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
  // Reused native allocation AND history belong to one extractor. No sorting.
  const std::vector<std::vector<std::pair<int, int>>> histories = {
      {{128, 1},
       {128, 1},
       {128, 2},
       {128, 5},
       {128, 9},
       {128, 2},
       {128, 1},
       {128, 9}},
      {{132, 9}, {128, 9}, {132, 9}, {133, 9}, {128, 2}, {132, 10}},
      {{6048, 1}, {6048, 2}, {6048, 5}, {6048, 2}},
      {{257, 1}, {128, 2}},  // C=N+1
      {{384, 1}, {128, 2}},  // C=N+W
      {{385, 1}, {128, 2}}   // C=N+W+1
  };
  for (const auto& sequence : histories) {
    ProbeImage source, original;
    siftgpu::StreamInputHistory history;
    int generation = 0;
    for (const auto& dimensions : sequence) {
      const int width = dimensions.first, height = dimensions.second;
      SCOPED_TRACE(::testing::Message() << width << 'x' << height);
      std::vector<float> values(width * height);
      const int offset = ++generation * 16;
      for (size_t i = 0; i < values.size(); ++i)
        values[i] = float(i + 1 + offset) / 65536.0f;
      ASSERT_TRUE(source.InitTexture(width, height));
      ASSERT_EQ(source.CopyFromHost(values.data()), cudaSuccess);
      ASSERT_TRUE(original.InitTexture(width * 2, height * 2));
      OriginalSample(original,
                     source);  // Frozen original, no candidate helper.
      auto& input = history.Begin(directory, width, height);
      input.WriteRows(0, height, values.data());
      input.Seal();
      siftgpu::StreamFilterOptions options;
      options.scratch_directory = directory;
      options.gpu_bytes = 256 * 1024;
      options.host_bytes = 128 * 1024;
      auto streamed = ProgramCU::SampleImageStreamed(
          input, width * 2, height * 2, true, false, options);
      std::vector<float> expected(values.size() * 4), actual(expected.size());
      ASSERT_EQ(original.CopyToHost(expected.data()), cudaSuccess);
      streamed.output->ReadRows(0, height * 2, actual.data());
      EXPECT_EQ(CountNonfinite(expected), 0);
      EXPECT_EQ(CountNonfinite(actual), 0);
      EXPECT_EQ(std::memcmp(expected.data(), actual.data(), actual.size() * 4),
                0);
      if (width % 4 == 0 && history.Capacity() > values.size()) {
        ProbeImage fresh_source, fresh_output;
        ASSERT_TRUE(fresh_source.InitTexture(width, height));
        ASSERT_EQ(fresh_source.CopyFromHost(values.data()), cudaSuccess);
        ASSERT_TRUE(fresh_output.InitTexture(width * 2, height * 2));
        OriginalSample(fresh_output, fresh_source);
        std::vector<float> fresh(expected.size());
        ASSERT_EQ(fresh_output.CopyToHost(fresh.data()), cudaSuccess);
        EXPECT_NE(std::memcmp(expected.data(), fresh.data(), fresh.size() * 4),
                  0)
            << "Sentinels must distinguish retained history from a fresh reset";
      }
      std::cout << "SIFT_INPUT_HISTORY width=" << width << " height=" << height
                << " capacity=" << history.Capacity()
                << " case=" << history.Case() << '\n';
      if (height >= 3) {
        // Right edge of odd output row consumes third source row's first value.
        const float literal =
            (float(width) * 0.5f + float(2 * width) * 0.5f) * 0.5f +
            (float(width + 1) * 0.5f + float(2 * width + 1) * 0.5f) * 0.5f;
        EXPECT_EQ(actual[4 * width - 1], (literal + offset) / 65536.0f);
      }
      siftgpu::SamplePlan plan(width,
                               height,
                               width * 2,
                               height * 2,
                               true,
                               false,
                               options.gpu_bytes,
                               options.host_bytes,
                               properties.textureAlignment,
                               history.Capacity());
      for (int row = 0; row < height * 2; ++row) {
        const auto window = plan.Window(row);
        EXPECT_EQ(window.origin * sizeof(float) % properties.textureAlignment,
                  0);
        EXPECT_LE(window.values, plan.source_values);
        if (row >= (height - 1) * 2)
          EXPECT_EQ(
              (window.values * sizeof(float)) % properties.textureAlignment,
              (std::min(size_t(history.Capacity()),
                        size_t((row / 2 + 2) * width + 1)) *
               sizeof(float)) %
                  properties.textureAlignment);
      }
      // Bind a non-16-byte extent inside a larger reused allocation. Hardware
      // boundary is compared to an exact-sized original resource, never zero.
      cudaTextureDesc descriptor{};
      descriptor.readMode = cudaReadModeElementType;
      ProbeImage staging, exact, observed, control;
      const int count = width * 2 + 1;
      std::vector<float> sentinels(count + 64, 0.875f);
      ASSERT_TRUE(staging.InitTexture(count + 64, 1));
      ASSERT_EQ(staging.CopyFromHost(sentinels.data()), cudaSuccess);
      ASSERT_TRUE(exact.InitTexture(count, 1));
      ASSERT_EQ(exact.CopyFromHost(sentinels.data()), cudaSuccess);
      ASSERT_TRUE(observed.InitTexture(4, 1));
      ASSERT_TRUE(control.InitTexture(4, 1));
      auto bounded = staging.BindTexture(
          descriptor, cudaCreateChannelDesc<float>(), count * 4);
      auto reference =
          exact.BindTexture(descriptor, cudaCreateChannelDesc<float>());
      ASSERT_NE(bounded.handle, 0);
      ASSERT_NE(reference.handle, 0);
      ObserveLinearFetches<<<1, 4>>>(bounded.handle, observed.Data(), count, 1);
      ObserveLinearFetches<<<1, 4>>>(
          reference.handle, control.Data(), count, 1);
      ASSERT_EQ(cudaGetLastError(), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      float a[4], b[4];
      ASSERT_EQ(observed.CopyToHost(a), cudaSuccess);
      ASSERT_EQ(control.CopyToHost(b), cudaSuccess);
      for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(a[i]));
        EXPECT_TRUE(std::isfinite(b[i]));
      }
      EXPECT_EQ(std::memcmp(a, b, sizeof(a)), 0);
      EXPECT_THROW(staging.BindTexture(descriptor,
                                       cudaCreateChannelDesc<float>(),
                                       size_t(staging.GetDataSize()) + 4),
                   std::invalid_argument);
      EXPECT_THROW(
          staging.BindTexture(descriptor, cudaCreateChannelDesc<float>(), 3),
          std::invalid_argument);
      EXPECT_LE(streamed.stats.gpu_staging_bytes, options.gpu_bytes);
      EXPECT_LE(streamed.stats.host_staging_bytes, options.host_bytes);
    }
  }
}

TEST(SiftOriginalResampling, AlignedFullPyramidOriginalFloatBytes) {
  int devices = 0;
  ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
  if (!devices) GTEST_SKIP() << "No CUDA device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  char pattern[] = "./sift-pyramid-history-XXXXXX";
  const char* directory = mkdtemp(pattern);
  ASSERT_NE(directory, nullptr);
  struct RemoveDirectory {
    const char* path;
    ~RemoveDirectory() { rmdir(path); }
  } cleanup{directory};
  struct Restore {
    int first = GlobalUtil::_octave_min_default;
    int count = GlobalUtil::_octave_num_default;
    int tight = GlobalUtil::_ForceTightPyramid;
    int fit = GlobalUtil::_FitMemoryCap;
    ~Restore() {
      GlobalUtil::_octave_min_default = first;
      GlobalUtil::_octave_num_default = count;
      GlobalUtil::_ForceTightPyramid = tight;
      GlobalUtil::_FitMemoryCap = fit;
    }
  } restore;
  GlobalUtil::_ForceTightPyramid = 1;
  GlobalUtil::_FitMemoryCap = 0;
  GlobalUtil::_octave_num_default = 3;
  for (int first : {-1, 0}) {
    GlobalUtil::_octave_min_default = first;
    SiftParam parameters;
    parameters.ParseSiftParam();
    std::unique_ptr<float[]> sigma_owner(parameters._sigma);
    PyramidCU resident(parameters);
    ProbeImage source;
    siftgpu::StreamInputHistory history;
    for (int width : {132, 128, 132, 133, 128, 132}) {
      const int height = 65;
      SCOPED_TRACE(::testing::Message() << width << " first=" << first);
      std::vector<float> values(width * height);
      for (size_t i = 0; i < values.size(); ++i)
        values[i] = float((i * 7919 + i / width * 31) % 65521) / 65521.0f;
      GLTexInput input;
      input.SetImageSize(width, height);
      input._pixel_data = values.data();
      resident.InitPyramid(width, height);
      resident.BuildPyramid(&input);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      ASSERT_TRUE(source.InitTexture(width, height));
      ASSERT_EQ(source.CopyFromHost(values.data()), cudaSuccess);
      source._exactDomain = width % 4 != 0;
      auto& disk_input = history.Begin(directory, width, height);
      disk_input.WriteRows(0, height, values.data());
      disk_input.Seal();
      siftgpu::StreamFilterOptions options;
      options.scratch_directory = directory;
      options.gpu_bytes = 256 * 1024;
      options.host_bytes = 128 * 1024;
      auto candidate = PyramidCU::BuildGaussianPyramidStreamed(
          parameters, disk_input, first, 3, options);
      std::vector<std::unique_ptr<ProbeImage>> levels;
      for (int octave = 0; octave < 3; ++octave) {
        auto expected = resident.GetBaseLevel(first + octave);
        ProbeImage buffer;
        ASSERT_TRUE(buffer.InitTexture(expected[0].GetImgWidth(),
                                       expected[0].GetImgHeight()));
        for (int level = 0; level < parameters._level_num; ++level) {
          auto image = std::make_unique<ProbeImage>();
          ASSERT_TRUE(image->InitTexture(expected[level].GetImgWidth(),
                                         expected[level].GetImgHeight()));
          if (level) {
            ProgramCU::FilterImage(image.get(),
                                   levels.back().get(),
                                   &buffer,
                                   parameters._sigma[level - 1]);
          } else if (!octave) {
            if (first == -1) {
              if (width % 4 == 0)
                OriginalSample(*image, source);
              else
                ProgramCU::SampleImageU(image.get(), &source, 1);
            }
            ProgramCU::FilterImage(image.get(),
                                   first == -1 ? image.get() : &source,
                                   &buffer,
                                   parameters.GetInitialSmoothSigma(first));
          } else {
            ProgramCU::SampleImageD(
                image.get(),
                levels[(octave - 1) * parameters._level_num +
                       parameters._level_ds - parameters._level_min]
                    .get());
            if (parameters._sigma_skip1 > 0)
              ProgramCU::FilterImage(
                  image.get(), image.get(), &buffer, parameters._sigma_skip1);
          }
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
          std::vector<float> oracle(image->GetImgWidth() *
                                    image->GetImgHeight()),
              actual(oracle.size());
          ASSERT_EQ(image->CopyToHost(oracle.data()), cudaSuccess);
          ASSERT_EQ(expected[level].CopyToHost(actual.data()), cudaSuccess);
          const auto oracle_nonfinite = CountNonfinite(oracle);
          const auto actual_nonfinite = CountNonfinite(actual);
          std::cout << "SIFT_PYRAMID_NUMERICAL w=" << width
                    << " first=" << first << " octave=" << octave
                    << " level=" << level
                    << " oracle_nonfinite=" << oracle_nonfinite
                    << " actual_nonfinite=" << actual_nonfinite << '\n';
          EXPECT_EQ(oracle_nonfinite, 0)
              << "Numerical qualification failed; raw bytes remain unchanged";
          EXPECT_EQ(actual_nonfinite, 0)
              << "Numerical qualification failed; raw bytes remain unchanged";
          EXPECT_EQ(
              std::memcmp(oracle.data(), actual.data(), oracle.size() * 4), 0)
              << "octave=" << octave << " level=" << level;
          auto& disk_level =
              candidate.levels[octave * parameters._level_num + level];
          ASSERT_EQ(disk_level->Width(), image->GetImgWidth());
          ASSERT_EQ(disk_level->Height(), image->GetImgHeight());
          disk_level->ReadRows(0, disk_level->Height(), actual.data());
          EXPECT_EQ(CountNonfinite(actual), 0);
          EXPECT_EQ(
              std::memcmp(oracle.data(), actual.data(), oracle.size() * 4), 0);
          levels.push_back(std::move(image));
        }
      }
    }
  }
}
#endif  // !defined(_WIN32)
}  // namespace
}  // namespace colmap
