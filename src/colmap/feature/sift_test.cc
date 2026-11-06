// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
#include <GL/glew.h>
#endif

#if defined(COLMAP_GUI_ENABLED)
#include <QApplication>
#else
#include "colmap/exe/gui.h"
#endif

#include "colmap/feature/sift.h"
#include "colmap/feature/sift_orientation_test_util.h"
#include "colmap/feature/utils.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/math/random.h"
#include "colmap/util/opengl_utils.h"
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
#include "colmap/util/cuda.h"
#include <vector>
// Legacy SiftPyramid/PyramidCU headers require vector in the enclosing scope.
using std::vector;
// clang-format off: legacy base-class headers must precede derived headers.
#include "thirdparty/SiftGPU/GlobalUtil.h"
#include "thirdparty/SiftGPU/GLTexImage.h"
#include "thirdparty/SiftGPU/CuTexImage.h"
// SiftGPU's directory-private definition also guards its public declarations.
#ifndef SIFTGPU_CUDA_ENABLED
#define SIFTGPU_CUDA_ENABLED
#include "thirdparty/SiftGPU/ProgramCU.h"
#include "thirdparty/SiftGPU/SiftGPU.h"
#include "thirdparty/SiftGPU/SiftPyramid.h"
#include "thirdparty/SiftGPU/PyramidCU.h"
#undef SIFTGPU_CUDA_ENABLED
#else
#include "thirdparty/SiftGPU/ProgramCU.h"
#include "thirdparty/SiftGPU/SiftGPU.h"
#include "thirdparty/SiftGPU/SiftPyramid.h"
#include "thirdparty/SiftGPU/PyramidCU.h"
#endif
// clang-format on
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#if defined(__linux__)
#include <csignal>

#include <sys/resource.h>
#endif
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

namespace colmap {
namespace {

Bitmap CreateImageWithSquare(const int size) {
  Bitmap bitmap(size, size, false);
  bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
  for (int r = size / 2 - size / 8; r < size / 2 + size / 8; ++r) {
    for (int c = size / 2 - size / 8; c < size / 2 + size / 8; ++c) {
      bitmap.SetPixel(r, c, BitmapColor<uint8_t>(255));
    }
  }
  return bitmap;
}

TEST(SiftHostAllocation, CpuOutputUnchangedAndGrowthRefuses) {
  for (bool covariant : {false, true}) {
    FeatureExtractionOptions options;
    options.use_gpu = false;
    options.sift->force_covariant_extractor = covariant;
    const Bitmap bitmap = CreateImageWithSquare(128);
    auto control = CreateSiftFeatureExtractor(options);
    ASSERT_NE(control, nullptr);
    FeatureKeypoints original_keypoints;
    FeatureDescriptors original_descriptors;
    ASSERT_TRUE(control->Extract(bitmap, &original_keypoints, &original_descriptors));
    uint64_t charged = 0;
    options.reserve_host_payload = [&](uint64_t bytes) { charged += bytes; };
    auto candidate = CreateSiftFeatureExtractor(options);
    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
    ASSERT_TRUE(candidate->Extract(bitmap, &keypoints, &descriptors));
    EXPECT_GT(charged, 0);
    EXPECT_EQ(keypoints, original_keypoints);
    ASSERT_EQ(descriptors.data.rows(), original_descriptors.data.rows());
    EXPECT_TRUE((descriptors.data.array() == original_descriptors.data.array()).all());
    options.reserve_host_payload = [](uint64_t) { throw std::runtime_error("growth refused"); };
    auto refused = CreateSiftFeatureExtractor(options);
    FeatureKeypoints refused_keypoints;
    FeatureDescriptors refused_descriptors;
    EXPECT_THROW(refused->Extract(bitmap, &refused_keypoints, &refused_descriptors), std::runtime_error);
    EXPECT_TRUE(refused_keypoints.empty());
    EXPECT_EQ(refused_descriptors.data.size(), 0);
  }
  FeatureExtractionOptions options;
  options.reserve_host_payload = [](uint64_t) {};
  EXPECT_THROW(options.ReserveHostPayload(std::numeric_limits<uint64_t>::max(), 2), std::overflow_error);
}

// Helper to create empty descriptors for testing.
FeatureDescriptors CreateEmptyDescriptors() {
  return FeatureDescriptors(FeatureExtractorType::SIFT,
                            FeatureDescriptorsData(0, 128));
}

// Helper to create reversed descriptors for testing matcher symmetry.
FeatureDescriptors CreateReversedDescriptors(const FeatureDescriptors& src) {
  return FeatureDescriptors(src.type, src.data.colwise().reverse());
}

void ValidateKeypoints(const FeatureKeypoints& keypoints,
                       const Bitmap& bitmap) {
  for (size_t i = 0; i < keypoints.size(); ++i) {
    EXPECT_GE(keypoints[i].x, 0);
    EXPECT_GE(keypoints[i].y, 0);
    EXPECT_LE(keypoints[i].x, bitmap.Width());
    EXPECT_LE(keypoints[i].y, bitmap.Height());
    EXPECT_GT(keypoints[i].ComputeScale(), 0);
    EXPECT_GT(keypoints[i].ComputeOrientation(), -M_PI);
    EXPECT_LT(keypoints[i].ComputeOrientation(), M_PI);
  }
}

void ValidateDescriptorNorms(const FeatureDescriptors& descriptors) {
  EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
  for (Eigen::Index i = 0; i < descriptors.data.rows(); ++i) {
    EXPECT_LT(std::abs(descriptors.data.row(i).cast<float>().norm() - 512), 1);
  }
}

void ExpectReversedMatches(const FeatureMatches& matches) {
  EXPECT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].point2D_idx1, 0);
  EXPECT_EQ(matches[0].point2D_idx2, 1);
  EXPECT_EQ(matches[1].point2D_idx1, 1);
  EXPECT_EQ(matches[1].point2D_idx2, 0);
}

void ExpectReversedInlierMatches(const TwoViewGeometry& two_view_geometry) {
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 2);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[1].point2D_idx2, 0);
}

TwoViewGeometry CreatePlanarTwoViewGeometry() {
  TwoViewGeometry tvg;
  tvg.config = TwoViewGeometry::PLANAR_OR_PANORAMIC;
  tvg.H = Eigen::Matrix3d::Identity();
  return tvg;
}

void RunGpuTest(const std::function<void()>& test_body) {
#if !defined(COLMAP_GPU_ENABLED)
  GTEST_SKIP() << "Requires OpenGL, CUDA or HIP support";
#else
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
  // The compute backend unconditionally requires a device when compiled in,
  // so skip rather than fail on hardware-less machines.
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "No CUDA or HIP devices available";
  }
#endif
  char app_name[] = "Test";
  int argc = 1;
  char* argv[] = {app_name};
  QApplication app(argc, argv);

  class TestThread : public Thread {
   public:
    std::function<void()> body;

   private:
    void Run() {
      opengl_context_.MakeCurrent();
      body();
    }
    OpenGLContextManager opengl_context_;
  };

  TestThread thread;
  thread.body = test_body;
  RunThreadWithOpenGLContext(&thread);
#endif
}

struct SiftCpuExtractionParams {
  std::string name;
  bool estimate_affine_shape;
  bool domain_size_pooling;
  bool force_covariant_extractor;
  bool upright;
  size_t expected_keypoints;
};

class SiftCpuExtractionTest
    : public ::testing::TestWithParam<SiftCpuExtractionParams> {};

TEST_P(SiftCpuExtractionTest, Nominal) {
  const auto& p = GetParam();
  const Bitmap bitmap = CreateImageWithSquare(256);

  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  options.use_gpu = false;
  options.sift->estimate_affine_shape = p.estimate_affine_shape;
  options.sift->domain_size_pooling = p.domain_size_pooling;
  options.sift->force_covariant_extractor = p.force_covariant_extractor;
  options.sift->upright = p.upright;
  auto extractor = CreateSiftFeatureExtractor(options);

  FeatureKeypoints keypoints;
  FeatureDescriptors descriptors;
  EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

  EXPECT_EQ(keypoints.size(), p.expected_keypoints);
  ValidateKeypoints(keypoints, bitmap);
  EXPECT_EQ(descriptors.data.rows(), p.expected_keypoints);
  ValidateDescriptorNorms(descriptors);
}

INSTANTIATE_TEST_SUITE_P(
    SiftCpuExtraction,
    SiftCpuExtractionTest,
    ::testing::Values(
        SiftCpuExtractionParams{"Sift", false, false, false, false, 22},
        SiftCpuExtractionParams{"CovariantSift", false, false, true, false, 22},
        SiftCpuExtractionParams{
            "CovariantAffineSift", true, false, false, false, 22},
        SiftCpuExtractionParams{
            "CovariantAffineSiftUpright", true, false, false, true, 10},
        SiftCpuExtractionParams{
            "CovariantDSPSift", false, true, false, false, 22},
        SiftCpuExtractionParams{
            "CovariantAffineDSPSift", true, true, false, false, 22}),
    [](const auto& info) { return info.param.name; });

TEST(SiftStreamingOptions, ExplicitComponentCapsAndUnsupportedConfigurations) {
  FeatureExtractionOptions options(FeatureExtractorType::SIFT);
  EXPECT_FALSE(options.sift->streaming);
  EXPECT_TRUE(options.sift->Check());
  options.sift->streaming = true;
  EXPECT_FALSE(options.sift->Check());
  options.sift->streaming_gpu_bytes = 65536;
  options.sift->streaming_host_bytes = 65536;
  options.sift->streaming_scratch_path = ".";
  EXPECT_TRUE(options.sift->Check());
  options.sift->first_octave = 1;
  EXPECT_FALSE(options.sift->Check());
  options.sift->first_octave = -1;
  options.sift->domain_size_pooling = true;
  EXPECT_FALSE(options.sift->Check());
  options.sift->domain_size_pooling = false;
  options.use_gpu = false;
  EXPECT_FALSE(options.Check());
  EXPECT_THROW(CreateSiftFeatureExtractor(options), std::exception);
}

TEST(ExtractSiftFeaturesGPU, Nominal) {
  RunGpuTest([] {
    const Bitmap bitmap = CreateImageWithSquare(256);

    FeatureExtractionOptions options(FeatureExtractorType::SIFT);
    options.use_gpu = true;
    options.sift->estimate_affine_shape = false;
    options.sift->domain_size_pooling = false;
    options.sift->force_covariant_extractor = false;
    auto extractor = CreateSiftFeatureExtractor(options);

    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
    EXPECT_TRUE(extractor->Extract(bitmap, &keypoints, &descriptors));

    EXPECT_GE(keypoints.size(), 12);
    ValidateKeypoints(keypoints, bitmap);
    EXPECT_GE(descriptors.data.rows(), 12);
    ValidateDescriptorNorms(descriptors);
  });
}

#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
#if !defined(_WIN32)
TEST(ExtractSiftFeaturesGPU, StreamedPublicOutputMatchesResidentOrderedBytes) {
  RunGpuTest([] {
    // Match native phase tests: scratch must be disk-backed, not a /tmp tmpfs.
    char pattern[] = "./sift-public-XXXXXX";
    const char* directory = mkdtemp(pattern);
    ASSERT_NE(directory, nullptr);
    const std::string scratch(directory);
    // Original public resident Extract includes GetFeatureVector coordinates,
    // affine keypoint layout, normalization and uint8 descriptor conversion.
    // Neither sorted output nor a candidate expansion helper is the oracle.
    // Exercise default output and active first/second whole-level limiters.
    for (int feature_limit : {8192, 8}) {
      SCOPED_TRACE(feature_limit);
      for (auto normalization : {SiftExtractionOptions::Normalization::L1_ROOT,
                                 SiftExtractionOptions::Normalization::L2}) {
        for (int first : {-1, 0}) {
          FeatureExtractionOptions resident_options(FeatureExtractorType::SIFT);
          resident_options.use_gpu = true;
          resident_options.gpu_index = "0";
          resident_options.max_image_size = 1024;
          resident_options.sift->first_octave = first;
          resident_options.sift->max_num_features = feature_limit;
          resident_options.sift->normalization = normalization;
          auto resident = CreateSiftFeatureExtractor(resident_options);
          ASSERT_NE(resident, nullptr);
          auto streamed_options = resident_options;
          streamed_options.host_budget_bytes = 256 * 1024 * 1024;
          streamed_options.sift->streaming = true;
          streamed_options.sift->streaming_gpu_bytes = 512 * 1024;
          streamed_options.sift->streaming_host_bytes = 512 * 1024;
          streamed_options.sift->streaming_scratch_path = scratch;
          auto streamed = CreateSiftFeatureExtractor(streamed_options);
          ASSERT_NE(streamed, nullptr);
          FeatureKeypoints actual;
          FeatureDescriptors actual_descriptors;
          const std::vector<int> widths =
              first == -1 ? std::vector<int>{132,
                                             128,
                                             132,
                                             256,
                                             256,
                                             259,
                                             128,
                                             257,
                                             259,
                                             260,
                                             264}
                          : std::vector<int>{259, 257, 259, 260};
          for (int width : widths) {
            SCOPED_TRACE(::testing::Message() << first << " " << width);
            Bitmap bitmap(width, 129, false);
            // Keep the strong square edges/nonzero detections, but exercise
            // byte-to-float division on intermediate grays, not only 0 and 255.
            for (int y = 0; y < bitmap.Height(); ++y)
              for (int x = 0; x < width; ++x) {
                const bool square = y >= 32 && y < 96 && x >= width / 2 - 32 &&
                                    x < width / 2 + 32;
                const int gray = (17 * x + 29 * y) % 32;
                bitmap.SetPixel(
                    x, y, BitmapColor<uint8_t>(square ? 224 + gray : gray));
              }
            const auto original = bitmap.RowMajorData();
            FeatureKeypoints expected;
            FeatureDescriptors expected_descriptors;
            ASSERT_TRUE(
                resident->Extract(bitmap, &expected, &expected_descriptors));
            ASSERT_FALSE(expected.empty());
            ASSERT_TRUE(
                streamed->Extract(bitmap, &actual, &actual_descriptors));
            ASSERT_EQ(actual.size(), expected.size());
            EXPECT_EQ(actual, expected);
            // Include signed-zero/float bit identity, not tolerance
            // comparisons.
            EXPECT_EQ(std::memcmp(actual.data(),
                                  expected.data(),
                                  actual.size() * sizeof(FeatureKeypoint)),
                      0);
            ASSERT_EQ(actual_descriptors.type, expected_descriptors.type);
            ASSERT_EQ(actual_descriptors.data.rows(),
                      expected_descriptors.data.rows());
            ASSERT_EQ(actual_descriptors.data.cols(), 128);
            EXPECT_EQ(std::memcmp(actual_descriptors.data.data(),
                                  expected_descriptors.data.data(),
                                  actual_descriptors.data.size()),
                      0);
            EXPECT_TRUE((actual_descriptors.data.array() != 0).any());
            for (const auto& key : actual) {
              EXPECT_TRUE(std::isfinite(key.x));
              EXPECT_TRUE(std::isfinite(key.y));
              EXPECT_TRUE(std::isfinite(key.a11));
              EXPECT_TRUE(std::isfinite(key.a12));
              EXPECT_TRUE(std::isfinite(key.a21));
              EXPECT_TRUE(std::isfinite(key.a22));
            }
            EXPECT_EQ(bitmap.RowMajorData(), original);
            EXPECT_TRUE(std::filesystem::is_empty(scratch));
          }
        }
      }
    }
    std::filesystem::remove(scratch);
  });
}
TEST(ExtractSiftFeaturesGPU,
     StreamedPublicFailureAndCancellationDoNotFallback) {
  RunGpuTest([] {
    // Match native phase tests: scratch must be disk-backed, not a /tmp tmpfs.
    char pattern[] = "./sift-public-XXXXXX";
    const char* directory = mkdtemp(pattern);
    ASSERT_NE(directory, nullptr);
    const std::string scratch(directory);
    FeatureExtractionOptions options(FeatureExtractorType::SIFT);
    options.use_gpu = true;
    options.gpu_index = "0";
    options.host_budget_bytes = 256 * 1024 * 1024;
    options.sift->first_octave = 0;
    options.sift->streaming = true;
    options.sift->streaming_gpu_bytes = 65536;
    options.sift->streaming_host_bytes = 65536;
    options.sift->streaming_scratch_path = scratch;
    Bitmap bitmap(257, 129, false);
    bitmap.Fill(BitmapColor<uint8_t>(0));
    FeatureKeypoints keys(1, FeatureKeypoint(3, 4));
    const auto original = keys;
    FeatureDescriptors descriptors(FeatureExtractorType::SIFT,
                                   FeatureDescriptorsData(1, 128));
    for (int i = 0; i < descriptors.data.size(); ++i)
      descriptors.data.data()[i] = (37 * i + 11) % 256;
    const auto original_descriptors = descriptors;
    const auto expect_unchanged = [&] {
      ASSERT_EQ(keys.size(), original.size());
      EXPECT_EQ(std::memcmp(keys.data(),
                            original.data(),
                            original.size() * sizeof(FeatureKeypoint)),
                0);
      EXPECT_EQ(descriptors.type, original_descriptors.type);
      ASSERT_EQ(descriptors.data.rows(), original_descriptors.data.rows());
      ASSERT_EQ(descriptors.data.cols(), original_descriptors.data.cols());
      EXPECT_EQ(std::memcmp(descriptors.data.data(),
                            original_descriptors.data.data(),
                            original_descriptors.data.size()),
                0);
    };
    for (int limit : {5, 200}) {
      int checks = 0;
      options.sift->streaming_cancelled = [&] { return ++checks > limit; };
      auto cancelled = CreateSiftFeatureExtractor(options);
      ASSERT_NE(cancelled, nullptr);
      EXPECT_THROW(cancelled->Extract(bitmap, &keys, &descriptors),
                   std::runtime_error);
      EXPECT_GT(checks,
                limit);  // Row staging, then native processing cancellation.
      expect_unchanged();
      EXPECT_TRUE(std::filesystem::is_empty(scratch));
      EXPECT_THROW(
          cancelled->Extract(bitmap, &keys, &descriptors),
          std::runtime_error);  // Failed history is never reused silently.
      expect_unchanged();
    }
    options.sift->streaming_cancelled = {};
    options.sift->streaming_scratch_path = scratch + "/missing";
    auto disk_error = CreateSiftFeatureExtractor(options);
    EXPECT_THROW(disk_error->Extract(bitmap, &keys, &descriptors),
                 std::runtime_error);
    expect_unchanged();
    EXPECT_TRUE(std::filesystem::is_empty(scratch));
    options.sift->streaming_scratch_path = scratch;
    options.sift->streaming_gpu_bytes = 1;
    auto component_no_fit = CreateSiftFeatureExtractor(options);
    ASSERT_NE(component_no_fit, nullptr);
    try {
      component_no_fit->Extract(bitmap, &keys, &descriptors);
      FAIL() << "Expected insufficient GPU staging capacity to be rejected";
    } catch (const std::invalid_argument& error) {
      EXPECT_STREQ(error.what(), "SIFT streamed KEY dimensions/capacity");
    }
    expect_unchanged();
    EXPECT_TRUE(std::filesystem::is_empty(scratch));
    options.sift->streaming_gpu_bytes = 65536;
    options.host_budget_bytes = 1;
    auto no_fit = CreateSiftFeatureExtractor(options);
    EXPECT_THROW(no_fit->Extract(bitmap, &keys, &descriptors),
                 std::runtime_error);
    expect_unchanged();
    EXPECT_TRUE(std::filesystem::is_empty(scratch));
    std::filesystem::remove(scratch);
  });
}
#endif

TEST(ExtractSiftFeaturesGPU, InputDomainAndImmutability) {
  RunGpuTest([] {
    FeatureExtractionOptions options(FeatureExtractorType::SIFT);
    options.use_gpu = true;
    options.gpu_index = "0";
    auto extractor = CreateSiftFeatureExtractor(options);
    ASSERT_NE(extractor, nullptr);
    ASSERT_TRUE(GlobalUtil::_UseCUDA);
    for (int width : {1, 2, 3, 4, 5, 257, 258, 259, 260}) {
      for (int height : {1, 2, 17}) {
        std::vector<float> pixels(width * height);
        std::iota(pixels.begin(), pixels.end(), 0.0f);
        const auto original = pixels;
        GLTexInput input;
        ASSERT_TRUE(input.SetImageData(width, height, pixels.data(),
                                      GL_LUMINANCE, GL_FLOAT));
        EXPECT_EQ(input.GetImgWidth(), width);
        EXPECT_EQ(input.GetImgHeight(), height);
        EXPECT_EQ(pixels, original);
        std::vector<uint8_t> bytes(width * height);
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = i % 251;
        const auto original_bytes = bytes;
        ASSERT_TRUE(input.SetImageData(width, height, bytes.data(),
                                      GL_LUMINANCE, GL_UNSIGNED_BYTE));
        ASSERT_EQ(input.GetImgWidth(), width);
        ASSERT_EQ(input.GetImgHeight(), height);
        const auto* converted = static_cast<const float*>(input._pixel_data);
        for (size_t i = 0; i < bytes.size(); ++i) {
          EXPECT_FLOAT_EQ(converted[i], bytes[i] / 255.0f);
        }
        EXPECT_EQ(bytes, original_bytes);
      }
    }
  });
}

TEST(ExtractSiftFeaturesGPU, ExactDomainCudaUpsampling) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "No CUDA or HIP devices available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  for (int width : {1, 2, 3, 4, 5, 257, 258, 259, 260}) {
    for (int height : {1, 3}) {
      SCOPED_TRACE(::testing::Message() << "width=" << width
                                      << " height=" << height);
      CuTexImage src;
      // Populate oversized, rounded capacity before reducing the logical size.
      // Rows are tightly packed: none of this tail is image content, even on
      // the last row. Poison makes accidental texture fetches observable.
      const int capacity_width = (width + 3) / 4 * 4;
      ASSERT_TRUE(src.InitTexture(capacity_width, height + 1));
      std::vector<float> input(capacity_width * (height + 1), -8192.0f);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          input[y * width + x] = 8.0f + 4.0f * x + 64.0f * y;
        }
      }
      src.CopyFromHost(input.data());
      src.SetImageSize(width, height);
      src._exactDomain = true;
      for (int log_scale : {1, 2, 3}) {
        SCOPED_TRACE(::testing::Message() << "log_scale=" << log_scale);
        const int scale = 1 << log_scale;
        const int dst_width = width * scale;
        const int dst_height = height * scale;
        CuTexImage dst;
        ASSERT_TRUE(dst.InitTexture(dst_width, dst_height));
        std::vector<float> output(dst_width * dst_height, -8192.0f);
        dst.CopyFromHost(output.data());
        ProgramCU::SampleImageU(&dst, &src, log_scale);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        dst.CopyToHost(output.data());
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        for (int y = 0; y < dst_height; ++y) {
          for (int x = 0; x < dst_width; ++x) {
            // Bilinear interpolation reproduces an affine plane exactly.
            // Constant extension clamps continuous coordinates independently,
            // not flattened texture indices. All values are binary-exact.
            const float sample_x =
                std::min(static_cast<float>(x) / scale,
                         static_cast<float>(width - 1));
            const float sample_y =
                std::min(static_cast<float>(y) / scale,
                         static_cast<float>(height - 1));
            const float expected = 8.0f + 4.0f * sample_x + 64.0f * sample_y;
            ASSERT_EQ(output[y * dst_width + x], expected)
                << "x=" << x << " y=" << y;
          }
        }
      }
    }
  }
}

// The control retains every original dense plane before any list/orientation
// consumer runs. It never regenerates and does not use PyramidCU's reuse path.
void BuildDenseKeyList(CuTexImage* key, CuTexImage* list, int* count) {
  std::vector<std::unique_ptr<CuTexImage>> hist;
  int width = (key->GetImgWidth() + 3) / 4;
  do {
    hist.push_back(std::make_unique<CuTexImage>());
    ASSERT_TRUE(hist.back()->InitTexture(width, key->GetImgHeight(), 4));
    if (hist.size() == 1) {
      ProgramCU::InitHistogram(key, hist.back().get());
    } else {
      ProgramCU::ReduceHistogram(hist[hist.size() - 2].get(),
                                 hist.back().get());
    }
    if (width == 1) break;
    width = (width + 3) / 4;
  } while (true);
  std::vector<int> rows(key->GetImgHeight() * 4);
  hist.back()->CopyToHost(rows.data());
  *count = std::accumulate(rows.begin(), rows.end(), 0);
  ASSERT_TRUE(list->InitTexture(*count, 1, 4));
  if (*count == 0) return;
  std::vector<int> seeds;
  for (size_t i = 0; i < rows.size(); ++i) {
    for (int n = 0; n < rows[i]; ++n) {
      seeds.insert(seeds.end(), {int(i % 4), int(i / 4), n, 0});
    }
  }
  list->CopyFromHost(seeds.data());
  for (int i = int(hist.size()) - 2; i >= 0; --i) {
    ProgramCU::GenerateList(list, hist[i].get());
  }
}

void ExpectNativeFloatBits(CuTexImage* control,
                           CuTexImage* candidate,
                           int floats) {
  std::vector<uint32_t> original(floats), regenerated(floats);
  control->CopyToHost(original.data());
  candidate->CopyToHost(regenerated.data());
  EXPECT_EQ(original, regenerated);
}

#if !defined(_WIN32)
// Nonzero synthetic gradient fixture: opposing directions produce two peaks.
// Resident ComputeOrientation is run on the full independent plane/list; no
// candidate data is used as the oracle, and all ordered xyso bytes are checked.
TEST(ExtractSiftFeaturesGPU,
     StreamedOrientationMatchesResidentOrderedExpansion) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA or HIP device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  struct Restore {
    int subpixel = GlobalUtil::_SubpixelLocalization;
    int sign = GlobalUtil::_KeepExtremumSign;
    int fixed = GlobalUtil::_FixedOrientation;
    int orientations = GlobalUtil::_MaxOrientation;
    int normalized = GlobalUtil::_NormalizedSIFT;
    int dynamic = GlobalUtil::_UseDynamicIndexing;
    ~Restore() {
      GlobalUtil::_SubpixelLocalization = subpixel;
      GlobalUtil::_KeepExtremumSign = sign;
      GlobalUtil::_FixedOrientation = fixed;
      GlobalUtil::_MaxOrientation = orientations;
      GlobalUtil::_NormalizedSIFT = normalized;
      GlobalUtil::_UseDynamicIndexing = dynamic;
    }
  } restore;
  GlobalUtil::_FixedOrientation = 0;
  for (auto shape : {std::pair<int, int>{65, 1031}, {64, 17}}) {
    const int width = shape.first, height = 73, count = shape.second;
    std::vector<float> gradient(size_t(width) * height * 2);
    std::vector<float> key(size_t(width) * height * 4, 0);
    std::vector<float> seeds(count * 4);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const size_t i = size_t(y) * width + x;
        gradient[2 * i] = 1.0f + float((x + y) % 5) / 32;
        gradient[2 * i + 1] = (x % 2) ? 0.13f : -3.01f;
        key[4 * i] = (x % 3) ? 1.0f : -1.0f;
        key[4 * i + 1] = float(x % 3 - 1) / 8;
        key[4 * i + 2] = float(y % 3 - 1) / 8;
        key[4 * i + 3] = float((x + y) % 3 - 1) / 8;
      }
    for (int i = 0; i < count; ++i) {
      // Include true global edges, odd tail, repeated positions and batches.
      const int seed[4] = {i % width, (i * 7) % height, 0, 0};
      std::memcpy(seeds.data() + 4 * i, seed, 16);
    }
    siftgpu::StreamedPlane disk_gradient(".", width, height, 2);
    siftgpu::StreamedPlane disk_key(".", width, height, 4);
    disk_gradient.WriteRows(0, height, gradient.data());
    disk_gradient.Seal();
    disk_key.WriteRows(0, height, key.data());
    disk_key.Seal();
    siftgpu::StreamListResult input;
    input.count = count;
    input.records = std::make_unique<siftgpu::StreamedPlane>(".", 1, count, 4);
    input.records->WriteRows(0, count, seeds.data());
    input.records->Seal();
    CuTexImage resident_gradient, resident_key, resident_list;
    ASSERT_TRUE(resident_gradient.InitTexture(width, height, 2));
    ASSERT_TRUE(resident_key.InitTexture(width, height, 4));
    ASSERT_TRUE(resident_list.InitTexture(count, 1, 4));
    ASSERT_EQ(resident_gradient.CopyFromHost(gradient.data()), cudaSuccess);
    ASSERT_EQ(resident_key.CopyFromHost(key.data()), cudaSuccess);
    for (int subpixel : {0, 1})
      for (int orientations : {1, 2}) {
        GlobalUtil::_SubpixelLocalization = subpixel;
        GlobalUtil::_KeepExtremumSign = subpixel;
        GlobalUtil::_MaxOrientation = orientations;
        std::vector<float> packed(count * 4), expected(count * 8);
        ASSERT_EQ(resident_list.CopyFromHost(seeds.data()), cudaSuccess);
        ProgramCU::ComputeOrientation(&resident_list,
                                      &resident_gradient,
                                      &resident_key,
                                      1.6f,
                                      1.2599211f,
                                      0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(resident_list.CopyToHost(packed.data()), cudaSuccess);
        const int expected_count =
            orientations > 1 ? test::ReshapeOriginalResidentOrientations(
                                   packed.data(), count, expected.data())
                             : count;
        if (orientations == 1)
          std::copy(packed.begin(), packed.end(), expected.begin());
        ASSERT_GT(expected_count, 0);
        if (orientations == 2) ASSERT_GT(expected_count, count);
        for (int repeat = 0; repeat < 2; ++repeat) {
          GlobalUtil::_NormalizedSIFT = repeat ? 0 : 1;
          GlobalUtil::_UseDynamicIndexing = repeat;
          siftgpu::StreamFilterOptions options;
          options.scratch_directory = ".";
          options.gpu_bytes = 65 * 19 * 8 + 180;
          options.host_bytes = 65 * 19 * 8 + 96;
          ASSERT_LT(options.gpu_bytes, disk_gradient.Bytes());
          ASSERT_LT(options.host_bytes, disk_gradient.Bytes());
          if (count > 1000) ASSERT_LT(options.host_bytes, input.records->Bytes());
          const uint64_t before = disk_gradient.BytesRead() +
                                  disk_key.BytesRead() +
                                  input.records->BytesRead();
          auto result = ProgramCU::ComputeOrientationStreamed(
              input, disk_gradient, disk_key, 1.6f, 1.2599211f, options);
          ASSERT_EQ(result.count, expected_count);
          ASSERT_TRUE(result.records && result.records->Complete());
          std::vector<float> actual(size_t(result.count) * 4);
          result.records->ReadRows(0, result.count, actual.data());
          EXPECT_EQ(
              std::memcmp(expected.data(), actual.data(), actual.size() * 4),
              0);
          const uint64_t orientation_reads =
              disk_gradient.BytesRead() + disk_key.BytesRead() +
              input.records->BytesRead() - before;
          // Independent original resident descriptor from the oracle expansion,
          // never from the candidate's xyso. Compare every ordered F32 byte.
          CuTexImage expanded_list, resident_descriptors;
          ASSERT_TRUE(expanded_list.InitTexture(expected_count, 1, 4));
          ASSERT_EQ(expanded_list.CopyFromHost(expected.data()), cudaSuccess);
          ProgramCU::ComputeDescriptor(
              &expanded_list, &resident_gradient, &resident_descriptors);
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
          std::vector<float> descriptor_oracle(size_t(expected_count) * 128);
          ASSERT_EQ(resident_descriptors.CopyToHost(descriptor_oracle.data()),
                    cudaSuccess);
          auto descriptor_options = options;
          descriptor_options.gpu_bytes = width * 29 * 8 + 788;
          descriptor_options.host_bytes = width * 29 * 8 + 656;
          ASSERT_LT(descriptor_options.gpu_bytes, disk_gradient.Bytes());
          const uint64_t descriptor_before =
              disk_gradient.BytesRead() + result.records->BytesRead();
          auto descriptors = ProgramCU::ComputeDescriptorStreamed(
              result, disk_gradient, descriptor_options);
          ASSERT_EQ(descriptors.count, expected_count);
          ASSERT_TRUE(descriptors.records && descriptors.records->Complete());
          for (int i = 0; i < expected_count; ++i) {
            float row[128];
            descriptors.records->ReadRows(i, 1, row);
            bool nonzero = false;
            for (float value : row) {
              EXPECT_TRUE(std::isfinite(value));
              nonzero = nonzero || value != 0.0f;
            }
            EXPECT_TRUE(nonzero);
            EXPECT_EQ(std::memcmp(row, descriptor_oracle.data() + i * 128, 512),
                      0);
          }
          EXPECT_LE(descriptors.stats.gpu_staging_bytes,
                    descriptor_options.gpu_bytes);
          EXPECT_LE(descriptors.stats.host_staging_bytes,
                    descriptor_options.host_bytes);
          EXPECT_EQ(descriptors.stats.file_read_bytes,
                    disk_gradient.BytesRead() + result.records->BytesRead() -
                        descriptor_before);
          EXPECT_EQ(descriptors.stats.h2d_bytes,
                    descriptors.stats.file_read_bytes);
          EXPECT_EQ(descriptors.stats.d2h_bytes,
                    uint64_t(expected_count) * 640);
          EXPECT_EQ(descriptors.stats.file_written_bytes,
                    uint64_t(expected_count) * 512);
          std::cout << "SIFT_STREAM_DESCRIPTOR " << subpixel << ' '
                    << orientations << ' ' << repeat << ' ' << descriptors.count
                    << ' ' << descriptors.stats.gpu_staging_bytes << ' '
                    << descriptors.stats.host_staging_bytes << ' '
                    << descriptors.stats.file_read_bytes << ' '
                    << descriptors.stats.file_written_bytes << ' '
                    << descriptors.stats.h2d_bytes << ' '
                    << descriptors.stats.d2h_bytes << '\n';
          descriptor_options.gpu_bytes = width * 8 + 788;
          EXPECT_THROW(ProgramCU::ComputeDescriptorStreamed(
                           result, disk_gradient, descriptor_options),
                       std::runtime_error);
          descriptor_options.gpu_bytes = options.gpu_bytes;
          descriptor_options.cancelled = [] { return true; };
          EXPECT_THROW(ProgramCU::ComputeDescriptorStreamed(
                           result, disk_gradient, descriptor_options),
                       std::runtime_error);
          EXPECT_LE(result.stats.gpu_staging_bytes, options.gpu_bytes);
          EXPECT_LE(result.stats.host_staging_bytes, options.host_bytes);
          // Descriptor reads are deliberately separate from orientation
          // counters.
          EXPECT_EQ(result.stats.file_read_bytes, orientation_reads);
          EXPECT_EQ(result.stats.file_written_bytes, result.records->Bytes());
          EXPECT_EQ(result.stats.h2d_bytes, result.stats.file_read_bytes);
          EXPECT_EQ(result.stats.d2h_bytes, uint64_t(count) * 32);
          for (int i = 0; i < result.count; ++i) {
            EXPECT_TRUE(std::isfinite(actual[4 * i + 3]));
            const float angle = actual[4 * i + 3];
            EXPECT_TRUE((angle >= 0 && angle < 0.3f) ||
                        (angle > 3.0f && angle < 3.5f));
          }
          std::cout << "SIFT_STREAM_ORIENTATION " << subpixel << ' '
                    << orientations << ' ' << repeat << ' ' << result.count
                    << ' ' << result.stats.gpu_staging_bytes << ' '
                    << result.stats.host_staging_bytes << ' '
                    << result.stats.file_read_bytes << ' '
                    << result.stats.file_written_bytes << ' '
                    << result.stats.h2d_bytes << ' ' << result.stats.d2h_bytes
                    << '\n';
#if defined(__linux__)
          const auto open_files = [] {
            return std::distance(
                std::filesystem::directory_iterator("/proc/self/fd"),
                std::filesystem::directory_iterator());
          };
          const auto before_failure = open_files();
#endif
          options.gpu_bytes =
              65 * 8 + 180;  // One row fits, actual sigma window does not.
          EXPECT_THROW(
              ProgramCU::ComputeOrientationStreamed(
                  input, disk_gradient, disk_key, 1.6f, 1.2599211f, options),
              std::runtime_error);
          options.gpu_bytes = 1;
          EXPECT_THROW(
              ProgramCU::ComputeOrientationStreamed(
                  input, disk_gradient, disk_key, 1.6f, 1.2599211f, options),
              std::runtime_error);
          options.gpu_bytes = 65 * 19 * 8 + 180;
          options.cancelled = [] { return true; };
          EXPECT_THROW(
              ProgramCU::ComputeOrientationStreamed(
                  input, disk_gradient, disk_key, 1.6f, 1.2599211f, options),
              std::runtime_error);
          int checks = 0;
          options.cancelled = [&] { return ++checks > 25; };
          EXPECT_THROW(
              ProgramCU::ComputeOrientationStreamed(
                  input, disk_gradient, disk_key, 1.6f, 1.2599211f, options),
              std::runtime_error);
          options.cancelled = {};
          options.host_bytes = 65 * 8 + 96;
          EXPECT_THROW(
              ProgramCU::ComputeOrientationStreamed(
                  input, disk_gradient, disk_key, 1.6f, 1.2599211f, options),
              std::runtime_error);
          options.host_bytes = 65 * 19 * 8 + 96;
#if defined(__linux__)
          EXPECT_EQ(open_files(), before_failure);
#endif
          siftgpu::StreamListResult zero;
          auto empty = ProgramCU::ComputeOrientationStreamed(
              zero, disk_gradient, disk_key, 1.6f, 1.2599211f, options);
          EXPECT_EQ(empty.count, 0);
          EXPECT_FALSE(empty.records);
          descriptor_options.cancelled = {};
          descriptor_options.gpu_bytes = width * 29 * 8 + 788;
          auto no_descriptors = ProgramCU::ComputeDescriptorStreamed(
              zero, disk_gradient, descriptor_options);
          EXPECT_EQ(no_descriptors.count, 0);
          EXPECT_FALSE(no_descriptors.records);
          descriptor_options.scratch_directory = "/missing-sift-directory";
          EXPECT_THROW(ProgramCU::ComputeDescriptorStreamed(
                           result, disk_gradient, descriptor_options),
                       std::runtime_error);
          descriptor_options.scratch_directory = ".";
          descriptor_options.host_bytes = width * 8 + 656;
          EXPECT_THROW(ProgramCU::ComputeDescriptorStreamed(
                           result, disk_gradient, descriptor_options),
                       std::runtime_error);
          descriptor_options.host_bytes = width * 29 * 8 + 656;
          int descriptor_checks = 0;
          descriptor_options.cancelled = [&] {
            return ++descriptor_checks > 25;
          };
          EXPECT_THROW(ProgramCU::ComputeDescriptorStreamed(
                           result, disk_gradient, descriptor_options),
                       std::runtime_error);
#if defined(__linux__)
          EXPECT_EQ(open_files(), before_failure);
#endif
        }
      }
  }
}

TEST(ExtractSiftFeaturesGPU,
     StreamedOrientationSecondLimiterMatchesOriginalWholeLevels) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA or HIP device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  struct Restore {
    int subpixel = GlobalUtil::_SubpixelLocalization;
    int sign = GlobalUtil::_KeepExtremumSign;
    int fixed = GlobalUtil::_FixedOrientation;
    int orientations = GlobalUtil::_MaxOrientation;
    int threshold = GlobalUtil::_FeatureCountThreshold;
    int truncate = GlobalUtil::_TruncateMethod;
    ~Restore() {
      GlobalUtil::_SubpixelLocalization = subpixel;
      GlobalUtil::_KeepExtremumSign = sign;
      GlobalUtil::_FixedOrientation = fixed;
      GlobalUtil::_MaxOrientation = orientations;
      GlobalUtil::_FeatureCountThreshold = threshold;
      GlobalUtil::_TruncateMethod = truncate;
    }
  } restore;
  GlobalUtil::_SubpixelLocalization = 1;
  GlobalUtil::_KeepExtremumSign = 0;
  GlobalUtil::_FixedOrientation = 0;
  GlobalUtil::_MaxOrientation = 2;
  SiftParam parameters;
  parameters.ParseSiftParam();
  std::unique_ptr<float[]> sigma_owner(parameters._sigma);
  class OriginalLimiter : public PyramidCU {
   public:
    using PyramidCU::PyramidCU;
    std::vector<int> Apply(const std::vector<int>& counts, int have_keys) {
      delete[] _levelFeatureNum;
      _levelFeatureNum = new int[counts.size()];
      std::copy(counts.begin(), counts.end(), _levelFeatureNum);
      _octave_num = int(counts.size()) / param._dog_level_num;
      _featureNum = std::accumulate(counts.begin(), counts.end(), 0);
      _keypoint_buffer.resize(size_t(_featureNum) * 4);
      LimitFeatureCount(have_keys);  // Actual original native limiter body.
      return {_levelFeatureNum, _levelFeatureNum + counts.size()};
    }
  } original_limiter(parameters);
  const int width = 65, height = 73;
  std::vector<float> gradient(size_t(width) * height * 2);
  std::vector<float> key(size_t(width) * height * 4, 0);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      const size_t i = size_t(y) * width + x;
      gradient[i * 2] = 1.0f;
      gradient[i * 2 + 1] = x % 2 ? 0.13f : -3.01f;
      key[i * 4] = 1.0f;
    }
  siftgpu::StreamDetectionPyramidResult detection;
  auto& pyramid = detection.pyramid;
  pyramid.first_octave = 0;
  pyramid.level_min = parameters._level_min;
  pyramid.level_num = parameters._level_num;
  pyramid.exact_domain = true;
  for (int i = 0; i < pyramid.level_num - 1; ++i) {
    auto plane =
        std::make_unique<siftgpu::StreamedPlane>(".", width, height, 2);
    plane->WriteRows(0, height, gradient.data());
    plane->Seal();
    pyramid.gradients.push_back(std::move(plane));
  }
  for (int j = 0; j < parameters._dog_level_num; ++j) {
    siftgpu::StreamDetectionLevel record;
    record.octave = 0;
    record.level = j + 2;
    record.detection.key =
        std::make_unique<siftgpu::StreamedPlane>(".", width, height, 4);
    record.detection.key->WriteRows(0, height, key.data());
    record.detection.key->Seal();
    detection.detections.push_back(std::move(record));
  }
  // The two passes straddle exact equality, whole-level overshoot and removal.
  // Counts are unequal so tile-local clipping/sorting cannot accidentally pass.
  const std::vector<int> detected = {11, 13, 17};
  ASSERT_EQ(parameters._dog_level_num, detected.size());
  bool first_truncated = false, second_truncated = false;
  for (int method : {0, 1, 2})
    for (int threshold : {0, 1, 11, 12, 24, 41}) {
      GlobalUtil::_TruncateMethod = method;
      GlobalUtil::_FeatureCountThreshold = threshold;
      auto counts = original_limiter.Apply(detected, 0);
      auto candidate_first = detected;
      siftgpu::LimitStreamedFeatureCounts(candidate_first, threshold, method);
      ASSERT_EQ(candidate_first, counts);
      first_truncated |= counts != detected;
      siftgpu::StreamFeatureListsResult lists;
      lists.level_counts = counts;
      lists.levels.resize(counts.size());
      std::vector<std::vector<float>> expected(counts.size());
      std::vector<int> expanded_counts(counts.size(), 0);
      for (size_t j = 0; j < counts.size(); ++j) {
        if (!counts[j]) continue;
        std::vector<float> seeds(size_t(counts[j]) * 4);
        for (int i = 0; i < counts[j]; ++i) {
          int seed[4] = {int(20 + i % 20), int(20 + i % 25), 0, 0};
          std::memcpy(seeds.data() + 4 * i, seed, 16);
        }
        auto& source = lists.levels[j];
        source.count = counts[j];
        source.records =
            std::make_unique<siftgpu::StreamedPlane>(".", 1, counts[j], 4);
        source.records->WriteRows(0, counts[j], seeds.data());
        source.records->Seal();
        CuTexImage resident_list, resident_key, resident_gradient;
        ASSERT_TRUE(resident_list.InitTexture(counts[j], 1, 4));
        ASSERT_TRUE(resident_key.InitTexture(width, height, 4));
        ASSERT_TRUE(resident_gradient.InitTexture(width, height, 2));
        ASSERT_EQ(resident_list.CopyFromHost(seeds.data()), cudaSuccess);
        ASSERT_EQ(resident_key.CopyFromHost(key.data()), cudaSuccess);
        ASSERT_EQ(resident_gradient.CopyFromHost(gradient.data()), cudaSuccess);
        ProgramCU::ComputeOrientation(
            &resident_list,
            &resident_gradient,
            &resident_key,
            parameters.GetLevelSigma(int(j) + parameters._level_min + 1),
            powf(2.0f, 1.0f / parameters._dog_level_num),
            0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(resident_list.CopyToHost(seeds.data()), cudaSuccess);
        expected[j].resize(size_t(counts[j]) * 8);
        expanded_counts[j] = test::ReshapeOriginalResidentOrientations(
            seeds.data(), counts[j], expected[j].data());
        ASSERT_GT(expanded_counts[j], counts[j]);
      }
      auto retained = original_limiter.Apply(expanded_counts, 1);
      second_truncated |= retained != expanded_counts;
      siftgpu::StreamFilterOptions options;
      options.scratch_directory = ".";
      options.gpu_bytes = width * 29 * 8 + 180;
      options.host_bytes = width * 29 * 8 + 96;
      auto result = PyramidCU::GetFeatureOrientationsStreamed(
          parameters, detection, lists, method, threshold, options);
      EXPECT_EQ(result.level_counts, retained);
      EXPECT_EQ(result.count,
                std::accumulate(retained.begin(), retained.end(), 0));
      for (size_t j = 0; j < retained.size(); ++j) {
        if (!retained[j]) {
          EXPECT_FALSE(result.levels[j].records);
          continue;
        }
        std::vector<float> actual(size_t(retained[j]) * 4);
        result.levels[j].records->ReadRows(0, retained[j], actual.data());
        EXPECT_EQ(
            std::memcmp(expected[j].data(), actual.data(), actual.size() * 4),
            0);
      }
      options.gpu_bytes = width * 61 * 8 + 788;
      options.host_bytes = width * 61 * 8 + 656;
      ASSERT_LT(options.gpu_bytes, uint64_t(width) * height * 8);
      ASSERT_LT(options.host_bytes, uint64_t(width) * height * 8);
      for (int first : {-1, 0, 1}) {
        pyramid.first_octave = first;
        auto extraction = PyramidCU::GetFeatureDescriptorsStreamed(
            parameters, detection, result, options);
        EXPECT_EQ(extraction.features.count, result.count);
        EXPECT_EQ(extraction.features.level_counts, retained);
        for (size_t j = 0; j < retained.size(); ++j) {
          if (!retained[j]) {
            EXPECT_FALSE(extraction.descriptors[j].records);
            EXPECT_FALSE(extraction.features.levels[j].records);
            continue;
          }
          CuTexImage original_list, original_gradient, original_descriptor;
          ASSERT_TRUE(original_list.InitTexture(retained[j], 1, 4));
          ASSERT_TRUE(original_gradient.InitTexture(width, height, 2));
          ASSERT_EQ(original_list.CopyFromHost(expected[j].data()),
                    cudaSuccess);
          ASSERT_EQ(original_gradient.CopyFromHost(gradient.data()),
                    cudaSuccess);
          ProgramCU::ComputeDescriptor(
              &original_list, &original_gradient, &original_descriptor);
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
          std::vector<float> oracle(size_t(retained[j]) * 128);
          ASSERT_EQ(original_descriptor.CopyToHost(oracle.data()), cudaSuccess);
          for (int row = 0; row < retained[j]; ++row) {
            float descriptor[128], keypoint[4];
            extraction.descriptors[j].records->ReadRows(row, 1, descriptor);
            EXPECT_EQ(std::memcmp(descriptor, oracle.data() + row * 128, 512),
                      0);
            extraction.features.levels[j].records->ReadRows(row, 1, keypoint);
            const float scale = first == -1 ? 0.5f : float(1 << first);
            const float offset = GlobalUtil::_LoweOrigin ? 0 : 0.5f;
            float converted[4];
            const float* source = expected[j].data() + row * 4;
            converted[0] = scale * (source[0] - 0.5f) + offset;
            converted[1] = scale * (source[1] - 0.5f) + offset;
            converted[2] = scale * source[2];
            const double twopi = 2.0 * 3.14159265358979323846;
            converted[3] = float(fmod(twopi - source[3], twopi));
            EXPECT_EQ(std::memcmp(converted, keypoint, 16), 0);
          }
        }
        std::cout << "SIFT_STREAM_DESCRIPTOR_LIMIT " << method << ' '
                  << threshold << ' ' << first << ' '
                  << extraction.features.count << ' '
                  << extraction.stats.gpu_staging_bytes << ' '
                  << extraction.stats.host_staging_bytes << ' '
                  << extraction.stats.file_read_bytes << ' '
                  << extraction.stats.file_written_bytes << ' '
                  << extraction.stats.h2d_bytes << ' '
                  << extraction.stats.d2h_bytes << '\n';
      }
      pyramid.first_octave = 0;
      std::cout << "SIFT_STREAM_ORIENTATION_LIMIT " << method << ' '
                << threshold << ' ' << result.count << ' '
                << result.stats.file_read_bytes << ' '
                << result.stats.file_written_bytes << '\n';
    }
  EXPECT_TRUE(first_truncated);
  EXPECT_TRUE(second_truncated);
}
TEST(ExtractSiftFeaturesGPU, StreamedPyramidMatchesResidentBuildPyramid) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA or HIP device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  struct RestoreGlobals {
    int first = GlobalUtil::_octave_min_default;
    int count = GlobalUtil::_octave_num_default;
    int fit = GlobalUtil::_FitMemoryCap;
    int tight = GlobalUtil::_ForceTightPyramid;
    ~RestoreGlobals() {
      GlobalUtil::_octave_min_default = first;
      GlobalUtil::_octave_num_default = count;
      GlobalUtil::_FitMemoryCap = fit;
      GlobalUtil::_ForceTightPyramid = tight;
    }
  } restore;
  GlobalUtil::_FitMemoryCap = 0;
  GlobalUtil::_ForceTightPyramid = 1;
  GlobalUtil::_octave_num_default = 4;
  // Resident BuildPyramid, not a second implementation of pyramid equations,
  // is the oracle. Resident allocations and comparison vectors are outside
  // the candidate's staging budget, and exist only in this qualification test.
  for (auto shape : {std::pair<int, int>{129, 513},
                     {128, 517},
                     {132, 517},
                     {17, 301},
                     {7, 2},
                     {1, 1},
                     {129, 513}}) {
    for (int first : {0, -1}) {
      if (first == -1 && shape.first % 4 == 0) continue;
      for (int configuration : {0, 1, 2}) {
        // Additional coverage stays on one bounded fixture rather than tripling
        // every tall-plane/native-oracle case.
        if (configuration && shape != std::make_pair(17, 301)) continue;
        // Legacy aligned mode has no nonempty height after enough shifts.
        const int octaves =
            configuration == 1 ? 1 : (shape.first % 4 == 0 ? 4 : 6);
        GlobalUtil::_octave_min_default = first;
        GlobalUtil::_octave_num_default = octaves;
        SiftParam parameters;
        if (configuration == 2) {
          // ParseSiftParam clamps the seed to level -1 (index 2), generating
          // a genuinely positive skip1 sigma without modifying production math.
          parameters._dog_level_num = 4;
          parameters._level_min = -3;
          parameters._level_max = -1;
        }
        parameters.ParseSiftParam();
        if (configuration == 2) {
          ASSERT_EQ(parameters._level_num, 3);
          ASSERT_EQ(parameters._level_ds - parameters._level_min, 2);
          ASSERT_GT(parameters._sigma_skip1, 0);
        }
        std::unique_ptr<float[]> sigma_owner(parameters._sigma);
        const int width = shape.first, height = shape.second;
        std::vector<float> values(size_t(width) * height);
        for (size_t i = 0; i < values.size(); ++i)
          values[i] = float((i * 7919 + i / width * 31) % 65521) / 65521.0f;
        GLTexInput resident_input;
        resident_input.SetImageSize(width, height);
        resident_input._pixel_data = values.data();
        PyramidCU resident(parameters);
        resident.InitPyramid(width, height);
        resident.BuildPyramid(&resident_input);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        siftgpu::StreamedPlane input(".", width, height);
        input.WriteRows(0, height, values.data());
        input.Seal();
        siftgpu::StreamFilterOptions options;
        options.scratch_directory = ".";
        const size_t payload =
            size_t(width * (first == -1 ? 2 : 1)) * 40 * sizeof(float);
        options.gpu_bytes = 2 * payload + 132;
        options.host_bytes = payload + 132;
        if (height > 500) {
          ASSERT_GT(input.Bytes(), options.gpu_bytes);
          ASSERT_GT(input.Bytes(), options.host_bytes);
        }
        auto candidate = PyramidCU::BuildPyramidStreamed(
            parameters, input, first, octaves, options);
        // Component payload counters only: no aggregate RAM/cache/VRAM claim.
        std::cout << "STREAMED_PYRAMID width=" << width << " height=" << height
                  << " first_octave=" << first << " octaves=" << octaves
                  << " level_min=" << parameters._level_min
                  << " levels_per_octave=" << parameters._level_num
                  << " seed_index="
                  << parameters._level_ds - parameters._level_min
                  << " sigma_skip1=" << parameters._sigma_skip1
                  << " level_count=" << candidate.levels.size()
                  << " input_bytes=" << input.Bytes()
                  << " gpu_cap=" << options.gpu_bytes
                  << " host_cap=" << options.host_bytes
                  << " gpu_staging=" << candidate.stats.gpu_staging_bytes
                  << " host_staging=" << candidate.stats.host_staging_bytes
                  << " file_read=" << candidate.stats.file_read_bytes
                  << " file_written=" << candidate.stats.file_written_bytes
                  << " h2d=" << candidate.stats.h2d_bytes
                  << " d2h=" << candidate.stats.d2h_bytes
                  << " scratch=" << candidate.stats.scratch_bytes
                  << " seconds=" << candidate.stats.seconds << std::endl;
        EXPECT_LE(candidate.stats.gpu_staging_bytes, options.gpu_bytes);
        EXPECT_LE(candidate.stats.host_staging_bytes, options.host_bytes);
        EXPECT_GT(candidate.stats.file_read_bytes, input.Bytes());
        EXPECT_GT(candidate.stats.file_written_bytes, input.Bytes());
        EXPECT_GT(candidate.stats.h2d_bytes, input.Bytes());
        EXPECT_GT(candidate.stats.d2h_bytes, input.Bytes());
        EXPECT_GT(candidate.stats.scratch_bytes, input.Bytes());
        ASSERT_EQ(candidate.levels.size(), octaves * parameters._level_num);
        ASSERT_EQ(candidate.dogs.size(), octaves * (parameters._level_num - 1));
        ASSERT_EQ(candidate.gradients.size(), candidate.dogs.size());
        if (height > 500) {
          ASSERT_GT(candidate.levels.front()->Bytes(), options.gpu_bytes);
          ASSERT_GT(candidate.levels.front()->Bytes(), options.host_bytes);
          EXPECT_GT(candidate.levels.front()->BytesRead(), 0);
        }
        for (int octave = 0; octave < octaves; ++octave) {
          CuTexImage* expected = resident.GetBaseLevel(first + octave);
          for (int level = 0; level < parameters._level_num; ++level) {
            SCOPED_TRACE(::testing::Message()
                         << width << "x" << height << " first=" << first
                         << " octave=" << octave << " level=" << level);
            auto& plane =
                *candidate.levels[octave * parameters._level_num + level];
            ASSERT_EQ(plane.Width(), expected[level].GetImgWidth());
            ASSERT_EQ(plane.Height(), expected[level].GetImgHeight());
            ASSERT_TRUE(plane.Complete());
            std::vector<float> oracle(plane.Bytes() / sizeof(float)),
                actual(oracle.size());
            ASSERT_EQ(expected[level].CopyToHost(oracle.data()), cudaSuccess);
            plane.ReadRows(0, plane.Height(), actual.data());
            EXPECT_EQ(std::memcmp(oracle.data(), actual.data(), plane.Bytes()),
                      0);
            if (level == 0) continue;
            const size_t index =
                octave * (parameters._level_num - 1) + level - 1;
            ASSERT_LT(index, candidate.dogs.size());
            ASSERT_LT(index, candidate.gradients.size());
            CuTexImage old_dog, old_gradient;
            ASSERT_TRUE(old_dog.InitTexture(plane.Width(), plane.Height()));
            ASSERT_TRUE(
                old_gradient.InitTexture(plane.Width(), plane.Height(), 2));
            // Independent resident old-control call, with the original pair
            // pointer and unchanged kernel dispatch, not a streamed oracle.
            ProgramCU::ComputeDOG(expected + level, &old_dog, &old_gradient);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            for (bool gradient : {false, true}) {
              auto& output = gradient ? *candidate.gradients[index]
                                      : *candidate.dogs[index];
              ASSERT_EQ(output.Width(), plane.Width());
              ASSERT_EQ(output.Height(), plane.Height());
              ASSERT_EQ(output.Channels(), gradient ? 2 : 1);
              ASSERT_TRUE(output.Complete());
              oracle.resize(output.Bytes() / sizeof(float));
              actual.resize(oracle.size());
              ASSERT_EQ(
                  (gradient ? old_gradient : old_dog).CopyToHost(oracle.data()),
                  cudaSuccess);
              output.ReadRows(0, output.Height(), actual.data());
              EXPECT_EQ(
                  std::memcmp(oracle.data(), actual.data(), output.Bytes()), 0);
            }
          }
        }
        const auto reads_before = input.BytesRead();
#if defined(__linux__)
        const auto open_files = [] {
          return std::distance(
              std::filesystem::directory_iterator("/proc/self/fd"),
              std::filesystem::directory_iterator());
        };
        const auto before_cancel = open_files();
#endif
        options.cancelled = [&] { return input.BytesRead() > reads_before; };
        EXPECT_THROW(PyramidCU::BuildGaussianPyramidStreamed(
                         parameters, input, first, octaves, options),
                     std::runtime_error);
#if defined(__linux__)
        EXPECT_EQ(open_files(), before_cancel);
#endif
        options.cancelled = {};
        if (configuration == 2 && first == 0) {
          size_t first_octave_checks = 0;
          {
            options.cancelled = [&] {
              ++first_octave_checks;
              return false;
            };
            auto first_only = PyramidCU::BuildGaussianPyramidStreamed(
                parameters, input, first, 1, options);
            ASSERT_EQ(first_only.levels.size(), parameters._level_num);
          }
#if defined(__linux__)
          const auto before_late_cancel = open_files();
#endif
          // A complete one-octave call includes its final cancellation check.
          // Advancing eight more checks in the multi-octave call necessarily
          // reaches second-octave sampling with all first-octave files
          // retained.
          size_t checks = 0;
          options.cancelled = [&] {
            return ++checks > first_octave_checks + 8;
          };
          EXPECT_THROW(PyramidCU::BuildGaussianPyramidStreamed(
                           parameters, input, first, octaves, options),
                       std::runtime_error);
          EXPECT_GT(checks, first_octave_checks + 8);
#if defined(__linux__)
          EXPECT_EQ(open_files(), before_late_cancel);
#endif
          // The same measured Gaussian call count puts cancellation inside
          // the subsequent DoG phase, with every Gaussian backing retained.
          checks = 0;
          options.cancelled = [&] {
            return ++checks > first_octave_checks + 8;
          };
          EXPECT_THROW(PyramidCU::BuildPyramidStreamed(
                           parameters, input, first, 1, options),
                       std::runtime_error);
          EXPECT_GT(checks, first_octave_checks + 8);
#if defined(__linux__)
          EXPECT_EQ(open_files(), before_late_cancel);
#endif
          options.cancelled = {};
        }
        const auto reads_after_cancel = input.BytesRead();
        options.host_bytes = 1;
        // Upsampling validates SamplePlan before the first filter; octave
        // zero validates StripePlan first. Both must refuse before input I/O.
        try {
          PyramidCU::BuildGaussianPyramidStreamed(
              parameters, input, first, octaves, options);
          FAIL() << "Expected insufficient host capacity to be rejected";
        } catch (const std::invalid_argument& error) {
          EXPECT_EQ(first, 0);
          EXPECT_STREQ(error.what(), "SIFT streaming dimensions/capacity");
        } catch (const std::runtime_error& error) {
          EXPECT_EQ(first, -1);
          EXPECT_STREQ(
              error.what(),
              "SIFT streamed sampling needs source rows and one output row");
        }
        EXPECT_EQ(input.BytesRead(), reads_after_cancel);
      }
    }
  }
}

TEST(ExtractSiftFeaturesGPU, StreamedDetectionPyramidMatchesAllResidentLevels) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA or HIP device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  class ResidentPreorientation : public PyramidCU {
   public:
    using PyramidCU::PyramidCU;
    void GetSimplifiedOrientation() override {}
    void DownloadKeypoints() override {}
    void GetFeatureDescriptors() override {}
  };
  struct RestoreDetectionGlobals {
    int threshold = GlobalUtil::_FeatureCountThreshold;
    int list_gpu = GlobalUtil::_ListGenGPU;
    int orientations = GlobalUtil::_MaxOrientation;
    int descriptors = GlobalUtil::_DescriptorPPT;
    int first = GlobalUtil::_octave_min_default;
    int count = GlobalUtil::_octave_num_default;
    int tight = GlobalUtil::_ForceTightPyramid;
    int fit = GlobalUtil::_FitMemoryCap;
    int truncate = GlobalUtil::_TruncateMethod;
    int subpixel = GlobalUtil::_SubpixelLocalization;
    ~RestoreDetectionGlobals() {
      GlobalUtil::_FeatureCountThreshold = threshold;
      GlobalUtil::_ListGenGPU = list_gpu;
      GlobalUtil::_MaxOrientation = orientations;
      GlobalUtil::_DescriptorPPT = descriptors;
      GlobalUtil::_octave_min_default = first;
      GlobalUtil::_octave_num_default = count;
      GlobalUtil::_ForceTightPyramid = tight;
      GlobalUtil::_FitMemoryCap = fit;
      GlobalUtil::_TruncateMethod = truncate;
      GlobalUtil::_SubpixelLocalization = subpixel;
    }
  } restore;
  GlobalUtil::_ListGenGPU = 1;
  GlobalUtil::_MaxOrientation = 0;
  GlobalUtil::_DescriptorPPT = 0;
  GlobalUtil::_octave_min_default = 0;
  GlobalUtil::_octave_num_default = 3;
  GlobalUtil::_ForceTightPyramid = 1;
  GlobalUtil::_FitMemoryCap = 0;
  for (int width : {17, 32, 129}) {
    const int height = width == 129 ? 257 : 65;
    std::vector<float> values(size_t(width) * height);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = float((i * 7919 + i / width * 31) % 65521) / 65521.0f;
    if (width == 129)
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
          const float dx = float(x % 16) - 7.2f;
          const float dy = float(y % 16) - 7.3f;
          values[size_t(y) * width + x] = std::exp(-(dx * dx + dy * dy) / 8.0f);
        }
    siftgpu::StreamedPlane input(".", width, height);
    input.WriteRows(0, height, values.data());
    input.Seal();
    SiftParam parameters;
    parameters.ParseSiftParam();
    std::unique_ptr<float[]> sigma_owner(parameters._sigma);
    GLTexInput resident_input;
    resident_input.SetImageSize(width, height);
    resident_input._pixel_data = values.data();
    ResidentPreorientation resident(parameters);
    resident.InitPyramid(width, height);
    resident.BuildPyramid(&resident_input);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    siftgpu::StreamFilterOptions options;
    options.scratch_directory = ".";
    options.gpu_bytes = 2 * 40 * width * sizeof(float) + 132;
    options.host_bytes = 40 * width * sizeof(float) + 132;
    for (int reverse : {0, 1}) {
      GlobalUtil::_TruncateMethod = reverse;
      GlobalUtil::_SubpixelLocalization = reverse;
      auto candidate = PyramidCU::BuildDetectionPyramidStreamed(
          parameters, input, 0, 3, options);
      ASSERT_EQ(candidate.detections.size(), 3 * parameters._dog_level_num);
      for (size_t i = 0; i < candidate.detections.size(); ++i) {
        auto& record = candidate.detections[i];
        const int oi = int(i) / parameters._dog_level_num;
        const int li = int(i) % parameters._dog_level_num;
        EXPECT_EQ(record.octave, reverse ? 2 - oi : oi);
        EXPECT_EQ(record.level,
                  2 + (reverse ? parameters._dog_level_num - 1 - li : li));
        auto* gaussian = resident.GetBaseLevel(record.octave);
        // Construct all three resident DoGs independently, retaining the
        // original Gaussian pointer relation and native kernel dispatch.
        CuTexImage dogs[3], gradient, key;
        const int w = gaussian->GetImgWidth(), h = gaussian->GetImgHeight();
        ASSERT_TRUE(gradient.InitTexture(w, h, 2));
        ASSERT_TRUE(key.InitTexture(w, h, 4));
        for (int j = 0; j < 3; ++j) {
          ASSERT_TRUE(dogs[j].InitTexture(w, h));
          ProgramCU::ComputeDOG(
              gaussian + record.level - 1 + j, dogs + j, &gradient);
        }
        ProgramCU::ComputeKEY(dogs + 1,
                              &key,
                              parameters._dog_threshold,
                              parameters._edge_threshold);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<float> oracle(size_t(w) * h * 4), actual(oracle.size());
        ASSERT_EQ(key.CopyToHost(oracle.data()), cudaSuccess);
        record.detection.key->ReadRows(0, h, actual.data());
        for (int y = 1; y < h - 1; ++y)
          for (int x = 1; x < w - 1; ++x) {
            const size_t offset = (size_t(y) * w + x) * 4;
            EXPECT_EQ(
                std::memcmp(oracle.data() + offset, actual.data() + offset, 16),
                0);
          }
        CuTexImage source;
        CuTexImage* previous = &key;
        for (auto& plane : record.detection.histograms) {
          CuTexImage output;
          ASSERT_TRUE(output.InitTexture(plane->Width(), h, 4));
          if (previous == &key)
            ProgramCU::InitHistogram(previous, &output);
          else
            ProgramCU::ReduceHistogram(previous, &output);
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
          oracle.resize(plane->Bytes() / sizeof(float));
          actual.resize(oracle.size());
          ASSERT_EQ(output.CopyToHost(oracle.data()), cudaSuccess);
          plane->ReadRows(0, h, actual.data());
          EXPECT_EQ(std::memcmp(oracle.data(), actual.data(), plane->Bytes()),
                    0);
          ASSERT_TRUE(source.InitTexture(plane->Width(), h, 4));
          ASSERT_EQ(source.CopyFromHost(oracle.data()), cudaSuccess);
          previous = &source;
        }
      }
      for (int method : {0, 1, 2}) {
        for (int threshold : {0, 1, 3, 100000}) {
          GlobalUtil::_TruncateMethod = method;
          GlobalUtil::_FeatureCountThreshold = threshold;
          resident.RunSIFT(
              &resident_input);  // Actual FIRST LimitFeatureCount(0).
          auto lists = PyramidCU::GenerateFeatureListsStreamed(
              candidate, method, threshold, options);
          EXPECT_EQ(lists.count, resident.GetFeatureNum());
          if (width == 129 && method == 0 && threshold == 0)
            EXPECT_GT(lists.count, 0);
          const int* counts = resident.GetLevelFeatureNum();
          for (size_t index = 0; index < lists.level_counts.size(); ++index)
            EXPECT_EQ(lists.level_counts[index], counts[index]);
        }
      }
    }
  }
}

TEST(ExtractSiftFeaturesGPU,
     StreamedKeyHistogramMatchesResidentAndIgnoresBorders) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA or HIP device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  struct RestoreSubpixel {
    int value = GlobalUtil::_SubpixelLocalization;
    ~RestoreSubpixel() { GlobalUtil::_SubpixelLocalization = value; }
  } restore;
  for (auto shape : {std::pair<int, int>{129, 513},
                     {128, 517},
                     {17, 65},
                     {7, 2},
                     {3, 5},
                     {1, 1},
                     {129, 513}}) {
    const int width = shape.first, height = shape.second;
    CuTexImage dogs[3], key;
    std::vector<std::unique_ptr<siftgpu::StreamedPlane>> input;
    std::vector<float> values(size_t(width) * height);
    for (int level = 0; level < 3; ++level) {
      // Isolated, slightly off-center extrema with nonzero Hessian and scale
      // curvature. Real KEY localization must pass, not an all-zero oracle.
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
          const float dx = float(x % 8) - 3.2f;
          const float dy = float(y % 8) - 3.3f;
          const float ds = float(level) - 1.1f;
          values[size_t(y) * width + x] =
              std::exp(-(dx * dx + dy * dy + ds * ds));
        }
      ASSERT_TRUE(dogs[level].InitTexture(width, height));
      ASSERT_EQ(dogs[level].CopyFromHost(values.data()), cudaSuccess);
      input.push_back(
          std::make_unique<siftgpu::StreamedPlane>(".", width, height));
      input.back()->WriteRows(0, height, values.data());
      input.back()->Seal();
    }
    ASSERT_TRUE(key.InitTexture(width, height, 4));
    for (int subpixel : {0, 1}) {
      GlobalUtil::_SubpixelLocalization = subpixel;
      for (float threshold : {0.01f, 0.8f, 1.0f}) {
        for (int rows : {1, 3, 7, 1}) {
          SCOPED_TRACE(::testing::Message()
                       << width << "x" << height << " rows=" << rows
                       << " subpixel=" << subpixel
                       << " threshold=" << threshold);
          siftgpu::StreamFilterOptions options;
          options.scratch_directory = ".";
          const size_t staging = std::min(height, rows + 2);
          options.gpu_bytes = std::max(7 * staging * width * sizeof(float),
                                       16 * (size_t(width) + (width + 3) / 4)) +
                              132;
          options.host_bytes = 4 * staging * width * sizeof(float);
          if (height > 500) {
            EXPECT_LT(options.gpu_bytes, input[0]->Bytes());
            EXPECT_LT(options.host_bytes, input[0]->Bytes());
          }
          uint64_t before = 0;
          for (auto& plane : input) before += plane->BytesRead();
          auto candidate = ProgramCU::ComputeKEYStreamed(
              *input[0], *input[1], *input[2], threshold, 10.0f, true, options);
          EXPECT_LE(candidate.stats.gpu_staging_bytes, options.gpu_bytes);
          EXPECT_LE(candidate.stats.host_staging_bytes, options.host_bytes);
          uint64_t after = 0, written = candidate.key->Bytes();
          for (auto& plane : input) after += plane->BytesRead();
          uint64_t hist_reads = candidate.key->Bytes();
          for (size_t i = 0; i < candidate.histograms.size(); ++i) {
            written += candidate.histograms[i]->Bytes();
            if (i + 1 < candidate.histograms.size())
              hist_reads += candidate.histograms[i]->Bytes();
          }
          EXPECT_EQ(candidate.stats.file_read_bytes,
                    after - before + hist_reads);
          EXPECT_EQ(candidate.stats.h2d_bytes, candidate.stats.file_read_bytes);
          EXPECT_EQ(candidate.stats.file_written_bytes, written);
          EXPECT_EQ(candidate.stats.scratch_bytes, written);
          EXPECT_EQ(
              candidate.stats.d2h_bytes,
              (after - before) / 3 * 4 + written - candidate.key->Bytes());
          std::cout << "STREAMED_KEY_HIST width=" << width
                    << " height=" << height << " rows=" << rows
                    << " gpu_cap=" << options.gpu_bytes
                    << " host_cap=" << options.host_bytes
                    << " gpu_staging=" << candidate.stats.gpu_staging_bytes
                    << " host_staging=" << candidate.stats.host_staging_bytes
                    << " file_read=" << candidate.stats.file_read_bytes
                    << " file_written=" << candidate.stats.file_written_bytes
                    << " h2d=" << candidate.stats.h2d_bytes
                    << " d2h=" << candidate.stats.d2h_bytes << std::endl;
          std::vector<float> actual(candidate.key->Bytes() / sizeof(float));
          candidate.key->ReadRows(0, height, actual.data());
          // Poison the entire resident scratch twice: border leftovers must
          // not enter any histogram channel (including odd-width tails).
          for (float poison : {19.0f, -37.0f}) {
            std::vector<float> oracle(actual.size(), poison);
            ASSERT_EQ(key.CopyFromHost(oracle.data()), cudaSuccess);
            ProgramCU::ComputeKEY(dogs + 1, &key, threshold, 10.0f);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_EQ(key.CopyToHost(oracle.data()), cudaSuccess);
            int features = 0;
            for (int y = 1; y < height - 1; ++y)
              for (int x = 1; x < width - 1; ++x) {
                const size_t offset = (size_t(y) * width + x) * 4;
                EXPECT_EQ(
                    std::memcmp(
                        oracle.data() + offset, actual.data() + offset, 16),
                    0);
                features += oracle[offset] != 0;
              }
            if (width > 8 && height > 8 && threshold == 0.01f)
              EXPECT_GT(features, 0);
            CuTexImage source;
            CuTexImage* previous = &key;
            int expected_width = (width + 3) / 4;
            for (auto& plane : candidate.histograms) {
              EXPECT_EQ(plane->Width(), expected_width);
              EXPECT_EQ(plane->Height(), height);
              CuTexImage output;
              ASSERT_TRUE(output.InitTexture(expected_width, height, 4));
              if (previous == &key)
                ProgramCU::InitHistogram(previous, &output);
              else
                ProgramCU::ReduceHistogram(previous, &output);
              ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
              std::vector<float> expected(plane->Bytes() / sizeof(float)),
                  got(expected.size());
              ASSERT_EQ(output.CopyToHost(expected.data()), cudaSuccess);
              plane->ReadRows(0, height, got.data());
              EXPECT_EQ(
                  std::memcmp(expected.data(), got.data(), plane->Bytes()), 0);
              // CuTexImage is not movable: transfer the independent resident
              // oracle through host storage between reductions, not candidate.
              ASSERT_TRUE(source.InitTexture(expected_width, height, 4));
              ASSERT_EQ(source.CopyFromHost(expected.data()), cudaSuccess);
              previous = &source;
              expected_width = (expected_width + 3) / 4;
            }
            ASSERT_EQ(candidate.histograms.back()->Width(), 1);
            if (poison == 19.0f && rows == 3) {
              // Independent resident seeds and original GenerateList dispatch.
              // Compare every int4 bit, not a sorted/capped feature set.
              std::vector<float> seeds;
              auto& root = *candidate.histograms.back();
              std::vector<float> roots(root.Bytes() / 4);
              root.ReadRows(0, height, roots.data());
              for (int y = 0; y < height; ++y) {
                int counts[4];
                std::memcpy(counts, roots.data() + y * 4, 16);
                for (int x = 0; x < 4; ++x)
                  for (int rank = 0; rank < counts[x]; ++rank) {
                    int seed[4] = {x, y, rank, 0};
                    float bits[4];
                    std::memcpy(bits, seed, 16);
                    seeds.insert(seeds.end(), bits, bits + 4);
                  }
              }
              siftgpu::StreamFilterOptions list_options;
              list_options.scratch_directory = ".";
              list_options.gpu_bytes = 132 + 32 * 7;
              list_options.host_bytes = 64 + 32 * 7;
              uint64_t reads_before_list = 0;
              for (auto& plane : candidate.histograms)
                reads_before_list += plane->BytesRead();
              auto list =
                  ProgramCU::GenerateListStreamed(candidate, list_options);
              uint64_t reads_after_list = 0;
              for (auto& plane : candidate.histograms)
                reads_after_list += plane->BytesRead();
              EXPECT_EQ(list.stats.file_read_bytes,
                        reads_after_list - reads_before_list);
              ASSERT_EQ(list.count, seeds.size() / 4);
              EXPECT_EQ(list.count, features);
              EXPECT_LE(list.stats.gpu_staging_bytes, list_options.gpu_bytes);
              EXPECT_LE(list.stats.host_staging_bytes, list_options.host_bytes);
              if (list.count) {
                CuTexImage resident_list;
                ASSERT_TRUE(resident_list.InitTexture(list.count, 1, 4));
                ASSERT_EQ(resident_list.CopyFromHost(seeds.data()),
                          cudaSuccess);
                for (int hi = int(candidate.histograms.size()) - 2; hi >= 0;
                     --hi) {
                  auto& plane = *candidate.histograms[hi];
                  CuTexImage hist;
                  ASSERT_TRUE(hist.InitTexture(plane.Width(), height, 4));
                  std::vector<float> values(plane.Bytes() / 4);
                  plane.ReadRows(0, height, values.data());
                  ASSERT_EQ(hist.CopyFromHost(values.data()), cudaSuccess);
                  ProgramCU::GenerateList(&resident_list, &hist);
                  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                }
                ASSERT_EQ(resident_list.CopyToHost(seeds.data()), cudaSuccess);
                std::vector<float> streamed(seeds.size());
                list.records->ReadRows(0, list.count, streamed.data());
                EXPECT_EQ(std::memcmp(
                              seeds.data(), streamed.data(), seeds.size() * 4),
                          0);
                EXPECT_EQ(list.stats.file_written_bytes, list.records->Bytes());
                EXPECT_GT(list.stats.file_read_bytes, 0);
                if (list.count > 32) {
                  EXPECT_GT(list.records->Bytes(), list_options.gpu_bytes);
                  EXPECT_GT(list.records->Bytes(), list_options.host_bytes);
                }
              } else {
                EXPECT_FALSE(list.records);
              }
              auto bad_options = list_options;
              bad_options.host_bytes = 1;
              EXPECT_THROW(
                  ProgramCU::GenerateListStreamed(candidate, bad_options),
                  std::runtime_error);
              if (list.count) {
                bad_options = list_options;
                bad_options.scratch_directory = "/missing-sift-directory";
                EXPECT_THROW(
                    ProgramCU::GenerateListStreamed(candidate, bad_options),
                    std::runtime_error);
                const auto root_reads = root.BytesRead();
                bad_options = list_options;
                bad_options.cancelled = [&] {
                  return root.BytesRead() > root_reads + root.Bytes();
                };
#if defined(__linux__)
                const auto descriptors_before = std::distance(
                    std::filesystem::directory_iterator("/proc/self/fd"),
                    std::filesystem::directory_iterator());
#endif
                EXPECT_THROW(
                    ProgramCU::GenerateListStreamed(candidate, bad_options),
                    std::runtime_error);
#if defined(__linux__)
                EXPECT_EQ(std::distance(std::filesystem::directory_iterator(
                                            "/proc/self/fd"),
                                        std::filesystem::directory_iterator()),
                          descriptors_before);
#endif
              }
              list_options.cancelled = [] { return true; };
              EXPECT_THROW(
                  ProgramCU::GenerateListStreamed(candidate, list_options),
                  std::runtime_error);
            }
          }
          options.cancelled = [] { return true; };
          EXPECT_THROW(ProgramCU::ComputeKEYStreamed(*input[0],
                                                     *input[1],
                                                     *input[2],
                                                     threshold,
                                                     10.0f,
                                                     true,
                                                     options),
                       std::runtime_error);
          options.cancelled = {};
          if (width == 17 && height == 65 && rows == 3 && !subpixel &&
              threshold == 0.01f) {
#if defined(__linux__)
            const auto open_files = [] {
              return std::distance(
                  std::filesystem::directory_iterator("/proc/self/fd"),
                  std::filesystem::directory_iterator());
            };
            const auto descriptors = open_files();
            int checks = 0;
            options.cancelled = [&] { return ++checks > 40; };
            EXPECT_THROW(ProgramCU::ComputeKEYStreamed(*input[0],
                                                       *input[1],
                                                       *input[2],
                                                       threshold,
                                                       10.0f,
                                                       true,
                                                       options),
                         std::runtime_error);
            EXPECT_EQ(open_files(), descriptors);
            options.cancelled = {};
            struct rlimit old_limit, limit;
            ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &old_limit), 0);
            limit = old_limit;
            limit.rlim_cur = 16;
            auto old_signal = std::signal(SIGXFSZ, SIG_IGN);
            ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);
            EXPECT_THROW(ProgramCU::ComputeKEYStreamed(*input[0],
                                                       *input[1],
                                                       *input[2],
                                                       threshold,
                                                       10.0f,
                                                       true,
                                                       options),
                         std::runtime_error);
            EXPECT_EQ(setrlimit(RLIMIT_FSIZE, &old_limit), 0);
            std::signal(SIGXFSZ, old_signal);
            EXPECT_EQ(open_files(), descriptors);
#endif
            options.host_bytes = 4 * 3 * width * sizeof(float) - 1;
            EXPECT_THROW(ProgramCU::ComputeKEYStreamed(*input[0],
                                                       *input[1],
                                                       *input[2],
                                                       threshold,
                                                       10.0f,
                                                       true,
                                                       options),
                         std::runtime_error);
          }
          options.gpu_bytes = 1;
          EXPECT_THROW(ProgramCU::ComputeKEYStreamed(*input[0],
                                                     *input[1],
                                                     *input[2],
                                                     threshold,
                                                     10.0f,
                                                     true,
                                                     options),
                       std::exception);
        }
      }
    }
  }
}

TEST(ExtractSiftFeaturesGPU,
     StreamedDogMatchesIndependentResidentAndFailsClosed) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA or HIP device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  for (auto shape : {std::pair<int, int>{129, 513},
                     {128, 517},
                     {17, 65},
                     {7, 2},
                     {3, 5},
                     {1, 1},
                     {129, 513}}) {
    const int width = shape.first, height = shape.second;
    siftgpu::StreamedPlane previous(".", width, height),
        current(".", width, height);
    CuTexImage resident[2], dog, gradient;
    std::vector<float> values(size_t(width) * height);
    for (int i = 0; i < 2; ++i) {
      for (size_t j = 0; j < values.size(); ++j)
        values[j] =
            float((j * 7919 + j / width * 31 + i * 1237) % 65521) / 65521.0f;
      ASSERT_TRUE(resident[i].InitTexture(width, height));
      ASSERT_EQ(resident[i].CopyFromHost(values.data()), cudaSuccess);
      auto& plane = i ? current : previous;
      plane.WriteRows(0, height, values.data());
      plane.Seal();
    }
    ASSERT_TRUE(dog.InitTexture(width, height));
    ASSERT_TRUE(gradient.InitTexture(width, height, 2));
    ProgramCU::ComputeDOG(resident + 1, &dog, &gradient);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int rows : {1, 3, 7, 1}) {
      SCOPED_TRACE(::testing::Message()
                   << width << "x" << height << " rows=" << rows);
      siftgpu::StreamFilterOptions options;
      options.scratch_directory = ".";
      const size_t staging = std::min(height, rows + 2);
      options.gpu_bytes = 5 * staging * width * sizeof(float) + 132;
      options.host_bytes = 2 * staging * width * sizeof(float);
      const auto before = previous.BytesRead() + current.BytesRead();
      auto result = ProgramCU::ComputeDOGStreamed(current, previous, options);
      EXPECT_THROW(ProgramCU::ComputeDOGStreamed(*result.gradient, previous, options),
                   std::invalid_argument);
      EXPECT_THROW(ProgramCU::FilterImageStreamed(*result.gradient, 1.0f, options),
                   std::invalid_argument);
      EXPECT_THROW(ProgramCU::SampleImageStreamed(*result.gradient, width, height,
                                                false, true, options),
                   std::invalid_argument);
      EXPECT_LE(result.stats.gpu_staging_bytes, options.gpu_bytes);
      EXPECT_LE(result.stats.host_staging_bytes, options.host_bytes);
      EXPECT_EQ(result.stats.file_read_bytes,
                previous.BytesRead() + current.BytesRead() - before);
      EXPECT_EQ(result.stats.h2d_bytes, result.stats.file_read_bytes);
      EXPECT_EQ(result.stats.d2h_bytes, result.stats.file_read_bytes / 2 * 3);
      EXPECT_EQ(result.stats.file_written_bytes, current.Bytes() * 3);
      EXPECT_EQ(result.stats.scratch_bytes, current.Bytes() * 3);
      std::cout << "STREAMED_DOG width=" << width << " height=" << height
                << " requested_rows=" << rows
                << " plane_bytes=" << current.Bytes()
                << " gpu_cap=" << options.gpu_bytes
                << " host_cap=" << options.host_bytes
                << " gpu_staging=" << result.stats.gpu_staging_bytes
                << " host_staging=" << result.stats.host_staging_bytes
                << " file_read=" << result.stats.file_read_bytes
                << " file_written=" << result.stats.file_written_bytes
                << " h2d=" << result.stats.h2d_bytes
                << " d2h=" << result.stats.d2h_bytes
                << " file_seconds=" << result.stats.file_seconds
                << " h2d_seconds=" << result.stats.h2d_seconds
                << " d2h_seconds=" << result.stats.d2h_seconds
                << " kernel_seconds=" << result.stats.kernel_seconds
                << " seconds=" << result.stats.seconds << std::endl;
      if (height > 500) {
        EXPECT_GT(current.Bytes(), options.gpu_bytes);
        EXPECT_GT(current.Bytes(), options.host_bytes);
        EXPECT_GT(result.stats.file_read_bytes, current.Bytes() * 2);
      }
      for (bool grad : {false, true}) {
        auto& plane = grad ? *result.gradient : *result.dog;
        std::vector<float> expected(plane.Bytes() / sizeof(float)),
            actual(expected.size());
        ASSERT_TRUE(plane.Complete());
        ASSERT_EQ((grad ? gradient : dog).CopyToHost(expected.data()),
                  cudaSuccess);
        plane.ReadRows(0, height, actual.data());
        EXPECT_EQ(std::memcmp(expected.data(), actual.data(), plane.Bytes()),
                  0);
      }
#if defined(__linux__)
      const auto open_files = [] {
        return std::distance(
            std::filesystem::directory_iterator("/proc/self/fd"),
            std::filesystem::directory_iterator());
      };
      const auto fds = open_files();
#endif
      const auto reads = current.BytesRead();
      options.cancelled = [&] { return current.BytesRead() > reads; };
      EXPECT_THROW(ProgramCU::ComputeDOGStreamed(current, previous, options),
                   std::runtime_error);
#if defined(__linux__)
      EXPECT_EQ(open_files(), fds);
#endif
      options.cancelled = {};
      options.scratch_directory = "/missing-sift-directory";
      EXPECT_THROW(ProgramCU::ComputeDOGStreamed(current, previous, options),
                   std::runtime_error);
      options.scratch_directory = ".";
      options.host_bytes = 1;
      const auto pre_refusal = current.BytesRead() + previous.BytesRead();
      EXPECT_THROW(ProgramCU::ComputeDOGStreamed(current, previous, options),
                   std::runtime_error);
      EXPECT_EQ(current.BytesRead() + previous.BytesRead(), pre_refusal);
#if defined(__linux__)
      // Actual output-file exhaustion, bounded to one byte; no disk filling.
      // Inputs and resident oracle were sealed before the temporary limit.
      struct RestoreLimit {
        struct rlimit old;
        using Handler = void (*)(int);
        Handler handler;
        RestoreLimit() : handler(std::signal(SIGXFSZ, SIG_IGN)) {
          if (getrlimit(RLIMIT_FSIZE, &old) != 0)
            throw std::runtime_error("getrlimit");
        }
        ~RestoreLimit() {
          setrlimit(RLIMIT_FSIZE, &old);
          std::signal(SIGXFSZ, handler);
        }
      } restore_limit;
      auto limit = restore_limit.old;
      limit.rlim_cur = 1;
      ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);
      options.host_bytes = 2 * staging * width * sizeof(float);
      EXPECT_THROW(ProgramCU::ComputeDOGStreamed(current, previous, options),
                   std::runtime_error);
      EXPECT_EQ(open_files(), fds);
#endif
    }
  }
}

// Phase constraints must refuse before I/O or CUDA allocations, never fall
// back to resident extraction. Unsupported octave values remain unsupported;
// aligned/default -1 is covered by the independent bounded native oracle.
TEST(ExtractSiftFeaturesGPU, StreamedPyramidRefusesUnsupportedDomains) {
  SiftParam parameters;
  parameters.ParseSiftParam();
  std::unique_ptr<float[]> sigma_owner(parameters._sigma);
  siftgpu::StreamedPlane input(".", 6048, 2);
  std::vector<float> data(6048 * 2);
  input.WriteRows(0, 2, data.data());
  input.Seal();
  siftgpu::StreamFilterOptions options;
  options.scratch_directory = "/must-not-create-scratch";
  options.host_bytes = options.gpu_bytes = 1 << 20;
  for (int first : {-2, 1})
    EXPECT_THROW(PyramidCU::BuildGaussianPyramidStreamed(
                     parameters, input, first, 1, options),
                 std::invalid_argument);
  // Legacy h=2 -> 1 -> 0 is not silently repaired into a new domain.
  EXPECT_THROW(
      PyramidCU::BuildGaussianPyramidStreamed(parameters, input, 0, 3, options),
      std::invalid_argument);
  for (int first : {-2, 1})
    EXPECT_THROW(
        PyramidCU::BuildPyramidStreamed(parameters, input, first, 1, options),
        std::invalid_argument);
  EXPECT_THROW(
      PyramidCU::BuildPyramidStreamed(parameters, input, 0, 3, options),
      std::invalid_argument);
  EXPECT_EQ(input.BytesRead(), 0);
}

TEST(ExtractSiftFeaturesGPU, StreamedGaussianMatchesResidentBytes) {
  if (GetNumCudaDevices() == 0) GTEST_SKIP() << "No CUDA or HIP device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  // The resident oracle is deliberately outside the streaming staging caps.
  // Rounded-domain callers pass the full rounded row, including its padding;
  // exact-domain callers pass the exact row. Filtering itself does not round.
  for (auto shape : {std::pair<int, int>{129, 257},
                     {128, 311},
                     {17, 67},
                     {7, 2},
                     {1, 1},
                     {129, 257}}) {
    const int width = shape.first, height = shape.second;
    std::vector<float> values(width * height), horizontal(values.size()),
        final(values.size()), candidate(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = float((i * 7919 + i / width * 31) % 65521) / 65521.0f;
    }
    siftgpu::StreamedPlane input(".", width, height);
    input.WriteRows(0, height, values.data());
    input.Seal();
    CuTexImage src, buf, dst;
    ASSERT_TRUE(src.InitTexture(width, height));
    ASSERT_TRUE(buf.InitTexture(width, height));
    ASSERT_TRUE(dst.InitTexture(width, height));
    ASSERT_EQ(src.CopyFromHost(values.data()), cudaSuccess);
    for (int fw = 5; fw <= 33; fw += 2) {
      SCOPED_TRACE(::testing::Message()
                   << width << "x" << height << " FW=" << fw);
      const float sigma = (fw - 1.5f) / (2 * GlobalUtil::_FilterWidthFactor);
      float coefficients[33];
      int actual_width;
      ProgramCU::CreateFilterKernel(sigma, coefficients, actual_width);
      ASSERT_EQ(actual_width, fw);
      ProgramCU::FilterImage(&dst, &src, &buf, sigma);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      ASSERT_EQ(buf.CopyToHost(horizontal.data()), cudaSuccess);
      ASSERT_EQ(dst.CopyToHost(final.data()), cudaSuccess);
      const size_t stripe_rows = std::min(height, fw - 1 + 7);
      const size_t payload = stripe_rows * width * sizeof(float);
      siftgpu::StreamFilterOptions options;
      options.scratch_directory = ".";
      options.gpu_bytes = 2 * payload + 132;
      options.host_bytes = payload + 132;
      if (height > 100) {
        ASSERT_GT(input.Bytes(), options.gpu_bytes);
        ASSERT_GT(input.Bytes(), options.host_bytes);
      }
      auto result = ProgramCU::FilterImageStreamed(input, sigma, options);
      // Component payload accounting, not aggregate host/GPU high-water proof.
      std::cout << "STREAMED_GAUSSIAN width=" << width << " height=" << height
                << " filter_width=" << fw << " plane_bytes=" << input.Bytes()
                << " gpu_cap=" << options.gpu_bytes
                << " host_cap=" << options.host_bytes
                << " gpu_staging=" << result.stats.gpu_staging_bytes
                << " host_staging=" << result.stats.host_staging_bytes
                << " file_read=" << result.stats.file_read_bytes
                << " file_written=" << result.stats.file_written_bytes
                << " h2d=" << result.stats.h2d_bytes
                << " d2h=" << result.stats.d2h_bytes
                << " scratch=" << result.stats.scratch_bytes
                << " seconds=" << result.stats.seconds << std::endl;
      EXPECT_LE(result.stats.gpu_staging_bytes, options.gpu_bytes);
      EXPECT_LE(result.stats.host_staging_bytes, options.host_bytes);
      EXPECT_EQ(result.stats.file_written_bytes, 2 * input.Bytes());
      EXPECT_GE(result.stats.file_read_bytes, 2 * input.Bytes());
      if (height > 100) {
        EXPECT_GT(result.stats.file_read_bytes, 2 * input.Bytes());
        EXPECT_GT(result.output->Bytes(), options.gpu_bytes);
        EXPECT_GT(result.output->Bytes(), options.host_bytes);
      }
      EXPECT_GE(result.stats.h2d_bytes, 2 * input.Bytes());
      EXPECT_GE(result.stats.d2h_bytes, 2 * input.Bytes());
      ASSERT_TRUE(result.horizontal->Complete());
      ASSERT_TRUE(result.output->Complete());
      result.horizontal->ReadRows(0, height, candidate.data());
      EXPECT_EQ(std::memcmp(horizontal.data(), candidate.data(), input.Bytes()),
                0);
      result.output->ReadRows(0, height, candidate.data());
      EXPECT_EQ(std::memcmp(final.data(), candidate.data(), input.Bytes()), 0);
      // Cancel after input stripes have been consumed. No result is
      // returned and the API must synchronously drain before freeing textures.
      const auto reads_before = input.BytesRead();
      options.cancelled = [&] {
        return input.BytesRead() - reads_before >= input.Bytes();
      };
      EXPECT_THROW(ProgramCU::FilterImageStreamed(input, sigma, options),
                   std::runtime_error);
    }
  }
}
#endif

TEST(ExtractSiftFeaturesGPU,
     DenseKeyRegenerationMatchesRetainedOriginalPlanes) {
  if (GetNumCudaDevices() == 0) {
    GTEST_SKIP() << "No CUDA or HIP devices available";
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  const int old_subpixel = GlobalUtil::_SubpixelLocalization;
  const int old_orientations = GlobalUtil::_MaxOrientation;
  const int old_fixed = GlobalUtil::_FixedOrientation;
  // Restore globals even after a fatal assertion.
  struct Restore {
    int subpixel, orientations, fixed;
    ~Restore() {
      GlobalUtil::_SubpixelLocalization = subpixel;
      GlobalUtil::_MaxOrientation = orientations;
      GlobalUtil::_FixedOrientation = fixed;
    }
  } restore{old_subpixel, old_orientations, old_fixed};
  GlobalUtil::_FixedOrientation = 0;
  bool multiple_orientations_seen = false;
  CuTexImage shared;
  ASSERT_TRUE(shared.InitTexture(132, 98, 4));
  for (auto shape : {std::pair<int, int>{131, 97},
                     {65, 49},
                     {132, 98},
                     {64, 48},
                     {131, 97}}) {
    const int width = shape.first, height = shape.second;
    SCOPED_TRACE(::testing::Message() << width << "x" << height);
    CuTexImage dogs[6], gradients[3];
    for (int d = 1; d < 6; ++d) {
      ASSERT_TRUE(dogs[d].InitTexture(width, height));
      std::vector<float> values(width * height, 0.0f);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          for (int j = 0; j < 3; ++j) {
            const float dx = x - (12.25f + j * 16.0f);
            const float dy = y - (height / 2 - 7.0f + 0.125f);
            const float ds = d - (j + 2.25f);
            values[y * width + x] +=
                0.5f *
                (std::exp(-(dx * dx + dy * dy) / 8.0f - ds * ds) +
                 std::exp(-(dx * dx + (dy - 10) * (dy - 10)) / 8.0f - ds * ds));
          }
        }
      }
      dogs[d].CopyFromHost(values.data());
    }
    for (auto& gradient : gradients) {
      ASSERT_TRUE(gradient.InitTexture(width, height, 2));
      std::vector<float> values(width * height * 2);
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          values[(y * width + x) * 2] = 1.0f;
          // Opposing lobes exercise packed multiple orientations.
          values[(y * width + x) * 2 + 1] = x % 2 ? 0.0f : 3.14159265f;
        }
      }
      gradient.CopyFromHost(values.data());
    }
    for (int subpixel : {0, 1}) {
      GlobalUtil::_SubpixelLocalization = subpixel;
      for (int orientations : {1, 2}) {
        GlobalUtil::_MaxOrientation = orientations;
        for (float threshold : {0.02f, 100.0f}) {
          CuTexImage original[3], lists[3];
          int counts[3] = {};
          // Independent old sequence: compute and retain all dense keys first.
          for (int j = 0; j < 3; ++j) {
            ASSERT_TRUE(original[j].InitTexture(width, height, 4));
            ProgramCU::ComputeKEY(dogs + j + 2, original + j, threshold, 10.0f);
          }
          for (int j = 0; j < 3; ++j) {
            ASSERT_NO_FATAL_FAILURE(
                BuildDenseKeyList(original + j, lists + j, counts + j));
            if (threshold < 1.0f) {
              ASSERT_GT(counts[j], 1);
            } else {
              ASSERT_EQ(counts[j], 0);
            }
          }
          // Reverse list traversal, then a separate forward orientation pass:
          // the last generated plane cannot accidentally satisfy both passes.
          CuTexImage regenerated_lists[3];
          shared.SetImageSize(width, height);
          for (int j = 2; j >= 0; --j) {
            ProgramCU::ComputeKEY(dogs + j + 2, &shared, threshold, 10.0f);
            std::vector<uint32_t> old_bits(width * height * 4);
            std::vector<uint32_t> new_bits(old_bits.size());
            original[j].CopyToHost(old_bits.data());
            shared.CopyToHost(new_bits.data());
            // ComputeKEY intentionally does not write the masked border.
            for (int y = 1; y < height - 1; ++y) {
              for (int x = 1; x < width - 1; ++x) {
                for (int k = 0; k < 4; ++k) {
                  EXPECT_EQ(old_bits[(y * width + x) * 4 + k],
                            new_bits[(y * width + x) * 4 + k]);
                }
              }
            }
            CuTexImage old_display, new_display;
            ASSERT_TRUE(old_display.InitTexture(width, height, 4));
            ASSERT_TRUE(new_display.InitTexture(width, height, 4));
            ProgramCU::DisplayConvertKEY(
                original + j, dogs + j + 2, &old_display);
            ProgramCU::DisplayConvertKEY(&shared, dogs + j + 2, &new_display);
            ExpectNativeFloatBits(
                &old_display, &new_display, width * height * 4);
            int count = -1;
            ASSERT_NO_FATAL_FAILURE(
                BuildDenseKeyList(&shared, regenerated_lists + j, &count));
            ASSERT_EQ(count, counts[j]);
            ExpectNativeFloatBits(lists + j, regenerated_lists + j, count * 4);
          }
          for (int j = 0; j < 3; ++j) {
            if (counts[j] == 0) continue;
            // Counts straddling a limiting boundary (including zero) retain
            // the identical prefix and ordering, not a sorted feature set.
            for (int cap : {0, 1, counts[j]}) {
              CuTexImage old_list, new_list;
              ASSERT_TRUE(old_list.InitTexture(counts[j], 1, 4));
              ASSERT_TRUE(new_list.InitTexture(counts[j], 1, 4));
              std::vector<uint32_t> bits(counts[j] * 4);
              lists[j].CopyToHost(bits.data());
              old_list.CopyFromHost(bits.data());
              regenerated_lists[j].CopyToHost(bits.data());
              new_list.CopyFromHost(bits.data());
              old_list.SetImageSize(cap, 1);
              new_list.SetImageSize(cap, 1);
              ProgramCU::ComputeOrientation(
                  &old_list, gradients + j, original + j, 1.6f, 1.259921f, 0);
              ProgramCU::ComputeKEY(dogs + j + 2, &shared, threshold, 10.0f);
              ProgramCU::ComputeOrientation(
                  &new_list, gradients + j, &shared, 1.6f, 1.259921f, 0);
              ExpectNativeFloatBits(&old_list, &new_list, cap * 4);
              if (cap > 0 && orientations == 2) {
                std::vector<uint32_t> oriented(cap * 4);
                old_list.CopyToHost(oriented.data());
                for (int n = 0; n < cap; ++n) {
                  const uint32_t packed = oriented[n * 4 + 3];
                  multiple_orientations_seen |=
                      (packed >> 16) != 65535 && (packed & 65535) != 65535 &&
                      (packed >> 16) != (packed & 65535);
                }
              }
              if (cap > 0 && orientations == 1) {
                CuTexImage old_descriptors, new_descriptors;
                ProgramCU::ComputeDescriptor(
                    &old_list, gradients + j, &old_descriptors);
                ProgramCU::ComputeDescriptor(
                    &new_list, gradients + j, &new_descriptors);
                ExpectNativeFloatBits(
                    &old_descriptors, &new_descriptors, cap * 128);
              }
            }
          }
          ASSERT_EQ(cudaGetLastError(), cudaSuccess);
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
      }
    }
  }
  EXPECT_TRUE(multiple_orientations_seen);
}

TEST(ExtractSiftFeaturesGPU, ArbitraryWidthsAndReuse) {
  RunGpuTest([] {
    FeatureExtractionOptions options(FeatureExtractorType::SIFT);
    options.use_gpu = true;
    options.gpu_index = "0";
    auto reused = CreateSiftFeatureExtractor(options);
    ASSERT_NE(reused, nullptr);
    // Repeated and descending sizes exercise FitPyramid as well as allocation,
    // with exact-domain -> legacy-domain -> exact-domain transitions.
    for (int octave : {-1, 0, 1}) {
      options.sift->first_octave = octave;
      reused = CreateSiftFeatureExtractor(options);
      for (int width : {259, 260, 257, 258, 260, 257, 1, 2, 3, 4, 5}) {
        for (bool rotated : {false, true}) {
          for (bool square : {false, true}) {
            if (square && width < 257) continue;
            SCOPED_TRACE(::testing::Message()
                         << "width=" << width << " rotated=" << rotated
                         << " first_octave=" << octave
                         << " fixture=" << (square ? "square" : "checker"));
            const int height = width < 6 ? 3 : 129;
            Bitmap bitmap(width, height, false);
            if (square) {
              // Match Nominal's 64-pixel square on a black background.
              // Unlike the fine checker, it has structure at coarse octaves.
              bitmap.Fill(BitmapColor<uint8_t>(0));
              for (int y = height / 2 - 32; y < height / 2 + 32; ++y) {
                for (int x = width / 2 - 32; x < width / 2 + 32; ++x) {
                  bitmap.SetPixel(x, y, BitmapColor<uint8_t>(255));
                }
              }
            } else {
              for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                  bitmap.SetPixel(x, y, BitmapColor<uint8_t>(
                      ((x / 5 + y / 7) % 2) * 255));
                }
              }
            }
            if (rotated) bitmap.Rot90(1);
            FeatureKeypoints keypoints, control_keypoints;
            FeatureDescriptors descriptors, control_descriptors;
            auto fresh = CreateSiftFeatureExtractor(options);
            ASSERT_NE(fresh, nullptr);
            ASSERT_TRUE(reused->Extract(bitmap, &keypoints, &descriptors));
            ASSERT_TRUE(fresh->Extract(bitmap, &control_keypoints, &control_descriptors));
            ValidateKeypoints(keypoints, bitmap);
            EXPECT_EQ(keypoints, control_keypoints);
            ASSERT_EQ(descriptors.data.rows(), control_descriptors.data.rows());
            EXPECT_TRUE((descriptors.data.array() == control_descriptors.data.array()).all());
            // The checker remains a reuse/immutability regression even when
            // smoothing and scale-space extrema rejection leave no features.
            if (square) EXPECT_FALSE(keypoints.empty());
          }
        }
      }
    }
  });
}
#endif

FeatureDescriptors CreateRandomFeatureDescriptors(const size_t num_features) {
  SetPRNGSeed(0);
  FeatureDescriptorsFloatData descriptors_float =
      FeatureDescriptorsFloatData::Zero(num_features, 128);
  std::vector<int> dims(128);
  std::iota(dims.begin(), dims.end(), 0);
  for (size_t i = 0; i < num_features; ++i) {
    std::shuffle(dims.begin(), dims.end(), *PRNG);
    for (size_t j = 0; j < 10; ++j) {
      descriptors_float(i, dims[j]) = 1.0f;
    }
  }
  L2NormalizeFeatureDescriptors(&descriptors_float);
  return FeatureDescriptors(
      FeatureExtractorType::SIFT,
      FeatureDescriptorsToUnsignedByte(descriptors_float));
}

void CheckEqualMatches(const FeatureMatches& matches1,
                       const FeatureMatches& matches2) {
  ASSERT_EQ(matches1.size(), matches2.size());
  for (size_t i = 0; i < matches1.size(); ++i) {
    EXPECT_EQ(matches1[i].point2D_idx1, matches2[i].point2D_idx1);
    EXPECT_EQ(matches1[i].point2D_idx2, matches2[i].point2D_idx2);
  }
}

TEST(CreateSiftGPUMatcherOpenGL, Nominal) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    EXPECT_NE(CreateSiftFeatureMatcher(options), nullptr);
  });
}

TEST(CreateSiftGPUMatcherCompute, Nominal) {
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = true;
  options.gpu_index = "0";
  options.max_num_matches = 1000;
  EXPECT_NE(CreateSiftFeatureMatcher(options), nullptr);
#endif
}

struct FeatureDescriptorIndexCacheHelper {
  explicit FeatureDescriptorIndexCacheHelper(
      const std::vector<FeatureMatcher::Image>& images)
      : index_cache(100, [this](const image_t image_id) {
          auto index = FeatureDescriptorIndex::Create();
          const auto& desc = this->image_descriptors_.at(image_id);
          index->Build(desc->ToFloat());
          return index;
        }) {
    for (const auto& image : images) {
      image_descriptors_.emplace(image.image_id, image.descriptors);
    }
  }

  ThreadSafeLRUCache<image_t, FeatureDescriptorIndex> index_cache;

 private:
  std::map<image_t, std::shared_ptr<const FeatureDescriptors>>
      image_descriptors_;
};

TEST(SiftCPUFeatureMatcher, Nominal) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image0, image1, image2});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = false;
  options.sift->cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  FeatureMatches matches;
  matcher->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);

  matcher->Match(image1, image2, &matches);
  ExpectReversedMatches(matches);

  matcher->Match(image0, image2, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image1, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
  matcher->Match(image0, image0, &matches);
  EXPECT_EQ(matches.size(), 0);
}

TEST(SiftCPUFeatureMatcher, TypeMismatch) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

  FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
  ASSERT_EQ(sift_desc.type, FeatureExtractorType::SIFT);

  FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
  undefined_desc.type = FeatureExtractorType::UNDEFINED;

  const FeatureMatcher::Image image_sift = {
      /*image_id=*/1,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(sift_desc)};
  const FeatureMatcher::Image image_undefined = {
      /*image_id=*/2,
      /*camera=*/&camera,
      /*keypoints=*/nullptr,
      std::make_shared<FeatureDescriptors>(undefined_desc)};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image_sift, image_undefined});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = CreateSiftFeatureMatcher(options);

  FeatureMatches matches;
  EXPECT_THROW(matcher->Match(image_sift, image_undefined, &matches),
               std::invalid_argument);
}

TEST(MatchGuidedSiftFeaturesCPU, TypeMismatch) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

  FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
  ASSERT_EQ(sift_desc.type, FeatureExtractorType::SIFT);

  FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
  undefined_desc.type = FeatureExtractorType::UNDEFINED;

  const FeatureMatcher::Image image_sift = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
      std::make_shared<FeatureDescriptors>(sift_desc)};
  const FeatureMatcher::Image image_undefined = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
      std::make_shared<FeatureDescriptors>(undefined_desc)};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image_sift, image_undefined});

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_brute_force_matcher = true;
  auto matcher = CreateSiftFeatureMatcher(options);

  TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

  EXPECT_THROW(matcher->MatchGuided(
                   1.0, image_sift, image_undefined, &two_view_geometry),
               std::invalid_argument);
}

TEST(MatchSiftFeaturesGPU, TypeMismatch) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 100, 200);

    FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
    FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
    undefined_desc.type = FeatureExtractorType::UNDEFINED;

    const FeatureMatcher::Image image_sift = {
        /*image_id=*/1,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(sift_desc)};
    const FeatureMatcher::Image image_undefined = {
        /*image_id=*/2,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(undefined_desc)};

    FeatureMatches matches;
    EXPECT_THROW(matcher->Match(image_sift, image_undefined, &matches),
                 std::invalid_argument);
  });
}

TEST(MatchGuidedSiftFeaturesGPU, TypeMismatch) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimpleRadial, 100.0, 100, 200);

    FeatureDescriptors sift_desc = CreateRandomFeatureDescriptors(2);
    FeatureDescriptors undefined_desc = CreateRandomFeatureDescriptors(2);
    undefined_desc.type = FeatureExtractorType::UNDEFINED;

    const FeatureMatcher::Image image_sift = {
        /*image_id=*/1,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
        std::make_shared<FeatureDescriptors>(sift_desc)};
    const FeatureMatcher::Image image_undefined = {
        /*image_id=*/2,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
        std::make_shared<FeatureDescriptors>(undefined_desc)};

    TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

    EXPECT_THROW(matcher->MatchGuided(
                     1.0, image_sift, image_undefined, &two_view_geometry),
                 std::invalid_argument);
  });
}

TEST(SiftCPUFeatureMatcherFaissVsBruteForce, Nominal) {
  FeatureMatchingOptions match_options;
  match_options.max_num_matches = 1000;

  auto TestFaissVsBruteForce = [](const FeatureMatchingOptions& options,
                                  const FeatureDescriptors& descriptors1,
                                  const FeatureDescriptors& descriptors2) {
    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
    const FeatureMatcher::Image image0 = {
        /*image_id=*/0,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(descriptors1)};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(descriptors2)};

    FeatureDescriptorIndexCacheHelper index_cache_helper(
        {image0, image1, image2});

    FeatureMatches matches_bf;
    FeatureMatches matches_faiss;

    FeatureMatchingOptions custom_options = options;
    custom_options.use_gpu = false;
    custom_options.sift->cpu_brute_force_matcher = true;
    auto bf_matcher = CreateSiftFeatureMatcher(custom_options);
    custom_options.sift->cpu_brute_force_matcher = false;
    custom_options.sift->cpu_descriptor_index_cache =
        &index_cache_helper.index_cache;
    auto faiss_matcher = CreateSiftFeatureMatcher(custom_options);

    bf_matcher->Match(image1, image2, &matches_bf);
    faiss_matcher->Match(image1, image2, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    const size_t num_matches = matches_bf.size();

    bf_matcher->Match(image0, image2, &matches_bf);
    faiss_matcher->Match(image0, image2, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    bf_matcher->Match(image1, image0, &matches_bf);
    faiss_matcher->Match(image1, image0, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    bf_matcher->Match(image0, image0, &matches_bf);
    faiss_matcher->Match(image0, image0, &matches_faiss);
    CheckEqualMatches(matches_bf, matches_faiss);

    return num_matches;
  };

  {
    const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    const FeatureDescriptors descriptors2 = CreateRandomFeatureDescriptors(50);
    FeatureMatchingOptions match_options;
    TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
  }

  {
    const FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2;
    descriptors2.data = descriptors1.data.colwise().reverse();
    descriptors2.type = descriptors1.type;
    FeatureMatchingOptions match_options;
    const size_t num_matches =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches, 50);
  }

  // Check the ratio test.
  {
    FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2 = descriptors1;

    FeatureMatchingOptions match_options;
    const size_t num_matches1 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches1, 50);

    descriptors2.data.row(49) = descriptors2.data.row(0);
    descriptors2.data(0, 0) += 50;
    descriptors2.data.row(0) = FeatureDescriptorsToUnsignedByte(
        descriptors2.data.row(0).cast<float>().normalized());
    descriptors2.data(49, 0) += 100;
    descriptors2.data.row(49) = FeatureDescriptorsToUnsignedByte(
        descriptors2.data.row(49).cast<float>().normalized());

    match_options.sift->max_ratio = 0.4;
    FeatureDescriptors descriptors1_top49;
    descriptors1_top49.data = descriptors1.data.topRows(49);
    descriptors1_top49.type = descriptors1.type;
    const size_t num_matches2 =
        TestFaissVsBruteForce(match_options, descriptors1_top49, descriptors2);
    EXPECT_EQ(num_matches2, 48);

    match_options.sift->max_ratio = 0.6;
    const size_t num_matches3 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches3, 49);
  }

  // Check the cross check.
  {
    FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
    FeatureDescriptors descriptors2 = descriptors1;
    descriptors1.data.row(0) = descriptors1.data.row(1);

    FeatureMatchingOptions match_options;

    match_options.sift->cross_check = false;
    const size_t num_matches1 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches1, 50);

    match_options.sift->cross_check = true;
    const size_t num_matches2 =
        TestFaissVsBruteForce(match_options, descriptors1, descriptors2);
    EXPECT_EQ(num_matches2, 48);
  }
}

TEST(MatchGuidedSiftFeaturesCPU, Nominal) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(0),
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};
  const FeatureMatcher::Image image3 = {
      /*image_id=*/3,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{100, 0}, {2, 0}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};

  FeatureDescriptorIndexCacheHelper index_cache_helper(
      {image0, image1, image2, image3});

  TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

  FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
  options.use_gpu = false;
  options.sift->cpu_descriptor_index_cache = &index_cache_helper.index_cache;
  auto matcher = CreateSiftFeatureMatcher(options);

  constexpr double kMaxError = 1.0;

  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  matcher->MatchGuided(kMaxError, image3, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);

  matcher->MatchGuided(kMaxError, image0, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
  matcher->MatchGuided(kMaxError, image1, image0, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
  matcher->MatchGuided(kMaxError, image0, image0, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
}

void TestGuidedMatchingWithCameraDistortion(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  // Test guided matching with essential matrix using calibrated cameras.
  // This exercises the code path that uses normalized coordinates.
  // Use the OPENCV model with strong radial and tangential distortion. Its
  // params are fx, fy, cx, cy, k1, k2, p1, p2. The distortion is strong enough
  // that the pixel-coordinate fundamental matrix finds no matches, but must
  // stay invertible over the keypoints used below, which is what bounds how
  // large these coefficients can be; p2 is left at zero for that reason.
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kOpenCV, 100.0, 100, 200);
  camera.params[4] = -0.5;  // k1
  camera.params[5] = 0.5;   // k2
  camera.params[6] = -0.5;  // p1

  // Two points on the epipolar line (v=0 in normalized coordinates).
  const Eigen::Vector2f img_point11 =
      camera.ImgFromCam({-0.5, 0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point12 =
      camera.ImgFromCam({0.4, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point21 =
      camera.ImgFromCam({0.3, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point22 =
      camera.ImgFromCam({-0.4, 0.1, 1.0}).value().cast<float>();

  const FeatureMatcher::Image image0 = {
      /*image_id=*/0,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(0),
      std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image0, image1, image2});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));
  two_view_geometry.F =
      FundamentalFromEssentialMatrix(camera.CalibrationMatrix(),
                                     two_view_geometry.E.value(),
                                     camera.CalibrationMatrix());

  constexpr double kMaxError = 1.0;

  // With uncalibrated cameras, the fundamental matrix is used with pixel
  // coordinates and no matches are expected to be found due to strong
  // distortion.
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ASSERT_EQ(two_view_geometry.inlier_matches.size(), 0);

  // With calibrated cameras, the essential matrix is used with normalized
  // coordinates and matches are expected to be found.
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  matcher->MatchGuided(kMaxError, image0, image2, &two_view_geometry);
  EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
}

// Guided matching for a spherical camera, with correspondences deliberately in
// the back hemisphere. Those pixels have no normalized image plane
// representation at all - CamFromImg fails for them - so they are only
// matchable via the full-sphere bearing.
void TestGuidedMatchingSpherical(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0, 512, 256);

  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(1, 0, 0));

  // Both points are behind both cameras, i.e. in the back hemisphere.
  const Eigen::Vector3d point3D1(0.3, 0.1, -2.0);
  const Eigen::Vector3d point3D2(-0.25, -0.15, -2.5);

  auto project = [&camera](const Eigen::Vector3d& point3D) {
    const std::optional<Eigen::Vector2d> image_point =
        camera.ImgFromCam(point3D);
    THROW_CHECK(image_point.has_value());
    // The premise of this test: these pixels are unprojectable through the
    // normalized image plane. If this ever starts failing, the test is no
    // longer exercising the back hemisphere.
    EXPECT_FALSE(camera.CamFromImg(*image_point).has_value());
    return image_point->cast<float>().eval();
  };

  const Eigen::Vector2f img_point11 = project(point3D1);
  const Eigen::Vector2f img_point12 = project(point3D2);
  const Eigen::Vector2f img_point21 = project(cam2_from_cam1 * point3D2);
  const Eigen::Vector2f img_point22 = project(cam2_from_cam1 * point3D1);

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  // Same as image2, but with the second correspondence replaced by a decoy far
  // off the epipolar great circle, which must be rejected.
  //
  // This is the load-bearing assertion. Recovering the matches above is
  // necessary but not sufficient: when every keypoint is unprojectable, the old
  // sentinel mapped them all to the same location, so the guided filter scored
  // every pair identically and degenerated into accepting everything - and
  // plain descriptor matching then produced the right answer anyway. Only a
  // decoy that the filter must actively reject distinguishes "the epipolar
  // constraint is evaluated correctly for back-hemisphere rays" from "the
  // constraint has stopped constraining anything".
  const Eigen::Vector2f img_point_decoy =
      project(cam2_from_cam1 * Eigen::Vector3d(2.0, -1.5, -0.5));
  const FeatureMatcher::Image image3 = {
      /*image_id=*/3,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          {img_point21.x(), img_point21.y()},
          {img_point_decoy.x(), img_point_decoy.y()}}),
      image2.descriptors};

  auto matcher = matcher_factory({image1, image2, image3});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(cam2_from_cam1);

  constexpr double kMaxError = 4.0;

  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  matcher->MatchGuided(kMaxError, image1, image3, &two_view_geometry);
  ASSERT_EQ(two_view_geometry.inlier_matches.size(), 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);
}

// One correspondence in the front hemisphere and one in the back, to verify
// the two are handled by the same code path rather than being swapped.
void TestGuidedMatchingSphericalMixedHemispheres(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  const Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kEquirectangular, /*focal_length=*/0, 512, 256);

  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(1, 0, 0));

  const Eigen::Vector3d point3D_front(0.2, 0.1, 2.0);
  const Eigen::Vector3d point3D_back(-0.25, -0.15, -2.5);

  auto project = [&camera](const Eigen::Vector3d& point3D) {
    const std::optional<Eigen::Vector2d> image_point =
        camera.ImgFromCam(point3D);
    THROW_CHECK(image_point.has_value());
    return image_point->cast<float>().eval();
  };

  const Eigen::Vector2f img_point11 = project(point3D_front);
  const Eigen::Vector2f img_point12 = project(point3D_back);
  const Eigen::Vector2f img_point21 = project(cam2_from_cam1 * point3D_back);
  const Eigen::Vector2f img_point22 = project(cam2_from_cam1 * point3D_front);

  EXPECT_TRUE(camera.CamFromImg(img_point11.cast<double>()).has_value());
  EXPECT_FALSE(camera.CamFromImg(img_point12.cast<double>()).has_value());

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  // Replaces the back-hemisphere correspondence with a decoy off the epipolar
  // great circle, so that the filter has to actively reject it. See the
  // comment in TestGuidedMatchingSpherical for why this is what discriminates.
  const Eigen::Vector2f img_point_decoy =
      project(cam2_from_cam1 * Eigen::Vector3d(2.0, -1.5, -0.5));
  const FeatureMatcher::Image image3 = {
      /*image_id=*/3,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          {img_point_decoy.x(), img_point_decoy.y()},
          {img_point22.x(), img_point22.y()}}),
      image2.descriptors};

  auto matcher = matcher_factory({image1, image2, image3});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(cam2_from_cam1);

  constexpr double kMaxError = 4.0;

  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  // Only the front-hemisphere correspondence survives.
  matcher->MatchGuided(kMaxError, image1, image3, &two_view_geometry);
  ASSERT_EQ(two_view_geometry.inlier_matches.size(), 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 0);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 1);
}

// A keypoint the camera cannot unproject must be rejected outright. It used to
// be relocated to a (1e6, 1e6) sentinel, which does not reject: the Sampson
// error is a ratio whose numerator and denominator scale together, so the
// residual converges to the finite squared distance from the partner to the
// epipolar line of the point at infinity in direction (1, 1, 0). Any partner
// near that line was therefore silently accepted.
void TestGuidedMatchingUnprojectableKeypoints(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  Camera camera =
      Camera::CreateFromModelId(1, CameraModelId::kOpenCV, 100.0, 100, 200);
  camera.params[4] = -0.5;  // k1
  camera.params[5] = 0.5;   // k2
  camera.params[6] = -0.5;  // p1

  // Well inside a region where the iterative undistortion does not converge.
  const Eigen::Vector2d unprojectable(50.0, 150.0);
  ASSERT_FALSE(camera.CamFromImg(unprojectable).has_value());

  auto project = [&camera](const Eigen::Vector3d& point3D) {
    const std::optional<Eigen::Vector2d> image_point =
        camera.ImgFromCam(point3D);
    THROW_CHECK(image_point.has_value());
    // Everything except the sentinel keypoint must be a normal, usable
    // keypoint, or the test would pass for the wrong reason.
    EXPECT_TRUE(camera.CamFromImg(*image_point).has_value());
    return image_point->cast<float>().eval();
  };

  // A translation with tx == ty, so that the epipolar line of the sentinel
  // direction (1, 1, 0) passes through the image center and the decoy below
  // can sit on it at a well-behaved location.
  const Rigid3d cam2_from_cam1(Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(1, 1, 1));
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  const Eigen::Vector3d point3D(-0.3, -0.2, 2.0);
  const Eigen::Vector2f img_point_good1 = project(point3D);
  const Eigen::Vector2f img_point_good2 = project(cam2_from_cam1 * point3D);

  // The old (1e6, 1e6) sentinel converges to the direction (1, 1, 0); its
  // epipolar line is where spurious matches used to concentrate, so the decoy
  // is placed exactly on it.
  const Eigen::Vector3d sentinel_line = E * Eigen::Vector3d(1, 1, 0);
  const double decoy_x = 0.2;
  const double decoy_y =
      -(sentinel_line.x() * decoy_x + sentinel_line.z()) / sentinel_line.y();
  const Eigen::Vector2f img_point_decoy = project({decoy_x, decoy_y, 1.0});

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          {static_cast<float>(unprojectable.x()),
           static_cast<float>(unprojectable.y())},
          {img_point_good1.x(), img_point_good1.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera,
      std::make_shared<FeatureKeypoints>(std::vector<FeatureKeypoint>{
          {img_point_good2.x(), img_point_good2.y()},
          {img_point_decoy.x(), img_point_decoy.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image1, image2});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.E = E;

  matcher->MatchGuided(/*max_error=*/1.0, image1, image2, &two_view_geometry);

  // Only the good pair survives; the unprojectable keypoint 0 matches nothing.
  ASSERT_EQ(two_view_geometry.inlier_matches.size(), 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
  EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);
}

TEST(MatchGuidedSiftFeaturesCPU, EssentialMatrix) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingWithCameraDistortion(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchGuidedSiftFeaturesCPU, Spherical) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingSpherical(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchGuidedSiftFeaturesCPU, SphericalMixedHemispheres) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingSphericalMixedHemispheres(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchGuidedSiftFeaturesCPU, UnprojectableKeypoints) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingUnprojectableKeypoints(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

void TestGuidedMatchingSharedFocal(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  // An UNCALIBRATED pair carrying solver-estimated intrinsics (camera1/camera2)
  // is guided-matched via the essential matrix in normalized coordinates, using
  // those estimated intrinsics rather than the images' cameras, whose focal
  // length is only a placeholder. Distortion is strong enough that the
  // pixel-coordinate F path finds nothing, and the placeholder focal is wrong
  // enough that normalizing with it finds nothing either, so the test passes
  // only if the estimated camera is the one used.
  //
  // As elsewhere on the E path, E is taken to relate undistorted rays.
  constexpr double kEstimatedFocal = 100.0;
  constexpr double kPlaceholderFocal = 500.0;
  // OPENCV params are fx, fy, cx, cy, k1, k2, p1, p2. Same distortion as in
  // TestGuidedMatchingWithCameraDistortion: strong, but invertible over the
  // keypoints used below.
  Camera camera = Camera::CreateFromModelId(
      1, CameraModelId::kOpenCV, kEstimatedFocal, 100, 200);
  camera.params[4] = -0.5;  // k1
  camera.params[5] = 0.5;   // k2
  camera.params[6] = -0.5;  // p1

  // The camera as stored in the database: same model and distortion, but the
  // focal length has not been recovered yet.
  Camera placeholder_camera = camera;
  placeholder_camera.SetFocalLength(kPlaceholderFocal);

  // Two points on the epipolar line (v=0 in normalized coordinates).
  const Eigen::Vector2f img_point11 =
      camera.ImgFromCam({-0.5, 0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point12 =
      camera.ImgFromCam({0.4, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point21 =
      camera.ImgFromCam({0.3, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point22 =
      camera.ImgFromCam({-0.4, 0.1, 1.0}).value().cast<float>();

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&placeholder_camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&placeholder_camera,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image1, image2});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));
  two_view_geometry.camera1 = camera;
  two_view_geometry.camera2 = camera;
  // F = K^-T E K^-1, as the estimator populates it for this config.
  two_view_geometry.F =
      FundamentalFromEssentialMatrix(camera.CalibrationMatrix(),
                                     two_view_geometry.E.value(),
                                     camera.CalibrationMatrix());

  constexpr double kMaxError = 1.0;

  // Matches are found only by normalizing with the estimated intrinsics. The
  // complementary case, an UNCALIBRATED pair without them falling back to the
  // F path, is covered by TestGuidedMatchingWithCameraDistortion.
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);
}

// The normalizing camera is not a function of the image alone: a shared-focal
// pair carries a focal length estimated per pair, so the same image matched
// against different partners must be renormalized. Guards the GPU matcher's
// feature-location cache, which keys on the image id.
void TestGuidedMatchingSharedFocalPerPairFocal(
    const std::function<std::unique_ptr<FeatureMatcher>(
        const std::vector<FeatureMatcher::Image>&)>& matcher_factory) {
  constexpr double kFocalA = 100.0;
  constexpr double kFocalB = 200.0;
  const Camera camera_a = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, kFocalA, 100, 200);
  const Camera camera_b = Camera::CreateFromModelId(
      2, CameraModelId::kSimplePinhole, kFocalB, 100, 200);

  // image1's pixels normalize to y = +-0.1 under camera_a, and to half that,
  // y = +-0.05, under camera_b. The relative pose is a pure x-translation, so a
  // match requires the partner's normalized y to agree.
  const Eigen::Vector2f img_point11 =
      camera_a.ImgFromCam({-0.5, 0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point12 =
      camera_a.ImgFromCam({0.4, -0.1, 1.0}).value().cast<float>();
  // Partner for the camera_a pair.
  const Eigen::Vector2f img_point21 =
      camera_a.ImgFromCam({0.3, -0.1, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point22 =
      camera_a.ImgFromCam({-0.4, 0.1, 1.0}).value().cast<float>();
  // Partner for the camera_b pair.
  const Eigen::Vector2f img_point31 =
      camera_b.ImgFromCam({0.3, -0.05, 1.0}).value().cast<float>();
  const Eigen::Vector2f img_point32 =
      camera_b.ImgFromCam({-0.4, 0.05, 1.0}).value().cast<float>();

  const FeatureMatcher::Image image1 = {
      /*image_id=*/1,
      /*camera=*/&camera_a,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point11.x(), img_point11.y()},
                                       {img_point12.x(), img_point12.y()}}),
      std::make_shared<FeatureDescriptors>(CreateRandomFeatureDescriptors(2))};
  const FeatureMatcher::Image image2 = {
      /*image_id=*/2,
      /*camera=*/&camera_a,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point21.x(), img_point21.y()},
                                       {img_point22.x(), img_point22.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};
  const FeatureMatcher::Image image3 = {
      /*image_id=*/3,
      /*camera=*/&camera_a,
      std::make_shared<FeatureKeypoints>(
          std::vector<FeatureKeypoint>{{img_point31.x(), img_point31.y()},
                                       {img_point32.x(), img_point32.y()}}),
      std::make_shared<FeatureDescriptors>(
          CreateReversedDescriptors(*image1.descriptors))};

  auto matcher = matcher_factory({image1, image2, image3});

  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
  two_view_geometry.E = EssentialMatrixFromPose(
      Rigid3d(Eigen::Quaterniond::Identity(), Eigen::Vector3d(1, 0, 0)));

  constexpr double kMaxError = 1.0;

  two_view_geometry.camera1 = camera_a;
  two_view_geometry.camera2 = camera_a;
  matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);

  // Same image1, different estimated focal: stale normalized keypoints from the
  // previous call would put image1 at y = +-0.1 instead of +-0.05, far outside
  // the ~1/f normalized threshold.
  two_view_geometry.camera1 = camera_b;
  two_view_geometry.camera2 = camera_b;
  matcher->MatchGuided(kMaxError, image1, image3, &two_view_geometry);
  ExpectReversedInlierMatches(two_view_geometry);
}

TEST(MatchGuidedSiftFeaturesCPU, SharedFocal) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingSharedFocal(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchGuidedSiftFeaturesCPU, SharedFocalPerPairFocal) {
  std::unique_ptr<FeatureDescriptorIndexCacheHelper> index_cache_helper;
  TestGuidedMatchingSharedFocalPerPairFocal(
      [&index_cache_helper](const std::vector<FeatureMatcher::Image>& images) {
        index_cache_helper =
            std::make_unique<FeatureDescriptorIndexCacheHelper>(images);
        FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
        options.use_gpu = false;
        options.sift->cpu_descriptor_index_cache =
            &index_cache_helper->index_cache;
        return CreateSiftFeatureMatcher(options);
      });
}

TEST(MatchSiftFeaturesGPU, Nominal) {
  RunGpuTest([] {
    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    const Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
    const FeatureMatcher::Image image0 = {
        /*image_id=*/0,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(2))};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        /*keypoints=*/nullptr,
        std::make_shared<FeatureDescriptors>(
            CreateReversedDescriptors(*image1.descriptors))};

    FeatureMatches matches;

    matcher->Match(image1, image2, &matches);
    ExpectReversedMatches(matches);

    matcher->Match(image1, image2, &matches);
    ExpectReversedMatches(matches);

    matcher->Match(image0, image2, &matches);
    EXPECT_EQ(matches.size(), 0);
    matcher->Match(image1, image0, &matches);
    EXPECT_EQ(matches.size(), 0);
    matcher->Match(image0, image0, &matches);
    EXPECT_EQ(matches.size(), 0);
  });
}

TEST(MatchSiftFeaturesCPUvsGPU, Nominal) {
  RunGpuTest([] {
    auto TestCPUvsGPU = [](const FeatureMatchingOptions& options,
                           const FeatureDescriptors& descriptors1,
                           const FeatureDescriptors& descriptors2) {
      const Camera camera = Camera::CreateFromModelId(
          1, CameraModelId::kSimplePinhole, 100.0, 100, 200);
      const FeatureMatcher::Image image0 = {
          /*image_id=*/0,
          /*camera=*/&camera,
          /*keypoints=*/nullptr,
          std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
      const FeatureMatcher::Image image1 = {
          /*image_id=*/1,
          /*camera=*/&camera,
          /*keypoints=*/nullptr,
          std::make_shared<FeatureDescriptors>(descriptors1)};
      const FeatureMatcher::Image image2 = {
          /*image_id=*/2,
          /*camera=*/&camera,
          /*keypoints=*/nullptr,
          std::make_shared<FeatureDescriptors>(descriptors2)};

      FeatureDescriptorIndexCacheHelper index_cache_helper(
          {image0, image1, image2});

      FeatureMatchingOptions custom_options = options;
      custom_options.use_gpu = true;
      custom_options.max_num_matches = 1000;
      auto gpu_matcher =
          THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(custom_options));
      custom_options.use_gpu = false;
      custom_options.sift->cpu_descriptor_index_cache =
          &index_cache_helper.index_cache;
      auto cpu_matcher = CreateSiftFeatureMatcher(custom_options);

      FeatureMatches matches_cpu;
      FeatureMatches matches_gpu;

      cpu_matcher->Match(image1, image2, &matches_cpu);
      gpu_matcher->Match(image1, image2, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      const size_t num_matches = matches_cpu.size();

      cpu_matcher->Match(image0, image2, &matches_cpu);
      gpu_matcher->Match(image0, image2, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      cpu_matcher->Match(image1, image0, &matches_cpu);
      gpu_matcher->Match(image1, image0, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      cpu_matcher->Match(image0, image0, &matches_cpu);
      gpu_matcher->Match(image0, image0, &matches_gpu);
      CheckEqualMatches(matches_cpu, matches_gpu);

      return num_matches;
    };

    {
      const FeatureDescriptors descriptors1 =
          CreateRandomFeatureDescriptors(50);
      const FeatureDescriptors descriptors2 =
          CreateRandomFeatureDescriptors(50);
      FeatureMatchingOptions match_options;
      TestCPUvsGPU(match_options, descriptors1, descriptors2);
    }

    {
      const FeatureDescriptors descriptors1 =
          CreateRandomFeatureDescriptors(50);
      FeatureDescriptors descriptors2;
      descriptors2.data = descriptors1.data.colwise().reverse();
      descriptors2.type = descriptors1.type;
      FeatureMatchingOptions match_options;
      const size_t num_matches =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches, 50);
    }

    // Check the ratio test.
    {
      FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
      FeatureDescriptors descriptors2 = descriptors1;

      FeatureMatchingOptions match_options;
      const size_t num_matches1 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches1, 50);

      descriptors2.data.row(49) = descriptors2.data.row(0);
      descriptors2.data(0, 0) += 50;
      descriptors2.data.row(0) = FeatureDescriptorsToUnsignedByte(
          descriptors2.data.row(0).cast<float>().normalized());
      descriptors2.data(49, 0) += 100;
      descriptors2.data.row(49) = FeatureDescriptorsToUnsignedByte(
          descriptors2.data.row(49).cast<float>().normalized());

      match_options.sift->max_ratio = 0.4;
      FeatureDescriptors descriptors1_top49;
      descriptors1_top49.data = descriptors1.data.topRows(49);
      descriptors1_top49.type = descriptors1.type;
      const size_t num_matches2 =
          TestCPUvsGPU(match_options, descriptors1_top49, descriptors2);
      EXPECT_EQ(num_matches2, 48);

      match_options.sift->max_ratio = 0.6;
      const size_t num_matches3 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches3, 49);
    }

    // Check the cross check.
    {
      FeatureDescriptors descriptors1 = CreateRandomFeatureDescriptors(50);
      FeatureDescriptors descriptors2 = descriptors1;
      descriptors1.data.row(0) = descriptors1.data.row(1);

      FeatureMatchingOptions match_options;

      match_options.sift->cross_check = false;
      const size_t num_matches1 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches1, 50);

      match_options.sift->cross_check = true;
      const size_t num_matches2 =
          TestCPUvsGPU(match_options, descriptors1, descriptors2);
      EXPECT_EQ(num_matches2, 48);
    }
  });
}

TEST(MatchGuidedSiftFeaturesGPU, Nominal) {
  RunGpuTest([] {
    Camera camera = Camera::CreateFromModelId(
        1, CameraModelId::kSimpleRadial, 100.0, 100, 200);
    const FeatureMatcher::Image image0 = {
        /*image_id=*/0,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(0),
        std::make_shared<FeatureDescriptors>(CreateEmptyDescriptors())};
    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{1, 0}, {2, 0}}),
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(2))};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{2, 0}, {1, 0}}),
        std::make_shared<FeatureDescriptors>(
            CreateReversedDescriptors(*image1.descriptors))};
    const FeatureMatcher::Image image3 = {
        /*image_id=*/3,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(
            std::vector<FeatureKeypoint>{{100, 0}, {1, 0}}),
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(2))};

    FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
    options.use_gpu = true;
    options.max_num_matches = 1000;
    auto matcher = THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));

    TwoViewGeometry two_view_geometry = CreatePlanarTwoViewGeometry();

    constexpr double kMaxError = 4.0;

    matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
    ExpectReversedInlierMatches(two_view_geometry);

    matcher->MatchGuided(kMaxError, image1, image2, &two_view_geometry);
    ExpectReversedInlierMatches(two_view_geometry);

    matcher->MatchGuided(kMaxError, image3, image2, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 1);
    EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx1, 1);
    EXPECT_EQ(two_view_geometry.inlier_matches[0].point2D_idx2, 0);

    matcher->MatchGuided(kMaxError, image0, image2, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
    matcher->MatchGuided(kMaxError, image1, image0, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
    matcher->MatchGuided(kMaxError, image0, image0, &two_view_geometry);
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), 0);
  });
}

TEST(MatchGuidedSiftFeaturesGPU, EssentialMatrix) {
  RunGpuTest([] {
    TestGuidedMatchingWithCameraDistortion(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

TEST(MatchGuidedSiftFeaturesGPU, Spherical) {
  RunGpuTest([] {
    TestGuidedMatchingSpherical(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

TEST(MatchGuidedSiftFeaturesGPU, SphericalMixedHemispheres) {
  RunGpuTest([] {
    TestGuidedMatchingSphericalMixedHemispheres(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

TEST(MatchGuidedSiftFeaturesGPU, UnprojectableKeypoints) {
  RunGpuTest([] {
    TestGuidedMatchingUnprojectableKeypoints(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

// The GPU tangent Sampson kernel must reproduce the CPU reference
// implementation on the same input. This is the check that distinguishes a
// genuine kernel bug from a plumbing bug, since the CPU path is independently
// tested above.
TEST(MatchGuidedSiftFeaturesCPUvsGPUGuided, EssentialMatrix) {
  const size_t kNumFeatures = 200;
  std::vector<Camera> cameras;
  cameras.push_back(Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 650.0, 1024, 768));
  cameras.push_back(Camera::CreateFromModelId(
      2, CameraModelId::kOpenCVFisheye, 350.0, 1024, 768));
  cameras.push_back(Camera::CreateFromModelId(
      3, CameraModelId::kEquirectangular, 0.0, 1000, 500));

  for (const Camera& camera : cameras) {
    SetPRNGSeed(42);
    FeatureKeypoints keypoints1(kNumFeatures);
    FeatureKeypoints keypoints2(kNumFeatures);
    for (size_t i = 0; i < kNumFeatures; ++i) {
      keypoints1[i] =
          FeatureKeypoint(RandomUniformReal<float>(1.0f, camera.width - 1.0f),
                          RandomUniformReal<float>(1.0f, camera.height - 1.0f));
      keypoints2[i] =
          FeatureKeypoint(RandomUniformReal<float>(1.0f, camera.width - 1.0f),
                          RandomUniformReal<float>(1.0f, camera.height - 1.0f));
    }

    const FeatureMatcher::Image image1 = {
        /*image_id=*/1,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(keypoints1),
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(kNumFeatures))};
    const FeatureMatcher::Image image2 = {
        /*image_id=*/2,
        /*camera=*/&camera,
        std::make_shared<FeatureKeypoints>(keypoints2),
        std::make_shared<FeatureDescriptors>(
            CreateRandomFeatureDescriptors(kNumFeatures))};

    TwoViewGeometry geometry;
    geometry.config = TwoViewGeometry::CALIBRATED;
    geometry.E = EssentialMatrixFromPose(
        Rigid3d(Eigen::Quaterniond(Eigen::AngleAxisd(
                    0.2, Eigen::Vector3d(0.3, 1.0, 0.2).normalized())),
                Eigen::Vector3d(1.0, 0.15, 0.05).normalized()));

    // A loose threshold so a non-trivial number of pairs survive the filter.
    constexpr double kMaxError = 30.0;

    FeatureDescriptorIndexCacheHelper index_cache_helper({image1, image2});
    FeatureMatchingOptions cpu_options(FeatureMatcherType::SIFT_BRUTEFORCE);
    cpu_options.use_gpu = false;
    cpu_options.sift->cpu_descriptor_index_cache =
        &index_cache_helper.index_cache;
    TwoViewGeometry cpu_geometry = geometry;
    CreateSiftFeatureMatcher(cpu_options)
        ->MatchGuided(kMaxError, image1, image2, &cpu_geometry);

    EXPECT_GT(cpu_geometry.inlier_matches.size(), 0u)
        << "model " << camera.ModelName();

    // Note that the assertions must live inside the lambda, since the body is
    // not executed if the GPU/OpenGL context is unavailable.
    RunGpuTest([&] {
      TwoViewGeometry gpu_geometry = geometry;
      FeatureMatchingOptions gpu_options(FeatureMatcherType::SIFT_BRUTEFORCE);
      gpu_options.use_gpu = true;
      gpu_options.max_num_matches = 4 * kNumFeatures;
      THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(gpu_options))
          ->MatchGuided(kMaxError, image1, image2, &gpu_geometry);

      // The CPU scores in double and the GPU in float, so a pair whose tangent
      // Sampson residual sits within float epsilon of kMaxError can flip
      // inclusion. Compare as sets and tolerate a few boundary flips rather
      // than requiring identical size and order.
      const auto to_set = [](const FeatureMatches& matches) {
        std::set<std::pair<point2D_t, point2D_t>> set;
        for (const auto& match : matches) {
          set.emplace(match.point2D_idx1, match.point2D_idx2);
        }
        return set;
      };
      const std::set<std::pair<point2D_t, point2D_t>> cpu_set =
          to_set(cpu_geometry.inlier_matches);
      const std::set<std::pair<point2D_t, point2D_t>> gpu_set =
          to_set(gpu_geometry.inlier_matches);
      size_t num_disagree = 0;
      for (const auto& match : cpu_set) num_disagree += !gpu_set.count(match);
      for (const auto& match : gpu_set) num_disagree += !cpu_set.count(match);
      EXPECT_LE(num_disagree, 4u) << "model " << camera.ModelName();
    });
  }
}

TEST(MatchGuidedSiftFeaturesGPU, SharedFocal) {
  RunGpuTest([] {
    TestGuidedMatchingSharedFocal(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

TEST(MatchGuidedSiftFeaturesGPU, SharedFocalPerPairFocal) {
  RunGpuTest([] {
    TestGuidedMatchingSharedFocalPerPairFocal(
        [](const std::vector<FeatureMatcher::Image>& images) {
          FeatureMatchingOptions options(FeatureMatcherType::SIFT_BRUTEFORCE);
          options.use_gpu = true;
          options.max_num_matches = 1000;
          return THROW_CHECK_NOTNULL(CreateSiftFeatureMatcher(options));
        });
  });
}

}  // namespace
}  // namespace colmap
