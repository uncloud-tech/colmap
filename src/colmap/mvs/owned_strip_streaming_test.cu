// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/patch_match_cuda_owned.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
template <class T>
void ExpectBits(const std::vector<T>& expected, const std::vector<T>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  if (expected.empty()) return;
  EXPECT_EQ(
      std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(T)),
      0);
}
void ExpectState(const StripQualificationSnapshot& a,
                 const StripQualificationSnapshot& b) {
  ExpectBits(a.cost.GetData(), b.cost.GetData());
  ExpectBits(a.selection.GetData(), b.selection.GetData());
  ExpectBits(a.mask.GetData(), b.mask.GetData());
  ExpectBits(a.subsequent_prng_draws, b.subsequent_prng_draws);
}
std::vector<int> Consistency(const Mat<uint8_t>& mask,
                             const std::vector<int>& sources) {
  std::vector<int> output;
  for (size_t r = 0; r < mask.GetHeight(); ++r) {
    for (size_t c = 0; c < mask.GetWidth(); ++c) {
      std::vector<int> ids;
      for (size_t s = 0; s < sources.size(); ++s)
        if (mask.Get(r, c, s)) ids.push_back(sources[s]);
      if (!ids.empty()) {
        output.insert(output.end(), {int(c), int(r), int(ids.size())});
        output.insert(output.end(), ids.begin(), ids.end());
      }
    }
  }
  return output;
}
std::vector<Image> Images(int width, int height) {
  const float K[9] = {
      90, 0, float(width) / 2, 0, 95, float(height) / 2, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<Image> images;
  for (int i = 0; i < 4; ++i) {
    const float T[3] = {0.07f * i, 0.01f * i, 0};
    images.emplace_back(
        "strip-photo-" + std::to_string(i), width, height, K, R, T);
    Bitmap bitmap(width, height, false);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        if (!bitmap.SetPixel(x,
                             y,
                             BitmapColor<uint8_t>(
                                 (17 * x + 31 * y + 7 * x * y + 29 * i) % 256)))
          throw std::runtime_error("fixture pixel failed");
    images.back().SetBitmap(std::move(bitmap));
  }
  return images;
}
constexpr size_t kInspectCap = 8 * 1024 * 1024;

void Compare(const PatchMatchOptions& options,
             const PatchMatch::Problem& problem,
             size_t columns,
             bool reject) {
  StripStreamingOptions streaming;
  streaming.directory = std::filesystem::current_path();
  streaming.columns = columns;
  streaming.host_staging_bytes = 1024 * 1024;
  streaming.device_staging_bytes = 8 * 1024 * 1024;
  streaming.disk_bytes = 9 *
                         problem.images->at(problem.ref_image_idx).GetWidth() *
                         problem.images->at(problem.ref_image_idx).GetHeight() *
                         problem.src_image_idxs.size();
  auto insufficient = streaming;
  const auto minimum = BuildStripStagingLayout(
      problem.images->at(problem.ref_image_idx).GetWidth(),
      problem.images->at(problem.ref_image_idx).GetHeight(),
      problem.src_image_idxs.size(),
      columns,
      options.filter);
  insufficient.host_staging_bytes = minimum.host_bytes - 1;
  EXPECT_THROW((PatchMatchCuda(options, problem, 32, insufficient)),
               std::invalid_argument);
  insufficient = streaming;
  insufficient.device_staging_bytes = minimum.device_bytes - 1;
  EXPECT_THROW((PatchMatchCuda(options, problem, 32, insufficient)),
               std::invalid_argument);
  insufficient = streaming;
  // Unfiltered backing has 8 bytes per source-pixel, not 9.
  insufficient.disk_bytes =
      streaming.disk_bytes / 9 * (options.filter ? 9 : 8) - 1;
  EXPECT_THROW((PatchMatchCuda(options, problem, 32, insufficient)),
               std::invalid_argument);
  PatchMatchCuda resident(options, problem, 32);
  std::vector<StripQualificationSnapshot> sweeps;
  resident.SetSweepObserver([&](int, int) {
    sweeps.push_back(resident.InspectQualification(kInspectCap));
  });
  resident.Run();
  const auto expected = resident.InspectQualification(kInspectCap);
  PatchMatchCuda streamed(options, problem, 32, streaming);
  size_t observed = 0;
  streamed.SetSweepObserver([&](int, int) {
    ASSERT_LT(observed, sweeps.size());
    ExpectState(sweeps[observed++], streamed.InspectQualification(kInspectCap));
  });
  EXPECT_THROW(streamed.GetDepthMap(), std::logic_error);
  EXPECT_THROW(streamed.GetSelProbMap(), std::logic_error);
  streamed.Run();
  EXPECT_EQ(observed, 8);
  ExpectState(expected, streamed.InspectQualification(kInspectCap));
  ExpectBits(resident.GetDepthMap().GetData(),
             streamed.GetDepthMap().GetData());
  ExpectBits(resident.GetNormalMap().GetData(),
             streamed.GetNormalMap().GetData());
  if (options.filter) {
    EXPECT_EQ(resident.GetConsistentImageIdxs(),
              Consistency(expected.mask, problem.src_image_idxs));
    if (reject) {
      EXPECT_TRUE(std::all_of(expected.mask.GetData().begin(),
                              expected.mask.GetData().end(),
                              [](uint8_t v) { return v == 0; }));
    }
  }
  EXPECT_GT(streamed.StripBytesRead(), streaming.disk_bytes);
  EXPECT_GT(streamed.StripBytesWritten(), streaming.disk_bytes);
  EXPECT_THROW(streamed.InspectQualification(1), std::invalid_argument);
  EXPECT_THROW(streamed.GetConsistentImageIdxs(), std::logic_error);
  // Cancellation after a real strip has executed: no partial output or retry.
  std::atomic<bool> cancelled{false};
  streaming.cancelled = &cancelled;
  PatchMatchCuda interrupted(options, problem, 32, streaming);
  interrupted.SetSweepObserver([&](int, int) { cancelled = true; });
  EXPECT_THROW(interrupted.Run(), std::runtime_error);
  EXPECT_THROW(interrupted.GetDepthMap(), std::logic_error);
  EXPECT_THROW(interrupted.InspectQualification(kInspectCap), std::logic_error);
  EXPECT_THROW(interrupted.Run(), std::logic_error);
  // Existing resident path accepts graph mode; only this seam rejects it.
  auto graph = options;
  graph.owned_cuda_graphs = true;
  EXPECT_THROW((PatchMatchCuda(graph, problem, 32, streaming)),
               std::invalid_argument);
}

TEST(OwnedStripStreaming, PhotoGeoResidentOracleAndCancellation) {
#ifndef __linux__
  GTEST_SKIP() << "native file strip backend unavailable";
#else
  int device = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&device));
  constexpr int block = 32;
  for (const auto shape : {std::pair<int, int>{3 * block + 5, 4 * block + 7},
                           std::pair<int, int>{13, 19}}) {
    SCOPED_TRACE(::testing::Message() << shape.first << "x" << shape.second);
    auto images = Images(shape.first, shape.second);
    PatchMatchOptions options;
    options.gpu_index = std::to_string(device);
    options.geom_consistency = false;
    options.depth_min = 1;
    options.depth_max = 5;
    options.sigma_spatial = 5;
    options.num_iterations = 2;
    options.num_samples = 3;
    options.filter = false;
    options.random_seed = 0;
    options.owned_cuda_graphs = false;
    // These are this run's PHOTO prerequisites, never historical GEO maps.
    std::vector<DepthMap> depths(4);
    std::vector<NormalMap> normals(4);
    for (int ref = 0; ref < 4; ++ref) {
      PatchMatch::Problem photo;
      photo.images = &images;
      photo.ref_image_idx = ref;
      photo.src_image_idxs = {(ref + 3) % 4, (ref + 1) % 4, (ref + 2) % 4};
      PatchMatchCuda producer(options, photo);
      producer.Run();
      depths[ref] = producer.GetDepthMap();
      normals[ref] = producer.GetNormalMap();
    }
    PatchMatch::Problem problem;
    problem.images = &images;
    problem.ref_image_idx = 0;
    problem.src_image_idxs = {3, 1, 2};
    Compare(options, problem, block, false);
    options.filter = true;
    options.filter_min_num_consistent =
        4;  // Genuine rejection: only 3 sources.
    Compare(options, problem, block, true);
    problem.depth_maps = &depths;
    problem.normal_maps = &normals;
    options.geom_consistency = true;
    options.filter_min_num_consistent = 1;
    Compare(options, problem, block, false);
    options.filter_min_num_consistent = 4;
    Compare(options, problem, block, true);
  }
#endif
}
}  // namespace
}  // namespace colmap::mvs::owned
