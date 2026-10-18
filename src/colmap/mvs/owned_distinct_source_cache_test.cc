// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap::mvs {
namespace {

// Real public owned solver: texture creation, curand and all four rotations
// remain production code. This is NOT a forced sample-path/intermediate test:
// the API does not expose draws, candidate costs, winners or final RNG states.
struct Output {
  DepthMap depth;
  NormalMap normal;
  Mat<float> probability;
  std::vector<std::pair<int, int>> sweeps;
};

void ExpectBitsEqual(const Mat<float>& a, const Mat<float>& b) {
  ASSERT_EQ(a.GetWidth(), b.GetWidth());
  ASSERT_EQ(a.GetHeight(), b.GetHeight());
  ASSERT_EQ(a.GetDepth(), b.GetDepth());
  ASSERT_EQ(a.GetData().size(), b.GetData().size());
  ASSERT_FALSE(a.GetData().empty());
  EXPECT_EQ(std::memcmp(a.GetData().data(),
                        b.GetData().data(),
                        a.GetData().size() * sizeof(float)),
            0);
  for (const float value : a.GetData()) EXPECT_TRUE(std::isfinite(value));
  for (const float value : b.GetData()) EXPECT_TRUE(std::isfinite(value));
}

TEST(OwnedDistinctSourceCache, RealSolverFinalMaps) {
  // Rectangular and nonmultiples of every supported tile in both orientations.
  constexpr int kWidth = 35;
  constexpr int kHeight = 19;
  const float K[9] = {24, 0, 9, 0, 24, 6, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  struct Case {
    int tile;
    int samples;
    int sources;
  };
  // S1/S20, repeated-source pressure, sample cap and dispatch fallback gates.
  // 15 sources/15 draws does NOT prove all 15 cache slots become occupied.
  const Case cases[] = {{16, 1, 1},
                        {16, 15, 1},
                        {16, 15, 3},
                        {16, 15, 15},
                        {16, 16, 3},
                        {8, 15, 3},
                        {16, 15, 20},
                        {32, 1, 1},
                        {32, 15, 3},
                        {32, 15, 20},
                        {32, 16, 20}};
  for (const bool geometry : {false, true}) {
    for (const bool filter : {false, true}) {
      // Owned geometry requires filtering; PHOTO supports both settings.
      // Rejection of unsupported flags is covered by the CPU execution-policy
      // contract and GeometryPreparationRejectsUnsupportedFlags host test.
      if (geometry && !filter) continue;
      for (const auto& test : cases) {
        SCOPED_TRACE(::testing::Message()
                     << "geometry=" << geometry << " filter=" << filter
                     << " tile=" << test.tile
                     << " samples=" << test.samples
                     << " sources=" << test.sources);
        std::vector<Image> images;
        std::vector<DepthMap> depths;
        std::vector<NormalMap> normals;
        PatchMatch::Problem problem;
        problem.ref_image_idx = 0;
        for (int image = 0; image <= test.sources; ++image) {
          const float T[3] = {0.04f * image, 0, 0};
          images.emplace_back("fixture", kWidth, kHeight, K, R, T);
          Bitmap bitmap(kWidth, kHeight, false);
          for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
              ASSERT_TRUE(bitmap.SetPixel(
                  x,
                  y,
                  BitmapColor<uint8_t>(
                      (17 * x + 31 * y + 7 * x * y + 11 * image) % 256)));
            }
          }
          images.back().SetBitmap(std::move(bitmap));
          depths.emplace_back(kWidth, kHeight, 1, 5);
          depths.back().Fill(3.0f);
          normals.emplace_back(kWidth, kHeight);
          for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
              normals.back().Set(y, x, 0, 0.0f);
              normals.back().Set(y, x, 1, 0.0f);
              normals.back().Set(y, x, 2, -1.0f);
            }
          }
          if (image != 0) problem.src_image_idxs.push_back(image);
        }
        problem.images = &images;
        problem.depth_maps = geometry ? &depths : nullptr;
        problem.normal_maps = geometry ? &normals : nullptr;
        PatchMatchOptions options;
        options.gpu_index = "0";
        options.random_seed = 0;
        options.depth_min = 1;
        options.depth_max = 5;
        options.sigma_spatial = 5;
        options.num_iterations = 1;
        options.num_samples = test.samples;
        options.geom_consistency = geometry;
        options.filter = filter;
        options.write_consistency_graph = false;
        options.allow_missing_files = false;
        options.filter_min_num_consistent = 1;
        const auto run = [&](bool cache, bool lookahead) {
          options.owned_distinct_source_cache = cache;
          options.owned_tex_lookahead = lookahead;
          PatchMatch solver(options, problem, PatchMatchExecution::kOwnedCuda);
          solver.Check();
          Output output;
          solver.PrepareOwned(test.tile);
          solver.SetOwnedSweepObserver([&](int iteration, int sweep) {
            output.sweeps.emplace_back(iteration, sweep);
          });
          solver.RunPreparedOwned();
          output.depth = solver.GetDepthMap();
          output.normal = solver.GetNormalMap();
          output.probability = solver.GetSelProbMap();
          return output;
        };
        const Output baseline = run(false, false);
        const Output cached = run(true, false);
        const std::vector<std::pair<int, int>> expected = {
            {1, 1}, {1, 2}, {1, 3}, {1, 4}};
        EXPECT_EQ(baseline.sweeps, expected);
        EXPECT_EQ(cached.sweeps, expected);
        EXPECT_EQ(baseline.depth.GetWidth(), kWidth);
        EXPECT_EQ(baseline.depth.GetHeight(), kHeight);
        ExpectBitsEqual(baseline.depth, cached.depth);
        ExpectBitsEqual(baseline.normal, cached.normal);
        ExpectBitsEqual(baseline.probability, cached.probability);
        // Include fallback cases: lookahead must not bypass the sample cap or
        // enable the unsupported tile8 cache specialization.
        {
          const Output lookahead = run(true, true);
          EXPECT_EQ(lookahead.sweeps, expected);
          ExpectBitsEqual(baseline.depth, lookahead.depth);
          ExpectBitsEqual(baseline.normal, lookahead.normal);
          ExpectBitsEqual(baseline.probability, lookahead.probability);
        }
      }
    }
  }
}

}  // namespace
}  // namespace colmap::mvs
