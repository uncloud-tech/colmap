// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_fixed_cohort_pipeline.h"
#include "colmap/mvs/patch_match_cuda_owned.h"

#include <cstring>
#include <limits>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
std::vector<Image> Images() {
  const float K[9] = {10, 0, 2, 0, 10, 2, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  std::vector<Image> images;
  for (int i = 0; i < 2; ++i) {
    const int width = i ? 3 : 5;
    const int height = i ? 3 : 4;
    images.emplace_back("fixed-" + std::to_string(i), width, height, K, R, T);
    Bitmap bitmap(width, height, false);
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        EXPECT_TRUE(bitmap.SetPixel(
            x, y, BitmapColor<uint8_t>(17 + 40 * i + x + 3 * y)));
    images.back().SetBitmap(std::move(bitmap));
  }
  return images;
}

std::array<FixedCohortReference, 2> References(int device, bool graphs) {
  std::array<FixedCohortReference, 2> references;
  for (int i = 0; i < 2; ++i) {
    auto& ref = references[i];
    // Deliberately non-image-index reference order.
    ref.reference_image_idx = 1 - i;
    ref.photo_sources = {i};
    ref.geo_sources = {i, i};
    auto& photo = ref.photo_options;
    photo.gpu_index = std::to_string(device);
    photo.geom_consistency = false;
    photo.depth_min = 1;
    photo.depth_max = 5;
    photo.sigma_spatial = 5;
    photo.num_iterations = 2;
    photo.num_samples = 2;
    photo.filter = false;
    photo.random_seed = 0;
    photo.num_threads = 1;
    photo.owned_cuda_graphs = graphs;
    ref.geo_options = photo;
    ref.geo_options.geom_consistency = true;
    ref.geo_options.filter = true;
    ref.geo_options.filter_min_num_consistent = 1;
  }
  return references;
}

FixedCohortBudget Budget() {
  // Independent, deliberately generous fixture bounds, NOT calibrated
  // production coefficients or evidence of a physical peak.
  constexpr uint64_t mib = 1024 * 1024;
  FixedCohortBudget b;
  b.caller_inputs = {mib, 0};
  b.frozen_inputs = {mib, 0};
  b.host_outputs = {mib, 0};
  b.snapshots = {0, mib};
  b.process = {16 * mib, 16 * mib};
  b.unknown = {8 * mib, 8 * mib};
  b.unknown_provenance =
      "tiny 5x4/3x3 fixture: 8MiB each for host metadata/allocator and "
      "device pitch/array/context uncertainty; not workload-calibrated";
  b.geo_staging_device = {{mib, mib}};
  b.headroom_reserve = {mib, mib};
  b.photo_private = {{{8 * mib, 8 * mib}, {8 * mib, 8 * mib}}};
  b.geo_private = b.photo_private;
  return b;
}

void ExpectBits(const std::vector<float>& expected,
                const std::vector<float>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  EXPECT_EQ(
      std::memcmp(
          expected.data(), actual.data(), expected.size() * sizeof(float)),
      0);
}

TEST(FixedCohortPipeline,
     PublicEntryMatchesFreshPrivateGeoInEveryExecutionMode) {
  auto images = Images();
  const auto original_pixels = images[0].GetBitmap().RowMajorData();
  int device = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&device));
  std::optional<FixedCohortResult> oracle;
  for (const bool graphs : {false, true}) {
    for (const auto dispatch : {PairDispatch::kSerial, PairDispatch::kPair}) {
      SCOPED_TRACE(graphs ? "graph" : "stream");
      SCOPED_TRACE(dispatch == PairDispatch::kSerial ? "serial" : "pair");
      const auto refs = References(device, graphs);
      const auto result = RunFixedCohortPhotoGeo(
          images,
          refs,
          Budget(),
          {128ull << 20, 128ull << 20},
          NativeRunDeadline::FromBudget(std::chrono::seconds(60)),
          dispatch);
      std::vector<DepthMap> depths(2);
      std::vector<NormalMap> normals(2);
      for (size_t i = 0; i < 2; ++i) {
        PatchMatch::Problem photo;
        photo.images = &images;
        photo.ref_image_idx = refs[i].reference_image_idx;
        photo.src_image_idxs = refs[i].photo_sources;
        PatchMatchCuda control(refs[i].photo_options, photo);
        control.Run();
        ExpectBits(control.GetDepthMap().GetData(),
                   result.photo[i].depth.GetData());
        ExpectBits(control.GetNormalMap().GetData(),
                   result.photo[i].normal.GetData());
        depths[refs[i].reference_image_idx] = result.photo[i].depth;
        normals[refs[i].reference_image_idx] = result.photo[i].normal;
      }
      for (size_t i = 0; i < 2; ++i) {
        PatchMatch::Problem problem;
        problem.images = &images;
        problem.ref_image_idx = refs[i].reference_image_idx;
        problem.src_image_idxs = refs[i].geo_sources;
        problem.depth_maps = &depths;
        problem.normal_maps = &normals;
        PatchMatchCuda control(refs[i].geo_options, problem);
        control.Run();
        ExpectBits(control.GetDepthMap().GetData(),
                   result.geo[i].depth.GetData());
        ExpectBits(control.GetNormalMap().GetData(),
                   result.geo[i].normal.GetData());
        if (oracle) {
          ExpectBits(oracle->photo[i].depth.GetData(),
                     result.photo[i].depth.GetData());
          ExpectBits(oracle->photo[i].normal.GetData(),
                     result.photo[i].normal.GetData());
          ExpectBits(oracle->geo[i].depth.GetData(),
                     result.geo[i].depth.GetData());
          ExpectBits(oracle->geo[i].normal.GetData(),
                     result.geo[i].normal.GetData());
        }
      }
      if (!oracle) oracle = result;
      EXPECT_EQ(images[0].GetBitmap().RowMajorData(), original_pixels);
    }
  }
}

TEST(FixedCohortPipeline, PublicEntryRefusesInvalidInputsBeforeDeviceWork) {
  const auto images = Images();
  const auto refs = References(0, false);
  const auto run = [&](const auto& r, FixedCohortBudget b, ByteCharge cap) {
    return RunFixedCohortPhotoGeo(
        images,
        r,
        b,
        cap,
        NativeRunDeadline::FromBudget(std::chrono::seconds(5)));
  };
  const ByteCharge cap{128ull << 20, 128ull << 20};
  EXPECT_THROW(run(refs, Budget(), {1, 1}), std::invalid_argument);
  auto wrong = refs;
  wrong[1].reference_image_idx = wrong[0].reference_image_idx;
  EXPECT_THROW(run(wrong, Budget(), cap), std::invalid_argument);
  wrong = refs;
  wrong[0].geo_sources = {2};
  EXPECT_THROW(run(wrong, Budget(), cap), std::invalid_argument);
  wrong = refs;
  wrong[0].geo_options.geom_consistency = false;
  EXPECT_THROW(run(wrong, Budget(), cap), std::invalid_argument);
  wrong = refs;
  wrong[0].geo_options.gpu_index = "1";
  EXPECT_THROW(run(wrong, Budget(), cap), std::invalid_argument);
  auto small = Budget();
  small.snapshots.device = 1;
  EXPECT_THROW(run(refs, small, cap), std::invalid_argument);
  std::atomic<bool> cancel{true};
  EXPECT_THROW(RunFixedCohortPhotoGeo(
                   images,
                   refs,
                   Budget(),
                   cap,
                   NativeRunDeadline::FromBudget(std::chrono::seconds(5)),
                   PairDispatch::kPair,
                   32,
                   &cancel),
               std::runtime_error);
}
TEST(FixedCohortPipeline, FailedPhotoCancelsAndDrainsWithoutStartingGeo) {
  auto images = Images();
  int device = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&device));
  const auto refs = References(device, true);
  std::atomic<bool> cancel{false};
  std::atomic<int> geo_sweeps{0};
  EXPECT_THROW(RunFixedCohortPhotoGeo(
                   images,
                   refs,
                   Budget(),
                   {128ull << 20, 128ull << 20},
                   NativeRunDeadline::FromBudget(std::chrono::seconds(60)),
                   PairDispatch::kPair,
                   32,
                   &cancel,
                   [&](bool geo, size_t, int, int) {
                     if (geo)
                       ++geo_sweeps;
                     else
                       cancel.store(true);
                   }),
               std::runtime_error);
  EXPECT_TRUE(cancel.load());
  EXPECT_EQ(geo_sweeps.load(), 0);
  // A new invocation owns fresh state, not cancelled/retained consumer state.
  cancel.store(false);
  EXPECT_NO_THROW(RunFixedCohortPhotoGeo(
      images,
      refs,
      Budget(),
      {128ull << 20, 128ull << 20},
      NativeRunDeadline::FromBudget(std::chrono::seconds(60)),
      PairDispatch::kPair,
      32,
      &cancel));
}

TEST(FixedCohortPipeline,
     InsufficientObservedHeadroomRefusesBeforeCopyOrSolver) {
  const auto images = Images();
  int device = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&device));
  auto budget = Budget();
  budget.frozen_inputs.host = 1ull << 62;
  std::atomic<bool> solver_started{false};
  EXPECT_THROW(RunFixedCohortPhotoGeo(
                   images,
                   References(device, false),
                   budget,
                   {std::numeric_limits<uint64_t>::max(),
                    std::numeric_limits<uint64_t>::max()},
                   NativeRunDeadline::FromBudget(std::chrono::seconds(5)),
                   PairDispatch::kPair,
                   32,
                   nullptr,
                   [&](bool, size_t, int, int) { solver_started = true; }),
               std::invalid_argument);
  EXPECT_FALSE(solver_started.load());
}
}  // namespace
}  // namespace colmap::mvs::owned
