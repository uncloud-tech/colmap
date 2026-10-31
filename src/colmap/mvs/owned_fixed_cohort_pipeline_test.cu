// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_cohort_lifecycle_test_util.h"
#include "colmap/mvs/owned_fixed_cohort_pipeline.h"
#include "colmap/mvs/patch_match_cuda_owned.h"

#include <cstring>
#include <limits>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
std::vector<Image> Images(int count = 2) {
  const float K[9] = {10, 0, 2, 0, 10, 2, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  std::vector<Image> images;
  for (int i = 0; i < count; ++i) {
    const int width = i % 2 ? 3 : 5;
    const int height = i % 2 ? 3 : 4;
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
  b.geo_staging_device = {mib, mib};
  b.headroom_reserve = {mib, mib};
  b.photo_private = {{8 * mib, 8 * mib}, {8 * mib, 8 * mib}};
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

// More logical producers than active owners; each source ordinal, including
// duplicates, must bind to its producer rather than the reused physical slot.
TEST(FixedCohortPipeline, FourJobsPreserveIdentityAcrossRollingOwners) {
  auto images = Images(4);
  int device = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&device));
  std::optional<FixedCohortResult> oracle;
  for (const bool graphs : {false, true}) {
    for (const size_t requested :
         {size_t{1}, size_t{2}, size_t{3}, size_t{4}, size_t{0}}) {
      auto pair = References(device, graphs);
      std::vector<FixedCohortReference> refs = {
          pair[0], pair[1], pair[0], pair[1]};
      const std::vector<int> ids = {3, 0, 2, 1};
      for (size_t i = 0; i < refs.size(); ++i) {
        refs[i].reference_image_idx = ids[i];
        refs[i].photo_sources = {
            ids[(i + 1) % 4], ids[(i + 2) % 4], ids[(i + 1) % 4]};
        refs[i].geo_sources = refs[i].photo_sources;
      }
      for (const auto storage : {FixedCohortPhotoSourceStorage::kPrivate,
                                 FixedCohortPhotoSourceStorage::kShared}) {
        auto budget = Budget();
        budget.photo_source_storage = storage;
        if (storage == FixedCohortPhotoSourceStorage::kShared)
          budget.photo_shared = {80, 80};  // four distinct global 5x4 layers
        budget.max_active_owners = requested ? requested : 1;
        budget.photo_private.resize(4, budget.photo_private.front());
        budget.geo_private.resize(4, budget.geo_private.front());
        budget.geo_staging_device.resize(4, budget.geo_staging_device.front());
        const auto active = requested
                                ? requested
                                : SelectFixedCohortActiveOwners(
                                      budget, {128ull << 20, 128ull << 20})
                                      .count;
        budget.max_active_owners = active;
        const auto dispatch =
            active == 1 ? PairDispatch::kSerial : PairDispatch::kPair;
        CohortLifecycleRendezvous lifecycle(active);
        const auto result = RunFixedCohortPhotoGeo(
            images,
            refs,
            budget,
            {128ull << 20, 128ull << 20},
            NativeRunDeadline::FromBudget(std::chrono::seconds(60)),
            dispatch,
            32,
            nullptr,
            [&](bool geo, size_t owner, int iteration, int sweep) {
              if (iteration == 1 && sweep == 1)
                lifecycle.Observe(geo ? 1 : 0, owner);
            });
        // First-sweep callbacks hold live solvers until all first-cohort
        // owners arrive, independently of requested/selected count. Enabling
        // this diagnostic synchronizes sweeps; no kernel overlap is claimed.
        EXPECT_EQ(lifecycle.Observed(0), active);
        EXPECT_EQ(lifecycle.Observed(1), active);
        EXPECT_EQ(result.actual_photo_shared.host,
                  storage == FixedCohortPhotoSourceStorage::kShared ? 80 : 0);
        EXPECT_EQ(result.actual_photo_shared.device,
                  result.actual_photo_shared.host);
        ASSERT_EQ(result.photo.size(), 4);
        ASSERT_EQ(result.geo.size(), 4);
        ASSERT_EQ(result.actual_geo_staging_bytes.size(), 4);
        const auto key =
            std::string(graphs ? "graph" : "stream") +
            (requested ? "_active" : "_auto_active") + std::to_string(active) +
            (storage == FixedCohortPhotoSourceStorage::kShared ? "_shared"
                                                               : "_private");
        RecordProperty(key + "_selected_active_owners", std::to_string(active));
        RecordProperty(key + "_observed_photo_owner_highwater",
                       std::to_string(lifecycle.Observed(0)));
        RecordProperty(key + "_observed_geo_owner_highwater",
                       std::to_string(lifecycle.Observed(1)));
        RecordProperty(key + "_actual_photo_shared_host",
                       std::to_string(result.actual_photo_shared.host));
        RecordProperty(key + "_actual_photo_shared_device",
                       std::to_string(result.actual_photo_shared.device));
        RecordProperty(key + "_declared_photo_shared_host",
                       std::to_string(budget.photo_shared.host));
        RecordProperty(key + "_declared_photo_shared_device",
                       std::to_string(budget.photo_shared.device));
        RecordProperty(key + "_before_copy_available_host",
                       std::to_string(result.before_copy_available.host));
        RecordProperty(key + "_before_copy_available_device",
                       std::to_string(result.before_copy_available.device));
        RecordProperty(key + "_before_geo_available_host",
                       std::to_string(result.before_geo_available.host));
        RecordProperty(key + "_before_geo_available_device",
                       std::to_string(result.before_geo_available.device));
        RecordProperty(key + "_snapshot_bytes",
                       std::to_string(result.actual_snapshot_bytes));
        RecordProperty(key + "_declared_snapshot_bytes",
                       std::to_string(budget.snapshots.device));
        const auto peak = FixedCohortPeaks(budget).peak;
        RecordProperty(key + "_declared_raw_host_peak",
                       std::to_string(peak.host));
        RecordProperty(key + "_declared_raw_device_peak",
                       std::to_string(peak.device));
        std::vector<DepthMap> depths(4);
        std::vector<NormalMap> normals(4);
        for (size_t i = 0; i < refs.size(); ++i) {
          RecordProperty(key + "_staging_bytes_job" + std::to_string(i),
                         std::to_string(result.actual_geo_staging_bytes[i]));
          RecordProperty(
              key + "_declared_staging_bytes_job" + std::to_string(i),
              std::to_string(budget.geo_staging_device[i]));
          PatchMatch::Problem problem;
          problem.images = &images;
          problem.ref_image_idx = ids[i];
          problem.src_image_idxs = refs[i].photo_sources;
          PatchMatchCuda control(refs[i].photo_options, problem);
          control.Run();
          ExpectBits(control.GetDepthMap().GetData(),
                     result.photo[i].depth.GetData());
          ExpectBits(control.GetNormalMap().GetData(),
                     result.photo[i].normal.GetData());
          depths[ids[i]] = result.photo[i].depth;
          normals[ids[i]] = result.photo[i].normal;
        }
        for (size_t i = 0; i < refs.size(); ++i) {
          PatchMatch::Problem problem;
          problem.images = &images;
          problem.ref_image_idx = ids[i];
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
        std::atomic<int> geo_sweeps{0};
        std::atomic<bool> late_photo{false};
        std::atomic<bool> tail_photo_started{false};
        EXPECT_THROW(
            RunFixedCohortPhotoGeo(
                images,
                refs,
                budget,
                {128ull << 20, 128ull << 20},
                NativeRunDeadline::FromBudget(std::chrono::seconds(60)),
                dispatch,
                32,
                nullptr,
                [&](bool geo, size_t job, int, int) {
                  if (geo) ++geo_sweeps;
                  if (!geo && job == 3) tail_photo_started = true;
                  if (!geo && job == 2) {
                    late_photo = true;
                    throw std::runtime_error("late PHOTO failure");
                  }
                }),
            std::runtime_error);
        EXPECT_TRUE(late_photo.load());
        if (active == 1) EXPECT_FALSE(tail_photo_started.load());
        EXPECT_EQ(geo_sweeps.load(), 0);
        if (!requested) {
          const auto retry = RunFixedCohortPhotoGeo(
              images,
              refs,
              budget,
              {128ull << 20, 128ull << 20},
              NativeRunDeadline::FromBudget(std::chrono::seconds(60)),
              dispatch);
          for (size_t i = 0; i < refs.size(); ++i) {
            ExpectBits(result.photo[i].depth.GetData(),
                       retry.photo[i].depth.GetData());
            ExpectBits(result.photo[i].normal.GetData(),
                       retry.photo[i].normal.GetData());
            ExpectBits(result.geo[i].depth.GetData(),
                       retry.geo[i].depth.GetData());
            ExpectBits(result.geo[i].normal.GetData(),
                       retry.geo[i].normal.GetData());
          }
        }
      }
    }
  }
}

TEST(FixedCohortPipeline, AllFourProducerMapsMustBeRetainedBeforeDeviceWork) {
  const auto images = Images(4);
  const auto pair = References(0, false);
  std::vector<FixedCohortReference> refs = {pair[0], pair[1], pair[0], pair[1]};
  const std::vector<int> ids = {3, 0, 2, 1};
  for (size_t i = 0; i < refs.size(); ++i) {
    refs[i].reference_image_idx = ids[i];
    refs[i].photo_sources = {ids[(i + 1) % 4]};
    refs[i].geo_sources = refs[i].photo_sources;
  }
  auto budget = Budget();
  budget.photo_private.resize(4, budget.photo_private.front());
  budget.geo_private.resize(4, budget.geo_private.front());
  budget.geo_staging_device.resize(4, budget.geo_staging_device.front());
  const auto run = [&](const FixedCohortBudget& b) {
    return RunFixedCohortPhotoGeo(
        images,
        refs,
        b,
        {128ull << 20, 128ull << 20},
        NativeRunDeadline::FromBudget(std::chrono::seconds(5)));
  };
  // Four maps = (20+9+20+9)*16 = 928 raw bytes; an active-pair
  // retention allowance alone is insufficient even with one aggregate margin.
  budget.snapshots.device = 464;
  EXPECT_THROW(run(budget), std::invalid_argument);
  budget = Budget();
  budget.photo_private.resize(4, budget.photo_private.front());
  budget.geo_private.resize(4, budget.geo_private.front());
  budget.geo_staging_device.resize(4, budget.geo_staging_device.front());
  budget.host_outputs.host = 928;  // Both returned phases require 1856.
  EXPECT_THROW(run(budget), std::invalid_argument);
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
  auto shared = Budget();
  shared.photo_source_storage = FixedCohortPhotoSourceStorage::kShared;
  shared.photo_shared = {39, 39};  // needs two global 5x4 layers, before CUDA
  EXPECT_THROW(run(refs, shared, cap), std::invalid_argument);
  shared.photo_source_storage = FixedCohortPhotoSourceStorage::kPrivate;
  EXPECT_THROW(run(refs, shared, cap), std::invalid_argument);
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
  std::vector<FixedCohortReference> vector_refs(refs.begin(), refs.end());
  EXPECT_THROW(run(std::vector<FixedCohortReference>{}, Budget(), cap),
               std::invalid_argument);
  vector_refs.push_back(refs.front());
  EXPECT_THROW(run(vector_refs, Budget(), cap), std::invalid_argument);
  vector_refs.assign(refs.begin(), refs.end());
  auto mismatched = Budget();
  mismatched.max_active_owners = 1;
  EXPECT_THROW(run(vector_refs, mismatched, cap), std::invalid_argument);
  mismatched = Budget();
  mismatched.geo_private.pop_back();
  EXPECT_THROW(run(vector_refs, mismatched, cap), std::invalid_argument);
  mismatched = Budget();
  mismatched.geo_staging_device.pop_back();
  EXPECT_THROW(run(vector_refs, mismatched, cap), std::invalid_argument);
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
