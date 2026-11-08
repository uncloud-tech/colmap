// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/patch_match_cuda_owned.h"
#if defined(COLMAP_STRIP_CUDA_ALLOCATIONS)
#include "colmap/mvs/owned_strip_allocation_test_util.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <tuple>
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

#if defined(COLMAP_STRIP_CUDA_ALLOCATIONS)
// Independent exact request inventory for this fixture, not the production
// rotation-layout builder. S=3 aliases normal shapes, so exact counts matter:
// an extra cost/selection twin must fail even if allocated and freed
// transiently.
void ExpectAllocations(const strip_test::Capture& capture,
                       size_t w,
                       size_t h,
                       size_t sources,
                       size_t columns,
                       bool filter,
                       bool geometry,
                       uint64_t staging_cap) {
  using strip_test::Kind;
  using Key = std::tuple<Kind, size_t, size_t, size_t>;
  std::map<Key, size_t> expected, actual;
  const auto add =
      [&](Kind kind, size_t row, size_t rows, size_t layers, size_t count) {
        expected[{kind, row, rows, layers}] += count;
      };
  const size_t extent = std::max(w, h);
  const size_t capacity = std::min(columns, extent);
  add(Kind::kPitched, capacity * 4, extent * sources, 1, 3);
  if (filter) add(Kind::kPitched, capacity, extent * sources, 1, 1);
  for (const auto shape :
       {std::pair<size_t, size_t>{w, h}, std::pair<size_t, size_t>{h, w}}) {
    add(Kind::kPitched, shape.first, shape.second, 1, 1);      // reference
    add(Kind::kPitched, shape.first * 4, shape.second, 1, 3);  // stats + depth
    add(Kind::kPitched, shape.first * 4, shape.second * 3, 1, 1);  // normal
  }
  const size_t perimeter = 2 * w + 2 * h - 4;
  add(Kind::kPitched, w * 48, h, 1, 1);  // full initialization RNG, once only
  add(Kind::kPitched, perimeter * 48, 1, 1, 1);        // retained boundary RNG
  add(Kind::kPitched, extent * 4, sources * 2, 1, 1);  // workspace
  add(Kind::kPitched, perimeter * 4, 1, 1, 9);  // eight inspectors + final
  add(Kind::kArray,
      w,
      h,
      1,
      2);  // temporary filter texture + reference parity0
  add(Kind::kArray, h, w, 1, 1);        // reference parity1
  add(Kind::kArray, w, h, sources, 1);  // resident source grayscale
  if (geometry)
    add(Kind::kArray, w * 4, h, sources, 1);  // resident source depth
  add(Kind::kArray, 43 * 4, sources, 1, 4);   // immutable poses
  uint64_t staging_bytes = 0;
  size_t staging_allocations = 0;
  for (const auto& a : capture.allocations) {
    ++actual[{a.kind, a.row_bytes, a.rows, a.layers}];
    std::cout << "OWNED_STRIP_TEST_ALLOCATION kind=" << int(a.kind)
              << " row_bytes=" << a.row_bytes << " rows=" << a.rows
              << " layers=" << a.layers << " returned_pitch=" << a.pitch
              << " allocated_event=" << a.allocated
              << " released_event=" << a.released << '\n';
    if (a.kind == Kind::kPitched) {
      EXPECT_GE(a.pitch, a.row_bytes);
      const bool staging =
          a.rows == extent * sources &&
          (a.row_bytes == capacity * 4 || (filter && a.row_bytes == capacity));
      if (staging) {
        ++staging_allocations;
        staging_bytes += a.pitch * a.rows;
        EXPECT_LT(a.allocated, capture.constructor_finished);
        EXPECT_GT(a.released, capture.last_sweep);
      }
    }
  }
  // Returned pitches give actual linear-allocation spans, including the full
  // initialization RNG transient. Array physical padding/context are not
  // inferred from their observed request extents.
  std::map<size_t, int64_t> live_changes;
  for (const auto& a : capture.allocations) {
    if (a.kind == Kind::kArray) continue;
    const auto span = static_cast<int64_t>(a.pitch * a.rows);
    live_changes[a.allocated] += span;
    if (a.released) live_changes[a.released] -= span;
  }
  int64_t live = 0, peak = 0;
  for (const auto& event : live_changes) {
    live += event.second;
    EXPECT_GE(live, 0);
    peak = std::max(peak, live);
  }
  EXPECT_EQ(live, 0);
  std::cout << "OWNED_STRIP_TEST_PEAK pitched_linear_span_bytes=" << peak
            << " staging_pitched_bytes=" << staging_bytes << '\n';
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(staging_allocations, filter ? 4 : 3);
  EXPECT_LE(staging_bytes, staging_cap);
  EXPECT_GT(capture.constructor_finished, 0);
  EXPECT_GT(capture.last_sweep, capture.constructor_finished);
  EXPECT_TRUE(capture.AllReleased());
}
#endif

#if defined(COLMAP_STRIP_CUDA_ALLOCATIONS)
void ExpectQueuedFailure(const PatchMatchOptions& options,
                         const PatchMatch::Problem& problem,
                         const StripStreamingOptions& streaming) {
  strip_test::Capture capture;
  CUDA_SAFE_CALL(
      cudaEventCreateWithFlags(&capture.fault_event, cudaEventDisableTiming));
  capture.fail_copy_after_submission = true;
  {
    strip_test::Scope capture_scope(capture);
    std::unique_ptr<PatchMatchCuda> solver;
    try {
      solver =
          std::make_unique<PatchMatchCuda>(options, problem, 32, streaming);
      solver->Run();
      ADD_FAILURE() << "queued-copy failure injection was not reached";
    } catch (const std::exception&) {
      EXPECT_TRUE(capture.fault_submitted);
      if (solver) {
        EXPECT_THROW(solver->GetDepthMap(), std::logic_error);
        EXPECT_THROW(solver->InspectQualification(kInspectCap),
                     std::logic_error);
      }
    }
    solver.reset();
  }
  EXPECT_TRUE(capture.fault_submitted);
  EXPECT_TRUE(capture.fault_drained);
  EXPECT_FALSE(capture.release_before_fault_drain);
  EXPECT_TRUE(capture.AllReleased());
  EXPECT_EQ(cudaEventQuery(capture.fault_event), cudaSuccess);
  CUDA_SAFE_CALL(cudaEventDestroy(capture.fault_event));
}
#endif

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
#if defined(COLMAP_STRIP_CUDA_ALLOCATIONS)
  strip_test::Capture capture;
  {
    strip_test::Scope capture_scope(capture);
#endif
    auto streamed =
        std::make_unique<PatchMatchCuda>(options, problem, 32, streaming);
#if defined(COLMAP_STRIP_CUDA_ALLOCATIONS)
    capture.constructor_finished = ++capture.sequence;
    // Opt-in real allocation mutants: the ordinary oracle MUST fail inventory
    // assertions even though these transient owners are freed before Run().
    if (const char* mutant = std::getenv("COLMAP_STEREO_FULL_VOLUME_MUTANT")) {
      const size_t w = problem.images->at(problem.ref_image_idx).GetWidth();
      const size_t h = problem.images->at(problem.ref_image_idx).GetHeight();
      const size_t sources = problem.src_image_idxs.size();
      const std::string kind(mutant);
      if (kind == "cost") {
        GpuMat<float> temporary(w, h, sources);
      } else if (kind == "rotation") {
        GpuMat<float> temporary(h, w, sources);
      } else if (kind == "mask") {
        GpuMat<uint8_t> temporary(w, h, sources);
      } else if (kind == "selection") {
        CudaSelectionStorage temporary(4 * w * h * sources);
      } else
        throw std::invalid_argument("unknown full-volume mutant");
    }
#endif
    size_t observed = 0;
    streamed->SetSweepObserver([&](int, int) {
#if defined(COLMAP_STRIP_CUDA_ALLOCATIONS)
      capture.last_sweep = ++capture.sequence;
#endif
      ASSERT_LT(observed, sweeps.size());
      ExpectState(sweeps[observed++],
                  streamed->InspectQualification(kInspectCap));
    });
    EXPECT_THROW(streamed->GetDepthMap(), std::logic_error);
    EXPECT_THROW(streamed->GetSelProbMap(), std::logic_error);
    streamed->Run();
    EXPECT_EQ(observed, 8);
    ExpectState(expected, streamed->InspectQualification(kInspectCap));
    ExpectBits(resident.GetDepthMap().GetData(),
               streamed->GetDepthMap().GetData());
    ExpectBits(resident.GetNormalMap().GetData(),
               streamed->GetNormalMap().GetData());
    if (options.filter) {
      EXPECT_EQ(resident.GetConsistentImageIdxs(),
                Consistency(expected.mask, problem.src_image_idxs));
      if (reject) {
        EXPECT_TRUE(std::all_of(expected.mask.GetData().begin(),
                                expected.mask.GetData().end(),
                                [](uint8_t v) { return v == 0; }));
      }
    }
    EXPECT_GT(streamed->StripBytesRead(), streaming.disk_bytes);
    EXPECT_GT(streamed->StripBytesWritten(), streaming.disk_bytes);
    EXPECT_THROW(streamed->InspectQualification(1), std::invalid_argument);
    EXPECT_THROW(streamed->GetConsistentImageIdxs(), std::logic_error);
    streamed.reset();
#if defined(COLMAP_STRIP_CUDA_ALLOCATIONS)
  }
  ExpectAllocations(capture,
                    problem.images->at(problem.ref_image_idx).GetWidth(),
                    problem.images->at(problem.ref_image_idx).GetHeight(),
                    problem.src_image_idxs.size(),
                    columns,
                    options.filter,
                    options.geom_consistency,
                    streaming.device_staging_bytes);
#endif
#if defined(COLMAP_STRIP_CUDA_ALLOCATIONS)
  ExpectQueuedFailure(options, problem, streaming);
#endif
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
#if !defined(__linux__) || !defined(COLMAP_STRIP_CUDA_ALLOCATIONS)
  GTEST_SKIP()
      << "native allocation qualification requires Linux static libraries";
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
