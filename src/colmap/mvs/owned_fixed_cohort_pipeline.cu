// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_fixed_cohort_pipeline.h"
#include "colmap/mvs/owned_fixed_cohort_validation.h"
#include "colmap/mvs/owned_fixed_cohort_workspace.h"

#if !defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/owned_geo_photo_input.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_tile.h"
#include "colmap/mvs/patch_match_cuda_owned.h"
#include "colmap/mvs/patch_match_execution.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>

namespace colmap::mvs::owned {
namespace {
void Require(bool valid, const char* message) {
  if (!valid) throw std::invalid_argument(message);
}

ByteCharge ObserveAvailable(int device) {
  const auto host = ObserveFixedCohortHostAvailable();
  CUDA_SAFE_CALL(cudaSetDevice(device));
  size_t free = 0, total = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free, &total));
  return {host, free};
}

}  // namespace

FixedCohortResult PreparedFixedCohortWorkspace::Run(
    const std::atomic<bool>* cancel) const {
  return RunFixedCohortPhotoGeo(images_, references_, budget_, capacity_,
                              deadline_, dispatch_, tile_threads_, cancel);
}

FixedCohortResult RunFixedCohortPhotoGeo(
    const std::vector<Image>& images,
    const std::vector<FixedCohortReference>& references,
    const FixedCohortBudget& budget,
    ByteCharge capacity,
    NativeRunDeadline deadline,
    PairDispatch dispatch,
    int tile_threads,
    const std::atomic<bool>* cancel,
    std::function<void(bool, size_t, int, int)> sweep_observer) {
  // Small metadata validation only before the envelope is accepted; no pixel
  // clone, context query, adapter or solver allocation precedes this boundary.
  ValidateOwnedTileThreads(tile_threads);
  Require(dispatch == PairDispatch::kSerial || dispatch == PairDispatch::kPair,
          "fixed-cohort invalid dispatch");
  Require(
      budget.max_active_owners == (dispatch == PairDispatch::kSerial ? 1u : 2u),
      "fixed-cohort active owners disagree with dispatch");
  const auto reserved = AdmitFixedCohortBudget(budget, capacity);
  ValidateFixedCohortPixels(images, references, budget);
  Require(
      deadline.budget.count() > 0 && deadline.end > deadline.entry &&
          deadline.entry <= NativeRunDeadline::Clock::now() &&
          NativeRunDeadline::FromBudget(deadline.budget, deadline.entry).end ==
              deadline.end,
      "fixed-cohort finite original deadline required");
  if (deadline.Remaining().count() <= 0)
    throw std::runtime_error("fixed-cohort deadline expired before setup");
  NativeRunDeadlineGuard guard(deadline);
  const auto check = [&] {
    guard.Check();
    if (cancel && cancel->load())
      throw std::runtime_error("fixed-cohort invocation cancelled");
  };
  check();
  const int device = FixedCohortDevice(references[0].photo_options);
  uint64_t caller_bytes = 0;
  for (const auto& image : images)
    caller_bytes = CheckedAdd(caller_bytes, image.GetBitmap().NumBytes());
  FixedCohortResult result;
  result.output_policy = budget.output_policy;
  if (budget.output_policy == FixedCohortOutputPolicy::kPhotoAndGeo)
    result.photo.resize(references.size());
  result.geo.resize(references.size());
  result.actual_geo_staging_bytes.resize(references.size());
  result.before_copy_available = ObserveAvailable(device);
  CheckFixedCohortPhaseHeadroom(
      budget, false, result.before_copy_available, {caller_bytes, 0});
  check();
  // One immutable invocation copy, shared read-only by private solver owners.
  auto frozen = images;
  const auto refs = references;
  std::vector<std::shared_ptr<const PhotoOutputSnapshot>> snapshots(
      refs.size());
  for (const bool geo : {false, true}) {
    check();
    if (geo) {
      uint64_t live_host = CheckedMultiply(2, caller_bytes);
      for (const auto& maps : result.photo)
        live_host = CheckedAdd(
            live_host,
            CheckedAdd(maps.depth.GetNumBytes(), maps.normal.GetNumBytes()));
      result.before_geo_available = ObserveAvailable(device);
      CheckFixedCohortPhaseHeadroom(budget,
                                    true,
                                    result.before_geo_available,
                                    {live_host, result.actual_snapshot_bytes});
    }
    const auto& charges = geo ? budget.geo_private : budget.photo_private;
    // Retain the entire once-margined peak, even in the smaller phase. No
    // per-owner re-margin or snapshot sharing credit is issued.
    const auto active =
        FixedCohortActivePeak(charges, budget.max_active_owners);
    const ByteCharge process{reserved.host - active.host,
                             reserved.device - active.device};
    // Use the once-margined reservation itself as the ledger capacity: spare
    // external cap space cannot admit more than the declared active envelope.
    PairCoordinator coordinator(reserved,
                                process,
                                budget.max_active_owners,
                                charges,
                                deadline.Remaining(),
                                {},
                                deadline);
    auto graphs = std::make_shared<GraphLaunchControl>(refs.size());
    const auto run = [&](size_t owner) {
      try {
        check();
        CUDA_SAFE_CALL(cudaSetDevice(device));
        coordinator.BeginPreparation(owner);
        check();
        const auto& ref = refs[owner];
        PatchMatch::Problem problem;
        problem.images = &frozen;
        problem.ref_image_idx = ref.reference_image_idx;
        problem.src_image_idxs = geo ? ref.geo_sources : ref.photo_sources;
        const auto& options = geo ? ref.geo_options : ref.photo_options;
        std::shared_ptr<const GeoPhotoInput> input;
        if (geo) {
          std::vector<GeoPhotoInput::Snapshot> sources;
          for (int index : problem.src_image_idxs) {
            const auto producer = std::find_if(
                refs.begin(), refs.end(), [&](const auto& candidate) {
                  return candidate.reference_image_idx == index;
                });
            Require(producer != refs.end(),
                    "fixed-cohort source producer missing");
            sources.push_back(snapshots[std::distance(refs.begin(), producer)]);
          }
          input = std::make_shared<GeoPhotoInput>(
              options, problem, snapshots[owner], std::move(sources));
          result.actual_geo_staging_bytes[owner] = input->SourceStagingBytes();
          CheckFixedCohortPhysicalCopies(budget,
                                         result.actual_snapshot_bytes,
                                         result.actual_geo_staging_bytes);
        }
        PatchMatchCuda solver(options, problem, tile_threads, std::move(input));
        if (options.owned_cuda_graphs)
          solver.SetGraphCoordinator(graphs, owner);
        if (cancel || sweep_observer)
          solver.SetSweepObserver([&](int iteration, int sweep) {
            if (sweep_observer) sweep_observer(geo, owner, iteration, sweep);
            check();
          });
        coordinator.ReadyAndWait(owner);
        check();
        solver.Run();
        check();
        coordinator.SolverDone(owner);
        coordinator.BeginOutput(owner);
        if (geo || budget.output_policy == FixedCohortOutputPolicy::kPhotoAndGeo) {
          auto& maps = geo ? result.geo[owner] : result.photo[owner];
          maps.depth = solver.GetDepthMap();
          maps.normal = solver.GetNormalMap();
        }
        if (!geo) snapshots[owner] = solver.SnapshotPhotoOutput();
        check();
        coordinator.EndOutput(owner);
        // solver drains/destructs here, before rolling retirement and join.
      } catch (...) {
        graphs->Cancel(std::current_exception());
        throw;
      }
    };
    RunRollingOwners(coordinator, run, [](size_t, const char*) {});
    check();
    if (!geo) {
      // Global PHOTO barrier: all successful owners drained, destroyed, joined.
      uint64_t actual = 0;
      for (const auto& snapshot : snapshots) {
        Require(static_cast<bool>(snapshot),
                "fixed-cohort PHOTO producer missing");
        actual = CheckedAdd(actual, snapshot->RetainedBytes());
      }
      CheckFixedCohortPhysicalCopies(
          budget, actual, result.actual_geo_staging_bytes);
      result.actual_snapshot_bytes = actual;
    }
  }
  check();
  return result;  // snapshots released only after all GEO owners joined/drained
}
}  // namespace colmap::mvs::owned
#endif
