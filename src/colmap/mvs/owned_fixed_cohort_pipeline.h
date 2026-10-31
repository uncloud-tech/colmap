// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/owned_fixed_cohort_budget.h"
#include "colmap/mvs/patch_match_options.h"

#include <array>
#include <atomic>
#include <functional>
#include <utility>
#include <vector>

namespace colmap::mvs::owned {

struct FixedCohortReference {
  int reference_image_idx = -1;
  // Already resolved native orders, never re-resolved, sorted or deduplicated.
  std::vector<int> photo_sources;
  std::vector<int> geo_sources;
  PatchMatchOptions photo_options;
  PatchMatchOptions geo_options;
};

struct FixedCohortResult {
  struct Maps {
    DepthMap depth;
    NormalMap normal;
  };
  FixedCohortOutputPolicy output_policy = FixedCohortOutputPolicy::kPhotoAndGeo;
  // GEO always has N maps in caller reference order, NOT image/worker order.
  // PHOTO has N maps by default; kFinalGeoOnly returns zero PHOTO entries,
  // explicitly omitted, NOT N empty valid maps. Consumers must check policy.
  std::vector<Maps> photo;
  std::vector<Maps> geo;
  static constexpr const char* qualification =
      "experimental_caller_declared_envelope/unqualified";
  ByteCharge before_copy_available;
  ByteCharge before_geo_available;
  // Logical padded payload actually constructed/uploaded, not CUDA physical
  // allocation bytes. Zero in default private mode. Released at PHOTO barrier.
  ByteCharge actual_photo_shared;
  uint64_t actual_snapshot_bytes = 0;
  std::vector<uint64_t> actual_geo_staging_bytes;
};

// Experimental finite closed-cohort native entry only. Not the default
// PatchMatch/Controller/CLI dispatch, a general scene pipeline or an estimator.
// Caller supplies resolved phase options and immutable input
// pixels/calibration; it must not mutate them during this synchronous call.
// Validation and envelope admission precede deep copies. The runner then owns
// frozen image copies. Every GEO source must have a PHOTO producer in this
// cohort. Duplicate source ordinals are preserved; self sources and duplicate
// producer references refuse.
//
// All PHOTO producers run successfully, publish independent COPY-backed device
// snapshots, drain and join before any GEO consumer starts. GEO copies snapshot
// maps into private solver buffers/source texture; all snapshots remain
// retained until final GEO drain/join. Opt-in budget.photo_source_storage
// shares only PHOTO source grayscale: one padded loaded-pixel union, retained
// host binding-validation payload and device array until PHOTO drain/join.
// Global padding may increase charges; default remains private. No GEO sharing,
// rolling generation, disk output, fusion or full-scene qualification is
// claimed.
//
// Serial dispatch requires one active owner; concurrent (kPair) dispatch
// requires at least two, bounded by budget.max_active_owners. Every per-job
// budget vector matches references. Returned maps are HOST copies selected by
// budget.output_policy and charged explicitly. kFinalGeoOnly skips both PHOTO
// host getters, not device snapshots; final GEO outputs and all device snapshot
// lifetimes remain unchanged. The caller owns resource-bound provenance and
// real external resource limits; lower-bound checks and RAM/cgroup/GPU-free
// observations do not certify fit. budget.already_live_process supplies
// separately justified classified storage actually resident at EACH observation
// (before copy / before GEO). It stays included in the full reservation/caps
// but is not charged twice against free. Exclude pixels/maps/snapshots and
// unknown overhead; default zero is conservative. Canonical production forecast
// delivery is not implemented by this experimental entry. One finite original
// deadline covers setup, both phases, drain and joins (existing fail-closed
// NativeRunDeadlineGuard semantics). Cancellation throws, never returns partial
// results; optional sweep-boundary checks may synchronize graph execution.
FixedCohortResult RunFixedCohortPhotoGeo(
    const std::vector<Image>& images,
    const std::vector<FixedCohortReference>& references,
    const FixedCohortBudget& budget,
    ByteCharge capacity,
    NativeRunDeadline deadline,
    PairDispatch dispatch = PairDispatch::kPair,
    int tile_threads = 32,
    const std::atomic<bool>* cancel = nullptr,
    // Diagnostic only; enables sweep synchronization, not throughput timing.
    std::function<void(bool, size_t, int, int)> sweep_observer = {});

// Compatibility for the original two-reference fixture. A serial invocation
// uses one active owner even when supplied with the old default pair budget.
inline FixedCohortResult RunFixedCohortPhotoGeo(
    const std::vector<Image>& images,
    const std::array<FixedCohortReference, 2>& references,
    const FixedCohortBudget& budget,
    ByteCharge capacity,
    NativeRunDeadline deadline,
    PairDispatch dispatch = PairDispatch::kPair,
    int tile_threads = 32,
    const std::atomic<bool>* cancel = nullptr,
    std::function<void(bool, size_t, int, int)> sweep_observer = {}) {
  auto compatible = budget;
  if (dispatch == PairDispatch::kSerial) compatible.max_active_owners = 1;
  return RunFixedCohortPhotoGeo(
      images,
      std::vector<FixedCohortReference>(references.begin(), references.end()),
      compatible,
      capacity,
      deadline,
      dispatch,
      tile_threads,
      cancel,
      std::move(sweep_observer));
}

}  // namespace colmap::mvs::owned
