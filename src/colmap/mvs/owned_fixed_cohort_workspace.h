// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_fixed_cohort_pipeline.h"

#include <filesystem>
#include <string>

namespace colmap::mvs::owned {

struct FixedCohortWorkspaceReference {
  std::string reference;
  std::vector<std::string> photo_sources;
  std::vector<std::string> geo_sources;
  PatchMatchOptions photo_options;
  PatchMatchOptions geo_options;
};

// Experimental data adapter, not a controller, forecast producer, or file
// writer. Owns immutable native pixels/calibration and exact resolved options.
// Native indices refer to Model order, never sparse reconstruction IDs.
class PreparedFixedCohortWorkspace {
 public:
  PreparedFixedCohortWorkspace(PreparedFixedCohortWorkspace&&) = default;
  PreparedFixedCohortWorkspace& operator=(PreparedFixedCohortWorkspace&&) =
      default;
  PreparedFixedCohortWorkspace(const PreparedFixedCohortWorkspace&) = delete;
  PreparedFixedCohortWorkspace& operator=(const PreparedFixedCohortWorkspace&) =
      delete;

  const std::vector<Image>& Images() const { return images_; }
  const std::vector<FixedCohortReference>& References() const {
    return references_;
  }
  const std::vector<std::string>& ImageNames() const { return names_; }

  FixedCohortOutputPolicy OutputPolicy() const { return budget_.output_policy; }

  // Retains ALL device PHOTO snapshots through final GEO drain, either policy.
  // Host PHOTO cardinality follows the frozen budget policy (see result).
  // Returns native depth/normal maps; no output paths or partial publication.
  // Uses the original preparation deadline, never creates a fresh budget.
  FixedCohortResult Run(const std::atomic<bool>* cancel = nullptr) const;

 private:
  PreparedFixedCohortWorkspace() = default;
  friend PreparedFixedCohortWorkspace PrepareFixedCohortWorkspace(
      const std::filesystem::path&,
      int,
      const std::vector<FixedCohortWorkspaceReference>&,
      const FixedCohortBudget&,
      ByteCharge,
      uint64_t,
      NativeRunDeadline,
      PairDispatch,
      int);
  std::vector<Image> images_;
  std::vector<std::string> names_;
  std::vector<FixedCohortReference> references_;
  FixedCohortBudget budget_;
  ByteCharge capacity_;
  NativeRunDeadline deadline_;
  PairDispatch dispatch_;
  int tile_threads_;
};

// Linux/cgroup-v2 only. Known UNDISTORTED COLMAP workspace only; caller keeps
// files immutable during preparation. All model images are loaded using
// existing native grayscale and calibration resizing routines. Names bind to
// Model indices without source sorting/deduplication or another controller
// source-resolution transformation. max_load_image_size is -1 or positive;
// phase max_image_size MUST remain -1. Depth/sigma/device and other phase
// options must already be fully resolved.
//
// budget is the existing RAW experimental native envelope, NOT a qualified
// estimator. Loaded pixels are caller_inputs; do NOT also put them in process
// or already_live_process. Cache duplicates exist only during preparation.
// loader_scratch_host is a separately justified RAW bound for decoder/resizer
// scratch (excluding those duplicates). Headers provide only a known logical
// minimum. Codec/OIIO/private transients are not physically bounded by headers;
// caller unknown/process allowances and external limits remain necessary.
// The loader peak (original cache bank + prepared bank + scratch + process
// + unknown) and native
// peak are each once-margined, with one shared unallocated headroom reserve.
// Real Linux/cgroup free-host observation is used, never caller-faked free RAM.
// Preparation conservatively takes no already_live_process credit; native Run
// alone uses the separately justified per-observation credits.
// GPU free observation remains at the existing native execution boundaries.
PreparedFixedCohortWorkspace PrepareFixedCohortWorkspace(
    const std::filesystem::path& workspace_path,
    int max_load_image_size,
    const std::vector<FixedCohortWorkspaceReference>& references,
    const FixedCohortBudget& budget,
    ByteCharge capacity,
    uint64_t loader_scratch_host,
    NativeRunDeadline deadline,
    PairDispatch dispatch,
    int tile_threads);

}  // namespace colmap::mvs::owned
