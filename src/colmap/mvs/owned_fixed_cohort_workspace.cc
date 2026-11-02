// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_fixed_cohort_workspace.h"

#include "colmap/mvs/owned_fixed_cohort_validation.h"
#include "colmap/mvs/owned_tile.h"
#include "colmap/mvs/workspace.h"
#include "colmap/scene/reconstruction.h"

#include <algorithm>
#include <limits>

#include <OpenImageIO/imageio.h>

namespace colmap::mvs::owned {
namespace {
void Require(bool valid, const char* message) {
  if (!valid) throw std::invalid_argument(message);
}

// Does not decode pixels. These checks protect Bitmap's signed-int allocation
// arithmetic and bind decoded dimensions to the undistorted camera dimensions.
// They do not bound codec allocations or establish physical fit.
uint64_t HeaderScratchMinimum(const Image& original, const Image& resized) {
  OIIO::ImageSpec config;
  config["oiio:reorient"] = 0;
  config["oiio:UnassociatedAlpha"] = 1;
  const auto input =
      OIIO::ImageInput::open(original.GetPath().u8string(), &config);
  if (!input) {
    OIIO::geterror();
    throw std::invalid_argument("fixed-cohort cannot read image header");
  }
  const auto spec = input->spec();
  input->close();
  Require(spec.width > 0 && spec.height > 0 && spec.depth == 1 &&
              spec.nchannels >= 1 && spec.nchannels <= 4 &&
              static_cast<size_t>(spec.width) == original.GetWidth() &&
              static_cast<size_t>(spec.height) == original.GetHeight(),
          "fixed-cohort header/camera shape mismatch or unsupported channels");
  const uint64_t channels = spec.nchannels >= 3 ? 3 : 1;
  const auto area = CheckedMultiply(spec.width, spec.height);
  Require(CheckedMultiply(area, channels) <=
              static_cast<uint64_t>(std::numeric_limits<int>::max()),
          "fixed-cohort decoded bitmap exceeds signed allocation range");
  const auto resized_area =
      CheckedMultiply(resized.GetWidth(), resized.GetHeight());
  Require(resized_area && resized_area <= static_cast<uint64_t>(
                                              std::numeric_limits<int>::max()),
          "fixed-cohort resized bitmap exceeds signed allocation range");
  // Source RGB (or grey), simultaneous converted grey, and resize destination.
  // Conservative logical scratch; cache-bank copies are charged separately.
  return CheckedAdd(CheckedMultiply(area, channels + 1), resized_area);
}
}  // namespace

void ValidateFixedCohortWorkspaceCameras(
    const std::filesystem::path& workspace_path) {
  // Model intentionally stores only K, so check the camera model before that
  // information is discarded. No new sparse-model parser or calibration math.
  Reconstruction reconstruction;
  reconstruction.Read(workspace_path / "sparse");
  for (const auto& [id, camera] : reconstruction.Cameras())
    Require(camera.IsUndistorted(),
            "fixed-cohort requires undistorted cameras");
}

PreparedFixedCohortWorkspace PrepareFixedCohortWorkspace(
    const std::filesystem::path& workspace_path,
    int max_load_image_size,
    const std::vector<FixedCohortWorkspaceReference>& references,
    const FixedCohortBudget& budget,
    ByteCharge capacity,
    uint64_t loader_scratch_host,
    NativeRunDeadline deadline,
    PairDispatch dispatch,
    int tile_threads) {
  Require(max_load_image_size == -1 || max_load_image_size > 0,
          "fixed-cohort invalid load image size");
  Require(dispatch == PairDispatch::kSerial || dispatch == PairDispatch::kPair,
          "fixed-cohort invalid dispatch");
  Require((dispatch == PairDispatch::kSerial ? budget.max_active_owners == 1
                                             : budget.max_active_owners >= 2),
          "fixed-cohort active owners disagree with dispatch");
  ValidateOwnedTileThreads(tile_threads);
  Require(
      deadline.budget.count() > 0 &&
          deadline.entry <= NativeRunDeadline::Clock::now() &&
          NativeRunDeadline::FromBudget(deadline.budget, deadline.entry).end ==
              deadline.end,
      "fixed-cohort finite original deadline required");
  if (deadline.Remaining().count() <= 0)
    throw std::runtime_error(
        "fixed-cohort deadline expired before preparation");
  NativeRunDeadlineGuard guard(deadline);
  guard.Check();
  const auto admitted = AdmitFixedCohortBudget(budget, capacity);

  ValidateFixedCohortWorkspaceCameras(workspace_path);
  guard.Check();
  PreparedFixedCohortWorkspace result;
  result.budget_ = budget;
  result.capacity_ = capacity;
  result.deadline_ = deadline;
  result.dispatch_ = dispatch;
  result.tile_threads_ = tile_threads;

  {
    Workspace::Options options;
    options.workspace_path = workspace_path;
    options.workspace_format = "COLMAP";
    options.image_as_rgb = false;
    // Reserve the full possible cache bank (eviction can only reduce it).
    // Destroy it before returning / entering native Run.
    options.cache_size =
        std::max(1.0, budget.caller_inputs.host / (1024.0 * 1024.0 * 1024.0));
    CachedWorkspace workspace(options);
    const auto& model = workspace.GetModel();
    result.images_ = model.images;
    for (size_t i = 0; i < result.images_.size(); ++i) {
      result.names_.push_back(model.GetImageName(i));
      if (max_load_image_size > 0)
        result.images_[i].Downsize(max_load_image_size, max_load_image_size);
    }
    for (const auto& named : references) {
      FixedCohortReference ref;
      ref.reference_image_idx = model.GetImageIdx(named.reference);
      for (const auto& name : named.photo_sources)
        ref.photo_sources.push_back(model.GetImageIdx(name));
      for (const auto& name : named.geo_sources)
        ref.geo_sources.push_back(model.GetImageIdx(name));
      ref.photo_options = named.photo_options;
      ref.geo_options = named.geo_options;
      result.references_.push_back(std::move(ref));
    }
    ValidateFixedCohortShape(result.images_, result.references_, budget);
    uint64_t known_scratch = 0;
    uint64_t original_bank = 0;
    for (size_t i = 0; i < result.images_.size(); ++i) {
      guard.Check();
      known_scratch =
          std::max(known_scratch,
                   HeaderScratchMinimum(model.images[i], result.images_[i]));
      original_bank = CheckedAdd(original_bank,
                                 CheckedMultiply(model.images[i].GetWidth(),
                                                 model.images[i].GetHeight()));
    }
    Require(loader_scratch_host >= known_scratch,
            "fixed-cohort loader scratch below known logical inventory");
    // Cache contains original grayscale pixels; the prepared bank may be
    // resized. One aggregate margin per lifecycle peak, never margin-on-margin.
    const auto loader_raw = CheckedAdd(
        CheckedAdd(original_bank, budget.caller_inputs.host),
        CheckedAdd(loader_scratch_host,
                   CheckedAdd(budget.process.host, budget.unknown.host)));
    const auto host_demand = std::max(admitted.host, MarginBytes(loader_raw));
    Require(
        CheckedAdd(host_demand, budget.headroom_reserve.host) <= capacity.host,
        "fixed-cohort loader peak exceeds host capacity");
    const auto host_free = ObserveFixedCohortHostAvailable();
    CheckFixedCohortHeadroom({host_demand, 0},
                             {host_free, 0},
                             {0, 0},
                             {budget.headroom_reserve.host, 0});
    for (size_t i = 0; i < result.images_.size(); ++i) {
      guard.Check();
      // GetBitmap decodes using the existing native RGB->grey policy. Rescale
      // using the same Bitmap/Image routines as CachedWorkspace's size option.
      Bitmap bitmap = workspace.GetBitmap(i);
      if (max_load_image_size > 0)
        bitmap.Rescale(result.images_[i].GetWidth(),
                       result.images_[i].GetHeight());
      result.images_[i].SetBitmap(std::move(bitmap));
    }
  }
  guard.Check();
  ValidateFixedCohortPixels(result.images_, result.references_, budget);
  return result;
}

#if !defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
std::vector<FixedCohortDeviceObservation> ObserveFixedCohortDevices(
    const NativeRunDeadlineGuard& guard) {
  guard.Check();
  throw std::runtime_error("visible device inventory requires native CUDA; no fallback");
}

FixedCohortCapacityObservation ObserveFixedCohortCapacities(int) {
  throw std::runtime_error(
      "fixed-cohort capacity observation requires native CUDA; unsupported "
      "build, no manual fallback");
}

FixedCohortResult PreparedFixedCohortWorkspace::Run(
    const std::atomic<bool>*) const {
  throw std::runtime_error(
      "fixed-cohort workspace execution requires native CUDA");
}
#endif
}  // namespace colmap::mvs::owned
