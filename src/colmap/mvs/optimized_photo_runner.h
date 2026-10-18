// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/optimized_photo_forecast.h"
#include "colmap/mvs/optimized_photo_options.h"
#include "colmap/mvs/optimized_photo_paths.h"
#include "colmap/mvs/owned_pair_image.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_rotation_storage.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/string.h"

#include <fstream>

namespace colmap::mvs {
namespace fs = std::filesystem;
inline std::string ReadOptimizedBytes(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::invalid_argument("cannot read optimized input: " +
                                path.string());
  std::string bytes((std::istreambuf_iterator<char>(input)),
                    std::istreambuf_iterator<char>());
  if (input.bad()) throw std::runtime_error("optimized input read failed");
  return bytes;
}
inline void CheckOptimizedPhotoInputs(const mvs::PatchMatchOptions& options,
                                      const mvs::PatchMatch::Problem& problem,
                                      owned::ByteCharge charge,
                                      const fs::path& /*workspace*/) {
  const auto& images = *problem.images;
  const auto& ref = images.at(problem.ref_image_idx);
  const uint64_t a = owned::ReadPairBitmapPixels(
      ref.GetPath(), ref.GetWidth(), ref.GetHeight());
  const uint64_t s = problem.src_image_idxs.size();
  if (!a || !s)
    throw std::invalid_argument("empty reference/source dimensions");
  uint64_t q = a, max_w = 0, max_h = 0;
  std::set<int> used{problem.ref_image_idx};
  for (int idx : problem.src_image_idxs) {
    if (!used.insert(idx).second)
      throw std::invalid_argument("duplicate effective source");
    const auto& image = images.at(idx);
    if (!image.GetWidth() || !image.GetHeight())
      throw std::invalid_argument("empty source dimensions");
    q = owned::CheckedAdd(
        q,
        owned::ReadPairBitmapPixels(
            image.GetPath(), image.GetWidth(), image.GetHeight()));
    max_w = std::max(max_w, static_cast<uint64_t>(image.GetWidth()));
    max_h = std::max(max_h, static_cast<uint64_t>(image.GetHeight()));
    if (!fs::is_regular_file(image.GetPath()))
      throw std::invalid_argument("source image missing");
  }
  const uint64_t cache = owned::CacheBytes(options.cache_size);
  const uint64_t retained = q;
  if (cache < retained)
    throw std::invalid_argument(
        "cache must retain complete private PHOTO input set");
  const uint64_t u =
      owned::CheckedMultiply(s, owned::CheckedMultiply(max_w, max_h));
  // Local decoded inputs and the PatchMatch owned snapshot coexist during
  // construction: each retains grayscale q in PHOTO.
  // Source grayscale staging u and result/reference scratch are separate;
  // preserve the existing 24a result/reference scratch allowance.
  const uint64_t staging = u;
  const uint64_t host = owned::CheckedAdd(
      cache,
      owned::CheckedAdd(owned::CheckedMultiply(2, retained),
                        std::max(staging, owned::CheckedMultiply(24, a))));
  // Persistent pitched layout plus source/pose arrays and both live reference
  // arrays during rebinding. This logical minimum is NOT a physical bound.
  uint64_t device = owned::CheckedAdd(
      owned::BuildOwnedRotationLayout(ref.GetWidth(), ref.GetHeight(), s)
          .logical_bytes,
      u);
  device = owned::CheckedAdd(device, owned::CheckedMultiply(2, a));
  device = owned::CheckedAdd(device, owned::CheckedMultiply(688, s));
  device =
      owned::CheckedAdd(device, 32);  // Retained calibration policy allowance.
  if (charge.host < host || charge.device < device)
    throw std::invalid_argument(
        "job reservation below logical workload envelope");
}

#if defined(COLMAP_CUDA_ENABLED) && defined(COLMAP_DOWNLOAD_ENABLED) && \
    !defined(COLMAP_HIP_ENABLED)
inline void RunOptimizedPhoto(
    const PatchMatchOptions& options,
    const fs::path& workspace,
    const fs::path& config,
    const fs::path& forecast_path,
    owned::ByteCharge capacity,
    int selected_tile,
    std::optional<NativeRunDeadline> execution_deadline = {}) {
  const auto wire =
      ParseOptimizedPhotoForecast(ReadOptimizedBytes(forecast_path));
  THROW_CHECK_EQ(
      ComputeSHA256(OptimizedPhotoOptions(options, workspace, config)),
      wire.options_sha256);
  THROW_CHECK(!options.geom_consistency);
  THROW_CHECK_EQ(options.max_image_size, -1);
  THROW_CHECK_GT(options.num_threads, 0);
  const auto devices = CSVToVector<int>(options.gpu_index);
  THROW_CHECK_EQ(devices.size(), 1);
  THROW_CHECK_GE(devices.front(), 0);
  THROW_CHECK_LE(owned::CacheBytes(options.cache_size),
                 wire.private_charge.host);
  const auto require = [&](const fs::path& path) {
    THROW_CHECK(wire.files.count(fs::canonical(path).string()))
        << "PHOTO forecast missing effective input " << path;
  };
  require(config);
  bool model_found = false;
  for (const auto* ext : {".bin", ".txt"}) {
    bool complete = true;
    for (const auto* stem : {"cameras", "images", "points3D"})
      complete = complete && fs::is_regular_file(workspace / "sparse" /
                                                 (std::string(stem) + ext));
    if (complete) {
      for (const auto* stem : {"cameras", "images", "points3D"})
        require(workspace / "sparse" / (std::string(stem) + ext));
      for (const auto* stem : {"rigs", "frames"}) {
        const auto optional = workspace / "sparse" / (std::string(stem) + ext);
        if (fs::exists(optional)) require(optional);
      }
      model_found = true;
    }
  }
  THROW_CHECK(model_found);
  // Covers header/hash scratch as well as decode. Physical qualification and
  // the single margin remain the reviewed forecaster's responsibility.
  owned::PairCoordinator coordinator(
      capacity,
      wire.process,
      size_t{1},
      std::vector<owned::ByteCharge>{wire.private_charge},
      std::chrono::milliseconds{300000},
      {},
      execution_deadline);
  std::vector<owned::BankFileIdentity> identities;
  for (const auto& file : wire.files) {
    THROW_CHECK_EQ(fs::canonical(file.first).string(), file.first);
    auto identity = owned::CaptureFileIdentity(file.first);
    THROW_CHECK_LE(identity.size, wire.private_charge.host)
        << "encoded-file hash buffer exceeds private host forecast";
    THROW_CHECK_EQ(ComputeSHA256(ReadOptimizedBytes(file.first)), file.second);
    owned::CheckFileIdentityUnchanged(identity);
    identities.push_back(std::move(identity));
  }
  const auto unchanged = [&] {
    for (const auto& identity : identities)
      owned::CheckFileIdentityUnchanged(identity);
  };
  const auto run = [&](size_t generation) {
    PatchMatchController controller(options,
                                    fs::canonical(workspace),
                                    "colmap",
                                    "option-all",
                                    config,
                                    PatchMatchExecution::kOwnedCuda);
    PatchMatchController::OwnedPairHooks hooks;
    hooks.owned_tile_threads = selected_tile;
    hooks.check_inputs = [&](size_t,
                             const PatchMatchOptions& effective,
                             const PatchMatch::Problem& problem) {
      unchanged();
      ValidateOptimizedPhotoOrder(
          wire, problem.ref_image_idx, problem.src_image_idxs);
      require(problem.images->at(problem.ref_image_idx).GetPath());
      const auto root = fs::canonical(workspace);
      const auto name = problem.images->at(problem.ref_image_idx)
                            .GetPath()
                            .lexically_relative(root / "images");
      ValidateOptimizedPhotoOutputs(root, name);
      for (const int idx : problem.src_image_idxs)
        require(problem.images->at(idx).GetPath());
      CheckOptimizedPhotoInputs(
          effective, problem, wire.private_charge, workspace);
    };
    hooks.metadata = [&](size_t,
                         const PatchMatchOptions&,
                         const PatchMatch::Problem&) { unchanged(); };
    controller.RunOwnedPairJob(coordinator, generation, hooks);
  };
  owned::RunRollingOwners(coordinator, run, [](size_t, const char*) {});
  unchanged();
}
#endif
}  // namespace colmap::mvs
