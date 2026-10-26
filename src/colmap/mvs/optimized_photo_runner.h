// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/optimized_photo_dispatch.h"
#include "colmap/mvs/optimized_photo_options.h"
#include "colmap/mvs/optimized_photo_prepared.h"
#include "colmap/mvs/optimized_photo_input_probe.h"
#include "colmap/mvs/optimized_photo_paths.h"
#include "colmap/mvs/optimized_photo_reservation.h"
#include "colmap/mvs/owned_pair_image.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_rotation_storage.h"
#include "colmap/mvs/owned_selection_boundary.h"
#include "colmap/mvs/patch_match.h"
#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/owned_graph.h"
#endif
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
inline void CheckOptimizedPhotoInputs(
    const mvs::PatchMatchOptions& options,
    const mvs::PatchMatch::Problem& problem,
    const OptimizedPhotoDispatch& dispatch,
    size_t generation) {
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
  const uint64_t staging =
      owned::CheckedMultiply(s, owned::CheckedMultiply(max_w, max_h));
  dispatch.WithCheckedInputs(
      generation, problem.ref_image_idx, problem.src_image_idxs,
      {cache, q, staging, static_cast<uint64_t>(ref.GetWidth()),
       static_cast<uint64_t>(ref.GetHeight()), s},
      [] {});
}

#if defined(COLMAP_CUDA_ENABLED) && defined(COLMAP_DOWNLOAD_ENABLED) && \
    !defined(COLMAP_HIP_ENABLED)
inline OptimizedConcurrencyObservation RunOptimizedPhoto(
    const PatchMatchOptions& options,
    const fs::path& workspace,
    const fs::path& config,
    const fs::path& forecast_path,
    owned::ByteCharge capacity,
    int selected_tile,
    std::optional<NativeRunDeadline> execution_deadline = {},
    const std::string& requested_concurrency = "1",
    const TrustedPhotoPhaseV1Delivery& phase_delivery = {},
    const OptimizedPhotoInputProbe& input_probe = {}) {
  const auto launch_bytes = ReadOptimizedBytes(forecast_path);
  const bool legacy_v2 =
      launch_bytes.rfind("COLMAP_OPTIMIZED_PHOTO_V2\n", 0) == 0;
  const auto dispatch = ParseOptimizedPhotoLaunch(
      launch_bytes, config, requested_concurrency, capacity, phase_delivery);
  if (input_probe) {
    input_probe([&](size_t generation, const PatchMatch::Problem& problem) {
      CheckOptimizedPhotoInputs(options, problem, dispatch, generation);
    });
    throw std::invalid_argument("PHOTO input probe completed without execution");
  }
  const auto& launch = dispatch.launch;
  const auto& phase_bindings = dispatch.bindings;
  owned::OwnedConcurrencyObserver observation(launch.admitted_n,
                                              launch.admitted_n);
  std::vector<owned::ByteCharge> charges;
  for (const auto& job : launch.jobs)
    charges.push_back(job.forecast.private_charge);
  owned::PairCoordinator coordinator(
      capacity,
      launch.jobs.front().forecast.process,
      launch.admitted_n,
      charges,
      std::chrono::milliseconds{300000},
      [&](const owned::PairEvent& event) { observation(event); },
      execution_deadline);
  // Shared cancellation only: each graph launches on its own owner stream.
  // Admission and the serialized host preparation boundary stay in force.
  std::shared_ptr<owned::GraphLaunchControl> graph_coordinator;
  if (options.owned_cuda_graphs && !legacy_v2) {
    const auto devices = CSVToVector<int>(options.gpu_index);
    if (devices.size() != 1 || devices.front() < 0)
      throw std::invalid_argument("PHOTO graphs require explicit GPU");
    graph_coordinator =
        std::make_shared<owned::GraphLaunchControl>(launch.admitted_n);
  }
  std::vector<owned::BankFileIdentity> identities;
  for (const auto& job : launch.jobs) {
    const auto& wire = job.forecast;
    THROW_CHECK_EQ(
        ComputeSHA256(OptimizedPhotoOptions(options, workspace, job.config)),
        wire.options_sha256);
    THROW_CHECK(!options.geom_consistency);
    THROW_CHECK_EQ(options.max_image_size, -1);
    THROW_CHECK_GT(options.num_threads, 0);
    const auto devices = CSVToVector<int>(options.gpu_index);
    THROW_CHECK_EQ(devices.size(), 1);
    THROW_CHECK_GE(devices.front(), 0);
    THROW_CHECK_LE(
        owned::CacheBytes(options.cache_size),
        phase_bindings.empty() ? wire.private_charge.host : wire.process.host);
    const auto require = [&](const fs::path& path) {
      THROW_CHECK(wire.files.count(fs::canonical(path).string()))
          << "PHOTO forecast missing effective input " << path;
    };
    require(job.config);
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
          const auto optional =
              workspace / "sparse" / (std::string(stem) + ext);
          if (fs::exists(optional)) require(optional);
        }
        model_found = true;
      }
    }
    THROW_CHECK(model_found);
    for (const auto& file : wire.files) {
      THROW_CHECK_EQ(fs::canonical(file.first).string(), file.first);
      auto identity = owned::CaptureFileIdentity(file.first);
      THROW_CHECK_LE(identity.size, wire.private_charge.host)
          << "encoded-file hash buffer exceeds private host forecast";
      THROW_CHECK_EQ(ComputeSHA256(ReadOptimizedBytes(file.first)),
                     file.second);
      owned::CheckFileIdentityUnchanged(identity);
      identities.push_back(std::move(identity));
    }
  }
  const auto unchanged = [&] {
    for (const auto& identity : identities)
      owned::CheckFileIdentityUnchanged(identity);
  };
  // One controller. Preparation owns ALL mutable workspace state (including
  // the CachedWorkspace LRU), not merely the synthetic selection-cache model.
  // Prepared PatchMatch objects snapshot their inputs; no workspace survives
  // this lease, and each CUDA solver retains its private selection state.
  PatchMatchController controller(options,
                                  fs::canonical(workspace),
                                  "colmap",
                                  "option-all",
                                  config,
                                  PatchMatchExecution::kOwnedCuda);
  owned::OwnedSharedController shared(
      launch.admitted_n, owned::kMaxSelectionSlots, launch.jobs.size());
  const auto run = [&](size_t generation) {
    const auto& job = launch.jobs.at(generation);
    const auto& wire = job.forecast;
    const auto require = [&](const fs::path& path) {
      THROW_CHECK(wire.files.count(fs::canonical(path).string()));
    };
    const auto owner = coordinator.Slot(generation);
    shared.Begin(owner);
    struct End {
      owned::OwnedSharedController& shared;
      size_t owner;
      ~End() { shared.End(owner); }
    } end{shared, owner};
    std::string prepared_effective_options;
    PatchMatchController::OwnedPairHooks hooks;
    hooks.owned_tile_threads = selected_tile;
    hooks.graph_coordinator = graph_coordinator;
    hooks.graph_owner = owner;
    hooks.stage_timing = [&](size_t job, const char* stage, uint64_t micros) {
      LOG(INFO) << "PHOTO_STAGE generation=" << job << " owner=" << owner
                << " mode=" << (graph_coordinator ? "graph" : "thread")
                << " stage=" << stage << " elapsed_us=" << micros;
    };
    hooks.check_inputs = [&](size_t,
                             const PatchMatchOptions& effective,
                             const PatchMatch::Problem& problem) {
      if (effective.geom_consistency)
        throw std::invalid_argument("PHOTO effective options are not photometric");
      prepared_effective_options =
          OptimizedPhotoOptions(effective, workspace, job.config);
      unchanged();
      require(problem.images->at(problem.ref_image_idx).GetPath());
      const auto root = fs::canonical(workspace);
      const auto name = problem.images->at(problem.ref_image_idx)
                            .GetPath()
                            .lexically_relative(root / "images");
      ValidateOptimizedPhotoOutputs(root, name);
      for (const int idx : problem.src_image_idxs)
        require(problem.images->at(idx).GetPath());
      CheckOptimizedPhotoInputs(effective, problem, dispatch, generation);
    };
    hooks.metadata = [&](size_t,
                         const PatchMatchOptions&,
                         const PatchMatch::Problem&) { unchanged(); };
    try {
      if (legacy_v2) {
        // Preserve today's V2 N1 execution as the compatibility oracle.
        shared.WithSelection(owner, [&](auto&) {
          controller.RunOwnedPairJob(coordinator, generation, hooks);
        });
        return;
      }
      coordinator.BeginPreparation(generation);
      const auto prep_start = std::chrono::steady_clock::now();
      auto prepared = shared.WithSelection(owner, [&](auto&) {
        return controller.PrepareSharedOwnedPhotoJob(
            job.config, generation, hooks);
      });
      hooks.stage_timing(generation,
                         "host_prepare",
                         std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - prep_start)
                             .count());
      // Bind this engine to the producer/generation and the effective PHOTO
      // snapshot captured during preparation, before dereference or CUDA work.
      CheckOptimizedPhotoPreparedJob(
          prepared, &controller, generation, prepared_effective_options,
          [&](const PatchMatchOptions& effective) {
            return OptimizedPhotoOptions(effective, workspace, job.config);
          });
      // The serialized selection boundary mutates under WithSelection above
      // (preparation/capture). Graph sweep execution runs OUTSIDE that lease,
      // exactly like the accepted prepared-job thread execution path.
      controller.ExecutePreparedOwnedPairJob(
          *prepared, coordinator, generation, hooks);
      // Destruction/drain remains before RunRollingOwners retirement/credit
      // reuse.
    } catch (...) {
      if (graph_coordinator)
        graph_coordinator->Cancel(std::current_exception());
      coordinator.Cancel(generation, std::current_exception());
      throw;
    }
  };
  owned::RunRollingOwners(coordinator, run, [](size_t, const char*) {});
  shared.Seal();
  unchanged();
  return observation.Observation();
}
#endif
}  // namespace colmap::mvs
