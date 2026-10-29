// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_fixed_cohort_pipeline.h"

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

// Runner-local Linux/cgroup-v2 observation, not a resource library/hard cap.
ByteCharge ObserveAvailable(int device) {
#if defined(__linux__)
  uint64_t available = 0;
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  while (std::getline(meminfo, line)) {
    std::istringstream fields(line);
    std::string key, unit;
    uint64_t value;
    if (fields >> key >> value >> unit && key == "MemAvailable:") {
      Require(unit == "kB", "fixed-cohort unexpected MemAvailable units");
      available = CheckedMultiply(value, 1024);
      break;
    }
  }
  Require(available > 0, "fixed-cohort RAM headroom query unavailable");
  std::ifstream membership("/proc/self/cgroup");
  std::filesystem::path group;
  while (std::getline(membership, line)) {
    if (line.rfind("0::/", 0) == 0) group = line.substr(3);
  }
  Require(!group.empty() && group.is_absolute(),
          "fixed-cohort cgroup-v2 membership unavailable");
  for (const auto& component : group)
    Require(component != "..",
            "fixed-cohort unsupported cgroup namespace path");
  std::filesystem::path root, mount;
  bool hierarchy_root = false;
  std::ifstream mounts("/proc/self/mountinfo");
  while (std::getline(mounts, line)) {
    if (line.find(" - cgroup2 ") == std::string::npos) continue;
    std::istringstream fields(line);
    std::string id, parent, dev, mount_root, mount_point;
    fields >> id >> parent >> dev >> mount_root >> mount_point;
    const auto relative = group.lexically_relative(mount_root);
    if (!relative.empty() && *relative.begin() != ".." &&
        mount_point.find('\\') == std::string::npos) {
      mount = mount_point;
      hierarchy_root = mount_root == "/";
      root = relative == "." ? mount : (mount / relative).lexically_normal();
      break;
    }
  }
  Require(!root.empty() && !mount.empty(),
          "fixed-cohort cgroup-v2 mount mapping unavailable");
  // Ancestor limits also constrain the leaf. Bound traversal inside this one
  // cgroup mount (never walk a filesystem root or inspect unrelated groups).
  for (size_t depth = 0;; ++depth) {
    Require(depth < 64, "fixed-cohort cgroup hierarchy too deep");
    std::string maximum, current;
    std::ifstream max_file(root / "memory.max");
    std::ifstream current_file(root / "memory.current");
    // Linux does not expose memory.max on the true hierarchy root: it has
    // no ancestor controller cap; MemAvailable still constrains this process.
    std::error_code error;
    const bool has_max = std::filesystem::exists(root / "memory.max", error);
    if (root == mount && hierarchy_root && !error && !has_max) break;
    Require(static_cast<bool>(max_file >> maximum) &&
                static_cast<bool>(current_file >> current),
            "fixed-cohort cgroup-v2 memory headroom query unavailable");
    const auto used = ParseBytes(current);
    if (maximum != "max") {
      const auto cap = ParseBytes(maximum);
      available = std::min(available, cap > used ? cap - used : 0);
    }
    if (root == mount) break;
    const auto parent = root.parent_path();
    Require(parent != root && parent != root.root_path(),
            "fixed-cohort cgroup mount traversal invalid");
    root = parent;
  }
  CUDA_SAFE_CALL(cudaSetDevice(device));
  size_t free = 0, total = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free, &total));
  return {available, free};
#else
  throw std::runtime_error("fixed-cohort RAM/cgroup headroom requires Linux");
#endif
}

int Device(const PatchMatchOptions& options) {
  Require(!options.gpu_index.empty(), "fixed-cohort device missing");
  const auto device = ParseBytes(options.gpu_index);
  Require(device <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
          "fixed-cohort device out of range");
  Require(options.gpu_index == std::to_string(device),
          "fixed-cohort requires one explicit canonical device");
  return static_cast<int>(device);
}

// Known lower-bound inventory only: reuse the current-owned solver envelope.
// Private source grayscale remains extra to resident-bank solver minima. GEO
// additionally creates float staging + private depth texture. Physical pitches,
// array allocation overhead and allocator/context storage require caller
// bounds.
void CheckPrivate(const std::vector<Image>& images,
                  int reference,
                  const std::vector<int>& sources,
                  bool geometry,
                  ByteCharge bound) {
  const auto& ref = images.at(reference);
  const auto minimum = ResidentPrivateLogicalMinimum(
      ref.GetWidth(), ref.GetHeight(), sources.size(), false);
  uint64_t width = 0, height = 0;
  for (int index : sources) {
    width = std::max(width, static_cast<uint64_t>(images.at(index).GetWidth()));
    height =
        std::max(height, static_cast<uint64_t>(images.at(index).GetHeight()));
  }
  const auto padded =
      CheckedMultiply(CheckedMultiply(width, height), sources.size());
  auto device = CheckedAdd(minimum.device, padded);
  if (geometry) device = CheckedAdd(device, CheckedMultiply(8, padded));
  Require(
      bound.host >= CheckedAdd(minimum.host, padded) && bound.device >= device,
      "fixed-cohort private bound below known logical inventory");
}

void Validate(const std::vector<Image>& images,
              const std::vector<FixedCohortReference>& refs,
              const FixedCohortBudget& budget) {
  Require(!images.empty(), "fixed-cohort images missing");
  ValidateFixedCohortJobShape(budget);
  Require(!refs.empty() && refs.size() == budget.photo_private.size(),
          "fixed-cohort reference/budget cardinality mismatch");
  const int device = Device(refs[0].photo_options);
  std::set<int> producers;
  for (const auto& ref : refs) {
    Require(ref.reference_image_idx >= 0 &&
                static_cast<size_t>(ref.reference_image_idx) < images.size() &&
                producers.insert(ref.reference_image_idx).second,
            "fixed-cohort invalid/duplicate reference");
  }
  uint64_t input_bytes = 0;
  for (const auto& image : images) {
    Require(image.GetWidth() && image.GetHeight() &&
                image.GetBitmap().IsGrey() &&
                static_cast<size_t>(image.GetBitmap().Width()) ==
                    image.GetWidth() &&
                static_cast<size_t>(image.GetBitmap().Height()) ==
                    image.GetHeight(),
            "fixed-cohort requires loaded grayscale pixels");
    for (size_t k = 0; k < 9; ++k)
      Require(std::isfinite(image.GetK()[k]) && std::isfinite(image.GetR()[k]),
              "fixed-cohort nonfinite calibration");
    for (size_t k = 0; k < 3; ++k)
      Require(std::isfinite(image.GetT()[k]), "fixed-cohort nonfinite pose");
    Require(std::abs(image.GetK()[1]) < 1e-6f &&
                std::abs(image.GetK()[3]) < 1e-6f &&
                std::abs(image.GetK()[6]) < 1e-6f &&
                std::abs(image.GetK()[7]) < 1e-6f &&
                std::abs(image.GetK()[8] - 1) < 1e-6f && image.GetK()[0] > 0 &&
                image.GetK()[4] > 0,
            "fixed-cohort unsupported intrinsics");
    input_bytes = CheckedAdd(input_bytes, image.GetBitmap().NumBytes());
  }
  Require(budget.caller_inputs.host >= input_bytes &&
              budget.frozen_inputs.host >= input_bytes,
          "fixed-cohort input copy bound insufficient");
  uint64_t maps = 0;
  for (size_t i = 0; i < refs.size(); ++i) {
    const auto& ref = refs[i];
    maps = CheckedAdd(
        maps,
        CheckedMultiply(
            CheckedMultiply(images[ref.reference_image_idx].GetWidth(),
                            images[ref.reference_image_idx].GetHeight()),
            16));
    for (const bool geo : {false, true}) {
      const auto& options = geo ? ref.geo_options : ref.photo_options;
      const auto& sources = geo ? ref.geo_sources : ref.photo_sources;
      Require(options.Check() && std::isfinite(options.sigma_spatial) &&
                  options.sigma_spatial > 0 && options.depth_min > 0 &&
                  options.depth_max > options.depth_min &&
                  std::isfinite(options.depth_max) &&
                  options.geom_consistency == geo &&
                  Device(options) == device && options.max_image_size == -1 &&
                  options.num_threads > 0 && !options.write_consistency_graph &&
                  !options.allow_missing_files && !sources.empty(),
              "fixed-cohort invalid phase options");
      ValidatePatchMatchExecution(PatchMatchExecution::kOwnedCuda,
                                  options.window_radius,
                                  options.window_step,
                                  geo,
                                  true,
                                  options.filter,
                                  options.write_consistency_graph,
                                  options.allow_missing_files);
      for (int source : sources) {
        Require(source >= 0 && static_cast<size_t>(source) < images.size() &&
                    source != ref.reference_image_idx &&
                    (!geo || producers.count(source)),
                "fixed-cohort missing source producer or invalid ordinal");
      }
      if (geo) {
        uint64_t width = 0, height = 0;
        for (int index : sources) {
          width =
              std::max(width, static_cast<uint64_t>(images[index].GetWidth()));
          height = std::max(height,
                            static_cast<uint64_t>(images[index].GetHeight()));
        }
        Require(
            budget.geo_staging_device[i] >=
                CheckedMultiply(CheckedMultiply(CheckedMultiply(width, height),
                                                sources.size()),
                                4),
            "fixed-cohort GEO staging bound below logical inventory");
      }
      CheckPrivate(images,
                   ref.reference_image_idx,
                   sources,
                   geo,
                   geo ? budget.geo_private[i] : budget.photo_private[i]);
    }
  }
  Require(budget.snapshots.device >= maps &&
              budget.host_outputs.host >= CheckedMultiply(2, maps),
          "fixed-cohort snapshot/output bound below logical maps");
}
}  // namespace

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
  Validate(images, references, budget);
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
  const int device = Device(references[0].photo_options);
  uint64_t caller_bytes = 0;
  for (const auto& image : images)
    caller_bytes = CheckedAdd(caller_bytes, image.GetBitmap().NumBytes());
  FixedCohortResult result;
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
        auto& maps = geo ? result.geo[owner] : result.photo[owner];
        maps.depth = solver.GetDepthMap();
        maps.normal = solver.GetNormalMap();
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
