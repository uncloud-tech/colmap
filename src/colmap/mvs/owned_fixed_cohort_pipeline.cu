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

#include "colmap/mvs/owned_inventory_cuda_driver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
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
  return ObserveFixedCohortCapacities(device).available;
}

}  // namespace

FixedCohortCapacityObservation ObserveFixedCohortCapacities(int device) {
  FixedCohortCapacityObservation result;
  const auto query = [](cudaError_t status) {
    if (status != cudaSuccess)
      throw std::runtime_error(
          std::string("fixed-cohort CUDA capacity query failed: ") +
          cudaGetErrorString(status));
  };
  query(cudaSetDevice(device));
  cudaDeviceProp properties{};
  query(cudaGetDeviceProperties(&properties, device));
  size_t free = 0, total = 0;
  query(cudaMemGetInfo(&free, &total));
  Require(free <= total && total > 0,
          "fixed-cohort CUDA capacity query returned invalid memory sizes");
  // Observe host after any CUDA context initialization, so newly-live context
  // storage is not mistaken for still-free host memory either.
  result.available.host = ObserveFixedCohortHostAvailable();
  result.available.device = free;
  result.device_total = total;
  result.device = device;
  result.device_name = properties.name;
  result.compute_capability_major = properties.major;
  result.compute_capability_minor = properties.minor;
  std::ostringstream uuid;
  uuid << std::hex << std::setfill('0');
  for (const auto byte : properties.uuid.bytes)
    uuid << std::setw(2)
         << static_cast<unsigned int>(static_cast<unsigned char>(byte));
  result.device_uuid = uuid.str();
  return result;
}

std::vector<FixedCohortDeviceObservation> ObserveFixedCohortDevices(
    const NativeRunDeadlineGuard& guard) {
  guard.Check();
  InventoryCudaDriver driver;
  const auto init = driver.Lookup<decltype(&cuInit)>("cuInit");
  const auto get_current = driver.Lookup<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent");
  const auto set_current = driver.Lookup<decltype(&cuCtxSetCurrent)>("cuCtxSetCurrent");
  const auto error_string = driver.Lookup<decltype(&cuGetErrorString)>("cuGetErrorString");
  const auto driver_query = [error_string](CUresult status) {
    if (status == CUDA_SUCCESS) return;
    const char* reason = nullptr;
    (void)error_string(status, &reason);
    throw std::runtime_error(std::string("CUDA inventory context query/restore failed: ") +
                             (reason ? reason : "unknown driver error"));
  };
  const auto initialized = init(0);
  if (initialized == CUDA_ERROR_NO_DEVICE) return {};
  driver_query(initialized);
  // Capture before any Runtime API call can select/create a primary context.
  // The current context may be non-primary or null; an ordinal is insufficient.
  CUcontext previous = nullptr;
  driver_query(get_current(&previous));
  struct RestoreContext {
    CUcontext context;
    decltype(set_current) set;
    bool restored = false;
    CUresult Restore() {
      const auto status = set(context);
      restored = status == CUDA_SUCCESS;
      return status;
    }
    ~RestoreContext() {
      if (!restored) {
        const auto status = Restore();
        if (status != CUDA_SUCCESS)
          std::fprintf(stderr,
                       "CUDA inventory: context restoration failed during cleanup: %d\n",
                       static_cast<int>(status));
      }
    }
  } restore{previous, set_current};
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice) {
    driver_query(restore.Restore());
    return {};
  }
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA visible inventory failed: ") +
                             cudaGetErrorString(status));
  if (count == 0) {
    driver_query(restore.Restore());
    return {};
  }
  std::vector<FixedCohortDeviceObservation> result;
  for (int device = 0; device < count; ++device) {
    guard.Check();
    FixedCohortDeviceObservation observed;
    observed.capacity.device = device;
    // Retain metadata even when context/free-memory observation fails.
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
      observed.capacity.device_name = properties.name;
      observed.capacity.device_total = properties.totalGlobalMem;
      observed.capacity.compute_capability_major = properties.major;
      observed.capacity.compute_capability_minor = properties.minor;
      std::ostringstream uuid;
      uuid << std::hex << std::setfill('0');
      for (const auto byte : properties.uuid.bytes)
        uuid << std::setw(2)
             << static_cast<unsigned int>(static_cast<unsigned char>(byte));
      observed.capacity.device_uuid = uuid.str();
    }
    try {
      observed.capacity = ObserveFixedCohortCapacities(device);
      observed.usable = true;
    } catch (const std::exception& error) {
      observed.reason = error.what();
      (void)cudaGetLastError();
    }
    guard.Check();
    result.push_back(std::move(observed));
  }
  driver_query(restore.Restore());
  return result;
}

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
  Require((dispatch == PairDispatch::kSerial ? budget.max_active_owners == 1
                                             : budget.max_active_owners >= 2),
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
    // This phase-local strong owner outlives every borrower, including graph
    // launches and failure drain/join. It dies before the GEO free observation.
    std::shared_ptr<const ResidentGrayscaleBank> photo_bank;
    if (!geo &&
        budget.photo_source_storage == FixedCohortPhotoSourceStorage::kShared) {
      std::vector<std::vector<int>> sources;
      for (const auto& ref : refs) sources.push_back(ref.photo_sources);
      HostGrayscaleBank host(frozen, sources, cancel);
      check();
      const auto bytes = CheckedMultiply(
          CheckedMultiply(host.Width(), host.Height()), host.Layers().size());
      Require(bytes <= budget.photo_shared.host &&
                  bytes <= budget.photo_shared.device,
              "fixed-cohort actual PHOTO union exceeds bound");
      photo_bank = std::make_shared<ResidentGrayscaleBank>(host, device);
      result.actual_photo_shared = {bytes, bytes};
      check();
    }
    const auto& charges = geo ? budget.geo_private : budget.photo_private;
    // Retain the entire once-margined peak, even in the smaller phase. No
    // per-owner re-margin or snapshot sharing credit is issued.
    const auto phase_count = FixedCohortPhaseActiveOwners(budget, geo);
    result.phase_owner_limits[geo] = phase_count;
    const auto active = FixedCohortActivePeak(charges, phase_count);
    const ByteCharge process{reserved.host - active.host,
                             reserved.device - active.device};
    // Use the once-margined reservation itself as the ledger capacity: spare
    // external cap space cannot admit more than the declared active envelope.
    PairCoordinator coordinator(reserved,
                                process,
                                phase_count,
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
        if (photo_bank) problem.photo_grayscale = photo_bank->BindJob(owner);
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
        if (geo ||
            budget.output_policy == FixedCohortOutputPolicy::kPhotoAndGeo) {
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
