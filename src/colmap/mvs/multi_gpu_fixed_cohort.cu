// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/multi_gpu_fixed_cohort.h"
#include "colmap/mvs/multi_gpu_workers.h"
#include "colmap/mvs/owned_fixed_cohort_validation.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_tile.h"
#include "colmap/mvs/patch_match_cuda_owned.h"

#if !defined(COLMAP_HIP_ENABLED)
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

namespace colmap::mvs::owned {
namespace {
constexpr size_t kNoJob = std::numeric_limits<size_t>::max();
struct CohortControls {
  std::unique_ptr<PairCoordinator> coordinator;
  std::shared_ptr<GraphLaunchControl> graphs;
};
}  // namespace

void CheckMultiGpuFixedCohortAvailable(const MultiGpuFixedCohortPlan& plan,
                                     uint64_t live_caller_pixels) {
  if (plan.phases[0].groups.size() != plan.phases[1].groups.size())
    throw std::invalid_argument("multi-GPU phase device sets differ");
  for (size_t d = 0; d < plan.phases[0].groups.size(); ++d) {
    const auto& group = plan.phases[0].groups[d];
    const auto observed = ObserveFixedCohortCapacities(group.device.ordinal);
    if (observed.device_uuid != group.device.uuid)
      throw std::runtime_error("multi-GPU approved UUID changed");
    const auto peak = std::max(group.peak_raw.device,
                               plan.phases[1].groups[d].peak_raw.device);
    if (CheckedAdd(MarginBytes(peak), group.device.reserve_bytes) >
        std::min(observed.available.device, group.device.capacity_bytes))
      throw std::runtime_error("multi-GPU device FREE no longer admits plan");
  }
  const auto host_raw = live_caller_pixels
      ? std::max(plan.phases[0].host_peak_raw, plan.phases[1].host_peak_raw)
      : plan.host_peak_raw;
  if (CheckedAdd(MarginBytes(host_raw), plan.host_reserve) > plan.host_capacity)
    throw std::runtime_error("multi-GPU host cap no longer admits exchange");
  CheckFixedCohortHeadroom({MarginBytes(host_raw), 0},
      {ObserveFixedCohortHostAvailable(), 0}, {live_caller_pixels, 0},
      {plan.host_reserve, 0});
}

FixedCohortResult RunMultiGpuFixedCohortPhotoGeo(
    const std::vector<Image>& images,
    const std::vector<FixedCohortReference>& refs,
    const FixedCohortBudget& budget,
    const MultiGpuFixedCohortPlan& plan,
    NativeRunDeadline deadline,
    int tile_threads,
    const std::atomic<bool>* cancel,
    std::function<void(bool, int, size_t, const char*)> observer) {
  ValidateOwnedTileThreads(tile_threads);
  ValidateMultiGpuFixedCohortPlan(images, refs, budget, plan);
  ValidateFixedCohortPixels(images, refs, budget);
  NativeRunDeadlineGuard guard(deadline);
  const auto check = [&] {
    guard.Check();
    if (cancel && cancel->load())
      throw std::runtime_error("multi-GPU invocation cancelled");
  };
  check();
  // Reobserve the whole approved device set before any frozen/exchange/bank
  // payload allocation. UUIDs are binding, not an auto-reselection suggestion.
  uint64_t caller_pixels = 0;
  for (const auto& image : images)
    caller_pixels = CheckedAdd(caller_pixels, image.GetBitmap().NumBytes());
  CheckMultiGpuFixedCohortAvailable(plan, caller_pixels);
  check();

  // The legacy Problem API takes a non-const vector pointer; owned solvers
  // borrow these pixels/calibration read-only. No code mutates this table.
  auto frozen = images;  // Caller AND the frozen copy stay charged.
  // Image-indexed exchange: source order is untouched and no per-GEO deep copy
  // of the complete exchange is made. One producer writes each reference slot.
  std::vector<DepthMap> photo_depth(images.size());
  std::vector<NormalMap> photo_normal(images.size());
  FixedCohortResult result;
  result.output_policy = budget.output_policy;
  result.geo.resize(refs.size());
  for (const bool geo : {false, true}) {
    check();
    const auto& phase = plan.phases[geo];
    uint64_t live_host = CheckedMultiply(2, caller_pixels);
    if (geo) {
      for (const auto& ref : refs) {
        const auto idx = ref.reference_image_idx;
        live_host = CheckedAdd(live_host,
            CheckedAdd(photo_depth[idx].GetNumBytes(), photo_normal[idx].GetNumBytes()));
      }
    }
    CheckFixedCohortHeadroom({MarginBytes(phase.host_peak_raw), 0},
        {ObserveFixedCohortHostAvailable(), 0}, {live_host, 0}, {plan.host_reserve, 0});
    MultiGpuWorkers workers;
    std::vector<std::vector<CohortControls>> controls(phase.groups.size());
    // Register ALL controls before the first device thread exists. Objects stay
    // alive through Run's full/partial launch drain, including unstarted GPUs.
    for (size_t d = 0; d < phase.groups.size(); ++d) {
      const auto& group = phase.groups[d];
      for (const auto& cohort : group.cohorts) {
        std::vector<ByteCharge> charges;
        for (size_t job : cohort.jobs) {
          std::vector<FixedCohortImageShape> shapes;
          for (const auto& image : frozen) shapes.push_back({image.GetWidth(), image.GetHeight()});
          uint64_t staging = 0;
          const auto& ref = refs.at(job);
          auto charge = FixedCohortPrivateLogicalMinimum(shapes, ref.reference_image_idx,
              geo ? ref.geo_sources : ref.photo_sources, geo, &staging, !geo);
          if (geo) charge.host = CheckedAdd(charge.host, staging);
          charges.push_back(charge);
        }
        ByteCharge fixed{CheckedAdd(group.device.fixed_raw.host, cohort.bank_raw.host),
                         CheckedAdd(group.device.fixed_raw.device, cohort.bank_raw.device)};
        ByteCharge capacity{CheckedAdd(fixed.host, cohort.private_peak_raw.host),
                            CheckedAdd(fixed.device, cohort.private_peak_raw.device)};
        CohortControls control;
        control.coordinator = std::make_unique<PairCoordinator>(capacity, fixed,
            cohort.active_owners, charges, deadline.Remaining(),
            std::function<void(const PairEvent&)>{}, deadline);
        control.graphs = std::make_shared<GraphLaunchControl>(cohort.jobs.size());
        auto* coordinator = control.coordinator.get();
        const auto graphs = control.graphs;
        workers.Register([coordinator, graphs](std::exception_ptr error) {
          coordinator->Cancel(kNoJob, error);
          graphs->Cancel(error);
        });
        controls[d].push_back(std::move(control));
      }
    }
    workers.Run(phase.groups.size(), [&](size_t d) {
      const auto& group = phase.groups[d];
      const int device = group.device.ordinal;
      for (size_t c = 0; c < group.cohorts.size(); ++c) {
        workers.Check();
        check();
        CUDA_SAFE_CALL(cudaSetDevice(device));
        const auto& cohort = group.cohorts[c];
        const auto observed = ObserveFixedCohortCapacities(device);
        if (observed.device_uuid != group.device.uuid ||
            CheckedAdd(MarginBytes(group.peak_raw.device), group.device.reserve_bytes) >
                std::min(observed.available.device, group.device.capacity_bytes))
          throw std::runtime_error("multi-GPU cohort device admission changed");
        auto& control = controls[d][c];
        auto& coordinator = *control.coordinator;
        std::shared_ptr<const ResidentGrayscaleBank> bank;
        if (!geo) {
          std::vector<std::vector<int>> sources;
          for (size_t job : cohort.jobs) sources.push_back(refs.at(job).photo_sources);
          HostGrayscaleBank host(frozen, sources, workers.CancellationToken());
          const auto bytes = CheckedMultiply(CheckedMultiply(host.Width(), host.Height()), host.Layers().size());
          if (bytes > cohort.bank_raw.host || bytes > cohort.bank_raw.device)
            throw std::runtime_error("multi-GPU PHOTO bank exceeds admitted union");
          workers.Check();
          check();
          bank = std::make_shared<ResidentGrayscaleBank>(host, device);
        }
        const auto run = [&](size_t owner) {
          try {
            workers.Check();
            check();
            CUDA_SAFE_CALL(cudaSetDevice(device));
            coordinator.BeginPreparation(owner);
            const size_t job = cohort.jobs.at(owner);
            const auto& ref = refs.at(job);
            if (observer) observer(geo, device, job, "begin");
            auto options = geo ? ref.geo_options : ref.photo_options;
            // Placement is the only changed option. The report binds this
            // exact job->device separately; numerical options remain original.
            options.gpu_index = std::to_string(device);
            PatchMatch::Problem problem;
            problem.images = &frozen;
            problem.ref_image_idx = ref.reference_image_idx;
            problem.src_image_idxs = geo ? ref.geo_sources : ref.photo_sources;
            if (geo) {
              problem.depth_maps = &photo_depth;
              problem.normal_maps = &photo_normal;
            } else {
              problem.photo_grayscale = bank->BindJob(owner);
            }
            workers.Check();
            check();
            PatchMatchCuda solver(options, problem, tile_threads);
            try {
              if (options.owned_cuda_graphs)
                solver.SetGraphCoordinator(control.graphs, owner);
              solver.SetSweepObserver([&](int, int) {
                workers.Check();
                check();
              });
              coordinator.ReadyAndWait(owner);
              workers.Check();
              check();
              solver.Run();
              workers.Check();
              check();
              coordinator.SolverDone(owner);
              coordinator.BeginOutput(owner);
              if (geo) {
                result.geo[job].depth = solver.GetDepthMap();
                result.geo[job].normal = solver.GetNormalMap();
              } else {
                photo_depth[ref.reference_image_idx] = solver.GetDepthMap();
                photo_normal[ref.reference_image_idx] = solver.GetNormalMap();
              }
              check();
              coordinator.EndOutput(owner);
              if (observer) observer(geo, device, job, "output");
            } catch (...) {
              // Publish before solver destruction/drain when an error crosses
              // the solver API or observer boundary.
              workers.Cancel(std::current_exception());
              throw;
            }
          } catch (...) {
            // Includes failures in setup or solver construction.
            workers.Cancel(std::current_exception());
            throw;
          }
        };
        RunRollingOwners(coordinator, run, [](size_t, const char*) {},
                         workers.OwnerThreadFactory());
        workers.Check();
        // Solver owners joined; bank destroyed before next cohort allocation.
      }
    }, MultiGpuWorkers::ThreadFactory{}, check);
    check();
    if (observer) observer(geo, -1, kNoJob, "joined");
    if (!geo) {
      for (const auto& ref : refs) {
        const auto idx = ref.reference_image_idx;
        if (photo_depth[idx].GetWidth() != frozen[idx].GetWidth() ||
            photo_depth[idx].GetHeight() != frozen[idx].GetHeight() ||
            photo_normal[idx].GetWidth() != frozen[idx].GetWidth() ||
            photo_normal[idx].GetHeight() != frozen[idx].GetHeight())
          throw std::runtime_error("multi-GPU PHOTO exchange incomplete");
      }
      // Complete successful PHOTO device set drained/joined; only now may
      // GEO read the immutable image-indexed exchange on any approved device.
    }
  }
  check();
  if (budget.output_policy == FixedCohortOutputPolicy::kPhotoAndGeo) {
    result.photo.resize(refs.size());
    for (size_t i = 0; i < refs.size(); ++i) {
      result.photo[i].depth = std::move(photo_depth[refs[i].reference_image_idx]);
      result.photo[i].normal = std::move(photo_normal[refs[i].reference_image_idx]);
    }
  }
  return result;  // All device owners drained; no snapshot or partial success.
}
}  // namespace colmap::mvs::owned
#endif
