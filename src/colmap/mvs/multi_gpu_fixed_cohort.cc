// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/multi_gpu_fixed_cohort.h"

#include "colmap/mvs/owned_fixed_cohort_validation.h"
#include "colmap/mvs/owned_tile.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace colmap::mvs::owned {
namespace {
void Require(bool valid, const char* message) {
  if (!valid) throw std::invalid_argument(message);
}
bool Same(PhasePlanCharge a, PhasePlanCharge b) {
  return a.host == b.host && a.device == b.device;
}
void CheckSame(const PhasePlan& a, const PhasePlan& b) {
  Require(a.geometry == b.geometry && a.host_peak_raw == b.host_peak_raw &&
              a.groups.size() == b.groups.size(),
          "multi-GPU phase plan changed");
  for (size_t d = 0; d < a.groups.size(); ++d) {
    const auto& x = a.groups[d];
    const auto& y = b.groups[d];
    Require(x.device.ordinal == y.device.ordinal &&
                x.device.uuid == y.device.uuid &&
                x.device.capacity_bytes == y.device.capacity_bytes &&
                x.device.reserve_bytes == y.device.reserve_bytes &&
                x.device.host_share_raw == y.device.host_share_raw &&
                Same(x.device.fixed_raw, y.device.fixed_raw) &&
                Same(x.peak_raw, y.peak_raw) &&
                x.assigned_work == y.assigned_work &&
                x.cohorts.size() == y.cohorts.size(),
            "multi-GPU device plan changed");
    for (size_t c = 0; c < x.cohorts.size(); ++c) {
      const auto& p = x.cohorts[c];
      const auto& q = y.cohorts[c];
      Require(p.jobs == q.jobs && p.active_owners == q.active_owners &&
                  Same(p.bank_raw, q.bank_raw) &&
                  Same(p.private_peak_raw, q.private_peak_raw),
              "multi-GPU cohort plan changed");
    }
  }
}
}  // namespace

MultiGpuFixedCohortPlan PlanMultiGpuFixedCohort(
    const std::vector<FixedCohortImageShape>& images,
    const std::vector<FixedCohortReference>& refs,
    const FixedCohortBudget& declarations,
    std::vector<PhasePlanDevice> devices,
    uint64_t host_capacity,
    size_t ceiling,
    uint64_t loader_scratch_host_raw) {
  Require(devices.size() >= 2 && !refs.empty() && ceiling > 0,
          "multi-GPU requires at least two approved devices, work and positive "
          "owner ceiling");
  ValidateFixedCohortLiveProcess(declarations);
  Require(
      (declarations.unknown.host || declarations.unknown.device) &&
          !declarations.unknown_provenance.empty(),
      "multi-GPU requires the existing explicit unknown allowance/provenance");
  uint64_t pixels = 0, maps = 0;
  PhasePlanRequest request;
  for (size_t i = 0; i < images.size(); ++i) {
    const auto& image = images[i];
    Require(image.width && image.height, "multi-GPU empty image");
    pixels = CheckedAdd(pixels, CheckedMultiply(image.width, image.height));
    // Identity is the frozen invocation model ordinal/geometry, never filename
    // similarity. The native adapter owns one immutable pixel/exchange table.
    request.images.push_back(
        {"invocation-image-" + std::to_string(i), image.width, image.height});
  }
  for (const auto& ref : refs) {
    const auto& shape = images.at(ref.reference_image_idx);
    maps = CheckedAdd(
        maps, CheckedMultiply(16, CheckedMultiply(shape.width, shape.height)));
  }
  request.common_host_raw =
      CheckedAdd(CheckedMultiply(2, pixels), CheckedMultiply(2, maps));
  request.host_capacity_bytes = host_capacity;
  request.host_reserve_bytes = declarations.headroom_reserve.host;
  request.active_owners_ceiling = ceiling;
  Require(host_capacity > request.host_reserve_bytes,
          "multi-GPU host reserve unavailable");
  // floor(available/1.03), without overflow and without a second margin.
  const auto available = host_capacity - request.host_reserve_bytes;
  const auto raw_capacity = CheckedAdd(CheckedMultiply(available / 103, 100),
                                       (available % 103) * 100 / 103);
  Require(raw_capacity > request.common_host_raw,
          "multi-GPU retained PHOTO/GEO exchange exceeds host capacity");
  const auto share = (raw_capacity - request.common_host_raw) / devices.size();
  for (auto& device : devices) {
    device.host_share_raw = share;
    device.fixed_raw = {
        CheckedAdd(declarations.process.host, declarations.unknown.host),
        CheckedAdd(declarations.process.device, declarations.unknown.device)};
    Require(device.fixed_raw.host && device.fixed_raw.device,
            "multi-GPU rolling coordinator requires positive per-device "
            "process/unknown allowances");
    device.reserve_bytes = declarations.headroom_reserve.device;
  }
  request.devices = std::move(devices);
  MultiGpuFixedCohortPlan result;
  result.host_capacity = host_capacity;
  result.host_reserve = request.host_reserve_bytes;
  result.active_owners_ceiling = ceiling;
  result.loader_scratch_host_raw = loader_scratch_host_raw;
  result.loader_host_raw =
      CheckedAdd(CheckedMultiply(2, pixels), loader_scratch_host_raw);
  for (const auto& device : request.devices)
    result.loader_host_raw =
        CheckedAdd(result.loader_host_raw, device.fixed_raw.host);
  result.host_peak_raw = result.loader_host_raw;
  for (const bool geo : {false, true}) {
    request.geometry = geo;
    request.jobs.clear();
    for (size_t i = 0; i < refs.size(); ++i) {
      const auto& ref = refs[i];
      const auto& sources = geo ? ref.geo_sources : ref.photo_sources;
      uint64_t staging = 0;
      auto charge = FixedCohortPrivateLogicalMinimum(
          images, ref.reference_image_idx, sources, geo, &staging, !geo);
      // Host-map GEO creates float padded source-depth staging in addition to
      // the grayscale packing charge. Ordered duplicates each consume a layer.
      if (geo) charge.host = CheckedAdd(charge.host, staging);
      PhasePlanJob job;
      job.id = i;
      job.reference = ref.reference_image_idx;
      for (int source : sources) job.sources.push_back(source);
      job.private_raw = {charge.host, charge.device};
      const auto& shape = images.at(ref.reference_image_idx);
      job.work = CheckedMultiply(CheckedMultiply(shape.width, shape.height),
                                 CheckedAdd(1, sources.size()));
      request.jobs.push_back(std::move(job));
    }
    result.phases[geo] = BuildMultiGpuPhasePlan(request);
    result.host_peak_raw =
        std::max(result.host_peak_raw, result.phases[geo].host_peak_raw);
  }
  Require(CheckedAdd(MarginBytes(result.host_peak_raw), result.host_reserve) <=
              host_capacity,
          "multi-GPU aggregate host admission failed");
  return result;
}

MultiGpuFixedCohortPlan PlanSingleDeviceHostFixedCohort(
    const std::vector<FixedCohortImageShape>& images,
    const std::vector<FixedCohortReference>& refs,
    const FixedCohortBudget& declarations,
    PhasePlanDevice device,
    uint64_t host_capacity,
    uint64_t disk_capacity,
    const std::filesystem::path& scratch_directory,
    const std::string& requested_storage,
    uint64_t loader_scratch_host_raw,
    int tile_threads) {
  ValidateOwnedTileThreads(tile_threads);
  ValidateFixedCohortLiveProcess(declarations);
  Require(device.ordinal >= 0 && !device.uuid.empty() &&
              device.capacity_bytes && scratch_directory.is_absolute(),
          "single-device HOST requires approved device and absolute scratch "
          "directory");
  Require(requested_storage == "auto" || requested_storage == "private" ||
              requested_storage == "shared",
          "invalid requested PHOTO source storage");
  Require((declarations.unknown.host || declarations.unknown.device) &&
              !declarations.unknown_provenance.empty(),
          "single-device HOST requires explicit unknown allowance/provenance");
  StereoAdmissionRequest request;
  request.images = images;
  request.capacity = {host_capacity, device.capacity_bytes};
  request.disk_capacity = disk_capacity;
  request.process = declarations.process;
  request.unknown = declarations.unknown;
  request.reserve = declarations.headroom_reserve;
  request.loader_scratch_host = loader_scratch_host_raw;
  request.block_width = tile_threads;
  Require(CheckedAdd(request.process.host, request.unknown.host) &&
              CheckedAdd(request.process.device, request.unknown.device),
          "single-device HOST requires positive process/unknown envelopes");
  for (const auto& ref : refs) {
    StereoAdmissionReference r;
    r.reference = ref.reference_image_idx;
    r.sources = {ref.photo_sources, ref.geo_sources};
    for (size_t phase = 0; phase < 2; ++phase) {
      const auto& options = phase ? ref.geo_options : ref.photo_options;
      r.policy[phase] = {options.filter && options.num_iterations > 0,
                         options.owned_cuda_graphs,
                         options.owned_async_pool};
    }
    request.references.push_back(std::move(r));
  }
  MultiGpuFixedCohortPlan result;
  result.host_capacity = host_capacity;
  result.host_reserve = declarations.headroom_reserve.host;
  result.active_owners_ceiling = 1;
  result.loader_scratch_host_raw = loader_scratch_host_raw;
  SingleDeviceHostExecution execution;
  execution.admission = PlanSingleDeviceStereoAdmission(request);
  execution.shapes = images;
  execution.scratch_directory = scratch_directory;
  execution.requested_photo_source_storage = requested_storage;
  execution.disk_capacity = disk_capacity;
  execution.process = declarations.process;
  execution.unknown = declarations.unknown;
  execution.unknown_provenance = declarations.unknown_provenance;
  execution.tile_threads = tile_threads;
  result.loader_host_raw = execution.admission.loader_host_raw;
  result.host_peak_raw = execution.admission.peak_raw.host;
  device.reserve_bytes = declarations.headroom_reserve.device;
  device.fixed_raw = {
      CheckedAdd(declarations.process.host, declarations.unknown.host),
      CheckedAdd(declarations.process.device, declarations.unknown.device)};
  const auto available = host_capacity - result.host_reserve;
  const auto raw = CheckedAdd(CheckedMultiply(available / 103, 100),
                              (available % 103) * 100 / 103);
  Require(raw >= execution.admission.common_host_raw,
          "single-device HOST common host exceeds cap");
  device.host_share_raw = raw - execution.admission.common_host_raw;
  for (size_t phase = 0; phase < 2; ++phase) {
    auto& output = result.phases[phase];
    output.geometry = phase;
    PhasePlanGroup group;
    group.device = device;
    for (size_t job = 0; job < refs.size(); ++job) {
      const auto& record = execution.admission.records[phase][job];
      group.cohorts.push_back(
          {{job}, 1, {}, {record.peak.host, record.peak.device}});
      group.peak_raw.host =
          std::max(group.peak_raw.host,
                   CheckedAdd(device.fixed_raw.host, record.peak.host));
      group.peak_raw.device =
          std::max(group.peak_raw.device,
                   CheckedAdd(device.fixed_raw.device, record.peak.device));
      group.assigned_work = CheckedAdd(
          group.assigned_work,
          CheckedMultiply(CheckedMultiply(record.width, record.height),
                          CheckedAdd(1, record.sources.size())));
    }
    output.host_peak_raw =
        CheckedAdd(execution.admission.common_host_raw, group.peak_raw.host);
    output.groups.push_back(std::move(group));
  }
  result.single_device_host = std::move(execution);
  return result;
}

#if !defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
void CheckMultiGpuFixedCohortAvailable(const MultiGpuFixedCohortPlan&,
                                       uint64_t) {
  throw std::runtime_error(
      "multi-GPU execution requires native CUDA; no fallback");
}
#endif

void ValidateMultiGpuFixedCohortPlan(
    const std::vector<Image>& images,
    const std::vector<FixedCohortReference>& refs,
    const FixedCohortBudget& budget,
    const MultiGpuFixedCohortPlan& plan) {
  // Existing explicit-one-device scientific validation remains authoritative.
  // Its inventory is a metadata floor only here; snapshot storage is never
  // allocated by this host-map executor.
  ValidateFixedCohortShape(images, refs, budget);
  std::vector<FixedCohortImageShape> shapes;
  for (const auto& image : images)
    shapes.push_back({image.GetWidth(), image.GetHeight()});
  std::vector<PhasePlanDevice> devices;
  for (const auto& group : plan.phases[0].groups)
    devices.push_back(group.device);
  MultiGpuFixedCohortPlan regenerated;
  if (plan.single_device_host) {
    const auto& single = *plan.single_device_host;
    Require(devices.size() == 1 && plan.active_owners_ceiling == 1 &&
                single.columns == 32 && single.shapes.size() == shapes.size(),
            "single-device HOST plan identity changed");
    for (size_t i = 0; i < shapes.size(); ++i)
      Require(shapes[i].width == single.shapes[i].width &&
                  shapes[i].height == single.shapes[i].height,
              "single-device HOST input dimensions changed");
    regenerated =
        PlanSingleDeviceHostFixedCohort(shapes,
                                        refs,
                                        budget,
                                        devices[0],
                                        plan.host_capacity,
                                        single.disk_capacity,
                                        single.scratch_directory,
                                        single.requested_photo_source_storage,
                                        plan.loader_scratch_host_raw,
                                        single.tile_threads);
    Require(single.process.host == budget.process.host &&
                single.process.device == budget.process.device &&
                single.unknown.host == budget.unknown.host &&
                single.unknown.device == budget.unknown.device &&
                single.unknown_provenance == budget.unknown_provenance,
            "single-device HOST allowance policy changed");
    Require(SameStereoAdmissionPlan(single.admission,
                                    regenerated.single_device_host->admission),
            "single-device HOST execution record changed");
  } else {
    regenerated = PlanMultiGpuFixedCohort(shapes,
                                          refs,
                                          budget,
                                          devices,
                                          plan.host_capacity,
                                          plan.active_owners_ceiling,
                                          plan.loader_scratch_host_raw);
  }
  Require(plan.host_peak_raw == regenerated.host_peak_raw &&
              plan.loader_host_raw == regenerated.loader_host_raw &&
              plan.host_reserve == regenerated.host_reserve,
          "multi-GPU aggregate plan changed");
  for (size_t phase = 0; phase < 2; ++phase)
    CheckSame(plan.phases[phase], regenerated.phases[phase]);
}
}  // namespace colmap::mvs::owned
