// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/exe/optimized_dense_plan_options.h"
#include "colmap/mvs/multi_gpu_fixed_cohort.h"
#include "colmap/mvs/owned_input_bank.h"

namespace colmap::dense_plan {

inline Json SingleDeviceHostJson(
    const mvs::owned::SingleDeviceHostExecution& execution,
    const std::string& source_storage = "private",
    bool snapshot = false) {
  std::vector<Json> phases;
  for (size_t phase = 0; phase < 2; ++phase) {
    std::vector<Json> records;
    for (const auto& record : execution.admission.records[phase]) {
      std::vector<Json> sources, planes;
      for (int source : record.sources) sources.emplace_back(source);
      for (const auto& plane : record.rotation_planes) {
        planes.push_back(Json::Object(
            {{"role", Json(mvs::owned::RotationRoleName(plane.role))},
             {"parity", Json(static_cast<int>(plane.parity))},
             {"row_bytes", Json(std::to_string(plane.row_bytes))},
             {"rows", Json(std::to_string(plane.rows))},
             {"depth", Json(std::to_string(plane.depth))},
             {"pitch_ceiling", Json(std::to_string(plane.pitch_ceiling))}}));
      }
      records.push_back(Json::Object(
          {{"reference", Json(record.reference)},
           {"width", Json(std::to_string(record.width))},
           {"height", Json(std::to_string(record.height))},
           {"sources", Json::Array(sources)},
           {"volume_mode",
            Json(record.mode == mvs::owned::StereoVolumeMode::kStripFile
                     ? "strip_file"
                     : "resident_private")},
           {"columns", Json(std::to_string(record.columns))},
           {"retain_mask", Json(record.policy.retain_mask)},
           {"owned_cuda_graphs", Json(record.policy.cuda_graphs)},
           {"owned_async_pool", Json(record.policy.async_pool)},
           {"initialization_host_raw",
            Json(std::to_string(record.initialization.host))},
           {"initialization_device_raw",
            Json(std::to_string(record.initialization.device))},
           {"steady_host_raw", Json(std::to_string(record.steady.host))},
           {"steady_device_raw", Json(std::to_string(record.steady.device))},
           {"peak_host_raw", Json(std::to_string(record.peak.host))},
           {"peak_device_raw", Json(std::to_string(record.peak.device))},
           {"initialization_prng_pitch",
            Json(std::to_string(record.initialization_prng_pitch))},
           {"staging_float_pitch",
            Json(std::to_string(record.staging_float_pitch))},
           {"staging_mask_pitch",
            Json(std::to_string(record.staging_mask_pitch))},
           {"staging_host_bytes",
            Json(std::to_string(record.staging_host_bytes))},
           {"staging_device_bytes",
            Json(std::to_string(record.staging_device_bytes))},
           {"scratch_disk_bytes",
            Json(std::to_string(record.scratch_disk_bytes))},
           {"cache_host_bytes", Json(std::to_string(record.cache_host_bytes))},
           {"padded_source_pixels",
            Json(std::to_string(record.padded_source_pixels))},
           {"rotation_planes", Json::Array(planes)},
           {"reason", Json(record.reason)}}));
    }
    phases.push_back(Json::Object({{"phase", Json(phase ? "geo" : "photo")},
                                   {"records", Json::Array(records)}}));
  }
  const auto& plan = execution.admission;
  return Json::Object(
      {{"status", Json("estimate_runtime_checked_not_physical_certification")},
       {"output_policy",
        Json(mvs::owned::FixedCohortOutputPolicyName(plan.output_policy))},
       {"peak_host_raw", Json(std::to_string(plan.peak_raw.host))},
       {"peak_device_raw", Json(std::to_string(plan.peak_raw.device))},
       {"requested_photo_source_storage",
        Json(execution.requested_photo_source_storage)},
       {"effective_photo_source_storage", Json(source_storage)},
       {"exchange",
        Json(snapshot ? "device_snapshots" : "immutable_host_photo")},
       {"scratch_directory", Json(execution.scratch_directory.string())},
       {"disk_capacity_bytes", Json(std::to_string(execution.disk_capacity))},
       {"process_host_raw", Json(std::to_string(execution.process.host))},
       {"process_device_raw", Json(std::to_string(execution.process.device))},
       {"unknown_host_raw", Json(std::to_string(execution.unknown.host))},
       {"unknown_device_raw", Json(std::to_string(execution.unknown.device))},
       {"unknown_provenance", Json(execution.unknown_provenance)},
       {"active_owners", Json(std::to_string(plan.active_owners))},
       {"output_disk_bytes", Json(std::to_string(plan.output_disk_bytes))},
       {"scratch_disk_bytes", Json(std::to_string(plan.scratch_disk_bytes))},
       {"required_disk_bytes", Json(std::to_string(plan.required_disk_bytes))},
       {"common_host_raw", Json(std::to_string(plan.common_host_raw))},
       {"admitted_host_with_reserve", Json(std::to_string(plan.admitted.host))},
       {"admitted_device_with_reserve",
        Json(std::to_string(plan.admitted.device))},
       {"tile_threads", Json(execution.tile_threads)},
       {"requested_active_owners_ceiling",
        Json(std::to_string(execution.requested_active_owners_ceiling))},
       {"pitch_policy",
        Json("per-plane estimated 512-byte rounding; pool workspace 128; "
             "packed selection exact; reject larger returned pitch")},
       {"cache_policy",
        Json(snapshot ? "no strip scratch or page-cache charge"
                      : "full scratch file charged; eviction advice is not a "
                        "guarantee")},
       {"phases", Json::Array(phases)}});
}

// Placement is an explicit binding, not a broad exception to scientific option
// equality. Numerical per-reference options remain in the existing phase
// report.
inline Json MultiGpuPlanJson(
    const mvs::owned::MultiGpuFixedCohortPlan& plan,
    const std::vector<mvs::owned::FixedCohortReference>& references,
    const std::vector<mvs::Image>& images,
    const std::string& allocation_fingerprint = "") {
  std::vector<Json> bindings;
  for (const auto& image : images) {
    const auto identity = mvs::owned::CaptureFileIdentity(image.GetPath());
    std::vector<Json> K, R, T;
    for (size_t k = 0; k < 9; ++k) {
      K.emplace_back(static_cast<double>(image.GetK()[k]));
      R.emplace_back(static_cast<double>(image.GetR()[k]));
    }
    for (size_t k = 0; k < 3; ++k)
      T.emplace_back(static_cast<double>(image.GetT()[k]));
    bindings.push_back(Json::Object(
        {{"path", Json(identity.path.string())},
         {"width", Json(std::to_string(image.GetWidth()))},
         {"height", Json(std::to_string(image.GetHeight()))},
         {"K", Json::Array(K)},
         {"R", Json::Array(R)},
         {"T", Json::Array(T)},
         {"size", Json(std::to_string(identity.size))},
         {"device", Json(std::to_string(identity.device))},
         {"inode", Json(std::to_string(identity.inode))},
         {"mtime_seconds", Json(std::to_string(identity.modified_seconds))},
         {"mtime_nanoseconds",
          Json(std::to_string(identity.modified_nanoseconds))},
         {"ctime_seconds", Json(std::to_string(identity.changed_seconds))},
         {"ctime_nanoseconds",
          Json(std::to_string(identity.changed_nanoseconds))}}));
  }
  std::vector<Json> scientific;
  for (const auto& ref : references) {
    std::vector<Json> photo, geo;
    for (int idx : ref.photo_sources) photo.emplace_back(idx);
    for (int idx : ref.geo_sources) geo.emplace_back(idx);
    scientific.push_back(
        Json::Object({{"reference", Json(ref.reference_image_idx)},
                      {"photo_sources", Json::Array(photo)},
                      {"geo_sources", Json::Array(geo)},
                      {"photo_options", OptionsJson(ref.photo_options)},
                      {"geo_options", OptionsJson(ref.geo_options)}}));
  }
  std::vector<Json> phases;
  for (const auto& phase : plan.phases) {
    std::vector<Json> groups;
    for (const auto& group : phase.groups) {
      std::vector<Json> cohorts;
      for (const auto& cohort : group.cohorts) {
        std::vector<Json> jobs;
        for (size_t job : cohort.jobs) jobs.emplace_back(static_cast<int>(job));
        cohorts.push_back(Json::Object(
            {{"jobs", Json::Array(jobs)},
             {"active_owners", Json(static_cast<int>(cohort.active_owners))},
             {"bank_host_raw", Json(std::to_string(cohort.bank_raw.host))},
             {"bank_device_raw", Json(std::to_string(cohort.bank_raw.device))},
             {"private_host_raw",
              Json(std::to_string(cohort.private_peak_raw.host))},
             {"private_device_raw",
              Json(std::to_string(cohort.private_peak_raw.device))}}));
      }
      groups.push_back(Json::Object(
          {{"ordinal", Json(group.device.ordinal)},
           {"uuid", Json(group.device.uuid)},
           {"capacity_bytes",
            Json(std::to_string(group.device.capacity_bytes))},
           {"reserve_bytes", Json(std::to_string(group.device.reserve_bytes))},
           {"host_share_raw",
            Json(std::to_string(group.device.host_share_raw))},
           {"fixed_host_raw",
            Json(std::to_string(group.device.fixed_raw.host))},
           {"fixed_device_raw",
            Json(std::to_string(group.device.fixed_raw.device))},
           {"host_peak_raw", Json(std::to_string(group.peak_raw.host))},
           {"device_peak_raw", Json(std::to_string(group.peak_raw.device))},
           {"cohorts", Json::Array(cohorts)}}));
    }
    phases.push_back(
        Json::Object({{"phase", Json(phase.geometry ? "geo" : "photo")},
                      {"groups", Json::Array(groups)}}));
  }
  return Json::Object(
      {{"schema_version", Json(1)},
       {"output_policy",
        Json(mvs::owned::FixedCohortOutputPolicyName(plan.output_policy))},
       {"allocation_fingerprint", Json(allocation_fingerprint)},
       {"execution_kind",
        Json(plan.snapshot_budget      ? "resident_snapshot"
             : plan.single_device_host ? "single_device_host_exchange"
                                       : "multi_gpu_host_exchange")},
       {"snapshot",
        plan.snapshot_inventory
            ? SingleDeviceHostJson(
                  *plan.snapshot_inventory,
                  plan.snapshot_budget->photo_source_storage ==
                          mvs::owned::FixedCohortPhotoSourceStorage::kShared
                      ? "shared"
                      : "private",
                  true)
            : Json()},
       {"snapshot_device_bytes",
        plan.snapshot_budget
            ? Json(std::to_string(plan.snapshot_budget->snapshots.device))
            : Json()},
       {"single_device_host",
        plan.single_device_host ? SingleDeviceHostJson(*plan.single_device_host)
                                : Json()},
       {"host_capacity_bytes", Json(std::to_string(plan.host_capacity))},
       {"host_reserve_bytes", Json(std::to_string(plan.host_reserve))},
       {"host_peak_raw", Json(std::to_string(plan.host_peak_raw))},
       {"loader_host_raw", Json(std::to_string(plan.loader_host_raw))},
       {"loader_scratch_host_raw",
        Json(std::to_string(plan.loader_scratch_host_raw))},
       {"active_owners_ceiling",
        Json(std::to_string(plan.active_owners_ceiling))},
       {"image_bindings", Json::Array(bindings)},
       {"reuse",
        Json(plan.snapshot_budget
                 ? "resident PHOTO device snapshots through final GEO drain"
             : plan.single_device_host
                 ? "private sources only; immutable HOST PHOTO exchange; no "
                   "bank or snapshot"
                 : "PHOTO padded source union within cohort only; GEO private "
                   "host-map inputs; no cross-cohort cache")},
       {"scientific_inputs", Json::Array(scientific)},
       {"phases", Json::Array(phases)}});
}
}  // namespace colmap::dense_plan
