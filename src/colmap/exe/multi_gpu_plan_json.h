// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/exe/optimized_dense_plan_options.h"
#include "colmap/mvs/multi_gpu_fixed_cohort.h"
#include "colmap/mvs/owned_input_bank.h"

namespace colmap::dense_plan {

// Placement is an explicit binding, not a broad exception to scientific option
// equality. Numerical per-reference options remain in the existing phase report.
inline Json MultiGpuPlanJson(const mvs::owned::MultiGpuFixedCohortPlan& plan,
                            const std::vector<mvs::owned::FixedCohortReference>& references,
                            const std::vector<mvs::Image>& images) {
  std::vector<Json> bindings;
  for (const auto& image : images) {
    const auto identity = mvs::owned::CaptureFileIdentity(image.GetPath());
    std::vector<Json> K, R, T;
    for (size_t k = 0; k < 9; ++k) {
      K.emplace_back(static_cast<double>(image.GetK()[k]));
      R.emplace_back(static_cast<double>(image.GetR()[k]));
    }
    for (size_t k = 0; k < 3; ++k) T.emplace_back(static_cast<double>(image.GetT()[k]));
    bindings.push_back(Json::Object({
        {"path", Json(identity.path.string())},
        {"width", Json(std::to_string(image.GetWidth()))},
        {"height", Json(std::to_string(image.GetHeight()))},
        {"K", Json::Array(K)}, {"R", Json::Array(R)}, {"T", Json::Array(T)},
        {"size", Json(std::to_string(identity.size))},
        {"device", Json(std::to_string(identity.device))},
        {"inode", Json(std::to_string(identity.inode))},
        {"mtime_seconds", Json(std::to_string(identity.modified_seconds))},
        {"mtime_nanoseconds", Json(std::to_string(identity.modified_nanoseconds))},
        {"ctime_seconds", Json(std::to_string(identity.changed_seconds))},
        {"ctime_nanoseconds", Json(std::to_string(identity.changed_nanoseconds))}}));
  }
  std::vector<Json> scientific;
  for (const auto& ref : references) {
    std::vector<Json> photo, geo;
    for (int idx : ref.photo_sources) photo.emplace_back(idx);
    for (int idx : ref.geo_sources) geo.emplace_back(idx);
    scientific.push_back(Json::Object({{"reference", Json(ref.reference_image_idx)},
        {"photo_sources", Json::Array(photo)}, {"geo_sources", Json::Array(geo)},
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
        cohorts.push_back(Json::Object({
            {"jobs", Json::Array(jobs)},
            {"active_owners", Json(static_cast<int>(cohort.active_owners))},
            {"bank_host_raw", Json(std::to_string(cohort.bank_raw.host))},
            {"bank_device_raw", Json(std::to_string(cohort.bank_raw.device))},
            {"private_host_raw", Json(std::to_string(cohort.private_peak_raw.host))},
            {"private_device_raw", Json(std::to_string(cohort.private_peak_raw.device))}}));
      }
      groups.push_back(Json::Object({
          {"ordinal", Json(group.device.ordinal)}, {"uuid", Json(group.device.uuid)},
          {"capacity_bytes", Json(std::to_string(group.device.capacity_bytes))},
          {"reserve_bytes", Json(std::to_string(group.device.reserve_bytes))},
          {"host_share_raw", Json(std::to_string(group.device.host_share_raw))},
          {"fixed_host_raw", Json(std::to_string(group.device.fixed_raw.host))},
          {"fixed_device_raw", Json(std::to_string(group.device.fixed_raw.device))},
          {"host_peak_raw", Json(std::to_string(group.peak_raw.host))},
          {"device_peak_raw", Json(std::to_string(group.peak_raw.device))},
          {"cohorts", Json::Array(cohorts)}}));
    }
    phases.push_back(Json::Object({{"phase", Json(phase.geometry ? "geo" : "photo")},
                                   {"groups", Json::Array(groups)}}));
  }
  return Json::Object({{"schema_version", Json(1)},
      {"host_capacity_bytes", Json(std::to_string(plan.host_capacity))},
      {"host_reserve_bytes", Json(std::to_string(plan.host_reserve))},
      {"host_peak_raw", Json(std::to_string(plan.host_peak_raw))},
      {"loader_host_raw", Json(std::to_string(plan.loader_host_raw))},
      {"loader_scratch_host_raw", Json(std::to_string(plan.loader_scratch_host_raw))},
      {"active_owners_ceiling", Json(std::to_string(plan.active_owners_ceiling))},
      {"image_bindings", Json::Array(bindings)},
      {"reuse", Json("PHOTO padded source union within cohort only; GEO private host-map inputs; no cross-cohort cache")},
      {"scientific_inputs", Json::Array(scientific)},
      {"phases", Json::Array(phases)}});
}
}  // namespace colmap::dense_plan
