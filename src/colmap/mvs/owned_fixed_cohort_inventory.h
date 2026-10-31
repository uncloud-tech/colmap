// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_fixed_cohort_budget.h"


namespace colmap::mvs::owned {

// Raw logical inventory only, not a calibrated physical-fit estimator. No
// margin here: native admission applies the single aggregate 3% margin.
struct FixedCohortImageShape {
  uint64_t width, height;
};
struct FixedCohortJobShape {
  int reference;
  std::vector<int> photo_sources, geo_sources;
};

inline ByteCharge FixedCohortPrivateLogicalMinimum(
    const std::vector<FixedCohortImageShape>& images,
    int reference,
    const std::vector<int>& sources,
    bool geometry,
    uint64_t* staging = nullptr,
    bool shared_photo = false) {
  const auto& ref = images.at(reference);
  const auto minimum = ResidentPrivateLogicalMinimum(
      ref.width, ref.height, sources.size(), false);
  uint64_t width = 0, height = 0;
  for (int index : sources) {
    width = std::max(width, images.at(index).width);
    height = std::max(height, images.at(index).height);
  }
  const auto padded =
      CheckedMultiply(CheckedMultiply(width, height), sources.size());
  if (staging) *staging = geometry ? CheckedMultiply(4, padded) : 0;
  auto device = CheckedAdd(minimum.device, shared_photo ? 0 : padded);
  if (geometry) device = CheckedAdd(device, CheckedMultiply(8, padded));
  // Retain per-owner host packing allowance in shared mode: binding validation
  // builds ordered Image copies (including duplicate ordinals). Its tight
  // pixels fit within this per-job padded transient. Not a host sharing credit.
  return {CheckedAdd(minimum.host, padded), device};
}

// Keeps caller-declared process/unknown/reserve/policy fields, replaces only
// logical payload inventory. Shape/options/closure validation remains native.
inline FixedCohortBudget BuildFixedCohortLogicalInventory(
    const std::vector<FixedCohortImageShape>& images,
    const std::vector<FixedCohortJobShape>& jobs,
    FixedCohortBudget budget) {
  uint64_t inputs = 0, maps = 0;
  for (const auto& image : images)
    inputs = CheckedAdd(inputs, CheckedMultiply(image.width, image.height));
  budget.photo_shared = {};
  const bool shared =
      budget.photo_source_storage == FixedCohortPhotoSourceStorage::kShared;
  if (shared) {
    std::vector<int> sources;
    uint64_t width = 0, height = 0;
    for (const auto& job : jobs) {
      for (int index : job.photo_sources) {
        const auto& image = images.at(index);
        if (std::find(sources.begin(), sources.end(), index) == sources.end())
          sources.push_back(index);
        width = std::max(width, image.width);
        height = std::max(height, image.height);
      }
    }
    const auto bytes =
        CheckedMultiply(CheckedMultiply(width, height), sources.size());
    budget.photo_shared = {bytes, bytes};
  }
  budget.photo_private.clear();
  budget.geo_private.clear();
  budget.geo_staging_device.clear();
  for (const auto& job : jobs) {
    const auto& ref = images.at(job.reference);
    maps = CheckedAdd(
        maps, CheckedMultiply(16, CheckedMultiply(ref.width, ref.height)));
    budget.photo_private.push_back(FixedCohortPrivateLogicalMinimum(
        images, job.reference, job.photo_sources, false, nullptr, shared));
    uint64_t staging = 0;
    budget.geo_private.push_back(FixedCohortPrivateLogicalMinimum(
        images, job.reference, job.geo_sources, true, &staging));
    budget.geo_staging_device.push_back(staging);
  }
  budget.caller_inputs = {inputs, 0};
  budget.frozen_inputs = {inputs, 0};
  budget.host_outputs = {
      CheckedMultiply(
          budget.output_policy == FixedCohortOutputPolicy::kPhotoAndGeo ? 2 : 1,
          maps),
      0};
  budget.snapshots = {0, maps};
  return budget;
}
// Componentwise policy, not a throughput or general maximality claim. Keep
// private for ties and incomparable admitted PHOTO/GEO counts.
inline bool PreferSharedFixedCohortCounts(std::array<size_t, 2> private_counts,
                                          std::array<size_t, 2> shared_counts) {
  return shared_counts[0] >= private_counts[0] &&
         shared_counts[1] >= private_counts[1] &&
         shared_counts != private_counts;
}

// Auto is a selection request, never a runtime storage enum. The returned
// canonical budget binds both actual allocation and reporting.
struct FixedCohortStorageSelection {
  FixedCohortBudget budget;
  std::string reason;
};

inline FixedCohortStorageSelection SelectFixedCohortPhotoSourceStorage(
    const std::vector<FixedCohortImageShape>& images,
    const std::vector<FixedCohortJobShape>& jobs,
    FixedCohortBudget declarations,
    ByteCharge capacity,
    size_t ceiling = 0,
    bool phase_counts = false) {
  for (const auto& image : images)
    if (!image.width || !image.height)
      throw std::invalid_argument("fixed-cohort empty image shape");
  std::array<FixedCohortBudget, 2> budgets;
  // Validate BOTH full effective ceilings before interpreting any capacity
  // refusal. No alternate plan may hide malformed input or arithmetic overflow.
  for (size_t i = 0; i < 2; ++i) {
    declarations.photo_source_storage =
        i ? FixedCohortPhotoSourceStorage::kShared
          : FixedCohortPhotoSourceStorage::kPrivate;
    budgets[i] = BuildFixedCohortLogicalInventory(images, jobs, declarations);
    budgets[i].phase_active_owners = {};
    budgets[i].max_active_owners =
        ceiling ? std::min(ceiling, jobs.size()) : jobs.size();
    AdmitFixedCohortBudget(budgets[i],
                           {std::numeric_limits<uint64_t>::max(),
                            std::numeric_limits<uint64_t>::max()});
  }
  std::array<std::string, 2> refusals;
  for (size_t i = 0; i < 2; ++i) {
    auto& budget = budgets[i];
    try {
      if (phase_counts) {
        const auto selected =
            SelectFixedCohortPhaseActiveOwners(budget, capacity, ceiling);
        budget.phase_active_owners = {selected[0].count, selected[1].count};
        budget.max_active_owners =
            std::max(selected[0].count, selected[1].count);
      } else {
        budget.max_active_owners =
            SelectFixedCohortActiveOwners(budget, capacity, ceiling).count;
      }
    } catch (const FixedCohortCapacityRefusal& error) {
      refusals[i] = error.what();
    }
  }
  if (!refusals[0].empty() && !refusals[1].empty())
    throw FixedCohortCapacityRefusal(
        "auto PHOTO source storage: private: " + refusals[0] +
        "; shared: " + refusals[1] +
        "; increase limiting capacity or reduce declared workload");
  if (!refusals[0].empty())
    return {budgets[1], "only shared fits complete pipeline"};
  if (!refusals[1].empty())
    return {budgets[0], "only private fits complete pipeline"};
  const auto p = std::array<size_t, 2>{
      FixedCohortPhaseActiveOwners(budgets[0], false),
      FixedCohortPhaseActiveOwners(budgets[0], true)};
  const auto s = std::array<size_t, 2>{
      FixedCohortPhaseActiveOwners(budgets[1], false),
      FixedCohortPhaseActiveOwners(budgets[1], true)};
  if (PreferSharedFixedCohortCounts(p, s))
    return {budgets[1], "shared dominates admitted PHOTO/GEO counts"};
  return {budgets[0], s == p ? "equal counts prefer private"
                           : "shared does not dominate; prefer private"};
}
}  // namespace colmap::mvs::owned
