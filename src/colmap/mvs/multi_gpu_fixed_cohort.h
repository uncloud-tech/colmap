// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/multi_gpu_phase_plan.h"
#include "colmap/mvs/owned_fixed_cohort_pipeline.h"
#include "colmap/mvs/owned_fixed_cohort_inventory.h"

namespace colmap::mvs::owned {

struct MultiGpuFixedCohortPlan {
  std::array<PhasePlan, 2> phases;
  uint64_t host_capacity = 0;
  uint64_t host_reserve = 0;
  uint64_t host_peak_raw = 0;
  uint64_t loader_scratch_host_raw = 0;
  uint64_t loader_host_raw = 0;
  size_t active_owners_ceiling = 1;
};

// Pure metadata adapter into the CPU assignment seam. Host exchange costs are
// charged under BOTH output policies. GEO packing includes its actual 4-byte
// padded host source-depth transient, not the snapshot adapter's device-only
// staging accounting. The original single-device inventory stays unchanged.
MultiGpuFixedCohortPlan PlanMultiGpuFixedCohort(
    const std::vector<FixedCohortImageShape>& images,
    const std::vector<FixedCohortReference>& references,
    const FixedCohortBudget& declarations,
    std::vector<PhasePlanDevice> devices,
    uint64_t host_capacity,
    size_t active_owners_ceiling,
    uint64_t loader_scratch_host_raw = 0);

// Revalidate a plan against regenerated complete metadata before pixel copies.
// Native entry additionally checks approved UUIDs and observed FREE.
void ValidateMultiGpuFixedCohortPlan(
    const std::vector<Image>& images,
    const std::vector<FixedCohortReference>& references,
    const FixedCohortBudget& validation_budget,
    const MultiGpuFixedCohortPlan& plan);

// Authoritative FREE/UUID checks before workspace pixels, and again before
// execution's frozen copy/exchange. Caps never raise observed availability.
// Nonzero live_caller_pixels is used only after preparation's cache/scratch
// has actually been destroyed; it credits exactly those already-live pixels
// against FREE, never against caps or process/unknown declarations.
void CheckMultiGpuFixedCohortAvailable(const MultiGpuFixedCohortPlan& plan,
                                     uint64_t live_caller_pixels = 0);

FixedCohortResult RunMultiGpuFixedCohortPhotoGeo(
    const std::vector<Image>& images,
    const std::vector<FixedCohortReference>& references,
    const FixedCohortBudget& validation_budget,
    const MultiGpuFixedCohortPlan& plan,
    NativeRunDeadline deadline,
    int tile_threads,
    const std::atomic<bool>* cancel = nullptr,
    // Diagnostic event seam; throwing cancels/drains the whole device set.
    // begin/output identify global jobs; joined uses device=-1/job=max(size_t).
    // bank_ready reports one completed PHOTO bank construction/upload (payload
    // bytes are that cohort's bank_raw.device, not predicted cross-cohort reuse).
    // cohort_drained follows owner join AND bank destruction. Both cohort
    // events identify the cohort by its first global job; no timing claim.
    std::function<void(bool, int, size_t, const char*)> observer = {});

}  // namespace colmap::mvs::owned
