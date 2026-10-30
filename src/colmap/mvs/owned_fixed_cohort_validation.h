// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_fixed_cohort_pipeline.h"

namespace colmap::mvs::owned {
// Pure validation against the existing native logical inventory. Shape-only
// validation never fabricates a Bitmap; final execution always checks pixels.
void ValidateFixedCohortShape(
    const std::vector<Image>& images,
    const std::vector<FixedCohortReference>& references,
    const FixedCohortBudget& budget);
void ValidateFixedCohortPixels(
    const std::vector<Image>& images,
    const std::vector<FixedCohortReference>& references,
    const FixedCohortBudget& budget);
int FixedCohortDevice(const PatchMatchOptions& options);
// Same Linux/cgroup-v2 host observation used by the native runner. No CUDA
// context creation, caller-supplied observations, or physical-fit
// certification.
uint64_t ObserveFixedCohortHostAvailable();
}  // namespace colmap::mvs::owned
