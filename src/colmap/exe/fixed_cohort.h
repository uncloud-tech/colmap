// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_fixed_cohort_workspace.h"
#include "colmap/mvs/owned_input_bank.h"

namespace colmap {
// Typed adapter only: no sorting, deduplication, global depth envelope, or
// reapplication of native controller source transformations.
std::vector<mvs::owned::FixedCohortWorkspaceReference>
ResolveFixedCohortWorkspacePlan(
    const std::vector<std::string>& names,
    const std::vector<mvs::owned::BankJob>& photo,
    const std::vector<mvs::owned::BankJob>& geo);
int RunExperimentalFixedCohort(int argc, char** argv);
}  // namespace colmap
