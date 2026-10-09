// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stdexcept>

namespace colmap {
namespace mvs {

// Per-instance experiment selection. Not a CLI, global, or scheduler policy.
enum class PatchMatchExecution { kStock, kOwnedCuda };

inline PatchMatchExecution ValidatePatchMatchExecution(
    const PatchMatchExecution execution,
    const int window_radius,
    const int window_step,
    const bool geom_consistency,
    const bool owned_cuda_available) {
  if (execution == PatchMatchExecution::kStock) {
    return execution;
  }
  if (execution != PatchMatchExecution::kOwnedCuda || !owned_cuda_available ||
      window_radius != 5 || window_step != 1 || geom_consistency) {
    throw std::invalid_argument(
        "Owned PatchMatch requires CUDA and photo window radius 5 / step 1");
  }
  return execution;
}

}  // namespace mvs
}  // namespace colmap
