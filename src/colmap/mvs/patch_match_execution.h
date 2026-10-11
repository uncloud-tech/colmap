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
    const bool owned_cuda_available,
    const bool filter = true,
    const bool write_consistency_graph = false,
    const bool allow_missing_files = false) {
  if (execution == PatchMatchExecution::kStock) {
    return execution;
  }
  if (execution != PatchMatchExecution::kOwnedCuda || !owned_cuda_available ||
      window_radius != 5 || window_step != 1) {
    throw std::invalid_argument(
        "Owned PatchMatch requires CUDA and window radius 5 / step 1");
  }
  if (geom_consistency &&
      (!filter || write_consistency_graph || allow_missing_files)) {
    throw std::invalid_argument(
        "Owned geometry requires filter=1 / graph=0 / allow_missing_files=0");
  }
  return execution;
}

}  // namespace mvs
}  // namespace colmap
