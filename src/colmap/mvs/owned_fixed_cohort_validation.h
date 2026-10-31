// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_fixed_cohort_pipeline.h"

#include <filesystem>

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
// context creation or physical-fit certification. proc_root permits isolated
// Linux proc/cgroup fixtures; production callers use the default, never a CLI
// override or caller-supplied available-byte value.
uint64_t ObserveFixedCohortHostAvailable(
    const std::filesystem::path& proc_root = "/proc");

struct FixedCohortCapacityObservation {
  ByteCharge available;
  uint64_t device_total = 0;  // Evidence only; NEVER used as available capacity.
  int device = -1;           // Process-visible CUDA ordinal.
  std::string device_name;
  std::string device_uuid;
};
// Selects/initializes the requested CUDA device and queries its free memory.
// No pixels, banks or solvers are allocated. Throws on query failure or a
// non-native-CUDA build; no fallback. Snapshot only, not a reservation.
FixedCohortCapacityObservation ObserveFixedCohortCapacities(int device);
}  // namespace colmap::mvs::owned
