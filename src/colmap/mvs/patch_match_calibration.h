// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <type_traits>

namespace colmap {
namespace mvs {
namespace owned {

// A launch-owned snapshot, never a pointer to mutable module calibration.
struct ReferenceCalibration {
  float K[4];
  float inv_K[4];
};
static_assert(sizeof(ReferenceCalibration) == 32);
static_assert(std::is_trivially_copyable<ReferenceCalibration>::value);

inline ReferenceCalibration SnapshotReferenceCalibration(const float K[4],
                                                         const float inv_K[4]) {
  ReferenceCalibration result;
  for (int i = 0; i < 4; ++i) {
    result.K[i] = K[i];
    result.inv_K[i] = inv_K[i];
  }
  return result;
}

}  // namespace owned
}  // namespace mvs
}  // namespace colmap
