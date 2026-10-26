// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/optimized_photo_forecast.h"
#include "colmap/mvs/owned_rotation_storage.h"

namespace colmap::mvs {
// Effective input dimensions observed by CheckOptimizedPhotoInputs.
struct OptimizedPhotoWorkload {
  uint64_t cache;
  uint64_t retained;
  uint64_t staging;
  uint64_t width;
  uint64_t height;
  uint64_t sources;
};
inline owned::ByteCharge LegacyAbsentBindingPhotoFloor(
    const OptimizedPhotoWorkload& workload) {
  const auto& w = workload;
  const auto a = owned::CheckedMultiply(w.width, w.height);
  // Historical private cache + two decoded banks + staging/result scratch.
  const auto host = owned::CheckedAdd(
      w.cache,
      owned::CheckedAdd(owned::CheckedMultiply(2, w.retained),
                        std::max(w.staging, owned::CheckedMultiply(24, a))));
  // Preserve the legacy mask/full-PRNG rotation layout only at this seam.
  // This logical minimum is not a physical allocation upper bound.
  auto device = owned::CheckedAdd(
      owned::BuildOwnedRotationLayout(w.width, w.height, w.sources)
          .logical_bytes,
      w.staging);
  device = owned::CheckedAdd(device, owned::CheckedMultiply(2, a));
  device = owned::CheckedAdd(device, owned::CheckedMultiply(688, w.sources));
  device = owned::CheckedAdd(device, 32);
  return {host, device};
}
inline void CheckOptimizedPhotoReservation(
    const OptimizedPhotoForecast& forecast,
    const OptimizedPhotoWorkload& workload,
    const PhotoPhaseV1Binding* binding = nullptr) {
  if (workload.cache < workload.retained)
    throw std::invalid_argument(
        "cache must retain complete private PHOTO input set");
  if (binding) {
    ValidatePhotoPhaseV1Binding(*binding, forecast);
    // The trusted phase inventory owns the physical bound. Its process charge
    // includes the single controller cache; private_charge excludes that cache.
    if (workload.cache > binding->process.host)
      throw std::invalid_argument("PHOTO cache exceeds process reservation");
    return;
  }
  const auto floor = LegacyAbsentBindingPhotoFloor(workload);
  if (forecast.private_charge.host < floor.host ||
      forecast.private_charge.device < floor.device)
    throw std::invalid_argument(
        "job reservation below logical workload envelope");
}
}  // namespace colmap::mvs
