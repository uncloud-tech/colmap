// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/optimized_photo_reservation.h"

#include <functional>

namespace colmap::mvs {
// TRUSTED source interface, not a parser or an authorization token. The adapter
// must attest the profile/source closure and exact launch bytes before
// returning per-job bindings. The CLI wire adapter consumes producer-supplied
// typed binding sections; never synthesize bindings from forecast charges.
// An absent adapter returns no evidence.
using TrustedPhotoPhaseV1Delivery =
    std::function<std::vector<PhotoPhaseV1Binding>(
        const std::string& exact_launch_bytes)>;

struct OptimizedPhotoDispatch {
  OptimizedPhotoLaunch launch;
  std::vector<PhotoPhaseV1Binding> bindings;

  // Production input boundary: effective order and binding-driven reservation
  // must pass before the caller can continue into solver preparation.
  template <typename Execute>
  void WithCheckedInputs(size_t generation,
                         int reference_image_idx,
                         const std::vector<int>& source_image_idxs,
                         const OptimizedPhotoWorkload& workload,
                         Execute&& execute) const {
    const auto& forecast = launch.jobs.at(generation).forecast;
    ValidateOptimizedPhotoOrder(
        forecast, reference_image_idx, source_image_idxs);
    CheckOptimizedPhotoReservation(
        forecast,
        workload,
        bindings.empty() ? nullptr : &bindings.at(generation));
    execute();
  }
};

// Adapter-aware production parse route. The four-argument overload remains the
// synthetic wire seam; production always goes through bound admission here.
inline OptimizedPhotoDispatch ParseOptimizedPhotoLaunch(
    const std::string& bytes,
    const std::filesystem::path& config,
    const std::string& requested,
    owned::ByteCharge capacity,
    const TrustedPhotoPhaseV1Delivery& delivery) {
  auto bindings =
      delivery ? delivery(bytes) : std::vector<PhotoPhaseV1Binding>{};
  auto launch = AdmitBoundOptimizedPhotoLaunch(
      bytes, config, requested, capacity, bindings);
  return {std::move(launch), std::move(bindings)};
}
}  // namespace colmap::mvs
