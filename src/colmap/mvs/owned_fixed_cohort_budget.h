// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/owned_resident_minima.h"

#include <array>

namespace colmap::mvs::owned {

// Experimental native caller contract, NOT a forecast producer or a physical
// allocation estimator. All fields are RAW (unmargined) bounds supplied by the
// caller. Keep original caller inputs AND frozen copies charged. Reserve both
// phases' returned host maps and all PHOTO snapshot copies for the full run.
// Per-owner bounds include setup/transients/private solver storage; the GEO
// bounds additionally include adapter staging + its private depth texture.
// Process covers classified process storage; unknown covers separately
// justified unclassified overhead. Neither is a sharing credit.
// Production dispatch requires a separately qualified canonical producer.
struct FixedCohortBudget {
  ByteCharge caller_inputs;
  ByteCharge frozen_inputs;
  ByteCharge host_outputs;
  ByteCharge snapshots;
  ByteCharge process;
  // Explicit unclassified physical overhead (pitch/array/context/allocator and
  // metadata). Must be justified independently, never free-minus-known filler.
  ByteCharge unknown;
  std::string unknown_provenance;
  // Sub-bounds ALREADY INSIDE geo_private, not an additional ledger charge.
  std::array<uint64_t, 2> geo_staging_device{};
  // Unallocated headroom in addition to the once-margined peak.
  ByteCharge headroom_reserve;
  std::array<ByteCharge, 2> photo_private;
  std::array<ByteCharge, 2> geo_private;
};

struct FixedCohortBudgetPeaks {
  ByteCharge photo;
  ByteCharge geo;
  ByteCharge peak;
};

inline FixedCohortBudgetPeaks FixedCohortPeaks(
    const FixedCohortBudget& budget) {
  const auto add = [](ByteCharge a, ByteCharge b) {
    return ByteCharge{CheckedAdd(a.host, b.host),
                      CheckedAdd(a.device, b.device)};
  };
  auto retained = add(budget.caller_inputs, budget.frozen_inputs);
  retained = add(retained, budget.host_outputs);
  retained = add(retained, budget.snapshots);
  retained = add(retained, budget.process);
  retained = add(retained, budget.unknown);
  auto photo = retained;
  auto geo = retained;
  for (size_t i = 0; i < 2; ++i) {
    photo = add(photo, budget.photo_private[i]);
    geo = add(geo, budget.geo_private[i]);
  }
  return {photo,
          geo,
          {std::max(photo.host, geo.host), std::max(photo.device, geo.device)}};
}

// Shape/cap gate only. The native entry additionally validates invocation
// lower bounds and actual snapshot pitches. No physical-fit certification.
inline ByteCharge AdmitFixedCohortBudget(const FixedCohortBudget& budget,
                                         ByteCharge capacity) {
  const auto peak = FixedCohortPeaks(budget).peak;
  const ByteCharge admitted{MarginBytes(peak.host), MarginBytes(peak.device)};
  if ((!budget.unknown.host && !budget.unknown.device) ||
      budget.unknown_provenance.empty() || !budget.caller_inputs.host ||
      !budget.frozen_inputs.host || !budget.host_outputs.host ||
      !budget.snapshots.device || budget.caller_inputs.device ||
      budget.frozen_inputs.device || budget.host_outputs.device ||
      budget.snapshots.host)
    throw std::invalid_argument("invalid fixed-cohort budget shape");
  for (size_t i = 0; i < 2; ++i) {
    if (!budget.photo_private[i].host || !budget.photo_private[i].device ||
        !budget.geo_private[i].host || !budget.geo_private[i].device ||
        !budget.geo_staging_device[i] ||
        budget.geo_staging_device[i] > budget.geo_private[i].device)
      throw std::invalid_argument("missing fixed-cohort owner bound");
  }
  const ByteCharge with_reserve{
      CheckedAdd(admitted.host, budget.headroom_reserve.host),
      CheckedAdd(admitted.device, budget.headroom_reserve.device)};
  if (!PairCoordinator::Fits(with_reserve, capacity))
    throw std::invalid_argument("fixed-cohort retained peak exceeds capacity");
  return admitted;
}

// Observed free is observational admission evidence, NOT a runtime hard cap.
// Subtract actual known live payload, not its forecast/reservation.
// Unclassified overhead is conservatively left in remaining demand when not
// measurable.
inline void CheckFixedCohortHeadroom(ByteCharge demand,
                                     ByteCharge observed_free,
                                     ByteCharge already_live,
                                     ByteCharge reserve) {
  const auto fits =
      [](uint64_t need, uint64_t free, uint64_t live, uint64_t keep) {
        const auto remaining = need > live ? need - live : 0;
        return free >= keep && remaining <= free - keep;
      };
  if (!fits(demand.host, observed_free.host, already_live.host, reserve.host) ||
      !fits(demand.device,
            observed_free.device,
            already_live.device,
            reserve.device))
    throw std::invalid_argument(
        "fixed-cohort observed remaining headroom insufficient");
}

inline void CheckFixedCohortPhysicalCopies(
    const FixedCohortBudget& budget,
    uint64_t snapshots,
    const std::array<uint64_t, 2>& staging) {
  const auto excess = [](uint64_t actual, uint64_t estimate) {
    return actual > estimate ? actual - estimate : 0;
  };
  const auto snapshot_extra = excess(snapshots, budget.snapshots.device);
  // Setup is serialized by PairCoordinator; each adapter's temporary pitched
  // staging is freed before the next preparation lease. Only the larger extra
  // staging peak can overlap retained snapshots/other live private owners.
  const auto staging_extra =
      std::max(excess(staging[0], budget.geo_staging_device[0]),
               excess(staging[1], budget.geo_staging_device[1]));
  const auto phases = FixedCohortPeaks(budget);
  if (CheckedAdd(phases.photo.device, snapshot_extra) >
          MarginBytes(phases.photo.device) ||
      CheckedAdd(CheckedAdd(phases.geo.device, snapshot_extra), staging_extra) >
          MarginBytes(phases.geo.device))
    throw std::invalid_argument(
        "fixed-cohort actual pitched discrepancies exceed original aggregate "
        "margin");
}

}  // namespace colmap::mvs::owned
