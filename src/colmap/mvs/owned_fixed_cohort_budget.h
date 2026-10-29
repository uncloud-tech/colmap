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
  // Actual classified process bytes already resident at each observation
  // [before copy, before GEO], INCLUDED in process, not another charge.
  // Each declaration needs independent provenance and must exclude caller
  // pixels/maps/snapshots and unknown overhead. Do not carry a PHOTO-only
  // allocation into GEO after destruction. Zero means no measured credit.
  std::array<ByteCharge, 2> already_live_process{};
  std::array<std::string, 2> already_live_process_provenance{};
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

inline void ValidateFixedCohortLiveProcess(const FixedCohortBudget& budget) {
  for (size_t i = 0; i < 2; ++i) {
    const auto live = budget.already_live_process[i];
    if (live.host > budget.process.host ||
        live.device > budget.process.device ||
        ((live.host || live.device) &&
         budget.already_live_process_provenance[i].empty()))
      throw std::invalid_argument(
          "fixed-cohort live process exceeds classified bound or lacks "
          "provenance");
  }
}

// Shape/cap gate only. The native entry additionally validates invocation
// lower bounds and actual snapshot pitches. No physical-fit certification.
inline ByteCharge AdmitFixedCohortBudget(const FixedCohortBudget& budget,
                                         ByteCharge capacity) {
  ValidateFixedCohortLiveProcess(budget);
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

// Shared public admission seam for the real runner's two observations.
// known_live_payload contains only actual pixels/returned maps/snapshots, never
// process allocations. Reservation and cap admission remain independent.
inline void CheckFixedCohortPhaseHeadroom(const FixedCohortBudget& budget,
                                          bool geo,
                                          ByteCharge observed_free,
                                          ByteCharge known_live_payload) {
  ValidateFixedCohortLiveProcess(budget);
  const auto peaks = FixedCohortPeaks(budget);
  // Snapshot pitch deltas may consume the original aggregate GEO margin;
  // CheckFixedCohortPhysicalCopies independently checks simultaneous deltas.
  const auto snapshot_bound =
      CheckedAdd(budget.snapshots.device,
                 MarginBytes(peaks.geo.device) - peaks.geo.device);
  const ByteCharge payload_bound{
      geo ? CheckedAdd(CheckedAdd(budget.caller_inputs.host,
                                  budget.frozen_inputs.host),
                       budget.host_outputs.host)
          : budget.caller_inputs.host,
      geo ? snapshot_bound : 0};
  if (!PairCoordinator::Fits(known_live_payload, payload_bound))
    throw std::invalid_argument(
        "fixed-cohort live payload exceeds classified bound");
  const auto process = budget.already_live_process[geo];
  const ByteCharge already_live{
      CheckedAdd(known_live_payload.host, process.host),
      CheckedAdd(known_live_payload.device, process.device)};
  const auto demand = geo ? peaks.geo : peaks.peak;
  CheckFixedCohortHeadroom(
      {MarginBytes(demand.host), MarginBytes(demand.device)},
      observed_free,
      already_live,
      budget.headroom_reserve);
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
