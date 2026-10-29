// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_fixed_cohort_budget.h"

#include <cassert>
#include <limits>
#include <stdexcept>

using namespace colmap::mvs::owned;

int main() {
  FixedCohortBudget budget;
  budget.caller_inputs = {100, 0};
  budget.frozen_inputs = {100, 0};
  budget.host_outputs = {200, 0};
  budget.snapshots = {0, 300};
  budget.process = {10, 20};
  budget.unknown = {1, 2};
  budget.unknown_provenance =
      "fixture: metadata and driver allocation uncertainty";
  budget.geo_staging_device = {{1, 1}};
  budget.photo_private = {{{30, 40}, {50, 60}}};
  budget.geo_private = {{{70, 80}, {90, 100}}};
  // All snapshots and outputs are reserved even before publication; private
  // PHOTO and GEO owners are mutually exclusive, never summed together.
  const auto peaks = FixedCohortPeaks(budget);
  assert(peaks.photo.host == 491 && peaks.photo.device == 422);
  assert(peaks.geo.host == 571 && peaks.geo.device == 502);
  assert(peaks.peak.host == 571 && peaks.peak.device == 502);
  // One canonical margin on the aggregate phase peak, never per item/phase.
  const auto admitted = AdmitFixedCohortBudget(budget, {589, 518});
  assert(admitted.host == 589 && admitted.device == 518);
  bool cap_refused = false;
  try {
    (void)AdmitFixedCohortBudget(budget, {588, 518});
  } catch (const std::invalid_argument&) {
    cap_refused = true;
  }
  assert(cap_refused);
  // Remaining-free checks subtract only actual already-live payload, not
  // reserved bounds; a full declared snapshot charge is NOT deducted twice.
  CheckFixedCohortHeadroom({500, 600}, {350, 450}, {200, 200}, {50, 50});
  bool headroom_refused = false;
  try {
    CheckFixedCohortHeadroom({500, 600}, {349, 450}, {200, 200}, {50, 50});
  } catch (const std::invalid_argument&) {
    headroom_refused = true;
  }
  assert(headroom_refused);
  // PHOTO raw device peak 422 -> original once-margined 435; GEO 502 ->
  // 518. A 13-byte snapshot discrepancy fits BOTH phases, without remargin.
  CheckFixedCohortPhysicalCopies(budget, 313, {{1, 1}});
  bool pitch_refused = false;
  try {
    CheckFixedCohortPhysicalCopies(budget, 314, {{1, 1}});
  } catch (const std::invalid_argument&) {
    pitch_refused = true;
  }
  assert(pitch_refused);
  bool combined_refused = false;
  try {
    // Snapshot +13 and staging +4 exceed GEO's existing 16-byte margin.
    CheckFixedCohortPhysicalCopies(budget, 313, {{5, 1}});
  } catch (const std::invalid_argument&) {
    combined_refused = true;
  }
  assert(combined_refused);
  bool live_refused = false;
  try {
    // Pitched +13 fits margin, but remaining actual free still refuses.
    CheckFixedCohortHeadroom(admitted, {389, 204}, {200, 313}, {0, 0});
  } catch (const std::invalid_argument&) {
    live_refused = true;
  }
  assert(live_refused);
  auto device_unknown_only = budget;
  device_unknown_only.unknown.host = 0;
  (void)AdmitFixedCohortBudget(device_unknown_only, {1000, 1000});
  auto reserve = budget;
  reserve.headroom_reserve = {1, 1};
  bool reserve_refused = false;
  try {
    (void)AdmitFixedCohortBudget(reserve, admitted);
  } catch (const std::invalid_argument&) {
    reserve_refused = true;
  }
  assert(reserve_refused);
  auto missing = budget;
  missing.unknown_provenance.clear();
  bool missing_refused = false;
  try {
    (void)AdmitFixedCohortBudget(missing, {1000, 1000});
  } catch (const std::invalid_argument&) {
    missing_refused = true;
  }
  assert(missing_refused);
  auto bad = budget;
  bad.process.host = std::numeric_limits<uint64_t>::max();
  bool refused = false;
  try {
    (void)FixedCohortPeaks(bad);
  } catch (const std::overflow_error&) {
    refused = true;
  }
  assert(refused);
}
