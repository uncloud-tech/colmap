// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_fixed_cohort_inventory.h"

#include <cassert>
#include <limits>
#include <stdexcept>

using namespace colmap::mvs::owned;

int main() {
  // Loaded union uses invocation image ordinals, preserving multiplicity in
  // each owner's private state. Global padding is NOT always a saving.
  const std::vector<FixedCohortImageShape> shapes = {{5, 4}, {3, 3}, {2, 7}};
  const std::vector<FixedCohortJobShape> jobs = {{1, {0, 2, 0}, {0, 0}},
                                                 {0, {2, 1}, {1}}};
  FixedCohortBudget shared_config;
  shared_config.photo_source_storage = FixedCohortPhotoSourceStorage::kShared;
  shared_config.unknown = {50, 60};
  shared_config.unknown_provenance = "explicit fixture allowance";
  const auto shared =
      BuildFixedCohortLogicalInventory(shapes, jobs, shared_config);
  auto private_config = shared_config;
  private_config.photo_source_storage = FixedCohortPhotoSourceStorage::kPrivate;
  const auto private_inventory =
      BuildFixedCohortLogicalInventory(shapes, jobs, private_config);
  // Three distinct layers, global extent 5x7, not five source ordinals.
  assert(shared.photo_shared.host == 105 && shared.photo_shared.device == 105);
  assert(shared.photo_private[0].host ==
         private_inventory.photo_private[0].host);
  assert(shared.photo_private[0].device + 105 ==
         private_inventory.photo_private[0].device);
  assert(shared.photo_private[1].device + 42 ==
         private_inventory.photo_private[1].device);
  assert(shared.geo_private[0].device ==
         private_inventory.geo_private[0].device);
  const auto shared_peaks = FixedCohortPeaks(shared);
  const auto private_peaks = FixedCohortPeaks(private_inventory);
  assert(shared_peaks.photo.device + 42 == private_peaks.photo.device);
  assert(shared_peaks.geo.device == private_peaks.geo.device);
  assert(shared_peaks.photo.host == private_peaks.photo.host + 105);
  auto permuted = jobs;
  permuted[0].photo_sources = {2, 0, 0};
  assert(BuildFixedCohortLogicalInventory(shapes, permuted, shared_config)
             .photo_shared.device == 105);
  // Zero overlap with opposed aspect ratios: explicit sharing costs MORE.
  const auto padded = BuildFixedCohortLogicalInventory(
      {{10, 1}, {1, 10}}, {{0, {1}, {1}}, {1, {0}, {0}}}, shared_config);
  assert(padded.photo_shared.device == 200);
  const auto admitted_shared = AdmitFixedCohortBudget(shared, {100000, 100000});
  (void)AdmitFixedCohortBudget(shared, admitted_shared);
  bool shared_cap_refused = false;
  try {
    (void)AdmitFixedCohortBudget(
        shared, {admitted_shared.host - 1, admitted_shared.device});
  } catch (const std::invalid_argument&) {
    shared_cap_refused = true;
  }
  assert(shared_cap_refused);
  const auto slack =
      MarginBytes(shared_peaks.geo.device) - shared_peaks.geo.device;
  auto staging_within = shared.geo_staging_device;
  staging_within[0] += slack;
  CheckFixedCohortPhysicalCopies(
      shared, shared.snapshots.device, staging_within);
  ++staging_within[0];
  bool shared_margin_refused = false;
  try {
    CheckFixedCohortPhysicalCopies(
        shared, shared.snapshots.device, staging_within);
  } catch (const std::invalid_argument&) {
    shared_margin_refused = true;
  }
  assert(shared_margin_refused);
  auto invalid_shared = shared;
  invalid_shared.photo_shared = {};
  bool missing_union_refused = false;
  try {
    (void)AdmitFixedCohortBudget(invalid_shared, {100000, 100000});
  } catch (const std::invalid_argument&) {
    missing_union_refused = true;
  }
  assert(missing_union_refused);
  FixedCohortBudget budget;
  budget.caller_inputs = {100, 0};
  budget.frozen_inputs = {100, 0};
  budget.host_outputs = {200, 0};
  budget.snapshots = {0, 300};
  budget.process = {10, 20};
  budget.unknown = {1, 2};
  budget.unknown_provenance =
      "fixture: metadata and driver allocation uncertainty";
  budget.geo_staging_device = {1, 1};
  budget.photo_private = {{30, 40}, {50, 60}};
  budget.geo_private = {{70, 80}, {90, 100}};
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
  CheckFixedCohortPhysicalCopies(budget, 313, {1, 1});
  // Simultaneous snapshot +13 and staging +3 exactly consume GEO's
  // original 16-byte margin; serialized staging extras are not summed.
  CheckFixedCohortPhysicalCopies(budget, 313, {4, 4});
  bool pitch_refused = false;
  try {
    CheckFixedCohortPhysicalCopies(budget, 314, {1, 1});
  } catch (const std::invalid_argument&) {
    pitch_refused = true;
  }
  assert(pitch_refused);
  bool combined_refused = false;
  try {
    // Snapshot +13 and staging +4 exceed GEO's existing 16-byte margin.
    CheckFixedCohortPhysicalCopies(budget, 313, {5, 1});
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
  // Public admission seam shared by both native phase observations. Raw host
  // demand is 1000: 500 classified live process + 100 caller + 400 other.
  auto live = budget;
  live.process.host = 500;
  live.frozen_inputs.host = 100;
  live.host_outputs.host = 100;
  live.unknown.host = 100;
  live.photo_private[0].host = live.geo_private[0].host = 50;
  live.photo_private[1].host = live.geo_private[1].host = 50;
  live.already_live_process = {{{500, 10}, {500, 5}}};
  live.already_live_process_provenance = {
      {"fixture classified allocations live before copy",
       "fixture classified allocations still live before GEO"}};
  for (const bool geo : {false, true}) {
    assert(AdmitFixedCohortBudget(live, {1030, 1000}).host == 1030);
    CheckFixedCohortPhaseHeadroom(live, geo, {450, 1000}, {100, 0});
    CheckFixedCohortPhaseHeadroom(live, geo, {430, 1000}, {100, 0});
    const auto refuses =
        [&](const FixedCohortBudget& b, ByteCharge free, ByteCharge payload) {
          bool rejected = false;
          try {
            CheckFixedCohortPhaseHeadroom(b, geo, free, payload);
          } catch (const std::invalid_argument&) {
            rejected = true;
          }
          assert(rejected);
        };
    refuses(live, {429, 1000}, {100, 0});
    // Device process credit is phase-specific, not carried over from PHOTO.
    // Before copy keeps the full peak reservation (518), not PHOTO's 435.
    const uint64_t remaining_device = geo ? 513 : 508;
    CheckFixedCohortPhaseHeadroom(live, geo, {430, remaining_device}, {100, 0});
    refuses(live, {430, remaining_device - 1}, {100, 0});
    auto incremental_process = live;
    incremental_process.already_live_process[geo].host = 400;
    refuses(incremental_process, {450, 1000}, {100, 0});
    auto malformed = live;
    malformed.already_live_process[geo].host = 501;
    refuses(malformed, {1000, 1000}, {100, 0});
    bool invalid_admission = false;
    try {
      (void)AdmitFixedCohortBudget(malformed, {1000, 1000});
    } catch (const std::invalid_argument&) {
      invalid_admission = true;
    }
    assert(invalid_admission);
    malformed = live;
    malformed.already_live_process[geo].device = 21;
    refuses(malformed, {1000, 1000}, {100, 0});
    malformed = live;
    malformed.already_live_process_provenance[geo].clear();
    refuses(malformed, {1000, 1000}, {100, 0});
    // Process cannot be included a second time as caller/map payload.
    refuses(live, {1000, 1000}, {600, 0});
  }
  // Actual snapshots can exceed their raw item bound inside the original
  // aggregate margin; the shared observation seam must not reimpose it.
  CheckFixedCohortPhaseHeadroom(live, true, {430, 200}, {100, 313});
  // No live-process declaration means no process deduction, even if free
  // would otherwise suggest that some unknown overhead is already resident.
  auto unmeasured = live;
  unmeasured.already_live_process = {};
  bool unmeasured_refused = false;
  try {
    CheckFixedCohortPhaseHeadroom(unmeasured, false, {450, 1000}, {100, 0});
  } catch (const std::invalid_argument&) {
    unmeasured_refused = true;
  }
  assert(unmeasured_refused);
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
  // Four logical jobs retain every map/output, but only resourcewise top-N
  // private inventories can coexist. Host and device maxima differ by job.
  auto four = budget;
  four.photo_private = {{10, 100}, {80, 20}, {30, 70}, {60, 40}};
  four.geo_private = four.photo_private;
  four.geo_staging_device = {1, 1, 1, 1};
  four.snapshots.device = 600;
  four.host_outputs.host = 400;
  four.max_active_owners = 2;
  auto four_peaks = FixedCohortPeaks(four);
  assert(four_peaks.photo.host == 751 && four_peaks.photo.device == 792);
  assert(AdmitFixedCohortBudget(four, {774, 816}).host == 774);
  four.max_active_owners = 1;
  four_peaks = FixedCohortPeaks(four);
  assert(four_peaks.photo.host == 691 && four_peaks.photo.device == 722);
  // Every snapshot discrepancy is aggregated before this public check;
  // serialized staging peaks use the largest excess, not the final job only.
  CheckFixedCohortPhysicalCopies(four, 621, {1, 1, 1, 1});
  // GEO's 22-byte margin covers snapshot +21 and staging +1, not +2.
  CheckFixedCohortPhysicalCopies(four, 621, {2, 2, 2, 2});
  bool early_staging_refused = false;
  try {
    CheckFixedCohortPhysicalCopies(four, 621, {3, 1, 1, 1});
  } catch (const std::invalid_argument&) {
    early_staging_refused = true;
  }
  assert(early_staging_refused);
  bool late_staging_refused = false;
  try {
    CheckFixedCohortPhysicalCopies(four, 621, {1, 1, 1, 5});
  } catch (const std::invalid_argument&) {
    late_staging_refused = true;
  }
  assert(late_staging_refused);
  four.geo_private.pop_back();
  bool shape_refused = false;
  try {
    (void)AdmitFixedCohortBudget(four, {1000, 1000});
  } catch (const std::invalid_argument&) {
    shape_refused = true;
  }
  assert(shape_refused);
  for (size_t active : {size_t{0}, size_t{3}}) {
    auto invalid = budget;
    invalid.max_active_owners = active;
    bool active_refused = false;
    try {
      (void)AdmitFixedCohortBudget(invalid, {1000, 1000});
    } catch (const std::invalid_argument&) {
      active_refused = true;
    }
    assert(active_refused);
  }
  auto missing_late_bound = budget;
  missing_late_bound.photo_private.resize(4, {30, 40});
  missing_late_bound.geo_private.resize(4, {70, 80});
  missing_late_bound.geo_staging_device = {1, 1, 1, 0};
  bool late_bound_refused = false;
  try {
    (void)AdmitFixedCohortBudget(missing_late_bound, {1000, 1000});
  } catch (const std::invalid_argument&) {
    late_bound_refused = true;
  }
  assert(late_bound_refused);
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
