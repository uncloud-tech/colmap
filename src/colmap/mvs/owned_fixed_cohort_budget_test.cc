// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_cohort_lifecycle_test_util.h"
#include "colmap/mvs/owned_fixed_cohort_inventory.h"
#include "colmap/mvs/owned_pair_run.h"

#include <atomic>
#include <cassert>
#include <limits>
#include <stdexcept>

using namespace colmap::mvs::owned;

int main() {
  // Observation is incremental FREE, never enlarged by caps or declarations.
  const auto free_caps = LimitFixedCohortAvailable({100, 200}, 90, 300);
  assert(free_caps.host == 90 && free_caps.device == 200);
  const auto uncapped = LimitFixedCohortAvailable({100, 200});
  assert(uncapped.host == 100 && uncapped.device == 200);
  const auto zero = LimitFixedCohortAvailable({100, 200}, 0, 0);
  assert(zero.host == 0 && zero.device == 0);
  const auto maximum = std::numeric_limits<uint64_t>::max();
  const auto large = LimitFixedCohortAvailable({maximum, maximum}, maximum, 1);
  assert(large.host == maximum && large.device == 1);
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
  // Storage policy compares both admitted phase counts, not their max/sum.
  assert(PreferSharedFixedCohortCounts({1, 2}, {2, 2}));
  assert(PreferSharedFixedCohortCounts({2, 1}, {2, 2}));
  assert(!PreferSharedFixedCohortCounts({1, 3}, {2, 2}));
  assert(!PreferSharedFixedCohortCounts({3, 1}, {2, 2}));
  assert(!PreferSharedFixedCohortCounts({2, 2}, {2, 2}));
  // Auto is a metadata-only request; equal admitted counts prefer private.
  const auto automatic = SelectFixedCohortPhotoSourceStorage(
      shapes, jobs, private_config, {100000, 100000});
  assert(automatic.budget.photo_source_storage ==
         FixedCohortPhotoSourceStorage::kPrivate);
  assert(automatic.budget.max_active_owners == 2);
  // Same canonical inputs, one margin. A large repeated PHOTO source and
  // tiny closed GEO producers isolate the storage/count decision.
  const std::vector<FixedCohortImageShape> auto_images =
      {{1, 1}, {1, 1}, {100, 100}};
  const std::vector<FixedCohortJobShape> auto_jobs =
      {{0, {2, 2, 2, 2}, {1}}, {1, {2, 2, 2, 2}, {0}}};
  auto auto_config = private_config;
  auto_config.max_active_owners = 1;
  const auto choose = [&](ByteCharge cap, bool phases = false,
                          size_t ceiling = 0) {
    return SelectFixedCohortPhotoSourceStorage(
        auto_images, auto_jobs, auto_config, cap, ceiling, phases);
  };
  for (bool phases : {false, true}) {
    const auto better = choose({1000000, 50000}, phases);
    assert(better.budget.photo_source_storage ==
           FixedCohortPhotoSourceStorage::kShared);
    assert(better.budget.max_active_owners == 2);
    assert(FixedCohortPhaseActiveOwners(better.budget, false) == 2);
    assert(better.reason == "shared dominates admitted PHOTO/GEO counts");
    assert(choose({1000000, 20000}, phases).reason ==
           "only shared fits complete pipeline");
    assert(choose({65000, 50000}, phases).reason ==
           "only private fits complete pipeline");
    assert(choose({1000000, 1000000}, phases).reason ==
           "equal counts prefer private");
    assert(choose({1000000, 50000}, phases, 1).reason ==
           "equal counts prefer private");
    bool neither = false;
    try {
      choose({0, 0}, phases);
    } catch (const FixedCohortCapacityRefusal& error) {
      const std::string message = error.what();
      neither = message.find("private:") != std::string::npos &&
                message.find("shared:") != std::string::npos &&
                message.find("host and device capacity") != std::string::npos;
    }
    assert(neither);
  }
  assert(auto_config.photo_source_storage ==
         FixedCohortPhotoSourceStorage::kPrivate);
  assert(auto_config.photo_private.size() == 2);  // Request not mutated.
  // No-overlap, opposed aspect ratios: global padding and extra host union
  // erase any gain. Even unlimited memory cannot promote a tied shared plan.
  const auto padded_auto = SelectFixedCohortPhotoSourceStorage(
      {{10, 1}, {1, 10}}, {{0, {1}, {1}}, {1, {0}, {0}}},
      auto_config, {1000000, 1000000});
  assert(padded_auto.budget.photo_source_storage ==
         FixedCohortPhotoSourceStorage::kPrivate);
  auto malformed_config = auto_config;
  malformed_config.unknown_provenance.clear();
  bool malformed_not_capacity = false;
  try {
    SelectFixedCohortPhotoSourceStorage(auto_images, auto_jobs,
                                        malformed_config, {0, 0});
  } catch (const FixedCohortCapacityRefusal&) {
    assert(false);
  } catch (const std::invalid_argument&) {
    malformed_not_capacity = true;
  }
  assert(malformed_not_capacity);
  // Overflow at the full ceiling must not masquerade as singleton refusal.
  auto overflowing_config = auto_config;
  overflowing_config.process.device = maximum - 100000;
  bool auto_overflow = false;
  try {
    SelectFixedCohortPhotoSourceStorage(auto_images, auto_jobs,
                                        overflowing_config, {0, 0});
  } catch (const std::overflow_error&) {
    auto_overflow = true;
  }
  assert(auto_overflow);
  bool shared_padding_overflow = false;
  try {
    SelectFixedCohortPhotoSourceStorage(
        {{1, 1}, {1, 1}, {1ull << 32, 1}, {1, 1ull << 32}},
        {{0, {2}, {1}}, {1, {3}, {0}}}, auto_config, {0, 0});
  } catch (const std::overflow_error&) {
    shared_padding_overflow = true;
  }
  assert(shared_padding_overflow);
  bool zero_shape = false;
  try {
    SelectFixedCohortPhotoSourceStorage(
        {{1, 1}, {1, 1}, {100, 100}, {0, 100}},
        auto_jobs, auto_config, {0, 0});
  } catch (const FixedCohortCapacityRefusal&) {
    assert(false);
  } catch (const std::invalid_argument&) {
    zero_shape = true;
  }
  assert(zero_shape);
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
  // FREE ceilings keep all declared process/unknown charges and one margin.
  // Even a cap claiming total memory cannot turn 588 free host bytes into 589.
  bool no_credit = false;
  try {
    (void)AdmitFixedCohortBudget(
        budget, LimitFixedCohortAvailable({588, 518}, maximum, maximum));
  } catch (const std::invalid_argument&) {
    no_credit = true;
  }
  assert(no_credit);
  assert(AdmitFixedCohortBudget(
             budget, LimitFixedCohortAvailable({589, 518})).host == 589);
  for (const auto available : {ByteCharge{0, 518}, ByteCharge{589, 0}}) {
    bool singleton_refused = false;
    try {
      (void)SelectFixedCohortActiveOwners(
          budget, LimitFixedCohortAvailable(available));
    } catch (const std::invalid_argument& error) {
      singleton_refused = std::string(error.what()).find("no singleton fits") !=
                          std::string::npos;
    }
    assert(singleton_refused);
  }
  auto observed_reserve = budget;
  observed_reserve.headroom_reserve = {1, 1};
  bool observed_reserve_refused = false;
  try {
    (void)AdmitFixedCohortBudget(
        observed_reserve, LimitFixedCohortAvailable({589, 518}));
  } catch (const std::invalid_argument&) {
    observed_reserve_refused = true;
  }
  assert(observed_reserve_refused);
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
  // Independent worked top-N example: retained host/device = 611/622.
  // Active totals: host 80,140,170,180; device 100,170,210,230.
  four.headroom_reserve = {7, 11};
  auto selected = SelectFixedCohortActiveOwners(four, {781, 827});
  assert(selected.count == 2 && selected.reason == "host and device capacity");
  assert(SelectFixedCohortActiveOwners(four, {10000, 755}).count == 1);
  assert(SelectFixedCohortActiveOwners(four, {812, 10000}).count == 3);
  assert(SelectFixedCohortActiveOwners(four, {822, 889}).count == 4);
  assert(SelectFixedCohortActiveOwners(four, {10000, 10000}, 3).count == 3);
  assert(SelectFixedCohortActiveOwners(four, {10000, 10000}, 99).count == 4);
  // Same envelope, asymmetric phases: GEO need not throttle PHOTO.
  auto asymmetric = four;
  asymmetric.photo_private.assign(4, {10, 10});
  const auto phases =
      SelectFixedCohortPhaseActiveOwners(asymmetric, {781, 827});
  assert(phases[0].count == 4 && phases[1].count == 2);
  assert(phases[0].reason == "job count");
  asymmetric.phase_active_owners = {4, 2};
  asymmetric.max_active_owners = 4;
  assert(FixedCohortPeaks(asymmetric).photo.host == 651);
  AdmitFixedCohortBudget(asymmetric, {781, 827});
  // PHOTO observation uses its actual phase peak only in the explicit mode.
  const auto photo_peak = FixedCohortPeaks(asymmetric).photo;
  CheckFixedCohortPhaseHeadroom(
      asymmetric,
      false,
      {CheckedAdd(MarginBytes(photo_peak.host),
                  asymmetric.headroom_reserve.host),
       CheckedAdd(MarginBytes(photo_peak.device),
                  asymmetric.headroom_reserve.device)},
      {});
  auto uniform = asymmetric;
  uniform.phase_active_owners = {};
  bool conservative = false;
  try {
    CheckFixedCohortPhaseHeadroom(
        uniform,
        false,
        {MarginBytes(photo_peak.host) + 7, MarginBytes(photo_peak.device) + 11},
        {});
  } catch (const std::invalid_argument&) {
    conservative = true;
  }
  assert(conservative);
  std::swap(asymmetric.photo_private, asymmetric.geo_private);
  asymmetric.phase_active_owners = {};
  const auto reversed =
      SelectFixedCohortPhaseActiveOwners(asymmetric, {781, 827});
  assert(reversed[0].count == 2 && reversed[1].count == 4);
  // Singleton fits and count two exceeds capacity, but count four overflows.
  // Capacity refusal must not hide malformed in-ceiling arithmetic.
  auto overflowing = four;
  overflowing.photo_private.assign(4, {10, 5000000000000000000ull});
  overflowing.geo_private = overflowing.photo_private;
  bool later_overflow_refused = false;
  try {
    SelectFixedCohortActiveOwners(overflowing,
                                 {10000, 6000000000000000000ull});
  } catch (const std::overflow_error&) {
    later_overflow_refused = true;
  }
  assert(later_overflow_refused);
  // An explicit ceiling excludes arithmetic outside the search envelope.
  assert(SelectFixedCohortActiveOwners(
             overflowing, {10000, 6000000000000000000ull}, 2)
             .count == 1);
  for (const bool geo : {false, true}) {
    auto hidden = four;
    (geo ? hidden.geo_private : hidden.photo_private)
        .assign(4, {10, 5000000000000000000ull});
    bool overflow = false;
    try {
      SelectFixedCohortPhaseActiveOwners(hidden,
                                         {10000, 6000000000000000000ull});
    } catch (const std::overflow_error&) {
      overflow = true;
    }
    assert(overflow);
    const auto limited = SelectFixedCohortPhaseActiveOwners(
        hidden, {10000, 6000000000000000000ull}, 2);
    assert(limited[geo].count == 1 && limited[!geo].count == 2);
  }
  for (const bool geo : {false, true}) {
    auto unequal = four;
    (geo ? unequal.photo_private : unequal.geo_private).assign(4, {10, 10});
    auto choices = SelectFixedCohortPhaseActiveOwners(unequal, {781, 10000});
    assert(choices[geo].count == 2 && choices[geo].reason == "host capacity");
    choices = SelectFixedCohortPhaseActiveOwners(unequal, {10000, 827});
    assert(choices[geo].count == 2 && choices[geo].reason == "device capacity");
    for (const size_t ceiling : {size_t{1}, size_t{3}, size_t{99}}) {
      choices =
          SelectFixedCohortPhaseActiveOwners(unequal, {10000, 10000}, ceiling);
      assert(choices[0].count == std::min(ceiling, size_t{4}));
      assert(choices[1].count == choices[0].count);
    }
    bool refused = false;
    try {
      SelectFixedCohortPhaseActiveOwners(unequal, {700, 10000});
    } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).find(geo ? "GEO no singleton"
                                               : "PHOTO no singleton") !=
                std::string::npos;
    }
    assert(refused);
  }
  for (const auto counts :
       {std::array<size_t, 2>{0, 1}, {1, 0}, {5, 1}, {1, 1}}) {
    auto invalid = four;
    invalid.max_active_owners = 4;
    invalid.phase_active_owners = counts;
    bool refused = false;
    try {
      AdmitFixedCohortBudget(invalid, {10000, 10000});
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused);
  }
  auto independent = four;
  for (auto& charge : independent.photo_private) charge.device = 10;
  for (auto& charge : independent.geo_private) charge.host = 10;
  const auto separate_limits = SelectFixedCohortPhaseActiveOwners(independent, {781, 755});
  assert(separate_limits[0].count == 2 && separate_limits[0].reason == "host capacity");
  assert(separate_limits[1].count == 1 && separate_limits[1].reason == "device capacity");
  independent.max_active_owners = 2;
  independent.phase_active_owners = {2, 1};
  const auto exact_phase_peak = FixedCohortPeaks(independent);
  assert(exact_phase_peak.photo.host == 751 && exact_phase_peak.photo.device == 642);
  assert(exact_phase_peak.geo.host == 621 && exact_phase_peak.geo.device == 722);
  const auto once = AdmitFixedCohortBudget(independent, {781, 755});
  assert(once.host == 774 && once.device == 744);
  // Positive private charges make demand strictly increase with count.
  // Across this capacity interval, count three remains the highest fit.
  for (uint64_t host_cap : {uint64_t{812}, uint64_t{817}, uint64_t{821}})
    assert(SelectFixedCohortActiveOwners(four, {host_cap, 10000}).count == 3);
  assert(SelectFixedCohortActiveOwners(four, {822, 10000}).count == 4);
  bool singleton_refused = false;
  try {
    SelectFixedCohortActiveOwners(four, {718, 10000});
  } catch (const std::invalid_argument& e) {
    singleton_refused =
        std::string(e.what()).find("no singleton fits: host capacity") !=
        std::string::npos;
  }
  assert(singleton_refused);
  assert(SelectFixedCohortActiveOwners(four, {781, 10000}).reason ==
         "host capacity");
  assert(SelectFixedCohortActiveOwners(four, {10000, 827}).reason ==
         "device capacity");
  for (size_t active : {size_t{3}, size_t{4}}) {
    four.max_active_owners = active;
    const auto peak = FixedCohortPeaks(four).peak;
    const ByteCharge cap{MarginBytes(peak.host) + 7,
                         MarginBytes(peak.device) + 11};
    assert(SelectFixedCohortActiveOwners(four, cap).count == active);
    assert(
        SelectFixedCohortActiveOwners(four, {cap.host - 1, cap.device}).count ==
        active - 1);
    const auto slack = MarginBytes(peak.device) - peak.device;
    CheckFixedCohortPhysicalCopies(four, 600 + slack, {1, 1, 1, 1});
    bool refused = false;
    try {
      CheckFixedCohortPhysicalCopies(four, 601 + slack, {1, 1, 1, 1});
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused);
  }
  // Execute the real rolling loop at the chosen counts. Cleanup must finish
  // before retirement/join makes credits reusable, including injected failure.
  for (size_t active : {size_t{3}, size_t{4}}) {
    for (bool fail : {false, true}) {
      CohortLifecycleRendezvous lifecycle(active, std::chrono::seconds(1));
      std::atomic<size_t> cleaned{0};
      std::vector<std::atomic<bool>> live(7);
      for (auto& value : live) value = false;
      struct Payload {
        std::atomic<bool>& live;
        explicit Payload(std::atomic<bool>& value) : live(value) {
          live = true;
        }
        ~Payload() { live = false; }
      };
      PairCoordinator coordinator({1 + active * 10, 1 + active * 10},
                                  {1, 1},
                                  active,
                                  std::vector<ByteCharge>(7, {10, 10}),
                                  std::chrono::seconds(3));
      bool failed = false;
      try {
        RunRollingOwners(
            coordinator,
            [&](size_t job) {
              coordinator.BeginPreparation(job);
              coordinator.ReadyAndWait(job);
              Payload payload(live[job]);
              if (!fail) lifecycle.Observe(0, job);
              if (fail && job == 2)
                throw std::runtime_error("injected owner failure");
              coordinator.SolverDone(job);
              coordinator.BeginOutput(job);
              coordinator.EndOutput(job);
            },
            [](size_t, const char*) {},
            PairThreadFactory{},
            [&](size_t job) {
              assert(!live[job]);
              ++cleaned;
            });
      } catch (const std::runtime_error& e) {
        if (std::string(e.what()) != "injected owner failure") throw;
        failed = true;
      }
      assert(failed == fail);
      if (!fail) assert(lifecycle.Observed(0) == active);
      for (const auto& value : live) assert(!value);
      assert(cleaned == coordinator.AdmittedCount());
      assert(coordinator.ReservedBytes().host == 0);
      assert(coordinator.ReservedBytes().device == 0);
    }
  }
  // A silent two-owner clamp cannot satisfy the three/four-owner proof.
  // Timeout must propagate through cancellation, cleanup, retirement and join.
  for (size_t target : {size_t{3}, size_t{4}}) {
    CohortLifecycleRendezvous lifecycle(target, std::chrono::milliseconds(50));
    PairCoordinator coordinator({41, 41},
                                {1, 1},
                                2,
                                std::vector<ByteCharge>(4, {10, 10}),
                                std::chrono::seconds(3));
    bool timed_out = false;
    try {
      RunRollingOwners(
          coordinator,
          [&](size_t job) {
            coordinator.BeginPreparation(job);
            coordinator.ReadyAndWait(job);
            lifecycle.Observe(0, job);
            coordinator.SolverDone(job);
            coordinator.BeginOutput(job);
            coordinator.EndOutput(job);
          },
          [](size_t, const char*) {});
    } catch (const std::runtime_error& e) {
      timed_out = std::string(e.what()) ==
                  "owner lifecycle rendezvous timed out";
    }
    assert(timed_out);
    assert(lifecycle.Observed(0) == 2);
    assert(coordinator.ReservedBytes().host == 0);
    assert(coordinator.ReservedBytes().device == 0);
  }
  four.max_active_owners = 1;
  four.headroom_reserve = {};
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
  // Preserve legacy manual pair-as-ceiling behavior for a singleton job;
  // automatic selection still reports only the one logical owner available.
  auto singleton = budget;
  singleton.photo_private.resize(1);
  singleton.geo_private.resize(1);
  singleton.geo_staging_device.resize(1);
  AdmitFixedCohortBudget(singleton, {10000, 10000});
  assert(SelectFixedCohortActiveOwners(singleton, {10000, 10000}).count == 1);
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
  for (int malformed = 0; malformed < 5; ++malformed) {
    auto invalid = budget;
    if (malformed == 0) invalid.unknown = {};
    if (malformed == 1) invalid.unknown_provenance.clear();
    if (malformed == 2) invalid.photo_private.back().host = 0;
    if (malformed == 3) invalid.geo_staging_device.back() = 0;
    if (malformed == 4) invalid.already_live_process[0].host = maximum;
    bool rejected = false;
    try { SelectFixedCohortPhaseActiveOwners(invalid, {}); }
    catch (const std::invalid_argument& e) {
      rejected = std::string(e.what()).find("no singleton fits") == std::string::npos;
    }
    assert(rejected);
  }
  // Exact CLI fixture inventory: each resource cap is derived from the same
  // canonical singleton, not a hand-tuned admission bypass.
  for (const bool heavy_photo : {false, true}) {
    for (const bool shared_photo : {false, true}) {
      auto config = budget;
      config.max_active_owners = 1;
      config.photo_source_storage = shared_photo ? FixedCohortPhotoSourceStorage::kShared
                                                 : FixedCohortPhotoSourceStorage::kPrivate;
      std::vector<FixedCohortJobShape> tiny_jobs;
      for (int i = 0; i < 3; ++i)
        tiny_jobs.push_back({i, std::vector<int>(heavy_photo ? 20 : 1, (i + 1) % 3),
                               std::vector<int>(heavy_photo ? 1 : 20, (i + 1) % 3)});
      auto tiny = BuildFixedCohortLogicalInventory({{20,10},{20,10},{20,10}}, tiny_jobs, config);
      const auto admitted_tiny = AdmitFixedCohortBudget(tiny, {maximum, maximum});
      const auto counts = SelectFixedCohortPhaseActiveOwners(
          tiny, {maximum, CheckedAdd(admitted_tiny.device, tiny.headroom_reserve.device)});
      assert(counts[heavy_photo ? 0 : 1].count == 1);
      assert(counts[heavy_photo ? 1 : 0].count == 3);
    }
  }
  // Mirror new actual-CLI AUTO fixture cap derivation, including the explicit
  // GPU-only allowance. It is test overhead, not estimator calibration.
  for (uint64_t unknown_device : {1ull << 20, 4ull << 20}) {
    FixedCohortBudget config;
    config.max_active_owners = 1;
    config.process = {1 << 20, 1 << 20};
    config.unknown = {1 << 20, unknown_device};
    config.unknown_provenance = "uncalibrated tiny fixture allowance";
    config.headroom_reserve = {1 << 20, 1 << 20};
    const std::vector<FixedCohortImageShape> tiny_images(3, {20, 10});
    std::vector<FixedCohortJobShape> tiny_jobs;
    for (int i = 0; i < 3; ++i)
      tiny_jobs.push_back({i, std::vector<int>(20, (i + 1) % 3),
                          {(i + 1) % 3}});
    for (const bool shared_mode : {false, true}) {
      for (size_t count : {1, 3}) {
        config.photo_source_storage =
            shared_mode ? FixedCohortPhotoSourceStorage::kShared
                        : FixedCohortPhotoSourceStorage::kPrivate;
        config.max_active_owners = count;
        const auto explicit_budget = BuildFixedCohortLogicalInventory(
            tiny_images, tiny_jobs, config);
        const auto admitted =
            AdmitFixedCohortBudget(explicit_budget, {maximum, maximum});
        const ByteCharge caps{
            1 << 30, CheckedAdd(admitted.device, config.headroom_reserve.device)};
        const auto chosen = SelectFixedCohortPhotoSourceStorage(
            tiny_images, tiny_jobs, config, caps, count);
        assert(chosen.budget.photo_source_storage ==
               explicit_budget.photo_source_storage);
        assert(chosen.budget.max_active_owners == count);
        assert(chosen.budget.photo_shared.device == (shared_mode ? 600 : 0));
        assert(chosen.budget.snapshots.device == explicit_budget.snapshots.device);
        if (shared_mode || count == 3) {
          const auto phases = SelectFixedCohortPhotoSourceStorage(
              tiny_images, tiny_jobs, config, caps, 0, true);
          assert(phases.budget.photo_source_storage ==
                 explicit_budget.photo_source_storage);
          assert(FixedCohortPhaseActiveOwners(phases.budget, false) == count);
          assert(FixedCohortPhaseActiveOwners(phases.budget, true) == 3);
        }
      }
    }
  }
  auto bad = budget;
  bad.process.host = std::numeric_limits<uint64_t>::max();
  bool refused = false;
  try {
    (void)FixedCohortPeaks(bad);
  } catch (const std::overflow_error&) {
    refused = true;
  }
  assert(refused);
  refused = false;
  try {
    SelectFixedCohortActiveOwners(bad, {1000, 1000});
  } catch (const std::overflow_error&) {
    refused = true;
  }
  assert(refused);
}
