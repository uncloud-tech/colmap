// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_stereo_admission.h"

#include <cassert>
#include <iostream>
#include <limits>

namespace {
template <class E, class F>
void Refuses(F action) {
  bool refused = false;
  try {
    action();
  } catch (const E&) {
    refused = true;
  }
  assert(refused);
}
}  // namespace

using namespace colmap::mvs::owned;

int main() {
  StereoAdmissionRequest request;
  request.images = {{640, 480}, {641, 481}, {639, 479}};
  StereoAdmissionReference reference;
  reference.reference = 0;
  reference.sources = {
      {{2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2}, {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1}}};
  auto second = reference;
  second.reference = 1;
  second.sources = {{{0}, {0}}};
  auto third = second;
  third.reference = 2;
  request.references = {reference, second, third};
  request.capacity = {2000000000, 2000000000};
  request.disk_capacity = 2000000000;
  request.process = {1000, 2000};
  request.unknown = {3000, 4000};
  request.reserve = {5000, 6000};
  const auto plan = PlanSingleDeviceStereoAdmission(request);
  assert(plan.active_owners == 1);
  assert(plan.records[0][0].mode == StereoVolumeMode::kResidentPrivate);
  assert(plan.records[1][0].sources == reference.sources[1]);
  assert(plan.records[0][0].scratch_disk_bytes == 0);
  assert(plan.records[0][0].initialization.device > 0);
  assert(plan.records[0][0].peak.host == 10766223);
  const auto snapshot =
      BuildSnapshotStereoInventory(request, FixedCohortBudget{});
  assert(snapshot.records[0][0].mode == StereoVolumeMode::kResidentPrivate);
  assert(snapshot.budget.snapshots.device == 14745632);
  assert(snapshot.records[1][0].staging_device_bytes == 16253952);
  assert(snapshot.budget.geo_staging_device[0] ==
         snapshot.records[1][0].staging_device_bytes);
  request.capacity.device = 60000000;
  const auto mixed = PlanSingleDeviceStereoAdmission(request);
  assert(mixed.records[0][0].mode == StereoVolumeMode::kStripFile);
  assert(mixed.records[0][1].mode == StereoVolumeMode::kResidentPrivate);
  assert(mixed.records[1][0].mode == StereoVolumeMode::kStripFile);
  assert(mixed.records[0][0].scratch_disk_bytes == 30412800);
  assert(mixed.records[0][0].cache_host_bytes == 30412800);
  assert(mixed.records[0][0].columns == 32);
  assert(mixed.required_disk_bytes ==
         mixed.output_disk_bytes + mixed.scratch_disk_bytes);
  const auto& strip = mixed.records[0][0];
  assert(SameStereoExecutionRecord(strip, strip));
  ValidateStereoExecutionBinding(
      strip, 0, 640, 480, strip.sources, false, strip.policy);
  auto wrong_sources = strip.sources;
  std::swap(wrong_sources[0], wrong_sources[1]);
  Refuses<std::invalid_argument>([&] {
    ValidateStereoExecutionBinding(
        strip, 0, 640, 480, wrong_sources, false, strip.policy);
  });
  auto changed = strip;
  changed.mode = StereoVolumeMode::kResidentPrivate;
  assert(!SameStereoExecutionRecord(strip, changed));
  changed = strip;
  changed.policy.async_pool = true;
  assert(!SameStereoExecutionRecord(strip, changed));
  changed = strip;
  ++changed.rotation_planes[0].pitch_ceiling;
  assert(!SameStereoExecutionRecord(strip, changed));
  changed = strip;
  std::swap(changed.sources[0], changed.sources[1]);
  assert(!SameStereoExecutionRecord(strip, changed));
  std::vector<RotationAllocationRecord> returned;
  const auto layout =
      BuildOwnedExecutionRotationLayout(640, 480, 11, true, true);
  for (size_t i = 0; i < layout.slot_layouts.size(); ++i)
    returned.push_back(MakeRotationAllocationRecord(
        layout.slot_layouts[i], strip.rotation_planes[i].pitch_ceiling));
  ValidateStereoReturnedPitches(strip,
                                returned,
                                strip.initialization_prng_pitch,
                                {strip.staging_float_pitch,
                                 strip.staging_float_pitch,
                                 strip.staging_float_pitch,
                                 strip.staging_mask_pitch});
  bool pitch_refused = false;
  returned[0].pitch_bytes += 1;
  try {
    ValidateStereoReturnedPitches(strip,
                                  returned,
                                  strip.initialization_prng_pitch,
                                  {strip.staging_float_pitch,
                                   strip.staging_float_pitch,
                                   strip.staging_float_pitch,
                                   strip.staging_mask_pitch});
  } catch (const std::invalid_argument&) {
    pitch_refused = true;
  }
  assert(pitch_refused);
  auto invalid = request;
  invalid.references[0].sources[0] = {0};
  Refuses<std::invalid_argument>(
      [&] { PlanSingleDeviceStereoAdmission(invalid); });
  assert(mixed.output_disk_bytes == 29491384);
  assert(mixed.required_disk_bytes == 59904184);
  assert(mixed.common_host_raw == 31334468);
  assert(mixed.loader_host_raw == 1847204);
  assert(strip.padded_source_pixels == 3391531);
  assert(strip.initialization.device > strip.steady.device);
  assert(strip.staging_host_bytes == 903680);
  assert(strip.staging_device_bytes == 14417920);
  assert(MarginBytes(100) == 103 && MarginBytes(101) == 105);

  // Fit exactly at one aggregate margin plus reserves; a byte below the strip
  // minimum fails instead of adding another margin or borrowing the reserve.
  auto boundary = request;
  boundary.capacity = mixed.admitted;
  const auto exact = PlanSingleDeviceStereoAdmission(boundary);
  assert(exact.admitted.host == boundary.capacity.host);
  assert(exact.admitted.device == boundary.capacity.device);
  --boundary.capacity.device;
  Refuses<StereoAdmissionRefusal>(
      [&] { PlanSingleDeviceStereoAdmission(boundary); });
  boundary = request;
  boundary.capacity.host = mixed.admitted.host - 1;
  try {
    PlanSingleDeviceStereoAdmission(boundary);
    assert(false);
  } catch (const StereoAdmissionRefusal& error) {
    assert(error.resource == StereoAdmissionResource::kHost);
  }
  boundary = request;
  boundary.disk_capacity = mixed.required_disk_bytes;
  PlanSingleDeviceStereoAdmission(boundary);
  --boundary.disk_capacity;
  try {
    PlanSingleDeviceStereoAdmission(boundary);
    assert(false);
  } catch (const StereoAdmissionRefusal& error) {
    assert(error.resource == StereoAdmissionResource::kDisk);
  }
  auto policy = request;
  policy.references[0].policy[1].cuda_graphs = true;
  try {
    PlanSingleDeviceStereoAdmission(policy);
    assert(false);
  } catch (const StereoAdmissionRefusal& error) {
    assert(error.resource == StereoAdmissionResource::kGraphConflict);
    assert(error.geometry && error.reference == 0);
  }
  std::rotate(policy.references.begin(),
              policy.references.begin() + 1,
              policy.references.end());
  try {
    PlanSingleDeviceStereoAdmission(policy);
    assert(false);
  } catch (const StereoAdmissionRefusal& error) {
    assert(error.resource == StereoAdmissionResource::kGraphConflict);
    assert(error.geometry && error.reference == 0);
  }
  std::rotate(policy.references.begin(),
              policy.references.end() - 1,
              policy.references.end());
  policy.capacity.device = 2000000000;
  assert(
      PlanSingleDeviceStereoAdmission(policy).records[1][0].policy.cuda_graphs);
  policy = request;
  policy.references[0].policy[0].async_pool = true;
  policy.references[0].policy[0].retain_mask = false;
  const auto pooled = PlanSingleDeviceStereoAdmission(policy);
  const auto& pooled_record = pooled.records[0][0];
  assert(pooled_record.policy.async_pool);
  assert(pooled_record.scratch_disk_bytes == 27033600);
  for (const auto& plane : pooled_record.rotation_planes) {
    assert(plane.role != RotationRole::kMask &&
           plane.role != RotationRole::kCost);
    if (plane.role == RotationRole::kWorkspace)
      assert(plane.pitch_ceiling == 2560);
  }
  auto bad_pitch = std::array<uint64_t, 4>{strip.staging_float_pitch + 1,
                                           strip.staging_float_pitch,
                                           strip.staging_float_pitch,
                                           strip.staging_mask_pitch};
  returned[0].pitch_bytes -= 1;
  Refuses<std::invalid_argument>([&] {
    ValidateStereoReturnedPitches(
        strip, returned, strip.initialization_prng_pitch, bad_pitch);
  });
  Refuses<std::invalid_argument>([&] {
    ValidateStereoReturnedPitches(strip,
                                  returned,
                                  strip.initialization_prng_pitch + 1,
                                  {strip.staging_float_pitch,
                                   strip.staging_float_pitch,
                                   strip.staging_float_pitch,
                                   strip.staging_mask_pitch});
  });
  invalid = request;
  invalid.active_owners = 2;
  Refuses<std::invalid_argument>(
      [&] { PlanSingleDeviceStereoAdmission(invalid); });
  invalid = request;
  invalid.references[0].sources[0].clear();
  Refuses<std::invalid_argument>(
      [&] { PlanSingleDeviceStereoAdmission(invalid); });
  invalid = request;
  invalid.references.pop_back();
  Refuses<std::invalid_argument>(
      [&] { PlanSingleDeviceStereoAdmission(invalid); });
  invalid = request;
  invalid.references.push_back(invalid.references[0]);
  Refuses<std::invalid_argument>(
      [&] { PlanSingleDeviceStereoAdmission(invalid); });
  invalid = request;
  invalid.references[0].sources[0] = {-1};
  Refuses<std::out_of_range>([&] { PlanSingleDeviceStereoAdmission(invalid); });
  invalid = request;
  invalid.strip_columns = 31;
  Refuses<std::invalid_argument>(
      [&] { PlanSingleDeviceStereoAdmission(invalid); });
  invalid = request;
  invalid.loader_scratch_host = std::numeric_limits<uint64_t>::max();
  Refuses<std::overflow_error>(
      [&] { PlanSingleDeviceStereoAdmission(invalid); });
  invalid = request;
  invalid.estimated_pitch_alignment = std::numeric_limits<uint64_t>::max();
  Refuses<std::overflow_error>(
      [&] { PlanSingleDeviceStereoAdmission(invalid); });
  auto wrong_shape = returned;
  wrong_shape[0].width *= 2;
  wrong_shape[0].element_bytes /= 2;
  Refuses<std::invalid_argument>([&] {
    ValidateStereoReturnedPitches(strip,
                                  wrong_shape,
                                  strip.initialization_prng_pitch,
                                  {strip.staging_float_pitch,
                                   strip.staging_float_pitch,
                                   strip.staging_float_pitch,
                                   strip.staging_mask_pitch});
  });
  invalid = request;
  invalid.images[0].width = 0;
  Refuses<std::invalid_argument>(
      [&] { PlanSingleDeviceStereoAdmission(invalid); });
  std::cout << "native stereo admission passed\n";
}
