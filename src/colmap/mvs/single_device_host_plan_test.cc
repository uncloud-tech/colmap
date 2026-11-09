// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/exe/multi_gpu_plan_json.h"
#include "colmap/mvs/multi_gpu_fixed_cohort.h"

#include <cassert>
#include <iostream>

using namespace colmap::mvs::owned;
int main() {
  const std::vector<FixedCohortImageShape> shapes{
      {640, 480}, {641, 481}, {639, 479}};
  std::vector<FixedCohortReference> refs(3);
  for (int i = 0; i < 3; ++i) {
    refs[i].reference_image_idx = i;
    refs[i].photo_options.gpu_index = "0";
    refs[i].geo_options.gpu_index = "0";
    refs[i].photo_sources =
        i == 0 ? std::vector<int>{2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2}
               : std::vector<int>{0};
    refs[i].geo_sources = refs[i].photo_sources;
  }
  FixedCohortBudget declarations;
  declarations.process = {1000, 2000};
  declarations.unknown = {3000, 4000};
  declarations.unknown_provenance = "test declared allowance";
  declarations.headroom_reserve = {5000, 6000};
  PhasePlanDevice device;
  device.ordinal = 0;
  device.uuid = "approved-device";
  device.capacity_bytes = 60000000;
  bool refused = false;
  try {
    PlanMultiGpuFixedCohort(
        shapes, refs, declarations, {device}, 2000000000, 1);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  assert(refused);
  const auto plan = PlanSingleDeviceHostFixedCohort(shapes,
                                                    refs,
                                                    declarations,
                                                    device,
                                                    2000000000,
                                                    2000000000,
                                                    "/scratch",
                                                    "auto",
                                                    0,
                                                    32);
  assert(plan.single_device_host);
  for (size_t phase = 0; phase < 2; ++phase) {
    assert(plan.phases[phase].groups.size() == 1);
    const auto& group = plan.phases[phase].groups[0];
    assert(group.cohorts.size() == 3);
    for (size_t job = 0; job < 3; ++job) {
      assert(group.cohorts[job].jobs == std::vector<size_t>{job});
      assert(group.cohorts[job].active_owners == 1);
      assert(group.cohorts[job].bank_raw.host == 0);
      assert(group.cohorts[job].bank_raw.device == 0);
    }
    const auto& record = plan.single_device_host->admission.records[phase][0];
    assert(record.mode == StereoVolumeMode::kStripFile);
    assert(record.columns > 32 && record.columns % 32 == 0);
    assert(record.sources == refs[0].photo_sources);
    assert(group.cohorts[0].private_peak_raw.device == record.peak.device);
  }
  const auto json =
      colmap::dense_plan::SingleDeviceHostJson(*plan.single_device_host).Dump();
  assert(json.find("\"effective_photo_source_storage\":\"private\"") !=
         std::string::npos);
  assert(json.find("\"requested_photo_source_storage\":\"auto\"") !=
         std::string::npos);
  assert(json.find("\"volume_mode\":\"strip_file\"") != std::string::npos);
  assert(json.find("\"requested_strip_columns\":\"0\"") != std::string::npos);
  assert(json.find("\"cache_host_bytes\":\"30412800\"") != std::string::npos);
  auto changed = *plan.single_device_host;
  changed.admission.records[0][0].policy.async_pool = true;
  assert(colmap::dense_plan::SingleDeviceHostJson(changed).Dump() != json);
  auto roomy = device;
  roomy.capacity_bytes = 2000000000;
  const auto preferred = PlanSingleDeviceStereoFixedCohort(shapes,
                                                           refs,
                                                           declarations,
                                                           roomy,
                                                           2000000000,
                                                           2000000000,
                                                           "/scratch",
                                                           "auto",
                                                           1);
  assert(preferred.snapshot_budget && preferred.snapshot_inventory);
  auto wrong_device = roomy;
  wrong_device.ordinal = 1;
  bool identity_refused = false;
  try {
    PlanSingleDeviceStereoFixedCohort(shapes,
                                      refs,
                                      declarations,
                                      wrong_device,
                                      2000000000,
                                      2000000000,
                                      "/scratch",
                                      "auto",
                                      1);
  } catch (const std::invalid_argument&) {
    identity_refused = true;
  }
  assert(identity_refused);
  assert(!preferred.single_device_host);
  const auto fallback = PlanSingleDeviceStereoFixedCohort(shapes,
                                                          refs,
                                                          declarations,
                                                          device,
                                                          2000000000,
                                                          2000000000,
                                                          "/scratch",
                                                          "auto",
                                                          1);
  assert(fallback.single_device_host && !fallback.snapshot_budget);
  for (const auto policy : {FixedCohortOutputPolicy::kFinalGeoOnly,
                            FixedCohortOutputPolicy::kPhotoAndGeo}) {
    declarations.output_policy = policy;
    const std::string name = policy == FixedCohortOutputPolicy::kFinalGeoOnly
                                 ? "final_geo_only"
                                 : "photo_and_geo";
    for (const auto& selected : {PlanSingleDeviceStereoFixedCohort(shapes,
                                                                   refs,
                                                                   declarations,
                                                                   roomy,
                                                                   2000000000,
                                                                   2000000000,
                                                                   "/scratch",
                                                                   "auto",
                                                                   1),
                                 PlanSingleDeviceStereoFixedCohort(shapes,
                                                                   refs,
                                                                   declarations,
                                                                   device,
                                                                   2000000000,
                                                                   2000000000,
                                                                   "/scratch",
                                                                   "auto",
                                                                   1)}) {
      assert(selected.output_policy == policy);
      const auto& metadata = selected.snapshot_inventory
                                 ? *selected.snapshot_inventory
                                 : *selected.single_device_host;
      assert(metadata.admission.output_policy == policy);
      const auto bound_json =
          colmap::dense_plan::SingleDeviceHostJson(metadata).Dump();
      assert(bound_json.find("\"output_policy\":\"" + name + "\"") !=
             std::string::npos);
      assert(bound_json.find("\"peak_host_raw\":\"" +
                             std::to_string(metadata.admission.peak_raw.host) +
                             "\"") != std::string::npos);
      assert(
          bound_json.find("\"peak_device_raw\":\"" +
                          std::to_string(metadata.admission.peak_raw.device) +
                          "\"") != std::string::npos);
      auto altered = metadata;
      altered.admission.output_policy =
          policy == FixedCohortOutputPolicy::kFinalGeoOnly
              ? FixedCohortOutputPolicy::kPhotoAndGeo
              : FixedCohortOutputPolicy::kFinalGeoOnly;
      assert(!SameStereoAdmissionPlan(metadata.admission, altered.admission));
      assert(colmap::dense_plan::SingleDeviceHostJson(altered).Dump() !=
             bound_json);
      altered = metadata;
      ++altered.admission.peak_raw.device;
      assert(!SameStereoAdmissionPlan(metadata.admission, altered.admission));
      assert(colmap::dense_plan::SingleDeviceHostJson(altered).Dump() !=
             bound_json);
    }
    auto second = roomy;
    second.ordinal = 1;
    second.uuid = "second-approved-device";
    const auto multi = PlanMultiGpuFixedCohort(
        shapes, refs, declarations, {roomy, second}, 2000000000, 1);
    assert(multi.output_policy == policy);
  }
  std::cout << "single-device HOST native adapter passed\n";
}
