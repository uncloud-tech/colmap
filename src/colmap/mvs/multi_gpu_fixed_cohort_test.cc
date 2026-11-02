// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/multi_gpu_fixed_cohort.h"

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {
TEST(MultiGpuFixedCohortPlan, HostExchangeIncludesBothPhasesAndOrderedFloatStaging) {
  const std::vector<FixedCohortImageShape> shapes{{5, 4}, {3, 3}};
  std::vector<FixedCohortReference> refs(2);
  for (int i = 0; i < 2; ++i) {
    refs[i].reference_image_idx = i;
    refs[i].photo_sources = {1 - i, 1 - i};
    refs[i].geo_sources = refs[i].photo_sources;
  }
  FixedCohortBudget declarations;
  declarations.output_policy = FixedCohortOutputPolicy::kFinalGeoOnly;
  declarations.process = {1, 1};
  declarations.unknown = {1, 1};
  declarations.unknown_provenance = "test-only declared unknown allowance";
  PhasePlanDevice first, second;
  first.ordinal = 0;
  first.uuid = "first";
  first.capacity_bytes = 1ull << 20;
  second = first;
  second.ordinal = 1;
  second.uuid = "second";
  const auto plan = PlanMultiGpuFixedCohort(shapes, refs, declarations,
                                           {first, second}, 1ull << 20, 1);
  uint64_t device_host = 0;
  for (const auto& group : plan.phases[1].groups) {
    device_host += group.peak_raw.host;
    ASSERT_EQ(group.cohorts.size(), 1);
    const auto& cohort = group.cohorts.front();
    ASSERT_EQ(cohort.jobs.size(), 1);
    // Job0: 24*20 + 18 gray packing + 4*18 float depth staging.
    // Job1: 24*9 + 40 gray packing + 4*40 float depth staging.
    EXPECT_EQ(cohort.private_peak_raw.host, cohort.jobs[0] == 0 ? 570 : 416);
    EXPECT_EQ(cohort.bank_raw.host, 0);
    EXPECT_EQ(cohort.bank_raw.device, 0);
  }
  // 2*(20+9) caller/frozen + 32*(20+9) PHOTO exchange AND GEO outputs.
  EXPECT_EQ(plan.phases[1].host_peak_raw - device_host, 986);
  EXPECT_EQ(plan.loader_host_raw, 62);  // 58 cache/prepared pixels + both device allowances.
  const auto loader_dominated = PlanMultiGpuFixedCohort(shapes, refs, declarations,
      {first, second}, 1ull << 20, 1, 10000);
  EXPECT_EQ(loader_dominated.loader_host_raw, 10062);
  EXPECT_EQ(loader_dominated.host_peak_raw, 10062);  // Sequential peak MAX, not phase SUM.
  EXPECT_LT(plan.phases[0].groups[0].device.host_share_raw, 1ull << 19);
  EXPECT_THROW(PlanMultiGpuFixedCohort(shapes, refs, declarations,
                                      {first, second}, 1000, 1), std::exception);
}
}  // namespace
}  // namespace colmap::mvs::owned
