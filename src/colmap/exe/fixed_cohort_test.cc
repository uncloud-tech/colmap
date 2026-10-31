// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/exe/fixed_cohort.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {
int Invoke(std::vector<std::string> args) {
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  return RunExperimentalFixedCohort(argv.size(), argv.data());
}
TEST(ExperimentalFixedCohort, WorkspacePlanPreservesPerReferenceOptionsAndOrder) {
  std::vector<mvs::owned::BankJob> photo(2), geo(2);
  for (size_t i = 0; i < 2; ++i) {
    photo[i].reference_image_idx = geo[i].reference_image_idx = i;
    photo[i].source_image_idxs = {2, 1, 2};
    geo[i].source_image_idxs = {1, 2, 1};
    photo[i].options.depth_min = geo[i].options.depth_min = 2 + i;
    photo[i].options.depth_max = geo[i].options.depth_max = 8 + i;
    photo[i].options.random_seed = geo[i].options.random_seed = 37 + i;
    photo[i].options.sigma_spatial = geo[i].options.sigma_spatial = 5;
  }
  const auto refs = ResolveFixedCohortWorkspacePlan({"a", "b", "c"}, photo, geo);
  ASSERT_EQ(refs.size(), 2);
  EXPECT_EQ(refs[0].photo_sources, (std::vector<std::string>{"c", "b", "c"}));
  EXPECT_EQ(refs[1].geo_sources, (std::vector<std::string>{"b", "c", "b"}));
  EXPECT_EQ(refs[0].photo_options.depth_min, 2);
  EXPECT_EQ(refs[1].photo_options.depth_min, 3);
  EXPECT_EQ(refs[1].geo_options.depth_max, 9);
  EXPECT_EQ(refs[1].geo_options.random_seed, 38);
  geo.pop_back();
  EXPECT_THROW(ResolveFixedCohortWorkspacePlan({"a", "b", "c"}, photo, geo),
               std::invalid_argument);
}
TEST(ExperimentalFixedCohort, Help) {
  // Linked runtime libraries may already have threads when this test starts.
  // Re-exec rather than running option parsing in a fork-only child.
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(Invoke({"experimental_fixed_cohort", "--help"}),
              ::testing::ExitedWithCode(EXIT_SUCCESS),
              "");
}
TEST(ExperimentalFixedCohort, MissingRequiredOptionsRefuses) {
  EXPECT_NE(Invoke({"experimental_fixed_cohort"}), EXIT_SUCCESS);
}
TEST(ExperimentalFixedCohort, UnknownOptionRefuses) {
  EXPECT_NE(Invoke({"experimental_fixed_cohort", "--not_an_option", "1"}),
            EXIT_SUCCESS);
}
}  // namespace
}  // namespace colmap
