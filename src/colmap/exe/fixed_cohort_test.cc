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
TEST(ExperimentalFixedCohort, Help) {
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
