// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/math/random.h"
#include "colmap/mvs/generated/resident_allocation_fingerprint.h"
#include "colmap/mvs/owned_strip_test_identity.h"
#include "colmap/util/version.h"

#include <iostream>

#include <glog/logging.h>
#include <gmock/gmock.h>

namespace {
// Keep ordinary test initialization and seed semantics identical to
// util/gtest_main.cc. Only this executable has the metadata entry point.
constexpr unsigned kDefaultTestPRNGSeed = 0;
class PRNGTestEventListener : public testing::EmptyTestEventListener {
 public:
  void OnTestStart(const testing::TestInfo&) override {
    colmap::SetPRNGSeed(kDefaultTestPRNGSeed);
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (const auto result = colmap::mvs::owned::StripTestIdentity(
          argc,
          argv,
          colmap::GetCommitRevision(),
          colmap::mvs::kResidentAllocationFingerprint,
          std::cout,
          std::cerr))
    return *result;
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleMock(&argc, argv);
  // Unknown GTest/GMock flags must not silently start scientific work either.
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument == "--help" || argument == "-h" || argument == "-?") continue;
    std::cerr << "unexpected strip harness test argument\n";
    return 1;
  }
  testing::UnitTest::GetInstance()->listeners().Append(
      new PRNGTestEventListener);
  return RUN_ALL_TESTS();
}
