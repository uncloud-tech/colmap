// SPDX-License-Identifier: BSD-3-Clause
// CPU-only CLI refusal tests for the opt-in bounded COHORT resident
// execution mode. Every path exercises the REAL standard CLI parser
// surface of RunPatchMatchStereoWithExecution: the raw-argv fail-closed
// guards and exclusivity checks run before any stock dispatch, workspace
// read or CUDA availability, so these tests are deterministic on
// CPU-only builds. The actual cohort EXECUTION path (wire gates, bank
// construction, owners) is covered by the CUDA-only surfaces.
#include "colmap/exe/mvs.h"
#include "colmap/util/testing.h"

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

namespace colmap {
namespace {

std::string ReadFileText(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

// Runs the real CLI dispatch with stderr captured to a file so the tests
// assert the EXACT guard reason (not just any failure): the fail-closed
// raw-argv/mutual-exclusivity checks run BEFORE the workspace, wire or
// CUDA availability can mask them.
int RunArgsExpectingError(const std::vector<std::string>& args,
                          std::string* stderr_log) {
  const auto log = CreateTestDir() / "cli-stderr.log";
  std::fflush(nullptr);
  const int saved = ::dup(::fileno(stderr));
  if (saved < 0) {
    throw std::runtime_error("stderr dup failed");
  }
  const int fd = ::open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    throw std::runtime_error("stderr capture open failed");
  }
  if (::dup2(fd, ::fileno(stderr)) != ::fileno(stderr)) {
    throw std::runtime_error("stderr redirect failed");
  }
  const int exit_code = ([&] {
    std::vector<char*> argv;
    for (auto& arg : args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    return RunPatchMatchStereoWithExecution(static_cast<int>(argv.size() - 1),
                                            argv.data(),
                                            mvs::PatchMatchExecution::kStock);
  })();
  std::fflush(nullptr);
  std::fflush(stderr);
  if (::dup2(saved, ::fileno(stderr)) != ::fileno(stderr)) {
    throw std::runtime_error("stderr restore failed");
  }
  ::close(saved);
  ::close(fd);
  if (stderr_log) *stderr_log = ReadFileText(log);
  return exit_code;
}

class ResidentCliCohortTest : public ::testing::Test {
 protected:
  void SetUp() override {
    workspace_ = CreateTestDir();
    forecast_ = CreateTestDir() / "stub.cohort-wire";
    std::ofstream file(forecast_);
    file << "placeholder\n";
  }

  std::vector<std::string> BaseArgs() const {
    return {"colmap",
            "--workspace_path",
            workspace_.string(),
            "--PatchMatchStereo.gpu_index",
            "0"};
  }

  std::filesystem::path workspace_;
  std::filesystem::path forecast_;
};

// Missing caps with the cohort forecast flag refuse before any stock
// dispatch; the asserted FAILURE REASON is the exact guard text, not a
// dummy downstream failure.
TEST_F(ResidentCliCohortTest, MissingCapsRefuse) {
  auto args = BaseArgs();
  args.push_back("--owned_run_cohort_forecast");
  args.push_back(forecast_.string());
  args.push_back("--owned_run_host_budget_bytes");
  args.push_back("1000");
  std::string log;
  EXPECT_EQ(RunArgsExpectingError(args, &log), EXIT_FAILURE);
  EXPECT_NE(log.find("Cohort resident execution requires "
                     "owned_run_host_budget_bytes"),
            std::string::npos)
      << log;
}

// A nonexistent cohort forecast file refuses (regular-file check); the
// reason is the file check, not the dense-library availability.
TEST_F(ResidentCliCohortTest, MissingForecastFileRefuses) {
  auto args = BaseArgs();
  args.push_back("--owned_run_cohort_forecast");
  args.push_back((forecast_.parent_path() / "absent.cohort-wire").string());
  args.push_back("--owned_run_host_budget_bytes");
  args.push_back("1000");
  args.push_back("--owned_run_device_budget_bytes");
  args.push_back("1000");
  args.push_back("--owned_run_max_concurrent_solvers");
  args.push_back("1");
  std::string log;
  EXPECT_EQ(RunArgsExpectingError(args, &log), EXIT_FAILURE);
  EXPECT_NE(log.find("not a regular file"), std::string::npos) << log;
}

// The two resident run modes are mutually exclusive (distinct wires,
// distinct magics): with everything else VALID (regular file, all caps)
// the asserted reason is the exclusivity guard itself, which runs before
// any workspace/wire/CUDA work can mask it.
TEST_F(ResidentCliCohortTest, FullBankAndCohortModesMutuallyExclusiveRefuse) {
  auto args = BaseArgs();
  args.push_back("--owned_run_forecast");
  args.push_back(forecast_.string());
  args.push_back("--owned_run_cohort_forecast");
  args.push_back(forecast_.string());
  args.push_back("--owned_run_host_budget_bytes");
  args.push_back("1000");
  args.push_back("--owned_run_device_budget_bytes");
  args.push_back("1000");
  args.push_back("--owned_run_max_concurrent_solvers");
  args.push_back("1");
  std::string log;
  EXPECT_EQ(RunArgsExpectingError(args, &log), EXIT_FAILURE);
  EXPECT_NE(log.find("mutually exclusive (distinct wires"), std::string::npos)
      << log;
}

// Raw-argv fail-closed: cohort flags combined with --project_path refuse
// with the post-parse guard reason. The project file is a VALID INI
// (sufficient required options parsed) so that options.Parse SUCCEEDS and
// the asserted reason is exactly the intended project_path guard — never
// a parse failure of a directory or other dummy-input failure.
TEST_F(ResidentCliCohortTest, ProjectPathCombinedRefuses) {
  const auto project_file = forecast_.parent_path() / "valid.ini";
  {
    std::ofstream ini(project_file);
    // Sufficient required options for a clean Parse: the required
    // workspace_path (boost program_options config syntax).
    ini << "workspace_path=" << workspace_.string() << "\n";
  }
  ASSERT_TRUE(std::filesystem::is_regular_file(project_file));
  auto args = BaseArgs();
  args.push_back("--project_path");
  args.push_back(project_file.string());
  args.push_back("--owned_run_cohort_forecast");
  args.push_back(forecast_.string());
  std::string log;
  EXPECT_EQ(RunArgsExpectingError(args, &log), EXIT_FAILURE);
  EXPECT_NE(log.find("--project_path combined with owned bank metadata"),
            std::string::npos)
      << log;
}

// A cohort/runner flag without the forecast (e.g. parser-discarded) must
// never silently fall through to the stock solver; the reason names the
// discarded mode request.
TEST_F(ResidentCliCohortTest, DiscardedRunnerFlagsRefuse) {
  auto args = BaseArgs();
  args.push_back("--owned_run_max_concurrent_solvers");
  args.push_back("1");
  std::string log;
  EXPECT_EQ(RunArgsExpectingError(args, &log), EXIT_FAILURE);
  EXPECT_NE(log.find("was requested on the command line "
                     "but was not applied"),
            std::string::npos)
      << log;
}

// Metadata and BOTH execution modes are mutually exclusive: with a VALID
// regular-file cohort wire and complete caps the asserted reason is the
// exclusivity guard (previously the guard was missing entirely for the
// cohort mode and the failure reason was a masked downstream failure).
TEST_F(ResidentCliCohortTest, MetadataModeMutuallyExclusiveRefuses) {
  for (const char* flag :
       {"--owned_run_cohort_forecast", "--owned_run_forecast"}) {
    auto args = BaseArgs();
    args.push_back("--owned_metadata_export");
    args.push_back((forecast_.parent_path() / "exported.contract").string());
    args.push_back(flag);
    args.push_back(forecast_.string());
    args.push_back("--owned_run_host_budget_bytes");
    args.push_back("1000");
    args.push_back("--owned_run_device_budget_bytes");
    args.push_back("1000");
    args.push_back("--owned_run_max_concurrent_solvers");
    args.push_back("1");
    std::string log;
    EXPECT_EQ(RunArgsExpectingError(args, &log), EXIT_FAILURE) << flag;
    EXPECT_NE(log.find("mutually exclusive; pass exactly one"),
              std::string::npos)
        << flag << ": " << log;
  }
}

// Negative and malformed caps refuse before the workspace is touched; the
// reason is the canonical unsigned parser.
TEST_F(ResidentCliCohortTest, MalformedCapsRefuse) {
  for (const char* value : {"abc", "-1", "0x10", "1.5"}) {
    auto args = BaseArgs();
    args.push_back("--owned_run_cohort_forecast");
    args.push_back(forecast_.string());
    args.push_back("--owned_run_host_budget_bytes");
    args.push_back(value);
    args.push_back("--owned_run_device_budget_bytes");
    args.push_back("1000");
    args.push_back("--owned_run_max_concurrent_solvers");
    args.push_back("1");
    std::string log;
    EXPECT_EQ(RunArgsExpectingError(args, &log), EXIT_FAILURE) << value;
    EXPECT_NE(log.find("canonical unsigned decimal literal"), std::string::npos)
        << value << ": " << log;
  }
}

}  // namespace
}  // namespace colmap