// SPDX-License-Identifier: BSD-3-Clause
// CPU-only CLI refusal tests for the opt-in bounded resident execution
// backend. Every path exercises the REAL standard CLI parser surface of
// RunPatchMatchStereoWithExecution: raw-argv fail-closed guards run before
// any stock dispatch, workspace read or CUDA availability, so these tests
// are deterministic on CPU-only builds. Registered with the normal CTest
// suite; the actual resident EXECUTION path (gate composition, bank
// construction, owners) is covered by the CUDA-only CLI/solver surfaces.
#include "colmap/exe/mvs.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

int RunArgs(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);
  return RunPatchMatchStereoWithExecution(static_cast<int>(argv.size() - 1),
                                          argv.data(),
                                          mvs::PatchMatchExecution::kStock);
}

class ResidentCliRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    workspace_ = CreateTestDir();
    forecast_ = CreateTestDir() / "stub.forecast";
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

#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
TEST_F(ResidentCliRunnerTest,
       OptimizedPhotoOptionsExportSucceedsAndBindsOptions) {
  const auto config = workspace_ / "photo.cfg";
  std::ofstream(config) << "nested/image.jpg\nsource.jpg\n";
  const auto export_options = [&](const char* filename, const char* samples) {
    auto args = BaseArgs();
    args.insert(args.end(),
                {"--config_path",
                 config.string(),
                 "--owned_photo_options_export",
                 (workspace_ / filename).string(),
                 "--PatchMatchStereo.geom_consistency",
                 "false",
                 "--PatchMatchStereo.num_samples",
                 samples});
    EXPECT_EQ(RunArgs(args), EXIT_SUCCESS);
    std::ifstream input(workspace_ / filename, std::ios::binary);
    EXPECT_TRUE(input.is_open());
    return std::string(std::istreambuf_iterator<char>(input), {});
  };
  const auto first = export_options("first.options", "15");
  EXPECT_EQ(first.find("COLMAP_OPTIMIZED_PHOTO_OPTIONS_V1\n"), 0);
  EXPECT_NE(first.find("opt_int num_samples 15\n"), std::string::npos);
  EXPECT_EQ(first, export_options("same.options", "15"));
  EXPECT_NE(first, export_options("changed.options", "14"));
  EXPECT_FALSE(std::filesystem::exists(workspace_ / "stereo"));
}
#endif

// Strict mode refuses before even attempting to open this empty workspace.
TEST_F(ResidentCliRunnerTest, OptimizedModeNeverFallsThroughToStock) {
  for (const auto& phase : {"true", "false"}) {
    auto args = BaseArgs();
    args.insert(args.end(),
                {"--optimized_only",
                 "true",
                 "--optimized_dispatch_receipt",
                 (workspace_ / "receipt.json").string(),
                 "--PatchMatchStereo.geom_consistency",
                 phase});
    EXPECT_EQ(RunArgs(args), EXIT_FAILURE);
    EXPECT_FALSE(std::filesystem::exists(workspace_ / "receipt.json"));
    EXPECT_FALSE(std::filesystem::exists(workspace_ / "stereo"));
  }
}

TEST_F(ResidentCliRunnerTest, OptimizedPhotoWithoutStrictModeRefuses) {
  auto args = BaseArgs();
  args.insert(args.end(), {"--owned_photo_forecast", forecast_.string()});
  EXPECT_EQ(RunArgs(args), EXIT_FAILURE);
  EXPECT_FALSE(std::filesystem::exists(workspace_ / "stereo"));
}

TEST_F(ResidentCliRunnerTest, OptimizedFusionRejectsOrdinaryBackend) {
  std::vector<std::string> args = {"colmap",
                                   "--workspace_path",
                                   workspace_.string(),
                                   "--output_path",
                                   (workspace_ / "fused.ply").string(),
                                   "--optimized_only",
                                   "true",
                                   "--optimized_dispatch_receipt",
                                   (workspace_ / "receipt.json").string(),
                                   "--StereoFusion.use_mmap",
                                   "false"};
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  EXPECT_EQ(RunStereoFuser(static_cast<int>(argv.size()), argv.data()),
            EXIT_FAILURE);
  EXPECT_FALSE(std::filesystem::exists(workspace_ / "receipt.json"));
  EXPECT_FALSE(std::filesystem::exists(workspace_ / "fused.ply"));
}

// Missing caps with the forecast flag refuse before any stock dispatch.
TEST_F(ResidentCliRunnerTest, MissingCapsRefuse) {
  auto args = BaseArgs();
  args.push_back("--owned_run_forecast");
  args.push_back(forecast_.string());
  args.push_back("--owned_run_host_budget_bytes");
  args.push_back("1000");
  EXPECT_EQ(RunArgs(args), EXIT_FAILURE);
}

// A nonexistent forecast file refuses (regular-file check) before the
// dense-library availability decision.
TEST_F(ResidentCliRunnerTest, MissingForecastFileRefuses) {
  auto args = BaseArgs();
  args.push_back("--owned_run_forecast");
  args.push_back((forecast_.parent_path() / "absent.forecast").string());
  args.push_back("--owned_run_host_budget_bytes");
  args.push_back("1000");
  args.push_back("--owned_run_device_budget_bytes");
  args.push_back("1000");
  args.push_back("--owned_run_max_concurrent_solvers");
  args.push_back("1");
  EXPECT_EQ(RunArgs(args), EXIT_FAILURE);
}

// Raw-argv fail-closed: any resident flag combined with --project_path
// refuses (the project-file reread can discard explicit mode requests).
TEST_F(ResidentCliRunnerTest, ProjectPathCombinedRefuses) {
  auto args = BaseArgs();
  args.push_back("--project_path");
  args.push_back(workspace_.string());
  args.push_back("--owned_run_forecast");
  args.push_back(forecast_.string());
  EXPECT_EQ(RunArgs(args), EXIT_FAILURE);
}

// A resident flag without the forecast (e.g. parser-discarded) must never
// silently fall through to the stock solver.
TEST_F(ResidentCliRunnerTest, DiscardedRunnerFlagsRefuse) {
  auto args = BaseArgs();
  args.push_back("--owned_run_max_concurrent_solvers");
  args.push_back("1");
  EXPECT_EQ(RunArgs(args), EXIT_FAILURE);
}

// Metadata and runner modes are mutually exclusive.
TEST_F(ResidentCliRunnerTest, MetadataModeMutuallyExclusiveRefuses) {
  auto args = BaseArgs();
  args.push_back("--owned_metadata_export");
  args.push_back((forecast_.parent_path() / "exported.contract").string());
  args.push_back("--owned_run_forecast");
  args.push_back(forecast_.string());
  args.push_back("--owned_run_host_budget_bytes");
  args.push_back("1000");
  args.push_back("--owned_run_device_budget_bytes");
  args.push_back("1000");
  args.push_back("--owned_run_max_concurrent_solvers");
  args.push_back("1");
  EXPECT_EQ(RunArgs(args), EXIT_FAILURE);
}

TEST_F(ResidentCliRunnerTest, N3N4N5MetadataModesRefuseExecutionFlags) {
  for (const char* mode :
       {"--owned_metadata_export", "--owned_metadata_verify"}) {
    for (const char* concurrency : {"3", "4", "5"}) {
      auto args = BaseArgs();
      const auto output = workspace_ / "metadata.contract";
      args.insert(args.end(),
                  {mode,
                   output.string(),
                   "--owned_run_host_budget_bytes",
                   "9999999999",
                   "--owned_run_device_budget_bytes",
                   "9999999999",
                   "--owned_run_max_concurrent_solvers",
                   concurrency});
      EXPECT_EQ(RunArgs(args), EXIT_FAILURE);
      EXPECT_FALSE(std::filesystem::exists(output));
      EXPECT_FALSE(std::filesystem::exists(workspace_ / "stereo"));
    }
  }
}

TEST_F(ResidentCliRunnerTest, ConcurrentCliRequiresCanonicalUnsigned) {
  for (const char* concurrency : {"03", "04", "05", "+3", " 4", "4 ", "3x"}) {
    auto args = BaseArgs();
    args.insert(args.end(),
                {"--owned_run_forecast",
                 forecast_.string(),
                 "--owned_run_host_budget_bytes",
                 "9999999999",
                 "--owned_run_device_budget_bytes",
                 "9999999999",
                 "--owned_run_max_concurrent_solvers",
                 concurrency});
    ::testing::internal::CaptureStderr();
    const int result = RunArgs(args);
    const auto diagnostic = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, EXIT_FAILURE) << concurrency;
    EXPECT_NE(diagnostic.find("canonical unsigned decimal literal"),
              std::string::npos)
        << diagnostic;
    EXPECT_FALSE(std::filesystem::exists(workspace_ / "stereo"));
  }
}

// With a CUDA-enabled binary this stops at the real forecast-file check,
// after optimized capability/concurrency policy and before workspace/GPU work.
// CPU-only binaries must instead report the capability refusal honestly.
TEST_F(ResidentCliRunnerTest, OptimizedN3N4N5ReachForecastBoundaryOrRefuseCpu) {
  for (const char* concurrency : {"3", "4", "5"}) {
    SCOPED_TRACE(concurrency);
    auto args = BaseArgs();
    args.insert(args.end(),
                {"--optimized_only",
                 "true",
                 "--optimized_dispatch_receipt",
                 (workspace_ / "receipt.json").string(),
                 "--owned_run_timeout_seconds",
                 "30",
                 "--PatchMatchStereo.num_threads",
                 "1",
                 "--PatchMatchStereo.owned_distinct_source_cache",
                 "true",
                 "--owned_run_forecast",
                 (workspace_ / "absent.forecast").string(),
                 "--owned_run_host_budget_bytes",
                 "9999999999",
                 "--owned_run_device_budget_bytes",
                 "9999999999",
                 "--owned_run_max_concurrent_solvers",
                 concurrency});
    ::testing::internal::CaptureStderr();
    const int result = RunArgs(args);
    const auto diagnostic = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, EXIT_FAILURE);
#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
    EXPECT_NE(diagnostic.find(
                  "resident forecast file is missing or not a regular file"),
              std::string::npos)
        << diagnostic;
#else
    EXPECT_NE(diagnostic.find("optimized_only requires CUDA owned"),
              std::string::npos)
        << diagnostic;
    EXPECT_EQ(diagnostic.find("resident forecast file is missing"),
              std::string::npos)
        << diagnostic;
#endif
    EXPECT_FALSE(std::filesystem::exists(workspace_ / "receipt.json"));
    EXPECT_FALSE(std::filesystem::exists(workspace_ / "stereo"));
  }
}

#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
TEST_F(ResidentCliRunnerTest, OptimizedPhotoCohortAndInvalidNRefusePolicy) {
  for (const char* phase : {"photo", "cohort", "geo"}) {
    const bool photo = std::string(phase) == "photo";
    const bool cohort = std::string(phase) == "cohort";
    const std::vector<std::string> values =
        photo || cohort ? std::vector<std::string>{"3", "4", "5"}
                        : std::vector<std::string>{"0", "6", "03", "05", "+3"};
    for (const auto& concurrency : values) {
      SCOPED_TRACE(std::string(phase) + concurrency);
      auto args = BaseArgs();
      args.insert(args.end(),
                  {"--optimized_only",
                   "true",
                   "--optimized_dispatch_receipt",
                   (workspace_ / "receipt.json").string(),
                   "--owned_run_timeout_seconds",
                   "30",
                   "--PatchMatchStereo.num_threads",
                   "1",
                   "--PatchMatchStereo.owned_distinct_source_cache",
                   "true",
                   "--PatchMatchStereo.geom_consistency",
                   photo ? "false" : "true",
                   photo    ? "--owned_photo_forecast"
                   : cohort ? "--owned_run_cohort_forecast"
                            : "--owned_run_forecast",
                   forecast_.string(),
                   "--owned_run_host_budget_bytes",
                   "9999999999",
                   "--owned_run_device_budget_bytes",
                   "9999999999",
                   "--owned_run_max_concurrent_solvers",
                   concurrency});
      ::testing::internal::CaptureStderr();
      const int result = RunArgs(args);
      const auto diagnostic = ::testing::internal::GetCapturedStderr();
      EXPECT_EQ(result, EXIT_FAILURE);
      EXPECT_NE(diagnostic.find("optimized_only requires PHOTO concurrency1"),
                std::string::npos)
          << diagnostic;
      EXPECT_FALSE(std::filesystem::exists(workspace_ / "receipt.json"));
      EXPECT_FALSE(std::filesystem::exists(workspace_ / "stereo"));
    }
  }
}
#endif

// Negative and malformed caps refuse before the workspace is touched.
TEST_F(ResidentCliRunnerTest, MalformedCapsRefuse) {
  for (const char* value : {"abc", "-1", "0x10", "1.5"}) {
    auto args = BaseArgs();
    args.push_back("--owned_run_forecast");
    args.push_back(forecast_.string());
    args.push_back("--owned_run_host_budget_bytes");
    args.push_back(value);
    args.push_back("--owned_run_device_budget_bytes");
    args.push_back("1000");
    args.push_back("--owned_run_max_concurrent_solvers");
    args.push_back("1");
    EXPECT_EQ(RunArgs(args), EXIT_FAILURE) << value;
  }
}

}  // namespace
}  // namespace colmap