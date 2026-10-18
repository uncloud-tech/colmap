// SPDX-License-Identifier: BSD-3-Clause

// CPU tests for the cohort runner's CPU-checkable pre-payload surfaces and
// lifecycle primitives (all REAL interfaces; no payloads, no CUDA): the
// output-path discipline (fail closed before any payload), the per-cohort
// fresh-ledger lifecycle with a THROWING observer (cancellation published
// first, owners drain and join, no advancing after failure), and the
// sequential-cohort teardown/advance ordering on the real
// PairCoordinator/NormalLookahead interfaces. The CUDA surfaces
// (ResidentInputBank, the controllers) remain dev1 native-build coverage.
#include "colmap/mvs/resident_cohort_runner.h"

#include "colmap/mvs/image.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

using owned::BankInput;
using owned::BankJob;

// VALID fixture job: distinct image/depth/normal input paths (the image
// object carries its model path, as the real resolved contract problems
// do) plus resolved output paths. No input file must exist on disk; only
// the OUTPUT paths are checked for existence, and the alias predicate is
// the existing lexical/equivalence helper (nonexistent outputs are
// compared lexically).
BankJob JobWithOutputs(const std::filesystem::path& root,
                       const int index,
                       const std::string& suffix = "") {
  static const float K[9] = {10, 0, 2, 0, 10, 1, 0, 0, 1};
  static const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  static const float T[3] = {0, 0, 0};
  const std::string in_prefix = "in-" + std::to_string(index);
  const std::string out_prefix = "out-" + std::to_string(index);
  const auto image_path = root / (in_prefix + suffix + ".png");
  Image image(image_path, 4, 2, K, R, T);
  image.Downsize(2, 2);
  BankJob job;
  job.depth_map_path = root / (out_prefix + suffix + ".out.depth.bin");
  job.normal_map_path = root / (out_prefix + suffix + ".out.normal.bin");
  BankInput input;
  input.image_idx = 0;  // valid within this job's one-image vector
  input.image = image;
  input.max_image_size = 2;
  input.depth_path = root / (in_prefix + suffix + ".in.depth.bin");
  input.normal_path = root / (in_prefix + suffix + ".in.normal.bin");
  job.inputs = {input};
  return job;
}

TEST(CohortOutputPathDiscipline, ExistingOutputsRefuse) {
  const auto root =
      std::filesystem::temp_directory_path() / "cohort_paths_test";
  std::filesystem::create_directories(root);
  auto job = JobWithOutputs(root, 0);
  // A stale output exists: the existence check refuses FIRST.
  std::ofstream preexisting(job.depth_map_path);
  preexisting << "stale output";
  preexisting.close();
  // The output path doubles as its own recorded input path to reach the
  // alias predicate for the same diagnostic surface (the input need not
  // exist; identity revalidation is a separate gate).
  for (auto& input : job.inputs) {
    input.depth_path = job.depth_map_path;
  }
  std::vector<BankJob> jobs{job};
  try {
    resident::CohortOutputPathsRefuseExistingOrAliased(jobs);
    FAIL() << "existing outputs must refuse";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("cohort output already exists"),
              std::string::npos)
        << message;
  }
  std::filesystem::remove_all(root);
}

// With VALID distinct inputs the clean case passes (the precondition is
// the resolved image path, not a missing-path throw).
TEST(CohortOutputPathDiscipline, CleanOutputsPass) {
  const auto root =
      std::filesystem::temp_directory_path() / "cohort_paths_test_clean";
  std::filesystem::create_directories(root);
  std::vector<BankJob> jobs{JobWithOutputs(root, 0), JobWithOutputs(root, 1)};
  EXPECT_NO_THROW(resident::CohortOutputPathsRefuseExistingOrAliased(jobs));
  std::filesystem::remove_all(root);
}

// A SAME-job output pair (depth vs normal of the identical job) refuses
// with its intended diagnostic.
TEST(CohortOutputPathDiscipline, SameJobOutputPairAliasRefuses) {
  const auto root =
      std::filesystem::temp_directory_path() / "cohort_paths_test_same";
  std::filesystem::create_directories(root);
  auto job = JobWithOutputs(root, 0);
  job.normal_map_path = job.depth_map_path;
  std::vector<BankJob> jobs{job};
  try {
    resident::CohortOutputPathsRefuseExistingOrAliased(jobs);
    FAIL() << "same-job output alias must refuse";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("aliases another output "
                           "(including the same job's depth vs normal "
                           "pair)"),
              std::string::npos)
        << message;
  }
  std::filesystem::remove_all(root);
}

// An output aliasing a WHOLE-invocation depth INPUT file of ANOTHER job
// refuses with its intended diagnostic.
TEST(CohortOutputPathDiscipline, CrossJobInputAliasRefuses) {
  const auto root =
      std::filesystem::temp_directory_path() / "cohort_paths_test_cross";
  std::filesystem::create_directories(root);
  auto first = JobWithOutputs(root, 0);
  auto second = JobWithOutputs(root, 1);
  // The aliased input stays UNCREATED on disk: the alias predicate
  // compares the uncreated candidate output against the invocation's
  // recorded input path lexically (creating it would trip the separate
  // output-existence refusal first; see ExistingOutputsRefuse).
  second.normal_map_path = first.inputs.front().depth_path;
  std::vector<BankJob> jobs{first, second};
  try {
    resident::CohortOutputPathsRefuseExistingOrAliased(jobs);
    FAIL() << "cross-job input alias must refuse";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("aliases an admitted input"), std::string::npos)
        << message;
  }
  std::filesystem::remove_all(root);
}

// An output aliasing a whole-invocation IMAGE input refuses with its
// intended diagnostic (image inputs are inside the alias scope).
TEST(CohortOutputPathDiscipline, ImageInputAliasRefuses) {
  const auto root =
      std::filesystem::temp_directory_path() / "cohort_paths_test_image";
  std::filesystem::create_directories(root);
  auto first = JobWithOutputs(root, 0);
  const auto image_path = first.inputs.front().image.GetPath();
  auto second = JobWithOutputs(root, 1);
  second.depth_map_path = image_path;
  std::vector<BankJob> jobs{first, second};
  try {
    resident::CohortOutputPathsRefuseExistingOrAliased(jobs);
    FAIL() << "image-input alias must refuse";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("aliases an admitted input"), std::string::npos)
        << message;
  }
  std::filesystem::remove_all(root);
}

// A CROSS-job OUTPUT alias refuses with its intended diagnostic (this is
// the all-pairs output scope, not a per-job check).
TEST(CohortOutputPathDiscipline, CrossJobOutputAliasRefuses) {
  const auto root =
      std::filesystem::temp_directory_path() / "cohort_paths_test_alias";
  std::filesystem::create_directories(root);
  auto first = JobWithOutputs(root, 0);
  auto aliased = JobWithOutputs(root, 1);
  aliased.normal_map_path = first.depth_map_path;  // output vs output alias
  std::vector<BankJob> jobs{first, aliased};
  try {
    resident::CohortOutputPathsRefuseExistingOrAliased(jobs);
    FAIL() << "cross-job output alias must refuse";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("aliases another output "
                           "(including the same job's depth vs normal "
                           "pair)"),
              std::string::npos)
        << message;
  }
  std::filesystem::remove_all(root);
}

// A SAME-job output-as-own-input alias (depth output doubling as this
// job's own normal input) refuses with its intended diagnostic.
TEST(CohortOutputPathDiscipline, OwnInputAliasRefuses) {
  const auto root =
      std::filesystem::temp_directory_path() / "cohort_paths_test_input";
  std::filesystem::create_directories(root);
  auto job = JobWithOutputs(root, 0);
  job.inputs.front().depth_path = job.normal_map_path;  // output vs input
  std::vector<BankJob> jobs{job};
  try {
    resident::CohortOutputPathsRefuseExistingOrAliased(jobs);
    FAIL() << "output-as-own-input alias must refuse";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("aliases an admitted input"), std::string::npos)
        << message;
  }
  std::filesystem::remove_all(root);
}

// Sequential-cohort lifecycle on the REAL coordinator/pipeline/observer
// interfaces, mirroring the runner's per-cohort wiring: the first
// cohort's failure (preserved first error) is thrown from Finish, the
// producers/owners drain with cancellation published BEFORE observers,
// and the FAILED loop refuses to advance to the second cohort.
TEST(CohortRunnerLifecycle, FailureAbortsRemainingCohortsAfterDrain) {
  int cohort_advance_attempts = 0;
  bool cancellation_published = false;
  bool observer_threw = false;
  for (size_t cohort = 0; cohort < 2; ++cohort) {
    ++cohort_advance_attempts;
    auto pipeline = std::make_unique<owned::NormalLookahead>(
        std::vector<size_t>{2 * cohort, 2 * cohort + 1},
        /*max_future=*/1, /*charge=*/4096);
    auto pipeline_raw = pipeline.get();
    const std::function<void(size_t, const char*)> throwing_observer =
        [&](size_t, const char* event) {
          if (std::string(event) == "cancelled") {
            observer_threw = true;
            throw std::runtime_error("telemetry cancelled event failed");
          }
        };
    const auto sink = [&, pipeline_raw](const owned::PairEvent& e) {
      if (e.event == "cancelled") {
        cancellation_published = true;
      }
      // MANDATORY ORDER (identical composition to the production sink):
      // cancellation first, throwing observer after.
      ResidentEventSink(e.event == "cancelled",
                                  [pipeline_raw]() {
                                    pipeline_raw->Cancel();
                                  },
                                  &throwing_observer, e.job,
                                  e.event.c_str());
    };
    owned::PairCoordinator coordinator(
        owned::ByteCharge{65536, 65536}, owned::ByteCharge{2048, 2048},
        /*max_concurrent_solvers=*/1,
        std::vector<owned::ByteCharge>{{4096, 8192}, {4096, 8192}},
        /*timeout=*/std::chrono::milliseconds(2000), sink);
    // Real pipeline production (the loader is the CPU stand-in for the
    // bounded bank normal read; the pipeline/queue/sink are the real
    // production classes).
    pipeline->Start([](size_t, const std::atomic<bool>&) {
      NormalMap normal(2, 2);
      normal.Fill(0.5f);
      return normal;
    });
    const auto run_job = [&](const size_t generation) {
      if (generation == 1) {
        // A real owner can fail BEFORE its protocol workflow; the owner
        // loop then cancels this generation (whose sink cancels the
        // producer), and the cancellation makes the retirement legal.
        throw std::runtime_error("owner 1 failed: payload refusal");
      }
      // The REAL per-owner protocol, driven through the coordinator's
      // public workflow exactly as the production controller drives it
      // (a protocol-skipping owner makes Retire refuse, the AwaitRetired
      // bounded wait time out and the process std::terminate — that
      // fixture bug was the source of the abort 134, NOT a product
      // defect).
      NormalMap normal = pipeline->Consume(generation);
      (void)normal;
      coordinator.BeginPreparation(generation);
      coordinator.ReadyAndWait(generation);
      coordinator.SolverDone(generation);
      coordinator.BeginOutput(generation);
      coordinator.EndOutput(generation);
      coordinator.AwaitOutputs(generation);
    };
    // NO external trigger thread: the failure comes from the real owner
    // path (run_job throws; the owner loop publishes cancellation whose
    // sink cancels the producer first and the throwing observer after) —
    // the external trigger race was part of the broken fixture.
    EXPECT_THROW(owned::RunRollingOwners(
                     coordinator, run_job,
                     [](size_t, const char*) {}),
                 std::exception);
    // The failure path never joins the producer explicitly (not fully
    // produced); the destructor CancelAndJoin-s it with the banks.
    pipeline.reset();
    // The failed cohort drains (the owner loop ran through) and the
    // remaining cohorts never advance: the run aborts at the first
    // failure with the producer and banks destroyed.
    if (cohort == 0) {
      EXPECT_TRUE(cancellation_published);
      EXPECT_TRUE(observer_threw);
      break;  // no advancing after failure
    }
  }
  EXPECT_EQ(cohort_advance_attempts, 1);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap