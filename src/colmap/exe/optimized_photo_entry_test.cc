// SPDX-License-Identifier: BSD-3-Clause
// CPU-executable in a CUDA+SHA256 native build: never creates a solver/device.
#include "colmap/exe/mvs.h"
#include "colmap/mvs/optimized_photo_runner.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/testing.h"

#include <fstream>
#include <type_traits>

#include <gtest/gtest.h>

namespace colmap {
namespace {
// Compile regression: only the exact shared PHOTO wrapper has private access.
template <class T, class = void>
struct HasPublicPreparedExecution : std::false_type {};
template <class T>
struct HasPublicPreparedExecution<
    T,
    std::void_t<decltype(&T::ExecutePreparedOwnedPairJob)>> : std::true_type {};
static_assert(!HasPublicPreparedExecution<mvs::PatchMatchController>::value);
// Including the real inline runner above also compiles its friend-only call.

TEST(OptimizedPhotoEntry, TrustedDeliveryReachesRealInputReservation) {
  const auto root = CreateTestDir();
  const auto config = root / "one.cfg";
  const std::string digest(64, 'a');
  // Synthetic evidence belongs only to this test callback, never to an adapter.
  const auto v2 = "COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                  "\n310000 100 200000 900000\n0 2 1 2\n1\n" + digest +
                  " 2f61\n";
  const auto wire = "COLMAP_OPTIMIZED_PHOTO_V3\n1 1 1\n" +
                    mvs::owned::Hex(config.string()) + "\n" +
                    mvs::owned::Hex(v2) + "\n";
  const auto forecast = root / "launch";
  {
    std::ofstream out(forecast);
    out << wire;
  }
  const float K[9] = {10, 0, 8, 0, 10, 6, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  std::vector<mvs::Image> images;
  for (const char* name : {"a.png", "b.png", "c.png"}) {
    const auto path = root / name;
    Bitmap bitmap(16, 12, true);
    bitmap.Fill(BitmapColor<uint8_t>(128));
    ASSERT_TRUE(bitmap.Write(path));
    images.emplace_back(path, 16, 12, K, R, T);
  }
  mvs::PatchMatch::Problem problem;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {1, 2};
  problem.images = &images;
  int delivered = 0;
  const mvs::TrustedPhotoPhaseV1Delivery delivery =
      [&](const std::string& bytes) {
        EXPECT_EQ(bytes, wire);
        ++delivered;
        return std::vector<mvs::PhotoPhaseV1Binding>{{"photo-phase-v1",
                                                      std::string(40, 'b'),
                                                      digest,
                                                      digest,
                                                      {310000, 100},
                                                      {200000, 900000},
                                                      0,
                                                      {1, 2}}};
      };
  const auto run = [&](const std::string& cache,
                       const mvs::TrustedPhotoPhaseV1Delivery& source,
                       const std::string& expected,
                       bool should_probe,
                       bool should_check) {
    SCOPED_TRACE(expected);
    const auto receipt = root / "receipt";
    bool probed = false;
    bool checked = false;
    const mvs::OptimizedPhotoInputProbe probe = [&](const auto& check) {
      probed = true;
      // This calls CheckOptimizedPhotoInputs, including real image-header reads
      // and the bound process-cache reservation, not WithCheckedInputs
      // directly.
      check(0, problem);
      checked = true;
    };
    std::vector<std::string> args = {
        "colmap",
        "--workspace_path",
        root.string(),
        "--config_path",
        config.string(),
        "--optimized_only",
        "true",
        "--owned_photo_forecast",
        forecast.string(),
        "--optimized_dispatch_receipt",
        receipt.string(),
        "--owned_run_max_concurrent_solvers",
        "1",
        "--owned_run_host_budget_bytes",
        "510000",
        "--owned_run_device_budget_bytes",
        "900100",
        "--owned_run_timeout_seconds",
        "60",
        "--PatchMatchStereo.num_threads",
        "1",
        "--PatchMatchStereo.geom_consistency",
        "false",
        "--PatchMatchStereo.filter",
        "false",
        "--PatchMatchStereo.num_samples",
        "2",
        "--PatchMatchStereo.cache_size",
        cache,
        "--PatchMatchStereo.owned_distinct_source_cache",
        "true"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    ::testing::internal::CaptureStderr();
    const int result =
        RunPatchMatchStereoWithExecution(argv.size(),
                                         argv.data(),
                                         mvs::PatchMatchExecution::kStock,
                                         source,
                                         probe);
    const auto diagnostic = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(result, 0);  // A probe can never publish successful execution.
    EXPECT_NE(diagnostic.find(expected), std::string::npos) << diagnostic;
    EXPECT_EQ(probed, should_probe);
    EXPECT_EQ(checked, should_check);
    EXPECT_FALSE(std::filesystem::exists(receipt));
  };
  // 0.00028 GiB is >private.host and <process.host. A legacy-floor fallback
  // would reject this; only the actual bound reservation admits it.
  run("0.00028",
      delivery,
      "PHOTO input probe completed without execution",
      true,
      true);
  EXPECT_EQ(delivered, 1);
  run("0.00030",
      delivery,
      "PHOTO cache exceeds process reservation",
      true,
      false);
  EXPECT_EQ(delivered, 2);
  run("0.00028",
      {},
      "PHOTO V3 requires delivered photo-phase-v1 binding",
      false,
      false);
  EXPECT_EQ(delivered, 2);
  run(
      "0.00028",
      [](const std::string&) {
        return std::vector<mvs::PhotoPhaseV1Binding>{};
      },
      "PHOTO V3 requires delivered photo-phase-v1 binding",
      false,
      false);
  // Identical legacy V2 bytes take the pre-existing unbound reservation path.
  {
    std::ofstream out(forecast);
    out << v2;
  }
  run("0.00018",
      {},
      "PHOTO input probe completed without execution",
      true,
      true);
}
TEST(OptimizedPhotoEntry, RunnerObjectsAttestAndAdmitTwoAtExactCaps) {
  const auto root = CreateTestDir();
  const auto config = root / "one.cfg";
  const auto config2 = root / "two.cfg";
  const auto forecast = root / "launch";
  const std::string digest(64, 'a');
  // TEST-EVIDENCE: real delivery object shape/options/SHA256, synthetic
  // canonical charges, not a claim of model qualification or GPU execution.
  mvs::PatchMatchOptions effective;
  effective.num_threads = 1;
  effective.geom_consistency = false;
  effective.filter = false;
  effective.num_samples = 2;
  effective.cache_size = 0.00028;
  effective.owned_distinct_source_cache = true;
  const auto options_bytes = [&](int ref) {
    return mvs::OptimizedPhotoOptions(effective, root, ref ? config2 : config);
  };
  const auto job_wire = [&](int ref) {
    return "COLMAP_OPTIMIZED_PHOTO_V2\n" + ComputeSHA256(options_bytes(ref)) +
           "\n310000 100 200000 900000\n" + std::to_string(ref) +
           " 2 2 3\n1\n" + digest + " 2f61\n";
  };
  const auto wire = "COLMAP_OPTIMIZED_PHOTO_V3\n2 2 2\n" +
                    mvs::owned::Hex(config.string()) + "\n" +
                    mvs::owned::Hex(job_wire(0)) + "\n" +
                    mvs::owned::Hex(config2.string()) + "\n" +
                    mvs::owned::Hex(job_wire(1)) + "\n";
  {
    std::ofstream out(forecast);
    out << wire;
  }
  const float K[9] = {10, 0, 8, 0, 10, 6, 0, 0, 1};
  const float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[3] = {0, 0, 0};
  std::vector<mvs::Image> images;
  for (const char* name : {"a.png", "b.png", "c.png", "d.png"}) {
    const auto path = root / name;
    Bitmap bitmap(16, 12, true);
    bitmap.Fill(BitmapColor<uint8_t>(128));
    ASSERT_TRUE(bitmap.Write(path));
    images.emplace_back(path, 16, 12, K, R, T);
  }
  mvs::PhotoPhaseV1DeliveryInputs delivered;
  delivered.exact_launch_bytes = wire;
  for (int ref = 0; ref < 2; ++ref) {
    delivered.bindings.push_back({"photo-phase-v1",
                                  mvs::kPhotoPhaseV1Lineage,
                                  mvs::kPhotoPhaseV1Fingerprint,
                                  ComputeSHA256(options_bytes(ref)),
                                  {310000, 100},
                                  {200000, 900000},
                                  ref,
                                  {2, 3},
                                  mvs::kPhotoPhaseV1SourceCommit});
    delivered.effective_jobs.push_back(
        {options_bytes(ref), {310000, 100}, {200000, 900000}, ref, {2, 3}});
  }
  delivered.sha256 = [](const std::string_view& bytes) {
    return ComputeSHA256(bytes);
  };
  int deliveries = 0;
  const mvs::PhotoPhaseV1RunnerDelivery runner = [&](const std::string& bytes) {
    EXPECT_EQ(bytes, wire);
    ++deliveries;
    return delivered;
  };
  std::string requested_n = "2";
  std::string host_cap = "710000";
  std::string device_cap = "1800100";
  std::string cohort_forecast;
  bool reverse_effective_sources = false;
  const auto run = [&](const std::string& expected,
                       int expected_checked,
                       bool conflicting_source = false,
                       bool from_wire = false) {
    int checked = 0;
    const mvs::OptimizedPhotoInputProbe probe = [&](const auto& check) {
      for (size_t i = 0; i < 2; ++i) {
        mvs::PatchMatch::Problem problem;
        problem.ref_image_idx = static_cast<int>(i);
        problem.src_image_idxs = reverse_effective_sources
                                     ? std::vector<int>{3, 2}
                                     : std::vector<int>{2, 3};
        problem.images = &images;
        check(i, problem);
        ++checked;
      }
    };
    const auto receipt = root / "receipt";
    std::vector<std::string> args = {
        "colmap",
        "--workspace_path",
        root.string(),
        "--config_path",
        config.string(),
        "--optimized_only",
        "true",
        "--owned_photo_forecast",
        forecast.string(),
        "--optimized_dispatch_receipt",
        receipt.string(),
        "--owned_run_max_concurrent_solvers",
        requested_n,
        "--owned_run_host_budget_bytes",
        host_cap,
        "--owned_run_device_budget_bytes",
        device_cap,
        "--owned_run_timeout_seconds",
        "60",
        "--PatchMatchStereo.num_threads",
        "1",
        "--PatchMatchStereo.geom_consistency",
        "false",
        "--PatchMatchStereo.filter",
        "false",
        "--PatchMatchStereo.num_samples",
        "2",
        "--PatchMatchStereo.cache_size",
        "0.00028",
        "--PatchMatchStereo.owned_distinct_source_cache",
        "true"};
    if (!cohort_forecast.empty())
      args.insert(args.end(), {"--owned_run_cohort_forecast", cohort_forecast});
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    ::testing::internal::CaptureStderr();
    mvs::TrustedPhotoPhaseV1Delivery trusted;
    if (conflicting_source) {
      trusted = [](const std::string&) {
        ADD_FAILURE() << "conflicting delivery must refuse before callbacks";
        return std::vector<mvs::PhotoPhaseV1Binding>{};
      };
    }
    const int result = RunPatchMatchStereoWithExecution(
        argv.size(),
        argv.data(),
        mvs::PatchMatchExecution::kStock,
        trusted,
        probe,
        from_wire ? mvs::PhotoPhaseV1RunnerDelivery{} : runner);
    const auto diagnostic = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(result,
              0);  // CPU probe never claims successful native execution.
    EXPECT_NE(diagnostic.find(expected), std::string::npos) << diagnostic;
    EXPECT_EQ(checked, expected_checked);
    EXPECT_FALSE(std::filesystem::exists(receipt));
  };
  // Exact process + 2*private caps: adding another margin would refuse. Both
  // jobs must reach the real bound reservation (cache > private.host).
  run("PHOTO input probe completed without execution", 2);
  EXPECT_EQ(deliveries, 1);
  delivered.bindings[1].source_image_idxs = {3, 2};
  run("PHOTO delivery per-reference mismatch", 0);
  delivered.bindings[1].source_image_idxs = {2, 3};
  delivered.exact_launch_bytes += "changed";
  run("PHOTO delivery launch bytes mismatch", 0);
  delivered.exact_launch_bytes = wire;
  delivered.effective_jobs[0].options_bytes += "changed";
  run("PHOTO delivery options mismatch", 0);
  delivered.effective_jobs[0].options_bytes = options_bytes(0);
  ++delivered.effective_jobs[0].private_charge.device;
  run("PHOTO delivery canonical charge mismatch", 0);
  --delivered.effective_jobs[0].private_charge.device;
  delivered.bindings[0].fingerprint = std::string(64, 'b');
  run("lineage-source_commit-incoherent", 0);
  delivered.bindings[0].fingerprint = mvs::kPhotoPhaseV1Fingerprint;
  delivered.sha256 = {};
  run("PHOTO delivery missing", 0);
  run("PHOTO conflicting delivery sources", 0, true);
  // The ordinary CLI source is now the V3 wire itself, not an injected
  // runner callback. The probe only avoids device execution after admission.
  const auto sections = [&] {
    std::string result;
    for (const auto& b : delivered.bindings) {
      result += "[BINDING]\nprofile=" + b.profile +
                "\nlineage_sha=" + b.lineage_sha +
                "\nfingerprint=" + b.fingerprint +
                "\noptions_sha256=" + b.options_sha256 +
                "\nprocess=310000 100\nprivate_charge=200000 900000"
                "\nreference_image_idx=" +
                std::to_string(b.reference_image_idx) +
                "\nsource_image_idxs=2 3\nsource_commit=" + b.source_commit +
                "\n[END]\n";
    }
    return result;
  };
  const auto write_wire = [&](const std::string& suffix) {
    std::ofstream out(forecast);
    out << wire << suffix;
  };
  write_wire(sections());
  run("PHOTO input probe completed without execution", 2, false, true);
  // Ordinary CLI wire source, exact process + top-N private caps, no extra
  // margin and no clamping. These checks run before any device execution.
  host_cap = "709999";
  run("PHOTO top-N private charges exceed caps", 0, false, true);
  host_cap = "710000";
  device_cap = "1800099";
  run("PHOTO top-N private charges exceed caps", 0, false, true);
  device_cap = "1800100";
  requested_n = "1";
  run("PHOTO rolling_active_jobs must equal CLI N", 0, false, true);
  requested_n = "2";
  reverse_effective_sources = true;
  run("PHOTO effective reference/source order mismatch", 0, false, true);
  reverse_effective_sources = false;
  cohort_forecast = (root / "unread.cohort").string();
  run("PHOTO requires owned_photo_forecast only", 0, false, true);
  cohort_forecast.clear();
  write_wire("");
  run("PHOTO V3 requires delivered photo-phase-v1 binding", 0, false, true);
  write_wire("[BINDING]\nunknown=value\n[END]\n");
  run("PHOTO malformed binding section: unknown key unknown", 0, false, true);
  write_wire("[BINDING]\nprofile=photo-phase-v1\n[END]\n");
  run("PHOTO malformed binding section: missing", 0, false, true);
  delivered.bindings[0].fingerprint =
      "d26041fafe7860b2653bafe6702ca97043881b14d953097112bcb4b7bde0117e";
  write_wire(sections());
  run("lineage-source_commit-incoherent", 0, false, true);
  EXPECT_EQ(deliveries, 7);
}
}  // namespace
}  // namespace colmap
