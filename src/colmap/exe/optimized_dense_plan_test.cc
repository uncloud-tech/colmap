// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/exe/optimized_dense_plan.h"

#include "colmap/exe/optimized_dense_plan_options.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#if defined(COLMAP_MVS_ENABLED) && defined(COLMAP_CUDA_ENABLED) &&      \
    defined(COLMAP_DOWNLOAD_ENABLED) && !defined(COLMAP_HIP_ENABLED) && \
    defined(__linux__)
#define COLMAP_DENSE_PLAN_NATIVE_TEST
#include "colmap/exe/mvs.h"
#include "colmap/mvs/optimized_photo_runner.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/resident_contract.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"
#endif

namespace colmap {
namespace {
int RunDensePlanCommand(std::vector<std::string> args) {
  args.insert(args.begin(), "colmap");
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  return RunOptimizedDensePlan(argv.size(), argv.data());
}
#ifdef COLMAP_DENSE_PLAN_NATIVE_TEST
void Write(const std::filesystem::path& path, const std::string& text) {
  std::ofstream output(path);
  ASSERT_TRUE(output.good());
  output << text;
}
#endif
TEST(OptimizedDensePlan, RealCommandRejectsUnknownOptionsWithoutWrites) {
  const auto root = CreateTestDir();
  EXPECT_NE(
      RunDensePlanCommand({"--log_path", (root / "must-not-exist").string()}),
      0);
  EXPECT_TRUE(std::filesystem::is_empty(root));
}
TEST(OptimizedDensePlan, RealCommandRejectsMissingWorkspaceWithoutOutput) {
  const auto root = CreateTestDir();
  EXPECT_NE(RunDensePlanCommand({"--workspace_path",
                                 (root / "missing").string(),
                                 "--source_cap",
                                 "20",
                                 "--num_samples",
                                 "7",
                                 "--gpu_index",
                                 "0",
                                 "--output_path",
                                 (root / "new.json").string()}),
            0);
  EXPECT_TRUE(std::filesystem::is_empty(root));
}

TEST(OptimizedDensePlan, InvalidLaunchPolicyRefusesBeforeWorkspaceOrOutput) {
  const auto root = CreateTestDir();
  const std::vector<std::string> base = {"--workspace_path",
                                         (root / "missing").string(),
                                         "--source_cap",
                                         "20",
                                         "--num_samples",
                                         "15",
                                         "--gpu_index",
                                         "0",
                                         "--output_path",
                                         (root / "plan.json").string()};
  for (const auto& policy : std::vector<std::vector<std::string>>{
           {"--cache_size_gb", "nan"},
           {"--cache_size_gb", "inf"},
           {"--cache_size_gb", "-1"},
           {"--cache_size_gb", "0"},
           {"--random_seed", "-2"},
           {"--random_seed", "1.5"},
           {"--owned_tex_lookahead", "0"},
           {"--owned_tex_lookahead", "False"},
           {"--owned_tex_lookahead", "true", "--owned_tex_lookahead", "false"},
           {"--unknown", "1"}}) {
    auto args = base;
    args.insert(args.end(), policy.begin(), policy.end());
    EXPECT_NE(RunDensePlanCommand(args), 0);
    EXPECT_TRUE(std::filesystem::is_empty(root));
  }
}

#ifndef COLMAP_DENSE_PLAN_NATIVE_TEST
TEST(OptimizedDensePlan, UnavailableNativeBuildIsActionableWithoutFallback) {
  const auto root = CreateTestDir();
  ::testing::internal::CaptureStderr();
  const auto result = RunDensePlanCommand({"--workspace_path",
                                           root.string(),
                                           "--source_cap",
                                           "8",
                                           "--num_samples",
                                           "15",
                                           "--gpu_index",
                                           "0",
                                           "--output_path",
                                           (root / "plan.json").string()});
  const auto diagnostic = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(result, 0);
  EXPECT_NE(diagnostic.find("requires Linux CUDA MVS"), std::string::npos);
  EXPECT_NE(diagnostic.find("No legacy fallback"), std::string::npos);
  EXPECT_TRUE(std::filesystem::is_empty(root));
}
#endif

#ifdef COLMAP_DENSE_PLAN_NATIVE_TEST
TEST(OptimizedDensePlan, RealStrictStereoCliPreservesPhaseAwareFilter) {
  const auto root = CreateTestDir();
  FLAGS_logtostderr = true;
  const auto check = [&](int tile, bool geo, bool la, const char* expected) {
    std::vector<std::string> args = {
        "colmap",
        "--workspace_path",
        root.string(),
        "--optimized_only",
        "true",
        "--optimized_dispatch_receipt",
        (root / "receipt.json").string(),
        "--owned_tile_threads",
        std::to_string(tile),
        "--PatchMatchStereo.owned_distinct_source_cache",
        "true",
        "--PatchMatchStereo.owned_tex_lookahead",
        la ? "true" : "false",
        "--PatchMatchStereo.num_threads",
        "1",
        "--PatchMatchStereo.filter",
        "false",
        "--PatchMatchStereo.geom_consistency",
        geo ? "true" : "false",
        "--owned_run_max_concurrent_solvers",
        "1",
        "--owned_run_timeout_seconds",
        "1",
        "--owned_photo_forecast",
        (root / "unread.forecast").string()};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    ::testing::internal::CaptureStderr();
    const auto result = RunPatchMatchStereoWithExecution(
        argv.size(), argv.data(), mvs::PatchMatchExecution::kStock);
    const auto diagnostic = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(result, 0);
    EXPECT_NE(diagnostic.find(expected), std::string::npos) << diagnostic;
    EXPECT_TRUE(std::filesystem::is_empty(root));
  };
  // Successful PHOTO option admission must reach the NEXT real CLI guard,
  // before opening even a forecast, model or device. GEO filter=false must
  // instead fail at strict option admission, with no fallback.
  for (int tile : {16, 32}) {
    for (bool lookahead : {false, true}) {
      check(tile,
            false,
            lookahead,
            "PHOTO requires a single-reference config_path");
      check(tile, true, lookahead, "GEO filter=true");
    }
  }
}

// Explicit opt-in because the surrounding native suite also runs on CPU build
// hosts. No dispatch mocks: each successful arm runs the real stereo CLI/CUDA
// solver and writes native depth/normal maps.
TEST(OptimizedPhotoExecution, RealCliV2V3N1IdentityAndN2Interleaving) {
  if (std::getenv("COLMAP_PHOTO_EXECUTION_TEST_GPU") == nullptr)
    GTEST_SKIP() << "set COLMAP_PHOTO_EXECUTION_TEST_GPU=1 on a CUDA executor";
  namespace fs = std::filesystem;
  const auto root = CreateTestDir();
  fs::create_directories(root / "sparse");
  fs::create_directories(root / "images");
  fs::create_directories(root / "stereo/depth_maps");
  fs::create_directories(root / "stereo/normal_maps");
  Write(root / "sparse/cameras.txt", "1 PINHOLE 16 12 10 10 8 6\n");
  Write(root / "sparse/images.txt",
        "1 1 0 0 0 0 0 0 1 a.png\n0 0 1\n"
        "2 1 0 0 0 -1 0 0 1 b.png\n0 0 1\n"
        "3 1 0 0 0 -2 0 0 1 c.png\n0 0 1\n");
  Write(root / "sparse/points3D.txt", "1 0 0 5 128 128 128 0 1 0 2 0 3 0\n");
  for (const char* name : {"a.png", "b.png", "c.png"}) {
    Bitmap bitmap(16, 12, true);
    bitmap.Fill(BitmapColor<uint8_t>(128));
    ASSERT_TRUE(bitmap.Write(root / "images" / name));
  }
  const auto c0 = root / "a.cfg", c1 = root / "b.cfg";
  Write(c0, "a.png\nb.png, c.png\n");
  Write(c1, "b.png\na.png, c.png\n");
  mvs::PatchMatchOptions options;
  options.geom_consistency = false;
  options.filter = false;
  options.gpu_index = "0";
  options.num_threads = 1;
  options.num_samples = 2;
  options.num_iterations = 1;
  options.random_seed = 0;
  options.depth_min = 0.1;
  options.depth_max = 10;
  options.cache_size = 0.001;
  options.owned_distinct_source_cache = true;
  options.owned_tex_lookahead = false;
  Write(root / "all.cfg",
        "a.png\nb.png, c.png\nb.png\na.png, c.png\nc.png\na.png, b.png\n");
  mvs::PatchMatchController metadata(options,
                                     root,
                                     "COLMAP",
                                     "option-all",
                                     root / "all.cfg",
                                     mvs::PatchMatchExecution::kStock);
  const auto graph = metadata.ResolveDensePlanGraph();
  const auto v2 = [&](const fs::path& config, size_t job) {
    const auto& resolved = graph.photo.at(job);
    std::string order = std::to_string(resolved.reference_image_idx) + " " +
                        std::to_string(resolved.source_image_idxs.size());
    for (const auto idx : resolved.source_image_idxs)
      order += " " + std::to_string(idx);
    std::string wire =
        "COLMAP_OPTIMIZED_PHOTO_V2\n" +
        ComputeSHA256(mvs::OptimizedPhotoOptions(options, root, config)) +
        "\n16777216 16777216 16777216 16777216\n" + order + "\n7\n";
    for (const auto& path : {config,
                             root / "sparse/cameras.txt",
                             root / "sparse/images.txt",
                             root / "sparse/points3D.txt",
                             root / "images/a.png",
                             root / "images/b.png",
                             root / "images/c.png"})
      wire += ComputeSHA256(mvs::ReadOptimizedBytes(path)) + " " +
              mvs::owned::Hex(fs::canonical(path).string()) + "\n";
    return wire;
  };
  const auto w0 = v2(c0, 0), w1 = v2(c1, 1);
  // TEST-EVIDENCE only: generous synthetic charges for this tiny numerical
  // fixture, not a calibrated workload admission or physical-fit claim.
  const auto binding = [&](const fs::path& config, size_t job) {
    const auto& resolved = graph.photo.at(job);
    std::string sources;
    for (const auto idx : resolved.source_image_idxs) {
      if (!sources.empty()) sources += " ";
      sources += std::to_string(idx);
    }
    return std::string("[BINDING]\nprofile=photo-phase-v1\nlineage_sha=") +
           mvs::kPhotoPhaseV1Lineage +
           "\nfingerprint=" + mvs::kPhotoPhaseV1Fingerprint +
           "\noptions_sha256=" +
           ComputeSHA256(mvs::OptimizedPhotoOptions(options, root, config)) +
           "\nprocess=16777216 16777216"
           "\nprivate_charge=16777216 16777216\nreference_image_idx=" +
           std::to_string(resolved.reference_image_idx) +
           "\nsource_image_idxs=" + sources +
           "\nsource_commit=" + mvs::kPhotoPhaseV1SourceCommit + "\n[END]\n";
  };
  const auto v3 = [&](int k, int n, bool two) {
    return "COLMAP_OPTIMIZED_PHOTO_V3\n" + std::to_string(k) + " " +
           std::to_string(n) + (two ? " 2\n" : " 1\n") +
           mvs::owned::Hex(c0.string()) + "\n" + mvs::owned::Hex(w0) + "\n" +
           (two ? mvs::owned::Hex(c1.string()) + "\n" + mvs::owned::Hex(w1) +
                      "\n"
                : "") +
           binding(c0, 0) + (two ? binding(c1, 1) : "");
  };
  int attempt = 0;
  const auto run = [&](const std::string& wire, int n) {
    const auto forecast = root / "launch";
    Write(forecast, wire);
    const auto receipt = root / ("receipt" + std::to_string(attempt++));
    std::vector<std::string> args = {
        "colmap",
        "--workspace_path",
        root.string(),
        "--config_path",
        c0.string(),
        "--optimized_only",
        "true",
        "--owned_photo_forecast",
        forecast.string(),
        "--optimized_dispatch_receipt",
        receipt.string(),
        "--owned_run_max_concurrent_solvers",
        std::to_string(n),
        "--owned_run_host_budget_bytes",
        "1073741824",
        "--owned_run_device_budget_bytes",
        "1073741824",
        "--owned_run_timeout_seconds",
        "60",
        "--PatchMatchStereo.gpu_index",
        "0",
        "--PatchMatchStereo.num_threads",
        "1",
        "--PatchMatchStereo.geom_consistency",
        "false",
        "--PatchMatchStereo.filter",
        "false",
        "--PatchMatchStereo.num_samples",
        "2",
        "--PatchMatchStereo.num_iterations",
        "1",
        "--PatchMatchStereo.random_seed",
        "0",
        "--PatchMatchStereo.depth_min",
        "0.1",
        "--PatchMatchStereo.depth_max",
        "10",
        "--PatchMatchStereo.cache_size",
        "0.001",
        "--PatchMatchStereo.owned_distinct_source_cache",
        "true",
        "--PatchMatchStereo.owned_tex_lookahead",
        "false"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    const int result = RunPatchMatchStereoWithExecution(
        argv.size(), argv.data(), mvs::PatchMatchExecution::kStock);
    return std::make_pair(result, receipt);
  };
  EXPECT_NE(run(v3(1, 2, true), 2).first, 0);   // N > supplied k.
  EXPECT_NE(run(v3(2, 2, false), 2).first, 0);  // N > job count.
  EXPECT_NE(run(v3(2, 2, true), 1).first, 0);   // Exact CLI N binding.
  const auto legacy = run(w0, 1);
  ASSERT_EQ(legacy.first, 0);
  const auto depth = root / "stereo/depth_maps/a.png.photometric.bin";
  const auto normal = root / "stereo/normal_maps/a.png.photometric.bin";
  const auto d0 = mvs::ReadOptimizedBytes(depth),
             n0 = mvs::ReadOptimizedBytes(normal);
  fs::remove(depth);
  fs::remove(normal);
  const auto single = run(v3(2, 1, false), 1);
  ASSERT_EQ(single.first, 0);
  EXPECT_EQ(d0, mvs::ReadOptimizedBytes(depth));
  EXPECT_EQ(n0, mvs::ReadOptimizedBytes(normal));
  EXPECT_EQ(mvs::ReadOptimizedBytes(legacy.second),
            mvs::ReadOptimizedBytes(single.second));
  fs::remove(depth);
  fs::remove(normal);
  // Capture an independent N1 oracle for the second reference too. N2 must
  // preserve each reference's bytes, not merely create the second output.
  const auto second =
      run("COLMAP_OPTIMIZED_PHOTO_V3\n2 1 1\n" + mvs::owned::Hex(c1.string()) +
              "\n" + mvs::owned::Hex(w1) + "\n" + binding(c1, 1),
          1);
  ASSERT_EQ(second.first, 0);
  const auto depth1 = root / "stereo/depth_maps/b.png.photometric.bin";
  const auto normal1 = root / "stereo/normal_maps/b.png.photometric.bin";
  const auto d1 = mvs::ReadOptimizedBytes(depth1);
  const auto n1 = mvs::ReadOptimizedBytes(normal1);
  fs::remove(depth1);
  fs::remove(normal1);
  const auto pair = run(v3(2, 2, true), 2);
  ASSERT_EQ(pair.first, 0);
  EXPECT_EQ(d0, mvs::ReadOptimizedBytes(depth));
  EXPECT_EQ(n0, mvs::ReadOptimizedBytes(normal));
  EXPECT_EQ(d1, mvs::ReadOptimizedBytes(depth1));
  EXPECT_EQ(n1, mvs::ReadOptimizedBytes(normal1));
  const auto receipt = mvs::ReadOptimizedBytes(pair.second);
  EXPECT_NE(receipt.find("\"rolling_active_jobs\":2"), std::string::npos);
  EXPECT_NE(receipt.find("\"max_concurrent_solvers\":2"), std::string::npos);
  EXPECT_NE(receipt.find("\"owner_intervals\":[{"), std::string::npos);
}

// No solver, maps, or device calls: real controller + command graph fixture.
class OptimizedDensePlanNative : public ::testing::TestWithParam<int> {};
TEST_P(OptimizedDensePlanNative,
       RealMetadataBeforePhotoAndExistingBankStillRequiresMaps) {
  const auto cap = std::to_string(GetParam());
  namespace fs = std::filesystem;
  const auto root = CreateTestDir();
  fs::create_directory(root / "sparse");
  fs::create_directory(root / "images");
  fs::create_directory(root / "stereo");
  Write(root / "sparse/cameras.txt", "1 PINHOLE 16 12 10 10 8 6\n");
  Write(root / "sparse/images.txt",
        "1 1 0 0 0 0 0 0 1 a.png\n0 0 1\n"
        "2 1 0 0 0 -1 0 0 1 b.png\n0 0 1\n"
        "3 1 0 0 0 -2 0 0 1 c.png\n0 0 1\n");
  Write(root / "sparse/points3D.txt", "1 0 0 5 128 128 128 0 1 0 2 0 3 0\n");
  for (const char* name : {"a.png", "b.png", "c.png"}) {
    Bitmap bitmap(16, 12, true);
    bitmap.Fill(BitmapColor<uint8_t>(128));
    ASSERT_TRUE(bitmap.Write(root / "images" / name));
  }
  Write(root / "stereo/patch-match.cfg",
        "a.png\n__auto__, " + cap + "\nb.png\n__auto__, " + cap +
            "\nc.png\n__auto__, " + cap + "\n");
  const auto before = GetRecursiveFileList(root);
  const auto bytes = [](const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(input)), {});
  };
  std::vector<std::string> original_bytes;
  for (const auto& path : before) original_bytes.push_back(bytes(path));
  mvs::PatchMatchOptions options;
  options.gpu_index = "0";
  options.num_threads = 1;
  options.num_samples = 7;
  options.cache_size = 2.5;
  options.random_seed = 42;
  options.owned_distinct_source_cache = true;
  options.owned_tex_lookahead = true;
  mvs::PatchMatchController controller(options, root, "COLMAP", "");
  const auto graph = controller.ResolveDensePlanGraph();
  ASSERT_EQ(graph.photo.size(), 3);
  ASSERT_EQ(graph.geo.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(graph.photo[i].source_image_idxs, graph.geo[i].source_image_idxs);
    EXPECT_EQ(graph.photo[i].source_image_idxs.size(), 2);
    EXPECT_EQ(graph.photo[i].options.num_samples, 7);
    EXPECT_FALSE(graph.photo[i].options.geom_consistency);
    EXPECT_FALSE(graph.photo[i].options.filter);
    EXPECT_TRUE(graph.geo[i].options.geom_consistency);
    EXPECT_DOUBLE_EQ(graph.geo[i].options.depth_min, 3.75);
    EXPECT_DOUBLE_EQ(graph.geo[i].options.depth_max, 6.25);
  }
  EXPECT_EQ(GetRecursiveFileList(root), before);
  mvs::PatchMatchController bank(options, root, "COLMAP", "");
  EXPECT_ANY_THROW(bank.ResolveOwnedBankContract());
  const auto output = root / "plan.json";
  const std::vector<std::string> args = {"--workspace_path",
                                         root.string(),
                                         "--source_cap",
                                         cap,
                                         "--num_samples",
                                         "7",
                                         "--gpu_index",
                                         "0",
                                         "--output_path",
                                         output.string()};
  auto policy_args = args;
  policy_args.insert(policy_args.end(),
                     {"--cache_size_gb", "2.5", "--random_seed", "42"});
  ASSERT_EQ(RunDensePlanCommand(policy_args), 0);
  std::ifstream input(output);
  const std::string json((std::istreambuf_iterator<char>(input)), {});
  EXPECT_NE(
      json.find(
          "\"dispatch_support\":{\"backend\":\"owned_cuda\",\"tiles\":[16,32]"),
      std::string::npos);
  EXPECT_NE(json.find("\"schema_version\":1"), std::string::npos);
  EXPECT_NE(json.find("\"concurrency_support\":" +
                      dense_plan::ConcurrencySupportJson(true, 7).Dump()),
            std::string::npos);
  EXPECT_NE(json.find("\"cache_size\":2.5"), std::string::npos);
  EXPECT_NE(json.find("\"random_seed\":42"), std::string::npos);
  EXPECT_NE(json.find("\"source_cap\":" + cap), std::string::npos);
  EXPECT_NE(json.find("\"num_samples\":7"), std::string::npos);
  EXPECT_NE(json.find("\"geom_consistency\":false"), std::string::npos);
  EXPECT_NE(json.find("fresh_controller_after_photo"), std::string::npos);
  EXPECT_NE(json.find("\"requested_options\":{\"max_image_size\":-1"),
            std::string::npos);
  EXPECT_NE(json.find("\"depth_min\":-1,\"depth_max\":-1"), std::string::npos);
  EXPECT_NE(json.find("\"owned_distinct_source_cache\":true"),
            std::string::npos);
  EXPECT_NE(json.find("\"owned_tex_lookahead\":true"), std::string::npos);
  EXPECT_NE(json.find("\"effective_options_wire_v1\":"), std::string::npos);
  EXPECT_EQ(json.find("\"options_wire_v1\":"), std::string::npos);
  EXPECT_NE(json.find("\"future_photo_dependencies\":[{"), std::string::npos);
  EXPECT_NE(json.find("\"sha256\":\""), std::string::npos);
  EXPECT_NE(RunDensePlanCommand(args), 0);
  std::ifstream unchanged(output);
  EXPECT_EQ(std::string((std::istreambuf_iterator<char>(unchanged)), {}), json);
  EXPECT_FALSE(fs::exists(root / "stereo/depth_maps"));
  for (size_t i = 0; i < before.size(); ++i)
    EXPECT_EQ(bytes(before[i]), original_bytes[i]);

  // A separate real exporter query produces OFF facts; never rewrite ON JSON.
  auto off_args = args;
  off_args.back() = (root / "plan-off.json").string();
  off_args.insert(off_args.end(),
                  {"--cache_size_gb",
                   "2.5",
                   "--random_seed",
                   "42",
                   "--owned_tex_lookahead",
                   "false"});
  ASSERT_EQ(RunDensePlanCommand(off_args), 0);
  const auto off_json = bytes(root / "plan-off.json");
  EXPECT_EQ(off_json.find("\"owned_tex_lookahead\":true"), std::string::npos);
  EXPECT_NE(off_json.find("\"owned_tex_lookahead\":false"), std::string::npos);
  EXPECT_NE(off_json.find("\"lookahead_modes\":[false,true]"),
            std::string::npos);
  auto off_options = options;
  off_options.owned_tex_lookahead = false;
  mvs::PatchMatchController off_controller(off_options, root, "COLMAP", "");
  const auto off_graph = off_controller.ResolveDensePlanGraph();
  ASSERT_EQ(off_graph.photo.size(), graph.photo.size());
  ASSERT_EQ(off_graph.geo.size(), graph.geo.size());
  for (const auto* phase : {&off_graph.photo, &off_graph.geo}) {
    const auto& on_phase = phase == &off_graph.photo ? graph.photo : graph.geo;
    for (size_t i = 0; i < phase->size(); ++i) {
      EXPECT_FALSE((*phase)[i].options.owned_tex_lookahead);
      EXPECT_TRUE((*phase)[i].options.owned_distinct_source_cache);
      EXPECT_EQ((*phase)[i].source_image_idxs, on_phase[i].source_image_idxs);
      auto expected = on_phase[i].options;
      expected.owned_tex_lookahead = false;
      EXPECT_EQ(dense_plan::OptionsJson((*phase)[i].options).Dump(),
                dense_plan::OptionsJson(expected).Dump());
    }
  }
  for (size_t i = 0; i < before.size(); ++i)
    EXPECT_EQ(bytes(before[i]), original_bytes[i]);

  auto mismatch = args;
  mismatch[3] = GetParam() == 8 ? "20" : "8";
  mismatch.back() = (root / "mismatch.json").string();
  EXPECT_NE(RunDensePlanCommand(mismatch), 0);
  EXPECT_FALSE(fs::exists(root / "mismatch.json"));

  // The actual complete GEO bank still refuses PHOTO-mode calls, then binds
  // exactly the planned fresh-controller order once real map inputs exist.
  auto photo_options = options;
  photo_options.geom_consistency = false;
  mvs::PatchMatchController invalid_bank(photo_options, root, "COLMAP", "");
  EXPECT_ANY_THROW(invalid_bank.ResolveOwnedBankJobs());
  fs::create_directory(root / "stereo/depth_maps");
  fs::create_directory(root / "stereo/normal_maps");
  for (const char* name : {"a.png", "b.png", "c.png"}) {
    mvs::DepthMap depth(16, 12, 3.75f, 6.25f);
    depth.Fill(5.0f);
    mvs::NormalMap normal(16, 12);
    normal.Fill(0.5f);
    depth.Write(root / "stereo/depth_maps" /
                (std::string(name) + ".photometric.bin"));
    normal.Write(root / "stereo/normal_maps" /
                 (std::string(name) + ".photometric.bin"));
  }
  mvs::PatchMatchController complete_bank(options, root, "COLMAP", "");
  const auto resolved = complete_bank.ResolveOwnedBankContractWithJobs();
  ASSERT_EQ(resolved.jobs.size(), graph.geo.size());
  for (size_t i = 0; i < graph.geo.size(); ++i)
    EXPECT_EQ(resolved.jobs[i].source_image_idxs,
              graph.geo[i].source_image_idxs);
}
// Geometrically valid sparse fixtures, not reconstruction-quality evidence.
// Every reference has cap actual AUTO-selected sources, independently of the
// PatchMatch sample count. Native metadata resolution never starts a solver.
class OptimizedDensePlanFullSources
    : public ::testing::TestWithParam<std::tuple<int, int>> {};
TEST_P(OptimizedDensePlanFullSources,
       NativeAutoSourcesMatchFreshGeoBankAfterCompletePhotoBarrier) {
  namespace fs = std::filesystem;
  const auto [cap, samples] = GetParam();
  const int count = cap + 1;
  const auto root = CreateTestDir();
  fs::create_directory(root / "sparse");
  fs::create_directory(root / "images");
  fs::create_directory(root / "stereo");
  Write(root / "sparse/cameras.txt", "1 PINHOLE 16 12 10 10 8 6\n");
  std::ostringstream images, track, config;
  images << std::setprecision(17);
  std::vector<std::string> names;
  for (int i = 0; i < count; ++i) {
    const double angle = 2 * std::acos(-1.0) * i / count;
    const double x = std::cos(angle), y = std::sin(angle);
    const std::string name = "ring" + std::to_string(i) + ".png";
    names.push_back(name);
    images << i + 1 << " 1 0 0 0 " << -2 * x << ' ' << -2 * y << " 0 1 " << name
           << '\n'
           << 8 - 4 * x << ' ' << 6 - 4 * y << " 1\n";
    track << ' ' << i + 1 << " 0";
    config << name << "\n__auto__, " << cap << '\n';
    Bitmap bitmap(16, 12, true);
    bitmap.Fill(BitmapColor<uint8_t>(128));
    ASSERT_TRUE(bitmap.Write(root / "images" / name));
  }
  Write(root / "sparse/images.txt", images.str());
  Write(root / "sparse/points3D.txt",
        "1 0 0 5 128 128 128 0" + track.str() + "\n");
  Write(root / "stereo/patch-match.cfg", config.str());
  const auto bytes = [](const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(input)), {});
  };
  const auto before = GetRecursiveFileList(root);
  std::vector<std::string> original;
  for (const auto& path : before) original.push_back(bytes(path));

  mvs::PatchMatchOptions options;
  options.gpu_index = "0";
  options.num_threads = 1;
  options.num_samples = samples;
  options.owned_distinct_source_cache = true;
  options.owned_tex_lookahead = true;
  ASSERT_EQ(options.depth_min, -1);
  ASSERT_EQ(options.depth_max, -1);
  mvs::PatchMatchController controller(options, root, "COLMAP", "");
  const auto graph = controller.ResolveDensePlanGraph();
  ASSERT_EQ(graph.names.size(), count);
  ASSERT_EQ(graph.photo.size(), count);
  ASSERT_EQ(graph.geo.size(), count);
  std::set<int> producers, references;
  std::set<std::string> outputs;
  for (const auto& photo : graph.photo) {
    EXPECT_TRUE(producers.insert(photo.reference_image_idx).second);
    EXPECT_TRUE(outputs.insert(photo.depth_map_path.string()).second);
    EXPECT_TRUE(outputs.insert(photo.normal_map_path.string()).second);
    EXPECT_FALSE(fs::exists(photo.depth_map_path));
    EXPECT_FALSE(fs::exists(photo.normal_map_path));
  }
  for (size_t i = 0; i < graph.geo.size(); ++i) {
    const auto& photo = graph.photo[i];
    const auto& geo = graph.geo[i];
    EXPECT_TRUE(references.insert(geo.reference_image_idx).second);
    EXPECT_EQ(photo.reference_image_idx, geo.reference_image_idx);
    ASSERT_EQ(photo.source_image_idxs.size(), cap);
    ASSERT_EQ(geo.source_image_idxs.size(), cap);
    EXPECT_EQ(photo.source_image_idxs, geo.source_image_idxs);
    const std::set<int> sources(geo.source_image_idxs.begin(),
                                geo.source_image_idxs.end());
    EXPECT_EQ(sources.size(), cap);
    EXPECT_EQ(sources.count(geo.reference_image_idx), 0);
    for (int source : sources) EXPECT_EQ(producers.count(source), 1);
    for (const auto* job : {&photo, &geo}) {
      EXPECT_EQ(job->options.num_samples, samples);
      EXPECT_EQ(job->options.max_image_size, -1);
      EXPECT_DOUBLE_EQ(job->options.depth_min, 3.75);
      EXPECT_DOUBLE_EQ(job->options.depth_max, 6.25);
      EXPECT_TRUE(job->options.owned_distinct_source_cache);
      EXPECT_TRUE(job->options.owned_tex_lookahead);
    }
    EXPECT_FALSE(photo.options.geom_consistency);
    EXPECT_FALSE(photo.options.filter);
    EXPECT_TRUE(geo.options.geom_consistency);
    EXPECT_TRUE(geo.options.filter);
  }
  EXPECT_EQ(references, producers);
  EXPECT_EQ(GetRecursiveFileList(root), before);

  const auto output = root / "plan.json";
  const std::vector<std::string> args = {"--workspace_path",
                                         root.string(),
                                         "--source_cap",
                                         std::to_string(cap),
                                         "--num_samples",
                                         std::to_string(samples),
                                         "--gpu_index",
                                         "0",
                                         "--output_path",
                                         output.string()};
  // Unsupported specialization refuses before creating any output.
  auto unsupported = args;
  unsupported[5] = "16";
  EXPECT_NE(RunDensePlanCommand(unsupported), 0);
  EXPECT_EQ(GetRecursiveFileList(root), before);
  ASSERT_EQ(RunDensePlanCommand(args), 0);
  const auto json = bytes(output);
  EXPECT_NE(json.find("\"source_cap\":" + std::to_string(cap)),
            std::string::npos);
  EXPECT_NE(json.find("\"num_samples\":" + std::to_string(samples)),
            std::string::npos);
  EXPECT_NE(json.find("\"requested_options\":{\"max_image_size\":-1"),
            std::string::npos);
  EXPECT_NE(json.find("\"depth_min\":-1,\"depth_max\":-1"), std::string::npos);
  EXPECT_NE(json.find("fresh_controller_after_photo"), std::string::npos);
  // Bind the real command's ordered graph to the real controller, not to a
  // substitute graph parser or explicit source list fed back through AUTO.
  for (const auto& photo : graph.photo) {
    for (const char* kind : {"depth", "normal"}) {
      const auto dependency =
          "\"reference\":" + std::to_string(photo.reference_image_idx) +
          ",\"kind\":\"" + kind +
          "\",\"width\":16,\"height\":12,\"channels\":" +
          (std::string(kind) == "depth" ? "1" : "3");
      EXPECT_NE(json.find(dependency), std::string::npos);
    }
  }
  for (const auto& geo : graph.geo) {
    std::ostringstream record;
    record << "\"reference\":" << geo.reference_image_idx << ",\"sources\":[";
    for (size_t j = 0; j < geo.source_image_idxs.size(); ++j) {
      if (j) record << ',';
      record << geo.source_image_idxs[j];
    }
    record << "],\"depth_min\":3.75,\"depth_max\":6.25";
    EXPECT_NE(json.find(record.str()), std::string::npos);
  }
  for (size_t i = 0; i < before.size(); ++i)
    EXPECT_EQ(bytes(before[i]), original[i]);
  EXPECT_FALSE(fs::exists(root / "stereo/depth_maps"));
  mvs::PatchMatchController missing_maps(options, root, "COLMAP", "");
  EXPECT_ANY_THROW(missing_maps.ResolveOwnedBankContractWithJobs());

  fs::create_directory(root / "stereo/depth_maps");
  fs::create_directory(root / "stereo/normal_maps");
  for (const auto& name : names) {
    mvs::DepthMap depth(16, 12, 3.75f, 6.25f);
    depth.Fill(5.0f);
    mvs::NormalMap normal(16, 12);
    normal.Fill(0.5f);
    depth.Write(root / "stereo/depth_maps" / (name + ".photometric.bin"));
    normal.Write(root / "stereo/normal_maps" / (name + ".photometric.bin"));
  }
  const auto complete_files = GetRecursiveFileList(root);
  std::vector<std::string> complete_bytes;
  for (const auto& path : complete_files) complete_bytes.push_back(bytes(path));
  // Deliberately retain the original per-reference AUTO selector. A fresh
  // native GEO controller must independently recover exactly the plan order.
  mvs::PatchMatchController bank(options, root, "COLMAP", "");
  const auto resolved = bank.ResolveOwnedBankContractWithJobs();
  ASSERT_EQ(resolved.jobs.size(), count);
  ASSERT_EQ(resolved.contract.problems.size(), count);
  EXPECT_EQ(resolved.contract.invocation_options.depth_min, -1);
  EXPECT_EQ(resolved.contract.invocation_options.depth_max, -1);
  for (size_t i = 0; i < graph.geo.size(); ++i) {
    const auto& planned = graph.geo[i];
    const auto& job = resolved.jobs[i];
    const auto& problem = resolved.contract.problems[i];
    EXPECT_EQ(job.reference_image_idx, planned.reference_image_idx);
    EXPECT_EQ(job.source_image_idxs, planned.source_image_idxs);
    EXPECT_EQ(problem.reference_image_idx, planned.reference_image_idx);
    EXPECT_EQ(job.depth_map_path, planned.depth_map_path);
    EXPECT_EQ(job.normal_map_path, planned.normal_map_path);
    EXPECT_EQ(problem.effective_options.num_samples, samples);
    EXPECT_DOUBLE_EQ(problem.effective_options.depth_min, 3.75);
    EXPECT_DOUBLE_EQ(problem.effective_options.depth_max, 6.25);
    EXPECT_TRUE(problem.effective_options.filter);
    EXPECT_TRUE(problem.effective_options.geom_consistency);
    EXPECT_TRUE(problem.effective_options.owned_distinct_source_cache);
    EXPECT_TRUE(problem.effective_options.owned_tex_lookahead);
    ASSERT_EQ(problem.inputs.size(), count);
    std::vector<int> contract_sources;
    int reference_count = 0;
    for (const auto& input : problem.inputs) {
      EXPECT_EQ(input.width, 16);
      EXPECT_EQ(input.height, 12);
      EXPECT_EQ(input.depth_header_width, 16);
      EXPECT_EQ(input.depth_header_height, 12);
      EXPECT_EQ(input.depth_header_depth, 1);
      EXPECT_EQ(input.normal_header_width, 16);
      EXPECT_EQ(input.normal_header_height, 12);
      EXPECT_EQ(input.normal_header_depth, 3);
      if (input.is_reference) {
        ++reference_count;
        EXPECT_EQ(input.image_idx, planned.reference_image_idx);
      } else {
        contract_sources.push_back(input.image_idx);
      }
    }
    EXPECT_EQ(reference_count, 1);
    EXPECT_EQ(contract_sources, planned.source_image_idxs);
  }
  EXPECT_EQ(GetRecursiveFileList(root), complete_files);
  for (size_t i = 0; i < complete_files.size(); ++i)
    EXPECT_EQ(bytes(complete_files[i]), complete_bytes[i]);
  EXPECT_EQ(bytes(root / "stereo/patch-match.cfg"), config.str());
  EXPECT_EQ(bytes(output), json);
}
INSTANTIATE_TEST_SUITE_P(SourceCapsAndSampleCounts,
                         OptimizedDensePlanFullSources,
                         ::testing::Combine(::testing::Values(8, 20),
                                            ::testing::Values(7, 15)));
INSTANTIATE_TEST_SUITE_P(SourceCaps,
                         OptimizedDensePlanNative,
                         ::testing::Values(8, 20));
#endif
}  // namespace
}  // namespace colmap
