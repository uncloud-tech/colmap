// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/exe/optimized_dense_plan_io.h"

#include "colmap/exe/optimized_dense_plan.h"
#include "colmap/exe/optimized_dense_plan_options.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif
#ifndef OPTIMIZED_DENSE_PLAN_STANDALONE_TEST
#include <gtest/gtest.h>
#endif

namespace colmap {
namespace {
void Check(bool value) {
  if (!value) throw std::runtime_error("dense plan IO assertion failed");
}
template <class F>
void Rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception&) {
    rejected = true;
  }
  Check(rejected);
}
void TestDensePlanJson() {
  using dense_plan::Json;
  Check(Json::Object({{"text", Json("a\"\\\n\t\x01")},
                      {"number", Json(20)},
                      {"flag", Json(true)},
                      {"unset", Json()},
                      {"sources", Json::Array({Json(3), Json(1)})}})
            .Dump() ==
        "{\"text\":\"a\\\"\\\\\\n\\t\\u0001\",\"number\":20,\"flag\":true,"
        "\"unset\":null,\"sources\":[3,1]}");
  Check(Json("caf\xc3\xa9").Dump() == "\"caf\xc3\xa9\"");
  Rejects([] { Json bad(std::string("\xc0\x80")); });
  Rejects([] { Json bad(std::numeric_limits<double>::infinity()); });
  Rejects([] { Json bad(std::numeric_limits<double>::quiet_NaN()); });
  Rejects([] {
    Json::Object({{"duplicate", Json(1)}, {"duplicate", Json(2)}});
  });
}
void TestDensePlanOptions() {
  Check(
      dense_plan::DispatchSupportJson(true, 15).Dump() ==
      "{\"backend\":\"owned_cuda\",\"tiles\":[16,32],\"num_samples_min\":1,"
      "\"num_samples_max\":15,\"cache\":true,\"lookahead\":true,"
      "\"lookahead_modes\":[false,true],\"lifetime\":\"preallocated_rotation\","
      "\"status\":\"compiled_planned_not_qualified\"}");
  Check(dense_plan::DispatchSupportJson(false, 7).Dump() == "null");
  Check(dense_plan::DispatchSupportJson(true, 16).Dump() == "null");
  Check(dense_plan::DispatchSupportJson(true, 0).Dump() == "null");
  mvs::PatchMatchOptions options;
  options.num_samples = 7;
  options.geom_consistency = false;
  const auto json = dense_plan::OptionsJson(options).Dump();
  Check(json.find("\"num_samples\":7") != std::string::npos);
  Check(json.find("\"geom_consistency\":false") != std::string::npos);
  Check(json.find("\"depth_min\":-1") != std::string::npos);
  Check(json.find("\"gpu_index\":\"-1\"") != std::string::npos);
}
void TestDensePlanConcurrency() {
  Check(dense_plan::ConcurrencySupportJson(true, 7).Dump() ==
        "{\"status\":\"compiled_supported_not_qualified\","
        "\"photo\":{\"full_bank\":1,\"cohort\":1},"
        "\"geo\":{\"full_bank\":5,\"cohort\":1}}");
  for (const auto& [cuda, samples] :
       std::vector<std::pair<bool, int>>{{false, 7}, {true, 0}, {true, 16}}) {
    Check(dense_plan::ConcurrencySupportJson(cuda, samples).Dump() == "null");
  }
  for (int samples : {1, 15}) {
    Check(dense_plan::ConcurrencySupportJson(true, samples).Dump() ==
          dense_plan::ConcurrencySupportJson(true, 7).Dump());
  }

  // Compare emitted ceilings against actual public dispatch validation, not
  // against a second copy of the policy helper. The golden above deliberately
  // makes a future policy expansion an explicit metadata-contract decision.
  std::vector<std::pair<std::string, dense_plan::Json>> phases;
  for (bool geo : {false, true}) {
    std::vector<std::pair<std::string, dense_plan::Json>> modes;
    for (bool cohort : {false, true}) {
      int maximum = 0;
      for (int n = 1; n <= 6; ++n) {
        bool supported = true;
        try {
          mvs::ValidateOptimizedStereoConcurrency(
              geo, std::to_string(n), !cohort, cohort);
        } catch (const std::invalid_argument&) {
          supported = false;
        }
        if (supported) {
          Check(n == maximum + 1);
          maximum = n;
          const auto receipt = mvs::OptimizedDispatchReceipt(
              geo ? "geo" : "photo", 16, 7, true, true, n);
          if (n == 1)
            Check(receipt.find("max_concurrent_solvers") == std::string::npos);
          else
            Check(receipt.find("\"max_concurrent_solvers\":" +
                               std::to_string(n)) != std::string::npos);
        }
      }
      modes.emplace_back(cohort ? "cohort" : "full_bank",
                         dense_plan::Json(maximum));
    }
    phases.emplace_back(geo ? "geo" : "photo", dense_plan::Json::Object(modes));
    Rejects([&] {
      mvs::ValidateOptimizedStereoConcurrency(geo, "2", false, false);
    });
    Rejects(
        [&] { mvs::ValidateOptimizedStereoConcurrency(geo, "2", true, true); });
  }
  Check(dense_plan::ConcurrencySupportJson(true, 7).Dump() ==
        dense_plan::Json::Object(
            {{"status", dense_plan::Json("compiled_supported_not_qualified")},
             {"photo", phases[0].second},
             {"geo", phases[1].second}})
            .Dump());
}
void TestDensePlanArguments() {
  std::vector<std::string> args = {"colmap",
                                   "--workspace_path",
                                   "workspace",
                                   "--source_cap",
                                   "20",
                                   "--num_samples",
                                   "7",
                                   "--gpu_index",
                                   "0",
                                   "--output_path",
                                   "plan.json"};
  const auto parse = [&] {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return ParseOptimizedDensePlanArguments(argv.size(), argv.data());
  };
  const auto parsed = parse();
  Check(parsed.source_cap == 20 && parsed.num_samples == 7);
  Check(!parsed.cache_size_gb && !parsed.random_seed);
  Check(parsed.owned_tex_lookahead);
  args.insert(args.end(), {"--owned_tex_lookahead", "false"});
  const auto off = parse();
  Check(!off.owned_tex_lookahead);
  mvs::PatchMatchOptions selected;
  dense_plan::ApplyLaunchPolicy(off, selected);
  Check(!selected.owned_tex_lookahead);
  Check(dense_plan::OptionsJson(selected).Dump().find(
            "\"owned_tex_lookahead\":false") != std::string::npos);
  args.back() = "true";
  dense_plan::ApplyLaunchPolicy(parse(), selected);
  Check(selected.owned_tex_lookahead);
  for (const char* bad : {"0", "1", "False", "TRUE", "invalid"}) {
    args.back() = bad;
    Rejects(parse);
  }
  args.back() = "false";
  args.insert(args.end(), {"--owned_tex_lookahead", "true"});
  Rejects(parse);
  args.resize(11);
  mvs::PatchMatchOptions defaults;
  const auto default_cache = defaults.cache_size;
  dense_plan::ApplyLaunchPolicy(parsed, defaults);
  Check(defaults.cache_size == default_cache && defaults.random_seed == -1);
  Check(defaults.owned_tex_lookahead);
  args.insert(args.end(), {"--cache_size_gb", "2.5", "--random_seed", "42"});
  const auto policy = parse();
  Check(policy.cache_size_gb == 2.5 && policy.random_seed == 42);
  dense_plan::ApplyLaunchPolicy(policy, defaults);
  const auto raw = dense_plan::OptionsJson(defaults).Dump();
  Check(raw.find("\"cache_size\":2.5") != std::string::npos);
  Check(raw.find("\"random_seed\":42") != std::string::npos);
  Check(defaults.depth_min == -1 && defaults.depth_max == -1);
  Check(defaults.sigma_spatial == -1);
  for (const char* bad :
       {"nan", "inf", "-inf", "0", "-1", "1garbage", "1e999"}) {
    args[12] = bad;
    Rejects(parse);
  }
  args[12] = "2.5";
  for (const char* bad : {"-2", "nan", "1.5", "2147483648", "3x"}) {
    args[14] = bad;
    Rejects(parse);
  }
  args[14] = "-1";
  Check(parse().random_seed == -1);
  args[14] = "0";
  Check(parse().random_seed == 0);
  args[14] = "2147483647";
  Check(parse().random_seed == 2147483647);
  args.resize(13);
  Check(parse().cache_size_gb == 2.5 && !parse().random_seed);
  args.resize(11);
  args.insert(args.end(), {"--random_seed", "0"});
  Check(!parse().cache_size_gb && parse().random_seed == 0);
  args.resize(11);
  args.insert(args.end(), {"--cache_size_gb", "2.5", "--cache_size_gb", "3"});
  Rejects(parse);
  args.resize(11);
  args.insert(args.end(), {"--random_seed", "0", "--random_seed", "1"});
  Rejects(parse);
  args.resize(11);
  const auto complete = args;
  args.erase(args.begin() + 1, args.begin() + 3);
  args.insert(args.end(), {"--cache_size_gb", "2.5"});
  Rejects(parse);
  args = complete;
  ValidateOptimizedDenseSourceCap("# native cfg\na.jpg\n__auto__, 20\n", 20);
  Rejects([] { ValidateOptimizedDenseSourceCap("a.jpg\n__auto__, 8\n", 20); });
  Rejects([] { ValidateOptimizedDenseSourceCap("a.jpg\nb.jpg,c.jpg\n", 20); });
  Rejects([] { ValidateOptimizedDenseSourceCap("a.jpg\n__all__\n", 20); });
  Rejects([] { ValidateOptimizedDenseSourceCap("a.jpg\n", 20); });
  args[6] = "20garbage";
  Rejects(parse);
  args[6] = "16";
  Rejects(parse);
  args[6] = "7";
  args.insert(args.end(), {"--log_path", "unexpected"});
  Rejects(parse);
}
void TestDensePlanExclusiveOutput() {
#if defined(__linux__)
  namespace fs = std::filesystem;
  auto dir =
      fs::temp_directory_path() / ("dense-plan-io-" + std::to_string(getpid()));
  Check(fs::create_directory(dir));
  try {
    const auto output = dir / "plan.json";
    dense_plan::WriteNewAtomic(output, "{\"ok\":true}\n");
    Rejects([&] { dense_plan::WriteNewAtomic(output, "replace"); });
    std::ifstream input(output);
    std::string text((std::istreambuf_iterator<char>(input)), {});
    Check(text == "{\"ok\":true}\n");
    fs::create_symlink("missing", dir / "symlink");
    Rejects([&] { dense_plan::WriteNewAtomic(dir / "symlink", "bad"); });
    Check(!fs::exists(dir / "missing"));
    Rejects(
        [&] { dense_plan::WriteNewAtomic(dir / "absent" / "plan", "bad"); });
    Check(std::distance(fs::directory_iterator(dir),
                        fs::directory_iterator()) == 2);
  } catch (...) {
    fs::remove_all(dir);
    throw;
  }
  fs::remove_all(dir);
#else
  Rejects([] { dense_plan::WriteNewAtomic("unsupported.json", "{}"); });
#endif
}
}  // namespace
}  // namespace colmap
#ifdef OPTIMIZED_DENSE_PLAN_STANDALONE_TEST
int main() {
  colmap::TestDensePlanJson();
  colmap::TestDensePlanArguments();
  colmap::TestDensePlanOptions();
  colmap::TestDensePlanConcurrency();
  colmap::TestDensePlanExclusiveOutput();
}
#else
TEST(OptimizedDensePlanIO, TypedJson) { colmap::TestDensePlanJson(); }
TEST(OptimizedDensePlanIO, TypedOptions) { colmap::TestDensePlanOptions(); }
TEST(OptimizedDensePlanIO, ConcurrencySupport) {
  colmap::TestDensePlanConcurrency();
}
TEST(OptimizedDensePlanIO, ArgumentsAndConfiguredCap) {
  colmap::TestDensePlanArguments();
}
TEST(OptimizedDensePlanIO, ExclusiveAtomicOutput) {
  colmap::TestDensePlanExclusiveOutput();
}
#endif
