// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/optimized_dispatch.h"

#include <charconv>
#include <cmath>
#include <filesystem>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace colmap {
// Read-only metadata command, not execution/admission/qualification.
// Five required argv retain native defaults. Optional --cache_size_gb must be
// positive finite and sets raw cache_size; --random_seed is an integer >=-1.
// Absence preserves native default cache_size and random_seed (-1).
// Optional --owned_tex_lookahead true|false defaults to true for compatibility.
// Schema v1: {schema_version,workspace_path,source_cap,num_samples,
// build:{revision,version,cuda_compiled,native_graph_supported},
// dispatch_support: null or {backend,tiles,num_samples_min,num_samples_max,
// cache,lookahead,lookahead_modes:[false,true],lifetime,
// status:"compiled_planned_not_qualified"},
// concurrency_support: null or {status:"compiled_supported_not_qualified",
// photo:{full_bank:1,cohort:1},geo:{full_bank:5,cohort:1}},
// Values are phase/mode concurrency ceilings derived from dispatch policy,
// NOT selected defaults, admission or workload/hardware qualification. N>1
// requires an exclusive full-bank forecast with exact active_jobs and resource
// checks; without that forecast only N1 is supported. Unsupported CUDA/sample
// dispatch emits null (no capability), never invented singleton support. Builds
// without native graph support still refuse the command without a plan file.
// images:[{index,name,path,width,height}],
// phases:{photo|geo:{source_resolution_history,requested_options,
// jobs:[{reference,sources,depth_min,depth_max,options,
// effective_options_wire_v1,outputs:{depth,normal}}]}},
// requested_options retains RAW invocation auto-depth/sigma (-1), with
// cache=true, selected lookahead=false|true, samples1..15,
// PHOTO filter=false / GEO filter=true.
// job.options contains EFFECTIVE resolved facts; effective_options_wire_v1 is
// INFORMATIONAL, never a launch binding. Actual per-reference native PHOTO
// options export (including its actual cfg path) remains admission authority.
// Never inject resolved historical depth/sigma values into native argv.
// input_files:[{path,sha256}],
// future_photo_dependencies:[{reference,kind,width,height,channels,path}]}.
// Indices address images in native model order. Sources retain native order.
// PHOTO and GEO use fresh controllers, not stock Run's repeated transformation.
// Future dependencies have NO invented identity/hash. GEO is a planned graph;
// complete actual PHOTO maps + exact native GEO bank revalidation are
// mandatory.
int RunOptimizedDensePlan(int argc, char** argv);

struct OptimizedDensePlanArguments {
  std::filesystem::path workspace_path;
  std::filesystem::path output_path;
  int source_cap;
  int num_samples;
  int gpu_index;
  std::optional<double> cache_size_gb;
  std::optional<int> random_seed;
  bool owned_tex_lookahead = true;
};

inline OptimizedDensePlanArguments ParseOptimizedDensePlanArguments(
    int argc, char** argv) {
  std::map<std::string, std::string> values;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key != "--workspace_path" && key != "--output_path" &&
        key != "--source_cap" && key != "--num_samples" &&
        key != "--gpu_index" && key != "--cache_size_gb" &&
        key != "--random_seed" && key != "--owned_tex_lookahead")
      throw std::invalid_argument("unsupported optimized_dense_plan option: " +
                                  key);
    if (++i == argc || std::string(argv[i]).empty() ||
        !values.emplace(key, argv[i]).second)
      throw std::invalid_argument("missing/duplicate argument: " + key);
  }
  if (!values.count("--workspace_path") || !values.count("--output_path") ||
      !values.count("--source_cap") || !values.count("--num_samples") ||
      !values.count("--gpu_index"))
    throw std::invalid_argument(
        "required: --workspace_path W --source_cap 8|20 "
        "--num_samples N --gpu_index I --output_path NEW");
  const auto integer = [&](const char* key) {
    const auto& text = values.at(key);
    int value;
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
      throw std::invalid_argument(std::string("invalid integer: ") + key);
    return value;
  };
  OptimizedDensePlanArguments args{values.at("--workspace_path"),
                                   values.at("--output_path"),
                                   integer("--source_cap"),
                                   integer("--num_samples"),
                                   integer("--gpu_index"),
                                   {},
                                   {}};
  if (values.count("--cache_size_gb")) {
    const auto& text = values.at("--cache_size_gb");
    size_t consumed = 0;
    const double cache = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(cache) || cache <= 0)
      throw std::invalid_argument("cache_size_gb must be positive and finite");
    args.cache_size_gb = cache;
  }
  if (values.count("--random_seed")) {
    args.random_seed = integer("--random_seed");
    if (*args.random_seed < -1)
      throw std::invalid_argument("random_seed must be -1 or nonnegative");
  }
  if (values.count("--owned_tex_lookahead")) {
    const auto& value = values.at("--owned_tex_lookahead");
    if (value != "true" && value != "false")
      throw std::invalid_argument("owned_tex_lookahead must be true or false");
    args.owned_tex_lookahead = value == "true";
  }
  if ((args.source_cap != 8 && args.source_cap != 20) ||
      (!mvs::OptimizedStereoCacheSupported(true, 16, args.num_samples) &&
       !mvs::OptimizedStereoCacheSupported(true, 32, args.num_samples)) ||
      args.gpu_index < 0)
    throw std::invalid_argument(
        "source_cap must be 8 or 20; num_samples=1..15; gpu_index>=0");
  return args;
}

// A cap is not the actual number of overlapping views, nor num_samples.
// Explicit lists / __all__ cannot establish the configured maximum and refuse.
inline void ValidateOptimizedDenseSourceCap(const std::string& config,
                                            int cap) {
  std::istringstream input(config);
  std::string line;
  bool reference = true;
  size_t jobs = 0;
  while (std::getline(input, line)) {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    line = line.substr(first, line.find_last_not_of(" \t\r") - first + 1);
    if (!reference) {
      const auto comma = line.find(',');
      auto selector = line.substr(0, comma);
      selector.erase(selector.find_last_not_of(" \t") + 1);
      if (comma == std::string::npos || selector != "__auto__")
        throw std::invalid_argument(
            "unknown configured source cap: require __auto__, cap");
      const auto count = line.substr(comma + 1);
      const auto start = count.find_first_not_of(" \t");
      if (start == std::string::npos ||
          count.substr(start) != std::to_string(cap))
        throw std::invalid_argument(
            "configured source cap differs from --source_cap");
      ++jobs;
    }
    reference = !reference;
  }
  if (!reference || !jobs)
    throw std::invalid_argument("empty or incomplete patch-match.cfg");
}
}  // namespace colmap
