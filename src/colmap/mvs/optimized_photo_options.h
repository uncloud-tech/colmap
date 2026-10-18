// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/patch_match_options.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <sstream>

namespace colmap::mvs {
// Options-only binding, deliberately NOT a complete owned-bank contract.
// Fixed exhaustive field order; changing this wire requires a version bump.
// Exact double bits, hex strings and classic-locale integers avoid ambiguity.
inline std::string OptimizedPhotoOptions(const PatchMatchOptions& options,
                                         const std::filesystem::path& workspace,
                                         const std::filesystem::path& config) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "COLMAP_OPTIMIZED_PHOTO_OPTIONS_V1\n";
#if defined(COLMAP_CUDA_ENABLED)
  out << "build_cuda_enabled 1\n";
#else
  out << "build_cuda_enabled 0\n";
#endif
#if defined(COLMAP_HIP_ENABLED)
  out << "build_hip_enabled 1\n";
#else
  out << "build_hip_enabled 0\n";
#endif
  const auto integer = [&](const char* key, int value) {
    out << "opt_int " << key << ' ' << value << '\n';
  };
  const auto boolean = [&](const char* key, bool value) {
    out << "opt_bool " << key << ' ' << (value ? 1 : 0) << '\n';
  };
  const auto real = [&](const char* key, double value) {
    static_assert(sizeof(double) == sizeof(uint64_t));
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    out << "opt_double " << key << ' ' << std::hex << std::setfill('0')
        << std::setw(16) << bits << std::dec << '\n';
  };
  const auto string = [&](const char* key, const std::string& value) {
    out << "opt_string " << key << ' ' << value.size() << ' '
        << owned::Hex(value) << '\n';
  };
  integer("max_image_size", options.max_image_size);
  string("gpu_index", options.gpu_index);
  real("depth_min", options.depth_min);
  real("depth_max", options.depth_max);
  integer("random_seed", options.random_seed);
  integer("window_radius", options.window_radius);
  integer("window_step", options.window_step);
  real("sigma_spatial", options.sigma_spatial);
  real("sigma_color", options.sigma_color);
  integer("num_samples", options.num_samples);
  boolean("owned_distinct_source_cache", options.owned_distinct_source_cache);
  boolean("owned_tex_lookahead", options.owned_tex_lookahead);
  real("ncc_sigma", options.ncc_sigma);
  real("min_triangulation_angle", options.min_triangulation_angle);
  real("incident_angle_sigma", options.incident_angle_sigma);
  integer("num_iterations", options.num_iterations);
  boolean("geom_consistency", options.geom_consistency);
  real("geom_consistency_regularizer", options.geom_consistency_regularizer);
  real("geom_consistency_max_cost", options.geom_consistency_max_cost);
  boolean("filter", options.filter);
  real("filter_min_ncc", options.filter_min_ncc);
  real("filter_min_triangulation_angle",
       options.filter_min_triangulation_angle);
  integer("filter_min_num_consistent", options.filter_min_num_consistent);
  real("filter_geom_consistency_max_cost",
       options.filter_geom_consistency_max_cost);
  boolean("write_consistency_graph", options.write_consistency_graph);
  boolean("allow_missing_files", options.allow_missing_files);
  integer("num_threads", options.num_threads);
  real("cache_size", options.cache_size);
  // Length/hex encoding binds paths without delimiter ambiguity.
  string("workspace", std::filesystem::canonical(workspace).string());
  string("config", std::filesystem::canonical(config).string());
  return out.str();
}
}  // namespace colmap::mvs
