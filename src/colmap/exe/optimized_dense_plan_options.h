// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/exe/optimized_dense_plan.h"
#include "colmap/exe/optimized_dense_plan_io.h"
#include "colmap/mvs/optimized_dispatch.h"
#include "colmap/mvs/patch_match_options.h"

namespace colmap::dense_plan {
inline void ApplyLaunchPolicy(const OptimizedDensePlanArguments& args,
                              mvs::PatchMatchOptions& options) {
  if (args.cache_size_gb) options.cache_size = *args.cache_size_gb;
  if (args.random_seed) options.random_seed = *args.random_seed;
  options.owned_tex_lookahead = args.owned_tex_lookahead;
}

// Compiled source capability, not execution telemetry or qualification.
inline Json DispatchSupportJson(bool cuda, int samples) {
  std::vector<Json> tiles;
  for (int tile : {16, 32}) {
    if (mvs::OptimizedStereoCacheSupported(cuda, tile, samples))
      tiles.emplace_back(tile);
  }
  if (tiles.empty()) return Json();
  return Json::Object(
      {{"backend", Json("owned_cuda")},
       {"tiles", Json::Array(tiles)},
       {"num_samples_min", Json(1)},
       {"num_samples_max", Json(mvs::owned::OwnedSourceCache<16>::kMaxSamples)},
       {"cache", Json(true)},
       {"lookahead", Json(true)},
       {"lookahead_modes", Json::Array({Json(false), Json(true)})},
       {"lifetime", Json("preallocated_rotation")},
       {"status", Json("compiled_planned_not_qualified")}});
}

inline Json OptionsJson(const mvs::PatchMatchOptions& options) {
  std::vector<std::pair<std::string, Json>> fields;
#define DENSE_OPTION(name) fields.emplace_back(#name, Json(options.name))
  DENSE_OPTION(max_image_size);
  DENSE_OPTION(gpu_index);
  DENSE_OPTION(depth_min);
  DENSE_OPTION(depth_max);
  DENSE_OPTION(random_seed);
  DENSE_OPTION(window_radius);
  DENSE_OPTION(window_step);
  DENSE_OPTION(sigma_spatial);
  DENSE_OPTION(sigma_color);
  DENSE_OPTION(num_samples);
  DENSE_OPTION(owned_distinct_source_cache);
  DENSE_OPTION(owned_tex_lookahead);
  DENSE_OPTION(ncc_sigma);
  DENSE_OPTION(min_triangulation_angle);
  DENSE_OPTION(incident_angle_sigma);
  DENSE_OPTION(num_iterations);
  DENSE_OPTION(geom_consistency);
  DENSE_OPTION(geom_consistency_regularizer);
  DENSE_OPTION(geom_consistency_max_cost);
  DENSE_OPTION(filter);
  DENSE_OPTION(filter_min_ncc);
  DENSE_OPTION(filter_min_triangulation_angle);
  DENSE_OPTION(filter_min_num_consistent);
  DENSE_OPTION(filter_geom_consistency_max_cost);
  DENSE_OPTION(write_consistency_graph);
  DENSE_OPTION(allow_missing_files);
  DENSE_OPTION(num_threads);
  DENSE_OPTION(cache_size);
#undef DENSE_OPTION
  return Json::Object(fields);
}
}  // namespace colmap::dense_plan
