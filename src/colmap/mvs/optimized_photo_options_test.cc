// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/optimized_photo_options.h"

#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>

int main() {
  namespace fs = std::filesystem;
  using namespace colmap::mvs;
  const auto root =
      fs::temp_directory_path() /
      ("colmap-photo-options-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(root);
  const auto config = root / "photo.cfg";
  std::ofstream(config) << "reference.jpg\nsource.jpg\n";
  PatchMatchOptions options;
  options.depth_min = 1.5;
  options.depth_max = -0.0;
  options.gpu_index = "0,1\n2";
  const auto bytes = OptimizedPhotoOptions(options, root, config);
  assert(bytes.find("COLMAP_OPTIMIZED_PHOTO_OPTIONS_V1\n") == 0);
  assert(bytes.find("opt_double depth_min 3ff8000000000000\n") !=
         std::string::npos);
  assert(bytes.find("opt_double depth_max 8000000000000000\n") !=
         std::string::npos);
  assert(bytes.find("opt_string gpu_index 5 302c310a32\n") !=
         std::string::npos);
  assert(bytes == OptimizedPhotoOptions(options, root / ".", config));
  // Preserve the accepted default V1 wire; only opt-in execution adds bindings.
  assert(bytes.find("owned_cuda_graphs") == std::string::npos);
  assert(bytes.find("owned_async_pool") == std::string::npos);
  auto graph = options;
  graph.owned_cuda_graphs = true;
  const auto graph_bytes = OptimizedPhotoOptions(graph, root, config);
  assert(graph_bytes != bytes);
  assert(graph_bytes.find("opt_bool owned_cuda_graphs 1\n") !=
         std::string::npos);
  // Every invocation option must affect the actual production binding.
  const std::function<void(PatchMatchOptions&)> changes[] = {
      [](auto& o) { ++o.max_image_size; },
      [](auto& o) { o.gpu_index = "1"; },
      [](auto& o) { ++o.depth_min; },
      [](auto& o) { o.depth_max = 0.0; },
      [](auto& o) { ++o.random_seed; },
      [](auto& o) { ++o.window_radius; },
      [](auto& o) { ++o.window_step; },
      [](auto& o) { ++o.sigma_spatial; },
      [](auto& o) { ++o.sigma_color; },
      [](auto& o) { ++o.num_samples; },
      [](auto& o) { o.owned_cuda_graphs = true; },
      [](auto& o) { o.owned_async_pool = true; },
      [](auto& o) {
        o.owned_distinct_source_cache = !o.owned_distinct_source_cache;
      },
      [](auto& o) { o.owned_tex_lookahead = !o.owned_tex_lookahead; },
      [](auto& o) { ++o.ncc_sigma; },
      [](auto& o) { ++o.min_triangulation_angle; },
      [](auto& o) { ++o.incident_angle_sigma; },
      [](auto& o) { ++o.num_iterations; },
      [](auto& o) { o.geom_consistency = !o.geom_consistency; },
      [](auto& o) { ++o.geom_consistency_regularizer; },
      [](auto& o) { ++o.geom_consistency_max_cost; },
      [](auto& o) { o.filter = !o.filter; },
      [](auto& o) { ++o.filter_min_ncc; },
      [](auto& o) { ++o.filter_min_triangulation_angle; },
      [](auto& o) { ++o.filter_min_num_consistent; },
      [](auto& o) { ++o.filter_geom_consistency_max_cost; },
      [](auto& o) { o.write_consistency_graph = !o.write_consistency_graph; },
      [](auto& o) { o.allow_missing_files = !o.allow_missing_files; },
      [](auto& o) { ++o.num_threads; },
      [](auto& o) { ++o.cache_size; }};
  for (const auto& change : changes) {
    auto changed = options;
    change(changed);
    assert(bytes != OptimizedPhotoOptions(changed, root, config));
  }
  std::ofstream(root / "other.cfg") << "other";
  assert(bytes != OptimizedPhotoOptions(options, root, root / "other.cfg"));
  fs::remove_all(root);
}
