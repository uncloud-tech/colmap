// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/fusion_dependencies.h"

#include <array>
#include <cassert>
#include <chrono>
#include <fstream>

int main() {
  namespace fs = std::filesystem;
  using colmap::mvs::FusionInputsAvailable;
  const auto root =
      fs::temp_directory_path() /
      ("colmap-fusion-deps-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(root);
  const std::array<fs::path, 3> paths{
      root / "image.jpg", root / "depth.bin", root / "normal.bin"};
  for (const auto& p : paths) std::ofstream(p) << "input";
  assert(
      FusionInputsAvailable("image.jpg", paths[0], paths[1], paths[2], true));
  for (size_t i = 0; i < paths.size(); ++i) {
    fs::remove(paths[i]);
    assert(!FusionInputsAvailable(
        "image.jpg", paths[0], paths[1], paths[2], false));
    bool rejected = false;
    try {
      FusionInputsAvailable("image.jpg", paths[0], paths[1], paths[2], true);
    } catch (const std::invalid_argument& e) {
      rejected =
          std::string(e.what()).find(paths[i].string()) != std::string::npos;
    }
    assert(rejected);
    std::ofstream(paths[i]) << "restored";
  }
  fs::remove_all(root);
}
