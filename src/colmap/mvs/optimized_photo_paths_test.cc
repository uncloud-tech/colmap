// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/optimized_photo_paths.h"

#include <cassert>
#include <chrono>
#include <fstream>

int main() {
  namespace fs = std::filesystem;
  using colmap::mvs::ValidateOptimizedPhotoOutputs;
  const auto temp =
      fs::temp_directory_path() /
      ("colmap-photo-paths-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto root = temp / "workspace";
  fs::create_directories(root / "stereo/depth_maps");
  fs::create_directories(root / "stereo/normal_maps");
  fs::create_directories(temp / "outside");
  const auto rejects = [&](const fs::path& name) {
    bool rejected = false;
    try {
      ValidateOptimizedPhotoOutputs(root, name);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  };
  const auto paths = ValidateOptimizedPhotoOutputs(root, "sub/image.jpg");
  assert(paths[0] == root / "stereo/depth_maps/sub/image.jpg.photometric.bin");
  assert(paths[1] == root / "stereo/normal_maps/sub/image.jpg.photometric.bin");
  rejects("../image.jpg");
  rejects("/image.jpg");
  for (const auto* folder : {"depth_maps", "normal_maps"}) {
    const auto nested = root / "stereo" / folder / "sub";
    fs::create_directory_symlink(temp / "outside", nested);
    rejects("sub/image.jpg");
    fs::remove(nested);
  }
  // Cross-map directory aliases must refuse too, even if still in stereo.
  fs::create_directory_symlink(root / "stereo/normal_maps",
                               root / "stereo/depth_maps/sub");
  rejects("sub/image.jpg");
  fs::remove(root / "stereo/depth_maps/sub");
  const auto output = root / "stereo/depth_maps/image.jpg.photometric.bin";
  std::ofstream(output) << "existing";
  rejects("image.jpg");
  fs::remove(output);
  fs::create_symlink(temp / "missing", output);
  rejects("image.jpg");
  fs::remove(output);
  fs::remove_all(temp);
}
