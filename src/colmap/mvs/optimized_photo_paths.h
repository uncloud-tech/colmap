// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_pair_validation.h"

#include <filesystem>
#include <vector>

namespace colmap::mvs {
// Validate before the controller writes either map. Inputs and output parents
// must remain immutable throughout the invocation (not a race-safe sandbox).
inline std::vector<std::filesystem::path> ValidateOptimizedPhotoOutputs(
    const std::filesystem::path& workspace, const std::filesystem::path& name) {
  namespace fs = std::filesystem;
  const auto root = fs::canonical(workspace);
  if (name.empty() || name.is_absolute())
    throw std::invalid_argument("invalid PHOTO reference name");
  for (const auto& part : name)
    if (part == ".." || part == ".")
      throw std::invalid_argument("invalid PHOTO reference name");
  std::vector<fs::path> outputs;
  for (const auto* folder : {"depth_maps", "normal_maps"}) {
    const auto dir = fs::weakly_canonical(root / "stereo" / folder);
    if (!owned::PathContains(root / "stereo", dir))
      throw std::invalid_argument("PHOTO output root escapes workspace");
    const auto path = dir / (name.string() + ".photometric.bin");
    if (fs::exists(path) || fs::is_symlink(fs::symlink_status(path)))
      throw std::invalid_argument("PHOTO output must be fresh");
    const auto resolved = fs::weakly_canonical(path);
    // Resolve the COMPLETE name, including nested existing parent symlinks.
    // Each map belongs to its own surface, not merely somewhere in stereo.
    if (!owned::PathContains(root / "stereo" / folder, resolved))
      throw std::invalid_argument("PHOTO final output escapes map root");
    outputs.push_back(resolved);
  }
  if (owned::PathsAlias(outputs[0], outputs[1]))
    throw std::invalid_argument("PHOTO outputs alias");
  return outputs;
}
}  // namespace colmap::mvs
