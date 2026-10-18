// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

namespace colmap::mvs {
// Shared by the all-input preflight and the per-image legacy skip path.
inline bool FusionInputsAvailable(const std::string& image_name,
                                  const std::filesystem::path& bitmap,
                                  const std::filesystem::path& depth,
                                  const std::filesystem::path& normal,
                                  bool require_complete_inputs) {
  for (const auto& path : {bitmap, depth, normal}) {
    if (!std::filesystem::is_regular_file(path)) {
      if (require_complete_inputs)
        throw std::invalid_argument("Missing fusion dependency for " +
                                    image_name + ": " + path.string());
      return false;
    }
  }
  return true;
}
}  // namespace colmap::mvs
