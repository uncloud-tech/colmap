// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_pair_validation.h"

#include <OpenImageIO/imageio.h>

#include <filesystem>
#include <stdexcept>

namespace colmap::mvs::owned {

// Header-only preflight using the SAME orientation/alpha configuration as
// Bitmap::Read. Inputs must remain immutable from this check through decode.
// With max_image_size=-1 and image_as_rgb=false, retained bytes are exactly W*H.
// Codec scratch allocations are NOT bounded by this logical retention check.
inline uint64_t ReadPairBitmapPixels(const std::filesystem::path& path,
                                     uint64_t expected_width,
                                     uint64_t expected_height) {
  OIIO::ImageSpec config;
  config["oiio:reorient"] = 0;
  config["oiio:UnassociatedAlpha"] = 1;
  auto input = OIIO::ImageInput::open(path.u8string(), &config);
  if (!input) {
    throw std::runtime_error("image header open failed: " + OIIO::geterror());
  }
  const auto& spec = input->spec();
  const int width = spec.width;
  const int height = spec.height;
  const int depth = spec.depth;
  const int channels = spec.nchannels;
  const bool deep = spec.deep;
  if (!input->close()) {
    throw std::runtime_error("image header close failed: " + input->geterror());
  }
  if (width <= 0 || height <= 0 || depth != 1 || deep || channels < 1 ||
      channels > 4 || static_cast<uint64_t>(width) != expected_width ||
      static_cast<uint64_t>(height) != expected_height) {
    throw std::invalid_argument("image header/model mismatch or unsupported layout: " + path.string());
  }
  return CheckedMultiply(static_cast<uint64_t>(width),
                         static_cast<uint64_t>(height));
}

}  // namespace colmap::mvs::owned
