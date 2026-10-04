// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define COLMAP_MASK_HOST_DEVICE __host__ __device__
#else
#define COLMAP_MASK_HOST_DEVICE
#endif

namespace colmap {
namespace mvs {

// Number of byte planes, avoiding overflow in the usual (sources + 7) / 8.
constexpr size_t PackedConsistencyMaskDepth(const size_t num_sources) {
  return num_sources / 8 + (num_sources % 8 != 0);
}

// Like GpuMatView<uint8_t>, but eight source flags share one byte at each
// pixel. Byte planes retain GpuMat's pitched, planar layout so its existing
// rotation and copy operations apply unchanged. The backing storage must be
// zero-initialized (including the unused bits of the last plane).
//
// Set is a non-atomic read/modify/write: one thread must own all source flags
// for a pixel. PatchMatch's column-wise sweep satisfies this requirement;
// distinct pixels never share a byte, even across pitched row boundaries.
struct PackedConsistencyMaskView {
  uint8_t* ptr;
  size_t pitch;
  size_t width;
  size_t height;
  size_t num_sources;

  COLMAP_MASK_HOST_DEVICE bool Get(const size_t row,
                                  const size_t col,
                                  const size_t source) const {
    return source < num_sources &&
           (ptr[pitch * ((source / 8) * height + row) + col] &
            (uint8_t{1} << (source % 8))) != 0;
  }

  COLMAP_MASK_HOST_DEVICE void Set(const size_t row,
                                  const size_t col,
                                  const size_t source,
                                  const uint8_t value) {
    if (source >= num_sources) {
      return;
    }
    uint8_t& byte = ptr[pitch * ((source / 8) * height + row) + col];
    const uint8_t bit = uint8_t{1} << (source % 8);
    byte = value ? (byte | bit) : (byte & ~bit);
  }
};

// Works with GpuMat and its lightweight view without requiring CUDA headers
// here, allowing the exact addressing logic to be tested on the host.
template <typename ByteMat>
PackedConsistencyMaskView MakePackedConsistencyMaskView(
    ByteMat& mat, const size_t num_sources) {
  return {mat.GetPtr(),
          mat.GetPitch(),
          mat.GetWidth(),
          mat.GetHeight(),
          num_sources};
}

// Expand directly into ConsistencyGraph's sparse (col, row, count, ids...)
// layout. Only real source flags are visited, never padding bits. The caller
// supplies a host copy in the original image orientation, after GpuMat::Rotate.
inline std::vector<int> UnpackConsistentImageIdxs(
    const PackedConsistencyMaskView mask,
    const std::vector<int>& src_image_idxs) {
  std::vector<int> consistent_image_idxs;
  std::vector<int> pixel_consistent_image_idxs;
  pixel_consistent_image_idxs.reserve(src_image_idxs.size());
  for (size_t r = 0; r < mask.height; ++r) {
    for (size_t c = 0; c < mask.width; ++c) {
      pixel_consistent_image_idxs.clear();
      for (size_t d = 0; d < src_image_idxs.size(); ++d) {
        if (mask.Get(r, c, d)) {
          pixel_consistent_image_idxs.push_back(src_image_idxs[d]);
        }
      }
      if (!pixel_consistent_image_idxs.empty()) {
        consistent_image_idxs.push_back(c);
        consistent_image_idxs.push_back(r);
        consistent_image_idxs.push_back(pixel_consistent_image_idxs.size());
        consistent_image_idxs.insert(consistent_image_idxs.end(),
                                     pixel_consistent_image_idxs.begin(),
                                     pixel_consistent_image_idxs.end());
      }
    }
  }
  return consistent_image_idxs;
}

}  // namespace mvs
}  // namespace colmap

#undef COLMAP_MASK_HOST_DEVICE
