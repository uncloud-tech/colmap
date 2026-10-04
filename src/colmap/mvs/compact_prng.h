// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define COLMAP_PRNG_HD __host__ __device__
#define COLMAP_PRNG_DEVICE __device__
#else
#define COLMAP_PRNG_HD
#define COLMAP_PRNG_DEVICE
#endif

namespace colmap {
namespace mvs {

// Benchmark-only, default off. Never compact launch-geometry seeded states.
inline bool UseCompactPrng(bool geometric, int seed, const char* opt_in) {
  return geometric && seed == 0 && opt_in && opt_in[0] == '1' &&
         opt_in[1] == '\0';
}

// Positive dimensions. All states stay in ORIGINAL image coordinates. Corners
// have one owner, so writes in one orientation survive into the next edge.
struct PrngPerimeter {
  size_t width;
  size_t height;
  struct Pixel {
    size_t row;
    size_t col;
  };

  COLMAP_PRNG_HD size_t Size() const {
    if (width == 1) return height;
    if (height == 1) return width;
    return 2 * width + 2 * (height - 2);
  }

  COLMAP_PRNG_HD size_t Index(size_t row, size_t col) const {
    if (width == 1) return row;
    if (height == 1 || row == 0) return col;
    if (row == height - 1) return width + col;
    return 2 * width + (col == 0 ? 0 : height - 2) + row - 1;
  }

  COLMAP_PRNG_HD Pixel Coordinate(size_t index) const {
    if (width == 1) return {index, 0};
    if (index < width) return {0, index};
    index -= width;
    if (index < width) return {height - 1, index};
    index -= width;
    if (index < height - 2) return {index + 1, 0};
    return {index - (height - 2) + 1, width - 1};
  }

  // Inverse of (r,c) -> (W-1-c,r), the CCW cuda_rotate.h mapping.
  // Row zero visits top/right/bottom/left; the last two are reversed.
  COLMAP_PRNG_HD Pixel RowZero(int orientation, size_t col) const {
    switch (orientation) {
      case 0: return {0, col};
      case 1: return {col, width - 1};
      case 2: return {height - 1, width - 1 - col};
      default: return {height - 1 - col, 0};
    }
  }
};

// The compact path supports row zero only, exactly the sweep's access pattern.
// The null compact pointer delegates unchanged to the ordinary full-map view.
// Template parameters keep this same access implementation host-testable.
template <typename State, typename FullView>
struct PrngRowView {
  FullView full;
  State* compact;
  PrngPerimeter perimeter;
  int orientation;

  COLMAP_PRNG_DEVICE State Get(size_t row, size_t col) const {
    if (!compact) return full.Get(row, col);
    const auto p = perimeter.RowZero(orientation, col);
    return compact[perimeter.Index(p.row, p.col)];
  }

  COLMAP_PRNG_DEVICE void Set(size_t row, size_t col, State value) const {
    if (!compact) {
      full.Set(row, col, value);
      return;
    }
    const auto p = perimeter.RowZero(orientation, col);
    compact[perimeter.Index(p.row, p.col)] = value;
  }
};

}  // namespace mvs
}  // namespace colmap
#undef COLMAP_PRNG_HD
#undef COLMAP_PRNG_DEVICE
