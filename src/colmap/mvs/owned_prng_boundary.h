// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_selection_scratch.h"

#if defined(__CUDACC__)
#define COLMAP_PRNG_BOUNDARY_HD __host__ __device__
#else
#define COLMAP_PRNG_BOUNDARY_HD
#endif

namespace colmap::mvs::owned {

// Canonical perimeter with each corner stored ONCE. Only row zero survives as
// a sweep's RNG state; the four CCW orientations visit these four boundaries.
// Initialization must still consume the original full-image states first.
class PrngBoundaryLayout {
 public:
  PrngBoundaryLayout(size_t width, size_t height)
      : width_(width), height_(height) {
    detail::CheckDimension(width);
    detail::CheckDimension(height);
    // Count() is a GpuMat width; keep its byte pitch within the int ABI too.
    if (Count() > static_cast<size_t>(INT_MAX) / 48) {
      throw std::overflow_error("PRNG boundary pitch exceeds kernel range");
    }
  }

  COLMAP_PRNG_BOUNDARY_HD size_t Count() const {
    return height_ == 1
               ? width_
               : (width_ == 1 ? height_ : 2 * width_ + 2 * height_ - 4);
  }

  // Call only for canonical boundary coordinates.
  COLMAP_PRNG_BOUNDARY_HD size_t Index(size_t row, size_t col) const {
    if (height_ == 1) return col;
    if (width_ == 1) return row;
    if (row == 0) return col;
    if (row == height_ - 1) return width_ + col;
    return 2 * width_ + 2 * (row - 1) + (col == width_ - 1);
  }

  COLMAP_PRNG_BOUNDARY_HD size_t SweepIndex(unsigned orientation,
                                            size_t col) const {
    switch (orientation) {
      case 0:
        return Index(0, col);
      case 1:
        return Index(col, width_ - 1);
      case 2:
        return Index(height_ - 1, width_ - 1 - col);
      default:
        return Index(height_ - 1 - col, 0);
    }
  }

 private:
  size_t width_, height_;
};

// Actual retained allocation inventory; the full-map initialization temporary
// is NOT a retained rotation owner. Legacy forecast inventory stays unchanged.
inline OwnedRotationLayout BuildOwnedBoundaryRotationLayout(
    size_t width, size_t height, size_t sources, bool retain_mask = true) {
  auto layout =
      BuildOwnedScratchRotationLayout(width, height, sources, retain_mask);
  const PrngBoundaryLayout boundary(width, height);
  for (auto it = layout.slots.begin(); it != layout.slots.end();) {
    if (it->role != RotationRole::kPrng) {
      ++it;
      continue;
    }
    layout.logical_bytes -= it->requested_span_bytes;
    if (it->parity == 1) {
      it = layout.slots.erase(it);
    } else {
      it->width = boundary.Count();
      it->height = 1;
      it->requested_span_bytes = detail::CheckedMultiply(boundary.Count(), 48);
      layout.logical_bytes =
          detail::CheckedAdd(layout.logical_bytes, it->requested_span_bytes);
      ++it;
    }
  }
  return layout;
}

// Deliberately mirrors the sweep's existing Get/Set seam, not a full-image
// view. The row argument is always zero. Canonical indexing aliases corner
// updates across adjacent orientations and keeps all later iterations on the
// same state.
template <class State>
struct PrngBoundaryView {
  State* states;
  PrngBoundaryLayout layout;
  unsigned orientation;
  COLMAP_PRNG_BOUNDARY_HD State Get(size_t, size_t col) const {
    return states[layout.SweepIndex(orientation, col)];
  }
  COLMAP_PRNG_BOUNDARY_HD void Set(size_t, size_t col, State state) const {
    states[layout.SweepIndex(orientation, col)] = state;
  }
};

}  // namespace colmap::mvs::owned

#undef COLMAP_PRNG_BOUNDARY_HD
