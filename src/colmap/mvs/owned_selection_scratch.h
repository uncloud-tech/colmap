// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_rotation_storage.h"

namespace colmap::mvs::owned {

// Actual allocation inventory only. Keep the legacy layout used by admission
// forecasts conservative until the candidate has separate physical
// qualification.
inline OwnedRotationLayout BuildOwnedScratchRotationLayout(
    size_t width, size_t height, size_t sources, bool retain_mask = true) {
  auto layout = BuildOwnedRotationLayout(width, height, sources);
  for (auto it = layout.slot_layouts.begin();
       it != layout.slot_layouts.end();) {
    if ((it->role == RotationRole::kSelectionCurrent && it->parity == 1) ||
        (it->role == RotationRole::kMask &&
         (!retain_mask || it->parity == 0))) {
      layout.logical_bytes -= it->requested_span_bytes;
      it = layout.slot_layouts.erase(it);
    } else {
      ++it;
    }
  }
  return layout;
}

// One uninitialized packed float volume, reinterpreted only after the preceding
// sweep's rotation has completed. Storage allocates Bytes() bytes and owns
// them; the enclosing solver MUST drain its stream before destruction (also on
// failure). No initialization, allocation, or completion occurs when selecting
// a view. Views use GpuMatView's slice-major layout, with no row/slice padding
// to read.
template <class Storage>
class OwnedSelectionScratch {
 public:
  OwnedSelectionScratch(size_t width, size_t height, size_t sources)
      : width_(width),
        height_(height),
        sources_(sources),
        bytes_(CheckedBytes(width, height, sources)),
        storage_(bytes_) {}
  OwnedSelectionScratch(const OwnedSelectionScratch&) = delete;
  OwnedSelectionScratch& operator=(const OwnedSelectionScratch&) = delete;

  size_t Bytes() const { return bytes_; }

  template <class ViewType>
  ViewType View(unsigned parity) {
    if (parity > 1) throw std::invalid_argument("Invalid selection parity");
    const size_t width = parity ? height_ : width_;
    const size_t height = parity ? width_ : height_;
    return {storage_.GetPtr(), width * sizeof(float), width, height, sources_};
  }

 private:
  static size_t CheckedBytes(size_t width, size_t height, size_t sources) {
    detail::CheckDimension(width);
    detail::CheckDimension(height);
    detail::CheckDimension(sources);
    // CudaRotate's row-pitch ABI is int, in both orientations.
    if (std::max(width, height) >
        static_cast<size_t>(INT_MAX) / sizeof(float)) {
      throw std::overflow_error("Selection scratch pitch exceeds kernel range");
    }
    return detail::CheckedMultiply(
        detail::CheckedMultiply(detail::CheckedMultiply(width, height),
                                sources),
        sizeof(float));
  }

  const size_t width_, height_, sources_, bytes_;
  Storage storage_;
};

}  // namespace colmap::mvs::owned
