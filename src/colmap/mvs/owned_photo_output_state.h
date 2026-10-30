// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <stdexcept>

namespace colmap::mvs::owned {
// Publication state only, not a scheduler. A throwing run remains ineligible.
class PhotoOutputState {
 public:
  void BeginRun() { completed_ = false; }
  void FinishRun() { completed_ = true; }
  void RequireSnapshot(bool geometry,
                       size_t width,
                       size_t height,
                       int rotation) const {
    if (!completed_ || geometry || width == 0 || height == 0 || rotation != 0) {
      throw std::logic_error(
          "PHOTO snapshot requires successful unrotated PHOTO output");
    }
  }

 private:
  bool completed_ = false;
};
}  // namespace colmap::mvs::owned
