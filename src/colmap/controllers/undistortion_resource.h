// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace colmap {

inline uint64_t UndistortionCheckedAdd(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("undistortion resource addition overflow");
  return a + b;
}

inline uint64_t UndistortionCheckedMultiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::overflow_error("undistortion resource multiplication overflow");
  return a * b;
}

// Known RGB8 vector payload only. Bitmap's int multiplications constrain each
// allocation. Indirect warp (and resized spherical clone) keeps source, full
// source-size target, and resized target simultaneously live. Decoder/filter/
// writer scratch, metadata, reconstruction copies and allocator costs are
// explicit policy, NOT a measured bound. No mapped-file credit or decode floor.
inline uint64_t UndistortionJobPayload(uint64_t width, uint64_t height,
                                      uint64_t target_width,
                                      uint64_t target_height,
                                      bool direct, bool copy) {
  const auto pixels = [](uint64_t w, uint64_t h) {
    if (!w || !h) throw std::invalid_argument("invalid undistortion dimensions");
    const auto bytes = UndistortionCheckedMultiply(
        UndistortionCheckedMultiply(w, h), 3);
    if (bytes > static_cast<uint64_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("undistortion bitmap exceeds int payload range");
    return bytes;
  };
  const auto source = pixels(width, height);
  const auto target = pixels(target_width, target_height);
  if (copy) return 0;
  return UndistortionCheckedAdd(
      source, direct ? target : UndistortionCheckedAdd(source, target));
}

// Same single aggregate ceil(3%) and outside reserve as fusion. Frozen native
// FREE/cgroup capacity, lowered by a user cap, never re-subtract live allocations.
class UndistortionLogicalBudget {
 public:
  UndistortionLogicalBudget(uint64_t capacity, uint64_t policy, uint64_t reserve)
      : capacity_(capacity), policy_(policy), reserve_(reserve) {}

  bool Fits(uint64_t payload) const {
    const auto raw = UndistortionCheckedAdd(payload, policy_);
    const auto margin = UndistortionCheckedAdd(
        UndistortionCheckedMultiply(raw / 100, 3), (raw % 100 * 3 + 99) / 100);
    return UndistortionCheckedAdd(UndistortionCheckedAdd(raw, margin), reserve_)
           <= capacity_;
  }

  // Largest ordered prefix within the thread ceiling. No sorting, resizing,
  // unbudgeted fallback or speculative decode. Batches drain before the next.
  size_t BatchSize(const std::vector<uint64_t>& jobs, size_t begin,
                   size_t threads) const {
    if (!threads) throw std::invalid_argument("undistortion needs workers");
    uint64_t bytes = 0;
    size_t count = 0;
    while (begin + count < jobs.size() && count < threads) {
      const auto next = UndistortionCheckedAdd(bytes, jobs[begin + count]);
      if (!Fits(next)) break;
      bytes = next;
      ++count;
    }
    if (!count && begin < jobs.size())
      throw std::runtime_error("undistortion logical host budget exhausted before decode (not a physical-fit guarantee)");
    return count;
  }

 private:
  uint64_t capacity_, policy_, reserve_;
};

}  // namespace colmap
