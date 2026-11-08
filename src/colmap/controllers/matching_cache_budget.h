// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace colmap {
inline uint64_t MatchingAdd(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("matching payload addition overflow");
  return a + b;
}
inline uint64_t MatchingMultiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::overflow_error("matching payload multiplication overflow");
  return a * b;
}
inline uint64_t MatchingMargined(uint64_t known,
                                 uint64_t policy,
                                 uint64_t reserve) {
  const auto raw = MatchingAdd(known, policy);
  return MatchingAdd(MatchingAdd(raw,
                                 MatchingAdd(MatchingMultiply(raw / 100, 3),
                                             (raw % 100 * 3 + 99) / 100)),
                     reserve);
}

// Independent LRUs need independent largest-N sums: their members need not
// agree. Values include known build/BLOB overlap, conservatively even on hits.
struct MatchingCacheInventory {
  std::vector<uint64_t> keypoints, descriptors, indexes;

  static uint64_t LargestSum(std::vector<uint64_t> bytes, size_t entries) {
    std::sort(bytes.begin(), bytes.end(), std::greater<uint64_t>());
    uint64_t sum = 0;
    for (size_t i = 0; i < std::min(entries, bytes.size()); ++i)
      sum = MatchingAdd(sum, bytes[i]);
    return sum;
  }
  // Select once, before workers or payload loads. Even one entry needs the
  // complete worker-pin/load envelope AND the largest pair headroom. Cache
  // capacity is not a promise that two images or all workers fit inside it.
  size_t Select(size_t maximum,
                uint64_t pinned,
                uint64_t pair,
                uint64_t capacity,
                uint64_t policy,
                uint64_t reserve) const {
    if (!maximum)
      throw std::invalid_argument("matching cache capacity is zero");
    auto fits = [&](size_t entries) {
      return MatchingMargined(
                 MatchingAdd(MatchingAdd(Retained(entries), pinned), pair),
                 policy,
                 reserve) <= capacity;
    };
    if (!fits(1)) {
      const auto required = MatchingMargined(
          MatchingAdd(MatchingAdd(Retained(1), pinned), pair), policy, reserve);
      throw std::runtime_error(
          "matching logical singleton refused with minimum cache and worker "
          "payload: required_bytes=" +
          std::to_string(required) + " capacity_bytes=" +
          std::to_string(capacity) + " (not a physical-fit guarantee)");
    }
    size_t low = 1, high = maximum;
    while (low < high) {
      const auto mid = low + (high - low) / 2 + 1;
      if (fits(mid))
        low = mid;
      else
        high = mid - 1;
    }
    return low;
  }
  uint64_t Retained(size_t entries) const {
    return MatchingAdd(MatchingAdd(LargestSum(keypoints, entries),
                                   LargestSum(descriptors, entries)),
                       LargestSum(indexes, entries));
  }
};
}  // namespace colmap
