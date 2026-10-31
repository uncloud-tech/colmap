// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace colmap::mvs {

class FusionResourceError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline uint64_t FusionCheckedAdd(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("fusion resource addition overflow");
  return a + b;
}

inline uint64_t FusionCheckedMultiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::overflow_error("fusion resource multiplication overflow");
  return a * b;
}

// Ordered, distinct, full-resolution inputs. No external image masks. RGB
// plus atomic mask are retained; converting char -> atomic overlaps both masks.
// Mapped float payload is evidence only, neither resident floor nor free credit.
struct FusionInputInventory {
  uint64_t retained_bytes = 0;
  uint64_t peak_bytes = 0;
  uint64_t mapped_bytes = 0;

  void AddImage(size_t width, size_t height) {
    if (!width || !height)
      throw std::invalid_argument("fusion resource dimensions must be positive");
    const auto area = FusionCheckedMultiply(width, height);
    const auto peak = FusionCheckedAdd(retained_bytes,
                                       FusionCheckedMultiply(5, area));
    const auto retained = FusionCheckedAdd(retained_bytes,
                                           FusionCheckedMultiply(4, area));
    const auto mapped = FusionCheckedAdd(mapped_bytes,
                                         FusionCheckedMultiply(16, area));
    retained_bytes = retained;
    peak_bytes = std::max(peak_bytes, std::max(peak, retained));
    mapped_bytes = mapped;
  }
};

// Logical payload accounting, NOT whole physical fit. Policy covers unknown
// decoder/scratch/metadata/allocator/traversal costs; reserve is outside the
// single aggregate ceil(3%) margin. Capacity is frozen before input allocation,
// never re-subtracted from newly observed FREE as inputs/outputs become live.
class FusionLogicalBudget {
 public:
  FusionLogicalBudget(uint64_t capacity, uint64_t policy, uint64_t reserve)
      : capacity_(capacity), policy_(policy), reserve_(reserve) {}

  void Check(uint64_t inputs, uint64_t outputs) const {
    const auto raw = FusionCheckedAdd(FusionCheckedAdd(inputs, outputs), policy_);
    // ceil(raw * 103 / 100), without an overflowing intermediate product.
    const auto margin = FusionCheckedAdd(
        FusionCheckedMultiply(raw / 100, 3),
        (raw % 100 * 3 + 99) / 100);
    const auto required = FusionCheckedAdd(FusionCheckedAdd(raw, margin), reserve_);
    if (required > capacity_)
      throw FusionResourceError("fusion logical host budget exhausted (not a physical-fit guarantee)");
  }

 private:
  uint64_t capacity_;
  uint64_t policy_;
  uint64_t reserve_;
};

// Single fusion worker and post-join merge only. Track capacities, not sizes.
// reserve's requested logical payload is checked with old storage still live;
// allocator bookkeeping/rounding remains policy, not a physical-fit claim.
class FusionOutputLedger {
 public:
  FusionOutputLedger(FusionLogicalBudget budget, uint64_t inputs)
      : budget_(budget), inputs_(inputs) {}

  uint64_t Bytes() const { return bytes_; }

  template <typename T>
  void Reserve(std::vector<T>& values, size_t capacity) {
    if (capacity <= values.capacity()) return;
    const auto old = FusionCheckedMultiply(values.capacity(), sizeof(T));
    const auto added = FusionCheckedMultiply(capacity, sizeof(T));
    budget_.Check(inputs_, FusionCheckedAdd(bytes_, added));
    values.reserve(capacity);
    // Standard reserve is at least the requested capacity. Use the actual
    // capacity afterwards too (supported native STL reserves exactly here).
    const auto actual = FusionCheckedMultiply(values.capacity(), sizeof(T));
    bytes_ = FusionCheckedAdd(bytes_ - old, actual);
    budget_.Check(inputs_, bytes_);
  }

  template <typename T>
  void GrowForAppend(std::vector<T>& values) {
    if (values.size() < values.capacity()) return;
    const auto capacity = values.capacity()
                              ? FusionCheckedMultiply(values.capacity(), 2)
                              : 1;
    if (capacity > values.max_size())
      throw std::overflow_error("fusion output capacity overflow");
    Reserve(values, static_cast<size_t>(capacity));
  }

  template <typename T>
  void Release(std::vector<T>& values) {
    bytes_ -= FusionCheckedMultiply(values.capacity(), sizeof(T));
    std::vector<T>().swap(values);
  }

 private:
  FusionLogicalBudget budget_;
  uint64_t inputs_;
  uint64_t bytes_ = 0;
};

}  // namespace colmap::mvs
