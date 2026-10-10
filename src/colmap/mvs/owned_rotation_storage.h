// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace colmap::mvs::owned {

enum class RotationRole {
  kPrng,
  kDepth,
  kNormal,
  kRefImage,
  kRefSum,
  kRefSquaredSum,
  kSelectionCurrent,
  kSelectionPrevious,
  kCost,
  kMask,
  kWorkspace
};

inline const char* RotationRoleName(RotationRole role) {
  switch (role) {
    case RotationRole::kPrng:
      return "prng";
    case RotationRole::kDepth:
      return "depth";
    case RotationRole::kNormal:
      return "normal";
    case RotationRole::kRefImage:
      return "ref_image";
    case RotationRole::kRefSum:
      return "ref_sum";
    case RotationRole::kRefSquaredSum:
      return "ref_squared_sum";
    case RotationRole::kSelectionCurrent:
      return "selection_current";
    case RotationRole::kSelectionPrevious:
      return "selection_previous";
    case RotationRole::kCost:
      return "cost";
    case RotationRole::kMask:
      return "mask";
    case RotationRole::kWorkspace:
      return "workspace";
  }
  throw std::invalid_argument("Invalid rotation role");
}

struct RotationSlotLayout {
  RotationRole role;
  unsigned parity;
  size_t width, height, depth, element_bytes, requested_span_bytes;
};

struct OwnedRotationLayout {
  size_t width, height, sources, logical_bytes;
  std::vector<RotationSlotLayout> slots;

  const RotationSlotLayout& At(RotationRole role, unsigned parity) const {
    RotationRoleName(role);
    if (parity > 1 || (role == RotationRole::kWorkspace && parity != 0)) {
      throw std::invalid_argument("Invalid rotation parity");
    }
    for (const auto& slot : slots) {
      if (slot.role == role && slot.parity == parity) return slot;
    }
    throw std::out_of_range("Missing rotation slot");
  }
};

namespace detail {
inline size_t CheckedMultiply(size_t a, size_t b) {
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
    throw std::overflow_error("Rotation size multiplication overflow");
  }
  return a * b;
}

inline size_t CheckedAdd(size_t a, size_t b) {
  if (a > std::numeric_limits<size_t>::max() - b) {
    throw std::overflow_error("Rotation size addition overflow");
  }
  return a + b;
}

inline void CheckDimension(size_t dimension) {
  if (dimension == 0 || dimension > static_cast<size_t>(INT_MAX)) {
    throw std::invalid_argument("Rotation dimension outside kernel int range");
  }
}

inline size_t ElementBytes(RotationRole role) {
  RotationRoleName(role);
  if (role == RotationRole::kPrng) return 48;
  if (role == RotationRole::kRefImage || role == RotationRole::kMask) return 1;
  return 4;
}

inline size_t Depth(RotationRole role, size_t sources) {
  RotationRoleName(role);
  switch (role) {
    case RotationRole::kNormal:
      return 3;
    case RotationRole::kWorkspace:
      return 2;
    case RotationRole::kSelectionCurrent:
    case RotationRole::kSelectionPrevious:
    case RotationRole::kCost:
    case RotationRole::kMask:
      return sources;
    default:
      return 1;
  }
}

inline size_t ValidateSlot(const RotationSlotLayout& slot) {
  RotationRoleName(slot.role);
  if (slot.parity > 1 ||
      (slot.role == RotationRole::kWorkspace && slot.parity != 0)) {
    throw std::invalid_argument("Invalid rotation parity");
  }
  CheckDimension(slot.width);
  CheckDimension(slot.height);
  CheckDimension(slot.depth);
  if (slot.element_bytes != ElementBytes(slot.role) ||
      slot.depth != Depth(slot.role, slot.depth)) {
    throw std::invalid_argument("Invalid rotation element size or depth");
  }
  const size_t row_bytes = CheckedMultiply(slot.width, slot.element_bytes);
  const size_t span =
      CheckedMultiply(CheckedMultiply(row_bytes, slot.height), slot.depth);
  if (span != slot.requested_span_bytes) {
    throw std::invalid_argument("Invalid requested rotation span");
  }
  return row_bytes;
}
}  // namespace detail

// Source-level logical inventory only: no pitch estimate or driver-memory bound.
inline OwnedRotationLayout BuildOwnedRotationLayout(size_t width,
                                                   size_t height,
                                                   size_t sources) {
  detail::CheckDimension(width);
  detail::CheckDimension(height);
  detail::CheckDimension(sources);
  OwnedRotationLayout layout{width, height, sources, 0, {}};
  layout.slots.reserve(21);
  for (unsigned role_index = 0; role_index <= 10; ++role_index) {
    const auto role = static_cast<RotationRole>(role_index);
    const bool workspace = role == RotationRole::kWorkspace;
    for (unsigned parity = 0; parity < (workspace ? 1u : 2u); ++parity) {
      RotationSlotLayout slot{
          role,
          parity,
          workspace ? std::max(width, height) : (parity ? height : width),
          workspace ? sources : (parity ? width : height),
          detail::Depth(role, sources),
          detail::ElementBytes(role),
          0};
      slot.requested_span_bytes = detail::CheckedMultiply(
          detail::CheckedMultiply(
              detail::CheckedMultiply(slot.width, slot.element_bytes),
              slot.height),
          slot.depth);
      detail::ValidateSlot(slot);
      layout.logical_bytes =
          detail::CheckedAdd(layout.logical_bytes, slot.requested_span_bytes);
      layout.slots.push_back(slot);
    }
  }
  return layout;
}

struct RotationAllocationRecord {
  RotationRole role;
  unsigned parity;
  size_t width, height, depth, element_bytes, pitch_bytes;
  size_t requested_span_bytes, pitched_span_bytes;
  const char* lifetime_phase;
};

// Returned pitch is allocation-specific. This span is NOT a physical allocation
// upper bound (driver padding, arrays, and context memory remain unaccounted).
inline RotationAllocationRecord MakeRotationAllocationRecord(
    const RotationSlotLayout& slot, size_t returned_pitch) {
  const size_t row_bytes = detail::ValidateSlot(slot);
  if (returned_pitch < row_bytes) {
    throw std::invalid_argument("Returned pitch is smaller than rotation row");
  }
  const size_t pitched_span = detail::CheckedMultiply(
      detail::CheckedMultiply(returned_pitch, slot.height), slot.depth);
  return {slot.role,
          slot.parity,
          slot.width,
          slot.height,
          slot.depth,
          slot.element_bytes,
          returned_pitch,
          slot.requested_span_bytes,
          pitched_span,
          "prepared_through_outputs"};
}

// Two immutable-shape owners, including for square images. No allocation,
// initialization, replacement, or stream completion is performed here. The
// enclosing owner MUST drain before member destruction, including constructor
// failure and cancellation. Physical shapes are checked via layouts/records by
// the caller, not through assumptions about T's API.
template <class T>
class OwnedRotationPair {
 public:
  OwnedRotationPair() = default;
  OwnedRotationPair(const OwnedRotationPair&) = delete;
  OwnedRotationPair& operator=(const OwnedRotationPair&) = delete;
  OwnedRotationPair(OwnedRotationPair&&) = delete;
  OwnedRotationPair& operator=(OwnedRotationPair&&) = delete;

  void Set(unsigned parity, std::unique_ptr<T> value) {
    CheckParity(parity);
    if (!value || slots_[parity]) {
      throw std::invalid_argument("Rotation slot must be nonnull and empty");
    }
    slots_[parity] = std::move(value);
  }

  T* Current() const { return At(current_parity_); }
  T* Alternate() const { return At(current_parity_ ^ 1); }
  T* At(unsigned parity) const {
    CheckParity(parity);
    if (!slots_[parity]) throw std::logic_error("Missing rotation owner");
    return slots_[parity].get();
  }
  T* operator->() const { return Current(); }
  unsigned CurrentParity() const { return current_parity_; }
  void Flip() {
    At(0);
    At(1);
    current_parity_ ^= 1;
  }

 private:
  static void CheckParity(unsigned parity) {
    if (parity > 1) throw std::invalid_argument("Invalid rotation parity");
  }

  std::unique_ptr<T> slots_[2];
  unsigned current_parity_ = 0;
};

}  // namespace colmap::mvs::owned
