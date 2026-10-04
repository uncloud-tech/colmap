// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

namespace colmap {
namespace mvs {
namespace rotation_storage {

// Compact layers have the same byte extent in either orientation.
struct RotationStoragePlan {
  size_t width, height, depth, element_bytes;
  size_t source_pitch, rotated_pitch, plane_bytes, backing_bytes;
};

struct RotationBacking {
  std::shared_ptr<void> allocation;
  size_t capacity_bytes = 0;
};

struct RotationStorageView {
  RotationStoragePlan layout{};
  RotationBacking backing;
};

inline std::optional<RotationStoragePlan> MakePlan(
    size_t width, size_t height, size_t depth, size_t element_bytes) noexcept {
  const size_t int_max = std::numeric_limits<int>::max();
  const size_t size_max = std::numeric_limits<size_t>::max();
  // Both rotation dimensions/pitches and PatchMatch's source-layer loops use int.
  if (!width || !height || !depth || !element_bytes || width > int_max ||
      height > int_max || depth > int_max || element_bytes > int_max / width ||
      element_bytes > int_max / height) {
    return std::nullopt;
  }
  const size_t source_pitch = width * element_bytes;
  const size_t rotated_pitch = height * element_bytes;
  if (height > size_max / source_pitch) return std::nullopt;
  const size_t plane_bytes = source_pitch * height;
  if (depth > size_max / plane_bytes) return std::nullopt;
  return RotationStoragePlan{width, height, depth, element_bytes,
                             source_pitch, rotated_pitch, plane_bytes,
                             plane_bytes * depth};
}

inline bool HasAddressRange(const RotationBacking& backing) noexcept {
  if (!backing.allocation || !backing.capacity_bytes) return false;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(backing.allocation.get());
  return backing.capacity_bytes <= std::numeric_limits<uintptr_t>::max() - begin;
}

inline std::optional<RotationStorageView> MakeView(
    const RotationBacking& backing, size_t width, size_t height,
    size_t depth, size_t element_bytes) noexcept {
  const auto plan = MakePlan(width, height, depth, element_bytes);
  if (!plan || backing.capacity_bytes != plan->backing_bytes ||
      !HasAddressRange(backing)) {
    return std::nullopt;
  }
  return RotationStorageView{*plan, backing};
}

inline bool IsValid(const RotationStorageView& view) noexcept {
  const auto& p = view.layout;
  const auto canonical = MakeView(view.backing, p.width, p.height,
                                  p.depth, p.element_bytes);
  return canonical && p.source_pitch == canonical->layout.source_pitch &&
         p.rotated_pitch == canonical->layout.rotated_pitch &&
         p.plane_bytes == canonical->layout.plane_bytes &&
         p.backing_bytes == canonical->layout.backing_bytes;
}

inline std::optional<RotationStorageView> ReorientView(
    const RotationStorageView& source, size_t width, size_t height) noexcept {
  const auto& p = source.layout;
  if (!IsValid(source) ||
      !((width == p.width && height == p.height) ||
        (width == p.height && height == p.width))) {
    return std::nullopt;
  }
  return MakeView(source.backing, width, height, p.depth, p.element_bytes);
}

// Typed prefix of an existing workspace, never an additional allocation.
// Capacity describes the usable prefix; shared ownership retains the full buffer.
inline std::optional<RotationStorageView> MakeWorkspaceView(
    const RotationStorageView& source, size_t width, size_t height,
    size_t element_bytes) noexcept {
  if (!IsValid(source)) return std::nullopt;
  const auto plan = MakePlan(width, height, 1, element_bytes);
  if (!plan || plan->backing_bytes > source.backing.capacity_bytes)
    return std::nullopt;
  auto backing = source.backing;
  backing.capacity_bytes = plan->backing_bytes;
  return MakeView(backing, width, height, 1, element_bytes);
}

inline bool SameOwner(const RotationBacking& a,
                      const RotationBacking& b) noexcept {
  return !a.allocation.owner_before(b.allocation) &&
         !b.allocation.owner_before(a.allocation);
}

inline bool ValidateReuse(const RotationStorageView& source,
                          const RotationStorageView& destination,
                          const RotationStorageView& workspace) noexcept {
  if (!IsValid(source) || !IsValid(destination) || !IsValid(workspace)) return false;
  const auto& s = source.layout;
  const auto& d = destination.layout;
  const auto& w = workspace.layout;
  if (d.width != s.height || d.height != s.width || d.depth != s.depth ||
      d.element_bytes != s.element_bytes || w.width != d.width ||
      w.height != d.height || w.depth != 1 || w.element_bytes != s.element_bytes ||
      !SameOwner(source.backing, destination.backing) ||
      source.backing.allocation.get() != destination.backing.allocation.get() ||
      SameOwner(source.backing, workspace.backing)) {
    return false;
  }
  const uintptr_t start = reinterpret_cast<uintptr_t>(source.backing.allocation.get());
  const uintptr_t scratch = reinterpret_cast<uintptr_t>(workspace.backing.allocation.get());
  // Address ranges were checked before adding their lengths.
  return start + s.backing_bytes <= scratch || scratch + w.backing_bytes <= start;
}

// Linear index of a compact pixel after a counter-clockwise quarter turn.
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
inline size_t RotatedIndex(size_t index, size_t width, size_t height) {
  return (width - 1 - index % width) * height + index / width;
}

// Each component is consumed into disjoint scratch before ANY of that component
// is overwritten. Other components remain untouched until their own turn.
// Requires a checked plan with element_bytes divisible by four.
template <class Rotate, class Scatter>
void ForEachWord(const RotationStoragePlan& plan,
                 Rotate&& rotate, Scatter&& scatter) {
  for (size_t layer = 0; layer < plan.depth; ++layer) {
    for (size_t word = 0; word < plan.element_bytes / 4; ++word) {
      rotate(layer, word);
      scatter(layer, word);
    }
  }
}

// Caller must supply a checked plan (the GPU adapter calls ValidateReuse first).
// Callbacks are invoked synchronously in this order; they may enqueue stream work.
// Exceptions propagate: never continue after failed submission or retry a layer.
template <class Rotate, class Copy>
void ForEachLayer(const RotationStoragePlan& plan, Rotate&& rotate, Copy&& copy) {
  for (size_t layer = 0; layer < plan.depth; ++layer) {
    const size_t offset = layer * plan.plane_bytes;
    rotate(layer, offset);
    copy(layer, offset);
  }
}

}  // namespace rotation_storage
}  // namespace mvs
}  // namespace colmap
