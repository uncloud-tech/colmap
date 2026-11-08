// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_prng_boundary.h"

namespace colmap::mvs::owned {

// The native planner and solver must omit the same file-backed owners. Keep
// this role filter shared; source-scaled byte subtraction is phase-dependent.
inline OwnedRotationLayout BuildOwnedExecutionRotationLayout(
    size_t width, size_t height, size_t sources, bool retain_mask, bool strip) {
  auto layout =
      BuildOwnedBoundaryRotationLayout(width, height, sources, retain_mask);
  if (strip) {
    auto& slots = layout.slot_layouts;
    slots.erase(
        std::remove_if(slots.begin(),
                       slots.end(),
                       [&](const auto& slot) {
                         const bool file_backed =
                             slot.role == RotationRole::kCost ||
                             slot.role == RotationRole::kSelectionCurrent ||
                             slot.role == RotationRole::kSelectionPrevious ||
                             slot.role == RotationRole::kMask;
                         if (file_backed)
                           layout.logical_bytes -= slot.requested_span_bytes;
                         return file_backed;
                       }),
        slots.end());
  }
  return layout;
}

struct OwnedRotationMemoryPhases {
  size_t initialization;
  size_t steady;
  size_t peak;
};

// Current owned CUDA path ONLY, not the legacy or optimized-photo algorithm.
// Logical requested bytes, without pitch/allocator overhead or any margin.
// InitRefImage + InitWorkspaceMemory retain the canonical reference image,
// depth/normal, packed current selection, previous selection, cost and
// workspace. InitRotationStorage adds compact boundary RNG while full-image RNG
// is still live, enqueues its stream-ordered release after compaction, then
// waits for physical release BEFORE any swapped owners (synchronous fallback
// on devices without stream-ordered allocator support). No initialization
// serialization across independent owners is assumed:
// each owner must reserve max(initialization, steady), not just retained
// inventory.
inline OwnedRotationMemoryPhases BuildOwnedRotationMemoryPhases(
    size_t width, size_t height, size_t sources, bool retain_mask = true) {
  const auto retained =
      BuildOwnedBoundaryRotationLayout(width, height, sources, retain_mask);
  size_t initialization = 0;
  for (const auto& slot : retained.slot_layouts) {
    if (slot.parity == 0) {
      initialization =
          detail::CheckedAdd(initialization, slot.requested_span_bytes);
    }
  }
  // The compact PRNG (parity zero above) and full initialization map coexist.
  const size_t full_prng =
      detail::CheckedMultiply(detail::CheckedMultiply(width, height), 48);
  initialization = detail::CheckedAdd(initialization, full_prng);
  return {initialization,
          retained.logical_bytes,
          std::max(initialization, retained.logical_bytes)};
}

}  // namespace colmap::mvs::owned
