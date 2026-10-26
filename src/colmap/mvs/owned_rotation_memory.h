// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_prng_boundary.h"

namespace colmap::mvs::owned {

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
// is still live, waits for compaction, then FREES full RNG BEFORE any swapped
// owners. No initialization serialization across independent owners is assumed:
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
