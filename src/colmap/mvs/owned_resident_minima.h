// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Shared native logical minima for the resident full-bank execution.
//
// Shared by current-owned pair, full-bank and cohort admission. The rotation
// envelope covers initialization compaction AND steady retained allocations;
// legacy layout APIs for other algorithms remain separate.
//
// These are the reviewed lower bounds the resident forecast ledger is
// validated against. They deliberately state LOGICAL minima (no additional
// margins; the caller's single canonical 3% margin is already inside any
// supplied charge), not physical allocatation guarantees.
//
// No CUDA/estimator dependencies: this must stay an ordinary C++ header so
// both the driver (colmap_owned_pair) and the product runner consume the
// same expressions. Unknown new source remains unqualified until review.

#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/owned_rotation_memory.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace colmap {
namespace mvs {
namespace owned {

// Single 3% forecast margin, rounded up; not a physical allocation bound.
// Identical integer semantics to the estimator's internal Margin (and the
// canonical Python margin_bytes), shared here so every consumer applies the
// same single margin exactly once.
inline uint64_t MarginBytes(uint64_t bytes) {
  return CheckedAdd(bytes,
                    CheckedAdd(CheckedMultiply(bytes / 100, 3),
                               (CheckedMultiply(bytes % 100, 3) + 99) / 100));
}

struct ResidentPrivateMinimum {
  uint64_t host = 0;
  uint64_t device = 0;
};

// Per-reference private logical minimum. `include_owner_normal` is the
// bounded-mode active reference normal holder (12 * area, 3 float channels):
// the owner retains one normal until retirement; the existing 24a
// result/reference scratch allowance covers a distinct holder, so these are
// additive and never double counted, and the bounded bank forecast excludes
// the retained reference normals exactly once (see
// HostInputBank::EstimateBounded). Device terms: the current-owned rotation
// phase envelope plus source/pose arrays (692 bytes per source including the
// private int layer mapping), both live reference arrays during rebinding
// (2 * area), and the retained 32-byte calibration policy allowance. The
// default retained mask conservatively covers filtered and unfiltered
// executions; requested spans are not pitched physical allocation guarantees.
inline ResidentPrivateMinimum ResidentPrivateLogicalMinimum(
    const uint64_t width,
    const uint64_t height,
    const size_t sources,
    const bool include_owner_normal) {
  const uint64_t area = CheckedMultiply(width, height);
  if (!area || !sources) {
    throw std::invalid_argument("empty resident problem");
  }
  ResidentPrivateMinimum minimum;
  minimum.host = CheckedMultiply(24, area);
  if (include_owner_normal) {
    minimum.host = CheckedAdd(minimum.host, CheckedMultiply(12, area));
  }
  uint64_t device = BuildOwnedRotationMemoryPhases(width, height, sources).peak;
  device = CheckedAdd(device, CheckedMultiply(2, area));
  device = CheckedAdd(device, CheckedMultiply(692, sources));
  device = CheckedAdd(device, 32);
  minimum.device = device;
  return minimum;
}

}  // namespace owned
}  // namespace mvs
}  // namespace colmap