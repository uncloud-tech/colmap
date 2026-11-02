// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// CPU-only placement metadata. Scientific source vectors/options are never
// rewritten by placement. All byte charges below are RAW; the planner applies
// the existing aggregate ceil(1.03 * raw) convention only at admission.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace colmap::mvs::owned {

struct PhasePlanCharge {
  uint64_t host = 0;
  uint64_t device = 0;
};

struct PhasePlanImage {
  // Invocation-local immutable identity (canonical image plus geometry/format;
  // GEO additionally binds the completed PHOTO exchange generation). Indexes
  // refer to this frozen table, not paths resolved again during placement.
  std::string identity;
  uint64_t width = 0;
  uint64_t height = 0;
};

struct PhasePlanJob {
  size_t id = 0;  // Global reference ordinal; unique, independent of image index.
  size_t reference = 0;
  // Exact ordered scientific read-set, including duplicates. Placement may
  // deduplicate its own bank union but must return these inputs unchanged.
  std::vector<size_t> sources;
  PhasePlanCharge private_raw;  // Complete owner setup/compute/output overlap.
  uint64_t work = 0;  // Deterministic pixel/source work proxy, not measured time.
};

struct PhasePlanDevice {
  int ordinal = -1;
  std::string uuid;
  uint64_t capacity_bytes = 0;
  uint64_t reserve_bytes = 0;
  // Static RAW per-device host envelope, excluding common retained pixels/maps.
  // Never give each device the invocation-wide host capacity.
  uint64_t host_share_raw = 0;
  PhasePlanCharge fixed_raw;  // Per-device process/unknown/metadata allowance.
};

struct PhasePlanRequest {
  bool geometry = false;
  std::vector<PhasePlanImage> images;
  std::vector<PhasePlanJob> jobs;
  std::vector<PhasePlanDevice> devices;
  size_t active_owners_ceiling = 1;
  // Common caller+frozen pixels, PHOTO exchange AND retained GEO host maps,
  // loader/output overlap and common process/unknown metadata. Not per GPU.
  uint64_t common_host_raw = 0;
  uint64_t host_capacity_bytes = 0;
  uint64_t host_reserve_bytes = 0;
};

struct PhasePlanCohort {
  // Input-relative order is preserved within each cohort. Exactly one cohort
  // owns each global ID. Cohorts on one GPU execute sequentially and are fully
  // drained/destroyed before the next bank is constructed.
  std::vector<size_t> jobs;
  size_t active_owners = 0;
  // PHOTO: padded union of distinct SOURCE images only. GEO: zero (no cache).
  PhasePlanCharge bank_raw;
  PhasePlanCharge private_peak_raw;
};

struct PhasePlanGroup {
  PhasePlanDevice device;
  std::vector<PhasePlanCohort> cohorts;
  // fixed_raw + max(cohort bank + resourcewise top-k private), not sum cohorts.
  PhasePlanCharge peak_raw;
  uint64_t assigned_work = 0;
};

struct PhasePlan {
  bool geometry = false;
  std::vector<PhasePlanGroup> groups;  // Approved device order, including idle.
  uint64_t host_peak_raw = 0;  // common + sum(group host peaks).
};

// Validates all identities, dimensions, unique IDs/device ordinals/UUIDs and
// checked arithmetic before admission. Refuses any complete singleton no-fit
// or aggregate-host violation without resizing/reordering/dropping work.
// Deterministic greedy: seed feasible useful work across devices; prefer lower
// incremental padded PHOTO source-union bytes among balanced feasible groups,
// then assigned work and approved device order. GEO uses the same placement
// function independently, but bank costs are zero. If extending a bank cannot
// fit, start a sequential admitted cohort at unchanged quality. Owner limits
// are bounded by the ceiling, available members and complete peak admission.
PhasePlan BuildMultiGpuPhasePlan(const PhasePlanRequest& request);

}  // namespace colmap::mvs::owned
