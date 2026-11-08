// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_fixed_cohort_inventory.h"
#include "colmap/mvs/owned_strip_storage.h"

namespace colmap::mvs::owned {

// Explicit single-device HOST-exchange inventory, not the snapshot or ordinary
// multi-GPU API. All charges are estimates, checked again at allocation time;
// CUDA does not promise the estimated pitch alignment below.
enum class StereoVolumeMode { kResidentPrivate, kStripFile };
enum class StereoAdmissionResource { kHost, kDevice, kDisk, kGraphConflict };
class StereoAdmissionRefusal : public std::runtime_error {
 public:
  StereoAdmissionRefusal(StereoAdmissionResource resource,
                         bool geometry,
                         size_t reference,
                         const std::string& message)
      : std::runtime_error(message),
        resource(resource),
        geometry(geometry),
        reference(reference) {}
  StereoAdmissionResource resource;
  bool geometry;
  size_t reference;
};

struct StereoExecutionPolicy {
  bool retain_mask = true;
  bool cuda_graphs = false;
  bool async_pool = false;
};
struct StereoAdmissionReference {
  int reference = -1;
  std::array<std::vector<int>, 2> sources;
  std::array<StereoExecutionPolicy, 2> policy;
};
struct StereoPitchPlane {
  RotationRole role;
  unsigned parity;
  uint64_t row_bytes, rows, depth, pitch_ceiling;
};
struct StereoExecutionRecord {
  bool geometry = false;
  int reference = -1;
  uint64_t width = 0, height = 0;
  std::vector<int> sources;  // Ordinals, including duplicates, never a set.
  StereoExecutionPolicy policy;
  StereoVolumeMode mode = StereoVolumeMode::kResidentPrivate;
  size_t columns = 0;
  ByteCharge initialization, steady, peak;
  std::vector<StereoPitchPlane> rotation_planes;
  uint64_t initialization_prng_pitch = 0;
  uint64_t staging_float_pitch = 0, staging_mask_pitch = 0;
  uint64_t staging_host_bytes = 0, staging_device_bytes = 0;
  uint64_t scratch_disk_bytes = 0, cache_host_bytes = 0;
  uint64_t padded_source_pixels = 0;
  std::string reason;
};
struct StereoAdmissionRequest {
  std::vector<FixedCohortImageShape> images;
  std::vector<StereoAdmissionReference> references;
  ByteCharge capacity, process, unknown, reserve;
  uint64_t loader_scratch_host = 0;
  uint64_t disk_capacity = 0;
  size_t active_owners = 1;
  size_t strip_columns = 32;
  size_t block_width = 32;
  uint64_t estimated_pitch_alignment = 512;
};
struct StereoAdmissionPlan {
  std::array<std::vector<StereoExecutionRecord>, 2> records;
  size_t active_owners = 1;
  ByteCharge peak_raw, admitted;
  uint64_t common_host_raw = 0, loader_host_raw = 0;
  uint64_t output_disk_bytes = 0, scratch_disk_bytes = 0;
  uint64_t required_disk_bytes = 0;
};

struct SnapshotStereoInventory {
  FixedCohortBudget budget;
  std::array<std::vector<StereoExecutionRecord>, 2> records;
};
// Complete source-native snapshot inventory; includes the existing conservative
// logical floor, actual phase owners, pitched GEO packing and transform copies.
SnapshotStereoInventory BuildSnapshotStereoInventory(
    const StereoAdmissionRequest& request, FixedCohortBudget declarations);

// Selects resident-private before strip-file, one owner, no bank/snapshot.
// Invalid dimensions/counts/overflow are errors, never alternate-mode triggers.
StereoAdmissionPlan PlanSingleDeviceStereoAdmission(
    const StereoAdmissionRequest& request);

// Exact execution binding, including mode, policies, ordered sources and all
// raw charges/estimated pitch planes. No tolerance or backend substitution.
bool SameStereoExecutionRecord(const StereoExecutionRecord& a,
                               const StereoExecutionRecord& b);
bool SameStereoAdmissionPlan(const StereoAdmissionPlan& a,
                             const StereoAdmissionPlan& b);

// Factory checks this binding before copying any reference/source pixels.
void ValidateStereoExecutionBinding(const StereoExecutionRecord& record,
                                    int reference,
                                    uint64_t width,
                                    uint64_t height,
                                    const std::vector<int>& sources,
                                    bool geometry,
                                    StereoExecutionPolicy policy);

// Reject an allocator pitch outside the admitted policy, even if another plane
// happened to use less. Exact role/shape binding prevents checking the wrong
// allocation. This is a requested-span check, NOT driver-memory certification.
void ValidateStereoReturnedPitches(
    const StereoExecutionRecord& record,
    const std::vector<RotationAllocationRecord>& rotation,
    uint64_t initialization_prng_pitch,
    const std::array<uint64_t, 4>& staging_pitches);

}  // namespace colmap::mvs::owned
