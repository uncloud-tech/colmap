// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#if !defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/cuda_texture.h"
#include "colmap/mvs/owned_photo_output.h"
#include "colmap/mvs/patch_match.h"

namespace colmap::mvs::owned {

// Opt-in fixed-cohort PHOTO -> GEO setup adapter. All PHOTO snapshots must be
// published before construction (global phase barrier). This is a COPY path:
// ordered source depths -> zero-padded pitched staging -> private depth
// texture; reference depth/normal -> the consumer's existing mutable solver
// buffers. No host/disk map round trip, no sharing/admission credit, no default
// dispatch. Caller keeps images/pixels frozen from PHOTO construction through
// GEO completion; camera metadata validation is NOT a pixel/content hash.
// GEO options are bound at adapter construction, separately from the producer's
// PHOTO options (e.g. geometry/filtering may deliberately differ by phase).
// Source ordinals, including duplicates, are copied verbatim, never
// sorted/deduplicated.
class GeoPhotoInput {
 public:
  using Snapshot = std::shared_ptr<const PhotoOutputSnapshot>;
  GeoPhotoInput(const PatchMatchOptions& options,
                const PatchMatch::Problem& problem,
                Snapshot reference,
                std::vector<Snapshot> ordered_sources);
  ~GeoPhotoInput() noexcept;
  GeoPhotoInput(const GeoPhotoInput&) = delete;
  GeoPhotoInput& operator=(const GeoPhotoInput&) = delete;

  // Setup-only. Reject mixed modes and changed ordered invocation bindings
  // before the consumer allocates GPU resources. Single explicit device only.
  void Validate(const PatchMatchOptions& options,
                const PatchMatch::Problem& problem) const;
  cudaTextureObject_t SourceDepthTexture() const {
    return source_depth_->GetObj();
  }
  void CopyReferenceMaps(OwnedCudaStream& stream,
                         GpuMat<float>& depth,
                         GpuMat<float>& normal) const;

  // Device-only queries (host metadata/handles/images remain extra).
  // Pitched physical allocations where queryable; CUDA array allocation
  // overhead is NOT queryable here, so texture bytes are a logical lower bound.
  // Retained snapshots live through the final consumer drain. Duplicate handles
  // count once. All producer/workspace/private grayscale/rotation/pool storage
  // remains EXTRA; no number below certifies total process peak or admission.
  size_t SnapshotRetainedBytes() const;
  size_t SourceStagingBytes() const { return source_staging_bytes_; }
  size_t SourceTextureLogicalBytes() const { return source_texture_bytes_; }
  size_t ReferenceCopyLogicalBytes() const {
    return reference_->Width() * reference_->Height() * 4 * sizeof(float);
  }
  // Staging lives only during setup, overlapping retained snapshots and the
  // new texture. Reference copies reuse the solver's already charged buffers
  // (their actual pitches are those of the supplied GpuMats), not extra owners.

 private:
  const PatchMatchOptions geo_options_;
  const Snapshot reference_;
  const std::vector<Snapshot> sources_;
  const int device_;
  size_t source_staging_bytes_ = 0;
  size_t source_texture_bytes_ = 0;
  std::unique_ptr<CudaArrayLayeredTexture<float>> source_depth_;
};
}  // namespace colmap::mvs::owned
#endif
