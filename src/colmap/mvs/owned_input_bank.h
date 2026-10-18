// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match_options.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace colmap::mvs::owned {

// Metadata only. Paths identify immutable admitted input files, not job indices.
struct BankInput {
  int image_idx = -1;
  Image image;
  std::filesystem::path depth_path;
  std::filesystem::path normal_path;
  int max_image_size = -1;
};

struct BankJob {
  PatchMatchOptions options;
  std::vector<Image> images;
  int reference_image_idx = -1;
  std::vector<int> source_image_idxs;
  std::vector<BankInput> inputs;
  std::filesystem::path depth_map_path;
  std::filesystem::path normal_map_path;
};

// One file's captured object identity: canonical resolved path, size,
// device/inode and full modified/changed timestamps. The same input object
// requires ALL fields to match; a same-shaped replacement (equal dimensions,
// different object or later change) is a DIFFERENT identity and must be
// rejected when validated against a captured invocation snapshot.
struct BankFileIdentity {
  std::filesystem::path path;
  uintmax_t size = 0;
  uintmax_t device = 0;
  uintmax_t inode = 0;
  int64_t modified_seconds = 0;
  int64_t modified_nanoseconds = 0;
  int64_t changed_seconds = 0;
  int64_t changed_nanoseconds = 0;
};

inline bool SameIdentity(const BankFileIdentity& a, const BankFileIdentity& b) {
  return a.path == b.path && a.size == b.size && a.device == b.device &&
         a.inode == b.inode &&
         a.modified_seconds == b.modified_seconds &&
         a.modified_nanoseconds == b.modified_nanoseconds &&
         a.changed_seconds == b.changed_seconds &&
         a.changed_nanoseconds == b.changed_nanoseconds;
}

// Runtime identity capture for one invocation snapshot input. Resolves the
// path canonically, then captures object identity plus modified/changed
// metadata; fails closed on non-regular files or unsupported platforms.
// Shared with the resident contract metadata capture.
BankFileIdentity CaptureFileIdentity(const std::filesystem::path& path);

// Validates a previously captured invocation identity against the CURRENT
// file state and throws when the file changed object, disappeared, was
// modified, or had its change metadata touched. Shared with the resident
// contract post-read revalidation.
void CheckFileIdentityUnchanged(const BankFileIdentity& identity);

// One bank layer's identity triple in role order (image, depth, normal).
struct BankLayerIdentity {
  BankFileIdentity image;
  BankFileIdentity depth;
  BankFileIdentity normal;
};

// Per resolved job, in job.inputs order: the admitted input's identity.
using BankJobIdentities = std::vector<BankLayerIdentity>;

struct BankLimits {
  uint64_t host_bytes = 0;
  uint64_t device_bytes = 0;
};

// Explicit reference-normal allowances for on-demand loading. Logical raw
// minima (no additional margin; the bank estimate's single 3% margin already
// covers the bank's own raw terms): not physical bounds.
struct NormalAllowance {
  // Active owned normal retained by each owner job until retirement:
  // 12 * width * height (3 float channels) of that job's reference layer.
  uint64_t owner_bytes = 0;
  // Ready-but-unhanded bounded prefetch buffers: max_future * owner_bytes.
  uint64_t future_bytes = 0;
  // One in-flight load peak: normal file payload plus, when downsizing, the
  // checked actual overlap of the full called allocation path — Rescale keeps
  // the original payload capacity, new_data and the shrink replacement live
  // (payload + two destination maps) while each smoothing pass keeps new_data
  // plus TWO full-source float temps (warp.cc smoothed vector AND the VLFeat
  // convolution buffer inside vl_imsmooth_f) plus the input-derived Gaussian
  // filters (filtery aliasing filterx when the sigmas match exactly). The
  // larger of the two checked peaks is charged; requested-size terms, with
  // malloc usable rounding the same internally uncounted estimate class as
  // the existing decoder internals.
  uint64_t transient_bytes = 0;
  // Number of distinct reference layers in the bank.
  size_t reference_layers = 0;
};

// Eager (default): the bulk constructor reads and retains reference normals.
// Bounded (experimental, opt-in): the constructor loads none; owners load
// their reference normal on demand via LoadReferenceNormal, optionally
// through a bounded NormalLookahead producer.
enum class BankNormals { kEager, kBounded };

// Margin-FREE raw bank estimate. The single canonical 3% forecast margin is
// applied by the CALLER (owned::MarginBytes) exactly once, AFTER adding its
// explicit unclassified allowance to the host raw total (the device raw
// never takes an allowance), matching the canonical Python estimator's
// allowance-before-margin composition. Margined estimates remain available
// through Estimate/EstimateBounded for the allowance-free minimum.
struct BankRawLimits {
  // Margin-free eager host raw (retained + metadata + staging allowances).
  uint64_t host_raw_bytes = 0;
  // Margin-free bounded host raw: eager host raw minus the retained
  // reference normals exactly once (underflow refuses).
  uint64_t bounded_host_raw_bytes = 0;
  // Margin-free device raw (never allowance-adjusted).
  uint64_t device_raw_bytes = 0;
};

struct BankWorkCounts {
  size_t grayscale_reads = 0;
  size_t depth_reads = 0;
  size_t normal_reads = 0;
};

struct BankLayer {
  BankInput input;
  Bitmap grayscale;
  DepthMap depth;
  // Only eager-mode reference layers retain a normal; bounded-mode reference
  // normals are loaded on demand and owned by the loading owner/pipeline.
  NormalMap reference_normal;
  bool is_reference = false;
};

// One fixed invocation owner; no mutable cache or concurrent loading.
class HostInputBank {
 public:
  // Eager metadata forecast (default mode): includes build staging, retained
  // reference normals (17 bytes/texel per reference layer: 1 gray + 4 depth
  // + 12 normal) and the unchanged non-normal construction staging/max
  // allowances, not the caller's private solver/process reservations, with
  // the single 3% margin applied once to the raw total. Logical payload
  // estimates are not physical allocation guarantees.
  static BankLimits Estimate(const std::vector<BankJob>& jobs);
  // Metadata-only capture of every admitted input's file identity, per job
  // in job.inputs order. Snapshot identity ONCE per invocation; validate the
  // snapshot against every later read (planner header checks, construction,
  // reload, on-demand normal loads). Constructor or estimator recapture is
  // NOT proof: a same-shaped replacement after the snapshot must reject the
  // invocation, never be re-admitted.
  static std::vector<BankJobIdentities> CaptureJobIdentities(
      const std::vector<BankJob>& jobs);
  // Bounded-mode bank forecast: identical to Estimate except that ALL
  // reference-normal retention (12 bytes/texel per reference layer) is
  // removed exactly once from the raw total before the same single 3%
  // margin. No other term (in particular no construction/decode staging
  // allowance) is attributed to normals or removed; device estimate and GPU
  // layout are unchanged. Owner/prefetch normal bytes come back explicitly
  // from EstimateNormalAllowance and are reserved by the caller separately;
  // they must never be re-added or re-margined into this bank forecast.
  static BankLimits EstimateBounded(const std::vector<BankJob>& jobs);
  // Margin-free raw estimates for callers that apply the single canonical
  // margin themselves (e.g. after adding an explicit unclassified
  // allowance); see BankRawLimits. Never re-margins a margined value.
  static BankRawLimits RawEstimate(const std::vector<BankJob>& jobs);
  // Explicit owner/normal allowances reported for runtime charge records.
  // max_future must be 1 or 2 (the bounded FIFO queue contract).
  static NormalAllowance EstimateNormalAllowance(const std::vector<BankJob>& jobs,
                                                 size_t max_future);
  HostInputBank(std::vector<BankJob> jobs, BankLimits limits,
                BankNormals normals = BankNormals::kEager,
                const BankJobIdentities* identity_snapshot = nullptr,
                size_t snapshot_count = 0);
  ~HostInputBank();
  HostInputBank(const HostInputBank&) = delete;
  HostInputBank& operator=(const HostInputBank&) = delete;
  HostInputBank(HostInputBank&&) noexcept;
  HostInputBank& operator=(HostInputBank&&) noexcept;

  const BankJob& Job(size_t job) const;
  size_t NumJobs() const;
  const std::vector<BankLayer>& Layers() const;
  size_t Width() const;
  size_t Height() const;
  int ReferenceLayer(size_t job) const;
  const std::vector<int>& SourceLayers(size_t job) const;
  // Eager mode only; bounded-mode banks keep no retained reference normal.
  const NormalMap& ReferenceNormal(size_t job) const;
  BankNormals NormalsMode() const;
  // Bounded mode: load exactly one reference normal by original bank layer
  // index. Returns a unique owned map (no shared mutable payload vector);
  // file identity is rechecked immediately before and after the read, paths/
  // header/channel/dimension validation and the exact Downsize semantics of
  // the eager constructor are preserved. Read-only: safe to call while owners
  // run; load counts are reported by the caller, not mutated here.
  NormalMap LoadReferenceNormal(size_t layer) const;
  const BankWorkCounts& WorkCounts() const;
  // Only after completed GPU upload. Metadata and eager reference normals
  // survive.
  void ReleaseUploadPayloads();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// CPU-side move-only owner of GPU resources. Borrow only after construction
// completes; all consuming streams must drain and owner threads join before
// destruction. Implementation resides in a CUDA translation unit.
class ResidentInputBank {
 public:
  ResidentInputBank(const HostInputBank& host, int device_index);
  ~ResidentInputBank();
  ResidentInputBank(const ResidentInputBank&) = delete;
  ResidentInputBank& operator=(const ResidentInputBank&) = delete;
  ResidentInputBank(ResidentInputBank&&) noexcept;
  ResidentInputBank& operator=(ResidentInputBank&&) noexcept;

  size_t Width() const;
  size_t Height() const;
  size_t NumLayers() const;
  int DeviceIndex() const;
  // uint64_t keeps this interface usable by ordinary C++ translation units;
  // CUDA consumers convert these borrowed handles to cudaTextureObject_t.
  uint64_t GrayscaleTexture() const;
  uint64_t ReferenceGrayscaleTexture() const;
  uint64_t DepthTexture() const;
  size_t LayerWidth(size_t layer) const;
  size_t LayerHeight(size_t layer) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace colmap::mvs::owned
