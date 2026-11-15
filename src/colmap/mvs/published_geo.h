// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/mat.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace colmap::mvs::owned {

// Orthogonal to PHOTO retention. Materializing APIs never accept kPublished.
enum class GeoRetention { kMaterialized, kPublished };
inline const char* GeoRetentionName(GeoRetention mode) {
  switch (mode) {
    case GeoRetention::kMaterialized:
      return "materialized";
    case GeoRetention::kPublished:
      return "published";
  }
  throw std::invalid_argument("invalid GEO retention mode");
}

struct PublishedGeoFile {
  std::filesystem::path path;
  uint64_t device = 0, inode = 0, bytes = 0;
  int64_t modified_seconds = 0, modified_nanoseconds = 0;
  int64_t changed_seconds = 0, changed_nanoseconds = 0;
};
struct PublishedGeoPair {
  PublishedGeoFile depth, normal;
};

// Linux, 4KiB pages, disk-backed files only. Caller owns a private namespace
// and forbids ALL body readers/hashes/hardlinks until successful completion.
// Same UID and mincore snapshots are not a security boundary or future cache
// guarantee. Timestamps need not distinguish unobserved transient changes.
// Observed verification, publication I/O, or cache-probe failures permanently
// invalidate this attempt; later repairs cannot make it complete. Rejected
// arguments and incomplete-output queries alone do not invalidate the attempt.
// No durability/atomic-pair promise.
// Synchronous borrow only: never copies or retains a Mat payload.
class PublishedGeoWriter {
 public:
  static constexpr uint64_t kPageBytes = 4096;
  static constexpr uint64_t kProbeWindowBytes = 4 * 1024 * 1024;
  static constexpr uint64_t kProbeBytes = kProbeWindowBytes / kPageBytes;
  // Initial Linux forecast profile allowances, NOT universal kernel bounds.
  // 2MiB file-granule padding; 128B per live body page for cache/xarray/folio
  // metadata (excludes inode/path metadata); retained inode/path allowance
  // 64KiB/file. Kept Linux 6.8 Mat-scale raw stats support that observed case,
  // not target certification. Existing process/unknown and ONE 3% remain.
  static constexpr uint64_t kCacheGranuleBytes = 2 * 1024 * 1024;
  static constexpr uint64_t kLiveMetadataPerPage = 128;
  static constexpr uint64_t kFileAllowanceBytes = 64 * 1024;
  static constexpr uint64_t kCapabilityProbeHostBytes = 8 * 1024 * 1024;

  PublishedGeoWriter(const std::filesystem::path& fresh_root,
                     std::vector<std::string> references,
                     const std::atomic<bool>* cancel = nullptr);
  PublishedGeoWriter(const PublishedGeoWriter&) = delete;
  PublishedGeoWriter& operator=(const PublishedGeoWriter&) = delete;
  void Publish(size_t job, const Mat<float>& depth, const Mat<float>& normal);
  // Identity/residency only. Never reads bodies or advises old files.
  void Verify() const;
  void VerifyComplete() const;
  void QualifyCache();
  const std::vector<PublishedGeoPair>& Products() const { return products_; }
  const std::vector<std::string>& References() const { return references_; }
  // Execution only, after approval and before solver allocation. Same real
  // writer/checker, fresh private scratch; no requested output during planning.
  static void Probe(const std::filesystem::path& directory,
                    const std::atomic<bool>* cancel = nullptr);

 private:
  void Check() const;
  PublishedGeoFile Write(const std::filesystem::path&, const Mat<float>&);
  std::filesystem::path root_;
  std::vector<std::string> references_;
  std::vector<PublishedGeoPair> products_;
  std::vector<PublishedGeoFile> files_;
  const std::atomic<bool>* cancel_;
  bool cache_qualified_ = false;
  // Logical verification state: const queries can observe irreversible failure.
  mutable bool failed_ = false;
};
}  // namespace colmap::mvs::owned
