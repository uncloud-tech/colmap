// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/generated/photo_phase_fingerprint.h"
#include "colmap/mvs/optimized_photo_dispatch.h"

#include <string_view>

namespace colmap::mvs {
// Trusted native/model-side observation, NEVER populated from launch forecasts.
// options_bytes is OptimizedPhotoOptions(effective options, workspace, config).
// Charges are the canonical model result for this exact effective order,
// already carrying the producer's margin. No capacity, N, k, or extra margin
// lives here.
struct PhotoPhaseV1Job {
  std::string options_bytes;
  owned::ByteCharge process;
  owned::ByteCharge private_charge;
  int reference_image_idx;
  std::vector<int> source_image_idxs;
};
using PhotoPhaseV1Hash = std::function<std::string(const std::string_view&)>;

// The caller supplies independent native/model evidence and ComputeSHA256 (the
// same hashing hook used by the resident runner). This is not a transport or an
// authorization token. Missing evidence fails closed; it is never inferred from
// a syntactically valid forecast. Snapshot inputs by value to prevent
// relabeling or later mutation. Default absent delivery in dispatch is
// deliberately intact.
inline TrustedPhotoPhaseV1Delivery MakeTrustedPhotoPhaseV1Delivery(
    std::string exact_launch_bytes,
    std::vector<PhotoPhaseV1Binding> bindings,
    std::vector<PhotoPhaseV1Job> effective_jobs,
    PhotoPhaseV1Hash sha256) {
  return [bytes = std::move(exact_launch_bytes),
          bindings = std::move(bindings),
          jobs = std::move(effective_jobs),
          sha256 = std::move(sha256)](const std::string& launch_bytes) {
    if (bytes != launch_bytes)
      throw std::invalid_argument("PHOTO delivery launch bytes mismatch");
    if (bindings.empty() || jobs.empty() || !sha256)
      throw std::invalid_argument("PHOTO delivery missing");
    if (bindings.size() != jobs.size())
      throw std::invalid_argument("PHOTO delivery count or order mismatch");
    for (size_t i = 0; i < bindings.size(); ++i) {
      const auto& b = bindings[i];
      const auto& job = jobs[i];
      if (b.profile != "photo-phase-v1")
        throw std::invalid_argument("PHOTO delivery unknown profile");
      if (b.lineage_sha.size() != 40 ||
          b.lineage_sha.find_first_not_of("0123456789abcdef") !=
              std::string::npos)
        throw std::invalid_argument("PHOTO delivery malformed lineage");
      if (!OptimizedDigest(b.fingerprint))
        throw std::invalid_argument("PHOTO delivery malformed fingerprint");
      if (!OptimizedDigest(b.options_sha256))
        throw std::invalid_argument("PHOTO delivery malformed options");
      // The build recomputes this closure over kPhotoPhaseV1SourceCommit.
      // The lineage is an ancestry base, NOT the revision being fingerprinted.
      if (b.source_commit.size() != 40 ||
          b.source_commit.find_first_not_of("0123456789abcdef") !=
              std::string::npos ||
          b.source_commit != kPhotoPhaseV1SourceCommit)
        throw std::invalid_argument("source_commit-revision-mismatch");
      if (b.fingerprint != kPhotoPhaseV1Fingerprint)
        throw std::invalid_argument("lineage-source_commit-incoherent");
      if (b.lineage_sha != kPhotoPhaseV1Lineage)
        throw std::invalid_argument("PHOTO delivery lineage incompatible");
      if (job.options_bytes.empty() ||
          b.options_sha256 != sha256(job.options_bytes))
        throw std::invalid_argument("PHOTO delivery options mismatch");
      if (b.reference_image_idx != job.reference_image_idx)
        throw std::invalid_argument("PHOTO delivery count or order mismatch");
      if (b.source_image_idxs != job.source_image_idxs)
        throw std::invalid_argument("PHOTO delivery per-reference mismatch");
      const auto nonzero = [](owned::ByteCharge c) {
        return c.host && c.device;
      };
      if (!nonzero(b.process) || !nonzero(b.private_charge) ||
          !nonzero(job.process) || !nonzero(job.private_charge))
        throw std::invalid_argument("PHOTO delivery absent or zero charges");
      if (b.process.host != job.process.host ||
          b.process.device != job.process.device ||
          b.private_charge.host != job.private_charge.host ||
          b.private_charge.device != job.private_charge.device)
        throw std::invalid_argument("PHOTO delivery canonical charge mismatch");
    }
    return bindings;
  };
}
// CLI supply point: the producer writes typed objects verbatim into V3.
// Charges are delivered model results, not derived from forecast numbers.
// Effective options are serialized by the native CLI for each actual config.
inline TrustedPhotoPhaseV1Delivery MakeWirePhotoPhaseV1Delivery(
    std::filesystem::path config,
    std::string requested,
    owned::ByteCharge capacity,
    std::function<std::string(const std::filesystem::path&)> native_options,
    PhotoPhaseV1Hash sha256) {
  return [config = std::move(config),
          requested = std::move(requested),
          capacity,
          native_options = std::move(native_options),
          sha256 = std::move(sha256)](const std::string& bytes) {
    // Preserve the absent-delivery refusal before any forecast admission.
    // Raw section framing cannot occur inside the hex-encoded job fields.
    if (bytes.rfind("COLMAP_OPTIMIZED_PHOTO_V3\n", 0) != 0 ||
        bytes.find("\n[BINDING") == std::string::npos)
      return std::vector<PhotoPhaseV1Binding>{};
    const auto launch =
        ParseOptimizedPhotoLaunch(bytes, config, requested, capacity);
    if (launch.bindings.empty()) return std::vector<PhotoPhaseV1Binding>{};
    std::vector<PhotoPhaseV1Job> jobs;
    for (size_t i = 0; i < launch.jobs.size(); ++i) {
      const auto& b = launch.bindings.at(i);
      jobs.push_back({native_options(launch.jobs[i].config),
                      b.process,
                      b.private_charge,
                      b.reference_image_idx,
                      b.source_image_idxs});
    }
    return MakeTrustedPhotoPhaseV1Delivery(
        bytes, launch.bindings, std::move(jobs), sha256)(bytes);
  };
}
// In-process RUNNER handoff from the staged sealed plan, never a disk/CLI
// transport. Jobs and bindings are in launch order; the producer has already
// applied its margin to both charge objects. The consumer adds no margin.
struct PhotoPhaseV1DeliveryInputs {
  std::string exact_launch_bytes;
  std::vector<PhotoPhaseV1Binding> bindings;
  std::vector<PhotoPhaseV1Job> effective_jobs;
  PhotoPhaseV1Hash sha256;
};
using PhotoPhaseV1RunnerDelivery =
    std::function<PhotoPhaseV1DeliveryInputs(const std::string&)>;

inline TrustedPhotoPhaseV1Delivery MakeTrustedPhotoPhaseV1Delivery(
    const PhotoPhaseV1RunnerDelivery& runner_delivery) {
  if (!runner_delivery) return {};
  return [runner_delivery](const std::string& exact_launch_bytes) {
    auto delivered = runner_delivery(exact_launch_bytes);
    return MakeTrustedPhotoPhaseV1Delivery(
        std::move(delivered.exact_launch_bytes),
        std::move(delivered.bindings),
        std::move(delivered.effective_jobs),
        std::move(delivered.sha256))(exact_launch_bytes);
  };
}
}  // namespace colmap::mvs
