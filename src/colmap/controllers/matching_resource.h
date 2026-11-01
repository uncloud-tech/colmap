// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/estimators/solvers/homography_matrix.h"
#include "colmap/geometry/pose.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/scene/database.h"
#include "colmap/util/hash_containers.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace colmap {
inline uint64_t MatchingAdd(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("matching payload addition overflow");
  return a + b;
}
inline uint64_t MatchingMultiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::overflow_error("matching payload multiplication overflow");
  return a * b;
}
inline uint64_t MatchingMargined(uint64_t known, uint64_t policy, uint64_t reserve) {
  const auto raw = MatchingAdd(known, policy);
  return MatchingAdd(MatchingAdd(raw, MatchingAdd(MatchingMultiply(raw / 100, 3),
                                                (raw % 100 * 3 + 99) / 100)), reserve);
}

// Logical live known payload, not RSS or a physical workspace guarantee. The
// producer ALSO writes results: TryAcquire must never wait while owning jobs.
class MatchingLedger : public std::enable_shared_from_this<MatchingLedger> {
 public:
  struct Ticket {
    std::shared_ptr<MatchingLedger> ledger;
    uint64_t bytes;
    ~Ticket() {
      std::lock_guard<std::mutex> lock(ledger->mutex_);
      ledger->used -= bytes;
    }
    void Grow(uint64_t additional) {
      std::lock_guard<std::mutex> lock(ledger->mutex_);
      const auto next = MatchingAdd(ledger->used.load(), additional);
      if (!ledger->Fits(next))
        throw std::runtime_error("matching logical host growth refused; workers must drain");
      ledger->used = next;
      ledger->peak = std::max(ledger->peak, next);
      bytes = MatchingAdd(bytes, additional);
    }
  };
  MatchingLedger(uint64_t capacity, uint64_t policy, uint64_t reserve)
      : capacity_(capacity), policy_(policy), reserve_(reserve) {}
  bool Fits(uint64_t bytes) const {
    return MatchingMargined(bytes, policy_, reserve_) <= capacity_;
  }
  void LowerCapacity(uint64_t observed) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = std::min(capacity_, observed);
    if (!Fits(used.load()))
      throw std::runtime_error("matching logical persistent payload refused after CUDA host observation");
  }
  uint64_t Capacity() const { return capacity_; }
  std::shared_ptr<Ticket> TryAcquire(uint64_t bytes) {
    auto ticket = std::shared_ptr<Ticket>(new Ticket{shared_from_this(), 0});
    std::lock_guard<std::mutex> lock(mutex_);
    if (!Fits(bytes)) throw std::runtime_error("matching logical singleton refused (not a physical-fit guarantee)");
    if (!Fits(MatchingAdd(used, bytes))) return nullptr;
    used = MatchingAdd(used, bytes);
    peak = std::max(peak, used.load());
    ticket->bytes = bytes;
    return ticket;
  }
  // The controller admits/grows credits. Tickets may retire on worker failure
  // too; retirement is synchronized. Clear queued owners AFTER workers join.
  std::atomic<uint64_t> used{0};
  uint64_t peak = 0;
 private:
  std::mutex mutex_;
  uint64_t capacity_, policy_, reserve_;
};

struct MatchingInventory {
  FlatHashMap<image_t, Database::MatchingFeatureShape> images;
  FlatHashMap<image_pair_t, uint64_t> stored_matches;
  uint64_t cache_payload = 0, index_payload = 0, max_rows = 0;
  uint64_t largest_cache_payload = 0, largest_index_payload = 0;

  static MatchingInventory Read(Database& database, const FeatureMatchingOptions& options) {
    MatchingInventory inventory;
    uint64_t total_keypoints = 0, total_descriptors = 0;
    for (const auto& image : database.ReadAllImages()) {
      const auto id = image.ImageId();
      const auto s = database.ReadMatchingFeatureShape(id);
      if ((s.keypoint_cols != 0 && s.keypoint_cols != 2 && s.keypoint_cols != 4 && s.keypoint_cols != 6) ||
          (s.keypoint_rows && !s.keypoint_cols) ||
          (s.descriptor_cols != 0 && s.descriptor_cols != 128) ||
          (s.descriptor_rows && (s.descriptor_cols != 128 || s.type != FeatureExtractorType::SIFT)) ||
          (!s.descriptor_rows && s.type != FeatureExtractorType::SIFT && s.type != FeatureExtractorType::UNDEFINED) ||
          s.keypoint_rows != s.descriptor_rows ||
          s.keypoint_bytes != MatchingMultiply(MatchingMultiply(s.keypoint_rows, s.keypoint_cols), sizeof(float)) ||
          s.descriptor_bytes != MatchingMultiply(s.descriptor_rows, s.descriptor_cols) ||
          s.keypoint_rows > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("budgeted matching requires consistent SIFT DB feature shapes");
      total_keypoints = MatchingAdd(total_keypoints, s.keypoint_rows);
      total_descriptors = MatchingAdd(total_descriptors, s.descriptor_rows);
      inventory.images.emplace(id, s);
      inventory.max_rows = std::max(inventory.max_rows, s.keypoint_rows);
      // Converted keypoints plus read-BLOB overlap, and byte descriptors.
      const auto cache_bytes = MatchingAdd(MatchingAdd(MatchingMultiply(s.keypoint_rows, sizeof(FeatureKeypoint)), s.keypoint_bytes),
                                          s.descriptor_bytes);
      inventory.cache_payload = MatchingAdd(inventory.cache_payload, cache_bytes);
      inventory.largest_cache_payload = std::max(inventory.largest_cache_payload, cache_bytes);
      if (!options.use_gpu && !options.sift->cpu_brute_force_matcher) {
        // FAISS source: ToFloat input + index storage + IDs + two centroid
        // tables (coarse quantizer and training input). Library workspace,
        // list capacity and bookkeeping remain explicit policy UNKNOWN.
        const auto n = s.descriptor_rows;
        auto bytes = MatchingMultiply(n, 128 * sizeof(float));
        bytes = MatchingAdd(bytes, MatchingMultiply(n, n < 512 ? 128 * sizeof(float) : 128 + sizeof(int64_t)));
        if (n >= 512) bytes = MatchingAdd(bytes,
            MatchingMultiply(static_cast<uint64_t>(4 * std::sqrt(n)), 2 * 128 * sizeof(float)));
        inventory.index_payload = MatchingAdd(inventory.index_payload, bytes);
        inventory.largest_index_payload = std::max(inventory.largest_index_payload, bytes);
      }
    }
    if (total_keypoints != database.NumKeypoints() || total_descriptors != database.NumDescriptors() ||
        inventory.max_rows != database.MaxNumKeypoints())
      throw std::invalid_argument("budgeted matching refuses orphan/inconsistent DB feature inventory");
    for (const auto& [pair, count] : database.ReadNumMatches()) {
      if (count < 0) throw std::invalid_argument("matching DB match count exceeds supported range");
      inventory.stored_matches.emplace(pair, count);
    }
    return inventory;
  }
  uint64_t MatchLimit(const FeatureMatchingOptions& options) const {
    return std::min<uint64_t>(options.max_num_matches, max_rows);
  }
  uint64_t HostPersistent(const FeatureMatchingOptions& options, size_t workers,
                          size_t verifiers = 1) const {
    // CU host sift_buffer retains capacity across pairs. Only non-guided byte
    // descriptors reach it here: GetBestMatch needs <=2*M ints. Reserve old+
    // new overlap once per worker; host storage is NOT device memory.
    const auto staging = options.use_gpu ? MatchingMultiply(MatchLimit(options), 4 * sizeof(int)) : 0;
    // LRU eviction does not destroy objects still shared by workers. Reserve
    // four image loads per matcher/verifier (both sides plus load/old overlap),
    // and four CPU indexes per matcher (two-entry private cache plus two old
    // local references during replacement). Once-per-DB terms alone would
    // incorrectly credit evicted but still live objects.
    auto bytes = MatchingAdd(MatchingAdd(cache_payload, index_payload), MatchingMultiply(staging, workers));
    bytes = MatchingAdd(bytes, MatchingMultiply(largest_cache_payload,
        MatchingMultiply(4, MatchingAdd(workers, verifiers))));
    return MatchingAdd(bytes, MatchingMultiply(largest_index_payload, MatchingMultiply(4, workers)));
  }
  uint64_t DevicePersistent(const FeatureMatchingOptions& options) const {
    if (!options.use_gpu) return 0;
    // SiftMatchCU::Allocate: M rounded to 32, 2 descriptor textures
    // (8M*4 floats), 2 locations (M*2 floats), M*M dot, match rows,
    // and optional CRT(M,ceil(M/8),32), clamped to FOUR channels by
    // CuTexImage::InitTexture. CUDA allocator/texture handles remain policy.
    if (max_rows > MatchLimit(options))
      throw std::invalid_argument("budgeted GPU matching refuses descriptor truncation at max_num_matches");
    if (!MatchLimit(options))
      throw std::invalid_argument("budgeted CUDA matching requires nonzero DB features and match limit");
    const auto m = MatchingMultiply(MatchingAdd(MatchLimit(options), 31) / 32, 32);
    auto bytes = MatchingAdd(MatchingMultiply(m, 2 * (8 * 4 + 2) * sizeof(float)),
                             MatchingMultiply(MatchingMultiply(m, m), sizeof(float)));
    bytes = MatchingAdd(bytes, MatchingMultiply(m, sizeof(float)));
    if (options.sift->cross_check) bytes = MatchingAdd(bytes,
        MatchingAdd(MatchingMultiply(MatchingMultiply(m, (m + 7) / 8), 4 * sizeof(float)),
                    MatchingMultiply(m, sizeof(float))));
    if (MatchingMultiply(m, m) >= static_cast<uint64_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("matching CUDA texture index range exceeded");
    return bytes;
  }
  uint64_t PairPayload(image_t a, image_t b, const FeatureMatchingOptions& options) const {
    const auto n1 = images.at(a).keypoint_rows, n2 = images.at(b).keypoint_rows;
    const auto sum = MatchingAdd(n1, n2);
    // Raw matches, by-value geometry input, inlier output and DB write BLOB
    // overlap. Existing DB matches are NOT bounded by max_num_matches.
    auto count = options.use_gpu ? MatchLimit(options) : n1;
    auto existing = stored_matches.find(ImagePairToPairId(a, b));
    if (existing != stored_matches.end()) count = std::max(count, existing->second);
    auto bytes = MatchingMultiply(count, 4 * sizeof(FeatureMatch));
    // Complete verifier keypoint->point vectors, not just matched features.
    bytes = MatchingAdd(bytes, MatchingMultiply(sum, 2 * sizeof(double)));
    // Known verifier-owned arrays: corresponding image points, bearing plus
    // Jacobian, spherical homography ray layouts, and three model masks.
    // Reserve the sum even though not every branch owns all simultaneously.
    bytes = MatchingAdd(bytes, MatchingMultiply(count,
        2 * sizeof(Eigen::Vector2d) + 2 * sizeof(CamRayWithJac) +
        sizeof(Eigen::Vector3d) + sizeof(CamRayWithImgPoint) + 3 * sizeof(char) +
        2 * sizeof(Eigen::Vector2d) + 3 * sizeof(double)));
    if (!options.use_gpu) {
      if (options.sift->cpu_brute_force_matcher) {
        bytes = MatchingAdd(bytes, MatchingMultiply(MatchingMultiply(n1, n2), sizeof(float)));
        bytes = MatchingAdd(bytes, MatchingMultiply(sum, 128 * sizeof(int)));
      } else {
        // Float query, two nearest-neighbour distances + int/int64 index
        // overlap and reciprocal arrays, conservatively both sides at once.
        bytes = MatchingAdd(bytes, MatchingMultiply(sum, 128 * sizeof(float) + 2 * (sizeof(float) + sizeof(int) + sizeof(int64_t))));
      }
      bytes = MatchingAdd(bytes, MatchingMultiply(sum, sizeof(int)));
    }
    // Estimator scratch (rays, residuals, solver/RANSAC internals), FAISS
    // workspaces, vector allocator slack, SQLite pages: UNKNOWN policy, not
    // coefficients copied from extraction or a theoretical physical bound.
    return bytes;
  }
};

// Shared exception channel, not a new executor. A failure stops all existing
// queues, waking output producers and the controller's blocking result pop.
struct MatchingFailure {
  std::mutex mutex;
  std::exception_ptr error;
  std::function<void()> stop_queues;
  std::function<void()> before_match;
  void Fail(std::exception_ptr next) {
    { std::lock_guard<std::mutex> lock(mutex); if (!error) error = next; }
    stop_queues();
  }
  void Rethrow() {
    std::lock_guard<std::mutex> lock(mutex);
    if (error) std::rethrow_exception(error);
  }
};
struct MatchingResources {
  MatchingInventory inventory;
  std::shared_ptr<MatchingLedger> host;
  std::shared_ptr<MatchingLedger::Ticket> persistent_host;
  std::vector<std::shared_ptr<MatchingLedger>> devices;
  std::vector<std::shared_ptr<MatchingLedger::Ticket>> persistent_devices;
};
}  // namespace colmap
