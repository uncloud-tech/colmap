// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Single-controller ownership and the serialized selection boundary for
// in-process PHOTO N1..Nk.
//
// Each prepared PHOTO solver owns its distinct-source device state. The
// production shared controller leases its complete CPU preparation, including
// workspace/LRU creation, decode, snapshot and destruction, through this
// boundary. No scheduler, kernel or numerical operation is changed.
//
// Verdict encoded by this header (the accompanying test carries the
// deterministic proof): the raw shared LRU is NOT proven race safe, so
// selection state is partitioned per owner and reachable only under an
// exclusive boundary lease. Production PHOTO acquires this boundary around
// all mutable controller preparation, not just this host-side cache model.
//
// Standard-library only (no CUDA, no COLMAP logging) so the contract can be
// exercised by bounded standalone CPU tests, mirroring
// owned_pair_coordinator.h.

#include "colmap/mvs/optimized_dispatch.h"
#include "colmap/mvs/owned_pair_coordinator.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace colmap::mvs::owned {

// Compiled slot bound of the owned distinct-source selection cache: one slot
// per processed sample and OwnedSourceCache::kMaxSamples == 15 samples, so the
// kernel always indexes inside 0..14. Exceeding it is a contract violation, not
// a wraparound, and is refused here instead of silently overwriting a slot.
inline constexpr int kMaxSelectionSlots = 15;

// Host-side model of that compiled rolling selection cache:
//   * linear probe of the slots in insertion order = a hit on the first match;
//   * a miss appends one slot (insertion order is the slot order);
//   * a repeat is a hit and never grows the cache;
//   * a new distinct source beyond the compiled slot bound refuses.
// Deliberately NOT thread safe: owners must reach it through
// OwnedSharedController::WithSelection, i.e. under the boundary lease.
class OwnedRollingSelectionCache {
 public:
  explicit OwnedRollingSelectionCache(int capacity = kMaxSelectionSlots)
      : capacity_(capacity) {
    if (capacity_ < 1 || capacity_ > kMaxSelectionSlots) {
      throw std::invalid_argument(
          "owned selection cache capacity outside the compiled slot bound");
    }
  }

  // Returns the slot index for `source`; `*hit` reports whether the slot was
  // already resident. Refuses a new distinct source past the compiled bound.
  int Select(int source, bool* hit) {
    for (int slot = 0; slot < Size(); ++slot) {
      if (slots_[static_cast<size_t>(slot)] == source) {
        if (hit != nullptr) *hit = true;
        ++hits_;
        return slot;
      }
    }
    if (Size() >= capacity_) {
      throw std::length_error(
          "owned selection cache exceeded the compiled distinct-source slots");
    }
    slots_.push_back(source);
    ++misses_;
    if (hit != nullptr) *hit = false;
    return Size() - 1;
  }

  void Reset() {
    slots_.clear();
    hits_ = misses_ = 0;
  }
  int Size() const noexcept { return static_cast<int>(slots_.size()); }
  int Capacity() const noexcept { return capacity_; }
  const std::vector<int>& Slots() const noexcept { return slots_; }
  uint64_t Hits() const noexcept { return hits_; }
  uint64_t Misses() const noexcept { return misses_; }

 private:
  const int capacity_;
  std::vector<int> slots_;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
};

// Exclusive selection boundary. At most one owner mutates selection state at a
// time; a nested acquisition from the same thread would deadlock, so it refuses
// instead of hanging. Counters are telemetry for the completion receipt.
class OwnedSelectionBoundary {
 public:
  explicit OwnedSelectionBoundary(size_t owners) : owners_(owners) {
    if (!owners_ ||
        owners_ > static_cast<size_t>(kOptimizedRollingStructuralMax)) {
      throw std::invalid_argument(
          "owned selection boundary requires 1..compiled-rolling owners");
    }
  }

  OwnedSelectionBoundary(const OwnedSelectionBoundary&) = delete;
  OwnedSelectionBoundary& operator=(const OwnedSelectionBoundary&) = delete;

  size_t Owners() const noexcept { return owners_; }

  // Runs `body` while holding the boundary lease. `owner` must be a declared
  // owner; exceptions from `body` propagate after the lease is released.
  template <class Body>
  decltype(auto) Run(size_t owner, Body&& body) {
    if (owner >= owners_) {
      throw std::invalid_argument("owned selection boundary: unknown owner");
    }
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (held_ && holder_ == std::this_thread::get_id()) {
        throw std::logic_error(
            "owned selection boundary: nested lease refused (would deadlock)");
      }
      if (held_) {
        ++waiters_;
        max_waiters_ = std::max(max_waiters_, waiters_);
      }
      idle_.wait(lock, [&] { return !held_; });
      if (waiters_) --waiters_;
      held_ = true;
      holder_ = std::this_thread::get_id();
      ++calls_;
    }
    struct Release {
      OwnedSelectionBoundary* boundary;
      ~Release() {
        std::lock_guard<std::mutex> lock(boundary->mutex_);
        boundary->held_ = false;
        boundary->holder_ = std::thread::id{};
        boundary->idle_.notify_all();
      }
    } release{this};
    return body();
  }

  // Leased bodies entered (successful acquisitions only).
  uint64_t SerializedCalls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return calls_;
  }
  // High-water mark of owners queued behind the lease (descriptive).
  size_t MaxWaiters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_waiters_;
  }

 private:
  const size_t owners_;
  mutable std::mutex mutex_;
  std::condition_variable idle_;
  bool held_ = false;
  std::thread::id holder_{};
  size_t waiters_ = 0;
  size_t max_waiters_ = 0;
  uint64_t calls_ = 0;
};

// ONE controller instance shared by `owners` in-process generations.
//
// Ownership rules, enforced rather than documented away:
//   * a declared owner enters at most one live generation at a time;
//   * by default the instance serves `owners` generations; rolling production
//     callers supply the frozen job count, with clean selection state per turn;
//   * selection state is owner-private and reachable only inside a live
//     generation, through the boundary lease;
//   * Seal requires every generation to have ended.
// Observed() and Intervals() are descriptive receipt telemetry, not resource
// bounds.
class OwnedSharedController {
 public:
  explicit OwnedSharedController(size_t owners,
                                 int selection_capacity = kMaxSelectionSlots,
                                 size_t generation_limit = 0)
      : boundary_(owners),
        owners_(owners),
        generation_limit_(generation_limit ? generation_limit : owners),
        live_(owners, false),
        interval_of_(owners, kNoInterval) {
    if (!owners_) {
      throw std::invalid_argument(
          "owned shared controller requires at least one owner");
    }
    for (size_t owner = 0; owner < owners_; ++owner) {
      selections_.emplace_back(selection_capacity);
    }
  }

  OwnedSharedController(const OwnedSharedController&) = delete;
  OwnedSharedController& operator=(const OwnedSharedController&) = delete;

  size_t Owners() const noexcept { return owners_; }

  // Enters the next in-process generation for `owner`; returns its generation
  // index.
  size_t Begin(size_t owner) {
    if (owner >= owners_) {
      throw std::invalid_argument("owned shared controller: unknown owner");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (sealed_) {
      throw std::logic_error("owned shared controller: instance is sealed");
    }
    if (live_[owner]) {
      throw std::logic_error(
          "owned shared controller: owner already has a live generation");
    }
    if (generations_ >= generation_limit_) {
      throw std::logic_error(
          "owned shared controller: no generation left (single-use instance)");
    }
    selections_.at(owner).Reset();
    live_[owner] = true;
    ++live_count_;
    observed_ = std::max(observed_, live_count_);
    const uint64_t now = Micros();
    intervals_.push_back({now, now});
    interval_of_[owner] = intervals_.size() - 1;
    return generations_++;
  }

  // Ends `owner`'s generation and closes its descriptive interval.
  void End(size_t owner) {
    if (owner >= owners_) {
      throw std::invalid_argument("owned shared controller: unknown owner");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!live_[owner]) {
      throw std::logic_error(
          "owned shared controller: owner has no live generation");
    }
    live_[owner] = false;
    --live_count_;
    const size_t index = interval_of_[owner];
    interval_of_[owner] = kNoInterval;
    intervals_.at(index).end_micros = Micros();
  }

  // No further generation may be claimed; requires every owner to have ended.
  void Seal() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (live_count_ != 0) {
      throw std::logic_error(
          "owned shared controller: seal while generations are live");
    }
    sealed_ = true;
  }

  bool Sealed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sealed_;
  }

  // Maximum simultaneous owners ever admitted (receipt `observed_n`).
  size_t Observed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return observed_;
  }

  // Generations admitted so far.
  size_t Generations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return generations_;
  }

  // Per-generation intervals in Begin order (receipt `owner_intervals`).
  std::vector<OptimizedOwnerInterval> Intervals() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return intervals_;
  }

  // Owner-private selection state, reachable only inside a live generation and
  // only under the exclusive boundary lease.
  template <class Body>
  decltype(auto) WithSelection(size_t owner, Body&& body) {
    if (owner >= owners_) {
      throw std::invalid_argument("owned shared controller: unknown owner");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!live_[owner]) {
        throw std::logic_error(
            "owned shared controller: selection outside a live generation");
      }
    }
    return boundary_.Run(
        owner, [&]() -> decltype(auto) { return body(selections_.at(owner)); });
  }

  // Boundary telemetry (descriptive).
  uint64_t SerializedSelections() const { return boundary_.SerializedCalls(); }
  size_t MaxWaiters() const { return boundary_.MaxWaiters(); }

 private:
  using Clock = std::chrono::steady_clock;
  static constexpr size_t kNoInterval = std::numeric_limits<size_t>::max();

  uint64_t Micros() const {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                              start_)
            .count());
  }

  OwnedSelectionBoundary boundary_;
  const size_t owners_;
  const size_t generation_limit_;
  std::vector<OwnedRollingSelectionCache> selections_;
  mutable std::mutex mutex_;
  std::vector<bool> live_;
  std::vector<size_t> interval_of_;
  std::vector<OptimizedOwnerInterval> intervals_;
  size_t live_count_ = 0;
  size_t observed_ = 0;
  size_t generations_ = 0;
  bool sealed_ = false;
  Clock::time_point start_ = Clock::now();
};

// Receipt telemetry built from the REAL coordinator event stream: adapter for
// PairCoordinator's sink. `admitted` opens an owner interval, `owner_joined`
// closes it, so `observed_n` and the intervals describe what was actually
// admitted and joined instead of what was requested. Descriptive only.
class OwnedConcurrencyObserver {
 public:
  OwnedConcurrencyObserver(int requested_n, int admitted_n)
      : requested_n_(requested_n), admitted_n_(admitted_n) {
    if (requested_n_ < 1 || admitted_n_ < 1 || requested_n_ < admitted_n_) {
      throw std::invalid_argument(
          "owned concurrency observer requires 1 <= admitted N <= requested N");
    }
  }

  // Sink-compatible: never fabricates an interval it did not observe.
  void operator()(const PairEvent& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t now = Micros();
    if (event.job != std::numeric_limits<size_t>::max() &&
        event.event == "admitted") {
      if (open_.count(event.job)) {
        throw std::logic_error("owned concurrency observer: duplicate admit");
      }
      open_[event.job] = intervals_.size();
      intervals_.push_back({now, now});
      ++live_;
      observed_ = std::max(observed_, live_);
    } else if (event.job != std::numeric_limits<size_t>::max() &&
               event.event == "owner_joined") {
      const auto found = open_.find(event.job);
      if (found == open_.end()) {
        throw std::logic_error(
            "owned concurrency observer: join without admitted owner");
      }
      intervals_.at(found->second).end_micros = now;
      open_.erase(found);
      if (live_) --live_;
    }
  }

  OptimizedConcurrencyObservation Observation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    OptimizedConcurrencyObservation observation;
    observation.requested_n = requested_n_;
    observation.admitted_n = admitted_n_;
    // An owner that never got admitted cannot be reported as observed; keep the
    // receipt honest with a floor of 1 instead of claiming zero parallelism.
    observation.observed_n = static_cast<int>(observed_ ? observed_ : 1);
    observation.owner_intervals = intervals_;
    return observation;
  }

  size_t Observed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return observed_;
  }

 private:
  using Clock = std::chrono::steady_clock;

  uint64_t Micros() const {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                              start_)
            .count());
  }

  const int requested_n_;
  const int admitted_n_;
  Clock::time_point start_ = Clock::now();
  mutable std::mutex mutex_;
  std::map<size_t, size_t> open_;
  std::vector<OptimizedOwnerInterval> intervals_;
  size_t live_ = 0;
  size_t observed_ = 0;
};

}  // namespace colmap::mvs::owned
