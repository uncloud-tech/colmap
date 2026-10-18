// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/owned_host_lookahead.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace colmap::mvs::owned {

// Bounded FIFO reference-normal prefetch for resident geometry owners.
//
// One CPU-only worker (never CUDA, never the coordinator) loads future
// generations at most `max_future` (strictly 1 or 2) ready-but-unhanded
// NormalMap buffers ahead of consumer progress, in immutable FIFO generation
// order over an immutable generation -> host-input-bank layer mapping,
// including the last generation. Each generation is loaded exactly once with
// no retry. Every buffer is a unique owned NormalMap: the producer queue owns
// it until Consume moves it to the single admitted owner generation, whose
// caller frame keeps it alive until owner retirement. There is no shared
// mutable payload vector and no second handoff of a generation.
//
// Host accounting: the caller reserves the pipeline charge (up to max_future
// buffers plus one decode transient; see
// HostInputBank::EstimateNormalAllowance) BEFORE any allocation; the
// reservation is reported through the observer at Start and released exactly
// once by CancelAndJoin/destruction, only after the in-flight read joined and
// every unhanded buffer is destroyed. Producer/ready/active buffers may
// coexist with owner forecasts; nothing here borrows live owner or process
// credits and the caller must reserve disjoint allowances to avoid double
// counting the same buffer holder.
//
// Failure model: cancellation is observed between reads; an in-flight read is
// always joined before any buffer or credit is released. The first loader
// exception is preserved and propagated to Join and to every Consume of an
// unproduced generation (no retry). A full ready queue is HEALTHY
// backpressure, never a failure: the producer's free-slot wait and every
// consumer handoff wait terminate only on Cancel()/cancellation, a stored
// producer error, or destruction — never on elapsed time. Because these
// waits are unbounded, the driver MUST wire coordinator/owner failure to
// Cancel() (prompt and nonjoining so owner threads cannot self-join the
// producer) and join only in parent cleanup (Join after owner drain, or the
// destructor). A blocked wait without such wiring is a driver bug, not a
// timeout path.
class NormalLookahead {
 public:
  static constexpr size_t kBatch = std::numeric_limits<size_t>::max();
  using Loader = std::function<NormalMap(size_t bank_layer,
                                         const std::atomic<bool>& cancelled)>;
  using Observer = std::function<void(const char* event, size_t generation,
                                      uint64_t reserved)>;

  // `bank_layers` is the immutable FIFO generation -> original bank layer
  // mapping; every entry is loaded once, in order. `max_future` bounds
  // ready-but-unhanded buffers to one or two. `charge` is the caller's
  // separately reserved host allowance for this pipeline; it is never
  // enforced here, only reported.
  NormalLookahead(std::vector<size_t> bank_layers, size_t max_future,
                  uint64_t charge, Observer observer = {})
      : bank_layers_(std::move(bank_layers)),
        buffers_(bank_layers_.size()),
        handed_(bank_layers_.size(), false),
        max_future_(max_future),
        charge_(charge),
        observer_(std::move(observer)) {
    if (max_future_ == 0 || max_future_ > 2) {
      throw std::invalid_argument(
          "bounded normal lookahead supports max_future of 1 or 2 only");
    }
    if (bank_layers_.empty()) {
      throw std::invalid_argument("empty normal lookahead mapping");
    }
  }
  NormalLookahead(const NormalLookahead&) = delete;
  NormalLookahead& operator=(const NormalLookahead&) = delete;
  ~NormalLookahead() { CancelAndJoin(); }

  size_t MaxFuture() const noexcept { return max_future_; }

  template <class Factory = HostPreparationThreadFactory>
  void Start(Loader loader, Factory factory = {}) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (started_) throw std::logic_error("normal lookahead already started");
      started_ = true;
      reserved_ = true;
      NotifyLocked("reserved", kBatch);
    }
    try {
      worker_ = factory([this, loader = std::move(loader)]() mutable {
        Produce(loader);
      });
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!first_error_) first_error_ = std::current_exception();
        NotifyLocked("failed", kBatch);
      }
      cv_.notify_all();
      CancelAndJoin();
      std::lock_guard<std::mutex> lock(mutex_);
      const auto error = first_error_;
      std::rethrow_exception(error);
    }
  }

  // Blocks until the unique buffer for exactly this FIFO generation is
  // ready, then moves it out. Each generation may be handed off exactly once,
  // only to the consumer that owns it. Producer loading order and per-
  // generation single handoff preserve the FIFO contract; handoff order of
  // distinct generations follows owner execution (owners are admitted FIFO).
  // The wait ends only on readiness, Cancel/cancellation, or a stored
  // producer error — never on elapsed time (backpressure is healthy).
  NormalMap Consume(size_t generation) {
    if (generation >= bank_layers_.size()) {
      throw std::invalid_argument("normal lookahead: invalid generation");
    }
    NormalMap normal;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!started_) throw std::logic_error("normal lookahead not started");
      if (handed_[generation]) {
        throw std::logic_error(
            "normal lookahead: duplicate generation handoff");
      }
      cv_.wait(lock, [&] {
        // An error stored by a later generation fails this unproduced one
        // immediately (no retry); a still-produced earlier buffer stays
        // deliverable so its owner can finish before the run unwinds.
        return cancelled_ || (first_error_ && produced_ <= generation) ||
               produced_ > generation;
      });
      if (produced_ > generation) {
        NotifyLocked("consume", generation);
        normal = std::move(buffers_[generation]);
        buffers_[generation] = NormalMap();
        handed_[generation] = true;
        ++handed_count_;
        cv_.notify_all();
        return normal;
      }
      if (first_error_) {
        std::rethrow_exception(first_error_);
      }
      throw std::runtime_error("normal lookahead cancelled");
    }
  }

  // Waits for the producer thread to terminate (full production including the
  // tail, cancellation, or failure), then propagates the first loader
  // exception. Full production requires consumption progress to unblock the
  // bounded queue; call only after all owners are done so the last
  // generations are handed off or the run has already failed. This is the
  // single join site for the success path.
  void Join() {
    if (!worker_.joinable()) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!started_) throw std::logic_error("normal lookahead not started");
    }
    if (worker_.joinable()) worker_.join();
    std::exception_ptr error;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (first_error_) error = first_error_;
    }
    if (error) {
      CancelAndJoin();
      std::rethrow_exception(error);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!fully_produced_) {
      throw std::runtime_error(
          "normal lookahead cancelled before full production");
    }
  }

  // Prompt, non-joining cancellation for coordinator/owner failure. Safe to
  // call from owner threads and more than once: it never joins, only marks
  // cancellation and wakes every producer/consumer wait. Joining the producer
  // happens exactly once in parent cleanup (Join or CancelAndJoin/
  // destruction), never concurrently with this signal. The explicit charge
  // stays reserved until CancelAndJoin — after the in-flight read joined and
  // owners drained — exactly as before.
  void Cancel() noexcept {
    cancelled_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(mutex_);
    cv_.notify_all();
  }

  // Joins the producer (including any in-flight read) after cancellation,
  // then destroys every unhanded buffer, then releases the charge exactly
  // once. Not for use from the producer thread or concurrently from multiple
  // cancellation sites; owner threads signal only Cancel().
  void CancelAndJoin() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    cancelled_ = true;
    cv_.notify_all();
    lock.unlock();
    if (worker_.joinable()) worker_.join();
    lock.lock();
    for (auto& buffer : buffers_) buffer = NormalMap();
    if (reserved_) {
      reserved_ = false;
      try {
        Notify("release", kBatch);
      } catch (...) {}
    }
  }

 private:
  void Produce(const Loader& loader) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto exit_locked = [&](bool produced_all) {
      fully_produced_ = produced_all;
      cv_.notify_all();
    };
    try {
      for (size_t generation = 0; generation < bank_layers_.size();
           ++generation) {
        // A full ready queue is healthy backpressure: wait for capacity,
        // cancellation, or a stored error, never on elapsed time.
        cv_.wait(lock, [&] {
          return cancelled_ || first_error_ ||
                 produced_ - handed_count_ < max_future_;
        });
        if (cancelled_ || first_error_) return exit_locked(false);
        lock.unlock();
        try {
          Notify("load_start", generation);
          NormalMap normal = loader(bank_layers_[generation], cancelled_);
          if (cancelled_) return;  // unique local buffer is destroyed here
          lock.lock();
          buffers_[generation] = std::move(normal);
        } catch (...) {
          lock.lock();
          if (!first_error_) first_error_ = std::current_exception();
          NotifyLocked("failed", generation);
          cv_.notify_all();
          exit_locked(false);
          return;
        }
        produced_ = generation + 1;
        NotifyLocked("load_done", generation);
        cv_.notify_all();
      }
      exit_locked(true);
      NotifyLocked("prep_done", kBatch);
    } catch (...) {
      if (!first_error_) first_error_ = std::current_exception();
      NotifyLocked("failed", kBatch);
      cv_.notify_all();
      exit_locked(false);
    }
  }

  void Notify(const char* event, size_t generation) {
    if (observer_) {
      observer_(event, generation, reserved_ ? charge_ : 0);
    }
  }
  void NotifyLocked(const char* event, size_t generation) {
    try {
      Notify(event, generation);
    } catch (...) {}
  }

  // Immutable mapping and per-generation unique buffers.
  const std::vector<size_t> bank_layers_;
  std::vector<NormalMap> buffers_;
  std::vector<bool> handed_;
  const size_t max_future_;
  const uint64_t charge_;
  Observer observer_;
  std::thread worker_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  size_t produced_ = 0;
  size_t handed_count_ = 0;
  bool started_ = false;
  bool reserved_ = false;
  bool fully_produced_ = false;
  // Observed by the worker between reads and by waiting consumers under the
  // same mutex; atomic so cancellation is visible to unlocked readers too.
  std::atomic<bool> cancelled_{false};
  std::exception_ptr first_error_;
};

}  // namespace colmap::mvs::owned