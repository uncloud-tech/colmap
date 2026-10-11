// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace colmap::mvs::owned {

// This reservation is additional to BOTH active owners and process credits.
// Subtraction avoids overflow, including malformed caller budgets.
inline bool CanReserveHostLookahead(uint64_t capacity,
                                    uint64_t process,
                                    uint64_t a,
                                    uint64_t b,
                                    uint64_t extra) {
  for (const uint64_t charge : {process, a, b, extra}) {
    if (charge > capacity) return false;
    capacity -= charge;
  }
  return extra != 0;
}

struct HostPreparationThreadFactory {
  std::thread operator()(std::function<void()> task) const {
    return std::thread(std::move(task));
  }
};

// One worker, one future pair, no device/coordinator access. T must own only
// host data before consumption. The caller must join before next admission,
// then consume only after that admission charges the new active owners.
// Keep the full transient charge until ownership transfer or joined cleanup;
// no early release based on sampled usage or estimated retained payload.
template <class T>
class HostLookahead {
 public:
  using Pair = std::array<std::unique_ptr<T>, 2>;
  using Observer = std::function<void(const char*, int, uint64_t)>;

  HostLookahead(uint64_t charge, Observer observer)
      : charge_(charge), observer_(std::move(observer)) {
    if (charge == 0) throw std::invalid_argument("zero lookahead charge");
  }
  HostLookahead(const HostLookahead&) = delete;
  HostLookahead& operator=(const HostLookahead&) = delete;
  ~HostLookahead() { CancelAndJoin(); }

  template <class Prepare, class Factory = HostPreparationThreadFactory>
  void Start(Prepare prepare, Factory factory = {}) {
    if (started_) throw std::logic_error("lookahead already started");
    started_ = true;
    reserved_ = true;
    try {
      Notify("reserved");
      worker_ = factory([this, prepare = std::move(prepare)]() mutable {
        try {
          Notify("prep_start");
          for (size_t job = 0; job != 2; ++job) {
            CheckCancelled();
            Notify("job_start", static_cast<int>(job));
            prepared_[job] = prepare(job, cancelled_);
            if (!prepared_[job]) {
              throw std::runtime_error("empty lookahead preparation");
            }
            CheckCancelled();
            Notify("job_done", static_cast<int>(job));
          }
          ready_ = true;
          Notify("prep_done");
        } catch (...) {
          error_ = std::current_exception();
          try { Notify("failed"); } catch (...) {}
        }
      });
    } catch (...) {
      const auto error = std::current_exception();
      try { Notify("failed"); } catch (...) {}
      CancelAndJoin();
      std::rethrow_exception(error);
    }
  }

  // Called before creating the next epoch's coordinator. An error must never
  // cross the next GPU admission boundary.
  void Join() {
    if (!started_) throw std::logic_error("lookahead not started");
    if (worker_.joinable()) worker_.join();
    if (error_) {
      const auto error = error_;
      CancelAndJoin();
      std::rethrow_exception(error);
    }
    if (!ready_) throw std::logic_error("lookahead not ready");
  }

  Pair Consume() {
    if (worker_.joinable() || !ready_ || !reserved_ || error_) {
      throw std::logic_error("lookahead consumed before joined readiness");
    }
    Notify("consume");
    Pair result = std::move(prepared_);
    ready_ = false;
    reserved_ = false;
    Notify("release");
    return result;
  }

  void CancelAndJoin() noexcept {
    cancelled_.store(true, std::memory_order_relaxed);
    if (worker_.joinable()) worker_.join();
    prepared_ = {};
    ready_ = false;
    if (reserved_) {
      reserved_ = false;
      try { Notify("release"); } catch (...) {}
    }
  }

 private:
  void CheckCancelled() const {
    if (cancelled_.load(std::memory_order_relaxed)) {
      throw std::runtime_error("host lookahead cancelled");
    }
  }
  void Notify(const char* event, int job = -1) {
    if (observer_) observer_(event, job, reserved_ ? charge_ : 0);
  }

  const uint64_t charge_;
  Observer observer_;
  Pair prepared_;
  std::thread worker_;
  std::atomic<bool> cancelled_{false};
  std::exception_ptr error_;
  bool started_ = false;
  bool reserved_ = false;
  bool ready_ = false;
};

}  // namespace colmap::mvs::owned
