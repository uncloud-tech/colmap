// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace colmap::mvs {

// Copy this value, never recreate it at a reference/cohort boundary.
struct NativeRunDeadline {
  using Clock = std::chrono::steady_clock;
  Clock::time_point entry;
  Clock::time_point end;
  std::chrono::milliseconds budget;

  std::chrono::milliseconds Remaining() const {
    return std::max(std::chrono::milliseconds::zero(),
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        end - Clock::now()));
  }

  static NativeRunDeadline FromBudget(std::chrono::milliseconds budget,
                                      Clock::time_point entry = Clock::now()) {
    if (budget.count() <= 0 ||
        budget > std::chrono::duration_cast<std::chrono::milliseconds>(
                     Clock::time_point::max() - entry))
      throw std::invalid_argument("owned run deadline budget out of range");
    return {entry, entry + budget, budget};
  }
};

// Covers blocking setup, CUDA drain and producer/owner joins outside the
// coordinator. On expiry, do NOT unwind retained storage or detach workers.
// This is an execution budget, not a second cleanup allowance.
class NativeRunDeadlineGuard {
 public:
  explicit NativeRunDeadlineGuard(std::optional<NativeRunDeadline> deadline)
      : deadline_(deadline) {
    if (deadline_) {
      worker_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_until(lock, deadline_->end, [this] { return done_; })) {
          std::fprintf(
              stderr,
              "owned run: absolute phase deadline expired; budget=%lld ms; "
              "setup/work/drain/join completion not proven; terminating "
              "without unwinding uncertain owners\n",
              static_cast<long long>(deadline_->budget.count()));
          std::fflush(stderr);
          std::abort();
        }
      });
    }
  }
  NativeRunDeadlineGuard(const NativeRunDeadlineGuard&) = delete;
  NativeRunDeadlineGuard& operator=(const NativeRunDeadlineGuard&) = delete;
  ~NativeRunDeadlineGuard() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done_ = true;
      cv_.notify_all();
    }
    if (worker_.joinable()) worker_.join();
  }
  void Check() const {
    if (deadline_ && NativeRunDeadline::Clock::now() >= deadline_->end)
      throw std::runtime_error(
          "owned run absolute phase deadline expired before success/admission");
  }

 private:
  std::optional<NativeRunDeadline> deadline_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool done_ = false;
  std::thread worker_;
};

// CLI-independent policy seam, also used by the actual command before work.
inline std::optional<NativeRunDeadline> ResolveOwnedRunDeadline(
    const std::string& seconds,
    bool optimized_only,
    bool owned_execution,
    NativeRunDeadline::Clock::time_point entry) {
  if (seconds.empty()) {
    if (optimized_only)
      throw std::invalid_argument(
          "optimized_only requires --owned_run_timeout_seconds (positive "
          "canonical integer)");
    return {};
  }
  if (!owned_execution)
    throw std::invalid_argument(
        "owned_run_timeout_seconds requires owned execution, not "
        "metadata/ordinary fallback");
  const auto invalid = [] {
    return std::invalid_argument(
        "owned_run_timeout_seconds must be a positive canonical integer within "
        "steady-clock range");
  };
  if (seconds.front() < '1' || seconds.front() > '9') throw invalid();
  uint64_t value = 0;
  for (const char c : seconds) {
    if (c < '0' || c > '9' ||
        value > (std::numeric_limits<uint64_t>::max() - (c - '0')) / 10)
      throw invalid();
    value = value * 10 + (c - '0');
  }
  const auto maximum = std::chrono::duration_cast<std::chrono::seconds>(
                           NativeRunDeadline::Clock::time_point::max() - entry)
                           .count();
  if (maximum < 1 || value > static_cast<uint64_t>(maximum)) throw invalid();
  return NativeRunDeadline::FromBudget(std::chrono::milliseconds(value * 1000),
                                       entry);
}

}  // namespace colmap::mvs
