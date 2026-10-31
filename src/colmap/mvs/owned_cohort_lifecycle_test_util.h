// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace colmap::mvs::owned {

// Test-only first-cohort rendezvous at existing owner/sweep observer seams.
// All first N owners must remain alive together inside the callback before any
// can leave. This proves owner lifetime overlap, NOT GPU kernel overlap/speed.
class CohortLifecycleRendezvous {
 public:
  explicit CohortLifecycleRendezvous(
      size_t target,
      std::chrono::milliseconds timeout = std::chrono::seconds(10))
      : target_(target), timeout_(timeout) {}

  void Observe(size_t phase, size_t job) {
    if (job >= target_) return;
    std::unique_lock<std::mutex> lock(mutex_);
    ++observed_.at(phase);
    cv_.notify_all();
    if (!cv_.wait_for(lock, timeout_, [&] {
          return timed_out_ || observed_[phase] == target_;
        })) {
      timed_out_ = true;
      cv_.notify_all();
    }
    if (timed_out_)
      throw std::runtime_error("owner lifecycle rendezvous timed out");
  }

  // Read only after all owner threads have joined.
  size_t Observed(size_t phase) const { return observed_.at(phase); }

 private:
  const size_t target_;
  const std::chrono::milliseconds timeout_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::array<size_t, 2> observed_{};
  bool timed_out_ = false;
};

}  // namespace colmap::mvs::owned
