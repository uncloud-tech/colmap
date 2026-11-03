// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace colmap::mvs::owned {

// One finite phase's device set. Register every live coordinator/graph control
// BEFORE Run; callbacks must be noexcept in effect and only publish/wake, never
// wait for workers. Controls outlive Run and all callbacks. Cancellation does
// not destroy controls, release credits, retire jobs or claim a partial result.
class MultiGpuWorkers {
 public:
  void Register(std::function<void(std::exception_ptr)> cancel) {
    if (started_) throw std::logic_error("device controls already sealed");
    controls_.push_back(std::move(cancel));
  }

  void Cancel(std::exception_ptr error) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_) return;
    error_ = error ? error : std::make_exception_ptr(std::runtime_error("multi-GPU cancelled"));
    cancelled_.store(true);
    for (const auto& cancel : controls_) {
      try { cancel(error_); } catch (...) { /* Preserve first error; fan out. */ }
    }
  }

  // Borrow only through Run/drain. Existing bounded bank constructors consume
  // this same token, so peer failure also interrupts host preparation.
  const std::atomic<bool>* CancellationToken() const { return &cancelled_; }

  void Check() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_) std::rethrow_exception(error_);
  }

  struct ThreadFactory {
    std::thread operator()(std::function<void()> task) const {
      return std::thread(std::move(task));
    }
  };

  // RunRollingOwners drains locally before rethrowing a failed owner launch.
  // Publish to every device BEFORE that local drain, preserving the original
  // launch error and waking sibling admission/graph waiters immediately. The
  // generic task parameter keeps std::function conversion inside this try.
  template <class Factory = ThreadFactory>
  auto OwnerThreadFactory(Factory factory = {}) {
    return [this, factory](auto task) mutable {
      try {
        Check();
        auto owner = factory(std::move(task));
        if (!owner.joinable())
          throw std::runtime_error("owner factory returned no worker");
        return owner;
      } catch (...) {
        Cancel(std::current_exception());
        throw;
      }
    };
  }

  template <class Work, class Factory = ThreadFactory>
  void Run(size_t count, Work work, Factory factory = {},
           std::function<void()> external_check = {}) {
    if (started_) throw std::logic_error("device workers already started");
    started_ = true;
    std::vector<std::thread> threads;
    // Reserve BEFORE launch so vector growth cannot destroy a joinable thread.
    threads.reserve(count);
    std::mutex monitor_mutex;
    std::condition_variable monitor_wake;
    bool finished = false;
    std::thread monitor;
    try {
      if (external_check) monitor = std::thread([&] {
        std::unique_lock<std::mutex> lock(monitor_mutex);
        while (!monitor_wake.wait_for(lock, std::chrono::milliseconds(20), [&] { return finished; })) {
          try { external_check(); }
          catch (...) { Cancel(std::current_exception()); return; }
        }
      });
      for (size_t device = 0; device < count; ++device) {
        Check();
        threads.push_back(factory([&, device] {
          try { Check(); work(device); }
          catch (...) { Cancel(std::current_exception()); }
        }));
        if (!threads.back().joinable())
          throw std::runtime_error("device factory returned no worker");
      }
    } catch (...) { Cancel(std::current_exception()); }
    for (auto& thread : threads) if (thread.joinable()) thread.join();
    {
      std::lock_guard<std::mutex> lock(monitor_mutex);
      finished = true;
      monitor_wake.notify_all();
    }
    if (monitor.joinable()) monitor.join();
    if (external_check) {
      try { external_check(); } catch (...) { Cancel(std::current_exception()); }
    }
    Check();
  }

 private:
  mutable std::mutex mutex_;
  bool started_ = false;
  std::atomic<bool> cancelled_{false};
  std::exception_ptr error_;
  std::vector<std::function<void(std::exception_ptr)>> controls_;
};

}  // namespace colmap::mvs::owned
