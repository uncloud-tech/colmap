// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_pair_coordinator.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace colmap::mvs::owned {

// Linux experiment telemetry. One <=4096-byte write per complete line keeps
// records atomic on the captured pipe. No process-wide logging redirection.
class PairEvents {
 public:
  static constexpr size_t kBatch = std::numeric_limits<size_t>::max();
  static uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  static uint64_t ThreadId() {
#if defined(__linux__)
    return static_cast<uint64_t>(syscall(SYS_gettid));
#else
    throw std::runtime_error("owned pair events require Linux qualification");
#endif
  }
  explicit PairEvents(std::string dispatch, size_t count = 2, bool rolling = false,
                      size_t max_concurrent_solvers = 3)
      : count_(count),
        dispatch_(std::move(dispatch)),
        rolling_(rolling),
        max_concurrent_solvers_(max_concurrent_solvers),
        owner_(count, 0),
        reference_(count, -1),
        device_(count, -1),
        solver_done_(count, false),
        slot_(count, kBatch) {
    if ((!rolling && count != 2 && count != 3) || (rolling && (!count || !max_concurrent_solvers))) {
      throw std::invalid_argument("bad event count");
    }
#if defined(__linux__)
    pid_ = static_cast<uint64_t>(getpid());
#else
    throw std::runtime_error("owned pair events require Linux qualification");
#endif
    start_ns_ = NowNs();
    // Identity is unique even when concurrent constructors observe the same
    // clock tick. Keep start_ns_ unchanged: the token is not a timing sample.
    batch_ = std::to_string(pid_) + "-" + std::to_string(NextBatchToken(start_ns_));
  }
  void BindSlot(size_t job, size_t slot) {
    RequireJob(job);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!rolling_ || slot >= max_concurrent_solvers_ || slot_[job] != kBatch)
      throw std::logic_error("invalid generation slot binding");
    slot_[job] = slot;
  }
  void Owner(size_t job) {
    RequireJob(job);
    std::lock_guard<std::mutex> lock(mutex_);
    owner_.at(job) = ThreadId();
  }
  void Identity(size_t job, int reference, int device) {
    RequireJob(job);
    if (reference < 0 || device < 0) throw std::invalid_argument("bad identity");
    std::lock_guard<std::mutex> lock(mutex_);
    reference_.at(job) = reference;
    device_.at(job) = device;
  }
  void Coordinator(const PairEvent& event) {
    if (event.job != kBatch) RequireJob(event.job);
    const uint64_t transition_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            event.timestamp.time_since_epoch()).count();
    std::lock_guard<std::mutex> lock(mutex_);
    if (rolling_ && !ready_ns_ && event.event == "ready") {
      ready_ns_ = transition_ns;
    }
    if (event.event == "both_ready" || event.event == "all_ready") {
      ready_ns_ = transition_ns;
    }
    if (event.event == "solver_end") {
      solver_done_.at(event.job) = true;
      solver_end_ns_ = std::max(solver_end_ns_, transition_ns);
    }
    std::string fields = "coord_seq=" + std::to_string(event.sequence) +
                         " transition_ns=" + std::to_string(transition_ns);
    if (event.event == "ledger_release") {
      fields += " reserved_host=0 reserved_device=0";
    }
    EmitLocked(event.job, event.event, fields);
  }
  void Emit(size_t job, const std::string& event,
            const std::string& fields = "") {
    std::lock_guard<std::mutex> lock(mutex_);
    EmitLocked(job, event, fields);
  }
  void Finished() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ready_ns_ ||
        !std::all_of(solver_done_.begin(), solver_done_.begin() + count_,
                     [](bool done) { return done; })) {
      throw std::logic_error("incomplete owned pair timing state");
    }
    EmitLocked(kBatch, "batch_finished",
               "status=success pair_total_ns=" + std::to_string(NowNs()-start_ns_) +
               " ready_window_ns=" + std::to_string(solver_end_ns_-ready_ns_) +
               " serial_scope=solver_only");
  }

 private:
  friend struct PairEventsTestAccess;

  static uint64_t NextBatchToken(uint64_t now) {
    uint64_t previous = last_batch_token_.load(std::memory_order_relaxed);
    while (true) {
      if (previous == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("owned pair batch identity exhausted");
      }
      const uint64_t next = std::max(now, previous + 1);
      if (last_batch_token_.compare_exchange_weak(
              previous, next, std::memory_order_relaxed)) {
        return next;
      }
    }
  }
  inline static std::atomic<uint64_t> last_batch_token_{0};

  void RequireJob(size_t job) const {
    if (job >= count_) throw std::invalid_argument("bad event job");
  }

  void EmitLocked(size_t job, const std::string& event,
                  const std::string& fields) {
    if (job != kBatch && job >= count_) throw std::invalid_argument("bad event job");
    if (event.find_first_of(" \r\n") != std::string::npos ||
        fields.find_first_of("\r\n") != std::string::npos) {
      throw std::invalid_argument("invalid event token/line");
    }
    const bool batch = job == kBatch;
    if (event == "ready" &&
        (batch || owner_[job] == 0 || reference_[job] < 0 || device_[job] < 0)) {
      throw std::logic_error("ready without resolved owner/reference/device");
    }
    std::ostringstream line;
    // Frame the record even if unrelated stdout ended in a partial line. The
    // leading newline is part of the SAME <=PIPE_BUF write as the record.
    line << "\nOWNED_PAIR_EVENT schema=" << (rolling_ ? 2 : 1) << " batch=" << batch_
         << " seq=" << ++sequence_ << " monotonic_ns=" << NowNs()
         << " pid=" << pid_ << " emitter_tid=" << ThreadId()
         << " owner_tid=" << (batch ? 0 : owner_[job])
         << " job=" << (batch ? "batch"
                             : rolling_ ? "J" + std::to_string(job)
                                        : (job == 0 ? "A" : job == 1 ? "B" : "C"))
         << " reference=" << (batch ? -1 : reference_[job])
         << " device_index=" << (batch ? -1 : device_[job])
         << " dispatch=" << dispatch_ << " event=" << event;
    if (rolling_) {
      line << " feed_mode=rolling queue_index="
           << (batch ? "-1" : std::to_string(job))
           << " generation=" << (batch ? 0 : job + 1)
           << " slot="
           << (batch || slot_[job] == kBatch ? "-1" : std::to_string(slot_[job]));
    }
    if (!fields.empty()) line << ' ' << fields;
    line << '\n';
    const std::string text = line.str();
    if (text.size() > 4096) throw std::length_error("event exceeds atomic line cap");
#if defined(__linux__)
    ssize_t result;
    do { result = write(STDOUT_FILENO, text.data(), text.size()); }
    while (result < 0 && errno == EINTR);
    if (result < 0 || static_cast<size_t>(result) != text.size()) {
      throw std::runtime_error("incomplete event write");
    }
#endif
  }
  const size_t count_;
  std::mutex mutex_;
  std::string dispatch_;
  std::string batch_;
  uint64_t pid_ = 0;
  uint64_t sequence_ = 0;
  uint64_t start_ns_ = 0;
  uint64_t ready_ns_ = 0;
  uint64_t solver_end_ns_ = 0;
  const bool rolling_;
  const size_t max_concurrent_solvers_;
  std::vector<uint64_t> owner_;
  std::vector<int> reference_;
  std::vector<int> device_;
  std::vector<bool> solver_done_;
  std::vector<size_t> slot_;
};

}  // namespace colmap::mvs::owned
