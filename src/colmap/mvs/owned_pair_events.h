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
  explicit PairEvents(std::string dispatch) : dispatch_(std::move(dispatch)) {
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
  void Owner(size_t job) {
    std::lock_guard<std::mutex> lock(mutex_);
    owner_.at(job) = ThreadId();
  }
  void Identity(size_t job, int reference, int device) {
    if (reference < 0 || device < 0) throw std::invalid_argument("bad identity");
    std::lock_guard<std::mutex> lock(mutex_);
    reference_.at(job) = reference;
    device_.at(job) = device;
  }
  void Coordinator(const PairEvent& event) {
    const uint64_t transition_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            event.timestamp.time_since_epoch()).count();
    std::lock_guard<std::mutex> lock(mutex_);
    if (event.event == "both_ready") ready_ns_ = transition_ns;
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
    if (!ready_ns_ || !solver_done_[0] || !solver_done_[1]) {
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

  void EmitLocked(size_t job, const std::string& event,
                  const std::string& fields) {
    if (job != kBatch && job > 1) throw std::invalid_argument("bad event job");
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
    line << "\nOWNED_PAIR_EVENT schema=1 batch=" << batch_
         << " seq=" << ++sequence_ << " monotonic_ns=" << NowNs()
         << " pid=" << pid_ << " emitter_tid=" << ThreadId()
         << " owner_tid=" << (batch ? 0 : owner_[job])
         << " job=" << (batch ? "batch" : (job == 0 ? "A" : "B"))
         << " reference=" << (batch ? -1 : reference_[job])
         << " device_index=" << (batch ? -1 : device_[job])
         << " dispatch=" << dispatch_ << " event=" << event;
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
  std::mutex mutex_;
  std::string dispatch_;
  std::string batch_;
  uint64_t pid_ = 0;
  uint64_t sequence_ = 0;
  uint64_t start_ns_ = 0;
  uint64_t ready_ns_ = 0;
  uint64_t solver_end_ns_ = 0;
  std::array<uint64_t, 2> owner_{{0, 0}};
  std::array<int, 2> reference_{{-1, -1}};
  std::array<int, 2> device_{{-1, -1}};
  std::array<bool, 2> solver_done_{{false, false}};
};

}  // namespace colmap::mvs::owned
