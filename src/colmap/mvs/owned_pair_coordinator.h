// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Finite two-or-three-owner coordination for the opt-in owned driver.
//
// This header is a standard-library-only finite state machine that admits,
// prepares, dispatches, schedules output for and retires two or three owned
// jobs that share one process/primary CUDA context. It deliberately has NO
// CUDA or COLMAP dependency (including no logging header) so that it can be
// exercised by bounded standalone CPU tests. The coordinator owns no device
// pointers and never allocates GPU memory; callers retain and retire their own
// resources and only report retirement here after those resources are gone.
//
// Failure model: the first recorded exception is preserved, every bounded
// waiter is notified, and no success or credit is fabricated. Resource credits
// are returned only after ALL active jobs prove retirement; cancellation alone
// releases no credits.

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace colmap {
namespace mvs {
namespace owned {

enum class PairDispatch { kSerial, kPair };
enum class EarlyGeometryPermits { kOff, kOn };

struct ByteCharge {
  uint64_t host = 0;
  uint64_t device = 0;
};

struct PairEvent {
  size_t job;
  std::string event;
  uint64_t sequence;
  std::chrono::steady_clock::time_point timestamp;
};

class PairCoordinator {
 private:
  using Clock = std::chrono::steady_clock;

  // Reserved -> Preparing -> Ready -> Running -> ComputeDone -> Outputting ->
  // OutputDone. Retirement is tracked separately so cancelled and unstarted
  // jobs can still retire.
  enum class JobState {
    kReserved,
    kPreparing,
    kReady,
    kRunning,
    kComputeDone,
    kOutputting,
    kOutputDone,
  };

  static constexpr size_t kNoJob = std::numeric_limits<size_t>::max();

  static bool CheckedAdd(const uint64_t a, const uint64_t b, uint64_t* out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
      return false;
    }
    *out = a + b;
    return true;
  }

  static const char* StateName(const JobState state) {
    switch (state) {
      case JobState::kReserved:
        return "reserved";
      case JobState::kPreparing:
        return "preparing";
      case JobState::kReady:
        return "ready";
      case JobState::kRunning:
        return "running";
      case JobState::kComputeDone:
        return "compute_done";
      case JobState::kOutputting:
        return "outputting";
      case JobState::kOutputDone:
        return "output_done";
    }
    return "unknown";
  }

  void RequireJob(const size_t job) const {
    if (job >= count_) {
      throw std::invalid_argument(count_ == 2
          ? "owned pair: job index must be 0 or 1"
          : "owned pair: job index must be 0, 1 or 2");
    }
  }

  static void ThrowState(const size_t job,
                         const char* api,
                         const JobState current) {
    throw std::logic_error(std::string("owned pair: ") + api +
                           " invalid state for job " + std::to_string(job) +
                           " (" + StateName(current) + ")");
  }

 public:
  // Validates and reserves the complete default TWO-job bundle before any
  // worker exists. A positive timeout is required; waits are bounded by it.
  PairCoordinator(PairDispatch dispatch,
                  ByteCharge capacity,
                  ByteCharge process_charge,
                  std::array<ByteCharge, 2> job_charges,
                  std::chrono::milliseconds timeout,
                  std::function<void(const PairEvent&)> sink = {},
                  EarlyGeometryPermits early_geometry_permits =
                      EarlyGeometryPermits::kOff)
      : PairCoordinator(dispatch, capacity, process_charge,
                        std::vector<ByteCharge>(job_charges.begin(),
                                                job_charges.end()),
                        timeout, std::move(sink), early_geometry_permits, 2) {}

  // Static-three only: same full retention in both arms. This is not a queue.
  PairCoordinator(PairDispatch dispatch,
                  ByteCharge capacity,
                  ByteCharge process_charge,
                  std::array<ByteCharge, 3> job_charges,
                  std::chrono::milliseconds timeout,
                  std::function<void(const PairEvent&)> sink,
                  size_t solver_permits)
      : PairCoordinator(dispatch, capacity, process_charge,
                        std::vector<ByteCharge>(job_charges.begin(),
                                                job_charges.end()),
                        timeout, std::move(sink), EarlyGeometryPermits::kOn,
                        solver_permits) {}

  size_t JobCount() const noexcept { return count_; }

 private:
  PairCoordinator(PairDispatch dispatch, ByteCharge capacity,
                  ByteCharge process_charge, std::vector<ByteCharge> job_charges,
                  std::chrono::milliseconds timeout,
                  std::function<void(const PairEvent&)> sink,
                  EarlyGeometryPermits early_geometry_permits,
                  size_t solver_permits)
      : dispatch_(dispatch),
        count_(job_charges.size()),
        solver_permits_(solver_permits),
        early_geometry_permits_(early_geometry_permits),
        timeout_(timeout),
        sink_(std::move(sink)) {
    state_.fill(JobState::kReserved);
    if (count_ != 2 && count_ != 3) {
      throw std::invalid_argument("owned pair: count must be 2 or 3");
    }
    if (count_ == 3 && (dispatch_ != PairDispatch::kPair ||
                       (solver_permits_ != 2 && solver_permits_ != 3))) {
      throw std::invalid_argument(
          "static three requires pair and 2 or 3 solver permits");
    }
    if (dispatch_ != PairDispatch::kSerial && dispatch_ != PairDispatch::kPair) {
      throw std::invalid_argument("owned pair: unsupported dispatch");
    }
    if (early_geometry_permits_ != EarlyGeometryPermits::kOff &&
        early_geometry_permits_ != EarlyGeometryPermits::kOn) {
      throw std::invalid_argument("owned pair: unsupported early geometry permits");
    }
    if (timeout_.count() <= 0) {
      throw std::invalid_argument("owned pair: timeout must be positive");
    }
    struct Row {
      const char* name;
      ByteCharge charge;
    };
    std::vector<Row> rows{{"process", process_charge}};
    const char* names[] = {"job_a", "job_b", "job_c"};
    for (size_t j = 0; j < count_; ++j) rows.push_back({names[j], job_charges[j]});
    ByteCharge reserved;
    for (const Row& row : rows) {
      if (row.charge.host == 0 || row.charge.device == 0) {
        throw std::invalid_argument(std::string("owned pair: zero ") +
                                    row.name + " charge");
      }
      if (!CheckedAdd(reserved.host, row.charge.host, &reserved.host) ||
          !CheckedAdd(reserved.device, row.charge.device, &reserved.device)) {
        throw std::overflow_error("owned pair: charge overflow");
      }
    }
    if (reserved.host > capacity.host) {
      throw std::invalid_argument("owned pair: host budget insufficient");
    }
    if (reserved.device > capacity.device) {
      throw std::invalid_argument("owned pair: device budget insufficient");
    }
    reserved_ = reserved;
    for (size_t j = 0; j < count_; ++j) Emit(j, "admitted");
  }

 public:
  PairCoordinator(const PairCoordinator&) = delete;
  PairCoordinator& operator=(const PairCoordinator&) = delete;

  // Acquires the single preparation lease and marks the job Preparing. Peers
  // wait for preceding preparation releases (A then B then optional C). Blocks,
  // bounded and cancellation-aware. No mutex is held across the return.
  void BeginPreparation(size_t job) {
    RequireJob(job);
    const Clock::time_point deadline = Clock::now() + timeout_;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (state_[job] != JobState::kReserved) {
        ThrowState(job, "BeginPreparation", state_[job]);
      }
      AwaitLocked(
          lock, "preparation", job,
          [&]() { return !prep_in_use_ && job == next_prep_; },
          deadline);
      prep_in_use_ = true;
      state_[job] = JobState::kPreparing;
    }
    Emit(job, "prep_start");
  }

  // Releases the preparation lease and marks the job Ready. Experimental
  // geometry Pair/ON grants this job's permit immediately; the caller must
  // enforce geometry-only admission. OFF and SERIAL retain the both-ready
  // barrier; serial leaves B's permit to SolverDone(0). No lifetime fence or
  // reservation is released by an early solver permit.
  void ReadyAndWait(size_t job) {
    RequireJob(job);
    const Clock::time_point deadline = Clock::now() + timeout_;
    std::vector<PairEvent> pending;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (state_[job] != JobState::kPreparing) {
        ThrowState(job, "ReadyAndWait", state_[job]);
      }
      prep_in_use_ = false;
      ++next_prep_;
      pending.push_back(MakeEventLocked(job, "prep_end"));
      state_[job] = JobState::kReady;
      ready_[job] = true;
      pending.push_back(MakeEventLocked(job, "ready"));
      const bool early = dispatch_ == PairDispatch::kPair &&
                         early_geometry_permits_ == EarlyGeometryPermits::kOn;
      if (early && (count_ == 2 || job < 2 || solver_permits_ == 3 ||
                    successful_initial_solver_)) {
        permit_[job] = true;
        pending.push_back(MakeEventLocked(job, "solver_release"));
      }
      if (!both_ready_closed_ && All(ready_)) {
        both_ready_closed_ = true;
        pending.push_back(MakeEventLocked(
            kNoJob, count_ == 2 ? "both_ready" : "all_ready"));
        if (!early && dispatch_ == PairDispatch::kPair) {
          permit_[0] = true;
          permit_[1] = true;
          pending.push_back(MakeEventLocked(0, "solver_release"));
          pending.push_back(MakeEventLocked(1, "solver_release"));
        } else if (dispatch_ == PairDispatch::kSerial) {
          permit_[0] = true;
          pending.push_back(MakeEventLocked(0, "solver_release"));
        }
      }
      cv_.notify_all();
    }
    FlushEvents(pending);
    {
      std::unique_lock<std::mutex> lock(mutex_);
      AwaitLocked(lock, "dispatch", job, [&]() { return permit_[job]; },
                  deadline);
      state_[job] = JobState::kRunning;
      solver_active_[job] = true;
    }
    Emit(job, "solver_start");
  }

  // Records a successful solver return. In serial mode only this releases the
  // peer's B permit, and only after A's solver has returned successfully.
  void SolverDone(size_t job) {
    RequireJob(job);
    std::vector<PairEvent> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_[job] != JobState::kRunning) {
        ThrowState(job, "SolverDone", state_[job]);
      }
      state_[job] = JobState::kComputeDone;
      solver_active_[job] = false;
      cv_.notify_all();
      pending.push_back(MakeEventLocked(job, "solver_end"));
      if (job < 2) successful_initial_solver_ = true;
      if (count_ == 3 && job < 2 && ready_[2] && !permit_[2] && !cancelled_) {
        permit_[2] = true;
        pending.push_back(MakeEventLocked(2, "solver_release"));
        cv_.notify_all();
      }
      if (dispatch_ == PairDispatch::kSerial && job == 0) {
        permit_[1] = true;
        pending.push_back(MakeEventLocked(1, "solver_release"));
        cv_.notify_all();
      }
    }
    FlushEvents(pending);
  }

  // Acquires the single output lease (ordered A/B/C for static three only).
  // Readiness (ComputeDone) is required. Bounded and cancellation-aware.
  void BeginOutput(size_t job) {
    RequireJob(job);
    const Clock::time_point deadline = Clock::now() + timeout_;
    PairEvent event;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (state_[job] != JobState::kComputeDone) {
        ThrowState(job, "BeginOutput", state_[job]);
      }
      AwaitLocked(lock, "output", job, [&]() {
        return !output_in_use_ && (count_ == 2 || job == next_output_);
      }, deadline);
      output_in_use_ = true;
      output_owner_ = job;
      state_[job] = JobState::kOutputting;
      event = MakeEventLocked(job, "output_start");
    }
    InvokeSink(event);
  }

  // Releases the output lease before any batch-final wait.
  void EndOutput(size_t job) {
    RequireJob(job);
    PairEvent event;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_[job] != JobState::kOutputting) {
        ThrowState(job, "EndOutput", state_[job]);
      }
      if (output_owner_ != job) {
        throw std::logic_error("owned pair: EndOutput by non-owner");
      }
      output_in_use_ = false;
      output_owner_ = kNoJob;
      state_[job] = JobState::kOutputDone;
      output_done_[job] = true;
      ++next_output_;
      event = MakeEventLocked(job, "output_end");
      cv_.notify_all();
    }
    InvokeSink(event);
  }

  // Batch-final wait: requires the output lease already released and waits for
  // all active OutputDone (or cancellation) while owned objects remain alive.
  void AwaitOutputs(size_t job) {
    RequireJob(job);
    const Clock::time_point deadline = Clock::now() + timeout_;
    std::unique_lock<std::mutex> lock(mutex_);
    if (state_[job] != JobState::kOutputDone) {
      ThrowState(job, "AwaitOutputs", state_[job]);
    }
    AwaitLocked(lock, "outputs", job,
                [&]() { return All(output_done_); }, deadline);
  }

  // Latches the first error, wakes every bounded waiter and releases no
  // credits. Never propagates an exception (noexcept) and never holds the
  // mutex while invoking the sink.
  void Cancel(size_t job, std::exception_ptr error) noexcept {
    PairEvent event;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!first_error_) {
        first_error_ = error ? error
                             : std::make_exception_ptr(
                                   std::runtime_error("owned pair: cancelled"));
      }
      PublishCancellationLocked();
      event = MakeEventLocked(job, "cancelled");
    }
    InvokeSinkGuarded(event);
  }

  // Failure-only ownership rendezvous. Publish cancellation FIRST so no new
  // solver can dispatch, then prove local CUDA completion without holding the
  // mutex. A running peer must finish or enter this same drain path before
  // either owner may unwind its retained buffers. Never-started peers have no
  // active solver and need not arrive. Unknown completion or timeout must stop
  // the process, NOT throw through owners and accidentally free live storage.
  template <typename Drain>
  void CancelAndAwaitCleanup(size_t job,
                             std::exception_ptr error,
                             Drain&& drain) noexcept {
    Cancel(job, error);
    try {
      RequireJob(job);
      drain();
      std::unique_lock<std::mutex> lock(mutex_);
      solver_active_[job] = false;
      cv_.notify_all();
      const auto deadline = Clock::now() + timeout_;
      if (!cv_.wait_until(lock, deadline, [&] {
            return std::none_of(solver_active_.begin(), solver_active_.begin() + count_,
                                [](bool active) { return active; });
          })) {
        std::terminate();
      }
    } catch (...) {
      std::terminate();
    }
  }

  // Called AFTER the real job resources have been destroyed. Supports
  // cancelled and unstarted jobs. Duplicate or unsafe retirement rejects.
  // Retirement records proven cleanup but NEVER releases batch credits; only
  // Finish does that, and only when ALL active retirements proved cleanup.
  void Retire(size_t job, bool completion_proven = true) {
    RequireJob(job);
    PairEvent event;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (retired_[job]) {
        throw std::logic_error("owned pair: duplicate retirement");
      }
      if (output_owner_ == job) {
        if (!cancelled_) {
          throw std::logic_error(
              "owned pair: retirement while output lease held");
        }
        // A failed writer never reaches EndOutput. After cancellation, actual
        // resource cleanup permits retiring that abandoned lease; this does
        // not release any byte credits or permit another job to resume.
        if (completion_proven) {
          output_in_use_ = false;
          output_owner_ = kNoJob;
        }
      }
      if (!cancelled_ && state_[job] != JobState::kOutputDone) {
        throw std::logic_error(
            "owned pair: unsafe retirement before output completion");
      }
      retired_[job] = true;
      if (completion_proven) {
        proven_retired_[job] = true;
      } else {
        proven_retired_[job] = false;
        if (!first_error_) {
          first_error_ = std::make_exception_ptr(
              std::runtime_error("owned pair: retirement completion unproven"));
        }
      }
      event = MakeEventLocked(job, "retired");
      cv_.notify_all();
    }
    InvokeSink(event);
  }

  // AFTER all active owner joins. Releases the full reservation exactly once,
  // only when ALL active retirements proved cleanup; a clean cancellation may release
  // and then rethrow its first failure. Unknown cleanup retains the full
  // reservation, emits no ledger_release, and throws (earlier first error wins).
  void Finish() {
    std::exception_ptr error;
    PairEvent release_event;
    bool release = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!All(retired_)) {
        // Missing retirement proof must retain credits, but must not replace
        // the originating failure with a secondary protocol diagnostic.
        if (first_error_) std::rethrow_exception(first_error_);
        throw std::logic_error(count_ == 2
            ? "owned pair: Finish before both jobs retired"
            : "owned pair: Finish before all three jobs retired");
      }
      if (All(proven_retired_)) {
        if (!credits_released_) {
          credits_released_ = true;
          reserved_ = ByteCharge{0, 0};
          release_event = MakeEventLocked(kNoJob, "ledger_release");
          release = true;
        }
      } else if (!first_error_) {
        first_error_ = std::make_exception_ptr(
            std::runtime_error("owned pair: retirement completion unproven"));
      }
      error = first_error_;
      if (!error && cancelled_) {
        error = std::make_exception_ptr(
            std::runtime_error("owned pair: run cancelled without error"));
      }
    }
    if (release) {
      try {
        InvokeSink(release_event);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error) {
          error = first_error_;
        }
      }
    }
    if (error) {
      std::rethrow_exception(error);
    }
  }

  // Full reservation until all active retirements proved cleanup AND Finish runs;
  // then zero. Unknown cleanup never releases.
  ByteCharge ReservedBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reserved_;
  }

 private:
  template <typename Predicate>
  void AwaitLocked(std::unique_lock<std::mutex>& lock,
                   const char* what,
                   size_t job,
                   Predicate predicate,
                   Clock::time_point deadline) {
    while (true) {
      // Fail closed even if the predicate is already satisfied: cancellation
      // must forbid new preparation and solver admission.
      if (cancelled_) {
        std::exception_ptr error = first_error_;
        if (!error) {
          error = std::make_exception_ptr(
              std::runtime_error("owned pair: cancelled"));
        }
        lock.unlock();
        std::rethrow_exception(error);
      }
      if (predicate()) {
        return;
      }
      if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
        if (!first_error_) {
          first_error_ = std::make_exception_ptr(std::runtime_error(
              std::string("owned pair: ") + what + " wait timed out"));
        }
        cancelled_ = true;
        const PairEvent event = MakeEventLocked(job, "cancelled");
        std::exception_ptr error = first_error_;
        cv_.notify_all();
        lock.unlock();
        InvokeSinkGuarded(event);
        std::rethrow_exception(error);
      }
    }
  }

  void FlushEvents(std::vector<PairEvent>& events) {
    for (const PairEvent& event : events) {
      InvokeSink(event);
    }
  }

  void PublishCancellationLocked() {
    cancelled_ = true;
    cv_.notify_all();
  }

  PairEvent MakeEventLocked(size_t job, const char* name) {
    PairEvent event;
    event.job = job;
    event.event = name;
    event.sequence = next_sequence_++;
    event.timestamp = Clock::now();
    return event;
  }

  void Emit(size_t job, const char* name) {
    PairEvent event;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      event = MakeEventLocked(job, name);
    }
    InvokeSink(event);
  }

  void InvokeSink(const PairEvent& event) {
    if (!sink_) {
      return;
    }
    try {
      sink_(event);
    } catch (...) {
      LatchError(std::current_exception());
      throw;
    }
  }

  void InvokeSinkGuarded(const PairEvent& event) noexcept {
    if (!sink_) {
      return;
    }
    try {
      sink_(event);
    } catch (...) {
    }
  }

  void LatchError(std::exception_ptr error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!first_error_) {
      first_error_ = error;
    }
    cancelled_ = true;
    cv_.notify_all();
  }

  bool All(const std::array<bool, 3>& values) const {
    return std::all_of(values.begin(), values.begin() + count_,
                       [](bool value) { return value; });
  }

  PairDispatch dispatch_;
  const size_t count_;
  const size_t solver_permits_;
  bool successful_initial_solver_ = false;
  EarlyGeometryPermits early_geometry_permits_;
  std::chrono::milliseconds timeout_;
  std::function<void(const PairEvent&)> sink_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  uint64_t next_sequence_ = 1;
  bool cancelled_ = false;
  std::exception_ptr first_error_;
  std::array<JobState, 3> state_;
  bool prep_in_use_ = false;
  size_t next_prep_ = 0;
  size_t next_output_ = 0;
  std::array<bool, 3> ready_{};
  bool both_ready_closed_ = false;
  std::array<bool, 3> permit_{};
  std::array<bool, 3> solver_active_{};
  bool output_in_use_ = false;
  size_t output_owner_ = kNoJob;
  std::array<bool, 3> output_done_{};
  std::array<bool, 3> retired_{};
  std::array<bool, 3> proven_retired_{};
  bool credits_released_ = false;
  ByteCharge reserved_;
};

}  // namespace owned
}  // namespace mvs
}  // namespace colmap
