# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic CPU tests for the finite owned PairCoordinator contract.

These tests compile the REAL header (src/colmap/mvs/owned_pair_coordinator.h)
against a standalone C++17 fake backend and against negative header mutants.
They prove host-side coordination only: never CUDA ordering, device memory,
product build, exactness or overlap. No sleep-based probabilistic interleaving
assertions are used; every wait is a bounded condition-variable latch and the
whole process is bounded by a subprocess timeout.

Run: python3 -m unittest discover -s benchmark/owned_patch_match -p 'test_*.py' -v
"""
import resource
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / 'src/colmap/mvs/owned_pair_coordinator.h'

PROGRAM = r'''
#include "colmap/mvs/owned_pair_coordinator.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace colmap::mvs::owned;
using namespace std::chrono_literals;

static int failures = 0;

static void OnTerminate() {
  std::fprintf(stderr, "terminate: uncaught exception in a worker\n");
  std::_Exit(3);
}

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #cond); \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

#define REQUIRE(cond)                                                      \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "REQUIRE failed at line %d: %s\n", __LINE__, #cond); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

static const ByteCharge CAP{1000, 2000};
static const ByteCharge PROC{10, 20};
static const std::array<ByteCharge, 2> JOBS{{ByteCharge{100, 200},
                                             ByteCharge{300, 400}}};

template <typename Fn>
bool Throws(Fn fn) {
  try {
    fn();
  } catch (...) {
    return true;
  }
  return false;
}

struct Latch {
  std::mutex mutex;
  std::condition_variable cv;
  bool flag = false;
  void Set() {
    std::lock_guard<std::mutex> lock(mutex);
    flag = true;
    cv.notify_all();
  }
  bool Wait(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, timeout, [this] { return flag; });
  }
};

struct EventLog {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<PairEvent> events;
  void Push(const PairEvent& event) {
    std::lock_guard<std::mutex> lock(mutex);
    events.push_back(event);
    cv.notify_all();
  }
  bool WaitFor(const std::string& name, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, timeout, [&] {
      for (const PairEvent& event : events) {
        if (event.event == name) {
          return true;
        }
      }
      return false;
    });
  }
  size_t Count(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex);
    size_t count = 0;
    for (const PairEvent& event : events) {
      if (event.event == name) {
        ++count;
      }
    }
    return count;
  }
  bool Seen(size_t job, const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex);
    for (const PairEvent& event : events) {
      if (event.job == job && event.event == name) {
        return true;
      }
    }
    return false;
  }
  uint64_t Sequence(size_t job, const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex);
    for (const PairEvent& event : events) {
      if (event.job == job && event.event == name) {
        return event.sequence;
      }
    }
    return 0;
  }
};

static void BoundedSettle(std::chrono::milliseconds duration) {
  std::mutex mutex;
  std::condition_variable cv;
  std::unique_lock<std::mutex> lock(mutex);
  cv.wait_for(lock, duration, [] { return false; });
}

static void RunOwner(PairCoordinator* c, size_t job, bool proven = true) {
  c->BeginPreparation(job);
  c->ReadyAndWait(job);
  c->SolverDone(job);
  c->BeginOutput(job);
  c->EndOutput(job);
  c->AwaitOutputs(job);
  c->Retire(job, proven);
}

static void ScenarioAdmissionAndLedger() {
  using D = PairDispatch;
  {
    PairCoordinator c(D::kPair, ByteCharge{410, 620}, PROC, JOBS, 1000ms);
    CHECK(c.ReservedBytes().host == 410 && c.ReservedBytes().device == 620);
  }
  CHECK(Throws([] {
    PairCoordinator c(D::kPair, ByteCharge{409, 620}, PROC, JOBS, 1000ms);
    (void)c;
  }));
  CHECK(Throws([] {
    PairCoordinator c(D::kPair, ByteCharge{410, 619}, PROC, JOBS, 1000ms);
    (void)c;
  }));
  // Pair does not fit even though A alone (process + job_a) would.
  CHECK(Throws([] {
    PairCoordinator c(D::kPair, ByteCharge{110, 220}, PROC, JOBS, 1000ms);
    (void)c;
  }));
  CHECK(Throws([] {
    PairCoordinator c(D::kPair, CAP, ByteCharge{0, 20}, JOBS, 1000ms);
    (void)c;
  }));
  CHECK(Throws([] {
    PairCoordinator c(D::kPair, CAP, PROC,
                      std::array<ByteCharge, 2>{{{0, 200}, {300, 400}}}, 1000ms);
    (void)c;
  }));
  CHECK(Throws([] {
    PairCoordinator c(D::kPair, ByteCharge{UINT64_MAX, UINT64_MAX}, PROC,
                      std::array<ByteCharge, 2>{{{UINT64_MAX, 1}, {1, 1}}},
                      1000ms);
    (void)c;
  }));
  CHECK(Throws([] {
    PairCoordinator c(D::kPair, CAP, PROC, JOBS, 0ms);
    (void)c;
  }));
}

static void ScenarioStateMisuse() {
  using D = PairDispatch;
  PairCoordinator c(D::kPair, CAP, PROC, JOBS, 5000ms);
  CHECK(Throws([&] { c.ReadyAndWait(0); }));
  CHECK(Throws([&] { c.SolverDone(0); }));
  CHECK(Throws([&] { c.BeginOutput(0); }));
  CHECK(Throws([&] { c.EndOutput(0); }));
  CHECK(Throws([&] { c.AwaitOutputs(0); }));
  CHECK(Throws([&] { c.BeginPreparation(2); }));
  CHECK(Throws([&] { c.SolverDone(7); }));
  CHECK(Throws([&] { c.Retire(2); }));
  CHECK(Throws([&] { c.Retire(0); }));
  CHECK(Throws([&] { c.Finish(); }));
}

static void ScenarioSerialDispatch() {
  EventLog log;
  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, 8000ms,
                    [&log](const PairEvent& e) { log.Push(e); });
  std::thread a([&] { RunOwner(&c, 0); });
  std::thread b([&] { RunOwner(&c, 1); });
  a.join();
  b.join();
  CHECK(log.Seen(0, "prep_end") && log.Seen(1, "prep_end"));
  CHECK(log.Count("both_ready") == 1);
  CHECK(log.Seen(0, "solver_release") && log.Seen(1, "solver_release"));
  CHECK(log.Seen(0, "solver_start") && log.Seen(1, "solver_start"));
  // serial: B's solver release/start must follow A's solver return.
  CHECK(log.Sequence(0, "solver_end") < log.Sequence(1, "solver_release"));
  CHECK(log.Sequence(1, "solver_release") < log.Sequence(1, "solver_start"));
  CHECK(log.Seen(0, "output_end") && log.Seen(1, "output_end"));
  // Credits stay reserved until Finish, which runs after both joins.
  CHECK(c.ReservedBytes().host == 410 && c.ReservedBytes().device == 620);
  c.Finish();
  CHECK(c.ReservedBytes().host == 0 && c.ReservedBytes().device == 0);
  CHECK(log.Count("ledger_release") == 1);
}

static void ScenarioPairRelease() {
  EventLog log;
  PairCoordinator c(PairDispatch::kPair, CAP, PROC, JOBS, 8000ms,
                    [&log](const PairEvent& e) { log.Push(e); });
  std::mutex barrier_mutex;
  std::condition_variable barrier_cv;
  int arrived = 0;
  std::atomic<bool> ok0{false};
  std::atomic<bool> ok1{false};
  auto arrive = [&]() {
    std::unique_lock<std::mutex> lock(barrier_mutex);
    ++arrived;
    barrier_cv.notify_all();
    return barrier_cv.wait_for(lock, 4000ms, [&] { return arrived == 2; });
  };
  std::thread a([&] {
    c.BeginPreparation(0);
    c.ReadyAndWait(0);
    ok0.store(arrive());
    c.SolverDone(0);
    c.BeginOutput(0);
    c.EndOutput(0);
    c.AwaitOutputs(0);
    c.Retire(0, true);
  });
  std::thread b([&] {
    c.BeginPreparation(1);
    c.ReadyAndWait(1);
    ok1.store(arrive());
    c.SolverDone(1);
    c.BeginOutput(1);
    c.EndOutput(1);
    c.AwaitOutputs(1);
    c.Retire(1, true);
  });
  a.join();
  b.join();
  // pair grants both solver permits once both jobs are ready: both jobs reach
  // the barrier before either records SolverDone.
  CHECK(ok0.load() && ok1.load());
  CHECK(log.Count("both_ready") == 1);
  CHECK(log.Seen(0, "solver_release") && log.Seen(1, "solver_release"));
  CHECK(log.Count("solver_start") == 2);
  CHECK(c.ReservedBytes().host == 410);  // full until Finish
  c.Finish();
  CHECK(c.ReservedBytes().host == 0);
  CHECK(log.Count("ledger_release") == 1);
}

static void ScenarioBothReadyGate() {
  EventLog log;
  Latch b_parked;
  Latch allow_b;
  PairCoordinator c(PairDispatch::kPair, CAP, PROC, JOBS, 8000ms,
                    [&log](const PairEvent& e) { log.Push(e); });
  std::thread a([&] {
    c.BeginPreparation(0);
    c.ReadyAndWait(0);
    c.SolverDone(0);
    c.BeginOutput(0);
    c.EndOutput(0);
    c.AwaitOutputs(0);
    c.Retire(0, true);
  });
  std::thread b([&] {
    c.BeginPreparation(1);
    b_parked.Set();
    allow_b.Wait(8000ms);
    c.ReadyAndWait(1);
    c.SolverDone(1);
    c.BeginOutput(1);
    c.EndOutput(1);
    c.AwaitOutputs(1);
    c.Retire(1, true);
  });
  REQUIRE(b_parked.Wait(4000ms));
  // The both-ready gate must hold A while B is parked: no solver may start.
  CHECK(!log.WaitFor("solver_start", 500ms));
  allow_b.Set();
  a.join();
  b.join();
  CHECK(log.Count("solver_start") == 2);
  CHECK(c.ReservedBytes().host == 410);  // full until Finish
  c.Finish();
  CHECK(c.ReservedBytes().host == 0);
}

static void ScenarioCreditGate() {
  EventLog log;
  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, 8000ms,
                    [&log](const PairEvent& e) { log.Push(e); });
  Latch retired0;
  Latch retired1;
  Latch allow_finish;
  auto owner_gated = [&](size_t job, Latch* retired) {
    c.BeginPreparation(job);
    c.ReadyAndWait(job);
    c.SolverDone(job);
    c.BeginOutput(job);
    c.EndOutput(job);
    c.AwaitOutputs(job);
    c.Retire(job, true);
    retired->Set();
    allow_finish.Wait(8000ms);
  };
  std::thread a([&] { owner_gated(0, &retired0); });
  std::thread b([&] { owner_gated(1, &retired1); });
  REQUIRE(retired0.Wait(4000ms) && retired1.Wait(4000ms));
  // Both owners retired and still alive: reservation must remain full and no
  // ledger_release may have been emitted yet.
  CHECK(c.ReservedBytes().host == 410 && c.ReservedBytes().device == 620);
  CHECK(log.Count("ledger_release") == 0);
  allow_finish.Set();
  a.join();
  b.join();
  // Still full after both joins, before Finish.
  CHECK(c.ReservedBytes().host == 410 && c.ReservedBytes().device == 620);
  c.Finish();
  CHECK(c.ReservedBytes().host == 0 && c.ReservedBytes().device == 0);
  CHECK(log.Count("ledger_release") == 1);
}

static void ScenarioCancellationNotify() {
  EventLog log;
  Latch a_threw;
  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, 8000ms,
                    [&log](const PairEvent& e) { log.Push(e); });
  std::thread a([&] {
    try {
      c.BeginPreparation(0);
      c.ReadyAndWait(0);
    } catch (...) {
      a_threw.Set();
    }
    c.Retire(0, true);
  });
  REQUIRE(log.WaitFor("ready", 4000ms));
  BoundedSettle(250ms);
  c.Cancel(0, std::make_exception_ptr(std::runtime_error("cancel-boom")));
  // Cancellation must wake the blocked waiter promptly (not at its deadline).
  CHECK(a_threw.Wait(2000ms));
  a.join();
  c.Retire(1, true);
  CHECK(c.ReservedBytes().host == 410);  // still full until Finish
  CHECK(Throws([&] { c.Finish(); }));
  CHECK(c.ReservedBytes().host == 0);
}

static void ScenarioStartupFailure() {
  using D = PairDispatch;
  PairCoordinator c(D::kSerial, CAP, PROC, JOBS, 8000ms);
  auto error = std::make_exception_ptr(std::runtime_error("prep-boom"));
  c.Cancel(0, error);
  CHECK(Throws([&] { c.BeginPreparation(0); }));
  CHECK(Throws([&] { c.BeginPreparation(1); }));
  CHECK(c.ReservedBytes().host == 410);  // cancellation releases no credits
  c.Retire(0, true);
  c.Retire(1, true);
  CHECK(c.ReservedBytes().host == 410);  // released only by Finish
  CHECK(Throws([&] { c.Finish(); }));  // clean cancellation releases then throws
  CHECK(c.ReservedBytes().host == 0 && c.ReservedBytes().device == 0);
  // A second cancellation with a different error must not overwrite the first.
  PairCoordinator c2(D::kSerial, CAP, PROC, JOBS, 5000ms);
  c2.Cancel(0, std::make_exception_ptr(std::runtime_error("first")));
  c2.Cancel(1, std::make_exception_ptr(std::runtime_error("second")));
  c2.Retire(0, true);
  c2.Retire(1, true);
  CHECK(c2.ReservedBytes().host == 410);
  bool first_survived = false;
  try {
    c2.Finish();
  } catch (const std::runtime_error& e) {
    first_survived = std::string(e.what()) == "first";
  }
  CHECK(first_survived);
  CHECK(c2.ReservedBytes().host == 0);
}

static void ScenarioSerialPeerNotDispatched() {
  EventLog log;
  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, 8000ms,
                    [&log](const PairEvent& e) { log.Push(e); });
  std::atomic<bool> b_dispatched{false};
  auto error = std::make_exception_ptr(std::runtime_error("solver-boom"));
  std::thread a([&] {
    c.BeginPreparation(0);
    c.ReadyAndWait(0);
    c.Cancel(0, error);
    c.Retire(0, true);
  });
  std::thread b([&] {
    try {
      c.BeginPreparation(1);
      c.ReadyAndWait(1);
      b_dispatched.store(true);
    } catch (...) {
    }
    try {
      c.Retire(1, true);
    } catch (...) {
    }
  });
  a.join();
  b.join();
  CHECK(!b_dispatched.load());
  CHECK(!log.Seen(1, "solver_start"));
  CHECK(c.ReservedBytes().host == 410);  // full until Finish
  CHECK(Throws([&] { c.Finish(); }));
  CHECK(c.ReservedBytes().host == 0);
}

static void ScenarioOutputFault() {
  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, 8000ms);
  auto error = std::make_exception_ptr(std::runtime_error("output-boom"));
  std::thread a([&] {
    try {
      c.BeginPreparation(0);
      c.ReadyAndWait(0);
      c.SolverDone(0);
      c.BeginOutput(0);
      c.Cancel(0, error);
      c.EndOutput(0);
      c.AwaitOutputs(0);
    } catch (...) {
    }
    try {
      c.Retire(0, true);
    } catch (...) {
    }
  });
  std::thread b([&] {
    try {
      c.BeginPreparation(1);
      c.ReadyAndWait(1);
      c.SolverDone(1);
      c.BeginOutput(1);
      c.EndOutput(1);
      c.AwaitOutputs(1);
    } catch (...) {
    }
    try {
      c.Retire(1, true);
    } catch (...) {
    }
  });
  a.join();
  b.join();
  CHECK(c.ReservedBytes().host == 410);  // full until Finish
  CHECK(Throws([&] { c.Finish(); }));
  CHECK(c.ReservedBytes().host == 0);
}

static void ScenarioRetireGuards() {
  EventLog log;
  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, 8000ms,
                    [&log](const PairEvent& e) { log.Push(e); });
  Latch a_output_lease;
  Latch allow_a_end;
  std::thread a([&] {
    c.BeginPreparation(0);
    c.ReadyAndWait(0);
    c.SolverDone(0);
    c.BeginOutput(0);
    a_output_lease.Set();
    allow_a_end.Wait(8000ms);
    c.EndOutput(0);
    c.AwaitOutputs(0);
    c.Retire(0, true);
  });
  std::thread b([&] {
    c.BeginPreparation(1);
    c.ReadyAndWait(1);
    c.SolverDone(1);
    c.BeginOutput(1);
    c.EndOutput(1);
    c.AwaitOutputs(1);
    c.Retire(1, true);
  });
  REQUIRE(a_output_lease.Wait(4000ms));
  // Single output lease: B must not have started output while A holds it.
  CHECK(!log.Seen(1, "output_start"));
  // Unsafe: A is not cancelled, has not completed output, and still holds the
  // output lease, so retirement must reject without touching the ledger.
  CHECK(Throws([&] { c.Retire(0, true); }));
  CHECK(Throws([&] { c.EndOutput(1); }));  // B has no active output lease
  CHECK(c.ReservedBytes().host == 410);
  allow_a_end.Set();
  a.join();
  b.join();
  CHECK(c.ReservedBytes().host == 410);  // full until Finish
  c.Finish();
  CHECK(c.ReservedBytes().host == 0);
}

static void ScenarioCleanupUnproven() {
  EventLog log;
  PairCoordinator c(PairDispatch::kPair, CAP, PROC, JOBS, 8000ms,
                    [&log](const PairEvent& e) { log.Push(e); });
  std::thread a([&] { RunOwner(&c, 0, /*proven=*/false); });
  std::thread b([&] { RunOwner(&c, 1, true); });
  a.join();
  b.join();
  CHECK(c.ReservedBytes().host == 410 && c.ReservedBytes().device == 620);
  CHECK(Throws([&] { c.Finish(); }));     // unknown cleanup must fail
  CHECK(c.ReservedBytes().host == 410);   // ...and release nothing
  CHECK(log.Count("ledger_release") == 0);
  bool unproven = false;
  try {
    c.Finish();
  } catch (const std::runtime_error& e) {
    unproven = std::string(e.what()) ==
               "owned pair: retirement completion unproven";
  }
  CHECK(unproven);
}

static void ScenarioTimeout() {
  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, 150ms);
  CHECK(Throws([&] {
    c.BeginPreparation(0);
    c.ReadyAndWait(0);
  }));
  c.Retire(0, true);
  CHECK(c.ReservedBytes().host == 410);  // peer not retired yet
  c.Retire(1, true);
  CHECK(c.ReservedBytes().host == 410);  // released only by Finish
  CHECK(Throws([&] { c.Finish(); }));
  CHECK(c.ReservedBytes().host == 0);
}

static void ScenarioSinkThrow() {
  using D = PairDispatch;
  CHECK(Throws([] {
    PairCoordinator c(D::kPair, CAP, PROC, JOBS, 1000ms,
                      [](const PairEvent&) {
                        throw std::runtime_error("sink-admitted");
                      });
    (void)c;
  }));
  EventLog log;
  PairCoordinator c(D::kSerial, CAP, PROC, JOBS, 8000ms,
                    [&log](const PairEvent& e) {
                      if (e.event == "prep_start") {
                        throw std::runtime_error("sink-prep");
                      }
                      log.Push(e);
                    });
  CHECK(Throws([&] { c.BeginPreparation(0); }));
  CHECK(Throws([&] { c.BeginPreparation(1); }));
  c.Retire(0, true);
  c.Retire(1, true);
  CHECK(c.ReservedBytes().host == 410);  // full until Finish
  CHECK(Throws([&] { c.Finish(); }));
  CHECK(c.ReservedBytes().host == 0);
}

static void ScenarioSinkReentrancy() {
  PairCoordinator* self = nullptr;
  PairCoordinator c(PairDispatch::kPair, CAP, PROC, JOBS, 8000ms,
                    [&self](const PairEvent&) {
                      if (self != nullptr) {
                        // Deadlocks if the sink is invoked under the mutex.
                        (void)self->ReservedBytes();
                      }
                    });
  self = &c;
  std::thread a([&] { RunOwner(&c, 0); });
  std::thread b([&] { RunOwner(&c, 1); });
  a.join();
  b.join();
  CHECK(c.ReservedBytes().host == 410);
  c.Finish();
  CHECK(c.ReservedBytes().host == 0);
}

int main() {
  std::set_terminate(OnTerminate);
  ScenarioAdmissionAndLedger();
  ScenarioStateMisuse();
  ScenarioSerialDispatch();
  ScenarioPairRelease();
  ScenarioBothReadyGate();
  ScenarioCreditGate();
  ScenarioCancellationNotify();
  ScenarioStartupFailure();
  ScenarioSerialPeerNotDispatched();
  ScenarioOutputFault();
  ScenarioRetireGuards();
  ScenarioCleanupUnproven();
  ScenarioTimeout();
  ScenarioSinkThrow();
  ScenarioSinkReentrancy();
  if (failures != 0) {
    std::fprintf(stderr, "%d pair coordinator check(s) failed\n", failures);
    return 1;
  }
  std::printf("all pair coordinator checks passed\n");
  return 0;
}
'''


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class PairCoordinatorTest(unittest.TestCase):
    def run_program(self, mutation=None):
        source = HEADER.read_text()
        if mutation is not None:
            source = mutation(source)
        temp = Path(tempfile.mkdtemp(prefix='owned-pair-'))
        header_dir = temp / 'colmap' / 'mvs'
        header_dir.mkdir(parents=True)
        (header_dir / 'owned_pair_coordinator.h').write_text(source)
        program = temp / 'program.cc'
        program.write_text(PROGRAM)
        binary = temp / 'program'
        command = ['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O1',
                   '-pthread', '-I', str(temp), str(program), '-o', str(binary)]
        compiled = subprocess.run(command, capture_output=True, text=True,
                                  timeout=60)
        (temp / 'compile.log').write_text(compiled.stdout + compiled.stderr)
        self.assertEqual(compiled.returncode, 0,
                         'compile failed:\n' + compiled.stdout + compiled.stderr)
        try:
            result = subprocess.run([str(binary)], capture_output=True,
                                    text=True, timeout=60, preexec_fn=no_core)
        except subprocess.TimeoutExpired:
            print(f'PAIR_COORDINATOR_TIMEOUT artifact={temp}')
            return None
        (temp / 'run.log').write_text(result.stdout + result.stderr)
        print(f'PAIR_COORDINATOR_ARTIFACT {temp} exit={result.returncode}')
        return result

    def test_real_header_behavior(self):
        result = self.run_program()
        self.assertIsNotNone(result, 'real header program hung / timed out')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mutant_missing_both_ready_gate(self):
        def mutation(source):
            old = 'if (!both_ready_closed_ && ready_[0] && ready_[1]) {'
            self.assertEqual(source.count(old), 1,
                             'both-ready gate anchor not unique')
            return source.replace(old, 'if (!both_ready_closed_) {  // mutant')

        result = self.run_program(mutation)
        self.assertTrue(result is None or result.returncode != 0,
                        'missing both-ready gate was not detected')

    def test_mutant_missing_cancellation_notification(self):
        def mutation(source):
            old = ('  void PublishCancellationLocked() {\n'
                   '    cancelled_ = true;\n'
                   '    cv_.notify_all();\n'
                   '  }')
            self.assertEqual(source.count(old), 1,
                             'cancellation-notify anchor not unique')
            return source.replace(
                old,
                '  void PublishCancellationLocked() {\n'
                '    cancelled_ = true;\n'
                '  }')

        result = self.run_program(mutation)
        self.assertTrue(result is None or result.returncode != 0,
                        'missing cancellation notification was not detected')

    def test_mutant_credit_early_release(self):
        def mutation(source):
            old = '      retired_[job] = true;'
            self.assertEqual(source.count(old), 1,
                             'retirement anchor not unique')
            return source.replace(
                old,
                '      retired_[job] = true;\n'
                '      reserved_ = ByteCharge{0, 0};  // mutant: release at Retire')

        result = self.run_program(mutation)
        self.assertTrue(result is None or result.returncode != 0,
                        'early credit release before Finish was not detected')


if __name__ == '__main__':
    unittest.main(verbosity=2)
