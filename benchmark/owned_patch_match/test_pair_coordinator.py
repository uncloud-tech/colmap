# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic CPU tests for the finite owned PairCoordinator contract.

These tests compile the REAL header (src/colmap/mvs/owned_pair_coordinator.h)
against a standalone C++17 fake backend and against negative header mutants.
They prove host-side coordination only: never CUDA ordering, device memory,
product build, exactness or overlap. No sleep-based probabilistic interleaving
assertions are used; every wait is a bounded condition-variable latch and the
whole process is bounded by a subprocess timeout.

Run: python3 -m unittest discover -s benchmark/owned_patch_match
     -p 'test_*.py' -v
"""

import resource
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src/colmap/mvs/owned_pair_coordinator.h"

PROGRAM = (
    "\n"
    '#include "colmap/mvs/owned_pair_coordinator.h"\n'
    '#include "colmap/mvs/owned_pair_run.h"\n'
    "\n"
    "#include <array>\n"
    "#include <atomic>\n"
    "#include <chrono>\n"
    "#include <condition_variable>\n"
    "#include <cstdint>\n"
    "#include <cstdio>\n"
    "#include <cstdlib>\n"
    "#include <exception>\n"
    "#include <functional>\n"
    "#include <mutex>\n"
    "#include <stdexcept>\n"
    "#include <string>\n"
    "#include <thread>\n"
    "#include <vector>\n"
    "\n"
    "using namespace colmap::mvs::owned;\n"
    "using namespace std::chrono_literals;\n"
    "\n"
    "static std::atomic<int> failures{0};\n"
    "\n"
    "static void OnTerminate() {\n"
    '  std::fprintf(stderr, "terminate: uncaught exception in a w'
    'orker\\n");\n'
    "  std::_Exit(3);\n"
    "}\n"
    "\n"
    "#define CHECK(cond)                                         "
    "               \\\n"
    "  do {                                                      "
    "               \\\n"
    "    if (!(cond)) {                                          "
    "               \\\n"
    '      std::fprintf(stderr, "CHECK failed at line %d: %s\\n", '
    "__LINE__, #cond); \\\n"
    "      ++failures;                                           "
    "               \\\n"
    "    }                                                       "
    "               \\\n"
    "  } while (0)\n"
    "\n"
    "#define REQUIRE(cond)                                       "
    "               \\\n"
    "  do {                                                      "
    "               \\\n"
    "    if (!(cond)) {                                          "
    "               \\\n"
    '      std::fprintf(stderr, "REQUIRE failed at line %d: %s\\n"'
    ", __LINE__, #cond); \\\n"
    "      std::exit(1);                                         "
    "               \\\n"
    "    }                                                       "
    "               \\\n"
    "  } while (0)\n"
    "\n"
    "static const ByteCharge CAP{1000, 2000};\n"
    "static const ByteCharge PROC{10, 20};\n"
    "static const std::array<ByteCharge, 2> JOBS{{ByteCharge{100,"
    " 200},\n"
    "                                             ByteCharge{300,"
    " 400}}};\n"
    "\n"
    "template <typename Fn>\n"
    "bool Throws(Fn fn) {\n"
    "  try {\n"
    "    fn();\n"
    "  } catch (...) {\n"
    "    return true;\n"
    "  }\n"
    "  return false;\n"
    "}\n"
    "\n"
    "struct Latch {\n"
    "  std::mutex mutex;\n"
    "  std::condition_variable cv;\n"
    "  bool flag = false;\n"
    "  void Set() {\n"
    "    std::lock_guard<std::mutex> lock(mutex);\n"
    "    flag = true;\n"
    "    cv.notify_all();\n"
    "  }\n"
    "  bool Wait(std::chrono::milliseconds timeout) {\n"
    "    std::unique_lock<std::mutex> lock(mutex);\n"
    "    return cv.wait_for(lock, timeout, [this] { return flag; "
    "});\n"
    "  }\n"
    "};\n"
    "\n"
    "struct EventLog {\n"
    "  std::mutex mutex;\n"
    "  std::condition_variable cv;\n"
    "  std::vector<PairEvent> events;\n"
    "  void Push(const PairEvent& event) {\n"
    "    std::lock_guard<std::mutex> lock(mutex);\n"
    "    events.push_back(event);\n"
    "    cv.notify_all();\n"
    "  }\n"
    "  bool WaitFor(const std::string& name, std::chrono::millise"
    "conds timeout) {\n"
    "    std::unique_lock<std::mutex> lock(mutex);\n"
    "    return cv.wait_for(lock, timeout, [&] {\n"
    "      for (const PairEvent& event : events) {\n"
    "        if (event.event == name) {\n"
    "          return true;\n"
    "        }\n"
    "      }\n"
    "      return false;\n"
    "    });\n"
    "  }\n"
    "  size_t Count(const std::string& name) {\n"
    "    std::lock_guard<std::mutex> lock(mutex);\n"
    "    size_t count = 0;\n"
    "    for (const PairEvent& event : events) {\n"
    "      if (event.event == name) {\n"
    "        ++count;\n"
    "      }\n"
    "    }\n"
    "    return count;\n"
    "  }\n"
    "  bool Seen(size_t job, const std::string& name) {\n"
    "    std::lock_guard<std::mutex> lock(mutex);\n"
    "    for (const PairEvent& event : events) {\n"
    "      if (event.job == job && event.event == name) {\n"
    "        return true;\n"
    "      }\n"
    "    }\n"
    "    return false;\n"
    "  }\n"
    "  uint64_t Sequence(size_t job, const std::string& name) {\n"
    "    std::lock_guard<std::mutex> lock(mutex);\n"
    "    for (const PairEvent& event : events) {\n"
    "      if (event.job == job && event.event == name) {\n"
    "        return event.sequence;\n"
    "      }\n"
    "    }\n"
    "    return 0;\n"
    "  }\n"
    "};\n"
    "\n"
    "static void BoundedSettle(std::chrono::milliseconds duration"
    ") {\n"
    "  std::mutex mutex;\n"
    "  std::condition_variable cv;\n"
    "  std::unique_lock<std::mutex> lock(mutex);\n"
    "  cv.wait_for(lock, duration, [] { return false; });\n"
    "}\n"
    "\n"
    "static void RunOwner(PairCoordinator* c, size_t job, bool pr"
    "oven = true) {\n"
    "  c->BeginPreparation(job);\n"
    "  c->ReadyAndWait(job);\n"
    "  c->SolverDone(job);\n"
    "  c->BeginOutput(job);\n"
    "  c->EndOutput(job);\n"
    "  c->AwaitOutputs(job);\n"
    "  c->Retire(job, proven);\n"
    "}\n"
    "\n"
    "static void ScenarioAdmissionAndLedger() {\n"
    "  using D = PairDispatch;\n"
    "  {\n"
    "    PairCoordinator c(D::kPair, ByteCharge{410, 620}, PROC, "
    "JOBS, 1000ms);\n"
    "    CHECK(c.ReservedBytes().host == 410 && c.ReservedBytes()"
    ".device == 620);\n"
    "  }\n"
    "  CHECK(Throws([] {\n"
    "    PairCoordinator c(D::kPair, ByteCharge{409, 620}, PROC, "
    "JOBS, 1000ms);\n"
    "    (void)c;\n"
    "  }));\n"
    "  CHECK(Throws([] {\n"
    "    PairCoordinator c(D::kPair, ByteCharge{410, 619}, PROC, "
    "JOBS, 1000ms);\n"
    "    (void)c;\n"
    "  }));\n"
    "  // Pair does not fit even though A alone (process + job_a)"
    " would.\n"
    "  CHECK(Throws([] {\n"
    "    PairCoordinator c(D::kPair, ByteCharge{110, 220}, PROC, "
    "JOBS, 1000ms);\n"
    "    (void)c;\n"
    "  }));\n"
    "  CHECK(Throws([] {\n"
    "    PairCoordinator c(D::kPair, CAP, ByteCharge{0, 20}, JOBS"
    ", 1000ms);\n"
    "    (void)c;\n"
    "  }));\n"
    "  CHECK(Throws([] {\n"
    "    PairCoordinator c(D::kPair, CAP, PROC,\n"
    "                      std::array<ByteCharge, 2>{{{0, 200}, {"
    "300, 400}}}, 1000ms);\n"
    "    (void)c;\n"
    "  }));\n"
    "  CHECK(Throws([] {\n"
    "    PairCoordinator c(D::kPair, ByteCharge{UINT64_MAX, UINT6"
    "4_MAX}, PROC,\n"
    "                      std::array<ByteCharge, 2>{{{UINT64_MAX"
    ", 1}, {1, 1}}},\n"
    "                      1000ms);\n"
    "    (void)c;\n"
    "  }));\n"
    "  CHECK(Throws([] {\n"
    "    PairCoordinator c(D::kPair, CAP, PROC, JOBS, 0ms);\n"
    "    (void)c;\n"
    "  }));\n"
    "}\n"
    "\n"
    "static void ScenarioStateMisuse() {\n"
    "  using D = PairDispatch;\n"
    "  PairCoordinator c(D::kPair, CAP, PROC, JOBS, 5000ms);\n"
    "  CHECK(Throws([&] { c.ReadyAndWait(0); }));\n"
    "  CHECK(Throws([&] { c.SolverDone(0); }));\n"
    "  CHECK(Throws([&] { c.BeginOutput(0); }));\n"
    "  CHECK(Throws([&] { c.EndOutput(0); }));\n"
    "  CHECK(Throws([&] { c.AwaitOutputs(0); }));\n"
    "  CHECK(Throws([&] { c.BeginPreparation(2); }));\n"
    "  CHECK(Throws([&] { c.SolverDone(7); }));\n"
    "  CHECK(Throws([&] { c.Retire(2); }));\n"
    "  CHECK(Throws([&] { c.Retire(0); }));\n"
    "  CHECK(Throws([&] { c.Finish(); }));\n"
    "}\n"
    "\n"
    "static void ScenarioSerialDispatch() {\n"
    "  EventLog log;\n"
    "  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, "
    "8000ms,\n"
    "                    [&log](const PairEvent& e) { log.Push(e)"
    "; });\n"
    "  std::thread a([&] { RunOwner(&c, 0); });\n"
    "  std::thread b([&] { RunOwner(&c, 1); });\n"
    "  a.join();\n"
    "  b.join();\n"
    '  CHECK(log.Seen(0, "prep_end") && log.Seen(1, "prep_end"));'
    "\n"
    '  CHECK(log.Count("both_ready") == 1);\n'
    '  CHECK(log.Seen(0, "solver_release") && log.Seen(1, "solver'
    '_release"));\n'
    '  CHECK(log.Seen(0, "solver_start") && log.Seen(1, "solver_s'
    'tart"));\n'
    "  // serial: B's solver release/start must follow A's solver"
    " return.\n"
    '  CHECK(log.Sequence(0, "solver_end") < log.Sequence(1, "sol'
    'ver_release"));\n'
    '  CHECK(log.Sequence(1, "solver_release") < log.Sequence(1, '
    '"solver_start"));\n'
    '  CHECK(log.Seen(0, "output_end") && log.Seen(1, "output_end'
    '"));\n'
    "  // Credits stay reserved until Finish, which runs after bo"
    "th joins.\n"
    "  CHECK(c.ReservedBytes().host == 410 && c.ReservedBytes().d"
    "evice == 620);\n"
    "  c.Finish();\n"
    "  CHECK(c.ReservedBytes().host == 0 && c.ReservedBytes().dev"
    "ice == 0);\n"
    '  CHECK(log.Count("ledger_release") == 1);\n'
    "}\n"
    "\n"
    "static void ScenarioPairRelease() {\n"
    "  EventLog log;\n"
    "  PairCoordinator c(PairDispatch::kPair, CAP, PROC, JOBS, 80"
    "00ms,\n"
    "                    [&log](const PairEvent& e) { log.Push(e)"
    "; });\n"
    "  std::mutex barrier_mutex;\n"
    "  std::condition_variable barrier_cv;\n"
    "  int arrived = 0;\n"
    "  std::atomic<bool> ok0{false};\n"
    "  std::atomic<bool> ok1{false};\n"
    "  auto arrive = [&]() {\n"
    "    std::unique_lock<std::mutex> lock(barrier_mutex);\n"
    "    ++arrived;\n"
    "    barrier_cv.notify_all();\n"
    "    return barrier_cv.wait_for(lock, 4000ms, [&] { return ar"
    "rived == 2; });\n"
    "  };\n"
    "  std::thread a([&] {\n"
    "    c.BeginPreparation(0);\n"
    "    c.ReadyAndWait(0);\n"
    "    ok0.store(arrive());\n"
    "    c.SolverDone(0);\n"
    "    c.BeginOutput(0);\n"
    "    c.EndOutput(0);\n"
    "    c.AwaitOutputs(0);\n"
    "    c.Retire(0, true);\n"
    "  });\n"
    "  std::thread b([&] {\n"
    "    c.BeginPreparation(1);\n"
    "    c.ReadyAndWait(1);\n"
    "    ok1.store(arrive());\n"
    "    c.SolverDone(1);\n"
    "    c.BeginOutput(1);\n"
    "    c.EndOutput(1);\n"
    "    c.AwaitOutputs(1);\n"
    "    c.Retire(1, true);\n"
    "  });\n"
    "  a.join();\n"
    "  b.join();\n"
    "  // pair grants both solver permits once both jobs are read"
    "y: both jobs reach\n"
    "  // the barrier before either records SolverDone.\n"
    "  CHECK(ok0.load() && ok1.load());\n"
    '  CHECK(log.Count("both_ready") == 1);\n'
    '  CHECK(log.Seen(0, "solver_release") && log.Seen(1, "solver'
    '_release"));\n'
    '  CHECK(log.Count("solver_start") == 2);\n'
    "  CHECK(c.ReservedBytes().host == 410);  // full until Finis"
    "h\n"
    "  c.Finish();\n"
    "  CHECK(c.ReservedBytes().host == 0);\n"
    '  CHECK(log.Count("ledger_release") == 1);\n'
    "}\n"
    "\n"
    "static void ScenarioBothReadyGate() {\n"
    "  EventLog log;\n"
    "  Latch b_parked;\n"
    "  Latch allow_b;\n"
    "  PairCoordinator c(PairDispatch::kPair, CAP, PROC, JOBS, 80"
    "00ms,\n"
    "                    [&log](const PairEvent& e) { log.Push(e)"
    "; });\n"
    "  std::thread a([&] {\n"
    "    c.BeginPreparation(0);\n"
    "    c.ReadyAndWait(0);\n"
    "    c.SolverDone(0);\n"
    "    c.BeginOutput(0);\n"
    "    c.EndOutput(0);\n"
    "    c.AwaitOutputs(0);\n"
    "    c.Retire(0, true);\n"
    "  });\n"
    "  std::thread b([&] {\n"
    "    c.BeginPreparation(1);\n"
    "    b_parked.Set();\n"
    "    allow_b.Wait(8000ms);\n"
    "    c.ReadyAndWait(1);\n"
    "    c.SolverDone(1);\n"
    "    c.BeginOutput(1);\n"
    "    c.EndOutput(1);\n"
    "    c.AwaitOutputs(1);\n"
    "    c.Retire(1, true);\n"
    "  });\n"
    "  REQUIRE(b_parked.Wait(4000ms));\n"
    "  // The both-ready gate must hold A while B is parked: no s"
    "olver may start.\n"
    '  CHECK(!log.WaitFor("solver_start", 500ms));\n'
    "  allow_b.Set();\n"
    "  a.join();\n"
    "  b.join();\n"
    '  CHECK(log.Count("solver_start") == 2);\n'
    "  CHECK(c.ReservedBytes().host == 410);  // full until Finis"
    "h\n"
    "  c.Finish();\n"
    "  CHECK(c.ReservedBytes().host == 0);\n"
    "}\n"
    "\n"
    "static void ScenarioCreditGate() {\n"
    "  EventLog log;\n"
    "  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, "
    "8000ms,\n"
    "                    [&log](const PairEvent& e) { log.Push(e)"
    "; });\n"
    "  Latch retired0;\n"
    "  Latch retired1;\n"
    "  Latch allow_finish;\n"
    "  auto owner_gated = [&](size_t job, Latch* retired) {\n"
    "    c.BeginPreparation(job);\n"
    "    c.ReadyAndWait(job);\n"
    "    c.SolverDone(job);\n"
    "    c.BeginOutput(job);\n"
    "    c.EndOutput(job);\n"
    "    c.AwaitOutputs(job);\n"
    "    c.Retire(job, true);\n"
    "    retired->Set();\n"
    "    allow_finish.Wait(8000ms);\n"
    "  };\n"
    "  std::thread a([&] { owner_gated(0, &retired0); });\n"
    "  std::thread b([&] { owner_gated(1, &retired1); });\n"
    "  REQUIRE(retired0.Wait(4000ms) && retired1.Wait(4000ms));\n"
    "  // Both owners retired and still alive: reservation must r"
    "emain full and no\n"
    "  // ledger_release may have been emitted yet.\n"
    "  CHECK(c.ReservedBytes().host == 410 && c.ReservedBytes().d"
    "evice == 620);\n"
    '  CHECK(log.Count("ledger_release") == 0);\n'
    "  allow_finish.Set();\n"
    "  a.join();\n"
    "  b.join();\n"
    "  // Still full after both joins, before Finish.\n"
    "  CHECK(c.ReservedBytes().host == 410 && c.ReservedBytes().d"
    "evice == 620);\n"
    "  c.Finish();\n"
    "  CHECK(c.ReservedBytes().host == 0 && c.ReservedBytes().dev"
    "ice == 0);\n"
    '  CHECK(log.Count("ledger_release") == 1);\n'
    "}\n"
    "\n"
    "static void ScenarioCancellationNotify() {\n"
    "  EventLog log;\n"
    "  Latch a_threw;\n"
    "  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, "
    "8000ms,\n"
    "                    [&log](const PairEvent& e) { log.Push(e)"
    "; });\n"
    "  std::thread a([&] {\n"
    "    try {\n"
    "      c.BeginPreparation(0);\n"
    "      c.ReadyAndWait(0);\n"
    "    } catch (...) {\n"
    "      a_threw.Set();\n"
    "    }\n"
    "    c.Retire(0, true);\n"
    "  });\n"
    '  REQUIRE(log.WaitFor("ready", 4000ms));\n'
    "  BoundedSettle(250ms);\n"
    '  c.Cancel(0, std::make_exception_ptr(std::runtime_error("ca'
    'ncel-boom")));\n'
    "  // Cancellation must wake the blocked waiter promptly (not"
    " at its deadline).\n"
    "  CHECK(a_threw.Wait(2000ms));\n"
    "  a.join();\n"
    "  c.Retire(1, true);\n"
    "  CHECK(c.ReservedBytes().host == 410);  // still full until"
    " Finish\n"
    "  CHECK(Throws([&] { c.Finish(); }));\n"
    "  CHECK(c.ReservedBytes().host == 0);\n"
    "}\n"
    "\n"
    "static void ScenarioStartupFailure() {\n"
    "  using D = PairDispatch;\n"
    "  PairCoordinator c(D::kSerial, CAP, PROC, JOBS, 8000ms);\n"
    '  auto error = std::make_exception_ptr(std::runtime_error("p'
    'rep-boom"));\n'
    "  c.Cancel(0, error);\n"
    "  CHECK(Throws([&] { c.BeginPreparation(0); }));\n"
    "  CHECK(Throws([&] { c.BeginPreparation(1); }));\n"
    "  CHECK(c.ReservedBytes().host == 410);  // cancellation rel"
    "eases no credits\n"
    "  c.Retire(0, true);\n"
    "  c.Retire(1, true);\n"
    "  CHECK(c.ReservedBytes().host == 410);  // released only by"
    " Finish\n"
    "  CHECK(Throws([&] { c.Finish(); }));  // clean cancellation"
    " releases then throws\n"
    "  CHECK(c.ReservedBytes().host == 0 && c.ReservedBytes().dev"
    "ice == 0);\n"
    "  // A second cancellation with a different error must not o"
    "verwrite the first.\n"
    "  PairCoordinator c2(D::kSerial, CAP, PROC, JOBS, 5000ms);\n"
    '  c2.Cancel(0, std::make_exception_ptr(std::runtime_error("f'
    'irst")));\n'
    '  c2.Cancel(1, std::make_exception_ptr(std::runtime_error("s'
    'econd")));\n'
    "  c2.Retire(0, true);\n"
    "  c2.Retire(1, true);\n"
    "  CHECK(c2.ReservedBytes().host == 410);\n"
    "  bool first_survived = false;\n"
    "  try {\n"
    "    c2.Finish();\n"
    "  } catch (const std::runtime_error& e) {\n"
    '    first_survived = std::string(e.what()) == "first";\n'
    "  }\n"
    "  CHECK(first_survived);\n"
    "  CHECK(c2.ReservedBytes().host == 0);\n"
    "}\n"
    "\n"
    "static void ScenarioSerialPeerNotDispatched() {\n"
    "  EventLog log;\n"
    "  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, "
    "8000ms,\n"
    "                    [&log](const PairEvent& e) { log.Push(e)"
    "; });\n"
    "  std::atomic<bool> b_dispatched{false};\n"
    '  auto error = std::make_exception_ptr(std::runtime_error("s'
    'olver-boom"));\n'
    "  std::thread a([&] {\n"
    "    c.BeginPreparation(0);\n"
    "    c.ReadyAndWait(0);\n"
    "    c.Cancel(0, error);\n"
    "    c.Retire(0, true);\n"
    "  });\n"
    "  std::thread b([&] {\n"
    "    try {\n"
    "      c.BeginPreparation(1);\n"
    "      c.ReadyAndWait(1);\n"
    "      b_dispatched.store(true);\n"
    "    } catch (...) {\n"
    "    }\n"
    "    try {\n"
    "      c.Retire(1, true);\n"
    "    } catch (...) {\n"
    "    }\n"
    "  });\n"
    "  a.join();\n"
    "  b.join();\n"
    "  CHECK(!b_dispatched.load());\n"
    '  CHECK(!log.Seen(1, "solver_start"));\n'
    "  CHECK(c.ReservedBytes().host == 410);  // full until Finis"
    "h\n"
    "  CHECK(Throws([&] { c.Finish(); }));\n"
    "  CHECK(c.ReservedBytes().host == 0);\n"
    "}\n"
    "\n"
    "static void ScenarioOutputFault() {\n"
    "  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, "
    "8000ms);\n"
    '  auto error = std::make_exception_ptr(std::runtime_error("o'
    'utput-boom"));\n'
    "  std::thread a([&] {\n"
    "    try {\n"
    "      c.BeginPreparation(0);\n"
    "      c.ReadyAndWait(0);\n"
    "      c.SolverDone(0);\n"
    "      c.BeginOutput(0);\n"
    "      c.Cancel(0, error);\n"
    "      c.EndOutput(0);\n"
    "      c.AwaitOutputs(0);\n"
    "    } catch (...) {\n"
    "    }\n"
    "    try {\n"
    "      c.Retire(0, true);\n"
    "    } catch (...) {\n"
    "    }\n"
    "  });\n"
    "  std::thread b([&] {\n"
    "    try {\n"
    "      c.BeginPreparation(1);\n"
    "      c.ReadyAndWait(1);\n"
    "      c.SolverDone(1);\n"
    "      c.BeginOutput(1);\n"
    "      c.EndOutput(1);\n"
    "      c.AwaitOutputs(1);\n"
    "    } catch (...) {\n"
    "    }\n"
    "    try {\n"
    "      c.Retire(1, true);\n"
    "    } catch (...) {\n"
    "    }\n"
    "  });\n"
    "  a.join();\n"
    "  b.join();\n"
    "  CHECK(c.ReservedBytes().host == 410);  // full until Finis"
    "h\n"
    "  CHECK(Throws([&] { c.Finish(); }));\n"
    "  CHECK(c.ReservedBytes().host == 0);\n"
    "}\n"
    "\n"
    "static void ScenarioRetireGuards() {\n"
    "  EventLog log;\n"
    "  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, "
    "8000ms,\n"
    "                    [&log](const PairEvent& e) { log.Push(e)"
    "; });\n"
    "  Latch a_output_lease;\n"
    "  Latch allow_a_end;\n"
    "  std::thread a([&] {\n"
    "    c.BeginPreparation(0);\n"
    "    c.ReadyAndWait(0);\n"
    "    c.SolverDone(0);\n"
    "    c.BeginOutput(0);\n"
    "    a_output_lease.Set();\n"
    "    allow_a_end.Wait(8000ms);\n"
    "    c.EndOutput(0);\n"
    "    c.AwaitOutputs(0);\n"
    "    c.Retire(0, true);\n"
    "  });\n"
    "  std::thread b([&] {\n"
    "    c.BeginPreparation(1);\n"
    "    c.ReadyAndWait(1);\n"
    "    c.SolverDone(1);\n"
    "    c.BeginOutput(1);\n"
    "    c.EndOutput(1);\n"
    "    c.AwaitOutputs(1);\n"
    "    c.Retire(1, true);\n"
    "  });\n"
    "  REQUIRE(a_output_lease.Wait(4000ms));\n"
    "  // Single output lease: B must not have started output whi"
    "le A holds it.\n"
    '  CHECK(!log.Seen(1, "output_start"));\n'
    "  // Unsafe: A is not cancelled, has not completed output, a"
    "nd still holds the\n"
    "  // output lease, so retirement must reject without touchin"
    "g the ledger.\n"
    "  CHECK(Throws([&] { c.Retire(0, true); }));\n"
    "  CHECK(Throws([&] { c.EndOutput(1); }));  // B has no activ"
    "e output lease\n"
    "  CHECK(c.ReservedBytes().host == 410);\n"
    "  allow_a_end.Set();\n"
    "  a.join();\n"
    "  b.join();\n"
    "  CHECK(c.ReservedBytes().host == 410);  // full until Finis"
    "h\n"
    "  c.Finish();\n"
    "  CHECK(c.ReservedBytes().host == 0);\n"
    "}\n"
    "\n"
    "static void ScenarioCleanupUnproven() {\n"
    "  EventLog log;\n"
    "  PairCoordinator c(PairDispatch::kPair, CAP, PROC, JOBS, 80"
    "00ms,\n"
    "                    [&log](const PairEvent& e) { log.Push(e)"
    "; });\n"
    "  std::thread a([&] { RunOwner(&c, 0, /*proven=*/false); });"
    "\n"
    "  std::thread b([&] { RunOwner(&c, 1, true); });\n"
    "  a.join();\n"
    "  b.join();\n"
    "  CHECK(c.ReservedBytes().host == 410 && c.ReservedBytes().d"
    "evice == 620);\n"
    "  CHECK(Throws([&] { c.Finish(); }));     // unknown cleanup"
    " must fail\n"
    "  CHECK(c.ReservedBytes().host == 410);   // ...and release "
    "nothing\n"
    '  CHECK(log.Count("ledger_release") == 0);\n'
    "  bool unproven = false;\n"
    "  try {\n"
    "    c.Finish();\n"
    "  } catch (const std::runtime_error& e) {\n"
    "    unproven = std::string(e.what()) ==\n"
    '               "owned pair: retirement completion unproven";'
    "\n"
    "  }\n"
    "  CHECK(unproven);\n"
    "}\n"
    "\n"
    "static void ScenarioTimeout() {\n"
    "  PairCoordinator c(PairDispatch::kSerial, CAP, PROC, JOBS, "
    "150ms);\n"
    "  CHECK(Throws([&] {\n"
    "    c.BeginPreparation(0);\n"
    "    c.ReadyAndWait(0);\n"
    "  }));\n"
    "  c.Retire(0, true);\n"
    "  CHECK(c.ReservedBytes().host == 410);  // peer not retired"
    " yet\n"
    "  c.Retire(1, true);\n"
    "  CHECK(c.ReservedBytes().host == 410);  // released only by"
    " Finish\n"
    "  CHECK(Throws([&] { c.Finish(); }));\n"
    "  CHECK(c.ReservedBytes().host == 0);\n"
    "}\n"
    "\n"
    "static void ScenarioSinkThrow() {\n"
    "  using D = PairDispatch;\n"
    "  CHECK(Throws([] {\n"
    "    PairCoordinator c(D::kPair, CAP, PROC, JOBS, 1000ms,\n"
    "                      [](const PairEvent&) {\n"
    '                        throw std::runtime_error("sink-admit'
    'ted");\n'
    "                      });\n"
    "    (void)c;\n"
    "  }));\n"
    "  EventLog log;\n"
    "  PairCoordinator c(D::kSerial, CAP, PROC, JOBS, 8000ms,\n"
    "                    [&log](const PairEvent& e) {\n"
    '                      if (e.event == "prep_start") {\n'
    '                        throw std::runtime_error("sink-prep"'
    ");\n"
    "                      }\n"
    "                      log.Push(e);\n"
    "                    });\n"
    "  CHECK(Throws([&] { c.BeginPreparation(0); }));\n"
    "  CHECK(Throws([&] { c.BeginPreparation(1); }));\n"
    "  c.Retire(0, true);\n"
    "  c.Retire(1, true);\n"
    "  CHECK(c.ReservedBytes().host == 410);  // full until Finis"
    "h\n"
    "  CHECK(Throws([&] { c.Finish(); }));\n"
    "  CHECK(c.ReservedBytes().host == 0);\n"
    "}\n"
    "\n"
    "static void ScenarioSinkReentrancy() {\n"
    "  PairCoordinator* self = nullptr;\n"
    "  PairCoordinator c(PairDispatch::kPair, CAP, PROC, JOBS, 80"
    "00ms,\n"
    "                    [&self](const PairEvent&) {\n"
    "                      if (self != nullptr) {\n"
    "                        // Deadlocks if the sink is invoked "
    "under the mutex.\n"
    "                        (void)self->ReservedBytes();\n"
    "                      }\n"
    "                    });\n"
    "  self = &c;\n"
    "  std::thread a([&] { RunOwner(&c, 0); });\n"
    "  std::thread b([&] { RunOwner(&c, 1); });\n"
    "  a.join();\n"
    "  b.join();\n"
    "  CHECK(c.ReservedBytes().host == 410);\n"
    "  c.Finish();\n"
    "  CHECK(c.ReservedBytes().host == 0);\n"
    "}\n"
    "\n"
    "\n"
    "static void ScenarioEarlyOrdering(PairDispatch dispatch,\n"
    "                                  EarlyGeometryPermits early"
    ") {\n"
    "  EventLog log;\n"
    "  Latch b_preparing, allow_b, a_solving;\n"
    "  PairCoordinator c(dispatch, CAP, PROC, JOBS, 8000ms,\n"
    "                    [&](const PairEvent& e) { log.Push(e); }"
    ", early);\n"
    "  std::thread a([&] {\n"
    "    c.BeginPreparation(0);\n"
    "    c.ReadyAndWait(0);\n"
    "    a_solving.Set();\n"
    "    c.SolverDone(0);\n"
    "    c.BeginOutput(0);\n"
    "    c.EndOutput(0);\n"
    "    c.AwaitOutputs(0);\n"
    "    c.Retire(0);\n"
    "  });\n"
    "  std::thread b([&] {\n"
    "    c.BeginPreparation(1);\n"
    "    b_preparing.Set();\n"
    "    REQUIRE(allow_b.Wait(4000ms));\n"
    "    c.ReadyAndWait(1);\n"
    "    c.SolverDone(1);\n"
    "    c.BeginOutput(1);\n"
    "    c.EndOutput(1);\n"
    "    c.AwaitOutputs(1);\n"
    "    c.Retire(1);\n"
    "  });\n"
    "  REQUIRE(b_preparing.Wait(4000ms));\n"
    "  const bool starts_early = dispatch == PairDispatch::kPair "
    "&&\n"
    "                            early == EarlyGeometryPermits::k"
    "On;\n"
    "  if (starts_early) {\n"
    "    REQUIRE(a_solving.Wait(4000ms));\n"
    '    CHECK(!log.Seen(1, "ready"));\n'
    '    CHECK(!log.Seen(0, "retired"));\n'
    "    CHECK(c.ReservedBytes().host == 410);\n"
    "  }\n"
    "  allow_b.Set();\n"
    "  a.join(); b.join();\n"
    '  CHECK(log.Count("solver_release") == 2);\n'
    '  CHECK(log.Count("solver_start") == 2);\n'
    '  CHECK(log.Count("output_end") == 2);\n'
    '  CHECK(log.Count("retired") == 2);\n'
    '  CHECK(log.Sequence(0, "prep_end") < log.Sequence(1, "prep_'
    'start"));\n'
    "  if (starts_early) {\n"
    '    CHECK(log.Sequence(0, "solver_start") < log.Sequence(1, '
    '"ready"));\n'
    "  } else {\n"
    '    CHECK(log.Sequence(1, "ready") < log.Sequence(0, "solver'
    '_start"));\n'
    "  }\n"
    "  if (dispatch == PairDispatch::kSerial) {\n"
    '    CHECK(log.Sequence(0, "solver_end") < log.Sequence(1, "s'
    'olver_start"));\n'
    "  }\n"
    "  CHECK(Throws([&] { c.Retire(0); }));\n"
    "  c.Finish(); c.Finish();\n"
    '  CHECK(log.Count("ledger_release") == 1);\n'
    "}\n"
    "\n"
    "static void ScenarioEarlyOutputLease() {\n"
    "  EventLog log;\n"
    "  Latch a_outputting, b_computed, allow_a_end;\n"
    "  PairCoordinator c(PairDispatch::kPair, CAP, PROC, JOBS, 80"
    "00ms,\n"
    "                    [&](const PairEvent& e) { log.Push(e); }"
    ",\n"
    "                    EarlyGeometryPermits::kOn);\n"
    "  std::thread a([&] {\n"
    "    c.BeginPreparation(0);\n"
    "    c.ReadyAndWait(0);\n"
    "    c.SolverDone(0);\n"
    "    CHECK(Throws([&] { c.SolverDone(0); }));\n"
    "    c.BeginOutput(0);\n"
    "    a_outputting.Set();\n"
    "    REQUIRE(allow_a_end.Wait(4000ms));\n"
    "    c.EndOutput(0);\n"
    "    CHECK(Throws([&] { c.EndOutput(0); }));\n"
    "    c.AwaitOutputs(0);\n"
    "    c.Retire(0);\n"
    "  });\n"
    "  std::thread b([&] {\n"
    "    c.BeginPreparation(1);\n"
    "    c.ReadyAndWait(1);\n"
    "    REQUIRE(a_outputting.Wait(4000ms));\n"
    "    c.SolverDone(1);\n"
    "    b_computed.Set();\n"
    "    c.BeginOutput(1);\n"
    "    c.EndOutput(1);\n"
    "    c.AwaitOutputs(1);\n"
    "    c.Retire(1);\n"
    "  });\n"
    "  REQUIRE(b_computed.Wait(4000ms));\n"
    '  CHECK(!log.Seen(1, "output_start"));\n'
    "  CHECK(c.ReservedBytes().host == 410);\n"
    "  allow_a_end.Set();\n"
    "  a.join(); b.join();\n"
    '  CHECK(log.Sequence(0, "output_end") < log.Sequence(1, "out'
    'put_start"));\n'
    "  c.Finish();\n"
    '  CHECK(log.Count("ledger_release") == 1);\n'
    "}\n"
    "\n"
    "// Failure occurs while the peer's solver is held by a deter"
    "ministic latch.\n"
    "// Resource cleanup must see that peer's successful completi"
    "on, not merely\n"
    "// cancellation. Exercise the real join/retirement wrapper a"
    "s production does.\n"
    "static void ScenarioEarlyFailure(const std::string& phase) {"
    "\n"
    "  EventLog log;\n"
    "  Latch peer_solving, failure_drained, allow_peer;\n"
    "  std::atomic<bool> peer_complete{false};\n"
    "  std::array<std::atomic<int>, 2> cleanup_count{};\n"
    "  PairCoordinator c(PairDispatch::kPair, CAP, PROC, JOBS, 80"
    "00ms,\n"
    "                    [&](const PairEvent& e) { log.Push(e); }"
    ",\n"
    "                    EarlyGeometryPermits::kOn);\n"
    '  const size_t failed = phase == "preparation" ? 1 : 0;\n'
    "  const size_t peer = 1 - failed;\n"
    "  auto error = std::make_exception_ptr(std::runtime_error(ph"
    "ase));\n"
    "  std::thread runner([&] {\n"
    "    bool caught = false;\n"
    "    try {\n"
    "      RunPairOwners(c, [&](size_t job) {\n"
    "        try {\n"
    "          c.BeginPreparation(job);\n"
    '          if (job == failed && phase == "preparation") {\n'
    "            REQUIRE(peer_solving.Wait(4000ms));\n"
    "            std::rethrow_exception(error);\n"
    "          }\n"
    "          c.ReadyAndWait(job);\n"
    "          if (job == peer) {\n"
    "            peer_solving.Set();\n"
    "            REQUIRE(allow_peer.Wait(4000ms));\n"
    "            peer_complete.store(true);\n"
    "            c.SolverDone(job);\n"
    "          } else {\n"
    "            REQUIRE(peer_solving.Wait(4000ms));\n"
    '            if (phase == "output") {\n'
    "              c.SolverDone(job);\n"
    "              c.BeginOutput(job);\n"
    "            }\n"
    "            std::rethrow_exception(error);\n"
    "          }\n"
    "          c.BeginOutput(job);\n"
    "          c.EndOutput(job);\n"
    "          c.AwaitOutputs(job);\n"
    "        } catch (...) {\n"
    "          c.CancelAndAwaitCleanup(job, std::current_exceptio"
    "n(), [&] {\n"
    "            if (job == failed) failure_drained.Set();\n"
    "          });\n"
    "          throw;\n"
    "        }\n"
    "      }, [](size_t, const char*) {}, PairThreadFactory{}, [&"
    "](size_t job) {\n"
    "        CHECK(peer_complete.load());\n"
    "        ++cleanup_count[job];\n"
    "        CHECK(c.ReservedBytes().host == 410);\n"
    "      });\n"
    "    } catch (const std::runtime_error& e) {\n"
    "      caught = std::string(e.what()) == phase;\n"
    "    }\n"
    "    CHECK(caught);\n"
    "  });\n"
    "  REQUIRE(failure_drained.Wait(4000ms));\n"
    "  CHECK(!peer_complete.load());\n"
    '  CHECK(log.Count("retired") == 0);\n'
    '  CHECK(log.Count("ledger_release") == 0);\n'
    "  CHECK(c.ReservedBytes().host == 410);\n"
    "  // Explicit external cancellation must not replace the ori"
    "ginal exception.\n"
    "  c.Cancel(peer, std::make_exception_ptr(std::runtime_error("
    '"external cancel")));\n'
    "  allow_peer.Set();\n"
    "  runner.join();\n"
    "  CHECK(cleanup_count[0] == 1 && cleanup_count[1] == 1);\n"
    '  CHECK(log.Count("retired") == 2);\n'
    '  CHECK(log.Count("ledger_release") == 1);\n'
    "  CHECK(c.ReservedBytes().host == 0);\n"
    "}\n"
    "\n"
    "int main() {\n"
    "  std::set_terminate(OnTerminate);\n"
    "  ScenarioEarlyOrdering(PairDispatch::kPair, EarlyGeometryPe"
    "rmits::kOn);\n"
    "  ScenarioEarlyOrdering(PairDispatch::kPair, EarlyGeometryPe"
    "rmits::kOff);\n"
    "  ScenarioEarlyOrdering(PairDispatch::kSerial, EarlyGeometry"
    "Permits::kOn);\n"
    "  ScenarioEarlyOrdering(PairDispatch::kSerial, EarlyGeometry"
    "Permits::kOff);\n"
    "  ScenarioEarlyOutputLease();\n"
    '  ScenarioEarlyFailure("preparation");\n'
    '  ScenarioEarlyFailure("solver");\n'
    '  ScenarioEarlyFailure("output");\n'
    "  ScenarioAdmissionAndLedger();\n"
    "  ScenarioStateMisuse();\n"
    "  ScenarioSerialDispatch();\n"
    "  ScenarioPairRelease();\n"
    "  ScenarioBothReadyGate();\n"
    "  ScenarioCreditGate();\n"
    "  ScenarioCancellationNotify();\n"
    "  ScenarioStartupFailure();\n"
    "  ScenarioSerialPeerNotDispatched();\n"
    "  ScenarioOutputFault();\n"
    "  ScenarioRetireGuards();\n"
    "  ScenarioCleanupUnproven();\n"
    "  ScenarioTimeout();\n"
    "  ScenarioSinkThrow();\n"
    "  ScenarioSinkReentrancy();\n"
    "  if (failures != 0) {\n"
    '    std::fprintf(stderr, "%d pair coordinator check(s) faile'
    'd\\n", failures.load());\n'
    "    return 1;\n"
    "  }\n"
    '  std::printf("all pair coordinator checks passed\\n");\n'
    "  return 0;\n"
    "}\n"
)


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class PairCoordinatorTest(unittest.TestCase):
    def run_program(self, mutation=None):
        source = HEADER.read_text()
        if mutation is not None:
            source = mutation(source)
        temp = Path(tempfile.mkdtemp(prefix="owned-pair-"))
        header_dir = temp / "colmap" / "mvs"
        header_dir.mkdir(parents=True)
        (header_dir / "owned_pair_coordinator.h").write_text(source)
        program = temp / "program.cc"
        program.write_text(PROGRAM)
        binary = temp / "program"
        command = [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O1",
            "-pthread",
            "-I",
            str(temp),
            "-I",
            str(ROOT / "src"),
            str(program),
            "-o",
            str(binary),
        ]
        compiled = subprocess.run(
            command, capture_output=True, text=True, timeout=60
        )
        (temp / "compile.log").write_text(compiled.stdout + compiled.stderr)
        self.assertEqual(
            compiled.returncode,
            0,
            "compile failed:\n" + compiled.stdout + compiled.stderr,
        )
        try:
            result = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                timeout=60,
                preexec_fn=no_core,
            )
        except subprocess.TimeoutExpired:
            print(f"PAIR_COORDINATOR_TIMEOUT artifact={temp}")
            return None
        (temp / "run.log").write_text(result.stdout + result.stderr)
        print(f"PAIR_COORDINATOR_ARTIFACT {temp} exit={result.returncode}")
        return result

    def test_real_header_behavior(self):
        result = self.run_program()
        self.assertIsNotNone(result, "real header program hung / timed out")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mutant_missing_early_permit(self):
        def mutation(source):
            old = "early_geometry_permits_ == EarlyGeometryPermits::kOn;"
            self.assertEqual(source.count(old), 1)
            return source.replace(
                old, "early_geometry_permits_ == EarlyGeometryPermits::kOff;"
            )

        result = self.run_program(mutation)
        self.assertTrue(
            result is None or result.returncode != 0,
            "missing early permit was not detected",
        )

    def test_mutant_duplicate_early_release(self):
        def mutation(source):
            old = (
                '        pending.push_back(MakeEventLocked(job, "solver_relea'
                'se"));'
            )
            self.assertEqual(source.count(old), 1)
            return source.replace(old, old + "\n" + old)

        result = self.run_program(mutation)
        self.assertTrue(
            result is None or result.returncode != 0,
            "duplicate early solver release was not detected",
        )

    def test_mutant_bypassed_peer_cleanup_fence(self):
        def mutation(source):
            old = (
                "return std::none_of(solver_active_.begin(),\n"
                "                                solver_active_.begin()"
                " + count_,\n"
                "                                [](bool active) { return"
                " active; });"
            )
            self.assertEqual(source.count(old), 1)
            return source.replace(old, "return true;")

        result = self.run_program(mutation)
        self.assertTrue(
            result is None or result.returncode != 0,
            "bypassed peer cleanup fence was not detected",
        )

    def test_mutant_missing_both_ready_gate(self):
        def mutation(source):
            old = "if (!rolling_ && !both_ready_closed_ && All(ready_)) {"
            self.assertEqual(
                source.count(old), 1, "both-ready gate anchor not unique"
            )
            return source.replace(
                old, "if (!rolling_ && !both_ready_closed_) {  // mutant"
            )

        result = self.run_program(mutation)
        self.assertTrue(
            result is None or result.returncode != 0,
            "missing both-ready gate was not detected",
        )

    def test_mutant_missing_cancellation_notification(self):
        def mutation(source):
            old = (
                "  void PublishCancellationLocked() {\n"
                "    cancelled_ = true;\n"
                "    cv_.notify_all();\n"
                "  }"
            )
            self.assertEqual(
                source.count(old), 1, "cancellation-notify anchor not unique"
            )
            return source.replace(
                old,
                "  void PublishCancellationLocked() {\n"
                "    cancelled_ = true;\n"
                "  }",
            )

        result = self.run_program(mutation)
        self.assertTrue(
            result is None or result.returncode != 0,
            "missing cancellation notification was not detected",
        )

    def test_mutant_credit_early_release(self):
        def mutation(source):
            old = "      retired_[job] = true;"
            self.assertEqual(
                source.count(old), 1, "retirement anchor not unique"
            )
            return source.replace(
                old,
                (
                    "      retired_[job] = true;\n"
                    "      reserved_ = ByteCharge{0, 0};  // mutant: release "
                    "at Retire"
                ),
            )

        result = self.run_program(mutation)
        self.assertTrue(
            result is None or result.returncode != 0,
            "early credit release before Finish was not detected",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
