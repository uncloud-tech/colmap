// SPDX-License-Identifier: BSD-3-Clause
// Dependency-free host regression: compile with -std=c++17 -pthread -Isrc.
#include "colmap/mvs/native_run_deadline.h"

#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_run.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

using namespace colmap::mvs::owned;
using namespace std::chrono_literals;

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "setup-expiry") {
    colmap::mvs::NativeRunDeadlineGuard guard(
        colmap::mvs::NativeRunDeadline::FromBudget(30ms));
    std::this_thread::sleep_for(5s);  // Synchronous setup/join never returns.
    return 1;
  }
  if (argc == 2 && std::string(argv[1]) == "retirement-expiry") {
    PairCoordinator coordinator({10, 10}, {1, 1}, size_t{1}, {{1, 1}}, 20ms);
    coordinator.AdmitNext();
    coordinator.AwaitRetired();
    return 1;
  }
  if (argc == 2 && std::string(argv[1]) == "drain-unknown") {
    PairCoordinator coordinator({10, 10}, {1, 1}, size_t{1}, {{1, 1}}, 20ms);
    coordinator.AdmitNext();
    coordinator.CancelAndAwaitCleanup(
        0, {}, [] { throw std::runtime_error("CUDA completion unknown"); });
    return 1;
  }
  if (argc == 2 && std::string(argv[1]) == "peer-drain-expiry") {
    PairCoordinator coordinator(
        {10, 10}, {1, 1}, size_t{2}, {{1, 1}, {1, 1}}, 20ms);
    coordinator.AdmitNext();
    coordinator.AdmitNext();
    for (size_t j = 0; j < 2; ++j) {
      coordinator.BeginPreparation(j);
      coordinator.ReadyAndWait(j);
    }
    coordinator.CancelAndAwaitCleanup(0, {}, [] {});
    return 1;
  }
  if (argc == 2 && std::string(argv[1]) == "cohort-no-reset") {
    const auto deadline = colmap::mvs::NativeRunDeadline::FromBudget(120ms);
    for (int cohort = 0; cohort < 2; ++cohort) {
      PairCoordinator coordinator(
          {10, 10}, {1, 1}, size_t{1}, {{1, 1}}, 500ms, {}, deadline);
      RunRollingOwners(
          coordinator,
          [&](size_t job) {
            coordinator.BeginPreparation(job);
            coordinator.ReadyAndWait(job);
            std::this_thread::sleep_for(80ms);
            coordinator.SolverDone(job);
            coordinator.BeginOutput(job);
            coordinator.EndOutput(job);
          },
          [](size_t, const char*) {});
    }
    return 1;
  }
  if (argc == 2 && std::string(argv[1]) == "no-reset") {
    const auto deadline = colmap::mvs::NativeRunDeadline::FromBudget(120ms);
    PairCoordinator coordinator(
        {10, 10}, {1, 1}, size_t{1}, {{1, 1}, {1, 1}}, 500ms, {}, deadline);
    RunRollingOwners(
        coordinator,
        [&](size_t job) {
          coordinator.BeginPreparation(job);
          coordinator.ReadyAndWait(job);
          std::this_thread::sleep_for(80ms);
          coordinator.SolverDone(job);
          coordinator.BeginOutput(job);
          coordinator.EndOutput(job);
        },
        [](size_t, const char*) {});
    return 1;
  }
  {
    using colmap::mvs::ResolveOwnedRunDeadline;
    const auto entry = colmap::mvs::NativeRunDeadline::Clock::now() - 100ms;
    const auto deadline = ResolveOwnedRunDeadline("2", true, true, entry);
    assert(deadline && deadline->end == entry + 2s);
    const auto copy = *deadline;  // Later reference/cohort receives same end.
    assert(copy.end == deadline->end);
    assert(copy.Remaining() < 1950ms);
    assert(!ResolveOwnedRunDeadline("", false, true, entry));
    for (const std::string text : {"",
                                   "0",
                                   "01",
                                   "+2",
                                   "-1",
                                   "1.0",
                                   " 2",
                                   "2 ",
                                   "18446744073709551616"}) {
      bool refused = false;
      try {
        ResolveOwnedRunDeadline(text, true, true, entry);
      } catch (const std::invalid_argument&) {
        refused = true;
      }
      assert(refused);
    }
    bool refused = false;
    try {
      ResolveOwnedRunDeadline("2", false, false, entry);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused);  // Orphan flag must never route ordinary execution.
  }
  {
    const auto deadline = colmap::mvs::NativeRunDeadline::FromBudget(500ms);
    PairCoordinator coordinator(
        {10, 10}, {1, 1}, size_t{1}, {{1, 1}}, 20ms, {}, deadline);
    RunRollingOwners(
        coordinator,
        [&](size_t job) {
          coordinator.BeginPreparation(job);
          coordinator.ReadyAndWait(job);
          std::this_thread::sleep_for(80ms);  // Valid work exceeds legacy wait.
          coordinator.SolverDone(job);
          coordinator.BeginOutput(job);
          coordinator.EndOutput(job);
          coordinator.AwaitOutputs(job);
        },
        [](size_t, const char*) {});
    assert(coordinator.ReservedBytes().host == 0);
  }
  {
    const auto deadline = colmap::mvs::NativeRunDeadline::FromBudget(1ms);
    PairCoordinator coordinator(
        {10, 10}, {1, 1}, size_t{1}, {{1, 1}}, 500ms, {}, deadline);
    coordinator.AdmitNext();
    std::this_thread::sleep_for(5ms);
    bool refused = false;
    try {
      coordinator.BeginPreparation(0);
    } catch (const std::runtime_error&) {
      refused = true;
    }
    assert(refused);  // Ready predicate cannot bypass expired execution gate.
    coordinator.Retire(0);
    coordinator.Joined(0);
    try {
      coordinator.Finish();
      assert(false);
    } catch (const std::runtime_error&) {
    }
  }
  {
    const auto deadline = colmap::mvs::NativeRunDeadline::FromBudget(1ms);
    std::this_thread::sleep_for(5ms);
    PairCoordinator coordinator(
        {10, 10}, {1, 1}, size_t{1}, {{1, 1}}, 500ms, {}, deadline);
    assert(coordinator.AdmitNext() == std::numeric_limits<size_t>::max());
  }
  {
    const auto deadline = colmap::mvs::NativeRunDeadline::FromBudget(500ms);
    PairCoordinator coordinator(
        {10, 10}, {1, 1}, size_t{2}, {{1, 1}, {1, 1}}, 20ms, {}, deadline);
    for (size_t job = 0; job < 2; ++job) {
      coordinator.AdmitNext();
      coordinator.BeginPreparation(job);
      coordinator.ReadyAndWait(job);
    }
    std::thread peer([&] {
      std::this_thread::sleep_for(80ms);
      coordinator.CancelAndAwaitCleanup(1, {}, [] {});
    });
    coordinator.CancelAndAwaitCleanup(0, {}, [] {});
    peer.join();  // Both drains proved before any retirement/credit release.
    assert(coordinator.ReservedBytes().host == 3);
    for (size_t job = 0; job < 2; ++job) {
      coordinator.Retire(job);
      coordinator.Joined(job);
    }
    bool failed = false;
    try {
      coordinator.Finish();
    } catch (const std::runtime_error&) {
      failed = true;
    }
    assert(failed);  // Clean cancellation is still NOT success.
    assert(coordinator.ReservedBytes().host == 0);
  }
  struct DeathCase {
    const char* mode;
    const char* reason;
    const char* budget;
  };
  for (const auto& test :
       {DeathCase{"retirement-expiry", "AwaitRetired", "20 ms"},
        DeathCase{"setup-expiry", "absolute phase deadline expired", "30 ms"},
        DeathCase{"drain-unknown", "local drain threw", "20 ms"},
        DeathCase{"peer-drain-expiry", "active peer drain not proven", "20 ms"},
        DeathCase{"no-reset", "AwaitRetired", "120 ms"},
        DeathCase{"cohort-no-reset", "AwaitRetired", "120 ms"}}) {
    const std::string mode = test.mode;
    const std::string command =
        "ulimit -c 0; \"" + std::string(argv[0]) + "\" " + mode + " 2>&1";
    FILE* pipe = popen(command.c_str(), "r");
    assert(pipe);
    std::string output;
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe)) output += buffer;
    const int status = pclose(pipe);
    assert(status != 0);
    assert(output.find(test.reason) != std::string::npos);
    assert(output.find(test.budget) != std::string::npos);
  }
}
