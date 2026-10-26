// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_graph_coordinator.h"

#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_selection_boundary.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <vector>

int main() {
  using namespace colmap::mvs::owned;
  // Real rolling admission/retirement + the production host selection lease.
  // A constrained forecast admits only one ready owner despite N=4; graph
  // orchestration must not invent credits or wait for unavailable peers. The
  // fifth generation is a tail and reuses a joined slot.
  for (size_t n = 1; n <= 4; ++n) {
    for (bool constrained : {false, true}) {
      const size_t jobs = 5;
      const uint64_t slots = constrained ? 1 : n;
      OwnedConcurrencyObserver observation(n, n);
      PairCoordinator admission(
          {1 + slots * 10, 1 + slots * 10},
          {1, 1},
          n,
          std::vector<ByteCharge>(jobs, {10, 10}),
          std::chrono::seconds(3),
          [&](const PairEvent& event) { observation(event); });
      OwnedSharedController shared(n, kMaxSelectionSlots, jobs);
      std::atomic<int> preparations{0}, completed{0};
      OwnerGraphLaunchState sweeps(n);
      RunRollingOwners(
          admission,
          [&](size_t generation) {
            const auto owner = admission.Slot(generation);
            shared.Begin(owner);
            admission.BeginPreparation(generation);
            shared.WithSelection(owner, [&](auto& selection) {
              assert(++preparations == 1);
              bool hit = true;
              assert(selection.Select(static_cast<int>(generation), &hit) == 0);
              assert(!hit);
              --preparations;
            });
            admission.ReadyAndWait(generation);
            sweeps.Check(owner);
            ++completed;
            admission.SolverDone(generation);
            admission.BeginOutput(generation);
            admission.EndOutput(generation);
            admission.AwaitOutputs(generation);
            shared.End(owner);
          },
          [](size_t, const char*) {});
      shared.Seal();
      assert(completed == jobs);
      assert(shared.SerializedSelections() == jobs);
      assert(observation.Observed() <= slots);
    }
  }

  OwnerGraphLaunchState sweeps(2);
  sweeps.Check(0);
  sweeps.Check(0);  // Repeated queued submissions never rendezvous.
  sweeps.Check(1);
  bool refused = false;
  try {
    sweeps.Check(2);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  assert(refused);
  sweeps.Cancel(std::make_exception_ptr(std::runtime_error("cancel")));
  for (size_t owner : {0, 1}) {
    refused = false;
    try {
      sweeps.Check(owner);
    } catch (const std::runtime_error&) {
      refused = true;
    }
    assert(refused);
  }
  OwnerGraphLaunchState next(2);
  next.Check(0);
  next.Check(1);
}
