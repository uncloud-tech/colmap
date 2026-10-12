// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_pair_coordinator.h"

#include <array>
#include <exception>
#include <functional>
#include <thread>
#include <utility>

namespace colmap::mvs::owned {

struct PairThreadFactory {
  std::thread operator()(std::function<void()> task) const {
    return std::thread(std::move(task));
  }
};

struct PairOwnerNoopCleanup {
  void operator()(size_t) const noexcept {}
};

// RunJob's scope must own all resources it consumes. Cleanup also destroys
// unconsumed inputs when an owner never starts or fails before RunJob consumes
// them. Both must finish before retirement. CUDA cleanup uncertainty must not
// return normally. Observer is telemetry, never an ownership-cleanup callback.
template <class RunJob, class Observer, class ThreadFactory = PairThreadFactory,
          class Cleanup = PairOwnerNoopCleanup>
void RunPairOwners(PairCoordinator& coordinator,
                   RunJob run_job,
                   Observer observer,
                   ThreadFactory create_thread = {},
                   Cleanup cleanup = {}) {
  std::array<std::thread, 3> owners;
  size_t started = 0;
  auto retire = [&](size_t job) {
    bool cleanup_proven = true;
    try {
      cleanup(job);
    } catch (...) {
      cleanup_proven = false;
      coordinator.Cancel(job, std::current_exception());
    }
    try {
      if (cleanup_proven) observer(job, "cleanup_complete");
    } catch (...) {
      coordinator.Cancel(job, std::current_exception());
    }
    try {
      coordinator.Retire(job, cleanup_proven);
    } catch (...) {
      coordinator.Cancel(job, std::current_exception());
    }
  };
  try {
    for (size_t job = 0; job < coordinator.JobCount(); ++job) {
      owners[job] = create_thread([&, job] {
        try {
          run_job(job);
        } catch (...) {
          coordinator.Cancel(job, std::current_exception());
        }
        retire(job);
      });
      ++started;
    }
  } catch (...) {
    coordinator.Cancel(std::numeric_limits<size_t>::max(),
                       std::current_exception());
    // Slots whose thread never existed own no resources. Mark them after
    // cancellation so a started owner waiting for its peer can drain and exit.
    for (size_t job = started; job < coordinator.JobCount(); ++job) retire(job);
  }
  for (auto& owner : owners) {
    if (owner.joinable()) owner.join();
  }
  try {
    observer(std::numeric_limits<size_t>::max(), "owners_joined");
  } catch (...) {
    coordinator.Cancel(std::numeric_limits<size_t>::max(),
                       std::current_exception());
  }
  coordinator.Finish();
}

}  // namespace colmap::mvs::owned
