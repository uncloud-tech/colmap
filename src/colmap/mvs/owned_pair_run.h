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

// RunJob's scope must own ALL job resources; returning or throwing means those
// objects were destroyed. CUDA cleanup uncertainty must not return normally.
// Observer is telemetry, never the owner of a backend/input/cache allocation.
template <class RunJob, class Observer, class ThreadFactory = PairThreadFactory>
void RunPairOwners(PairCoordinator& coordinator,
                   RunJob run_job,
                   Observer observer,
                   ThreadFactory create_thread = {}) {
  std::array<std::thread, 2> owners;
  size_t started = 0;
  auto retire = [&](size_t job) {
    try {
      observer(job, "cleanup_complete");
    } catch (...) {
      coordinator.Cancel(job, std::current_exception());
    }
    try {
      coordinator.Retire(job, true);
    } catch (...) {
      coordinator.Cancel(job, std::current_exception());
    }
  };
  try {
    for (size_t job = 0; job < 2; ++job) {
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
    for (size_t job = started; job < 2; ++job) retire(job);
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
