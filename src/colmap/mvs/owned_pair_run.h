// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_pair_coordinator.h"

#include <array>
#include <exception>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

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
template <class RunJob,
          class Observer,
          class ThreadFactory = PairThreadFactory,
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

// Bounded FIFO postjoin turnover. Job indices are immutable generation tokens;
// no callback ever resolves a mutable slot to discover its job identity.
template <class RunJob,
          class Observer,
          class ThreadFactory = PairThreadFactory,
          class Cleanup = PairOwnerNoopCleanup>
void RunRollingOwners(PairCoordinator& coordinator,
                      RunJob run_job,
                      Observer observer,
                      ThreadFactory create_thread = {},
                      Cleanup cleanup = {}) {
  if (!coordinator.Rolling()) throw std::logic_error("not a rolling feed");
  const size_t batch = std::numeric_limits<size_t>::max();
  std::vector<std::thread> owners(coordinator.JobCount());
  size_t joined = 0;
  auto retire = [&](size_t job) {
    bool proven = true;
    try {
      cleanup(job);
    } catch (...) {
      proven = false;
      coordinator.Cancel(job, std::current_exception());
    }
    try {
      if (proven) observer(job, "cleanup_complete");
    } catch (...) {
      coordinator.Cancel(job, std::current_exception());
    }
    try {
      coordinator.Retire(job, proven);
    } catch (...) {
      coordinator.Cancel(job, std::current_exception());
    }
  };
  while (true) {
    try {
      while (true) {
        const size_t job = coordinator.AdmitNext();
        if (job == batch) break;
        try {
          owners[job] = create_thread([&, job] {
            try {
              run_job(job);
            } catch (...) {
              coordinator.Cancel(job, std::current_exception());
            }
            retire(job);
          });
          if (!owners[job].joinable())
            throw std::runtime_error("thread factory returned no owner");
        } catch (...) {
          coordinator.Cancel(job, std::current_exception());
          retire(job);  // Admitted input whose thread never existed.
          break;
        }
      }
    } catch (...) {
      // Admission telemetry can throw after atomic assignment but before
      // return.
      coordinator.Cancel(batch, std::current_exception());
      const size_t count = coordinator.AdmittedCount();
      if (count) retire(count - 1);
    }
    if (joined == coordinator.AdmittedCount()) break;
    const size_t job = coordinator.AwaitRetired();
    if (owners[job].joinable()) owners[job].join();
    // std::thread's invocation state has now been destroyed. Clear the parent
    // thread handle too before publishing joined eligibility for the next head.
    owners[job] = std::thread{};
    try {
      coordinator.Joined(job);
    } catch (...) {
      coordinator.Cancel(batch, std::current_exception());
    }
    ++joined;
  }
  try {
    observer(batch, "owners_joined");
  } catch (...) {
    coordinator.Cancel(batch, std::current_exception());
  }
  coordinator.Finish();
}

}  // namespace colmap::mvs::owned
