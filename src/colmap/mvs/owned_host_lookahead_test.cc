// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/owned_host_lookahead.h"
#include "colmap/mvs/owned_pair_run.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {

struct Lifetime {
  std::atomic<int> live{0};
  std::atomic<int> destroyed{0};
  std::atomic<int> peak{0};
};

// Payload is owned, not a view into a builder's scratch storage.
struct Snapshot {
  Snapshot(Lifetime& lifetime, size_t job)
      : lifetime(lifetime), payload(8, static_cast<int>(job + 10)) {
    const int count = ++lifetime.live;
    int peak = lifetime.peak.load();
    while (peak < count && !lifetime.peak.compare_exchange_weak(peak, count)) {}
  }
  ~Snapshot() {
    --lifetime.live;
    ++lifetime.destroyed;
  }
  Snapshot(const Snapshot&) = delete;
  Snapshot& operator=(const Snapshot&) = delete;

  Lifetime& lifetime;
  std::vector<int> payload;
};

using Lookahead = HostLookahead<Snapshot>;
constexpr uint64_t kCharge = 32;

struct Observation {
  std::string event;
  int job;
  uint64_t reserved;
  int live;
  int destroyed;
};

// Start's synchronous notification precedes the worker; Join precedes all
// reads and cleanup notifications. No observer vector access races the worker.
struct Journal {
  explicit Journal(Lifetime& lifetime) : lifetime(lifetime) {}

  void Record(const char* event, int job, uint64_t reserved) {
    entries.push_back({event,
                       job,
                       reserved,
                       lifetime.live.load(),
                       lifetime.destroyed.load()});
  }
  Lookahead::Observer Observer() {
    return [this](const char* event, int job, uint64_t reserved) {
      Record(event, job, reserved);
    };
  }
  void ExpectRelease(int live, int destroyed) const {
    int releases = 0;
    for (const auto& entry : entries) {
      if (entry.event == "release") {
        ++releases;
        EXPECT_EQ(entry.reserved, 0);
        EXPECT_EQ(entry.live, live);
        EXPECT_EQ(entry.destroyed, destroyed);
      } else {
        EXPECT_EQ(entry.reserved, kCharge);
      }
    }
    EXPECT_EQ(releases, 1);
  }

  Lifetime& lifetime;
  std::vector<Observation> entries;
};

TEST(OwnedHostLookahead, UnstartedAndPreExecutionOwnersDropSnapshotsBeforeRetirement) {
  // Reproduce consumption after active admission, then each failure boundary
  // before an actual owner can take its snapshot from the outer handoff array.
  for (const int fail_launch : {0, 1, -1}) {
    SCOPED_TRACE(fail_launch);
    Lifetime lifetime;
    Lookahead future(kCharge, {});
    future.Start([&](size_t slot, const std::atomic<bool>&) {
      return std::make_unique<Snapshot>(lifetime, slot);
    });
    future.Join();
    std::atomic<int> release_count{0};
    PairCoordinator coordinator(PairDispatch::kPair, {100, 100}, {10, 10},
        {{{20, 20}, {20, 20}}}, std::chrono::milliseconds(1000),
        [&](const PairEvent& event) {
          if (event.event == "ledger_release") {
            EXPECT_EQ(lifetime.live.load(), 0);
            ++release_count;
          }
        });
    auto handoff = future.Consume();
    ASSERT_EQ(lifetime.live.load(), 2);
    int launched = 0;
    auto factory = [&](std::function<void()> fn) {
      if (launched++ == fail_launch) throw std::runtime_error("owner launch failed");
      return std::thread(std::move(fn));
    };
    auto run = [&](size_t slot) {
      if (fail_launch == -1 && slot == 0) {
        throw std::runtime_error("owner identity event failed before execute");
      }
      auto owned = std::move(handoff[slot]);
      coordinator.BeginPreparation(slot);
      coordinator.ReadyAndWait(slot);
      coordinator.SolverDone(slot);
      coordinator.BeginOutput(slot);
      coordinator.EndOutput(slot);
      coordinator.AwaitOutputs(slot);
    };
    auto observer = [&](size_t slot, const char* event) {
      if (std::string(event) == "cleanup_complete") EXPECT_EQ(handoff[slot], nullptr);
    };
    EXPECT_THROW(RunPairOwners(coordinator, run, observer, factory,
                             [&](size_t slot) { handoff[slot].reset(); }),
                 std::runtime_error);
    EXPECT_EQ(lifetime.live.load(), 0);
    EXPECT_EQ(lifetime.destroyed.load(), 2);
    EXPECT_EQ(release_count.load(), 1);
    EXPECT_EQ(coordinator.ReservedBytes().host, 0);
  }
}

TEST(OwnedHostLookahead, UnprovenInputCleanupCannotReleaseActiveCredits) {
  PairCoordinator coordinator(PairDispatch::kPair, {100, 100}, {10, 10},
      {{{20, 20}, {20, 20}}}, std::chrono::milliseconds(1000));
  auto no_thread = [](std::function<void()>) -> std::thread {
    throw std::runtime_error("owner launch failed");
  };
  EXPECT_THROW(RunPairOwners(coordinator, [](size_t) {},
      [](size_t, const char*) {}, no_thread,
      [](size_t) { throw std::runtime_error("input cleanup not proven"); }),
      std::runtime_error);
  EXPECT_EQ(coordinator.ReservedBytes().host, 50);
  EXPECT_EQ(coordinator.ReservedBytes().device, 50);
}

TEST(OwnedHostLookahead, ReservationUsesSameBudgetWithoutOverflow) {
  EXPECT_TRUE(CanReserveHostLookahead(100, 20, 25, 25, 30));
  EXPECT_FALSE(CanReserveHostLookahead(99, 20, 25, 25, 30));
  EXPECT_FALSE(CanReserveHostLookahead(100, 20, 25, 25, 0));
  // The active pair fits, but that does not make lookahead eligible. The
  // caller can still use its synchronous fallback with the same budget.
  EXPECT_LE(20 + 25 + 25, 70);
  EXPECT_FALSE(CanReserveHostLookahead(70, 20, 25, 25, 1));
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  EXPECT_TRUE(CanReserveHostLookahead(kMax, kMax - 3, 1, 1, 1));
  EXPECT_FALSE(CanReserveHostLookahead(kMax, kMax, 1, 1, 1));
  EXPECT_FALSE(CanReserveHostLookahead(kMax, 1, kMax, 0, 1));
  EXPECT_FALSE(CanReserveHostLookahead(kMax, 1, 0, kMax, 1));
  EXPECT_FALSE(CanReserveHostLookahead(kMax, 1, 0, 0, kMax));
  EXPECT_FALSE(CanReserveHostLookahead(0, 0, 0, 0, 1));
  EXPECT_FALSE(CanReserveHostLookahead(10, 11, 0, 0, 1));
}

TEST(OwnedHostLookahead, SequentialPairJoinsBeforeConsumerAdmission) {
  Lifetime lifetime;
  Journal journal(lifetime);
  int consumer_admissions = 0;
  std::promise<void> a_entered;
  std::promise<void> allow_a;
  auto a_gate = allow_a.get_future();
  std::vector<size_t> jobs;
  std::vector<std::thread::id> workers;
  Lookahead::Pair pair;
  {
    Lookahead lookahead(kCharge, journal.Observer());
    lookahead.Start([&](size_t job, const std::atomic<bool>&) {
      jobs.push_back(job);
      workers.push_back(std::this_thread::get_id());
      auto snapshot = std::make_unique<Snapshot>(lifetime, job);
      if (job == 0) {
        a_entered.set_value();
        a_gate.wait();
      }
      return snapshot;
    });
    a_entered.get_future().wait();
    EXPECT_EQ(lifetime.live.load(), 1);
    EXPECT_THROW(lookahead.Consume(), std::logic_error);
    EXPECT_THROW(lookahead.Start([](size_t, const std::atomic<bool>&) {
                   return std::unique_ptr<Snapshot>();
                 }),
                 std::logic_error);
    EXPECT_EQ(consumer_admissions, 0);
    allow_a.set_value();
    lookahead.Join();
    EXPECT_EQ(consumer_admissions, 0);
    EXPECT_EQ(jobs, (std::vector<size_t>{0, 1}));
    ASSERT_EQ(workers.size(), 2);
    EXPECT_EQ(workers[0], workers[1]);
    EXPECT_NE(workers[0], std::this_thread::get_id());
    EXPECT_EQ(lifetime.live.load(), 2);
    EXPECT_EQ(lifetime.peak.load(), 2);
    // This is deliberately a consumer-only admission counter, not a GPU mock.
    ++consumer_admissions;
    pair = lookahead.Consume();
    ASSERT_NE(pair[0], nullptr);
    ASSERT_NE(pair[1], nullptr);
    EXPECT_EQ(pair[0]->payload, std::vector<int>(8, 10));
    EXPECT_EQ(pair[1]->payload, std::vector<int>(8, 11));
    journal.ExpectRelease(2, 0);
    EXPECT_THROW(lookahead.Consume(), std::logic_error);
    EXPECT_EQ(consumer_admissions, 1);
  }
  EXPECT_EQ(lifetime.live.load(), 2);
  EXPECT_EQ(pair[0]->payload, std::vector<int>(8, 10));
  EXPECT_EQ(pair[1]->payload, std::vector<int>(8, 11));
  pair = {};
  EXPECT_EQ(lifetime.live.load(), 0);
  EXPECT_EQ(lifetime.destroyed.load(), 2);
  journal.ExpectRelease(2, 0);
}

TEST(OwnedHostLookahead, CompletedWorkerStillRequiresExplicitJoin) {
  Lifetime lifetime;
  std::promise<void> ready;
  Lookahead lookahead(kCharge, [&](const char* event, int, uint64_t) {
    if (std::string(event) == "prep_done") ready.set_value();
  });
  lookahead.Start([&](size_t job, const std::atomic<bool>&) {
    return std::make_unique<Snapshot>(lifetime, job);
  });
  ready.get_future().wait();
  EXPECT_THROW(lookahead.Consume(), std::logic_error);
  lookahead.Join();
  auto pair = lookahead.Consume();
  EXPECT_EQ(lifetime.live.load(), 2);
}

TEST(OwnedHostLookahead, PreparationErrorsStopJobsAndCleanBeforeRelease) {
  for (const size_t failed_job : {0, 1}) {
    SCOPED_TRACE(failed_job);
    Lifetime lifetime;
    Journal journal(lifetime);
    std::vector<size_t> jobs;
    Lookahead lookahead(kCharge, journal.Observer());
    lookahead.Start([&](size_t job, const std::atomic<bool>&) {
      jobs.push_back(job);
      auto snapshot = std::make_unique<Snapshot>(lifetime, job);
      if (job == failed_job) {
        throw std::runtime_error(job == 0 ? "prep A failed" : "prep B failed");
      }
      return snapshot;
    });
    try {
      lookahead.Join();
      ADD_FAILURE() << "Preparation error did not propagate";
    } catch (const std::runtime_error& error) {
      EXPECT_EQ(std::string(error.what()),
                failed_job == 0 ? "prep A failed" : "prep B failed");
    }
    EXPECT_EQ(jobs, failed_job == 0 ? std::vector<size_t>{0}
                                  : (std::vector<size_t>{0, 1}));
    EXPECT_EQ(lifetime.live.load(), 0);
    EXPECT_EQ(lifetime.destroyed.load(), static_cast<int>(failed_job + 1));
    journal.ExpectRelease(0, static_cast<int>(failed_job + 1));
    EXPECT_THROW(lookahead.Consume(), std::logic_error);
    lookahead.CancelAndJoin();
    journal.ExpectRelease(0, static_cast<int>(failed_job + 1));
  }
}

TEST(OwnedHostLookahead, ThreadFactoryFailureReleasesWithoutStartingBuilder) {
  Lifetime lifetime;
  Journal journal(lifetime);
  int builder_calls = 0;
  Lookahead lookahead(kCharge, journal.Observer());
  EXPECT_THROW(
      lookahead.Start(
          [&](size_t job, const std::atomic<bool>&) {
            ++builder_calls;
            return std::make_unique<Snapshot>(lifetime, job);
          },
          [](std::function<void()>) -> std::thread {
            throw std::runtime_error("thread creation failed");
          }),
      std::runtime_error);
  EXPECT_EQ(builder_calls, 0);
  EXPECT_EQ(lifetime.live.load(), 0);
  EXPECT_EQ(lifetime.destroyed.load(), 0);
  journal.ExpectRelease(0, 0);
  EXPECT_THROW(lookahead.Consume(), std::logic_error);
}

TEST(OwnedHostLookahead, CancellationDrainsRunningBuilderBeforeRelease) {
  Lifetime lifetime;
  Journal journal(lifetime);
  std::promise<void> entered;
  std::atomic<int> builder_calls{0};
  std::atomic<bool> saw_cancel{false};
  Lookahead lookahead(kCharge, journal.Observer());
  lookahead.Start([&](size_t job, const std::atomic<bool>& cancelled) {
    ++builder_calls;
    auto snapshot = std::make_unique<Snapshot>(lifetime, job);
    entered.set_value();
    while (!cancelled.load(std::memory_order_relaxed)) {
      std::this_thread::yield();
    }
    saw_cancel.store(true);
    return snapshot;
  });
  entered.get_future().wait();
  EXPECT_EQ(lifetime.live.load(), 1);
  lookahead.CancelAndJoin();
  EXPECT_TRUE(saw_cancel.load());
  EXPECT_EQ(builder_calls.load(), 1);
  EXPECT_EQ(lifetime.live.load(), 0);
  EXPECT_EQ(lifetime.destroyed.load(), 1);
  journal.ExpectRelease(0, 1);
  EXPECT_THROW(lookahead.Join(), std::runtime_error);
  lookahead.CancelAndJoin();
  journal.ExpectRelease(0, 1);
}

TEST(OwnedHostLookahead, OutputFailureUnwindsAndDrainsActivePreparation) {
  Lifetime lifetime;
  Journal journal(lifetime);
  std::promise<void> entered;
  std::atomic<bool> drained{false};
  try {
    Lookahead lookahead(kCharge, journal.Observer());
    lookahead.Start([&](size_t job, const std::atomic<bool>& cancelled) {
      auto snapshot = std::make_unique<Snapshot>(lifetime, job);
      entered.set_value();
      while (!cancelled.load(std::memory_order_relaxed)) {
        std::this_thread::yield();
      }
      drained.store(true);
      return snapshot;
    });
    entered.get_future().wait();
    // Simulate an active epoch's output failure, not a preparation failure.
    throw std::runtime_error("active epoch output failed");
  } catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()), "active epoch output failed");
  }
  EXPECT_TRUE(drained.load());
  EXPECT_EQ(lifetime.live.load(), 0);
  EXPECT_EQ(lifetime.destroyed.load(), 1);
  journal.ExpectRelease(0, 1);
}

TEST(OwnedHostLookahead, ObserverExceptionsDoNotLeakOrEscapeDestructor) {
  // Cover both synchronous and worker notifications, including throwing from
  // failure reporting itself and from release during scope-unwind cleanup.
  for (const std::string throwing_event : {"reserved",
                                           "job_done",
                                           "prep_done",
                                           "consume",
                                           "release"}) {
    SCOPED_TRACE(throwing_event);
    Lifetime lifetime;
    Journal journal(lifetime);
    {
      Lookahead lookahead(
          kCharge, [&](const char* event, int job, uint64_t reserved) {
            journal.Record(event, job, reserved);
            if (event == throwing_event || std::string(event) == "failed") {
              throw std::runtime_error("observer failed");
            }
          });
      auto prepare = [&](size_t job, const std::atomic<bool>&) {
        return std::make_unique<Snapshot>(lifetime, job);
      };
      if (throwing_event == "reserved") {
        EXPECT_THROW(lookahead.Start(prepare), std::runtime_error);
      } else {
        lookahead.Start(prepare);
        if (throwing_event == "job_done" || throwing_event == "prep_done") {
          EXPECT_THROW(lookahead.Join(), std::runtime_error);
        } else {
          lookahead.Join();
          EXPECT_THROW(lookahead.Consume(), std::runtime_error);
        }
      }
    }
    EXPECT_EQ(lifetime.live.load(), 0);
    const int expected = throwing_event == "reserved" ? 0
                         : throwing_event == "job_done" ? 1
                                                        : 2;
    EXPECT_EQ(lifetime.destroyed.load(), expected);
    // Consume transfers before release, so a throwing release sees owners
    // that are subsequently destroyed by Consume's exceptional exit.
    journal.ExpectRelease(throwing_event == "release" ? 2 : 0,
                          throwing_event == "release" ? 0 : expected);
  }
}

TEST(OwnedHostLookahead, ThrowingReleaseObserverIsSuppressedDuringDestruction) {
  Lifetime lifetime;
  Journal journal(lifetime);
  {
    Lookahead lookahead(
        kCharge, [&](const char* event, int job, uint64_t reserved) {
          journal.Record(event, job, reserved);
          if (std::string(event) == "release") {
            throw std::runtime_error("release observer failed");
          }
        });
    lookahead.Start([&](size_t job, const std::atomic<bool>&) {
      return std::make_unique<Snapshot>(lifetime, job);
    });
    lookahead.Join();
  }
  EXPECT_EQ(lifetime.live.load(), 0);
  journal.ExpectRelease(0, 2);
}

}  // namespace
}  // namespace colmap::mvs::owned
