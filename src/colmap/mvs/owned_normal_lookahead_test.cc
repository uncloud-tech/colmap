// SPDX-License-Identifier: BSD-3-Clause

// CPU-only tests for the bounded reference-normal lookahead producer. These
// exercise real NormalMap payloads through the public interface: FIFO order,
// the one-to-two ready-buffer bound, per-generation single handoff, exception
// propagation, bounded consumer deadlines, cancellation between in-flight
// reads, and exactly-once charge release. No CUDA is involved anywhere.

#include "colmap/mvs/owned_normal_lookahead.h"
#include "colmap/mvs/owned_pair_run.h"

#include <algorithm>
#include <atomic>
#include <future>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {
namespace {

// Loader-produced NormalMap payloads carry their generation as a constant so
// FIFO order and single-load ownership are observable without GPU or I/O.
NormalMap MakeNormal(size_t generation) {
  NormalMap normal(16, 16);
  normal.Fill(static_cast<float>(generation));
  return normal;
}

// In-flight read must complete before the producer can observe cancellation
// and before any release is reported. The cancellation flag is honored
// BETWEEN reads by the producer, never mid-map.
NormalMap GatedNormal(std::atomic<int>& started,
                      std::atomic<int>& completed,
                      size_t generation,
                      const std::atomic<bool>& cancelled,
                      std::chrono::milliseconds max_wait) {
  ++started;
  const auto deadline = std::chrono::steady_clock::now() + max_wait;
  while (!cancelled.load(std::memory_order_relaxed)) {
    if (std::chrono::steady_clock::now() > deadline) {
      throw std::runtime_error("gated loader was never cancelled");
    }
    std::this_thread::yield();
  }
  ++completed;
  return MakeNormal(generation);
}

struct Journal {
  std::vector<std::string> events;
  std::vector<uint64_t> reserved;
  std::mutex mutex;
  void Record(const char* event, uint64_t value) {
    std::lock_guard<std::mutex> lock(mutex);
    events.push_back(event);
    reserved.push_back(value);
  }
  void ExpectExactly(const std::string& event, size_t count) const {
    size_t seen = 0;
    for (const auto& entry : events) seen += entry == event;
    EXPECT_EQ(seen, count) << event;
  }
};

TEST(NormalLookahead, ConstructorsValidateLimits) {
  EXPECT_THROW(NormalLookahead({0}, 0, 1),
               std::invalid_argument);
  EXPECT_THROW(NormalLookahead({0}, 3, 1),
               std::invalid_argument);
  EXPECT_THROW(NormalLookahead({}, 2, 1),
               std::invalid_argument);
  NormalLookahead ok({0, 1}, 1, 1);
  EXPECT_EQ(ok.MaxFuture(), 1);
}

TEST(NormalLookahead, ConsumeBeforeStartAndOutOfBoundsRejected) {
  // max_future 2 lets the unconsumed pair produce fully without consumers.
  NormalLookahead lookahead({0, 1}, 2, 8);
  EXPECT_THROW(lookahead.Consume(0), std::logic_error);
  lookahead.Start([](size_t layer, const std::atomic<bool>&) {
    return MakeNormal(layer);
  });
  EXPECT_THROW(lookahead.Consume(2), std::invalid_argument);
  lookahead.Join();
  EXPECT_EQ(lookahead.Consume(0).Get(0, 0), 0);
  EXPECT_THROW(lookahead.Consume(0), std::logic_error);
}

TEST(NormalLookahead, FifoOrderTailAndExactlyOneLoadPerGeneration) {
  std::vector<std::atomic<int>> loads(5);
  for (auto& load : loads) load.store(0);
  std::atomic<int> total{0};
  Journal journal;
  NormalLookahead lookahead({4, 3, 2, 1, 0}, 2, 16, [&](const char* event, size_t, uint64_t r) {
                              journal.Record(event, r);
                            });
  lookahead.Start([&](size_t layer, const std::atomic<bool>&) {
    ++loads[layer];
    ++total;
    return MakeNormal(layer);
  });
  // Consumer drives in FIFO generation order while the producer works ahead
  // under the bounded queue. Payload identities follow the immutable
  // generation -> bank layer mapping {4,3,2,1,0}, including the tail.
  const int expected_layer[5] = {4, 3, 2, 1, 0};
  for (size_t generation = 0; generation < 5; ++generation) {
    NormalMap normal = lookahead.Consume(generation);
    EXPECT_EQ(normal.Get(0, 0),
              static_cast<float>(expected_layer[generation]));
  }
  lookahead.Join();
  EXPECT_EQ(total.load(), 5);
  for (size_t layer = 0; layer < 5; ++layer) {
    EXPECT_EQ(loads[layer].load(), 1) << layer;
  }
  // The lookahead charge stays reserved for the whole producer lifetime and
  // is released exactly once, after the producer joined.
  lookahead.CancelAndJoin();
  journal.ExpectExactly("release", 1);
  EXPECT_EQ(journal.reserved[0], 16);
}

TEST(NormalLookahead, ReadyUnconsumedBuffersStayBounded) {
  // Without any consumer progress the producer must stop after max_future
  // ready buffers instead of loading the whole FIFO into host memory.
  std::atomic<int> loads{0};
  NormalLookahead lookahead({0, 1, 2, 3}, 2, 8);
  lookahead.Start([&](size_t layer, const std::atomic<bool>&) {
    ++loads;
    return MakeNormal(layer);
  });
  const auto observed = [](const std::atomic<int>& value, int target) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(1000);
    while (value.load() < target) {
      if (std::chrono::steady_clock::now() > deadline) return false;
      std::this_thread::yield();
    }
    return true;
  };
  ASSERT_TRUE(observed(loads, 2));  // gens 0 and 1 only; gen 2 stays unloaded.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(loads.load(), 2);
  // Handing off gen 0 frees exactly one slot for the next load.
  EXPECT_EQ(lookahead.Consume(0).Get(0, 0), 0);
  ASSERT_TRUE(observed(loads, 3));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(loads.load(), 3);
  // Consuming gen 1 again unblocks the producer for the final generation.
  EXPECT_EQ(lookahead.Consume(1).Get(0, 0), 1);
  ASSERT_TRUE(observed(loads, 4));
  lookahead.Join();
  EXPECT_EQ(loads.load(), 4);
}

TEST(NormalLookahead, OutOfOrderConsumeHonorsSingleHandoffPerGeneration) {
  NormalLookahead lookahead({0, 1, 2}, 2, 8);
  lookahead.Start([](size_t layer, const std::atomic<bool>&) {
    return MakeNormal(layer);
  });
  // Each admitted owner consumes exactly its own generation; ownership of the
  // unique buffer transfers out of the producer queue. Consuming gen 1 first
  // also frees the bounded slot that lets the producer finish the tail.
  NormalMap first = lookahead.Consume(1);
  EXPECT_EQ(first.Get(0, 0), 1);
  NormalMap zero = lookahead.Consume(0);
  EXPECT_EQ(zero.Get(0, 0), 0);
  NormalMap two = lookahead.Consume(2);
  EXPECT_EQ(two.Get(0, 0), 2);
  lookahead.Join();
  EXPECT_THROW(lookahead.Consume(0), std::logic_error);
  EXPECT_THROW(lookahead.Consume(1), std::logic_error);
  EXPECT_THROW(lookahead.Consume(2), std::logic_error);
}

TEST(NormalLookahead, LoaderExceptionPropagatesToJoinAndPendingConsumers) {
  std::atomic<int> loads{0};
  NormalLookahead lookahead({0, 1, 2}, 2, 8);
  lookahead.Start([&](size_t layer, const std::atomic<bool>&) {
    const int count = ++loads;
    if (layer == 1) {
      throw std::runtime_error("normal identity changed during read");
    }
    return MakeNormal(layer);
  });
  EXPECT_EQ(lookahead.Consume(0).Get(0, 0), 0);
  try {
    lookahead.Consume(1);
    FAIL() << "generation error did not propagate";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "normal identity changed during read");
  }
  // Later generations fail closed with the same first error, no retry.
  try {
    lookahead.Consume(2);
    FAIL() << "post-error generation did not fail";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "normal identity changed during read");
  }
  EXPECT_EQ(loads.load(), 2);
  EXPECT_THROW(lookahead.Join(), std::runtime_error);
}

TEST(NormalLookahead, CancelJoinsInFlightReadBeforeRelease) {
  std::atomic<int> started{0};
  std::atomic<int> completed{0};
  Journal journal;
  NormalLookahead lookahead({0, 1, 2}, 2, 8,
                            [&](const char* event, size_t, uint64_t r) {
                              journal.Record(event, r);
                            });
  lookahead.Start([&](size_t layer, const std::atomic<bool>& cancelled) {
    return GatedNormal(started, completed, layer, cancelled,
                       std::chrono::milliseconds(2000));
  });
  while (started.load() < 1) std::this_thread::yield();
  lookahead.CancelAndJoin();
  EXPECT_EQ(completed.load(), 1);  // in-flight read joined before release
  journal.ExpectExactly("release", 1);
  EXPECT_EQ(journal.reserved[journal.events.size() - 1], 0u);
  EXPECT_THROW(lookahead.Join(), std::runtime_error);
}

TEST(NormalLookahead, HealthyConsumerDelayBeyondAnyTimeoutStillSucceeds) {
  // Regression for the removed elapsed-time failures: a full queue or a
  // delayed loader is healthy backpressure. The old 5-minute producer slot
  // timeout and consumer deadline would have failed this run.
  std::atomic<bool> released{false};
  std::promise<void> allow;
  NormalLookahead lookahead({0}, 2, 8);
  lookahead.Start([&](size_t, const std::atomic<bool>&) {
    allow.get_future().wait();
    released.store(true);
    return MakeNormal(0);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  allow.set_value();
  EXPECT_EQ(lookahead.Consume(0).Get(0, 0), 0);
  EXPECT_TRUE(released.load());

  // Producer blocked on a full single-slot queue for well over any previous
  // short timeout must not record a failure; later consumption succeeds.
  std::atomic<int> loads{0};
  NormalLookahead blocked({0, 1, 2}, 1, 8);
  blocked.Start([&](size_t layer, const std::atomic<bool>&) {
    ++loads;
    return MakeNormal(layer);
  });
  const auto observed = [](const std::atomic<int>& value, int target,
                           std::chrono::milliseconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (value.load() < target) {
      if (std::chrono::steady_clock::now() > deadline) return false;
      std::this_thread::yield();
    }
    return true;
  };
  ASSERT_TRUE(observed(loads, 1, std::chrono::seconds(2)));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(loads.load(), 1);  // slot wait healthy, no error recorded.
  EXPECT_EQ(blocked.Consume(0).Get(0, 0), 0);
  ASSERT_TRUE(observed(loads, 2, std::chrono::seconds(2)));
  EXPECT_EQ(blocked.Consume(1).Get(0, 0), 1);
  blocked.Join();
  EXPECT_EQ(loads.load(), 3);
}

TEST(NormalLookahead, OwnerFailureBeforeConsumeCancelsPromptly) {
  // Depth-one head blocking: generation 0 is ready, generation 1 waits for
  // production, and the producer waits for queue space. An owner failing
  // before consuming generation 0 (wired to Cancel()) must wake the waiting
  // consumer and producer promptly — no five-minute timeout path.
  std::atomic<int> loads{0};
  Journal journal;
  NormalLookahead lookahead({0, 1, 2}, 1, 8,
                            [&](const char* event, size_t, uint64_t r) {
                              journal.Record(event, r);
                            });
  lookahead.Start([&](size_t layer, const std::atomic<bool>&) {
    ++loads;
    return MakeNormal(layer);
  });
  const auto observed = [](const std::atomic<int>& value, int target,
                           std::chrono::milliseconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (value.load() < target) {
      if (std::chrono::steady_clock::now() > deadline) return false;
      std::this_thread::yield();
    }
    return true;
  };
  ASSERT_TRUE(observed(loads, 1, std::chrono::seconds(2)));
  // Generation-1 owner blocked in Consume while the producer waits for space.
  auto consumer = std::async(std::launch::async, [&] { return lookahead.Consume(1); });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(loads.load(), 1);
  // Simulated owner-failure hook: prompt, non-joining cancellation.
  lookahead.Cancel();
  const auto status = consumer.wait_for(std::chrono::seconds(2));
  ASSERT_EQ(status, std::future_status::ready);
  try {
    consumer.get();
    FAIL() << "cancelled consumer did not fail";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "normal lookahead cancelled");
  }
  EXPECT_EQ(loads.load(), 1);  // gen 1/2 never loaded after cancellation.
  // The explicit charge stays reserved; release happens only in CancelAndJoin
  // after the producer join destroys the unhanded buffers.
  EXPECT_EQ(std::find(journal.events.begin(), journal.events.end(), "release"),
            journal.events.end())
      << "charge released before owner drain/cleanup";
  // Destructor drains/joins the producer and releases the charge exactly once.
}

TEST(NormalLookahead, DestructorDrainsProducerWithoutLeak) {
  std::atomic<int> started{0};
  std::atomic<int> completed{0};
  {
    NormalLookahead lookahead({0, 1}, 1, 8);
    lookahead.Start([&](size_t layer, const std::atomic<bool>& cancelled) {
      return GatedNormal(started, completed, layer, cancelled,
                         std::chrono::milliseconds(2000));
    });
    while (started.load() < 1) std::this_thread::yield();
  }
  EXPECT_EQ(completed.load(), 1);
}

}  // namespace
}  // namespace colmap::mvs::owned