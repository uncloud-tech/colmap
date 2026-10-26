// SPDX-License-Identifier: BSD-3-Clause
// CPU lifecycle regression for the resident runner's mandatory failure-sink
// ordering (Sol 3cdc fix 5): a THROWING cancellation observer must never
// disable the producer cancellation, and the owner loop must drain and join
// (no stranded owner) on the real coordinator/NormalLookahead primitives.
//
// This regression runs the REAL classes (PairCoordinator, NormalLookahead,
// RunRollingOwners) and the production sink ordering helper
// (ResidentEventSink). The CUDA bank surfaces themselves remain dev1
// native-build coverage.
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/owned_normal_lookahead.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/resident_forecast.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

TEST(ResidentRunnerSinkOrder, CancelBeforeObserver) {
  // The production ordering helper: cancellation is published FIRST.
  std::string order;
  bool cancelled = false;
  const std::function<void()> do_cancel = [&] {
    cancelled = true;
    order.push_back('c');
  };
  const std::function<void(size_t, const char*)> observer = [&](size_t,
                                                                const char*) {
    // A throwing telemetry observer must not strand anyone; the ordering
    // assertion lives in the recorded sequence checked below.
    order.push_back('o');
    throw std::runtime_error("observer telemetry failure");
  };
  EXPECT_THROW(ResidentEventSink(/*cancelled_event=*/true,
                                 do_cancel,
                                 &observer,
                                 /*job=*/0,
                                 "cancelled"),
               std::exception);
  EXPECT_EQ(order, "co");
  // Non-cancelled events: observer runs, no cancellation (the throwing
  // observer still throws; the ordering is what this test pins).
  cancelled = false;
  order.clear();
  EXPECT_THROW(
      ResidentEventSink(
          /*cancelled_event=*/false, do_cancel, &observer, 0, "admitted"),
      std::exception);
  EXPECT_EQ(order, "o");
  EXPECT_FALSE(cancelled);
  // Observer-free sinks still cancel.
  cancelled = false;
  ResidentEventSink(
      /*cancelled_event=*/true, do_cancel, nullptr, 0, "cancelled");
  EXPECT_TRUE(cancelled);
}

// Real lifecycle: a blocked consumer must unwind and the owner loop must
// drain/join even when the cancellation observer throws.
TEST(ResidentRunnerLifecycle, ThrowingCancellationObserverStillDrains) {
  // One owner slot. Coordinator timeouts do not bound Consume waits;
  // the test runner must enforce an external timeout for regressions.
  const std::function<void(size_t, const char*)> throwing_observer =
      [](size_t, const char* event) {
        if (std::string(event) == "cancelled") {
          throw std::runtime_error("telemetry cancelled event failed");
        }
      };
  auto pipeline =
      std::make_unique<owned::NormalLookahead>(std::vector<size_t>{0, 1},
                                               /*max_future=*/1,
                                               /*charge=*/4096);
  auto pipeline_raw = pipeline.get();
  const auto sink = [pipeline_raw,
                     &throwing_observer](const owned::PairEvent& e) {
    // IDENTICAL composition to the production runner sink (Sol 3cdc fix 5).
    ResidentEventSink(
        e.event == "cancelled",
        [pipeline_raw]() { pipeline_raw->Cancel(); },
        &throwing_observer,
        e.job,
        e.event.c_str());
  };
  owned::PairCoordinator coordinator(
      owned::ByteCharge{65536, 65536},
      owned::ByteCharge{16384, 16384},
      /*max_concurrent_solvers=*/1,
      std::vector<owned::ByteCharge>{{4096, 4096}, {4096, 4096}},
      /*timeout=*/std::chrono::milliseconds(2000),
      sink);
  // The loader blocks until cancellation is observed, then fails (no
  // payload production is needed for this regression).
  bool producer_started = false;
  pipeline->Start([&](size_t layer, const std::atomic<bool>& cancelled) {
    producer_started = true;
    NormalMap normal(2, 2);
    while (!cancelled.load(std::memory_order_relaxed)) {
      std::this_thread::yield();
    }
    throw std::runtime_error("loader observed cancellation");
    return normal;
  });
  // The owner consumes generation 0 and blocks in Consume until the
  // cancellation unblocks it.
  const auto run_job = [&](const size_t generation) {
    NormalMap normal = pipeline->Consume(generation);
    normal.Fill(0.0f);
  };
  // Trigger the failure from outside: cancellation published through the
  // sink with the THROWING observer on the cancelled event.
  std::thread trigger([&] {
    // Bounded settling: cancellation is sticky, so a late trigger still
    // cancels and still unblocks the consumer; no hang either way.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    coordinator.Cancel(
        0,
        std::make_exception_ptr(std::runtime_error("triggered cancellation")));
  });
  // RunRollingOwners must drain and join despite the throwing observer,
  // and the preserved first error propagates from Finish.
  EXPECT_THROW(
      owned::RunRollingOwners(coordinator, run_job, [](size_t, const char*) {}),
      std::exception);
  trigger.join();
  // Join the producer before observing its non-atomic state.
  pipeline.reset();
  EXPECT_TRUE(producer_started);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap