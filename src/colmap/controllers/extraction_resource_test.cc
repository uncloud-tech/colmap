// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/controllers/extraction_resource.h"

#include <future>
#include <limits>
#include <gtest/gtest.h>

namespace colmap {
TEST(ExtractionResource, HeaderOverlapAndOverflow) {
  // RGB decode overlaps grayscale conversion (4P); thumbnail and rotation
  // together retain original + two thumbnails, not merely final dimensions.
  EXPECT_EQ(ExtractionBitmapPayload(8, 4, 3, 4), 128);
  EXPECT_EQ(ExtractionBitmapPayload(8, 4, 1, 4), 40);
  EXPECT_EQ(ExtractionBitmapPayload(1, 1, 1, 3200), 5);
  EXPECT_EQ(ExtractionBitmapPayload(1, 1, 1, 3200, false), 2);
  EXPECT_THROW(ExtractionBitmapPayload(0, 1, 1, 3200), std::invalid_argument);
  EXPECT_THROW(ExtractionMultiply(std::numeric_limits<uint64_t>::max(), 2),
               std::overflow_error);
}
TEST(ExtractionResource, ExactMarginReserveAndSingleton) {
  ExtractionBudget exact(114, 10, 0); // ceil((100 + 10) * 1.03)
  auto ticket = exact.Acquire(100, [] { return false; });
  EXPECT_EQ(exact.Used(), 100);
  EXPECT_THROW(ticket->Grow(1), std::runtime_error);
  EXPECT_EQ(exact.Used(), 100);
  ticket.reset();
  EXPECT_EQ(exact.Used(), 0);
  ExtractionBudget small(114, 10, 1);
  EXPECT_THROW(small.Acquire(100, [] { return false; }), std::runtime_error);
}
TEST(ExtractionResource, RetirementBackpressureAndCancellation) {
  ExtractionBudget budget(103, 0, 0);
  auto first = budget.Acquire(100, [] { return false; });
  std::promise<void> waiting;
  std::once_flag notified;
  auto next = std::async(std::launch::async, [&] {
    return budget.Acquire(100, [&] {
      std::call_once(notified, [&] { waiting.set_value(); });
      return false;
    });
  });
  waiting.get_future().wait();
  first.reset();
  auto second = next.get();
  EXPECT_EQ(budget.Used(), 100);
  budget.Cancel();
  EXPECT_THROW(budget.Acquire(1, [] { return false; }), std::runtime_error);
  second.reset();
  EXPECT_EQ(budget.Used(), 0);
}
TEST(ExtractionResource, CancellationWakesBlockedAdmission) {
  ExtractionBudget budget(103, 0, 0);
  auto first = budget.Acquire(100, [] { return false; });
  std::promise<void> waiting;
  std::once_flag notified;
  auto next = std::async(std::launch::async, [&] {
    return budget.Acquire(100, [&] {
      std::call_once(notified, [&] { waiting.set_value(); });
      return false;
    });
  });
  waiting.get_future().wait();
  budget.Cancel();
  EXPECT_THROW(next.get(), std::runtime_error);
  first.reset();
  EXPECT_EQ(budget.Used(), 0);
  EXPECT_EQ(budget.Peak(), 100);
}
TEST(ExtractionResource, GrowthNeverWaitsWhileHoldingCredit) {
  ExtractionBudget budget(206, 0, 0);
  auto a = budget.Acquire(100, [] { return false; });
  auto b = budget.Acquire(100, [] { return false; });
  EXPECT_THROW(a->Grow(1), std::runtime_error);
  EXPECT_THROW(b->Grow(1), std::runtime_error);
  EXPECT_EQ(budget.Used(), 200);
}
}  // namespace colmap
