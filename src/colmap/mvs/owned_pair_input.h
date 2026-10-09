// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <exception>
#include <future>
#include <memory>
#include <vector>

namespace colmap::mvs::owned {

// Keep captured inputs alive until BOTH futures and the pool's independent
// exception-checker queue are consumed. Only then can Stop safely join workers.
// Cancel must be noexcept and publish failure before any resource unwinds.
template <class Pool, class Cancel>
void DrainPairInputs(std::unique_ptr<Pool>& pool,
                     std::vector<std::shared_future<void>>& futures,
                     std::exception_ptr first_error,
                     Cancel cancel) {
  for (auto& future : futures) {
    try {
      future.get();
    } catch (...) {
      if (!first_error) first_error = std::current_exception();
      cancel(first_error);
    }
  }
  try {
    pool->Wait();
  } catch (...) {
    if (!first_error) first_error = std::current_exception();
    cancel(first_error);
  }
  pool.reset();
  if (first_error) std::rethrow_exception(first_error);
}

}  // namespace colmap::mvs::owned
