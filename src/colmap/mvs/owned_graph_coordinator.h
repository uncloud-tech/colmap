// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <atomic>
#include <cstddef>
#include <exception>
#include <stdexcept>

namespace colmap::mvs::owned {
// Invocation cancellation only, NOT a submission coordinator. There is no
// worker, queue, mutex, cross-owner wait or device fence. Admission and slot
// reuse belong to PairCoordinator; each owner drains only its own stream.
class OwnerGraphLaunchState {
 public:
  explicit OwnerGraphLaunchState(size_t owners) : owners_(owners) {
    if (!owners) throw std::invalid_argument("Empty graph owners");
  }
  void Check(size_t owner) const {
    if (owner >= owners_) throw std::invalid_argument("Unknown graph owner");
    if (cancelled_.load())
      throw std::runtime_error("Graph invocation cancelled");
  }
  void Cancel(std::exception_ptr error) {
    if (!error) throw std::invalid_argument("Cancellation requires an error");
    cancelled_.store(true);
  }

 private:
  const size_t owners_;
  std::atomic<bool> cancelled_{false};
};
}  // namespace colmap::mvs::owned
