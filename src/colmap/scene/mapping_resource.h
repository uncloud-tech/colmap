// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace colmap {

inline uint64_t MappingAdd(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("mapping payload addition overflow");
  return a + b;
}

inline uint64_t MappingMultiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::overflow_error("mapping payload multiplication overflow");
  return a * b;
}

// Initial DB/model payload and graph growth only, NOT a physical SfM cap.
class MappingLedger : public std::enable_shared_from_this<MappingLedger> {
 public:
  struct Ticket {
    Ticket() = default;
    Ticket(const Ticket&) = delete;
    Ticket& operator=(const Ticket&) = delete;
    std::shared_ptr<MappingLedger> ledger;
    uint64_t bytes = 0;
    ~Ticket() {
      if (ledger) Release(bytes);
    }
    void Release(uint64_t retired) {
      std::lock_guard<std::mutex> lock(ledger->mutex_);
      if (retired > bytes) std::terminate();
      ledger->used_ -= retired;
      bytes -= retired;
    }
    void Absorb(const std::shared_ptr<Ticket>& other) {
      std::lock_guard<std::mutex> lock(ledger->mutex_);
      if (other->ledger != ledger) std::terminate();
      bytes += other->bytes;
      other->bytes = 0;
    }
  };

  MappingLedger(uint64_t capacity, uint64_t policy, uint64_t reserve)
      : capacity_(capacity), policy_(policy), reserve_(reserve) {}

  std::shared_ptr<Ticket> Acquire(uint64_t bytes) {
    auto ticket = std::make_shared<Ticket>();
    ticket->ledger = shared_from_this();
    std::lock_guard<std::mutex> lock(mutex_);
    const auto next = MappingAdd(used_, bytes);
    const auto raw = MappingAdd(next, policy_);
    const auto margin = MappingAdd((raw / 100) * 3, (raw % 100 * 3 + 99) / 100);
    if (MappingAdd(MappingAdd(raw, margin), reserve_) > capacity_)
      throw std::runtime_error("mapping logical host payload/growth refused; physical fit UNKNOWN");
    used_ = next;
    peak_ = std::max(peak_, next);
    ticket->bytes = bytes;
    return ticket;
  }

  uint64_t Peak() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_;
  }

  uint64_t Used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return used_;
  }

 private:
  mutable std::mutex mutex_;
  uint64_t capacity_, policy_, reserve_, used_ = 0, peak_ = 0;
};

// Preserve element order, and keep the original vector intact on refusal.
// Both allocations are charged until the old allocation actually dies.
// Allocator internals remain UNKNOWN, not a hard physical memory guarantee.
template <typename T>
void MappingReserve(std::vector<T>& values, size_t capacity,
                    const std::shared_ptr<MappingLedger::Ticket>& credit) {
  if (capacity <= values.capacity()) return;
  if (!credit) {
    values.reserve(capacity);
    return;
  }
  if (capacity > values.max_size()) throw std::length_error("mapping vector shape");
  const auto old_bytes = MappingMultiply(values.capacity(), sizeof(T));
  auto overlap = credit->ledger->Acquire(MappingMultiply(capacity, sizeof(T)));
  {
    std::vector<T> replacement;
    replacement.reserve(capacity);
    if (replacement.capacity() > capacity) {
      auto extra = credit->ledger->Acquire(
          MappingMultiply(replacement.capacity() - capacity, sizeof(T)));
      overlap->Absorb(extra);
    }
    replacement.insert(replacement.end(), values.begin(), values.end());
    values.swap(replacement);
  }
  credit->Release(old_bytes);
  credit->Absorb(overlap);
}

}  // namespace colmap
