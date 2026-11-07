// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/extraction_resource.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>

namespace colmap {

// Reserve streamed working owners on the existing phase ledger. Public output
// arrays are separately image-owned through the database writer.
inline std::shared_ptr<ExtractionBudget::Ticket> ReserveSiftStreamingScratch(
    uint64_t staging_bytes,
    uint64_t metadata_bytes,
    const std::function<std::shared_ptr<ExtractionBudget::Ticket>(uint64_t)>&
        acquire,
    const std::function<void(uint64_t)>& reserve) {
  const auto bytes = ExtractionAdd(staging_bytes, metadata_bytes);
  if (acquire) {
    // Reuse the existing phase ledger without retaining its image ticket.
    // Unlike extractor history/capacities, this working set can drain while a
    // reader is waiting for admission. Its credit dies after the working
    // owners.
    auto phase = acquire(0);
    if (!phase) throw std::invalid_argument("missing SIFT phase host credit");
    return phase->Fork(bytes, false);
  }
  // Preserve the optional callback behavior for direct explicit native callers.
  if (reserve) reserve(bytes);
  return {};
}

// Known owned live payload only, NOT an aggregate RSS/page-cache certificate.
// The caller includes decoded Bitmap and retained output capacities, then
// checks again with the actual final count before allocating the public result
// arrays.
inline uint64_t CheckSiftStreamingPayload(uint64_t bitmap_bytes,
                                          uint64_t retained_bytes,
                                          uint64_t staging_bytes,
                                          uint64_t metadata_bytes,
                                          uint64_t result_count,
                                          uint64_t result_stride,
                                          uint64_t host_cap,
                                          uint64_t policy_bytes,
                                          uint64_t reserve_bytes) {
  auto add = [](uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
      throw std::overflow_error("SIFT streaming host payload overflow");
    return a + b;
  };
  if (result_stride &&
      result_count > std::numeric_limits<uint64_t>::max() / result_stride)
    throw std::overflow_error("SIFT streaming result payload overflow");
  const uint64_t payload = add(add(add(bitmap_bytes, retained_bytes),
                                   add(staging_bytes, metadata_bytes)),
                               result_count * result_stride);
  if (!host_cap || add(add(payload, policy_bytes), reserve_bytes) > host_cap)
    throw std::runtime_error("SIFT streaming known host payload exceeds cap");
  return payload;
}

}  // namespace colmap
