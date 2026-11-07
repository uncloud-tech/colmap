// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace colmap {

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
