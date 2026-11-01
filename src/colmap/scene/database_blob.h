// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace colmap {

// Validate stored full-width signed dimensions BEFORE narrowing or resizing.
// SQLite's BLOB byte count and the feature consumers use signed int ranges.
inline void ValidateDatabaseBlobShape(int64_t rows,
                                      int64_t cols,
                                      uint64_t bytes,
                                      uint64_t scalar_bytes,
                                      uint64_t max_index) {
  const auto max_dimension =
      static_cast<uint64_t>(std::numeric_limits<int>::max());
  if (rows < 0 || cols < 0 || scalar_bytes == 0 ||
      static_cast<uint64_t>(rows) > max_dimension ||
      static_cast<uint64_t>(cols) > max_dimension ||
      static_cast<uint64_t>(rows) > max_index ||
      static_cast<uint64_t>(cols) > max_index)
    throw std::runtime_error("unsupported database matrix dimensions");
  const auto r = static_cast<uint64_t>(rows);
  const auto c = static_cast<uint64_t>(cols);
  // Division checks avoid signed/unsigned product overflow, even for an
  // implementation with a narrower matrix Index or a larger Scalar.
  if ((c && r > max_index / c) ||
      (c && r > std::numeric_limits<uint64_t>::max() / c / scalar_bytes) ||
      r * c * scalar_bytes != bytes)
    throw std::runtime_error("database matrix shape/BLOB length mismatch");
}

}  // namespace colmap
