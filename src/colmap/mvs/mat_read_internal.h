// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <istream>
#include <vector>

namespace colmap {
namespace mvs {
namespace internal {

// Mat-only seam for exercising chunked reads and non-native byte order.
// little_endian specifies host byte order; the file is always little-endian.
constexpr size_t kFloatMapReadChunkBytes = 1024 * 1024;
void ReadFloatMapPayload(
    std::istream& stream,
    std::vector<float>& data,
    bool little_endian,
    size_t chunk_bytes = kFloatMapReadChunkBytes);

}  // namespace internal
}  // namespace mvs
}  // namespace colmap
