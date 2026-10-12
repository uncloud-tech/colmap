// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include "colmap/mvs/mat_read_internal.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <vector>

namespace colmap {
namespace mvs {

namespace {

// Accept the canonical decimal W&H&D& header emitted by Mat<float>::Write.
size_t ReadDimension(std::istream& stream) {
  size_t value = 0;
  bool has_digit = false;
  char c;
  while (stream.get(c)) {
    if (c == '&') {
      THROW_CHECK(has_digit && value > 0) << "Invalid float map dimension";
      return value;
    }
    THROW_CHECK(c >= '0' && c <= '9') << "Invalid float map header";
    const size_t digit = c - '0';
    THROW_CHECK_LE(value, (std::numeric_limits<size_t>::max() - digit) / 10)
        << "Float map dimension overflow";
    value = value * 10 + digit;
    has_digit = true;
  }
  THROW_CHECK(false) << "Incomplete float map header";
  return 0;
}

}  // namespace

namespace internal {

void ReadFloatMapPayload(std::istream& stream,
                         std::vector<float>& data,
                         const bool little_endian,
                         const size_t chunk_bytes) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "Float maps require IEEE-754 binary32");
  THROW_CHECK_GE(chunk_bytes, sizeof(float));
  THROW_CHECK_LE(chunk_bytes, std::numeric_limits<std::streamsize>::max());
  const size_t chunk_floats = chunk_bytes / sizeof(float);
  for (size_t offset = 0; offset < data.size();) {
    const size_t count = std::min(chunk_floats, data.size() - offset);
    const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
    char* const dst = reinterpret_cast<char*>(data.data() + offset);
    stream.read(dst, bytes);
    THROW_CHECK(stream && stream.gcount() == bytes)
        << "Short float map payload read";
    if (!little_endian) {
      // Swap object representations, never float values (including NaN bits).
      for (size_t i = 0; i < count; ++i) {
        std::reverse(dst + i * sizeof(float), dst + (i + 1) * sizeof(float));
      }
    }
    offset += count;
  }
}

}  // namespace internal

template <>
void Mat<float>::Read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);

  const size_t width = ReadDimension(file);
  const size_t height = ReadDimension(file);
  const size_t depth = ReadDimension(file);
  const size_t max_size = std::numeric_limits<size_t>::max();
  THROW_CHECK_LE(width, max_size / height) << path;
  const size_t area = width * height;
  THROW_CHECK_LE(area, max_size / depth) << path;
  const size_t count = area * depth;
  THROW_CHECK_LE(count, data_.max_size()) << path;
  THROW_CHECK_LE(count, max_size / sizeof(float)) << path;
  const size_t bytes = count * sizeof(float);
  THROW_CHECK_LE(bytes, std::numeric_limits<std::streamoff>::max()) << path;

  // Check the remaining file length before allocating the destination. Extra
  // trailing bytes remain accepted, as with the original reader.
  const auto payload_begin = file.tellg();
  THROW_CHECK(payload_begin != std::streampos(-1)) << path;
  file.seekg(0, std::ios::end);
  const auto file_end = file.tellg();
  THROW_CHECK(file_end != std::streampos(-1) && file_end >= payload_begin)
      << path;
  THROW_CHECK_GE(file_end - payload_begin, static_cast<std::streamoff>(bytes))
      << "Truncated float map: " << path;
  file.seekg(payload_begin);
  THROW_CHECK(file) << path;

  data_.resize(count);
  internal::ReadFloatMapPayload(file, data_, IsLittleEndian());
  width_ = width;
  height_ = height;
  depth_ = depth;
  file.close();
}

template <>
void Mat<float>::Write(const std::filesystem::path& path) const {
  std::ofstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);
  file << width_ << "&" << height_ << "&" << depth_ << "&";
  WriteBinaryLittleEndian<float>(&file, {data_.data(), data_.size()});
  file.close();
}

}  // namespace mvs
}  // namespace colmap
