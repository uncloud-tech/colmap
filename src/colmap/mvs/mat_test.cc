// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/mat.h"

#include "colmap/mvs/mat_read_internal.h"
#include "colmap/util/endian.h"
#include "colmap/util/testing.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

class MatReadTest : public testing::Test {
 protected:
  void TearDown() override { std::filesystem::remove_all(test_dir_); }

  void WriteFile(const std::string& contents) {
    std::ofstream file(path_, std::ios::binary);
    file.write(contents.data(), contents.size());
    ASSERT_TRUE(file.good());
  }

  const std::filesystem::path test_dir_ = CreateTestDir();
  const std::filesystem::path path_ = test_dir_ / "map.bin";
};

std::string FloatPayload(const size_t count) {
  // Signed zeros, infinities, subnormals, and distinct quiet/signaling NaNs.
  const uint32_t bits[] = {0x00000000,
                           0x80000000,
                           0x3f800000,
                           0xbf800000,
                           0x7f800000,
                           0xff800000,
                           0x00000001,
                           0x7fc12345,
                           0x7f812345,
                           0xffc54321};
  std::string payload(count * sizeof(float), '\0');
  for (size_t i = 0; i < count; ++i) {
    const uint32_t value = bits[i % (sizeof(bits) / sizeof(bits[0]))];
    for (size_t byte = 0; byte < sizeof(float); ++byte) {
      payload[i * sizeof(float) + byte] =
          static_cast<char>((value >> (8 * byte)) & 0xff);
    }
  }
  return payload;
}

TEST_F(MatReadTest, MatchesLegacyBitsAndLayouts) {
  for (const size_t depth : {1, 3}) {
    const size_t width = 5;
    const size_t height = 2;
    const std::string payload = FloatPayload(width * height * depth);
    WriteFile("5&2&" + std::to_string(depth) + "&" + payload);
    Mat<float> mat;
    mat.Read(path_);
    std::istringstream old_stream(payload);
    std::vector<float> old_data(width * height * depth);
    ReadBinaryLittleEndian<float>(&old_stream, &old_data);
    EXPECT_EQ(mat.GetWidth(), width);
    EXPECT_EQ(mat.GetHeight(), height);
    EXPECT_EQ(mat.GetDepth(), depth);
    EXPECT_EQ(std::memcmp(mat.GetPtr(), old_data.data(), mat.GetNumBytes()), 0);
    // Check the planar slice/row/column indexing without copying float values.
    for (size_t slice = 0; slice < depth; ++slice) {
      for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
          const size_t idx = slice * width * height + row * width + col;
          uint32_t expected;
          std::memcpy(&expected, old_data.data() + idx, sizeof(expected));
          if ((expected & 0x7f800000) == 0x7f800000) {
            continue;  // Non-finite bits are covered by memcmp above.
          }
          uint32_t bits;
          const float value = mat.Get(row, col, slice);
          std::memcpy(&bits, &value, sizeof(bits));
          EXPECT_EQ(bits, expected);
        }
      }
    }
  }
}

TEST_F(MatReadTest, ProductionChunkBoundaries) {
  const size_t chunk = internal::kFloatMapReadChunkBytes / sizeof(float);
  for (const size_t count : {chunk - 1, chunk, chunk + 1, 2 * chunk + 1}) {
    const std::string payload = FloatPayload(count);
    WriteFile(std::to_string(count) + "&1&1&" + payload);
    Mat<float> mat;
    mat.Read(path_);
    std::vector<float> old_data(count);
    std::istringstream old_stream(payload);
    ReadBinaryLittleEndian<float>(&old_stream, &old_data);
    EXPECT_EQ(std::memcmp(mat.GetPtr(), old_data.data(), mat.GetNumBytes()), 0);
  }
}

TEST_F(MatReadTest, RejectsInvalidDimensionsAndHeadersBeforeResize) {
  const std::string max = std::to_string(std::numeric_limits<size_t>::max());
  const std::string headers[] = {
      "", "1&", "1&1&", "&1&1&", "0&1&1&", "1&0&1&", "1&1&0&",
      "-1&1&1&", "1x1&1&", "1&1&1x", "a&1&1&",
      max + "0&1&1&", max + "&2&1&", "1&" + max + "&2&",
      max + "&1&1&", std::to_string(std::numeric_limits<size_t>::max() /
                                    sizeof(float) + 1) + "&1&1&",
      std::to_string(std::numeric_limits<std::streamoff>::max() /
                     sizeof(float) + 1) + "&1&1&"};
  for (const auto& header : headers) {
    SCOPED_TRACE(header);
    WriteFile(header);
    Mat<float> mat(1, 1, 1);
    mat.Fill(42);
    EXPECT_THROW(mat.Read(path_), std::exception);
    EXPECT_EQ(mat.GetData().size(), 1);
    EXPECT_EQ(mat.Get(0, 0), 42);
    EXPECT_EQ(mat.GetWidth(), 1);
    EXPECT_EQ(mat.GetHeight(), 1);
    EXPECT_EQ(mat.GetDepth(), 1);
  }
}

TEST_F(MatReadTest, TruncationAndMinimumDimensions) {
  const std::string payload = FloatPayload(6);
  for (size_t bytes = 0; bytes < payload.size(); ++bytes) {
    WriteFile("2&1&3&" + payload.substr(0, bytes));
    Mat<float> mat;
    EXPECT_THROW(mat.Read(path_), std::exception);
    EXPECT_TRUE(mat.GetData().empty());
  }
  WriteFile("1&1&1&" + FloatPayload(1) + "trailing bytes");
  Mat<float> mat;
  EXPECT_NO_THROW(mat.Read(path_));
  EXPECT_EQ(mat.GetData().size(), 1);
}

class CountingReadBuffer : public std::stringbuf {
 public:
  explicit CountingReadBuffer(const std::string& data, const bool short_read)
      : std::stringbuf(data), short_read_(short_read) {}
  size_t calls = 0;

 protected:
  std::streamsize xsgetn(char* dst, const std::streamsize count) override {
    ++calls;
    return std::stringbuf::xsgetn(dst, short_read_ ? count - 1 : count);
  }

 private:
  const bool short_read_;
};

TEST(Mat, ChunkedPayloadAndEndianBits) {
  const std::string payload = FloatPayload(10);
  for (const bool little_endian : {false, true}) {
    for (const size_t chunk_bytes : {4, 8, 13, 40, 44}) {
      CountingReadBuffer buffer(payload, false);
      std::istream stream(&buffer);
      std::vector<float> data(10);
      internal::ReadFloatMapPayload(stream, data, little_endian, chunk_bytes);
      std::string expected = payload;
      if (!little_endian) {
        for (size_t offset = 0; offset < expected.size(); offset += 4) {
          std::reverse(expected.begin() + offset, expected.begin() + offset + 4);
        }
      }
      EXPECT_EQ(std::memcmp(data.data(), expected.data(), expected.size()), 0);
      const size_t chunk_floats = chunk_bytes / sizeof(float);
      EXPECT_EQ(buffer.calls, (data.size() + chunk_floats - 1) / chunk_floats);
    }
  }
}

TEST(Mat, RejectsShortPayloadReads) {
  const std::string payload = FloatPayload(10);
  CountingReadBuffer buffer(payload, true);
  std::istream stream(&buffer);
  std::vector<float> data(10);
  EXPECT_THROW(internal::ReadFloatMapPayload(stream, data, true, 8),
               std::exception);
  EXPECT_EQ(buffer.calls, 1);
  for (const size_t bytes : {0, 7, 8, 39}) {
    std::istringstream truncated(payload.substr(0, bytes));
    EXPECT_THROW(internal::ReadFloatMapPayload(truncated, data, true, 8),
                 std::exception);
  }
  std::istringstream valid(payload);
  EXPECT_THROW(internal::ReadFloatMapPayload(valid, data, true, 3),
               std::exception);
}

TEST(Mat, Empty) {
  Mat<int> mat;
  EXPECT_EQ(mat.GetWidth(), 0);
  EXPECT_EQ(mat.GetHeight(), 0);
  EXPECT_EQ(mat.GetDepth(), 0);
  EXPECT_EQ(mat.GetNumBytes(), 0);
}

TEST(Mat, NonEmpty) {
  Mat<int> mat(1, 2, 3);
  EXPECT_EQ(mat.GetWidth(), 1);
  EXPECT_EQ(mat.GetHeight(), 2);
  EXPECT_EQ(mat.GetDepth(), 3);
  EXPECT_EQ(mat.GetNumBytes(), 24);
}

TEST(Mat, GetSet) {
  Mat<int> mat(1, 2, 3);

  EXPECT_EQ(mat.GetNumBytes(), 24);

  mat.Set(0, 0, 0, 1);
  mat.Set(0, 0, 1, 2);
  mat.Set(0, 0, 2, 3);
  mat.Set(1, 0, 0, 4);
  mat.Set(1, 0, 1, 5);
  mat.Set(1, 0, 2, 6);

  EXPECT_EQ(mat.Get(0, 0, 0), 1);
  EXPECT_EQ(mat.Get(0, 0, 1), 2);
  EXPECT_EQ(mat.Get(0, 0, 2), 3);
  EXPECT_EQ(mat.Get(1, 0, 0), 4);
  EXPECT_EQ(mat.Get(1, 0, 1), 5);
  EXPECT_EQ(mat.Get(1, 0, 2), 6);

  int slice[3];
  mat.GetSlice(0, 0, slice);
  EXPECT_EQ(slice[0], 1);
  EXPECT_EQ(slice[1], 2);
  EXPECT_EQ(slice[2], 3);
  mat.GetSlice(1, 0, slice);
  EXPECT_EQ(slice[0], 4);
  EXPECT_EQ(slice[1], 5);
  EXPECT_EQ(slice[2], 6);
}

TEST(Mat, Fill) {
  Mat<int> mat(1, 2, 3);

  EXPECT_EQ(mat.GetNumBytes(), 24);

  mat.Fill(10);
  mat.Set(0, 0, 0, 10);
  mat.Set(0, 0, 1, 10);
  mat.Set(0, 0, 2, 10);
  mat.Set(1, 0, 0, 10);
  mat.Set(1, 0, 1, 10);
  mat.Set(1, 0, 2, 10);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
