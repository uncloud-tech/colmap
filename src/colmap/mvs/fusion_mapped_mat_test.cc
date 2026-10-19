// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/fusion_mapped_mat.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifndef COLMAP_FUSION_MAPPED_MAT_STANDALONE_TEST
#include "colmap/mvs/mat.h"

#include <gtest/gtest.h>
#endif

namespace colmap::mvs {
namespace {
void CheckAt(const bool condition, const int line) {
  if (!condition) {
    throw std::runtime_error("Fusion mapped matrix test failed at line " +
                             std::to_string(line));
  }
}
#define Check(condition) CheckAt((condition), __LINE__)

template <typename F>
void ThrowsAt(const int line, F&& fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::exception&) {
    threw = true;
  }
  CheckAt(threw, line);
}
#define Throws(...) ThrowsAt(__LINE__, __VA_ARGS__)

#if defined(__unix__) || defined(__APPLE__)
struct TestFile {
  TestFile() {
    char pattern[] = "/tmp/colmap-fusion-mapped-XXXXXX";
    const int fd = mkstemp(pattern);
    Check(fd != -1);
    close(fd);
    path = pattern;
  }
  ~TestFile() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
  void Write(const std::string& header, const size_t count = 12) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << header;
    for (size_t i = 0; i < count; ++i) {
      const float value = static_cast<float>(i) - 2.5f;
      out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    out.close();
    Check(!out.fail());
  }
  std::filesystem::path path;
};

void RunReaderChecks() {
  static_assert(!std::is_copy_constructible_v<FusionMappedMat>);
  static_assert(!std::is_copy_assignable_v<FusionMappedMat>);
  static_assert(std::is_nothrow_move_constructible_v<FusionMappedMat>);
  static_assert(std::is_nothrow_move_assignable_v<FusionMappedMat>);
  TestFile file;
  // Six header bytes deliberately leave the float payload unaligned.
  file.Write("2&2&3&");
  {
    FusionMappedMat mat(file.path, 2, 2, 3);
    Check(mat.GetWidth() == 2 && mat.GetHeight() == 2 && mat.GetDepth() == 3);
    for (size_t c = 0; c < 3; ++c) {
      for (size_t r = 0; r < 2; ++r) {
        for (size_t x = 0; x < 2; ++x) {
          Check(mat.Get(r, x, c) ==
                static_cast<float>(c * 4 + r * 2 + x) - 2.5f);
        }
      }
    }
    Throws([&] { mat.Get(2, 0); });
    Throws([&] { mat.Get(0, 2); });
    Throws([&] { mat.Get(0, 0, 3); });
    Throws([&] { mat.Get(std::numeric_limits<size_t>::max(), 0); });
    mat.VerifyUnchanged();
    FusionMappedMat moved(std::move(mat));
    Check(mat.GetWidth() == 0);
    Throws([&] { mat.Get(0, 0); });
    Throws([&] { mat.VerifyUnchanged(); });
    FusionMappedMat assigned(file.path, 2, 2, 3);
    assigned = std::move(moved);
    Check(assigned.Get(1, 1, 2) == 8.5f);
    assigned.VerifyUnchanged();
  }
  {
    // Include signed infinities and quiet NaNs with distinct signs/payloads
    // in every planar channel, retaining the deliberately unaligned header.
    const uint32_t bits[] = {0x7f800000,
                             0xff800000,
                             0x7fc00001,
                             0xffc12345,
                             0xff800000,
                             0x7fc54321,
                             0x7f800000,
                             0xffc00001,
                             0x7fc12345,
                             0xffc54321,
                             0xff800000,
                             0x7f800000};
    {
      std::ofstream out(file.path, std::ios::binary | std::ios::trunc);
      out << "2&2&3&";
      out.write(reinterpret_cast<const char*>(bits), sizeof(bits));
      out.close();
      Check(!out.fail());
    }
    FusionMappedMat mapped(file.path, 2, 2, 3);
#ifndef COLMAP_FUSION_MAPPED_MAT_STANDALONE_TEST
    Mat<float> stock;
    stock.Read(file.path);
#endif
    for (size_t c = 0; c < 3; ++c) {
      for (size_t r = 0; r < 2; ++r) {
        for (size_t x = 0; x < 2; ++x) {
          const float value = mapped.Get(r, x, c);
          uint32_t actual;
          std::memcpy(&actual, &value, sizeof(actual));
          Check(actual == bits[c * 4 + r * 2 + x]);
#ifndef COLMAP_FUSION_MAPPED_MAT_STANDALONE_TEST
          const float stock_value = stock.Get(r, x, c);
          uint32_t stock_bits;
          std::memcpy(&stock_bits, &stock_value, sizeof(stock_bits));
          Check(actual == stock_bits);
#endif
        }
      }
    }
    mapped.VerifyUnchanged();
  }
  file.Write("2&2&1&", 4);
  Check(FusionMappedMat(file.path, 2, 2, 1).Get(1, 1) == 0.5f);
  for (const std::string header : {"",
                                   "2&2&",
                                   "2&2&3",
                                   "-2&2&3&",
                                   "+2&2&3&",
                                   " 2&2&3&",
                                   "0&2&3&",
                                   "2x2&3&",
                                   "2&&3&",
                                   "2&2&2&",
                                   "3&2&3&",
                                   "184467440737095516160&2&3&"}) {
    file.Write(header);
    Throws([&] { FusionMappedMat mat(file.path, 2, 2, 3); });
  }
  for (const size_t count : {0, 11, 13}) {
    file.Write("2&2&3&", count);
    Throws([&] { FusionMappedMat mat(file.path, 2, 2, 3); });
  }
  for (const size_t size : {53, 55}) {
    file.Write("2&2&3&");
    std::filesystem::resize_file(file.path, size);
    Throws([&] { FusionMappedMat mat(file.path, 2, 2, 3); });
  }
  file.Write("2&2&3&");
  Throws([&] { FusionMappedMat mat(file.path, 0, 2, 3); });
  Throws([&] { FusionMappedMat mat(file.path, 2, 2, 2); });
  Throws([&] {
    FusionMappedMat mat(file.path, std::numeric_limits<size_t>::max(), 2, 3);
  });
  Throws([&] {
    FusionMappedMat mat(
        file.path, std::numeric_limits<size_t>::max() / 4 + 1, 1, 1);
  });
  Throws([&] { FusionMappedMat mat(file.path.parent_path(), 2, 2, 3); });
  {
    FusionMappedMat mat(file.path, 2, 2, 3);
    auto timestamp = std::filesystem::last_write_time(file.path);
    std::filesystem::last_write_time(file.path,
                                     timestamp + std::chrono::seconds(2));
    Throws([&] { mat.VerifyUnchanged(); });
  }
  {
    FusionMappedMat mat(file.path, 2, 2, 3);
    // Some test filesystems expose coarse ctime resolution.
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    Check(chmod(file.path.c_str(), 0400) == 0);
    Throws([&] { mat.VerifyUnchanged(); });
    Check(chmod(file.path.c_str(), 0600) == 0);
  }
  {
    FusionMappedMat mat(file.path, 2, 2, 3);
    std::filesystem::resize_file(file.path, 1);
    // Never load from a truncated mapping: verification is the safe boundary.
    Throws([&] { mat.VerifyUnchanged(); });
  }
  file.Write("2&2&3&");
  {
    FusionMappedMat mat(file.path, 2, 2, 3);
    TestFile replacement;
    replacement.Write("2&2&3&");
    std::filesystem::rename(replacement.path, file.path);
    Throws([&] { mat.VerifyUnchanged(); });
  }
  std::filesystem::remove(file.path);
  Throws([&] { FusionMappedMat mat(file.path, 2, 2, 3); });
  Check(mkfifo(file.path.c_str(), 0600) == 0);
  Throws([&] { FusionMappedMat mat(file.path, 2, 2, 3); });
}
#else
void RunReaderChecks() {
  Throws([] { FusionMappedMat mat("unsupported", 1, 1, 1); });
}
#endif

#undef Check
#undef Throws

#ifndef COLMAP_FUSION_MAPPED_MAT_STANDALONE_TEST
TEST(FusionMappedMat, ReaderContract) { RunReaderChecks(); }
#endif
}  // namespace
}  // namespace colmap::mvs

#ifdef COLMAP_FUSION_MAPPED_MAT_STANDALONE_TEST
int main() { colmap::mvs::RunReaderChecks(); }
#endif
