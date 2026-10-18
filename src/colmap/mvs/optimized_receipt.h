// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace colmap::mvs {
inline void ValidateOptimizedReceiptPath(const std::filesystem::path& path) {
  if (path.empty() || std::filesystem::exists(path) ||
      std::filesystem::is_symlink(std::filesystem::symlink_status(path)) ||
      !std::filesystem::is_directory(
          path.parent_path().empty() ? "." : path.parent_path())) {
    throw std::invalid_argument(
        "optimized dispatch receipt requires a NEW path in an existing "
        "directory");
  }
#ifdef _WIN32
  throw std::invalid_argument("optimized receipt publication requires POSIX");
#endif
}
// Publish complete bytes atomically without replacing another actor's file.
inline void PublishOptimizedReceipt(const std::filesystem::path& path,
                                    const std::string& bytes) {
  ValidateOptimizedReceiptPath(path);
#ifndef _WIN32
  std::string temporary = path.string() + ".tmp.XXXXXX";
  int fd = ::mkstemp(temporary.data());
  if (fd < 0) throw std::runtime_error("receipt temporary creation failed");
  try {
    size_t written = 0;
    while (written < bytes.size()) {
      const auto n =
          ::write(fd, bytes.data() + written, bytes.size() - written);
      if (n <= 0) throw std::runtime_error("receipt write failed");
      written += static_cast<size_t>(n);
    }
    if (::fsync(fd)) throw std::runtime_error("receipt fsync failed");
    const int status = ::close(fd);
    fd = -1;
    if (status) throw std::runtime_error("receipt close failed");
    if (::link(temporary.c_str(), path.c_str()))
      throw std::runtime_error(
          "receipt publication failed (target exists or inaccessible)");
    ::unlink(temporary.c_str());
  } catch (...) {
    if (fd >= 0) ::close(fd);
    ::unlink(temporary.c_str());
    throw;
  }
#endif
}
}  // namespace colmap::mvs
