// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

namespace colmap::mvs {

// Read-only, full-resolution native float map for fusion. The caller must keep
// the file immutable for the entire mapping lifetime, including all task joins.
// Payloads need not be aligned: Get uses byte copies, never float-pointer
// casts. Only supported native little-endian IEEE-754 platforms are accepted.
class FusionMappedMat {
 public:
  FusionMappedMat(const std::filesystem::path& path,
                  size_t width,
                  size_t height,
                  size_t channels);
  ~FusionMappedMat();
  FusionMappedMat(FusionMappedMat&&) noexcept;
  FusionMappedMat& operator=(FusionMappedMat&&) noexcept;
  FusionMappedMat(const FusionMappedMat&) = delete;
  FusionMappedMat& operator=(const FusionMappedMat&) = delete;

  float Get(size_t row, size_t col, size_t channel = 0) const;
  size_t GetWidth() const;
  size_t GetHeight() const;
  size_t GetDepth() const;

  // Detect identity/size/mtime/ctime changes at safe phase boundaries. This is
  // not protection against concurrent truncation between validation and loads.
  void VerifyUnchanged() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace colmap::mvs
