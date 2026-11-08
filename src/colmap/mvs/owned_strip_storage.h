// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>

namespace colmap::mvs::owned {

// Native qualification only. These are staging/scratch caps, not a bound on
// the remaining resident reference/source owners or aggregate process memory.
struct StripStreamingOptions {
  std::filesystem::path directory;
  size_t columns = 0;
  uint64_t host_staging_bytes = 0;
  uint64_t device_staging_bytes = 0;
  uint64_t disk_bytes = 0;
  const std::atomic<bool>* cancelled = nullptr;
};

struct StripStagingLayout {
  size_t columns, height;
  uint64_t values, host_bytes, device_bytes;
};
StripStagingLayout BuildStripStagingLayout(
    size_t width, size_t height, size_t sources, size_t columns, bool mask);

uint64_t StripCheckedProduct(uint64_t a, uint64_t b);
// Global block boundaries must match the resident kernel, including tail lanes.
void ValidateStripColumns(size_t columns, size_t block_width);
std::pair<size_t, size_t> StripCanonicalPixel(
    size_t width, size_t height, unsigned orientation, size_t row, size_t col);

enum class StripPlane { kCost, kSelection, kMask };

// One exclusive, unlinked, physically reserved disk file. Layout is canonical
// source-major cost(float), previous selection(float), optional mask(uint8).
// Only the enclosing solver may interpret completion; partial commits are NOT
// restartable. Each transfer uses one bounded canonical row buffer internally.
class StripFileStore {
 public:
  StripFileStore(const std::filesystem::path& directory,
                 size_t width,
                 size_t height,
                 size_t sources,
                 bool mask,
                 uint64_t disk_budget,
                 const std::atomic<bool>* cancelled = nullptr);
  ~StripFileStore();
  StripFileStore(const StripFileStore&) = delete;
  StripFileStore& operator=(const StripFileStore&) = delete;

  uint64_t Bytes() const;
  uint64_t BytesRead() const;
  uint64_t BytesWritten() const;
  size_t RowBufferBytes() const;
  void CheckCancelled() const;
  // Buffer is packed source-major [source][oriented row][local column].
  void Read(StripPlane plane,
            unsigned orientation,
            size_t first_column,
            size_t columns,
            void* buffer);
  void Write(StripPlane plane,
             unsigned orientation,
             size_t first_column,
             size_t columns,
             const void* buffer);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace colmap::mvs::owned
