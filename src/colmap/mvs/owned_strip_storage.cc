// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_strip_storage.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/magic.h>
#include <sys/vfs.h>
#endif
#endif

namespace colmap::mvs::owned {
uint64_t StripCheckedProduct(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::overflow_error("strip byte size overflow");
  return a * b;
}
StripStagingLayout BuildStripStagingLayout(
    size_t w, size_t h, size_t s, size_t columns, bool mask) {
  if (!w || !h || !s || !columns || w > INT_MAX || h > INT_MAX || s > INT_MAX ||
      columns > INT_MAX)
    throw std::invalid_argument(
        "strip staging dimensions outside native range");
  const size_t extent = std::max(w, h);
  columns = std::min(columns, extent);
  const uint64_t values =
      StripCheckedProduct(StripCheckedProduct(columns, extent), s);
  const uint64_t payload = StripCheckedProduct(values, 4);
  const uint64_t row = StripCheckedProduct(extent, 4);
  if (payload > std::numeric_limits<uint64_t>::max() - row)
    throw std::overflow_error("strip host staging sum overflow");
  return {columns,
          extent,
          values,
          payload + row,
          StripCheckedProduct(values, mask ? 13 : 12)};
}

void ValidateStripColumns(size_t columns, size_t block_width) {
  if ((block_width != 8 && block_width != 16 && block_width != 32) ||
      !columns || columns > INT_MAX || columns % block_width)
    throw std::invalid_argument("strip columns must be block aligned");
}
std::pair<size_t, size_t> StripCanonicalPixel(
    size_t w, size_t h, unsigned o, size_t r, size_t c) {
  if (!w || !h || o > 3 || r >= (o % 2 ? w : h) || c >= (o % 2 ? h : w))
    throw std::out_of_range("strip oriented coordinate");
  switch (o) {
    case 0:
      return {r, c};
    case 1:
      return {c, w - 1 - r};
    case 2:
      return {h - 1 - r, w - 1 - c};
    default:
      return {h - 1 - c, r};
  }
}

struct StripFileStore::Impl {
  size_t width, height, sources;
  bool mask;
  uint64_t area, bytes, read = 0, written = 0;
  const std::atomic<bool>* cancelled;
  std::vector<unsigned char> row;
#ifdef __linux__
  int fd = -1;
  ~Impl() {
    if (fd >= 0) close(fd);
  }
#endif
  Impl(size_t w, size_t h, size_t s, bool m, const std::atomic<bool>* c)
      : width(w),
        height(h),
        sources(s),
        mask(m),
        area(StripCheckedProduct(w, h)),
        bytes(StripCheckedProduct(StripCheckedProduct(area, s), m ? 9 : 8)),
        cancelled(c) {
    if (!w || !h || !s || w > INT_MAX || h > INT_MAX || s > INT_MAX ||
        bytes > uint64_t(std::numeric_limits<int64_t>::max()))
      throw std::invalid_argument("strip file dimensions outside native range");
    row.resize(StripCheckedProduct(std::max(w, h), sizeof(float)));
  }
  void Check() const {
    if (cancelled && cancelled->load())
      throw std::runtime_error("stereo strip cancelled");
  }
  [[noreturn]] void Fail(const char* operation) const {
    throw std::runtime_error(std::string("stereo strip ") + operation + ": " +
                             std::strerror(errno));
  }
#ifdef __linux__
  void Io(bool write, void* buffer, size_t length, uint64_t offset) {
    auto* p = static_cast<unsigned char*>(buffer);
    while (length) {
      Check();
      const size_t chunk = std::min(length, size_t(SSIZE_MAX));
      const ssize_t n = write ? pwrite(fd, p, chunk, off_t(offset))
                              : pread(fd, p, chunk, off_t(offset));
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) Fail(write ? "write" : "read");
      if (!n) throw std::runtime_error("stereo strip short I/O");
      p += n;
      length -= size_t(n);
      offset += uint64_t(n);
      (write ? written : read) += uint64_t(n);
    }
  }
#endif
  void Transfer(bool write,
                StripPlane plane,
                unsigned o,
                size_t first,
                size_t columns,
                void* buffer) {
    Check();
    const size_t ow = o % 2 ? height : width;
    const size_t oh = o % 2 ? width : height;
    if (o > 3 || !columns || first >= ow || columns > ow - first || !buffer)
      throw std::invalid_argument("invalid strip transfer range");
    if (plane != StripPlane::kCost && plane != StripPlane::kSelection &&
        plane != StripPlane::kMask)
      throw std::invalid_argument("invalid strip plane");
    if (plane == StripPlane::kMask && !mask)
      throw std::invalid_argument("strip mask absent");
    const size_t element = plane == StripPlane::kMask ? 1 : 4;
    const uint64_t base =
        plane == StripPlane::kCost
            ? 0
            : StripCheckedProduct(area * sources,
                                  plane == StripPlane::kSelection ? 4 : 8);
    const auto a = StripCanonicalPixel(width, height, o, 0, first);
    const auto b =
        StripCanonicalPixel(width, height, o, oh - 1, first + columns - 1);
    const size_t r0 = std::min(a.first, b.first),
                 r1 = std::max(a.first, b.first);
    const size_t c0 = std::min(a.second, b.second),
                 c1 = std::max(a.second, b.second);
    auto* tile = static_cast<unsigned char*>(buffer);
#ifdef __linux__
    for (size_t s = 0; s < sources; ++s) {
      for (size_t r = r0; r <= r1; ++r) {
        const uint64_t offset =
            base + ((s * height + r) * width + c0) * element;
        const size_t count = (c1 - c0 + 1) * element;
        if (!write) Io(false, row.data(), count, offset);
        for (size_t c = c0; c <= c1; ++c) {
          size_t tr = 0, tc = 0;
          switch (o) {
            case 0:
              tr = r;
              tc = c;
              break;
            case 1:
              tr = width - 1 - c;
              tc = r;
              break;
            case 2:
              tr = height - 1 - r;
              tc = width - 1 - c;
              break;
            case 3:
              tr = c;
              tc = height - 1 - r;
              break;
          }
          auto* value = tile + ((s * oh + tr) * columns + tc - first) * element;
          if (write)
            std::memcpy(row.data() + (c - c0) * element, value, element);
          else
            std::memcpy(value, row.data() + (c - c0) * element, element);
        }
        if (write) Io(true, row.data(), count, offset);
      }
    }
    // This unlinked scratch file is not a durable checkpoint: pwrite/pread
    // provide coherence without per-transfer writeback or cache eviction.
    // Admission charges the full file size to aggregate host page cache.
#else
    (void)element;
    (void)write;
    (void)base;
    (void)tile;
    (void)r0;
    (void)r1;
    (void)c0;
    (void)c1;
    throw std::runtime_error("stereo strip file backend unavailable");
#endif
  }
};

StripFileStore::StripFileStore(const std::filesystem::path& directory,
                               size_t w,
                               size_t h,
                               size_t s,
                               bool mask,
                               uint64_t budget,
                               const std::atomic<bool>* cancelled)
    : impl_(std::make_unique<Impl>(w, h, s, mask, cancelled)) {
  impl_->Check();
  if (directory.empty())
    throw std::invalid_argument("strip scratch directory is required");
  if (impl_->bytes > budget)
    throw std::invalid_argument("strip disk budget below minimum");
#ifdef __linux__
  if (impl_->bytes > uint64_t(std::numeric_limits<off_t>::max()))
    throw std::overflow_error("strip file offset overflow");
  std::string pattern = (directory / "colmap-stereo-XXXXXX").string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back(0);
#ifdef __linux__
  impl_->fd = mkostemp(name.data(), O_CLOEXEC);
#else
  impl_->fd = mkstemp(name.data());
#endif
  if (impl_->fd < 0) impl_->Fail("create");
  if (unlink(name.data())) impl_->Fail("unlink");
  if (fcntl(impl_->fd, F_SETFD, FD_CLOEXEC) < 0) impl_->Fail("close-on-exec");
#ifdef __linux__
  struct statfs fs;
  if (fstatfs(impl_->fd, &fs)) impl_->Fail("filesystem");
  if (fs.f_type == TMPFS_MAGIC || fs.f_type == RAMFS_MAGIC)
    throw std::invalid_argument("stereo strip scratch must be disk backed");
#endif
  const int error = posix_fallocate(impl_->fd, 0, off_t(impl_->bytes));
  if (error) {
    errno = error;
    impl_->Fail("reserve disk");
  }
#else
  (void)directory;
  throw std::runtime_error("stereo strip file backend unavailable");
#endif
}
StripFileStore::~StripFileStore() = default;
uint64_t StripFileStore::Bytes() const { return impl_->bytes; }
uint64_t StripFileStore::BytesRead() const { return impl_->read; }
uint64_t StripFileStore::BytesWritten() const { return impl_->written; }
size_t StripFileStore::RowBufferBytes() const { return impl_->row.size(); }
void StripFileStore::CheckCancelled() const { impl_->Check(); }
void StripFileStore::Read(
    StripPlane p, unsigned o, size_t f, size_t n, void* b) {
  impl_->Transfer(false, p, o, f, n, b);
}
void StripFileStore::Write(
    StripPlane p, unsigned o, size_t f, size_t n, const void* b) {
  impl_->Transfer(true, p, o, f, n, const_cast<void*>(b));
}
}  // namespace colmap::mvs::owned
