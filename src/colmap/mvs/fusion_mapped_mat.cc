// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/fusion_mapped_mat.h"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace colmap::mvs {
namespace {
size_t CheckedMultiply(const size_t a, const size_t b) {
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
    throw std::runtime_error("Fusion mapped matrix size overflow");
  }
  return a * b;
}
}  // namespace

struct FusionMappedMat::Impl {
  std::filesystem::path path;
  size_t width = 0;
  size_t height = 0;
  size_t channels = 0;
  size_t payload_offset = 0;
  size_t size = 0;
#if defined(__unix__) || defined(__APPLE__)
  int fd = -1;
  void* mapping = MAP_FAILED;
  struct stat identity{};

  ~Impl() {
    if (mapping != MAP_FAILED) {
      munmap(mapping, size);
    }
    if (fd != -1) {
      close(fd);
    }
  }

  bool SameIdentity(const struct stat& other) const {
#if defined(__APPLE__)
    const auto mtime = identity.st_mtimespec;
    const auto ctime = identity.st_ctimespec;
    const auto other_mtime = other.st_mtimespec;
    const auto other_ctime = other.st_ctimespec;
#else
    const auto mtime = identity.st_mtim;
    const auto ctime = identity.st_ctim;
    const auto other_mtime = other.st_mtim;
    const auto other_ctime = other.st_ctim;
#endif
    return S_ISREG(other.st_mode) && identity.st_dev == other.st_dev &&
           identity.st_ino == other.st_ino &&
           identity.st_size == other.st_size &&
           mtime.tv_sec == other_mtime.tv_sec &&
           mtime.tv_nsec == other_mtime.tv_nsec &&
           ctime.tv_sec == other_ctime.tv_sec &&
           ctime.tv_nsec == other_ctime.tv_nsec;
  }

  void Verify() const {
    struct stat current{};
    struct stat named{};
    if (fstat(fd, &current) != 0 || stat(path.c_str(), &named) != 0 ||
        !SameIdentity(current) || !SameIdentity(named)) {
      throw std::runtime_error("Fusion mapped matrix changed: " +
                               path.string());
    }
  }

  size_t ReadDimension() {
    size_t value = 0;
    size_t digits = 0;
    // Bound header work, including arbitrarily many leading zeros.
    constexpr size_t kMaxDigits = std::numeric_limits<size_t>::digits10 + 1;
    while (payload_offset < size) {
      char ch;
      ssize_t count;
      do {
        count = pread(fd, &ch, 1, static_cast<off_t>(payload_offset));
      } while (count < 0 && errno == EINTR);
      if (count != 1) {
        throw std::runtime_error("Cannot read fusion mapped matrix header");
      }
      ++payload_offset;
      if (ch == '&' && digits != 0 && value != 0) {
        return value;
      }
      if (ch < '0' || ch > '9' || ++digits > kMaxDigits ||
          value > (std::numeric_limits<size_t>::max() - (ch - '0')) / 10) {
        throw std::runtime_error("Invalid fusion mapped matrix header");
      }
      value = value * 10 + (ch - '0');
    }
    throw std::runtime_error("Truncated fusion mapped matrix header");
  }
#else
  void Verify() const {
    throw std::runtime_error("Fusion mapped matrix requires POSIX mmap");
  }
#endif
};

FusionMappedMat::FusionMappedMat(const std::filesystem::path& path,
                                 const size_t width,
                                 const size_t height,
                                 const size_t channels)
    : impl_(std::make_unique<Impl>()) {
  const uint32_t one = 1;
  const float float_one = 1.0f;
  const unsigned char expected[] = {0, 0, 128, 63};
  if (CHAR_BIT != 8 || sizeof(float) != 4 ||
      !std::numeric_limits<float>::is_iec559 ||
      *reinterpret_cast<const unsigned char*>(&one) != 1 ||
      std::memcmp(&float_one, expected, sizeof(expected)) != 0) {
    throw std::runtime_error(
        "Fusion mapped matrix requires native little-endian IEEE-754 floats");
  }
  // Native files have no endian marker: foreign-endian payloads cannot in
  // general be detected. The opt-in caller must supply native immutable files.
  if (width == 0 || height == 0 || (channels != 1 && channels != 3)) {
    throw std::invalid_argument("Invalid fusion mapped matrix dimensions");
  }
  const size_t bytes = CheckedMultiply(
      CheckedMultiply(CheckedMultiply(width, height), channels), sizeof(float));
  impl_->path = std::filesystem::absolute(path);
  impl_->width = width;
  impl_->height = height;
  impl_->channels = channels;
#if defined(__unix__) || defined(__APPLE__)
  // NONBLOCK prevents a malformed FIFO input from blocking before fstat.
  impl_->fd = open(impl_->path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  if (impl_->fd == -1 || fstat(impl_->fd, &impl_->identity) != 0 ||
      !S_ISREG(impl_->identity.st_mode) || impl_->identity.st_size <= 0 ||
      static_cast<uintmax_t>(impl_->identity.st_size) >
          std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Cannot open regular fusion mapped matrix: " +
                             path.string());
  }
  impl_->size = static_cast<size_t>(impl_->identity.st_size);
  impl_->Verify();
  if (impl_->ReadDimension() != width || impl_->ReadDimension() != height ||
      impl_->ReadDimension() != channels) {
    throw std::runtime_error("Fusion mapped matrix dimensions mismatch");
  }
  if (bytes != impl_->size - impl_->payload_offset) {
    throw std::runtime_error("Fusion mapped matrix payload size mismatch");
  }
  impl_->Verify();
  impl_->mapping =
      mmap(nullptr, impl_->size, PROT_READ, MAP_PRIVATE, impl_->fd, 0);
  if (impl_->mapping == MAP_FAILED) {
    throw std::runtime_error("Cannot map fusion matrix: " + path.string());
  }
  impl_->Verify();
  // Checks bracket setup, not concurrent writes. In particular, truncating a
  // live mapping can cause SIGBUS. Owners must keep files immutable and keep
  // this object alive until all reader tasks have joined.
#else
  (void)bytes;
  impl_->Verify();
#endif
}

FusionMappedMat::~FusionMappedMat() = default;
FusionMappedMat::FusionMappedMat(FusionMappedMat&&) noexcept = default;
FusionMappedMat& FusionMappedMat::operator=(FusionMappedMat&&) noexcept =
    default;

float FusionMappedMat::Get(const size_t row,
                           const size_t col,
                           const size_t channel) const {
  if (!impl_ || row >= impl_->height || col >= impl_->width ||
      channel >= impl_->channels) {
    throw std::out_of_range("Fusion mapped matrix index out of range");
  }
#if defined(__unix__) || defined(__APPLE__)
  const size_t index = (channel * impl_->height + row) * impl_->width + col;
  float value;
  std::memcpy(&value,
              static_cast<const unsigned char*>(impl_->mapping) +
                  impl_->payload_offset + index * sizeof(float),
              sizeof(value));
  return value;
#else
  throw std::runtime_error("Fusion mapped matrix requires POSIX mmap");
#endif
}

size_t FusionMappedMat::GetWidth() const { return impl_ ? impl_->width : 0; }
size_t FusionMappedMat::GetHeight() const { return impl_ ? impl_->height : 0; }
size_t FusionMappedMat::GetDepth() const { return impl_ ? impl_->channels : 0; }

void FusionMappedMat::VerifyUnchanged() const {
  if (!impl_) {
    throw std::runtime_error("Fusion mapped matrix was moved from");
  }
  impl_->Verify();
}
}  // namespace colmap::mvs
