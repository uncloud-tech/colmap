// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/published_geo.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#ifdef __linux__
#include <fcntl.h>
#include <linux/magic.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <unistd.h>
#endif

namespace colmap::mvs::owned {
namespace {
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void ValidateMat(const Mat<float>& map) {
  const auto width = map.GetWidth(), height = map.GetHeight();
  const auto channels = map.GetDepth();
  const auto maximum = std::numeric_limits<size_t>::max();
  Require(width && height && channels &&
              width <= size_t(std::numeric_limits<int>::max()) &&
              height <= size_t(std::numeric_limits<int>::max()) &&
              width <= maximum / height,
          "published GEO map dimensions outside native range");
  const auto area = width * height;
  Require(area <= maximum / channels, "published GEO map size overflow");
  const auto count = area * channels;
  Require(count <= maximum / sizeof(float) && count == map.GetData().size() &&
              count * sizeof(float) <=
                  uint64_t(std::numeric_limits<int64_t>::max()) -
                      PublishedGeoWriter::kPageBytes,
          "published GEO map payload/offset mismatch");
}
#ifdef __linux__
struct File {
  int fd;
  ~File() {
    if (fd >= 0) close(fd);
  }
  void Close() {
    const int owned = fd;
    fd = -1;  // Linux close must not be retried, including EINTR.
    Require(close(owned) == 0, "published GEO close failed");
  }
};
PublishedGeoFile Identity(int fd, const std::filesystem::path& path) {
  struct stat info{};
  Require(fstat(fd, &info) == 0 && S_ISREG(info.st_mode) &&
              info.st_uid == geteuid() && info.st_nlink == 1 &&
              info.st_size > 0,
          "published GEO file identity/ownership invalid");
  return {path,
          uint64_t(info.st_dev),
          uint64_t(info.st_ino),
          uint64_t(info.st_size),
          info.st_mtim.tv_sec,
          info.st_mtim.tv_nsec,
          info.st_ctim.tv_sec,
          info.st_ctim.tv_nsec};
}
void VerifyFile(const PublishedGeoFile& expected) {
  Require(std::filesystem::canonical(expected.path.parent_path()) ==
              expected.path.parent_path(),
          "published GEO parent changed");
  File file{open(expected.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
  Require(file.fd >= 0, "published GEO reopen failed");
  const auto actual = Identity(file.fd, expected.path);
  Require(actual.device == expected.device && actual.inode == expected.inode &&
              actual.bytes == expected.bytes &&
              actual.modified_seconds == expected.modified_seconds &&
              actual.modified_nanoseconds == expected.modified_nanoseconds &&
              actual.changed_seconds == expected.changed_seconds &&
              actual.changed_nanoseconds == expected.changed_nanoseconds,
          "published GEO identity changed");
  std::vector<unsigned char> residency(PublishedGeoWriter::kProbeBytes);
  for (uint64_t offset = 0; offset < actual.bytes;) {
    const auto length = size_t(std::min<uint64_t>(
        PublishedGeoWriter::kProbeWindowBytes, actual.bytes - offset));
    void* mapping = mmap(nullptr,
                         length,
                         PROT_NONE,
                         MAP_SHARED,
                         file.fd,
                         static_cast<off_t>(offset));
    Require(mapping != MAP_FAILED, "published GEO mmap unsupported");
    // Do not touch the mapping. mincore emits ONE BYTE per page, not a bitmap.
    const int status = mincore(mapping, length, residency.data());
    const int unmapped = munmap(mapping, length);
    Require(status == 0 && unmapped == 0, "published GEO mincore/unmap failed");
    const auto pages = (length + PublishedGeoWriter::kPageBytes - 1) /
                       PublishedGeoWriter::kPageBytes;
    for (size_t i = 0; i < pages; ++i)
      Require(!(residency[i] & 1), "published GEO cache residency is nonzero");
    offset += length;
  }
  file.Close();
}
#endif
}  // namespace

PublishedGeoWriter::PublishedGeoWriter(const std::filesystem::path& root,
                                       std::vector<std::string> references,
                                       const std::atomic<bool>* cancel)
    : root_(std::filesystem::absolute(root).lexically_normal()),
      references_(std::move(references)),
      cancel_(cancel) {
  Check();
  Require(!references_.empty(), "published GEO requires references");
  std::set<std::string> unique;
  for (const auto& name : references_) {
    const std::filesystem::path path(name);
    Require(!path.empty() && !path.is_absolute() &&
                path.lexically_normal() == path && unique.insert(name).second,
            "invalid published GEO reference");
    for (const auto& part : path)
      Require(part != "." && part != "..", "invalid published GEO component");
  }
#ifdef __linux__
  const uint16_t endian = 1;
  Require(*reinterpret_cast<const unsigned char*>(&endian) == 1 &&
              sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
              sysconf(_SC_PAGESIZE) == long(kPageBytes),
          "published GEO requires little-endian float32 Linux/4KiB pages");
  Require(
      std::filesystem::canonical(root_.parent_path()) == root_.parent_path(),
      "published GEO parent missing or symlinked");
  struct statfs filesystem{};
  Require(statfs(root_.parent_path().c_str(), &filesystem) == 0 &&
              filesystem.f_type != TMPFS_MAGIC &&
              filesystem.f_type != RAMFS_MAGIC,
          "published GEO requires disk-backed filesystem");
  Require(mkdir(root_.c_str(), 0700) == 0, "fresh published GEO root required");
#else
  throw std::runtime_error("published GEO requires Linux");
#endif
}
void PublishedGeoWriter::Check() const {
  if (cancel_ && cancel_->load())
    throw std::runtime_error("published GEO cancelled");
}
void PublishedGeoWriter::Verify() const {
  Check();
#ifdef __linux__
  struct stat root{};
  Require(lstat(root_.c_str(), &root) == 0 && S_ISDIR(root.st_mode) &&
              root.st_uid == geteuid() && !(root.st_mode & 0077),
          "published GEO namespace is no longer private");
  for (const auto& file : files_) {
    Check();
    VerifyFile(file);
  }
#else
  throw std::runtime_error("published GEO requires Linux");
#endif
}
PublishedGeoFile PublishedGeoWriter::Write(const std::filesystem::path& path,
                                           const Mat<float>& map) {
  Check();
  Require(map.GetWidth() && map.GetHeight() && map.GetNumBytes(),
          "empty published GEO map");
#ifdef __linux__
  std::filesystem::create_directories(path.parent_path());
  Require(std::filesystem::canonical(path.parent_path()) == path.parent_path(),
          "published GEO parent symlink");
  File file{open(path.c_str(),
                 O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                 0600)};
  Require(file.fd >= 0, "published GEO exclusive create failed");
  const auto header = std::to_string(map.GetWidth()) + "&" +
                      std::to_string(map.GetHeight()) + "&" +
                      std::to_string(map.GetDepth()) + "&";
  const auto write_all = [&](const void* data, size_t remaining) {
    const auto* bytes = static_cast<const char*>(data);
    while (remaining) {
      Verify();  // Prior files BEFORE any new advice, even while normal writes.
      const auto count =
          write(file.fd, bytes, std::min<size_t>(remaining, 1048576));
      if (count < 0 && errno == EINTR) continue;
      Require(count > 0, "published GEO write failed");
      bytes += count;
      remaining -= count;
    }
  };
  write_all(header.data(), header.size());
  write_all(map.GetPtr(), map.GetNumBytes());
  const auto identity = Identity(file.fd, path);
  Require(identity.bytes == header.size() + map.GetNumBytes(),
          "published GEO size mismatch");
  Require(fdatasync(file.fd) == 0, "published GEO sync failed");
  Verify();  // Never mask a reader violation by advising previous files.
  const auto rounded =
      ((identity.bytes + kPageBytes - 1) / kPageBytes) * kPageBytes;
  Require(
      rounded <= uint64_t(std::numeric_limits<off_t>::max()) &&
          posix_fadvise(
              file.fd, 0, static_cast<off_t>(rounded), POSIX_FADV_DONTNEED) ==
              0,
      "published GEO advice failed");
  file.Close();
  VerifyFile(identity);
  return identity;
#else
  (void)path;
  throw std::runtime_error("published GEO requires Linux");
#endif
}
void PublishedGeoWriter::Publish(size_t job,
                                 const Mat<float>& depth,
                                 const Mat<float>& normal) {
  Verify();
  Require(job == products_.size() && job < references_.size(),
          "published GEO caller order/count changed");
  Require(depth.GetDepth() == 1 && normal.GetDepth() == 3 &&
              depth.GetWidth() == normal.GetWidth() &&
              depth.GetHeight() == normal.GetHeight(),
          "published GEO pair shape mismatch");
  ValidateMat(depth);
  ValidateMat(normal);
  const auto name = references_[job] + ".geometric.bin";
  const auto d = Write(root_ / "stereo" / "depth_maps" / name, depth);
  files_.push_back(d);
  const auto n = Write(root_ / "stereo" / "normal_maps" / name, normal);
  files_.push_back(n);
  Verify();
  products_.push_back({d, n});
}
void PublishedGeoWriter::VerifyComplete() const {
  Verify();
  Require(products_.size() == references_.size() &&
              files_.size() == 2 * references_.size(),
          "published GEO output incomplete");
}
void PublishedGeoWriter::QualifyCache() {
  Verify();
  if (!cache_qualified_) {
    Probe(root_.parent_path(), cancel_);
    cache_qualified_ = true;
  }
}
void PublishedGeoWriter::Probe(const std::filesystem::path& directory,
                               const std::atomic<bool>* cancel) {
#ifdef __linux__
  auto pattern = (directory / ".published-geo-probe-XXXXXX").string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back(0);
  Require(mkdtemp(name.data()) != nullptr, "published GEO probe create failed");
  const std::filesystem::path scratch(name.data());
  try {
    PublishedGeoWriter writer(scratch / "cohort", {"probe"}, cancel);
    Mat<float> depth(257, 1, 1), normal(257, 1, 3);
    writer.Publish(0, depth, normal);
    writer.VerifyComplete();
    std::filesystem::remove_all(scratch);
  } catch (...) {
    std::filesystem::remove_all(scratch);
    throw;
  }
#else
  (void)directory;
  (void)cancel;
  throw std::runtime_error("published GEO requires Linux");
#endif
}
}  // namespace colmap::mvs::owned
