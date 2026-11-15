// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/published_geo.h"

#include <atomic>
#include <cassert>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace colmap::mvs;
using namespace colmap::mvs::owned;
namespace {
template <class F>
void Refuses(F action) {
  bool refused = false;
  try {
    action();
  } catch (const std::exception&) {
    refused = true;
  }
  assert(refused);
}
#ifdef PUBLISHED_GEO_WRAP_IO
int fail_sync = 0, fail_close = 0, fail_advice = 0, fail_write = 0;
extern "C" int __real_fdatasync(int);
extern "C" int __real_close(int);
extern "C" int __real_posix_fadvise(int, off_t, off_t, int);
extern "C" ssize_t __real_write(int, const void*, size_t);
extern "C" int __wrap_fdatasync(int fd) {
  if (fail_sync) {
    errno = EIO;
    return -1;
  }
  return __real_fdatasync(fd);
}
extern "C" int __wrap_close(int fd) {
  const int result = __real_close(fd);
  if (fail_close) {
    fail_close = 0;
    errno = EIO;
    return -1;
  }
  return result;
}
extern "C" int __wrap_posix_fadvise(int fd,
                                    off_t offset,
                                    off_t count,
                                    int advice) {
  if (fail_advice) return ENOSYS;
  return __real_posix_fadvise(fd, offset, count, advice);
}
extern "C" ssize_t __wrap_write(int fd, const void* data, size_t count) {
  if (fail_write) {
    errno = EIO;
    return -1;
  }
  return __real_write(fd, data, count);
}
#endif
}  // namespace
int main() {
  const auto root = std::filesystem::temp_directory_path() /
                    ("published-geo-test-" + std::to_string(getpid()));
  std::filesystem::remove_all(root);
  Mat<float> depth(3, 1, 1), normal(3, 1, 3);
  depth.Fill(1.0f);
  normal.Fill(-2.0f);
  {
    PublishedGeoWriter writer(root, {"nested/image.jpg"});
    writer.Publish(0, depth, normal);
    writer.VerifyComplete();
    assert(writer.Products().size() == 1);
    const auto path = writer.Products()[0].depth.path;
    std::ifstream input(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(input)), {});
    const std::string expected =
        std::string("3&1&1&") +
        std::string("\0\0\x80\x3f\0\0\x80\x3f\0\0\x80\x3f", 12);
    assert(bytes == expected);
    // A body reader is a contract violation, NOT an occasion to re-evict.
    bool refused = false;
    try {
      writer.VerifyComplete();
    } catch (const std::runtime_error&) {
      refused = true;
    }
    assert(refused);
  }
#ifdef PUBLISHED_GEO_NATIVE_MAT
  // Actual legacy API oracle, after cohort completion (body reads now allowed).
  const auto legacy_depth = root / "legacy-depth.bin";
  const auto legacy_normal = root / "legacy-normal.bin";
  depth.Write(legacy_depth);
  normal.Write(legacy_normal);
  const auto bytes = [](const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), {});
  };
  assert(bytes(legacy_depth) ==
         bytes(root / "stereo/depth_maps/nested/image.jpg.geometric.bin"));
  assert(bytes(legacy_normal) ==
         bytes(root / "stereo/normal_maps/nested/image.jpg.geometric.bin"));
#endif
  // Ordinary filesystem failures and identity changes use the REAL writer.
  Refuses([&] { PublishedGeoWriter duplicate(root, {"other"}); });
  Refuses(
      [&] { PublishedGeoWriter missing(root / "absent" / "child", {"x"}); });
  Refuses([&] { PublishedGeoWriter invalid(root / "invalid", {"../escape"}); });
  PublishedGeoWriter ordered(root / "ordered", {"b", "a"});
  Refuses([&] { ordered.Publish(1, depth, normal); });
  Refuses([&] { ordered.VerifyComplete(); });
  ordered.Publish(0, depth, normal);
  ordered.Publish(1, depth, normal);
  ordered.VerifyComplete();
  const auto first = ordered.Products()[0].depth.path;
  const auto alias = root / "alias";
  assert(link(first.c_str(), alias.c_str()) == 0);
  Refuses([&] { ordered.Verify(); });
  assert(unlink(alias.c_str()) == 0);
  // ctime is also bound: removing an illicit hardlink does not repair identity.
  Refuses([&] { ordered.Verify(); });
  std::atomic<bool> cancelled{false};
  PublishedGeoWriter cancelled_writer(root / "cancelled", {"x"}, &cancelled);
  cancelled.store(true);
  Refuses([&] { cancelled_writer.Publish(0, depth, normal); });
  Refuses([&] { cancelled_writer.VerifyComplete(); });
  PublishedGeoWriter prior_read(root / "prior-read", {"first", "second"});
  prior_read.Publish(0, depth, normal);
  std::ifstream early(prior_read.Products()[0].depth.path, std::ios::binary);
  assert(early.get() == '3');
  Refuses([&] { prior_read.Publish(1, depth, normal); });
  assert(prior_read.Products().size() == 1);
  assert(!std::filesystem::exists(
      root / "prior-read/stereo/depth_maps/second.geometric.bin"));
  PublishedGeoWriter collision(root / "collision", {"x"});
  std::filesystem::create_directories(root / "collision/stereo/depth_maps");
  const auto collision_path =
      root / "collision/stereo/depth_maps/x.geometric.bin";
  std::ofstream(collision_path) << "preserved";
  Refuses([&] { collision.Publish(0, depth, normal); });
  assert(std::filesystem::file_size(collision_path) == 9);
  PublishedGeoWriter symlinked(root / "symlinked", {"x"});
  std::filesystem::create_directories(root / "symlinked/stereo/depth_maps");
  std::filesystem::create_symlink(
      collision_path, root / "symlinked/stereo/depth_maps/x.geometric.bin");
  Refuses([&] { symlinked.Publish(0, depth, normal); });
  PublishedGeoWriter replaced(root / "replaced", {"x"});
  replaced.Publish(0, depth, normal);
  const auto replaced_path = replaced.Products()[0].normal.path;
  std::filesystem::remove(replaced_path);
  std::ofstream(replaced_path) << "replacement";
  Refuses([&] { replaced.VerifyComplete(); });
  // mlock negative, only skipped if ordinary RLIMIT/capability forbids pinning.
  PublishedGeoWriter pinned(root / "pinned", {"x"});
  pinned.Publish(0, depth, normal);
  int fd = open(pinned.Products()[0].depth.path.c_str(), O_RDONLY);
  assert(fd >= 0);
  void* page = mmap(nullptr, 4096, PROT_READ, MAP_SHARED, fd, 0);
  assert(page != MAP_FAILED);
  if (mlock(page, 4096) == 0) {
    assert(posix_fadvise(fd, 0, 4096, POSIX_FADV_DONTNEED) == 0);
    Refuses([&] { pinned.VerifyComplete(); });
    assert(munlock(page, 4096) == 0);
  } else {
    std::cout << "mlock negative unavailable: " << errno
              << " (not qualified)\n";
  }
  assert(munmap(page, 4096) == 0 && close(fd) == 0);
  PublishedGeoWriter::Probe(root);
  // Actual RLIMIT_FSIZE failure, no process-global limit/signal mutation.
  const pid_t child = fork();
  assert(child >= 0);
  if (!child) {
    signal(SIGXFSZ, SIG_IGN);
    struct rlimit limit {
      8, 8
    };
    if (setrlimit(RLIMIT_FSIZE, &limit)) _exit(2);
    PublishedGeoWriter limited(root / "limited", {"x"});
    try {
      limited.Publish(0, depth, normal);
    } catch (...) {
      _exit(0);
    }
    _exit(1);
  }
  int status = 0;
  assert(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
         WEXITSTATUS(status) == 0);
#ifdef PUBLISHED_GEO_WRAP_IO
  for (int failure = 0; failure < 4; ++failure) {
    PublishedGeoWriter failed(root / ("failed-" + std::to_string(failure)),
                              {"x"});
    fail_write = failure == 0;
    fail_sync = failure == 1;
    fail_advice = failure == 2;
    fail_close = failure == 3;
    Refuses([&] { failed.Publish(0, depth, normal); });
    fail_write = fail_sync = fail_advice = fail_close = 0;
    Refuses([&] { failed.VerifyComplete(); });
    assert(failed.Products().empty());
  }
#endif
  std::filesystem::remove_all(root);
  std::cout << "published GEO exact bytes / owned reader detection passed\n";
}
