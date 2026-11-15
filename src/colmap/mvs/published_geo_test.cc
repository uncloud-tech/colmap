// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/published_geo.h"

#include <atomic>
#include <cassert>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace colmap::mvs;
using namespace colmap::mvs::owned;
namespace {
// Metadata only: diagnostics must not refault bodies or repair cache state.
void PathState(const char* phase, const std::filesystem::path& path) {
  struct stat info{};
  const int result = lstat(path.c_str(), &info);
  const int error = result ? errno : 0;
  std::cerr << "[published-geo refusal] " << phase << " path=" << path
            << " lstat=" << result << " errno=" << error;
  if (!result) {
    std::cerr << " dev=" << info.st_dev << " ino=" << info.st_ino
              << " uid=" << info.st_uid << " mode=" << info.st_mode
              << " nlink=" << info.st_nlink << " bytes=" << info.st_size
              << " mtime=" << info.st_mtim.tv_sec << ':' << info.st_mtim.tv_nsec
              << " ctime=" << info.st_ctim.tv_sec << ':'
              << info.st_ctim.tv_nsec;
  }
  std::cerr << '\n';
}
template <class F>
void Refuses(const char* name,
             const char* expected,
             const std::filesystem::path& path,
             F action,
             const PublishedGeoFile* published = nullptr) {
  bool refused = false;
  try {
    action();
  } catch (const std::exception&) {
    refused = true;
  }
  // Only diagnose after failure: successful checks must not gain logging/stat
  // delays between publication, link/unlink, and identity verification.
  if (!refused) {
    std::cerr << "[published-geo refusal] FAILED case=" << name
              << ": action returned normally; expected std::exception ("
              << expected << ")\n";
    PathState("after unexpected return", path);
    PathState("parent", path.parent_path());
    if (published) {
      std::cerr << "[published-geo refusal] recorded dev=" << published->device
                << " ino=" << published->inode << " bytes=" << published->bytes
                << " mtime=" << published->modified_seconds << ':'
                << published->modified_nanoseconds
                << " ctime=" << published->changed_seconds << ':'
                << published->changed_nanoseconds << '\n';
    }
  }
  assert(refused);
}
void RemainsFailed(PublishedGeoWriter& writer,
                   const std::filesystem::path& path,
                   const Mat<float>& depth,
                   const Mat<float>& normal) {
  const char* names[] = {"failed-attempt-verify",
                         "failed-attempt-complete",
                         "failed-attempt-publish",
                         "failed-attempt-qualify"};
  for (int operation = 0; operation < 4; ++operation) {
    Refuses(names[operation], "attempt previously failed", path, [&] {
      try {
        switch (operation) {
          case 0:
            writer.Verify();
            break;
          case 1:
            writer.VerifyComplete();
            break;
          case 2:
            writer.Publish(writer.Products().size(), depth, normal);
            break;
          case 3:
            writer.QualifyCache();
            break;
        }
      } catch (const std::runtime_error& error) {
        assert(std::string(error.what()) ==
               "published GEO attempt previously failed");
        throw;
      }
    });
  }
}
#ifdef PUBLISHED_GEO_WRAP_IO
int fail_sync = 0, fail_close = 0, fail_advice = 0, fail_write = 0;
// Inject only the timestamp observation seen in CI, on this owned inode.
// Actual files, link counts, I/O, residency, and all other metadata stay real.
const PublishedGeoFile* identical_ctime = nullptr;
extern "C" int __real_fstat(int, struct stat*);
extern "C" int __wrap_fstat(int fd, struct stat* info) {
  const int result = __real_fstat(fd, info);
  if (!result && identical_ctime &&
      uint64_t(info->st_dev) == identical_ctime->device &&
      uint64_t(info->st_ino) == identical_ctime->inode) {
    info->st_ctim.tv_sec = identical_ctime->changed_seconds;
    info->st_ctim.tv_nsec = identical_ctime->changed_nanoseconds;
  }
  return result;
}
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
  struct statfs filesystem{};
  assert(statfs(root.parent_path().c_str(), &filesystem) == 0);
  std::cerr << "[published-geo refusal] root=" << root << " uid=" << getuid()
            << " euid=" << geteuid() << " page_bytes=" << sysconf(_SC_PAGESIZE)
            << " fs_type=" << filesystem.f_type
            << " fs_block_bytes=" << filesystem.f_bsize << '\n';
  PathState("temporary parent", root.parent_path());
  Mat<float> depth(3, 1, 1), normal(3, 1, 3);
  depth.Fill(1.0f);
  normal.Fill(-2.0f);
#ifdef PUBLISHED_GEO_WRAP_IO
  {
    PublishedGeoWriter observed(root, {"x"});
    observed.Publish(0, depth, normal);
    const auto published = observed.Products()[0].depth;
    identical_ctime = &published;
    const auto alias = root / "alias";
    assert(link(published.path.c_str(), alias.c_str()) == 0);
    Refuses(
        "observed-hardlink", "identity/ownership invalid", published.path, [&] {
          try {
            observed.Verify();
          } catch (const std::runtime_error& error) {
            assert(std::string(error.what()) ==
                   "published GEO file identity/ownership invalid");
            throw;
          }
        });
    assert(unlink(alias.c_str()) == 0);
    std::cerr
        << "[published-geo refusal] injected observation: identical ctime "
           "on owned inode after real link/unlink\n";
    Refuses(
        "observed-hardlink-undone",
        "previous verification failure is final",
        published.path,
        [&] { observed.Verify(); },
        &published);
    RemainsFailed(observed, published.path, depth, normal);
    identical_ctime = nullptr;
  }
  std::filesystem::remove_all(root);
#endif
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
  // Mat's constructor historically does unchecked products; publication must
  // not turn a wrapped allocation into a success-shaped malformed map file.
  const auto wrapped_width = std::numeric_limits<size_t>::max() / 2 + 2;
  Mat<float> malformed_depth(wrapped_width, 2, 1);
  Mat<float> malformed_normal(wrapped_width, 2, 3);
  PublishedGeoWriter malformed(root / "malformed", {"x"});
  Refuses("wrapped-mat-publish", "invalid dimensions", root / "malformed", [&] {
    malformed.Publish(0, malformed_depth, malformed_normal);
  });
  Refuses(
      "wrapped-mat-incomplete", "output incomplete", root / "malformed", [&] {
        malformed.VerifyComplete();
      });
  // Rejected arguments and incomplete queries do not invalidate an attempt.
  malformed.Verify();
  malformed.Publish(0, depth, normal);
  malformed.VerifyComplete();
  // Ordinary filesystem failures and identity changes use the REAL writer.
  Refuses("duplicate-root", "fresh root required", root, [&] {
    PublishedGeoWriter duplicate(root, {"other"});
  });
  Refuses("missing-parent", "parent missing", root / "absent", [&] {
    PublishedGeoWriter missing(root / "absent" / "child", {"x"});
  });
  Refuses("escaping-reference", "invalid component", root / "invalid", [&] {
    PublishedGeoWriter invalid(root / "invalid", {"../escape"});
  });
  PublishedGeoWriter ordered(root / "ordered", {"b", "a"});
  Refuses("out-of-order", "caller order/count changed", root / "ordered", [&] {
    ordered.Publish(1, depth, normal);
  });
  Refuses("wrong-pair-shape", "pair shape mismatch", root / "ordered", [&] {
    ordered.Publish(0, depth, depth);
  });
  Refuses("ordered-incomplete", "output incomplete", root / "ordered", [&] {
    ordered.VerifyComplete();
  });
  ordered.Publish(0, depth, normal);
  Refuses(
      "partially-complete-query", "output incomplete", root / "ordered", [&] {
        ordered.VerifyComplete();
      });
  ordered.Verify();
  ordered.Publish(1, depth, normal);
  ordered.VerifyComplete();
  const auto first = ordered.Products()[0].depth.path;
  const auto alias = root / "alias";
  assert(link(first.c_str(), alias.c_str()) == 0);
  Refuses("hardlink-present", "identity/ownership invalid", first, [&] {
    ordered.Verify();
  });
  assert(unlink(alias.c_str()) == 0);
  // An observed hardlink violation is final, even when ctime aliases after
  // unlink.
  Refuses(
      "hardlink-removed",
      "observed hardlink failure is final after unlink",
      first,
      [&] { ordered.Verify(); },
      &ordered.Products()[0].depth);
  std::atomic<bool> cancelled{false};
  PublishedGeoWriter cancelled_writer(root / "cancelled", {"x"}, &cancelled);
  cancelled.store(true);
  Refuses("cancelled-publish", "cancelled", root / "cancelled", [&] {
    cancelled_writer.Publish(0, depth, normal);
  });
  Refuses("cancelled-complete", "cancelled", root / "cancelled", [&] {
    cancelled_writer.VerifyComplete();
  });
  PublishedGeoWriter prior_read(root / "prior-read", {"first", "second"});
  prior_read.Publish(0, depth, normal);
  std::ifstream early(prior_read.Products()[0].depth.path, std::ios::binary);
  assert(early.get() == '3');
  Refuses("prior-body-read",
          "cache residency nonzero",
          prior_read.Products()[0].depth.path,
          [&] { prior_read.Publish(1, depth, normal); });
  assert(prior_read.Products().size() == 1);
  assert(!std::filesystem::exists(
      root / "prior-read/stereo/depth_maps/second.geometric.bin"));
  PublishedGeoWriter collision(root / "collision", {"x"});
  std::filesystem::create_directories(root / "collision/stereo/depth_maps");
  const auto collision_path =
      root / "collision/stereo/depth_maps/x.geometric.bin";
  std::ofstream(collision_path) << "preserved";
  Refuses("existing-file", "exclusive create failed", collision_path, [&] {
    collision.Publish(0, depth, normal);
  });
  assert(std::filesystem::file_size(collision_path) == 9);
  PublishedGeoWriter symlinked(root / "symlinked", {"x"});
  std::filesystem::create_directories(root / "symlinked/stereo/depth_maps");
  std::filesystem::create_symlink(
      collision_path, root / "symlinked/stereo/depth_maps/x.geometric.bin");
  Refuses("symlink-file",
          "exclusive create failed",
          root / "symlinked/stereo/depth_maps/x.geometric.bin",
          [&] { symlinked.Publish(0, depth, normal); });
  PublishedGeoWriter replaced(root / "replaced", {"x"});
  replaced.Publish(0, depth, normal);
  const auto replaced_path = replaced.Products()[0].normal.path;
  std::filesystem::remove(replaced_path);
  std::ofstream(replaced_path) << "replacement";
  Refuses("replaced-file", "identity changed", replaced_path, [&] {
    replaced.VerifyComplete();
  });
  // mlock negative, only skipped if ordinary RLIMIT/capability forbids pinning.
  PublishedGeoWriter pinned(root / "pinned", {"x"});
  pinned.Publish(0, depth, normal);
  int fd = open(pinned.Products()[0].depth.path.c_str(), O_RDONLY);
  assert(fd >= 0);
  void* page = mmap(nullptr, 4096, PROT_READ, MAP_SHARED, fd, 0);
  assert(page != MAP_FAILED);
  if (mlock(page, 4096) == 0) {
    assert(posix_fadvise(fd, 0, 4096, POSIX_FADV_DONTNEED) == 0);
    Refuses("pinned-page",
            "cache residency nonzero",
            pinned.Products()[0].depth.path,
            [&] { pinned.VerifyComplete(); });
    assert(munlock(page, 4096) == 0);
  } else {
    std::cout << "mlock negative unavailable: " << errno
              << " (not qualified)\n";
  }
  assert(munmap(page, 4096) == 0 && close(fd) == 0);
  PublishedGeoWriter::Probe(root);
  // A deliberate reader violation cannot be repaired by external cache advice.
  PublishedGeoWriter read_reset(root / "read-reset", {"x", "y"});
  read_reset.Publish(0, depth, normal);
  const auto read_path = read_reset.Products()[0].depth.path;
  fd = open(read_path.c_str(), O_RDONLY);
  assert(fd >= 0);
  char byte = 0;
  assert(pread(fd, &byte, 1, 0) == 1 && byte == '3');
  Refuses("observed-cache-read", "cache residency nonzero", read_path, [&] {
    try {
      read_reset.Verify();
    } catch (const std::runtime_error& error) {
      assert(std::string(error.what()) ==
             "published GEO cache residency is nonzero");
      throw;
    }
  });
  assert(posix_fadvise(fd, 0, 4096, POSIX_FADV_DONTNEED) == 0);
  page = mmap(nullptr, 4096, PROT_NONE, MAP_SHARED, fd, 0);
  assert(page != MAP_FAILED);
  unsigned char resident = 1;
  assert(mincore(page, 4096, &resident) == 0 && !(resident & 1));
  assert(munmap(page, 4096) == 0 && close(fd) == 0);
  RemainsFailed(read_reset, read_path, depth, normal);
  assert(read_reset.Products().size() == 1);
  assert(!std::filesystem::exists(
      root / "read-reset/stereo/depth_maps/y.geometric.bin"));
  PublishedGeoWriter namespace_reset(root / "namespace-reset", {"x"});
  assert(chmod((root / "namespace-reset").c_str(), 0755) == 0);
  Refuses("observed-namespace-permission",
          "namespace no longer private",
          root / "namespace-reset",
          [&] { namespace_reset.Verify(); });
  assert(chmod((root / "namespace-reset").c_str(), 0700) == 0);
  RemainsFailed(namespace_reset, root / "namespace-reset", depth, normal);
  // Actual RLIMIT_FSIZE failure, no process-global limit/signal mutation.
  const pid_t child = fork();
  assert(child >= 0);
  if (!child) {
    signal(SIGXFSZ, SIG_IGN);
    struct rlimit limit{8, 8};
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
  PublishedGeoWriter probe_failure(root / "probe-failure", {"x"});
  fail_sync = 1;
  Refuses("cache-probe-io-failure",
          "probe sync failed",
          root / "probe-failure",
          [&] { probe_failure.QualifyCache(); });
  fail_sync = 0;
  RemainsFailed(probe_failure, root / "probe-failure", depth, normal);
  const char* failure_names[] = {
      "write-error", "sync-error", "advice-error", "close-error"};
  for (int failure = 0; failure < 4; ++failure) {
    PublishedGeoWriter failed(root / ("failed-" + std::to_string(failure)),
                              {"x"});
    fail_write = failure == 0;
    fail_sync = failure == 1;
    fail_advice = failure == 2;
    fail_close = failure == 3;
    Refuses(failure_names[failure],
            "injected I/O error",
            root / ("failed-" + std::to_string(failure)),
            [&] { failed.Publish(0, depth, normal); });
    fail_write = fail_sync = fail_advice = fail_close = 0;
    RemainsFailed(
        failed, root / ("failed-" + std::to_string(failure)), depth, normal);
    Refuses(failure_names[failure],
            "output incomplete after injected I/O error",
            root / ("failed-" + std::to_string(failure)),
            [&] { failed.VerifyComplete(); });
    assert(failed.Products().empty());
  }
#endif
  std::filesystem::remove_all(root);
  std::cout << "published GEO exact bytes / owned reader detection passed\n";
}
