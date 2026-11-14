// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Linux-only, test-only post-open failure probe for the terminal PLY/VIS
// writers.
#if defined(__linux__)
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <exception>
#include <filesystem>

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace colmap {
namespace terminal_writer_test {

// Only the forked child gets the resource limit. Exit 10 means the actual
// writer silently returned despite a verified EFBIG; 11 means it reported an
// exception. All other statuses are harness failures, NOT writer RED evidence.
template <typename Writer>
void ExpectFileSizeFailure(const std::filesystem::path& path,
                           const rlim_t limit,
                           Writer writer) {
  const pid_t pid = fork();
  ASSERT_NE(pid, -1);
  if (pid == 0) {
    alarm(15);
    struct rlimit old_limit;
    if (getrlimit(RLIMIT_FSIZE, &old_limit) != 0 ||
        limit > old_limit.rlim_max || signal(SIGXFSZ, SIG_IGN) == SIG_ERR) {
      _exit(70);
    }
    const struct rlimit child_limit = {limit, old_limit.rlim_max};
    if (setrlimit(RLIMIT_FSIZE, &child_limit) != 0) _exit(71);
    bool threw = false;
    try {
      writer();
    } catch (const std::exception& error) {
      threw = true;
      dprintf(STDERR_FILENO,
              "writer exception path=%s: %s\n",
              path.c_str(),
              error.what());
    } catch (...) {
      _exit(72);
    }
    // Independently verify that the writer reached the limit and the kernel
    // rejects a subsequent real append. A setup/open failure cannot pass this.
    std::error_code ec;
    if (std::filesystem::file_size(path, ec) != limit || ec) _exit(73);
    const int fd = open(path.c_str(), O_WRONLY | O_APPEND);
    if (fd < 0) _exit(74);
    errno = 0;
    const ssize_t result = write(fd, "!", 1);
    const int error = errno;
    close(fd);
    if (result != -1 || error != EFBIG) _exit(75);
    dprintf(STDERR_FILENO,
            "verified post-open EFBIG path=%s limit=%llu writer=%s\n",
            path.c_str(),
            static_cast<unsigned long long>(limit),
            threw ? "THREW" : "SILENT_RETURN");
    _exit(threw ? 11 : 10);
  }
  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(pid, &status, 0);
  } while (waited == -1 && errno == EINTR);
  ASSERT_EQ(waited, pid);
  ASSERT_TRUE(WIFEXITED(status)) << "raw wait status=" << status;
  EXPECT_EQ(WEXITSTATUS(status), 11)
      << "10 = verified silent-return RED; 70..75 = invalid probe";
}

}  // namespace terminal_writer_test
}  // namespace colmap
#endif  // defined(__linux__)
