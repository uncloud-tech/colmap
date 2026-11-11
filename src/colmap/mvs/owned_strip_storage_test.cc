// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_strip_storage.h"

#include <cassert>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
#ifdef __linux__
#include <sys/resource.h>
#endif

#ifdef STRIP_STORAGE_WRAP_IO
#include <unistd.h>
static size_t io_chunk = SIZE_MAX;
static bool io_eof = false;
static int io_error = 0;
static std::atomic<bool>* cancel_after_io = nullptr;
extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" ssize_t __real_pwrite(int, const void*, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void* data, size_t size, off_t offset) {
  if (io_eof) return 0;
  if (io_error) {
    errno = std::exchange(io_error, 0);
    return -1;
  }
  const auto n = __real_pread(fd, data, std::min(size, io_chunk), offset);
  if (cancel_after_io) cancel_after_io->store(true);
  return n;
}
extern "C" ssize_t __wrap_pwrite(int fd,
                                 const void* data,
                                 size_t size,
                                 off_t offset) {
  if (io_error) {
    errno = std::exchange(io_error, 0);
    return -1;
  }
  const auto n = __real_pwrite(fd, data, std::min(size, io_chunk), offset);
  if (cancel_after_io) cancel_after_io->store(true);
  return n;
}
#endif

using namespace colmap::mvs::owned;
namespace {
template <class F>
void Refuses(F f) {
  bool refused = false;
  try {
    f();
  } catch (const std::exception&) {
    refused = true;
  }
  assert(refused);
}
size_t Files() {
  return std::distance(std::filesystem::directory_iterator("."),
                       std::filesystem::directory_iterator{});
}
}  // namespace
int main() {
#ifndef __linux__
  std::cout << "disk strip backend requires Linux\n";
  return 77;
#else
  const auto staging = BuildStripStagingLayout(101, 135, 3, 32, true);
  assert(staging.host_bytes == 52380);
  assert(staging.device_bytes == 168480);
  assert(BuildStripStagingLayout(3, 2, 1, 32, false).values == 9);
  Refuses([] { BuildStripStagingLayout(0, 2, 1, 32, true); });
  Refuses([] {
    BuildStripStagingLayout(INT32_MAX, INT32_MAX, INT32_MAX, 32, true);
  });
  Refuses([] { StripCheckedProduct(UINT64_MAX, 2); });
  for (size_t block : {8, 16, 32}) {
    ValidateStripColumns(block, block);
    ValidateStripColumns(3 * block, block);
    Refuses([&] { ValidateStripColumns(block + 1, block); });
  }
  Refuses([] { ValidateStripColumns(0, 32); });
  Refuses([] { ValidateStripColumns(32, 7); });
  const size_t files = Files();
  const auto descriptors = [] {
    return std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                         std::filesystem::directory_iterator{});
  };
  const auto initial_descriptors = descriptors();
  {
    // Independent literal CudaRotate mapping: canonical [1 2 3;4 5 6].
    StripFileStore store(".", 3, 2, 1, true, 54);
    const float canonical[] = {1, 2, 3, 4, 5, 6};
    store.Write(StripPlane::kCost, 0, 0, 3, canonical);
    const std::vector<std::vector<float> > expected = {{1, 2, 3, 4, 5, 6},
                                                       {3, 6, 2, 5, 1, 4},
                                                       {6, 5, 4, 3, 2, 1},
                                                       {4, 1, 5, 2, 6, 3}};
    for (unsigned orientation = 0; orientation < 4; ++orientation) {
      float result[6] = {};
      store.Read(
          StripPlane::kCost, orientation, 0, orientation % 2 ? 2 : 3, result);
      assert(std::memcmp(
                 result, expected[orientation].data(), sizeof(result)) == 0);
    }
    assert(store.Bytes() == 54);
    assert(store.BytesWritten() == 24);
    assert(store.BytesRead() == 96);
    assert(store.RowBufferBytes() == 12);
    assert(Files() == files);  // File is unlinked while still actively used.
    Refuses([&] { store.Read(StripPlane::kCost, 4, 0, 1, nullptr); });
    Refuses([&] { store.Read(StripPlane::kCost, 0, 3, 1, nullptr); });
  }
#ifdef STRIP_STORAGE_WRAP_IO
  {
    StripFileStore store(".", 3, 2, 1, false, 48);
    const float input[] = {1, 2, 3, 4, 5, 6};
    float output[6] = {};
    // Real file syscalls limited to 3 bytes, including split float elements.
    io_chunk = 3;
    store.Write(StripPlane::kCost, 0, 0, 3, input);
    store.Read(StripPlane::kCost, 0, 0, 3, output);
    assert(std::memcmp(input, output, sizeof(input)) == 0);
    io_chunk = SIZE_MAX;
    io_eof = true;
    Refuses([&] { store.Read(StripPlane::kCost, 0, 0, 3, output); });
    io_eof = false;
    io_error = EINTR;
    store.Write(StripPlane::kCost, 0, 0, 3, input);
    io_error = EINTR;
    store.Read(StripPlane::kCost, 0, 0, 3, output);
    assert(std::memcmp(input, output, sizeof(input)) == 0);
    io_error = EIO;
    Refuses([&] { store.Write(StripPlane::kCost, 0, 0, 3, input); });
    io_error = EIO;
    Refuses([&] { store.Read(StripPlane::kCost, 0, 0, 3, output); });
  }
  {
    std::atomic<bool> stop{false};
    StripFileStore store(".", 3, 2, 1, false, 48, &stop);
    float tile[6] = {};
    io_chunk = 3;
    cancel_after_io = &stop;
    Refuses([&] { store.Write(StripPlane::kCost, 0, 0, 3, tile); });
    assert(store.BytesWritten() == 3);
    stop = false;
    Refuses([&] { store.Read(StripPlane::kCost, 0, 0, 3, tile); });
    assert(store.BytesRead() == 3);
    cancel_after_io = nullptr;
    io_chunk = SIZE_MAX;
  }
#endif
  // More than three aligned strips plus a tail in both orientations. Write
  // each canonical value through a rotated ragged tile, then read canonically.
  for (size_t block : {8, 16, 32}) {
    const size_t w = 3 * block + 5, h = 4 * block + 7, sources = 3;
    const size_t values = w * h * sources;
    StripFileStore store(".", w, h, sources, true, 9 * values);
    for (unsigned orientation = 0; orientation < 4; ++orientation) {
      const size_t ow = orientation % 2 ? h : w, oh = orientation % 2 ? w : h;
      std::set<std::pair<size_t, size_t> > seen;
      for (size_t first = 0; first < ow; first += block) {
        const size_t columns = std::min(block, ow - first);
        std::vector<float> tile(columns * oh * sources);
        std::vector<uint8_t> mask(tile.size());
        for (size_t r = 0; r < oh; ++r) {
          for (size_t c = 0; c < columns; ++c) {
            const auto p = StripCanonicalPixel(w, h, orientation, r, first + c);
            assert(seen.insert(p).second);
            for (size_t s = 0; s < sources; ++s) {
              const auto i = (s * oh + r) * columns + c;
              tile[i] = float((s * h + p.first) * w + p.second + 1);
              mask[i] = uint8_t((p.first + p.second + s) % 2);
            }
          }
        }
        store.Write(
            StripPlane::kSelection, orientation, first, columns, tile.data());
        store.Write(
            StripPlane::kMask, orientation, first, columns, mask.data());
        std::vector<float> reload(tile.size());
        store.Read(
            StripPlane::kSelection, orientation, first, columns, reload.data());
        assert(tile == reload);
      }
      assert(seen.size() == w * h);
      std::vector<float> canonical(values);
      std::vector<uint8_t> mask(values);
      store.Read(StripPlane::kSelection, 0, 0, w, canonical.data());
      store.Read(StripPlane::kMask, 0, 0, w, mask.data());
      for (size_t s = 0; s < sources; ++s)
        for (size_t r = 0; r < h; ++r)
          for (size_t c = 0; c < w; ++c) {
            const auto i = (s * h + r) * w + c;
            assert(canonical[i] == float(i + 1));
            assert(mask[i] == uint8_t((r + c + s) % 2));
          }
    }
    // Commit only one pixel; unrelated regions and the other plane survive.
    const float changed[] = {-1, -2, -3};
    // Full-height single-column tile: change all rows only in canonical col0.
    std::vector<float> column(h * sources, -1);
    store.Write(StripPlane::kSelection, 0, 0, 1, column.data());
    std::vector<float> result(values);
    store.Read(StripPlane::kSelection, 0, 0, w, result.data());
    assert(result[0] == changed[0]);
    assert(result[1] == 2);
    assert(result[w + 1] == float(w + 2));
  }
  Refuses([] { StripFileStore bad(".", 3, 2, 1, true, 53); });
  Refuses([] { StripFileStore bad(".", 0, 2, 1, false, 100); });
  Refuses([] {
    StripFileStore bad("missing-strip-directory", 3, 2, 1, false, 100);
  });
  std::atomic<bool> cancelled{true};
  Refuses([&] { StripFileStore bad(".", 3, 2, 1, false, 100, &cancelled); });
  cancelled = false;
  {
    StripFileStore store(".", 3, 2, 1, false, 100, &cancelled);
    float tile[6] = {};
    cancelled = true;
    Refuses([&] { store.Write(StripPlane::kCost, 0, 0, 3, tile); });
    Refuses([&] { store.Read(StripPlane::kCost, 0, 0, 3, tile); });
    cancelled = false;
    Refuses([&] { store.Read(StripPlane::kMask, 0, 0, 3, tile); });
  }
  // Actual kernel-enforced file-size exhaustion, not a mock counter/OOM.
  struct rlimit original;
  assert(getrlimit(RLIMIT_FSIZE, &original) == 0);
  auto limited = original;
  limited.rlim_cur = 0;
  const auto old_signal = std::signal(SIGXFSZ, SIG_IGN);
  assert(setrlimit(RLIMIT_FSIZE, &limited) == 0);
  Refuses([] { StripFileStore bad(".", 3, 2, 1, false, 100); });
  assert(setrlimit(RLIMIT_FSIZE, &original) == 0);
  std::signal(SIGXFSZ, old_signal);
#ifdef __linux__
  if (std::filesystem::is_directory("/dev/shm"))
    Refuses([] { StripFileStore bad("/dev/shm", 3, 2, 1, false, 100); });
#endif
  assert(Files() == files);
  assert(descriptors() == initial_descriptors);
  std::cout
      << "strip storage: real spill/reload, 4 orientations, ragged strips, "
         "bounds, exhaustion, cancellation and unlink passed\n";
#endif
}
