# SPDX-License-Identifier: BSD-3-Clause
"""Bounded CPU tests of SIFT file-plane and stripe admission, not CUDA."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class StreamedFilterTest(unittest.TestCase):
    def test_sampler_texture_limits_cuda_and_hip(self):
        # Compile actual admission for CUDA 13 (no removed property) and HIP
        # (no CUDA-only query). API doubles are not native SDK qualification.
        source = (ROOT / "src/thirdparty/SiftGPU/ProgramCU.cu").read_text()
        sampler = source.split("ProgramCU::SampleImageStreamed(", 1)[1]
        admission = sampler.split("  int device;", 1)[1].split(
            "  CheckStreamCancellation", 1
        )[0]
        code = (
            r"""
#include "src/thirdparty/SiftGPU/StreamedPlane.h"
#include <cassert>
#include <string>
#include <type_traits>
using namespace siftgpu;
enum { cudaSuccess, cudaFailure, cudaChannelFormatKindFloat };
struct cudaDeviceProp {
  size_t textureAlignment;
#if defined(COLMAP_HIP_ENABLED)
  int maxTexture1DLinear;
#endif
};
#if !defined(COLMAP_HIP_ENABLED)
struct cudaChannelFormatDesc { int x, y, z, w, f; };
template<class T> cudaChannelFormatDesc cudaCreateChannelDesc() {
  static_assert(std::is_same<T, float>::value, "float sampler format");
  return {32, 0, 0, 0, cudaChannelFormatKindFloat};
}
#endif
#if defined(COLMAP_HIP_ENABLED)
constexpr int query_calls = 2;
#else
constexpr int query_calls = 3;
#endif
static int fail_stage = 0;
static int calls = 0;
static size_t limit = 328;
int cudaGetDevice(int* device) {
  ++calls; *device = 2;
  return fail_stage == 1 ? cudaFailure : cudaSuccess;
}
int cudaGetDeviceProperties(cudaDeviceProp* properties, int device) {
  ++calls; assert(device == 2); properties->textureAlignment = 256;
#if defined(COLMAP_HIP_ENABLED)
  properties->maxTexture1DLinear = int(limit);
#endif
  return fail_stage == 2 ? cudaFailure : cudaSuccess;
}
#if !defined(COLMAP_HIP_ENABLED)
int cudaDeviceGetTexture1DLinearMaxWidth(
    size_t* width, const cudaChannelFormatDesc* format, int device) {
  ++calls; assert(device == 2);
  assert(format->x == 32 && format->y == 0 && format->z == 0 &&
         format->w == 0 && format->f == cudaChannelFormatKindFloat);
  *width = limit;
  return fail_stage == 3 ? cudaFailure : cudaSuccess;
}
#endif
struct Input {
  int Width() const { return 132; }
  int Height() const { return 9; }
  size_t CapacityValues() const { return 132 * 9; }
};
void Admit() {
  Input input;
  const int width = 264, height = 18;
  const bool up = true, exact = false;
  StreamFilterOptions options;
  options.gpu_bytes = 4096; options.host_bytes = 4096;
  int device;
"""
            + admission
            + r"""
}
int main() {
  // Literal SamplePlan source extent: 2W+1 plus 63 alignment elements.
  for (size_t capacity : {size_t(328), size_t(329)}) {
    limit = capacity; calls = 0; Admit(); assert(calls == query_calls);
  }
  limit = 327;
  bool refused = false;
  try { Admit(); } catch (const std::runtime_error& error) {
    refused = std::string(error.what()) ==
        "SIFT sampling CUDA linear texture limit";
  }
  assert(refused);
  for (fail_stage = 1; fail_stage <= query_calls; ++fail_stage) {
    limit = 328; calls = 0; refused = false;
    try { Admit(); } catch (const std::runtime_error& error) {
      refused = std::string(error.what()) == "SIFT sampling CUDA properties";
    }
    assert(refused && calls == fail_stage);
  }
}
"""
        )
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            test = pathlib.Path(directory) / "sampler.cc"
            binary = pathlib.Path(directory) / "sampler"
            test.write_text(code)
            for backend in ("COLMAP_CUDA_ENABLED", "COLMAP_HIP_ENABLED"):
                with self.subTest(backend=backend):
                    subprocess.run(
                        [
                            "c++",
                            "-std=c++17",
                            "-Wall",
                            "-Wextra",
                            "-Werror",
                            f"-D{backend}",
                            "-I",
                            str(ROOT),
                            str(test),
                            "-o",
                            str(binary),
                        ],
                        check=True,
                    )
                    subprocess.run([str(binary)], check=True)

    def test_aligned_sampling_extent_and_history(self):
        # Agreed seams: native sampler admission/windows and per-extractor
        # history.
        code = r"""
#include "src/thirdparty/SiftGPU/StreamedPlane.h"
#include <cassert>
using namespace siftgpu;
template<class F> void Refuses(F f) {
  bool caught = false;
  try { f(); } catch (const std::exception&) { caught = true; }
  assert(caught);
}
int main() {
  SamplePlan plan(132, 9, 264, 18, true, false, 4096, 4096, 256);
  // 2W+1 plus worst 63-float alignment slack.
  assert(plan.source_values == 328);
  assert(plan.gpu_bytes == (328 + 264 + 33) * 4);
  auto seam = plan.Window(3);
  assert(seam.origin == 128 && seam.values == 269); // Includes flat index 396.
  auto last = plan.Window(17);
  assert(last.origin == 1024 && last.values == 164); // Resource ends at 1188.
  assert((1188 - last.origin) % 64 == 1188 % 64);
  SamplePlan one(128, 1, 256, 2, true, false, 4096, 4096, 256);
  assert(one.Window(1).values == 128);
  Refuses([] { SamplePlan(132, 9, 264, 18, true, false, 2499, 4096, 256); });
  Refuses([] { SamplePlan(132, 9, 264, 18, true, false, 4096, 1311, 256); });
  Refuses([] { SamplePlan(132, 9, 264, 18, true, false, 4096, 4096, 3); });
  Refuses([&] { plan.Window(18); });
  StreamInputHistory history;
  auto& first = history.Begin(".", 132, 3);
  std::vector<float> old(396, 0.875f);
  first.WriteRows(0, 3, old.data()); first.Seal();
  auto& shrink = history.Begin(".", 128, 2);
  std::vector<float> fresh(256, 0.125f);
  shrink.WriteRows(0, 2, fresh.data()); shrink.Seal();
  assert(history.Capacity() == 396);
  assert(shrink.Width() == 128 && shrink.Height() == 2);
  float bits[3]; shrink.ReadFlat(255, 3, bits);
  assert(bits[0] == 0.125f && bits[1] == 0.875f && bits[2] == 0.875f);
  SamplePlan retained(128, 2, 256, 4, true, false, 4096, 4096, 256, 396);
  assert(retained.Window(3).origin == 128);
  // Through N+W, not just two active rows.
  assert(retained.Window(3).values == 257);
  auto& grown = history.Begin(".", 133, 3);
  std::vector<float> newer(399, 0.25f);
  grown.WriteRows(0, 3, newer.data()); grown.Seal();
  assert(history.Capacity() == 399);
  grown.ReadFlat(398, 1, bits); assert(bits[0] == 0.25f);
  history.Invalidate();
  Refuses([&] { history.Begin(".", 128, 2); });
  // Extents ending at N+1, N+W, N+W+1. No fresh substitute can pass.
  for (int capacity : {257, 384, 385}) {
    StreamInputHistory tail;
    auto& prior = tail.Begin(".", capacity, 1);
    std::vector<float> sentinels(capacity, 0.875f);
    prior.WriteRows(0, 1, sentinels.data()); prior.Seal();
    auto& active = tail.Begin(".", 128, 2);
    std::vector<float> prefix(256, -0.0f);
    active.WriteRows(0, 2, prefix.data()); active.Seal();
    float boundary[2]; active.ReadFlat(255, 2, boundary);
    assert(std::memcmp(boundary, prefix.data(), 4) == 0);
    assert(boundary[1] == 0.875f);
    SamplePlan partial(128, 2, 256, 4, true, false, 4096, 4096, 256, capacity);
    assert(partial.Window(3).values == size_t(capacity - 128));
    assert(active.Bytes() == size_t(capacity) * 4);
  }
  StreamInputHistory incomplete;
  incomplete.Begin(".", 128, 2);
  Refuses([&] { incomplete.Begin(".", 128, 2); });

}
"""
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            source = pathlib.Path(tmp) / "test.cc"
            source.write_text(code)
            binary = pathlib.Path(tmp) / "test"
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(ROOT),
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)

    def test_orientation_expansion_matches_frozen_resident(self):
        code = r"""
#include "StreamedPlane.h"
#include "src/colmap/feature/sift_orientation_test_util.h"
#include <cstring>
#include <iostream>
#include <vector>
int main() {
  constexpr int count = 10;
  const uint32_t packed[count] = {0xffffffffu, 0xffff0000u, 0x80000000u,
      0x40004000u, 0x0001ffffu, 0x00000001u, 0xfffe0001u, 0x0001fffeu,
      0xffff7fffu, 0x80018000u};
  float input[count*4];
  for (int i = 0; i < count; ++i) {
    input[i*4] = i + 0.125f; input[i*4+1] = -i - 0.25f;
    input[i*4+2] = i + 1.5f;
    std::memcpy(input + i*4+3, packed+i, 4);
  }
  float expected[count*8], actual[count*8];
  const int n = colmap::test::ReshapeOriginalResidentOrientations(
      input, count, expected);
  if (n != 13) return 1;
  const int got = siftgpu::ReshapePackedOrientations(input, count, actual);
  if (got != n || std::memcmp(actual, expected, n*16)) return 2;
  for (int batch : {1, 3, 7, 10}) {
    int produced = 0;
    for (int first = 0; first < count; first += batch) {
      produced += siftgpu::ReshapePackedOrientations(input + first*4,
          std::min(batch, count-first), actual + produced*4);
    }
    if (produced != n || std::memcmp(actual, expected, n*16)) return 3;
  }
  if (siftgpu::ReshapePackedOrientations(input, 0, actual) != 0) return 4;
  std::cout << "SIFT_ORIENTATION_ORIGINAL_EXPANSION "
               "records=10 expanded=13 batches=4\n";
}
"""
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            temp = pathlib.Path(directory)
            source, binary = temp / "test.cc", temp / "test"
            source.write_text(code)
            candidate = (
                ROOT / "src/thirdparty/SiftGPU/StreamedPlane.h"
            ).read_text()
            header = temp / "StreamedPlane.h"
            header.write_text(candidate)
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(temp),
                    "-I",
                    str(ROOT),
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)
            # Mutate only private temporary copies: each acceptance concern
            # must fail against the unchanged independent original oracle.
            mutations = {
                "sentinel": ("if (orientations[0] != 65535)", "if (true)"),
                "duplicate": (
                    "orientations[1] != 65535 && "
                    "orientations[1] != orientations[0]",
                    "orientations[1] != 65535",
                ),
                "conversion": (
                    "float(factor * orientations[0])",
                    "float(factor * orientations[0]) + 0.001f",
                ),
                "order": (
                    "  return retained;\n}\n\n// One-feature batch:",
                    "  if (retained > 1) for (int c = 0; c < 4; ++c) "
                    "std::swap(des[-4+c], des[-8+c]);\n  return retained;"
                    "\n}\n\n// One-feature batch:",
                ),
            }
            for concern, (old, new) in mutations.items():
                with self.subTest(mutation=concern):
                    self.assertEqual(candidate.count(old), 1)
                    header.write_text(candidate.replace(old, new))
                    subprocess.run(
                        [
                            "c++",
                            "-std=c++17",
                            "-O2",
                            "-Wall",
                            "-Wextra",
                            "-Werror",
                            "-I",
                            str(temp),
                            "-I",
                            str(ROOT),
                            str(source),
                            "-o",
                            str(binary),
                        ],
                        check=True,
                    )
                    result = subprocess.run([str(binary)], capture_output=True)
                    self.assertEqual(
                        result.returncode,
                        2,
                        f"expected byte/count mismatch for {concern}",
                    )
            print("SIFT_ORIENTATION_ORACLE_MUTANTS rejected=4")

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "GNU syscall wrapping on Linux"
    )
    def test_plane_and_budget(self):
        code = r"""
#include "src/thirdparty/SiftGPU/StreamedPlane.h"
#include <cassert>
#include <climits>
#include <cstring>
#include <sys/resource.h>
#include <signal.h>
#include <filesystem>
using namespace siftgpu;
static bool interrupt_io = false, fail_io = false, zero_io = false;
static std::string directory;
extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" ssize_t __real_pwrite(int, const void*, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void* p, size_t n, off_t o) {
  if (fail_io) { errno = EIO; return -1; }
  if (zero_io) return 0;
  if (interrupt_io) { interrupt_io = false; errno = EINTR; return -1; }
  return __real_pread(fd, p, std::min(n, size_t(3)), o);
}
extern "C" ssize_t __wrap_pwrite(int fd, const void* p, size_t n, off_t o) {
  if (fail_io) { errno = ENOSPC; return -1; }
  if (interrupt_io) { interrupt_io = false; errno = EINTR; return -1; }
  return __real_pwrite(fd, p, std::min(n, size_t(5)), o);
}
template<class F> void Refuses(F f) {
  bool caught = false;
  try { f(); } catch (const std::exception&) { caught = true; }
  assert(caught);
}
int main(int argc, char** argv) {
  assert(argc == 2);
  directory = argv[1];
  Refuses([] { StripePlan(17, 100, 33, 100, 100); });
  SamplePlan up(17, 101, 34, 202, true, true, 4096, 4096);
  assert(up.host_bytes == 136 && up.gpu_bytes == 404);
  SamplePlan aligned(16, 10, 32, 20, true, false, 4096, 4096);
  Refuses([] { SamplePlan(17, 1, 8, 2, false, true, 4096, 4096); });
  Refuses([] { SamplePlan(17, 100, 34, 200, true, true, 403, 4096); });
  Refuses([] { SamplePlan(17, 100, 34, 200, true, true, 4096, 135); });
  SamplePlan one(1, 1, 1, 1, false, true, 140, 4);
  assert(one.host_bytes == 4 && one.gpu_bytes == 140);
  auto plan = StripePlan(17, 100, 33, 2*40*17*4+132, 40*17*4+132);
  assert(plan.rows == 8 && plan.staging_rows == 40);
  assert(plan.gpu_bytes == 5572 && plan.host_bytes == 2852);
  Refuses([] { StripePlan(INT_MAX, 100, 33, SIZE_MAX, SIZE_MAX); });
  auto minimum = StripePlan(17, 100, 33, 2*33*17*4+132, 33*17*4+132);
  assert(minimum.rows == 1);
  Refuses([] { StripePlan(17, 100, 33, 2*33*17*4+131, 33*17*4+132); });
  Refuses([] { StripePlan(17, 100, 33, 2*33*17*4+132, 33*17*4+131); });
  Refuses([] { StreamedPlane bad(directory, INT_MAX, INT_MAX); });
  Refuses([] { StreamedPlane bad(directory, 0, 10); });
  const auto open_files = [] { return std::distance(
      std::filesystem::directory_iterator("/proc/self/fd"),
      std::filesystem::directory_iterator()); };
  const auto before = open_files();
  { StreamedPlane scoped(directory, 1, 1); }
  assert(open_files() == before);
  Refuses([] { StreamedPlane bad("/missing-sift-directory", 1, 1); });
  {
    StreamInputHistory owned;
    auto& input = owned.Begin(directory, 3, 4);
    float values[12] = {};
    input.WriteRows(0, 4, values); input.Seal();
    assert(open_files() == before + 1);
    owned.Invalidate();
    // Unlinked reserved blocks have no owner now.
    assert(open_files() == before);
  }
  {
    StreamedPlane flat(directory, 3, 3);
    float data[9] = {1,2,3,4,5,6,7,8,9}, copy[5] = {};
    Refuses([&] { flat.ReadFlat(2, 5, copy); });
    flat.WriteRows(0, 3, data);
    flat.Seal();
    interrupt_io = true;
    flat.ReadFlat(2, 5, copy); // Partial/EINTR exact IO crosses two seams.
    const float literal[5] = {3,4,5,6,7};
    assert(std::memcmp(copy, literal, sizeof(copy)) == 0);
    assert(flat.BytesRead() == 20);
    Refuses([&] { flat.ReadFlat(9, 1, copy); });
    Refuses([&] { flat.ReadFlat(8, SIZE_MAX, copy); });
    Refuses([&] { flat.ReadFlat(0, 0, copy); });
    Refuses([&] { flat.ReadFlat(2, 5, nullptr); });
    Refuses([&] { flat.ReadFlat(2, 5, copy, [] { return true; }); });
    zero_io = true;
    Refuses([&] { flat.ReadFlat(2, 5, copy); });
    zero_io = false;
    fail_io = true;
    Refuses([&] { flat.ReadFlat(2, 5, copy); });
    fail_io = false;
  }
  // Failed/cancelled prefix updates must not publish a fake valid transaction.
  for (bool cancelled : {false, true}) {
    StreamInputHistory history;
    auto& prior = history.Begin(directory, 4, 3);
    float data[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    prior.WriteRows(0, 3, data); prior.Seal();
    auto& partial = history.Begin(directory, 4, 2);
    partial.WriteRows(0, 1, data);
    int checks = 0;
    fail_io = !cancelled;
    Refuses([&] { partial.WriteRows(1, 1, data, [&] {
      return cancelled && ++checks > 1;
    }); });
    fail_io = false;
    Refuses([&] { partial.Seal(); });
    Refuses([&] { history.Begin(directory, 4, 2); });
  }
  KeyPlan key(17, 100, 7*5*17*4+132, 4*5*17*4);
  assert(key.rows == 3 && key.staging_rows == 5);
  assert(key.gpu_bytes == 2512 && key.host_bytes == 1360);
  KeyPlan tiny_key(1, 1, 164, 16);
  assert(tiny_key.rows == 1 && tiny_key.staging_rows == 1);
  assert(tiny_key.gpu_bytes == 164 && tiny_key.host_bytes == 16);
  Refuses([] { KeyPlan(1, 1, 163, 16); });
  KeyPlan large_key(6201, 4130, 7*3*6201*4+132, 4*3*6201*4);
  assert(large_key.rows == 1 && large_key.staging_rows == 3);
  assert(large_key.gpu_bytes < size_t(6201)*4130*4);
  assert(large_key.host_bytes < size_t(6201)*4130*4);
  Refuses([] { KeyPlan(17, 100, 7*3*17*4+131, 4096); });
  Refuses([] { KeyPlan(17, 100, 4096, 4*3*17*4-1); });
  Refuses([] { KeyPlan(INT_MAX, 100, SIZE_MAX, SIZE_MAX); });
  // Literal packed-angle oracle: no reimplementation of the conversion.
  float packed_angles[16] = {1,2,3,0, 4,5,6,0, 7,8,9,0, 10,11,12,0};
  const uint32_t angle_bits[4] = {
      0x80000000u, 0x40004000u, 0xffff0000u, 0xffffffffu};
  for (int i = 0; i < 4; ++i)
    std::memcpy(packed_angles + i*4+3, angle_bits+i, 4);
  float expanded_angles[32] = {};
  assert(ReshapePackedOrientations(packed_angles, 4, expanded_angles) == 4);
  assert(expanded_angles[0] == 1 && expanded_angles[4] == 1 &&
         expanded_angles[8] == 4 && expanded_angles[12] == 7);
  const uint32_t expected_angle_bits[4] = {0u, 0x404910a4u, 0x3fc910a4u, 0u};
  for (int i = 0; i < 4; ++i) {
    uint32_t bits; std::memcpy(&bits, expanded_angles + i*4+3, 4);
    assert(bits == expected_angle_bits[i]);
  }
  assert(ReshapePackedOrientations(packed_angles, 0, expanded_angles) == 0);
  DescriptorPlan descriptor(17, 5, 1468, 1336);
  assert(descriptor.gpu_bytes == 1468 && descriptor.host_bytes == 1336);
  Refuses([] { DescriptorPlan(17, 5, 1467, 1336); });
  Refuses([] { DescriptorPlan(17, 5, 1468, 1335); });
  Refuses([] { DescriptorPlan(INT_MAX, 5, SIZE_MAX, SIZE_MAX); });
  {
    StreamedPlane descriptors(directory, 1, 2, 128);
    float record[128]; for (int i = 0; i < 128; ++i) record[i] = float(i);
    descriptors.WriteRows(0, 1, record);
    descriptors.WriteRows(1, 1, record);
    descriptors.Seal();
    assert(descriptors.Width() == 1 && descriptors.RowBytes() == 512);
    float copy[128]; descriptors.ReadRecord(0, 1, copy);
    assert(std::memcmp(record, copy, 512) == 0);
    Refuses([&] { descriptors.ReadRows(0, 1, copy, [] { return true; }); });
    assert(descriptors.BytesWritten() == 1024 &&
           descriptors.BytesRead() == 512);
    StreamedPlane failed_descriptor(directory, 1, 2, 128);
    fail_io = true;
    Refuses([&] { failed_descriptor.WriteRows(0, 1, record); });
    fail_io = false;
    Refuses([&] { failed_descriptor.Seal(); });
    assert(!failed_descriptor.Complete());
    StreamedPlane cancelled_descriptor(directory, 1, 2, 128);
    int descriptor_checks = 0;
    Refuses([&] { cancelled_descriptor.WriteRows(0, 1, record,
        [&] { return ++descriptor_checks > 2; }); });
    assert(cancelled_descriptor.BytesWritten() == 10);
    Refuses([&] { cancelled_descriptor.Seal(); });
    zero_io = true;
    Refuses([&] { descriptors.ReadRecord(0, 0, copy); });
    zero_io = false;
  }
  OrientationPlan orientation(17, 5, 132 + 48 + 17*5*8, 96 + 17*5*8);
  assert(orientation.gpu_bytes == 860 && orientation.host_bytes == 776);
  Refuses([] { OrientationPlan(17, 5, 859, 776); });
  Refuses([] { OrientationPlan(17, 5, 860, 775); });
  Refuses([] { OrientationPlan(INT_MAX, 5, SIZE_MAX, SIZE_MAX); });
  {
    StreamedPlane expanded(directory, 1, 6, 4);
    float xyso[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    expanded.Reserve();
    expanded.WriteRows(0, 2, xyso);
    expanded.SealPrefix(2);
    assert(expanded.Complete() && expanded.Height() == 2 &&
           expanded.Bytes() == 32);
    float copy[8]; expanded.ReadRows(0, 2, copy);
    assert(std::memcmp(copy, xyso, 32) == 0);
    Refuses([&] { expanded.SealPrefix(1); });
  }
  {
    StreamedPlane partial(directory, 1, 4, 4);
    float xyso[4] = {1,2,3,4}; partial.WriteRows(0, 1, xyso);
    Refuses([&] { partial.SealPrefix(2); });
    Refuses([&] { partial.SealPrefix(0); });
    assert(!partial.Complete());
    partial.SealPrefix(1);
  }
  ListPlan list_plan(132 + 2*16*7, 2*16*7 + 64);
  assert(list_plan.records == 7);
  assert(list_plan.gpu_bytes == 356 && list_plan.host_bytes == 288);
  Refuses([] { ListPlan(163, 96); });
  Refuses([] { ListPlan(164, 95); });
  StreamedPlane records(directory, 3, 2, 4);
  float record[12] = {1,2,3,4,5,6,7,8,9,10,11,12}, record_copy[12];
  records.WriteRows(0, 1, record);
  records.WriteRows(1, 1, record);
  records.Seal();
  records.ReadRows(1, 1, record_copy);
  float cell[4];
  records.ReadRecord(2, 1, cell);
  assert(std::memcmp(cell, record + 8, 16) == 0);
  Refuses([&] { records.ReadRecord(3, 1, cell); });
  Refuses([&] { records.ReadRecord(0, 2, cell); });
  assert(records.RowBytes() == 48 && records.Bytes() == 96);
  assert(records.BytesRead() == 64 && records.BytesWritten() == 96);
  assert(std::memcmp(record, record_copy, sizeof(record)) == 0);
  DogPlan dog(17, 100, 5*5*17*4+132, 2*5*17*4);
  assert(dog.rows == 3 && dog.staging_rows == 5);
  assert(dog.gpu_bytes == 1832 && dog.host_bytes == 680);
  DogPlan tiny(1, 1, 152, 8);
  assert(tiny.rows == 1 && tiny.staging_rows == 1);
  Refuses([] { DogPlan(17, 100, 5*3*17*4+131, 4096); });
  Refuses([] { DogPlan(17, 100, 4096, 2*3*17*4-1); });
  Refuses([] { DogPlan(INT_MAX, 100, SIZE_MAX, SIZE_MAX); });
  Refuses([] { StreamedPlane bad(directory, 3, 4, 3); });
  StreamedPlane gradient(directory, 3, 2, 2);
  assert(gradient.Width() == 3 && gradient.Channels() == 2);
  assert(gradient.RowBytes() == 24 && gradient.Bytes() == 48);
  float pairs[12] = {1,2,3,4,5,6,7,8,9,10,11,12}, copied[12] = {};
  gradient.WriteRows(0, 1, pairs);
  gradient.WriteRows(1, 1, pairs+6);
  gradient.Seal();
  gradient.ReadRows(0, 2, copied);
  assert(std::memcmp(pairs, copied, sizeof(pairs)) == 0);
  assert(gradient.BytesRead() == 48 && gradient.BytesWritten() == 48);
  StreamedPlane plane(directory, 3, 4);
  assert(!plane.Complete());
  float values[12] = {1,2,3,4,5,6,7,8,9,10,11,12}, result[12] = {};
  Refuses([&] { plane.ReadRows(0, 1, result); }); // unpublished data
  interrupt_io = true;
  plane.WriteRows(0, 4, values);
  assert(plane.BytesWritten() == sizeof(values));
  plane.Seal();
  interrupt_io = true;
  plane.ReadRows(0, 4, result);
  assert(plane.BytesRead() == sizeof(values));
  assert(std::memcmp(values, result, sizeof(values)) == 0);
  zero_io = true;
  Refuses([&] { plane.ReadRows(0, 1, result); });
  zero_io = false;
  // Scratch is unlinked immediately.
  assert(std::filesystem::is_empty(directory));
  StreamedPlane cancelled(directory, 3, 4);
  int checks = 0;
  Refuses([&] {
    cancelled.WriteRows(0, 4, values, [&] { return ++checks > 2; });
  });
  assert(cancelled.BytesWritten() == 10 && !cancelled.Complete());
  Refuses([&] { cancelled.Seal(); });
  Refuses([&] { plane.ReadRows(3, 2, result); });
  Refuses([&] { plane.WriteRows(0, 1, values); });
  Refuses([&] { plane.ReadRows(0, 1, result, [] { return true; }); });
  StreamedPlane failed(directory, 3, 4);
  fail_io = true;
  Refuses([&] { failed.WriteRows(0, 1, values); });
  assert(!failed.Complete() && failed.BytesWritten() == 0);
  fail_io = false;
  Refuses([&] { failed.Seal(); });
  Refuses([&] { failed.WriteRows(0, 4, values, [] { return true; }); });
  assert(!failed.Complete());
  // Real bounded disk write failure, no filling a disk or forced OOM.
  StreamInputHistory grow_failure;
  auto& small = grow_failure.Begin(directory, 1, 1);
  small.WriteRows(0, 1, values); small.Seal();
  signal(SIGXFSZ, SIG_IGN);
  struct rlimit old, limit;
  assert(getrlimit(RLIMIT_FSIZE, &old) == 0);
  limit = old; limit.rlim_cur = 16;
  assert(setrlimit(RLIMIT_FSIZE, &limit) == 0);
  StreamedPlane limited(directory, 3, 4);
  Refuses([&] { limited.Reserve(); });
  Refuses([&] { limited.WriteRows(0, 4, values); });
  assert(!limited.Complete());
  Refuses([&] { grow_failure.Begin(directory, 3, 4); });
  assert(setrlimit(RLIMIT_FSIZE, &old) == 0);
  Refuses([&] { grow_failure.Begin(directory, 1, 1); });
}
"""
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            temp = pathlib.Path(directory)
            source = temp / "test.cc"
            binary = temp / "test"
            source.write_text(code)
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(ROOT),
                    str(source),
                    "-Wl,--wrap=pread",
                    "-Wl,--wrap=pwrite",
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            scratch = temp / "scratch"
            scratch.mkdir()
            subprocess.run([str(binary), str(scratch)], check=True)
            self.assertEqual(list(scratch.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
