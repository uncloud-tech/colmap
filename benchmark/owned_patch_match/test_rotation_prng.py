# SPDX-License-Identifier: BSD-3-Clause
"""CPU/source checks only: no CUDA compiler, runtime, or GPU is used."""

import pathlib
import re
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADER = ROOT / "src/colmap/mvs/gpu_mat_prng.h"
SOURCE = ROOT / "src/colmap/mvs/gpu_mat_prng.cu"

# Frozen baseline bodies from 011f9a55: compare tokens, not formatting.
KERNEL = (
    "__global__ void InitRandomStateKernel(GpuMatView<curandState"
    "> output) {\n"
    "  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;\n"
    "  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  const size_t uniqueBlockIndex = blockIdx.y * gridDim.x + b"
    "lockIdx.x;\n"
    "  const size_t id = uniqueBlockIndex * blockDim.y * blockDim"
    ".x +\n"
    "                    threadIdx.y * blockDim.x + threadIdx.x;\n"
    "  // Each thread gets same seed, a different sequence number"
    ", no offset.\n"
    "  if (col < output.GetWidth() && row < output.GetHeight()) {"
    "\n"
    "    curand_init(id, 0, 0, &output.GetRef(row, col));\n"
    "  }\n"
    "}"
)
LEGACY = """GpuMatPRNG::GpuMatPRNG(const int width, const int height)
    : GpuMat(width, height) {
  InitRandomStateKernel<<<gridSize_, blockSize_>>>(View());
}"""
INITIALIZE = """owned::ScopedStreamCompletion completion(stream);
  InitRandomStateKernel<<<gridSize_, blockSize_, 0, stream>>>(View());
  completion.Wait();"""


def tokens(text):
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    return re.findall(r"\w+|[^\s]", text)


def body(source, signature):
    start = source.index("{", source.index(signature))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start + 1 : end - 1]


def check_contract(header, source):
    kernel_start = source.index("__global__ void InitRandomStateKernel")
    kernel_end = source.index("\n\n}  // namespace", kernel_start)
    assert tokens(source[kernel_start:kernel_end]) == tokens(KERNEL)
    legacy_start = source.index("GpuMatPRNG::GpuMatPRNG(")
    legacy_end = source.index("\n\n#if", legacy_start)
    assert tokens(source[legacy_start:legacy_end]) == tokens(LEGACY)
    guard = "#if !defined(COLMAP_HIP_ENABLED)"
    h_cuda = header.split(guard)[1].split("#endif")[0]
    assert "struct UninitializedTag {};" in h_cuda
    assert re.search(
        r"GpuMatPRNG\(int width, int height, UninitializedTag\)", h_cuda
    )
    assert "void Reinitialize(cudaStream_t stream);" in h_cuda
    cu_cuda = source.split(guard)[1].split("#endif")[0]
    assert tokens(body(cu_cuda, "const cudaStream_t stream)")) == tokens(
        "Reinitialize(stream);"
    )
    assert tokens(body(cu_cuda, "UninitializedTag)")) == []
    assert tokens(body(cu_cuda, "void GpuMatPRNG::Reinitialize(")) == tokens(
        INITIALIZE
    )
    assert cu_cuda.count(": GpuMat(width, height)") == 2


STUB = r"""
#pragma once
#include <cassert>
#include <cstddef>
#include <vector>
using std::size_t;
using cudaStream_t = int;
#define __global__
struct dim3 { size_t x, y, z; };
inline dim3 blockIdx{}, blockDim{}, threadIdx{}, gridDim{};
struct curandState { size_t seed = 99999, sequence = 99999, offset = 99999; };
inline void curand_init(size_t seed, size_t sequence, size_t offset,
                        curandState* state) {
  *state = {seed, sequence, offset};
}
namespace colmap { namespace mvs {
struct Launch { dim3 grid, block; int stream; size_t width, height; };
inline std::vector<Launch> launches;
inline std::vector<int> completions, waits;
inline int allocations = 0;
template <class T> struct GpuMatView {
  T* ptr; size_t width, height;
  size_t GetWidth() const { return width; }
  size_t GetHeight() const { return height; }
  T& GetRef(size_t row, size_t col) { return ptr[row * width + col]; }
};
template <class T> class GpuMat {
 public:
  GpuMat(int width, int height) : width_(width), height_(height),
      states_(width * height), blockSize_{32, 16, 1},
      gridSize_{(size_t(width) + 31) / 32, (size_t(height) + 15) / 16, 1} {
    ++allocations;
  }
  GpuMatView<T> View() { return {states_.data(), width_, height_}; }
 protected:
  size_t width_, height_;
  std::vector<T> states_;
  dim3 blockSize_, gridSize_;
};
namespace owned {
struct ScopedStreamCompletion {
  explicit ScopedStreamCompletion(int stream) : stream_(stream) {
    completions.push_back(stream);
  }
  void Wait() { waits.push_back(stream_); }
  int stream_;
};
}
template <class Kernel>
void FakeLaunch(dim3 grid, dim3 block, int shared, int stream,
                Kernel kernel, GpuMatView<curandState> view) {
  assert(shared == 0);
  launches.push_back({grid, block, stream, view.width, view.height});
  gridDim = grid; blockDim = block;
  for (size_t y = 0; y < grid.y; ++y)
    for (size_t x = 0; x < grid.x; ++x)
      for (size_t ty = 0; ty < block.y; ++ty)
        for (size_t tx = 0; tx < block.x; ++tx) {
          blockIdx = {x, y, 0}; threadIdx = {tx, ty, 0};
          kernel(view);
        }
}
}}
"""

HARNESS = r"""
using namespace colmap::mvs;
void CheckStates(GpuMatPRNG& prng) {
  auto view = prng.View();
  for (size_t row = 0; row < view.height; ++row)
    for (size_t col = 0; col < view.width; ++col) {
      auto state = view.GetRef(row, col);
      size_t grid_x = (view.width + 31) / 32;
      size_t id = ((row / 16) * grid_x + col / 32) * 16 * 32 +
                  (row % 16) * 32 + col % 32;
      assert(state.seed == id && state.sequence == 0 && state.offset == 0);
    }
}
int main() {
  constexpr int stream = 73;
  GpuMatPRNG current(35, 19, stream);
  assert(allocations == 1 && launches.size() == 1);
  assert(completions.size() == 1 && waits.size() == 1);
  CheckStates(current);
  GpuMatPRNG alternate(19, 35, GpuMatPRNG::UninitializedTag{});
  assert(allocations == 2 && launches.size() == 1);
  assert(completions.size() == 1 && waits.size() == 1);
  assert(alternate.View().GetRef(0, 0).seed == 99999);
  for (int rotation = 0; rotation < 20; ++rotation) {
    auto& destination = rotation % 2 == 0 ? alternate : current;
    destination.Reinitialize(stream);
    CheckStates(destination);
    const auto& launch = launches.back();
    assert(launch.width == (rotation % 2 == 0 ? 19u : 35u));
    assert(launch.height == (rotation % 2 == 0 ? 35u : 19u));
    assert(launch.grid.x == (launch.width + 31) / 32);
    assert(launch.grid.y == (launch.height + 15) / 16);
    assert(launch.block.x == 32 && launch.block.y == 16);
  }
  assert(allocations == 2 && launches.size() == 21);
  assert(completions.size() == 21 && waits.size() == 21);
  for (size_t i = 0; i < 21; ++i) {
    assert(launches[i].stream == stream);
    assert(completions[i] == stream && waits[i] == stream);
  }
  GpuMatPRNG legacy(3, 5);
  assert(allocations == 3 && launches.size() == 22);
  assert(launches.back().stream == 0 && waits.size() == 21);
  CheckStates(legacy);
}
"""


def run_host(header, source, harness=HARNESS):
    # Keep actual header/method/kernel bodies. Adapt only launch syntax and
    # GPU dependencies to CPU spies; this does not validate real CUDA behavior.
    source = re.sub(
        (
            "InitRandomStateKernel<<<gridSize_, blockSize_, 0, stream>>>\\"
            "(View\\(\\)\\)"
        ),
        "FakeLaunch(gridSize_, blockSize_, 0, stream, "
        "InitRandomStateKernel, View())",
        source,
    )
    source = source.replace(
        "InitRandomStateKernel<<<gridSize_, blockSize_>>>(View())",
        "FakeLaunch(gridSize_, blockSize_, 0, 0, "
        "InitRandomStateKernel, View())",
    )
    with tempfile.TemporaryDirectory(prefix="rotation-prng-") as tmp:
        tmp = pathlib.Path(tmp)
        includes = tmp / "colmap/mvs"
        includes.mkdir(parents=True)
        (includes / "gpu_mat.h").write_text(STUB)
        (includes / "gpu_mat_prng.h").write_text(header)
        cpp = tmp / "probe.cc"
        cpp.write_text(source + harness)
        binary = tmp / "probe"
        compiled = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-I",
                str(tmp),
                str(cpp),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if compiled.returncode:
            raise AssertionError(compiled.stderr)
        return subprocess.run(
            [str(binary)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=tmp,
        )


class RotationPrngTest(unittest.TestCase):
    def setUp(self):
        self.header = HEADER.read_text()
        self.source = SOURCE.read_text()

    def test_frozen_source_contract(self):
        check_contract(self.header, self.source)

    def test_actual_bodies_forward_and_preserve_21_initializations(self):
        result = run_host(self.header, self.source)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_negative_missing_constructor_initialization(self):
        mutated = self.source.replace("Reinitialize(stream);", "", 1)
        self.assertNotEqual(mutated, self.source)
        with self.assertRaises(AssertionError):
            check_contract(self.header, mutated)
        self.assertNotEqual(run_host(self.header, mutated).returncode, 0)

    def test_negative_wrong_forwarded_stream(self):
        mutated = self.source.replace(
            "Reinitialize(stream);", "Reinitialize(0);", 1
        )
        self.assertNotEqual(mutated, self.source)
        with self.assertRaises(AssertionError):
            check_contract(self.header, mutated)
        self.assertNotEqual(run_host(self.header, mutated).returncode, 0)

    def test_negative_kernel_seed_arithmetic_and_bounds(self):
        for old, new in [
            ("curand_init(id, 0, 0", "curand_init(id, 1, 0"),
            (
                "threadIdx.y * blockDim.x + threadIdx.x;",
                "threadIdx.y * blockDim.y + threadIdx.x;",
            ),
            ("col < output.GetWidth()", "col <= output.GetWidth()"),
        ]:
            with self.subTest(mutation=new):
                mutated = self.source.replace(old, new, 1)
                self.assertNotEqual(mutated, self.source)
                with self.assertRaises(AssertionError):
                    check_contract(self.header, mutated)

    def test_negative_removed_completion(self):
        mutated = self.source.replace("completion.Wait();", "", 1)
        self.assertNotEqual(mutated, self.source)
        with self.assertRaises(AssertionError):
            check_contract(self.header, mutated)
        self.assertNotEqual(run_host(self.header, mutated).returncode, 0)

    def test_negative_added_startup_initialization(self):
        signature = "UninitializedTag)"
        start = self.source.index("{", self.source.index(signature))
        mutated = (
            self.source[: start + 1]
            + "Reinitialize(0);"
            + self.source[start + 1 :]
        )
        with self.assertRaises(AssertionError):
            check_contract(self.header, mutated)
        self.assertNotEqual(run_host(self.header, mutated).returncode, 0)

    def test_hip_keeps_only_legacy_api(self):
        harness = (
            "\n"
            "#include <type_traits>\n"
            "using namespace colmap::mvs;\n"
            "template <class T, class = void> struct HasTag : std::false_"
            "type {};\n"
            "template <class T> struct HasTag<T, std::void_t<typename T::"
            "UninitializedTag>>\n"
            "    : std::true_type {};\n"
            "template <class T, class = void> struct HasReinitialize : st"
            "d::false_type {};\n"
            "template <class T> struct HasReinitialize<T, std::void_t<dec"
            "ltype(&T::Reinitialize)>>\n"
            "    : std::true_type {};\n"
            "static_assert(!HasTag<GpuMatPRNG>::value);\n"
            "static_assert(!HasReinitialize<GpuMatPRNG>::value);\n"
            "int main() {\n"
            "  GpuMatPRNG legacy(3, 5);\n"
            "  assert(allocations == 1 && launches.size() == 1 && waits.e"
            "mpty());\n"
            "}\n"
        )
        result = run_host(
            self.header, "#define COLMAP_HIP_ENABLED\n" + self.source, harness
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_negative_hip_guard_removed(self):
        guard = "#if !defined(COLMAP_HIP_ENABLED)"
        with self.assertRaises((AssertionError, IndexError)):
            check_contract(self.header.replace(guard, "#if 1", 1), self.source)
        with self.assertRaises((AssertionError, IndexError)):
            check_contract(self.header, self.source.replace(guard, "#if 1", 1))


if __name__ == "__main__":
    unittest.main()
