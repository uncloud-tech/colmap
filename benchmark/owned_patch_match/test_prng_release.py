# SPDX-License-Identifier: BSD-3-Clause
"""Host checks only; owned_normal_rotation_test contains the CUDA regression."""

import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STUB = r"""
#pragma once
#include <cassert>
#include <stdexcept>
using cudaStream_t = int;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr int cudaDevAttrMemoryPoolsSupported = 1;
struct curandState {};
inline int supported = 1, error = 0, frees = 0, asyncs = 0, waits = 0;
inline int last_stream = -1;
inline void* released = nullptr;
inline int cudaGetDevice(int* d) { *d = 0; return 0; }
inline int cudaDeviceGetAttribute(int* v, int, int) { *v = supported; return 0; }
inline int cudaStreamSynchronize(int s) { ++waits; last_stream = s; return 0; }
inline int cudaFree(void* p) { if (p) ++frees; released = p; return error; }
#if CUDART_VERSION >= 11020
inline int cudaFreeAsync(void* p, int s) {
  ++asyncs; released = p; last_stream = s; return error;
}
#endif
#define CUDA_SAFE_CALL(call) do { if ((call) != 0) throw std::runtime_error("CUDA"); } while (0)
namespace colmap::mvs {
template<class T> class GpuMat {
 public:
  GpuMat(int, int) : array_ptr_(new T) {}
  ~GpuMat() { if (array_ptr_) { cudaFree(array_ptr_); delete array_ptr_; } }
  T* GetPtr() { return array_ptr_; }
 protected:
  T* array_ptr_;
};
}
"""
HARNESS = r"""
using namespace colmap::mvs;
GpuMatPRNG::GpuMatPRNG(int w, int h) : GpuMat(w, h) {}
int main() {
#ifndef COLMAP_HIP_ENABLED
  for (int pools : {0, 1}) {
    supported = pools;
    int before = frees, before_async = asyncs, before_wait = waits;
    curandState* p;
    {
      GpuMatPRNG map(3, 5);
      p = map.GetPtr();
      map.ReleaseAfterInitialization(73);
      assert(map.GetPtr() == nullptr && released == p && last_stream == 73);
#if CUDART_VERSION >= 11020
      assert(frees == before + !pools);
      assert(asyncs == before_async + pools);
      assert(waits == before_wait + !pools);
#else
      assert(frees == before + 1 && asyncs == before_async);
      assert(waits == before_wait + 1);
#endif
      const int calls = frees + asyncs + waits;
      map.ReleaseAfterInitialization(73); // Empty release is idempotent.
      assert(frees + asyncs + waits == calls);
    }
    assert(frees == before + (CUDART_VERSION < 11020 || !pools));
    delete p; // The fake CUDA free above does not release host test storage.
  }
  supported = 1;
  {
    GpuMatPRNG map(3, 5);
    auto* p = map.GetPtr();
    error = 1;
    try { map.ReleaseAfterInitialization(73); assert(false); }
    catch (const std::runtime_error&) {}
    assert(map.GetPtr() == p); // Failed submission must retain cleanup ownership.
    error = 0;
  }
#endif
}
"""


class PrngReleaseTest(unittest.TestCase):
    def test_real_header_ownership_and_version_fallback(self):
        for version, hip in [(12080, False), (11010, False), (12080, True)]:
            with (
                self.subTest(version=version, hip=hip),
                tempfile.TemporaryDirectory() as tmp,
            ):
                tmp = pathlib.Path(tmp)
                include = tmp / "colmap/mvs"
                include.mkdir(parents=True)
                (include / "gpu_mat.h").write_text(STUB)
                (include / "gpu_mat_prng.h").write_bytes(
                    (ROOT / "src/colmap/mvs/gpu_mat_prng.h").read_bytes()
                )
                (tmp / "probe.cc").write_text(
                    '#include "colmap/mvs/gpu_mat_prng.h"\n' + HARNESS
                )
                command = [
                    "g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    f"-DCUDART_VERSION={version}", "-I", str(tmp),
                    str(tmp / "probe.cc"), "-o", str(tmp / "probe"),
                ]
                if hip:
                    command.insert(1, "-DCOLMAP_HIP_ENABLED")
                compiled = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                subprocess.run([str(tmp / "probe")], check=True)

    def test_release_retires_before_swapped_staging(self):
        source = (ROOT / "src/colmap/mvs/patch_match_cuda_owned.cu").read_text()
        source = source.split("void PatchMatchCuda::InitRotationStorage() {")[1]
        source = source.split("void PatchMatchCuda::EmitRotationAllocations")[0]
        release = source.index(
            "init_rand_state_map_->ReleaseAfterInitialization(stream_->Get());"
        )
        self.assertLess(source.index("CompactOwnedPrngBoundary<<<"), release)
        self.assertLess(release, source.index("completion.Wait();"))
        self.assertLess(
            source.index("completion.Wait();"), source.index("depth_map_.Set(1,")
        )


if __name__ == "__main__":
    unittest.main()
