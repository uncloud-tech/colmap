# SPDX-License-Identifier: BSD-3-Clause
"""CPU linker regression only: not a CUDA execution qualification.

Exercise the test capture at the application ABI seam using a stub runtime in a
separate object. Actual queued copies/events remain covered by the GPU oracle.
"""

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src/colmap/mvs/owned_strip_allocation_test_util.h"
FAKE_CUDA = """
#pragma once
#include <cstddef>
#define CUDARTAPI
#define CUDART_VERSION 12000
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1;
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaArray_t = void*;
using cudaMemPool_t = void*;
using cudaMemcpyKind = int;
struct cudaExtent { size_t width, height, depth; };
struct cudaChannelFormatDesc { int x, y, z, w; };
extern "C" int cudaEventRecord(cudaEvent_t, cudaStream_t);
extern "C" int cudaEventRecord_ptsz(cudaEvent_t, cudaStream_t);
"""


class InterpositionTest(unittest.TestCase):
    def test_both_application_abis_and_bypass_mutant(self):
        header = HEADER.read_text()
        declarations = re.findall(
            r"cudaError_t CUDARTAPI __real_(cuda\w+)\((.*?)\);",
            header,
            re.S,
        )
        signatures = dict(declarations)
        stream_apis = [
            "cudaMallocAsync", "cudaMallocFromPoolAsync", "cudaFreeAsync",
            "cudaStreamSynchronize", "cudaMemcpy2DAsync",
        ]
        for name in stream_apis:
            signatures[name + "_ptsz"] = signatures[name]
        runtime = []
        for name, args in signatures.items():
            types = args.split(",")
            parameters = ",".join(f"{t.strip()} a{i}" for i, t in enumerate(types))
            body = ""
            if name.removesuffix("_ptsz") in stream_apis:
                body = f"assert(expected_ptsz == {int(name.endswith('_ptsz'))});"
            if name.startswith("cudaMalloc"):
                body += "*a0 = static_cast<void*>(storage);"
                if name == "cudaMallocPitch":
                    body += "*a1 = a2;"
            runtime.append(f'extern "C" int {name}({parameters}) {{ {body} return 0; }}')
        cmake = (ROOT / "src/colmap/mvs/CMakeLists.txt").read_text()
        apis = re.search(r"foreach\(api (cudaMalloc\b.*?)\)", cmake, re.S)[1].split()
        with tempfile.TemporaryDirectory(prefix="strip-abi-", dir=ROOT) as tmp:
            tmp = Path(tmp)
            (tmp / "cuda_runtime.h").write_text(FAKE_CUDA)
            (tmp / "runtime.cc").write_text(
                '#include "cuda_runtime.h"\n#include <cassert>\nint expected_ptsz = 0;\nstatic char storage[4096];\n'
                + "\n".join(runtime)
                + '\nextern "C" int cudaEventRecord(cudaEvent_t, cudaStream_t) { assert(expected_ptsz == 0); return 0; }'
                + '\nextern "C" int cudaEventRecord_ptsz(cudaEvent_t, cudaStream_t) { assert(expected_ptsz == 1); return 0; }'
            )
            for suffix in ("", "_ptsz"):
                # Include capture definitions in one TU only.
                driver = f'#include "{HEADER}"\n#include <cassert>\n'
                driver += "extern int expected_ptsz;\n"
                driver += '\nextern "C" {\n' + "\n".join(
                    f"int {name}({args});" for name, args in signatures.items()
                ) + "\n}\n"
                driver += '''
int main() {
  using namespace colmap::mvs::owned::strip_test;
  Capture capture;
  Scope scope(capture);
  void* p = nullptr;
  auto stream = reinterpret_cast<void*>(123);
'''
                driver += f"expected_ptsz = {int(bool(suffix))};\n"
                for alloc in (f"cudaMallocAsync{suffix}(&p, 64, stream)",
                              f"cudaMallocFromPoolAsync{suffix}(&p, 64, nullptr, stream)"):
                    driver += f"assert({alloc} == 0);\n"
                    driver += "assert(!capture.allocations.empty());\n"
                    driver += "assert(capture.allocations.back().released == 0);\n"
                    driver += f"assert(cudaFreeAsync{suffix}(p, stream) == 0);\n"
                    driver += "assert(!capture.AllReleased());\n"
                    driver += f"assert(cudaStreamSynchronize{suffix}(stream) == 0);\n"
                    driver += "assert(capture.AllReleased());\n"
                driver += "capture.fail_copy_after_submission = true;\n"
                driver += f"assert(cudaMemcpy2DAsync{suffix}(p, 8, p, 8, 8, 1, 0, stream) == cudaErrorInvalidValue);\n"
                driver += "assert(capture.fault_submitted && !capture.fault_drained);\n"
                driver += f"assert(cudaStreamSynchronize{suffix}(stream) == 0);\n"
                driver += "assert(capture.fault_drained && !capture.release_before_fault_drain);\n}\n"
                (tmp / "driver.cc").write_text(driver)
                command = ["g++", "-std=c++17", f"-I{tmp}", str(tmp / "driver.cc"),
                           str(tmp / "runtime.cc"), "-o", str(tmp / "test")]
                flags = [f"-Wl,--wrap={api}" for api in apis]
                subprocess.run(command + flags, check=True, capture_output=True)
                result = subprocess.run([str(tmp / "test")], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                if suffix:
                    # Bypassing a real ABI relocation must break the proof even
                    # though all wrapper definitions remain in the executable.
                    subprocess.run(command + [f for f in flags if f != "-Wl,--wrap=cudaFreeAsync_ptsz"]
                                   + ["-Wl,--defsym=__real_cudaFreeAsync_ptsz=cudaFreeAsync_ptsz"],
                                   check=True, capture_output=True)
                    result = subprocess.run([str(tmp / "test")], capture_output=True)
                    self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
