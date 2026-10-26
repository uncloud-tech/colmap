# SPDX-License-Identifier: BSD-3-Clause
"""Compile actual owned stream/timer and ctor/dtor against CPU CUDA API stubs.
These tests prove host contracts ONLY, never GPU ordering or CUDA ABI/codegen.
"""

import resource
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MVS = ROOT / "src/colmap/mvs"
STUB = (
    "\n"
    "#pragma once\n"
    "#include <cassert>\n"
    "#include <cstdint>\n"
    "#include <stdexcept>\n"
    "#include <string>\n"
    "#include <vector>\n"
    "using cudaStream_t = void*;\n"
    "using cudaEvent_t = void*;\n"
    "using cudaError_t = int;\n"
    "constexpr int cudaSuccess=0;\n"
    "constexpr unsigned cudaStreamNonBlocking=1;\n"
    "namespace fake {\n"
    "struct Call { std::string name; void* arg; };\n"
    "inline std::vector<Call> calls;\n"
    "inline uintptr_t next=1;\n"
    "inline int device=7;\n"
    "inline bool pending=false;\n"
    "inline bool fail_init=false;\n"
    "inline void log(const char* name,void* arg=nullptr) { calls."
    "push_back({name,arg}); }\n"
    "}\n"
    "inline int cudaGetDevice(int* x) { *x=fake::device; return 0"
    "; }\n"
    "inline int cudaSetDevice(int x) { fake::device=x; return 0; "
    "}\n"
    "inline int cudaStreamCreateWithFlags(void** x,unsigned f) {\n"
    " assert(f==cudaStreamNonBlocking); *x=reinterpret_cast<void*"
    ">(fake::next++);\n"
    ' fake::log("create",*x); return 0;\n'
    "}\n"
    "inline int cudaStreamSynchronize(void* x) {\n"
    ' assert(x); fake::pending=false; fake::log("sync",x); return'
    " 0;\n"
    "}\n"
    "inline int cudaStreamDestroy(void* x) { assert(!fake::pendin"
    'g); fake::log("destroy",x); return 0; }\n'
    "inline int cudaEventCreate(void** x) { *x=reinterpret_cast<v"
    "oid*>(fake::next++); return 0; }\n"
    'inline int cudaEventDestroy(void* x) { fake::log("event_dest'
    'roy",x); return 0; }\n'
    'inline int cudaEventRecord(void*,void* stream) { fake::log("'
    'record",stream); return 0; }\n'
    'inline int cudaEventSynchronize(void* x) { fake::log("event_'
    'wait",x); return 0; }\n'
    "inline int cudaEventElapsedTime(float* x,void*,void*) { *x=1"
    '2; fake::log("elapsed"); return 0; }\n'
    "namespace colmap {\n"
    "inline void CudaSafeCall(int x) { if(x) throw std::runtime_e"
    'rror("fake CUDA error"); }\n'
    "inline void SetBestCudaDevice(int x) { fake::device=x; }\n"
    "}\n"
    "#define CUDA_SAFE_CALL(x) ::colmap::CudaSafeCall(x)\n"
    '#define CUDA_CHECK() ::fake::log("check")\n'
)
PROGRAM = (
    "\n"
    '#include "colmap/mvs/owned_cuda_execution.h"\n'
    '#include "colmap/mvs/owned_tile.h"\n'
    '#include "colmap/mvs/patch_match_execution.h"\n'
    "#include <memory>\n"
    "namespace colmap { namespace mvs {\n"
    "struct PatchMatchOptions {\n"
    ' std::string gpu_index="0"; int window_radius=5; int window_'
    "step=1;\n"
    " bool geom_consistency=false;\n"
    "};\n"
    "struct PatchMatch { struct Problem {}; };\n"
    "namespace owned {\n"
    "struct Resource {\n"
    ' ~Resource() { assert(!fake::pending); fake::log("resource_d'
    'estroy"); }\n'
    "};\n"
    "class PatchMatchCuda {\n"
    " public:\n"
    " PatchMatchCuda(const PatchMatchOptions&,const PatchMatch::P"
    "roblem&,int=32);\n"
    " ~PatchMatchCuda();\n"
    " void InitRefImage() { resource_=std::make_unique<Resource>("
    "); fake::pending=true; }\n"
    " void InitSourceImages() { if(fake::fail_init) throw std::ru"
    'ntime_error("injected init failure"); }\n'
    " void InitTransforms() {}\n"
    " void InitWorkspaceMemory() {}\n"
    " void InitRotationStorage() {}\n"
    " void EmitRotationAllocations() {}\n"
    " private:\n"
    " PatchMatchOptions options_; const int owned_tile_threads_; "
    "PatchMatch::Problem problem_;\n"
    " std::unique_ptr<OwnedCudaStream> stream_;\n"
    " int ref_width_,ref_height_,rotation_in_half_pi_;\n"
    " std::unique_ptr<Resource> resource_;\n"
    "};\n"
    "// ACTUAL_CONSTRUCTOR_AND_DESTRUCTOR\n"
    "}}}\n"
    "int main() {\n"
    " using namespace colmap::mvs;\n"
    " using namespace colmap::mvs::owned;\n"
    " fake::calls.clear();\n"
    " void* expected=nullptr;\n"
    " {\n"
    "  OwnedCudaStream stream;\n"
    "  expected=stream.Get();\n"
    '  { OwnedCudaTimer timer(expected); timer.Print("cpu stub");'
    " }\n"
    "  stream.Synchronize();\n"
    " }\n"
    " bool waited=false,elapsed=false;\n"
    " for(const auto& c:fake::calls) {\n"
    '  if(c.name=="record" || c.name=="sync" || c.name=="destroy"'
    ") assert(c.arg==expected);\n"
    '  if(c.name=="event_wait") waited=true;\n'
    '  if(c.name=="elapsed") { assert(waited); elapsed=true; }\n'
    " }\n"
    ' assert(elapsed && fake::calls.back().name=="destroy");\n'
    " {\n"
    "  OwnedCudaStream stream;\n"
    "  bool caught=false;\n"
    "  try {\n"
    "   Resource temporary_texture;\n"
    "   ScopedStreamCompletion completion(stream.Get());\n"
    "   fake::pending=true;\n"
    '   throw std::runtime_error("local async API failure");\n'
    "  } catch(const std::runtime_error&) { caught=true; }\n"
    "  assert(caught && !fake::pending);\n"
    " }\n"
    " for(int tile:{0,4,64,-8}) {\n"
    "  fake::calls.clear(); fake::device=7; bool caught=false;\n"
    "  try { PatchMatchCuda engine(PatchMatchOptions{},PatchMatch"
    "::Problem{},tile); }\n"
    "  catch(const std::invalid_argument&) { caught=true; }\n"
    "  assert(caught && fake::calls.empty() && fake::device==7);\n"
    " }\n"
    " for(bool fail:{false,true}) {\n"
    "  fake::calls.clear(); fake::fail_init=fail; bool caught=fal"
    "se;\n"
    "  try { PatchMatchCuda engine(PatchMatchOptions{},PatchMatch"
    "::Problem{}); fake::pending=true; }\n"
    "  catch(const std::runtime_error&) { caught=true; }\n"
    "  assert(caught==fail); assert(!fake::pending);\n"
    "  int sync=-1,resource=-1,destroy=-1;\n"
    "  for(size_t i=0;i<fake::calls.size();++i) {\n"
    '   if(fake::calls[i].name=="sync" && resource<0) sync=int(i)'
    ";\n"
    '   if(fake::calls[i].name=="resource_destroy") resource=int('
    "i);\n"
    '   if(fake::calls[i].name=="destroy") destroy=int(i);\n'
    "  }\n"
    "  assert(sync>=0 && sync<resource && resource<destroy);\n"
    " }\n"
    "}\n"
)


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class CpuExecution(unittest.TestCase):
    def run_contract(self, mutation=None):
        stream = (MVS / "owned_cuda_execution.h").read_text()
        source = (MVS / "patch_match_cuda_owned.cu").read_text()
        ctor = source[
            source.index("PatchMatchCuda::PatchMatchCuda(") : source.index(
                "void PatchMatchCuda::Run()"
            )
        ]
        if mutation == "timer_legacy":
            stream = stream.replace(
                "cudaEventRecord(start_, stream_)",
                "cudaEventRecord(start_, nullptr)",
            )
        elif mutation == "missing_constructor_drain":
            old = "    stream_->DrainBeforeDestruction();\n    throw;"
            self.assertIn(old, ctor)
            ctor = ctor.replace(old, "    throw;")
        elif mutation == "missing_destructor_drain":
            old = (
                "  if (stream_) {\n    stream_->DrainBeforeDestruction();\n  }"
            )
            self.assertIn(old, ctor)
            ctor = ctor.replace(old, "  // mutant: no drain")
        # Retained CPU scratch, not a product build and not linked with CUDA.
        temp = Path(tempfile.mkdtemp(prefix="owned-cpu-contract-"))
        fake = temp / "colmap/util"
        fake.mkdir(parents=True)
        (fake / "cudacc.h").write_text(STUB)
        (fake / "logging.h").write_text(
            "#pragma once\n"
            "#include <sstream>\n"
            "#define LOG(x) std::ostringstream()\n"
        )
        (fake / "string.h").write_text(
            "#pragma once\n"
            "#include <string>\n"
            "namespace colmap { template<class... T> std::string StringPr"
            "intf(const char* s,T...) { return s; } }\n"
        )
        target = temp / "colmap/mvs"
        target.mkdir(parents=True)
        (target / "owned_cuda_execution.h").write_text(stream)
        program = temp / "contract.cc"
        program.write_text(
            PROGRAM.replace("// ACTUAL_CONSTRUCTOR_AND_DESTRUCTOR", ctor)
        )
        command = [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(temp),
            "-I",
            str(ROOT / "src"),
            str(program),
            "-o",
            str(temp / "contract"),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30
        )
        (temp / "compile.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run(
            [str(temp / "contract")],
            capture_output=True,
            text=True,
            timeout=10,
            preexec_fn=no_core,
        )
        (temp / "run.log").write_text(result.stdout + result.stderr)
        print(
            f"CPU_STUB_ARTIFACT {temp} "
            f"mutation={mutation} exit={result.returncode}"
        )
        return result

    def test_actual_host_lifetime_and_event_forwarding(self):
        result = self.run_contract()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mutant_legacy_event_is_rejected(self):
        self.assertNotEqual(self.run_contract("timer_legacy").returncode, 0)

    def test_mutant_constructor_unwind_is_rejected(self):
        self.assertNotEqual(
            self.run_contract("missing_constructor_drain").returncode, 0
        )

    def test_mutant_destructor_order_is_rejected(self):
        self.assertNotEqual(
            self.run_contract("missing_destructor_drain").returncode, 0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
