# SPDX-License-Identifier: BSD-3-Clause
"""Compile actual owned stream/timer and ctor/dtor against CPU CUDA API stubs.
These tests prove host contracts ONLY, never GPU ordering or CUDA ABI/codegen.
"""
from pathlib import Path
import resource
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MVS = ROOT / 'src/colmap/mvs'
STUB = r'''
#pragma once
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess=0;
constexpr unsigned cudaStreamNonBlocking=1;
namespace fake {
struct Call { std::string name; void* arg; };
inline std::vector<Call> calls;
inline uintptr_t next=1;
inline int device=7;
inline bool pending=false;
inline bool fail_init=false;
inline void log(const char* name,void* arg=nullptr) { calls.push_back({name,arg}); }
}
inline int cudaGetDevice(int* x) { *x=fake::device; return 0; }
inline int cudaSetDevice(int x) { fake::device=x; return 0; }
inline int cudaStreamCreateWithFlags(void** x,unsigned f) {
 assert(f==cudaStreamNonBlocking); *x=reinterpret_cast<void*>(fake::next++);
 fake::log("create",*x); return 0;
}
inline int cudaStreamSynchronize(void* x) {
 assert(x); fake::pending=false; fake::log("sync",x); return 0;
}
inline int cudaStreamDestroy(void* x) { assert(!fake::pending); fake::log("destroy",x); return 0; }
inline int cudaEventCreate(void** x) { *x=reinterpret_cast<void*>(fake::next++); return 0; }
inline int cudaEventDestroy(void* x) { fake::log("event_destroy",x); return 0; }
inline int cudaEventRecord(void*,void* stream) { fake::log("record",stream); return 0; }
inline int cudaEventSynchronize(void* x) { fake::log("event_wait",x); return 0; }
inline int cudaEventElapsedTime(float* x,void*,void*) { *x=12; fake::log("elapsed"); return 0; }
namespace colmap {
inline void CudaSafeCall(int x) { if(x) throw std::runtime_error("fake CUDA error"); }
inline void SetBestCudaDevice(int x) { fake::device=x; }
}
#define CUDA_SAFE_CALL(x) ::colmap::CudaSafeCall(x)
#define CUDA_CHECK() ::fake::log("check")
'''
PROGRAM = r'''
#include "colmap/mvs/owned_cuda_execution.h"
#include "colmap/mvs/patch_match_execution.h"
#include <memory>
namespace colmap { namespace mvs {
struct PatchMatchOptions {
 std::string gpu_index="0"; int window_radius=5; int window_step=1;
 bool geom_consistency=false;
};
struct PatchMatch { struct Problem {}; };
namespace owned {
struct Resource {
 ~Resource() { assert(!fake::pending); fake::log("resource_destroy"); }
};
class PatchMatchCuda {
 public:
 PatchMatchCuda(const PatchMatchOptions&,const PatchMatch::Problem&);
 ~PatchMatchCuda();
 void InitRefImage() { resource_=std::make_unique<Resource>(); fake::pending=true; }
 void InitSourceImages() { if(fake::fail_init) throw std::runtime_error("injected init failure"); }
 void InitTransforms() {}
 void InitWorkspaceMemory() {}
 private:
 PatchMatchOptions options_; PatchMatch::Problem problem_;
 std::unique_ptr<OwnedCudaStream> stream_;
 int ref_width_,ref_height_,rotation_in_half_pi_;
 std::unique_ptr<Resource> resource_;
};
// ACTUAL_CONSTRUCTOR_AND_DESTRUCTOR
}}}
int main() {
 using namespace colmap::mvs;
 using namespace colmap::mvs::owned;
 fake::calls.clear();
 void* expected=nullptr;
 {
  OwnedCudaStream stream;
  expected=stream.Get();
  { OwnedCudaTimer timer(expected); timer.Print("cpu stub"); }
  stream.Synchronize();
 }
 bool waited=false,elapsed=false;
 for(const auto& c:fake::calls) {
  if(c.name=="record" || c.name=="sync" || c.name=="destroy") assert(c.arg==expected);
  if(c.name=="event_wait") waited=true;
  if(c.name=="elapsed") { assert(waited); elapsed=true; }
 }
 assert(elapsed && fake::calls.back().name=="destroy");
 {
  OwnedCudaStream stream;
  bool caught=false;
  try {
   Resource temporary_texture;
   ScopedStreamCompletion completion(stream.Get());
   fake::pending=true;
   throw std::runtime_error("local async API failure");
  } catch(const std::runtime_error&) { caught=true; }
  assert(caught && !fake::pending);
 }
 for(bool fail:{false,true}) {
  fake::calls.clear(); fake::fail_init=fail; bool caught=false;
  try { PatchMatchCuda engine(PatchMatchOptions{},PatchMatch::Problem{}); fake::pending=true; }
  catch(const std::runtime_error&) { caught=true; }
  assert(caught==fail); assert(!fake::pending);
  int sync=-1,resource=-1,destroy=-1;
  for(size_t i=0;i<fake::calls.size();++i) {
   if(fake::calls[i].name=="sync" && resource<0) sync=int(i);
   if(fake::calls[i].name=="resource_destroy") resource=int(i);
   if(fake::calls[i].name=="destroy") destroy=int(i);
  }
  assert(sync>=0 && sync<resource && resource<destroy);
 }
}
'''


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class CpuExecution(unittest.TestCase):
    def run_contract(self, mutation=None):
        stream=(MVS/'owned_cuda_execution.h').read_text()
        source=(MVS/'patch_match_cuda_owned.cu').read_text()
        ctor=source[source.index('PatchMatchCuda::PatchMatchCuda('):
                    source.index('void PatchMatchCuda::Run()')]
        if mutation=='timer_legacy':
            stream=stream.replace('cudaEventRecord(start_, stream_)','cudaEventRecord(start_, nullptr)')
        elif mutation=='missing_constructor_drain':
            old='    stream_->DrainBeforeDestruction();\n    throw;'
            self.assertIn(old,ctor)
            ctor=ctor.replace(old,'    throw;')
        elif mutation=='missing_destructor_drain':
            old='  if (stream_) {\n    stream_->DrainBeforeDestruction();\n  }'
            self.assertIn(old,ctor)
            ctor=ctor.replace(old,'  // mutant: no drain')
        # Retained CPU scratch, not a product build and not linked with CUDA.
        temp=Path(tempfile.mkdtemp(prefix='owned-cpu-contract-'))
        fake=temp/'colmap/util'; fake.mkdir(parents=True)
        (fake/'cudacc.h').write_text(STUB)
        (fake/'logging.h').write_text('#pragma once\n#include <sstream>\n#define LOG(x) std::ostringstream()\n')
        (fake/'string.h').write_text('#pragma once\n#include <string>\nnamespace colmap { template<class... T> std::string StringPrintf(const char* s,T...) { return s; } }\n')
        target=temp/'colmap/mvs';target.mkdir(parents=True)
        (target/'owned_cuda_execution.h').write_text(stream)
        program=temp/'contract.cc'
        program.write_text(PROGRAM.replace('// ACTUAL_CONSTRUCTOR_AND_DESTRUCTOR',ctor))
        command=['g++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(temp),
                 '-I',str(ROOT/'src'),str(program),'-o',str(temp/'contract')]
        result=subprocess.run(command,capture_output=True,text=True,timeout=30)
        (temp/'compile.log').write_text(result.stdout+result.stderr)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        result=subprocess.run([str(temp/'contract')],capture_output=True,text=True,
                              timeout=10,preexec_fn=no_core)
        (temp/'run.log').write_text(result.stdout+result.stderr)
        print(f'CPU_STUB_ARTIFACT {temp} mutation={mutation} exit={result.returncode}')
        return result

    def test_actual_host_lifetime_and_event_forwarding(self):
        result=self.run_contract()
        self.assertEqual(result.returncode,0,result.stderr)

    def test_mutant_legacy_event_is_rejected(self):
        self.assertNotEqual(self.run_contract('timer_legacy').returncode,0)

    def test_mutant_constructor_unwind_is_rejected(self):
        self.assertNotEqual(self.run_contract('missing_constructor_drain').returncode,0)

    def test_mutant_destructor_order_is_rejected(self):
        self.assertNotEqual(self.run_contract('missing_destructor_drain').returncode,0)


if __name__=='__main__':
    unittest.main(verbosity=2)
