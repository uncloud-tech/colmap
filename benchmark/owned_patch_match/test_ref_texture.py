# SPDX-License-Identifier: BSD-3-Clause
"""Actual texture header and owned preparation/bind methods with CPU CUDA stubs.
No CUDA runtime, device execution, ABI, or numerical equivalence is tested.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_cpu_execution import STUB
from test_rotation_reuse import body

ROOT = Path(__file__).resolve().parents[2]
MVS = ROOT / 'src/colmap/mvs'
CUDA = r'''
#include <cstring>
#include <map>
using cudaArray_t = void*;
using cudaTextureObject_t = uintptr_t;
struct cudaExtent { size_t width,height,depth; };
struct cudaPitchedPtr { void* ptr; size_t pitch,xsize,ysize; };
struct cudaChannelFormatDesc { size_t bytes; };
struct cudaTextureDesc { int addressMode[3]; int filterMode,readMode,normalizedCoords; };
struct cudaResourceDesc { int resType; struct { struct { cudaArray_t array; } array; } res; };
struct cudaMemcpy3DParms { cudaExtent extent; int kind; cudaPitchedPtr srcPtr; cudaArray_t dstArray; };
constexpr int cudaArrayLayered=1,cudaResourceTypeArray=2,cudaMemcpyDeviceToDevice=3,
 cudaMemcpyHostToDevice=4,cudaAddressModeBorder=5,cudaFilterModePoint=6,cudaReadModeNormalizedFloat=7;
inline cudaExtent make_cudaExtent(size_t w,size_t h,size_t d) { return {w,h,d}; }
inline cudaPitchedPtr make_cudaPitchedPtr(void* p,size_t pitch,size_t w,size_t h) { return {p,pitch,w,h}; }
template<class T> cudaChannelFormatDesc cudaCreateChannelDesc() { return {sizeof(T)}; }
namespace fake {
inline int allocations=0, creates=0, copies=0, fail_alloc=0,fail_create=0;
inline bool fail_copy=false;
inline int fail_wait=0;
inline std::map<void*,cudaExtent> arrays;
inline std::map<uintptr_t,void*> textures;
inline std::map<void*,int> contents;
}
inline int cudaMalloc3DArray(void** a,const cudaChannelFormatDesc* f,cudaExtent e,int flags) {
 assert(f->bytes==1 && flags==cudaArrayLayered);
 fake::log("alloc"); if(++fake::allocations==fake::fail_alloc) return 1;
 *a=reinterpret_cast<void*>(fake::next++); fake::arrays[*a]=e; return 0;
}
inline int cudaCreateTextureObject(uintptr_t* t,const cudaResourceDesc* r,const cudaTextureDesc* d,void*) {
 fake::log("texture"); if(++fake::creates==fake::fail_create) return 1;
 assert(r->resType==cudaResourceTypeArray);
 for(int mode:d->addressMode) assert(mode==cudaAddressModeBorder);
 assert(d->filterMode==cudaFilterModePoint && d->readMode==cudaReadModeNormalizedFloat && !d->normalizedCoords);
 *t=fake::next++; fake::textures[*t]=r->res.array.array; return 0;
}
inline int cudaDestroyTextureObject(uintptr_t t) {
 assert(!fake::pending); assert(fake::textures.erase(t)==1); fake::log("texture_free"); return 0;
}
inline int cudaFreeArray(void* a) {
 // A never-published array may be freed while unrelated preparation is queued.
 for(auto t:fake::textures) assert(t.second!=a);
 assert(fake::arrays.erase(a)==1); fake::contents.erase(a); fake::log("array_free"); return 0;
}
inline int cudaMemcpy3DAsync(const cudaMemcpy3DParms* p,void* s) {
 assert(s); const auto e=fake::arrays.at(p->dstArray);
 assert(p->extent.width==e.width && p->extent.height==e.height && p->extent.depth==e.depth);
 assert(p->srcPtr.pitch==64 && p->srcPtr.xsize==e.width && p->srcPtr.ysize==e.height);
 assert(p->kind==cudaMemcpyDeviceToDevice);
 fake::log("copy"); ++fake::copies; fake::pending=true;
 fake::contents[p->dstArray]=*static_cast<int*>(p->srcPtr.ptr);
 if(fake::fail_copy) { fake::fail_copy=false; return 1; } return 0;
}
inline int cudaMemcpy3D(const cudaMemcpy3DParms*) { return 0; }
'''
GPU_MAT = r'''
#pragma once
#include "colmap/mvs/owned_cuda_execution.h"
namespace colmap { namespace mvs {
template<class T> struct GpuMat {
 size_t w,h,d=1; int value=0;
 size_t GetWidth() const { return w; } size_t GetHeight() const { return h; }
 size_t GetDepth() const { return d; } size_t GetPitch() const { return 64; }
 const T* GetPtr() const { return reinterpret_cast<const T*>(&value); }
};
}}
'''
PROGRAM = r'''
#include "colmap/mvs/cuda_texture.h"
#include <memory>
namespace colmap { namespace mvs {
struct Bitmap { std::vector<uint8_t> RowMajorData() const { return {1}; } };
struct Image {
 size_t GetWidth() const { return 7; } size_t GetHeight() const { return 3; }
 Bitmap GetBitmap() const { return {}; }
};
struct PatchMatchOptions {
 std::string gpu_index="0"; int window_radius=5,window_step=1;
 bool geom_consistency=false; float sigma_spatial=1,sigma_color=1;
};
struct PatchMatch { struct Problem {
 const std::vector<Image>* images; int ref_image_idx=0; std::vector<int> src_image_idxs{1};
}; };
struct GpuMatRefImage {
 std::unique_ptr<GpuMat<uint8_t>> image;
 GpuMatRefImage(size_t w,size_t h):image(std::make_unique<GpuMat<uint8_t>>(GpuMat<uint8_t>{w,h})) {}
 void Filter(const uint8_t*,int,int,float,float,void*) { fake::pending=true; }
};
namespace owned {
inline int BuildOwnedRotationLayout(size_t,size_t,size_t) { return 0; }
enum class PatchMatchExecution { kOwnedCuda };
inline void ValidatePatchMatchExecution(PatchMatchExecution,int,int,bool,bool) {}
struct RefPair {
 std::unique_ptr<GpuMatRefImage> current;
 void Set(int,std::unique_ptr<GpuMatRefImage> p) { current=std::move(p); }
 GpuMatRefImage* operator->() { return current.get(); }
};
class PatchMatchCuda {
 public:
 PatchMatchCuda(const PatchMatchOptions&,const PatchMatch::Problem&); ~PatchMatchCuda();
 void InitRefImage(); void BindRefImageTexture();
 void InitSourceImages() {} void InitTransforms() {} void InitWorkspaceMemory() {}
 void InitRotationStorage() {} void EmitRotationAllocations() {}
 void DrainForCleanup() noexcept { stream_->DrainBeforeDestruction(); }
 PatchMatchOptions options_; PatchMatch::Problem problem_;
 std::unique_ptr<OwnedCudaStream> stream_;
 size_t ref_width_,ref_height_; int rotation_in_half_pi_,rotation_layout_;
 std::unique_ptr<CudaArrayLayeredTexture<uint8_t>> ref_image_textures_[2];
 CudaArrayLayeredTexture<uint8_t>* ref_image_texture_=nullptr;
 RefPair ref_image_;
};
// ACTUAL_METHODS
}}}
void Reset() {
 assert(fake::arrays.empty() && fake::textures.empty());
 fake::calls.clear(); fake::allocations=fake::creates=fake::copies=0;
 fake::fail_alloc=fake::fail_create=fake::fail_wait=0; fake::fail_copy=false;
}
int main(int argc, char**) {
 using namespace colmap::mvs; using namespace colmap::mvs::owned;
 std::vector<Image> images(2); PatchMatch::Problem problem{&images};
 if(argc>1) {
  std::set_terminate([] { std::_Exit(91); });
  fake::fail_wait=2; // Wait and guard drain both fail: no owner may unwind.
  PatchMatchCuda p({},problem); return 1;
 }
 for(int iterations:{0,1,3}) {
  Reset();
  {
   PatchMatchCuda p({},problem);
   assert(fake::arrays.size()==2 && fake::textures.size()==2 && fake::copies==1);
   assert(fake::contents.size()==1); // spare uninitialized
   auto even=p.ref_image_textures_[0]->GetObj(), odd=p.ref_image_textures_[1]->GetObj();
   assert(even!=odd);
   for(int i=1;i<=4*iterations;++i) {
    p.stream_->Synchronize(); // real pre-Rotate reader fence
    p.rotation_in_half_pi_=i%4;
    p.ref_image_->image->w=i%2?3:7; p.ref_image_->image->h=i%2?7:3;
    p.ref_image_->image->value=i; fake::pending=true; // same-stream rotate write
    p.BindRefImageTexture();
    assert(!fake::pending && p.ref_image_texture_->GetObj()==(i%2?odd:even));
    assert(fake::contents.at(fake::textures.at(p.ref_image_texture_->GetObj()))==i);
   }
   assert(fake::copies==1+4*iterations && fake::allocations==2 && fake::creates==2);
   p.DrainForCleanup(); assert(fake::arrays.size()==2);
  }
  assert(fake::arrays.empty() && fake::textures.empty());
 }
 for(int mode=0;mode<6;++mode) {
  Reset();
  if(mode==0) fake::fail_alloc=1;
  if(mode==1) fake::fail_alloc=2;
  if(mode==2) fake::fail_create=1;
  if(mode==3) fake::fail_create=2;
  if(mode==4) fake::fail_copy=true;
  if(mode==5) fake::fail_wait=1;
  bool caught=false;
  try { PatchMatchCuda p({},problem); } catch(const std::runtime_error&) { caught=true; }
  assert(caught && !fake::pending && fake::arrays.empty() && fake::textures.empty());
  bool synced=false, copied=false; int syncs=0;
  for(const auto& c:fake::calls) {
   if(c.name=="copy") copied=true;
   if(c.name=="sync") { synced=true; ++syncs; }
   if(c.name=="texture_free") assert(synced);
  }
  if(mode>=4) assert(copied && syncs>=(mode==5?3:2));
 }
 // Rotation-time failure keeps both resident owners through explicit drain.
 for(bool fail_copy:{false,true}) {
  Reset();
  {
   PatchMatchCuda p({},problem);
   p.stream_->Synchronize();
   p.rotation_in_half_pi_=1;
   p.ref_image_->image->w=3; p.ref_image_->image->h=7;
   fake::fail_copy=fail_copy; fake::fail_wait=fail_copy?0:1;
   bool caught=false;
   try { p.BindRefImageTexture(); } catch(const std::runtime_error&) { caught=true; }
   assert(caught && !fake::pending && fake::arrays.size()==2 && fake::textures.size()==2);
   p.DrainForCleanup(); assert(fake::arrays.size()==2);
  }
  assert(fake::arrays.empty());
 }
 // Legacy factory still delegates to the same waited copy boundary.
 Reset();
 cudaTextureDesc d{}; for(int& m:d.addressMode) m=cudaAddressModeBorder;
 d.filterMode=cudaFilterModePoint; d.readMode=cudaReadModeNormalizedFloat;
 { auto t=CudaArrayLayeredTexture<uint8_t>::FromGpuMat(d,GpuMat<uint8_t>{7,3},reinterpret_cast<void*>(1));
   assert(fake::copies==1 && !fake::pending);
   bool caught=false;
   try { t->CopyFromGpuMat(GpuMat<uint8_t>{3,7},reinterpret_cast<void*>(1)); }
   catch(const std::runtime_error&) { caught=true; }
   assert(caught && fake::copies==1);
 }
}
'''


class RefTexture(unittest.TestCase):
    def test_actual_header_preparation_refresh_and_failures(self):
        cu = (MVS / 'patch_match_cuda_owned.cu').read_text()
        methods = '\n'.join(body(cu, signature) for signature in (
            'PatchMatchCuda::PatchMatchCuda(', 'PatchMatchCuda::~PatchMatchCuda()',
            'void PatchMatchCuda::InitRefImage()', 'void PatchMatchCuda::BindRefImageTexture()'))
        # Stub only external CUDA/GpuMat boundaries; use real completion guard.
        stub = STUB.replace('assert(x); fake::pending=false;',
                            'assert(x); if(fake::fail_wait>0) { --fake::fail_wait; fake::log("sync"); return 1; } fake::pending=false;')
        stub = stub.replace('inline bool pending=false;', 'inline bool pending=false; inline int fail_wait=0;')
        cuda = CUDA.replace('inline int fail_wait=0;', '')
        with tempfile.TemporaryDirectory(prefix='ref-texture-cpu-') as tmp:
            tmp = Path(tmp)
            files = {
                'colmap/util/cudacc.h': stub + cuda,
                'colmap/util/logging.h': '#pragma once\n#include <sstream>\n#define LOG(x) std::ostringstream()\n#define THROW_CHECK_EQ(a,b) do { if((a)!=(b)) throw std::runtime_error("shape"); } while(0)\n#define THROW_CHECK_GT(a,b) do { if(!((a)>(b))) throw std::runtime_error("shape"); } while(0)\n',
                'colmap/util/string.h': '#pragma once\nnamespace colmap { template<class... T> std::string StringPrintf(const char* s,T...) { return s; } }\n',
                'colmap/mvs/gpu_mat.h': GPU_MAT,
                'test.cc': PROGRAM.replace('// ACTUAL_METHODS', methods),
            }
            for name, text in files.items():
                path = tmp / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            command = ['g++', '-std=c++17', '-I', str(tmp), '-I', str(ROOT / 'src'),
                       str(tmp / 'test.cc'), '-o', str(tmp / 'test')]
            build = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(tmp / 'test')], capture_output=True, text=True, timeout=5)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            unknown = subprocess.run([str(tmp / 'test'), 'unknown'], capture_output=True,
                                     text=True, timeout=5)
            self.assertEqual(unknown.returncode, 91, unknown.stdout + unknown.stderr)

    def test_source_fences_and_descriptor(self):
        cu = (MVS / 'patch_match_cuda_owned.cu').read_text()
        bind = body(cu, 'void PatchMatchCuda::BindRefImageTexture()')
        self.assertNotIn('make_unique', bind)
        self.assertNotIn('::FromGpuMat', bind)
        self.assertIn('rotation_in_half_pi_ % 2', bind)
        self.assertEqual(bind.count('CopyFromGpuMat('), 1)
        rotate = body(cu, 'void PatchMatchCuda::Rotate()')
        self.assertLess(rotate.index('ref_image_.Flip();'), rotate.index('BindRefImageTexture();'))
        run = body(cu, 'void PatchMatchCuda::RunWithWindowSizeAndStep()')
        self.assertIn('stream_->Synchronize();\n\n      Rotate();', run)
        # Preserve the original descriptor exactly, merely move it to prep.
        import test_rotation_reuse
        base = subprocess.check_output(['git', 'show', test_rotation_reuse.BASE +
                                        ':src/colmap/mvs/patch_match_cuda_owned.cu'], cwd=ROOT, text=True)
        old = body(base, 'void PatchMatchCuda::BindRefImageTexture()')
        descriptor = old[old.index('  cudaTextureDesc'):old.index('  ref_image_texture_ =')]
        self.assertIn(descriptor, body(cu, 'void PatchMatchCuda::InitRefImage()'))
        for name in ('EmitRotationAllocations',):
            sig = 'void PatchMatchCuda::' + name + '()'
            current = subprocess.check_output(['git', 'show', 'HEAD:src/colmap/mvs/patch_match_cuda_owned.cu'], cwd=ROOT, text=True)
            self.assertEqual(body(cu, sig), body(current, sig))


if __name__ == '__main__':
    unittest.main()
