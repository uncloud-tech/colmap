# SPDX-License-Identifier: BSD-3-Clause
"""Actual texture header and owned preparation/bind methods with CPU CUDA stubs.
No CUDA runtime, device execution, ABI, or numerical equivalence is tested.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

from test_cpu_execution import STUB
from test_rotation_reuse import body

ROOT = Path(__file__).resolve().parents[2]
MVS = ROOT / "src/colmap/mvs"
CUDA = (
    "\n"
    "#include <cstring>\n"
    "#include <map>\n"
    "using cudaArray_t = void*;\n"
    "using cudaTextureObject_t = uintptr_t;\n"
    "struct cudaExtent { size_t width,height,depth; };\n"
    "struct cudaPitchedPtr { void* ptr; size_t pitch,xsize,ysize;"
    " };\n"
    "struct cudaChannelFormatDesc { size_t bytes; };\n"
    "struct cudaTextureDesc { int addressMode[3]; int filterMode,"
    "readMode,normalizedCoords; };\n"
    "struct cudaResourceDesc { int resType; struct { struct { cud"
    "aArray_t array; } array; } res; };\n"
    "struct cudaMemcpy3DParms { cudaExtent extent; int kind; cuda"
    "PitchedPtr srcPtr; cudaArray_t dstArray; };\n"
    "constexpr int cudaArrayLayered=1,cudaResourceTypeArray=2,cud"
    "aMemcpyDeviceToDevice=3,\n"
    " cudaMemcpyHostToDevice=4,cudaAddressModeBorder=5,cudaFilter"
    "ModePoint=6,cudaReadModeNormalizedFloat=7;\n"
    "inline cudaExtent make_cudaExtent(size_t w,size_t h,size_t d"
    ") { return {w,h,d}; }\n"
    "inline cudaPitchedPtr make_cudaPitchedPtr(void* p,size_t pit"
    "ch,size_t w,size_t h) { return {p,pitch,w,h}; }\n"
    "template<class T> cudaChannelFormatDesc cudaCreateChannelDes"
    "c() { return {sizeof(T)}; }\n"
    "namespace fake {\n"
    "inline int allocations=0, creates=0, copies=0, fail_alloc=0,"
    "fail_create=0;\n"
    "inline bool fail_copy=false;\n"
    "inline int fail_wait=0;\n"
    "inline std::map<void*,cudaExtent> arrays;\n"
    "inline std::map<uintptr_t,void*> textures;\n"
    "inline std::map<void*,int> contents;\n"
    "}\n"
    "inline int cudaMalloc3DArray(void** a,const cudaChannelForma"
    "tDesc* f,cudaExtent e,int flags) {\n"
    " assert(f->bytes==1 && flags==cudaArrayLayered);\n"
    ' fake::log("alloc"); if(++fake::allocations==fake::fail_allo'
    "c) return 1;\n"
    " *a=reinterpret_cast<void*>(fake::next++); fake::arrays[*a]="
    "e; return 0;\n"
    "}\n"
    "inline int cudaCreateTextureObject(uintptr_t* t,const cudaRe"
    "sourceDesc* r,const cudaTextureDesc* d,void*) {\n"
    ' fake::log("texture"); if(++fake::creates==fake::fail_create'
    ") return 1;\n"
    " assert(r->resType==cudaResourceTypeArray);\n"
    " for(int mode:d->addressMode) assert(mode==cudaAddressModeBo"
    "rder);\n"
    " assert(d->filterMode==cudaFilterModePoint && d->readMode==c"
    "udaReadModeNormalizedFloat && !d->normalizedCoords);\n"
    " *t=fake::next++; fake::textures[*t]=r->res.array.array; ret"
    "urn 0;\n"
    "}\n"
    "inline int cudaDestroyTextureObject(uintptr_t t) {\n"
    " assert(!fake::pending); assert(fake::textures.erase(t)==1);"
    ' fake::log("texture_free"); return 0;\n'
    "}\n"
    "inline int cudaFreeArray(void* a) {\n"
    " // A never-published array may be freed while unrelated pre"
    "paration is queued.\n"
    " for(auto t:fake::textures) assert(t.second!=a);\n"
    " assert(fake::arrays.erase(a)==1); fake::contents.erase(a); "
    'fake::log("array_free"); return 0;\n'
    "}\n"
    "inline int cudaMemcpy3DAsync(const cudaMemcpy3DParms* p,void"
    "* s) {\n"
    " assert(s); const auto e=fake::arrays.at(p->dstArray);\n"
    " assert(p->extent.width==e.width && p->extent.height==e.heig"
    "ht && p->extent.depth==e.depth);\n"
    " assert(p->srcPtr.pitch==64 && p->srcPtr.xsize==e.width && p"
    "->srcPtr.ysize==e.height);\n"
    " assert(p->kind==cudaMemcpyDeviceToDevice);\n"
    ' fake::log("copy"); ++fake::copies; fake::pending=true;\n'
    " fake::contents[p->dstArray]=*static_cast<int*>(p->srcPtr.pt"
    "r);\n"
    " if(fake::fail_copy) { fake::fail_copy=false; return 1; } re"
    "turn 0;\n"
    "}\n"
    "inline int cudaMemcpy3D(const cudaMemcpy3DParms*) { return 0"
    "; }\n"
)
GPU_MAT = r"""
#pragma once
#include "colmap/mvs/owned_cuda_execution.h"
#include "colmap/mvs/owned_tile.h"
namespace colmap { namespace mvs {
template<class T> struct GpuMat {
 size_t w,h,d=1; int value=0;
 size_t GetWidth() const { return w; } size_t GetHeight() const { return h; }
 size_t GetDepth() const { return d; } size_t GetPitch() const { return 64; }
 const T* GetPtr() const { return reinterpret_cast<const T*>(&value); }
};
}}
"""
PROGRAM = (
    "\n"
    '#include "colmap/mvs/cuda_texture.h"\n'
    "#include <memory>\n"
    "namespace colmap { namespace mvs {\n"
    "struct Bitmap { std::vector<uint8_t> RowMajorData() const { "
    "return {1}; } };\n"
    "struct Image {\n"
    " size_t GetWidth() const { return 7; } size_t GetHeight() co"
    "nst { return 3; }\n"
    " Bitmap GetBitmap() const { return {}; }\n"
    "};\n"
    "struct PatchMatchOptions {\n"
    ' std::string gpu_index="0"; int window_radius=5,window_step='
    "1;\n"
    " bool geom_consistency=false; float sigma_spatial=1,sigma_co"
    "lor=1;\n"
    "};\n"
    "struct PatchMatch { struct Problem {\n"
    " const std::vector<Image>* images; int ref_image_idx=0; std:"
    ":vector<int> src_image_idxs{1};\n"
    "}; };\n"
    "struct GpuMatRefImage {\n"
    " std::unique_ptr<GpuMat<uint8_t>> image;\n"
    " GpuMatRefImage(size_t w,size_t h):image(std::make_unique<Gp"
    "uMat<uint8_t>>(GpuMat<uint8_t>{w,h})) {}\n"
    " void Filter(const uint8_t*,int,int,float,float,void*) { fak"
    "e::pending=true; }\n"
    "};\n"
    "namespace owned {\n"
    "inline int BuildOwnedRotationLayout(size_t,size_t,size_t) { "
    "return 0; }\n"
    "enum class PatchMatchExecution { kOwnedCuda };\n"
    "inline void ValidatePatchMatchExecution(PatchMatchExecution,"
    "int,int,bool,bool) {}\n"
    "struct RefPair {\n"
    " std::unique_ptr<GpuMatRefImage> current;\n"
    " void Set(int,std::unique_ptr<GpuMatRefImage> p) { current=s"
    "td::move(p); }\n"
    " GpuMatRefImage* operator->() { return current.get(); }\n"
    "};\n"
    "class PatchMatchCuda {\n"
    " public:\n"
    " PatchMatchCuda(const PatchMatchOptions&,const PatchMatch::P"
    "roblem&,int=32); ~PatchMatchCuda();\n"
    " void InitRefImage(); void BindRefImageTexture();\n"
    " void InitSourceImages() {} void InitTransforms() {} void In"
    "itWorkspaceMemory() {}\n"
    " void InitRotationStorage() {} void EmitRotationAllocations("
    ") {}\n"
    " void DrainForCleanup() noexcept { stream_->DrainBeforeDestr"
    "uction(); }\n"
    " PatchMatchOptions options_; const int owned_tile_threads_; "
    "PatchMatch::Problem problem_;\n"
    " std::unique_ptr<OwnedCudaStream> stream_;\n"
    " size_t ref_width_,ref_height_; int rotation_in_half_pi_,rot"
    "ation_layout_;\n"
    " std::unique_ptr<CudaArrayLayeredTexture<uint8_t>> ref_image"
    "_textures_[2];\n"
    " CudaArrayLayeredTexture<uint8_t>* ref_image_texture_=nullpt"
    "r;\n"
    " RefPair ref_image_;\n"
    "};\n"
    "// ACTUAL_METHODS\n"
    "}}}\n"
    "void Reset() {\n"
    " assert(fake::arrays.empty() && fake::textures.empty());\n"
    " fake::calls.clear(); fake::allocations=fake::creates=fake::"
    "copies=0;\n"
    " fake::fail_alloc=fake::fail_create=fake::fail_wait=0; fake:"
    ":fail_copy=false;\n"
    "}\n"
    "int main(int argc, char**) {\n"
    " using namespace colmap::mvs; using namespace colmap::mvs::o"
    "wned;\n"
    " std::vector<Image> images(2); PatchMatch::Problem problem{&"
    "images};\n"
    " if(argc>1) {\n"
    "  std::set_terminate([] { std::_Exit(91); });\n"
    "  fake::fail_wait=2; // Wait and guard drain both fail: no o"
    "wner may unwind.\n"
    "  PatchMatchCuda p({},problem); return 1;\n"
    " }\n"
    " for(int iterations:{0,1,3}) {\n"
    "  Reset();\n"
    "  {\n"
    "   PatchMatchCuda p({},problem);\n"
    "   assert(fake::arrays.size()==2 && fake::textures.size()==2"
    " && fake::copies==1);\n"
    "   assert(fake::contents.size()==1); // spare uninitialized\n"
    "   auto even=p.ref_image_textures_[0]->GetObj(), odd=p.ref_i"
    "mage_textures_[1]->GetObj();\n"
    "   assert(even!=odd);\n"
    "   for(int i=1;i<=4*iterations;++i) {\n"
    "    p.stream_->Synchronize(); // real pre-Rotate reader fenc"
    "e\n"
    "    p.rotation_in_half_pi_=i%4;\n"
    "    p.ref_image_->image->w=i%2?3:7; p.ref_image_->image->h=i"
    "%2?7:3;\n"
    "    p.ref_image_->image->value=i; fake::pending=true; // sam"
    "e-stream rotate write\n"
    "    p.BindRefImageTexture();\n"
    "    assert(!fake::pending && p.ref_image_texture_->GetObj()="
    "=(i%2?odd:even));\n"
    "    assert(fake::contents.at(fake::textures.at(p.ref_image_t"
    "exture_->GetObj()))==i);\n"
    "   }\n"
    "   assert(fake::copies==1+4*iterations && fake::allocations="
    "=2 && fake::creates==2);\n"
    "   p.DrainForCleanup(); assert(fake::arrays.size()==2);\n"
    "  }\n"
    "  assert(fake::arrays.empty() && fake::textures.empty());\n"
    " }\n"
    " for(int mode=0;mode<6;++mode) {\n"
    "  Reset();\n"
    "  if(mode==0) fake::fail_alloc=1;\n"
    "  if(mode==1) fake::fail_alloc=2;\n"
    "  if(mode==2) fake::fail_create=1;\n"
    "  if(mode==3) fake::fail_create=2;\n"
    "  if(mode==4) fake::fail_copy=true;\n"
    "  if(mode==5) fake::fail_wait=1;\n"
    "  bool caught=false;\n"
    "  try { PatchMatchCuda p({},problem); } catch(const std::run"
    "time_error&) { caught=true; }\n"
    "  assert(caught && !fake::pending && fake::arrays.empty() &&"
    " fake::textures.empty());\n"
    "  bool synced=false, copied=false; int syncs=0;\n"
    "  for(const auto& c:fake::calls) {\n"
    '   if(c.name=="copy") copied=true;\n'
    '   if(c.name=="sync") { synced=true; ++syncs; }\n'
    '   if(c.name=="texture_free") assert(synced);\n'
    "  }\n"
    "  if(mode>=4) assert(copied && syncs>=(mode==5?3:2));\n"
    " }\n"
    " // Rotation-time failure keeps both resident owners through"
    " explicit drain.\n"
    " for(bool fail_copy:{false,true}) {\n"
    "  Reset();\n"
    "  {\n"
    "   PatchMatchCuda p({},problem);\n"
    "   p.stream_->Synchronize();\n"
    "   p.rotation_in_half_pi_=1;\n"
    "   p.ref_image_->image->w=3; p.ref_image_->image->h=7;\n"
    "   fake::fail_copy=fail_copy; fake::fail_wait=fail_copy?0:1;"
    "\n"
    "   bool caught=false;\n"
    "   try { p.BindRefImageTexture(); } catch(const std::runtime"
    "_error&) { caught=true; }\n"
    "   assert(caught && !fake::pending && fake::arrays.size()==2"
    " && fake::textures.size()==2);\n"
    "   p.DrainForCleanup(); assert(fake::arrays.size()==2);\n"
    "  }\n"
    "  assert(fake::arrays.empty());\n"
    " }\n"
    " // Legacy factory still delegates to the same waited copy b"
    "oundary.\n"
    " Reset();\n"
    " cudaTextureDesc d{}; for(int& m:d.addressMode) m=cudaAddres"
    "sModeBorder;\n"
    " d.filterMode=cudaFilterModePoint; d.readMode=cudaReadModeNo"
    "rmalizedFloat;\n"
    " { auto t=CudaArrayLayeredTexture<uint8_t>::FromGpuMat(d,Gpu"
    "Mat<uint8_t>{7,3},reinterpret_cast<void*>(1));\n"
    "   assert(fake::copies==1 && !fake::pending);\n"
    "   bool caught=false;\n"
    "   try { t->CopyFromGpuMat(GpuMat<uint8_t>{3,7},reinterpret_"
    "cast<void*>(1)); }\n"
    "   catch(const std::runtime_error&) { caught=true; }\n"
    "   assert(caught && fake::copies==1);\n"
    " }\n"
    "}\n"
)


class RefTexture(unittest.TestCase):
    def test_actual_header_preparation_refresh_and_failures(self):
        cu = (MVS / "patch_match_cuda_owned.cu").read_text()
        methods = "\n".join(
            body(cu, signature)
            for signature in (
                "PatchMatchCuda::PatchMatchCuda(",
                "PatchMatchCuda::~PatchMatchCuda()",
                "void PatchMatchCuda::InitRefImage()",
                "void PatchMatchCuda::BindRefImageTexture()",
            )
        )
        # Stub only external CUDA/GpuMat boundaries; use real completion guard.
        stub = STUB.replace(
            "assert(x); fake::pending=false;",
            (
                "assert(x); if(fake::fail_wait>0) { --fake::fail_wait; fake::"
                'log("sync"); return 1; } fake::pending=false;'
            ),
        )
        stub = stub.replace(
            "inline bool pending=false;",
            "inline bool pending=false; inline int fail_wait=0;",
        )
        cuda = CUDA.replace("inline int fail_wait=0;", "")
        with tempfile.TemporaryDirectory(prefix="ref-texture-cpu-") as tmp:
            tmp = Path(tmp)
            files = {
                "colmap/util/cudacc.h": stub + cuda,
                ("colmap/util/logging.h"): (
                    "#pragma once\n"
                    "#include <sstream>\n"
                    "#define LOG(x) std::ostringstream()\n"
                    "#define THROW_CHECK_EQ(a,b) do { if((a)!=(b)) throw std:"
                    ':runtime_error("shape"); } while(0)\n'
                    "#define THROW_CHECK_GT(a,b) do { if(!((a)>(b))) throw st"
                    'd::runtime_error("shape"); } while(0)\n'
                ),
                ("colmap/util/string.h"): (
                    "#pragma once\n"
                    "namespace colmap { template<class... T> std::string Stri"
                    "ngPrintf(const char* s,T...) { return s; } }\n"
                ),
                "colmap/mvs/gpu_mat.h": GPU_MAT,
                "test.cc": PROGRAM.replace("// ACTUAL_METHODS", methods),
            }
            for name, text in files.items():
                path = tmp / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            command = [
                "g++",
                "-std=c++17",
                "-I",
                str(tmp),
                "-I",
                str(ROOT / "src"),
                str(tmp / "test.cc"),
                "-o",
                str(tmp / "test"),
            ]
            build = subprocess.run(
                command, capture_output=True, text=True, timeout=30
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run(
                [str(tmp / "test")], capture_output=True, text=True, timeout=5
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            unknown = subprocess.run(
                [str(tmp / "test"), "unknown"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(
                unknown.returncode, 91, unknown.stdout + unknown.stderr
            )

    def test_source_fences_and_descriptor(self):
        cu = (MVS / "patch_match_cuda_owned.cu").read_text()
        bind = body(cu, "void PatchMatchCuda::BindRefImageTexture()")
        self.assertNotIn("make_unique", bind)
        self.assertNotIn("::FromGpuMat", bind)
        self.assertIn("rotation_in_half_pi_ % 2", bind)
        self.assertEqual(bind.count("CopyFromGpuMat("), 1)
        rotate = body(cu, "void PatchMatchCuda::Rotate()")
        self.assertLess(
            rotate.index("ref_image_.Flip();"),
            rotate.index("BindRefImageTexture();"),
        )
        run = body(cu, "void PatchMatchCuda::RunWithWindowSizeAndStep()")
        self.assertIn("stream_->Synchronize();\n\n      Rotate();", run)
        # Preserve the original descriptor exactly, merely move it to prep.
        import test_rotation_reuse

        base = subprocess.check_output(
            [
                "git",
                "show",
                test_rotation_reuse.BASE
                + ":src/colmap/mvs/patch_match_cuda_owned.cu",
            ],
            cwd=ROOT,
            text=True,
        )
        old = body(base, "void PatchMatchCuda::BindRefImageTexture()")
        descriptor = old[
            old.index("  cudaTextureDesc") : old.index("  ref_image_texture_ =")
        ]
        self.assertIn(
            descriptor, body(cu, "void PatchMatchCuda::InitRefImage()")
        )
        for name in ("EmitRotationAllocations",):
            sig = "void PatchMatchCuda::" + name + "()"
            current = subprocess.check_output(
                [
                    "git",
                    "show",
                    "HEAD:src/colmap/mvs/patch_match_cuda_owned.cu",
                ],
                cwd=ROOT,
                text=True,
            )
            self.assertEqual(body(cu, sig), body(current, sig))


if __name__ == "__main__":
    unittest.main()
