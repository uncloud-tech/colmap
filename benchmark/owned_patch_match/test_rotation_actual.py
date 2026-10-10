# SPDX-License-Identifier: BSD-3-Clause
"""Compile actual spare/Rotate/record bodies with host buffer spies, not CUDA."""
from pathlib import Path
import re
import resource
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CU = ROOT / 'src/colmap/mvs/patch_match_cuda_owned.cu'


def body(text, signature):
    start = text.index(signature)
    pos = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[pos] == '{') - (text[pos] == '}')
        pos += 1
    return text[start:pos]


CPP = r'''
#include "colmap/mvs/owned_rotation_storage.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
using namespace colmap::mvs::owned;
int allocations=0, frees=0, initializations=0, normals=0;
struct curandState { uint32_t words[12]; };
std::vector<std::string> logs;
struct Log {
 std::ostringstream text;
 template<class T> Log& operator<<(const T& value) { text<<value; return *this; }
 ~Log() { logs.push_back(text.str()); }
};
#define LOG(x) Log()
#define THROW_CHECK_EQ(a,b) assert((a)==(b))
#define THROW_CHECK_GE(a,b) assert((a)>=(b))
struct Stream { int Get() const { return 7; } };
template<class T> struct GpuMat {
 size_t w,h,d,pitch; int id; std::vector<int> cells;
 GpuMat(size_t width,size_t height,size_t depth=1): w(width),h(height),d(depth),
   pitch(((width*sizeof(T)+63)/64)*64 + 64*(allocations+1)),id(++allocations),cells(w*h*d,-777) {}
 ~GpuMat() { ++frees; }
 size_t GetWidth() const { return w; } size_t GetHeight() const { return h; }
 size_t GetDepth() const { return d; } size_t GetPitch() const { return pitch; }
 const T* GetPtr() const { return nullptr; } // used only within unevaluated sizeof
 GpuMat* View() { return this; }
 void FillWithScalar(int v, int stream) { assert(stream==7); std::fill(cells.begin(),cells.end(),v); }
 void Rotate(GpuMat* target,int stream) {
   assert(stream==7 && target!=this && target->id!=id);
   assert(target->w==h && target->h==w && target->d==d);
   for(size_t z=0;z<d;++z) for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x)
     target->cells[z*w*h+(w-1-x)*h+y]=cells[z*w*h+y*w+x];
 }
};
struct GpuMatPRNG: GpuMat<curandState> {
 struct UninitializedTag {};
 GpuMatPRNG(size_t w,size_t h,int stream):GpuMat(w,h) { Reinitialize(stream); }
 GpuMatPRNG(size_t w,size_t h,UninitializedTag):GpuMat(w,h) {}
 void Reinitialize(int stream) { assert(stream==7); ++initializations; std::fill(cells.begin(),cells.end(),-888); }
};
struct GpuMatRefImage {
 std::unique_ptr<GpuMat<uint8_t>> image;
 std::unique_ptr<GpuMat<float>> sum_image,squared_sum_image;
 GpuMatRefImage(size_t w,size_t h):image(new GpuMat<uint8_t>(w,h)),
   sum_image(new GpuMat<float>(w,h)),squared_sum_image(new GpuMat<float>(w,h)) {}
};
void RotateNormalMap(GpuMat<float>* normal) { assert(normal->d==3); ++normals; }
class PatchMatchCuda {
 public:
 struct Problem { int ref_image_idx=7; std::vector<int> src_image_idxs; } problem_;
 size_t ref_width_,ref_height_; int rotation_in_half_pi_=0,binds=0,configs=0;
 std::unique_ptr<Stream> stream_=std::make_unique<Stream>();
 OwnedRotationLayout rotation_layout_;
 OwnedRotationPair<GpuMatPRNG> rand_state_map_;
 OwnedRotationPair<GpuMatRefImage> ref_image_;
 OwnedRotationPair<GpuMat<float>> depth_map_,normal_map_,sel_prob_map_,prev_sel_prob_map_,cost_map_;
 OwnedRotationPair<GpuMat<uint8_t>> consistency_mask_;
 std::unique_ptr<GpuMat<float>> global_workspace_;
 PatchMatchCuda(size_t w,size_t h,size_t s):ref_width_(w),ref_height_(h),rotation_layout_(BuildOwnedRotationLayout(w,h,s)) {
   problem_.src_image_idxs.resize(s);
   rand_state_map_.Set(0,std::make_unique<GpuMatPRNG>(w,h,7));
   ref_image_.Set(0,std::make_unique<GpuMatRefImage>(w,h));
   depth_map_.Set(0,std::make_unique<GpuMat<float>>(w,h));
   normal_map_.Set(0,std::make_unique<GpuMat<float>>(w,h,3));
   sel_prob_map_.Set(0,std::make_unique<GpuMat<float>>(w,h,s));
   prev_sel_prob_map_.Set(0,std::make_unique<GpuMat<float>>(w,h,s));
   cost_map_.Set(0,std::make_unique<GpuMat<float>>(w,h,s));
   global_workspace_=std::make_unique<GpuMat<float>>(std::max(w,h),s,2);
   InitRotationStorage(); EmitRotationAllocations();
 }
 void InitRotationStorage(); void EmitRotationAllocations(); void Rotate();
 void BindRefImageTexture() { ++binds; assert(ref_image_->image->w==depth_map_->w); }
 void ComputeCudaConfig() { ++configs; assert(cost_map_->w==depth_map_->w); }
};
// ACTUAL_METHODS
std::vector<int> Rotated(const std::vector<int>& cells,size_t w,size_t h,size_t d) {
 std::vector<int> result(cells.size());
 for(size_t z=0;z<d;++z) for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x)
   result[z*w*h+(w-1-x)*h+y]=cells[z*w*h+y*w+x];
 return result;
}
void Check(size_t w,size_t h,size_t sources,bool filter) {
 allocations=frees=initializations=normals=0; logs.clear();
 {
  PatchMatchCuda p(w,h,sources);
  assert(allocations==21 && frees==0 && initializations==1 && logs.size()==21);
  for(const auto& line:logs) std::cout<<line<<'\n';
  std::vector<int> mask_expected;
  for(int step=0;step<20;++step) {
   auto* current=p.sel_prob_map_.Current(); auto* previous=p.prev_sel_prob_map_.Current();
   assert(current!=previous && current!=p.prev_sel_prob_map_.Alternate());
   for(size_t i=0;i<current->cells.size();++i) current->cells[i]=100000*step+int(i);
   const auto expected=Rotated(current->cells,current->w,current->h,current->d);
   if(filter && step==19) {
     p.consistency_mask_->FillWithScalar(0,7);
     for(size_t i=0;i<p.consistency_mask_->cells.size();++i) p.consistency_mask_->cells[i]=int(i)%3;
     mask_expected=Rotated(p.consistency_mask_->cells,current->w,current->h,current->d);
   }
   p.Rotate();
   assert(p.prev_sel_prob_map_->cells==expected);
   assert(p.rotation_in_half_pi_==(step+1)%4);
   for(unsigned parity:{p.rand_state_map_.CurrentParity(),p.depth_map_.CurrentParity(),
       p.normal_map_.CurrentParity(),p.ref_image_.CurrentParity(),p.cost_map_.CurrentParity(),
       p.sel_prob_map_.CurrentParity(),p.prev_sel_prob_map_.CurrentParity(),p.consistency_mask_.CurrentParity()})
     assert(parity==unsigned((step+1)%2));
   if(filter && step==19) {
     auto& consistency_mask_=p.consistency_mask_; auto& stream_=p.stream_;
     // ACTUAL_FINAL_MASK_ROTATION
     assert(p.consistency_mask_->cells==mask_expected);
   }
   assert(allocations==21 && frees==0);
  }
  assert(initializations==21 && normals==20 && p.binds==20 && p.configs==20);
  if(!filter) for(unsigned parity:{0,1}) for(int v:p.consistency_mask_.At(parity)->cells) assert(v==-777);
 }
 assert(frees==21);
}
int main() { Check(35,19,2,true); Check(17,17,3,false); }
'''


class ActualRotation(unittest.TestCase):
    def program(self, text):
        methods = '\n'.join(body(text, 'void PatchMatchCuda::' + name + '()')
                            for name in ('InitRotationStorage', 'EmitRotationAllocations', 'Rotate'))
        methods = re.sub(r'<<<.*?>>>', '', methods, flags=re.S)
        mask = re.search(r'consistency_mask_\.Alternate\(\)->Rotate\([^;]+;', text).group(0)
        return CPP.replace('// ACTUAL_METHODS', methods).replace('// ACTUAL_FINAL_MASK_ROTATION', mask)

    def compile_run(self, program):
        with tempfile.TemporaryDirectory(prefix='rotation-actual-') as tmp:
            src, exe = Path(tmp) / 'test.cc', Path(tmp) / 'test'
            src.write_text(program)
            build = subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'src'),
                                    str(src), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            return subprocess.run([str(exe)], capture_output=True, text=True, timeout=10,
                                  preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))

    def test_actual_bodies_and_literal_records(self):
        run = self.compile_run(self.program(CU.read_text()))
        self.assertEqual(run.returncode, 0, run.stderr)
        lines = run.stdout.splitlines()
        self.assertEqual(len(lines), 42)
        keys = 'reference role parity width height depth element_bytes pitch_bytes requested_span_bytes pitched_span_bytes phase'.split()
        for line in lines:
            prefix, *fields = line.split(' ')
            self.assertEqual(prefix, 'OWNED_ROTATION_ALLOCATION')
            self.assertEqual([field.split('=')[0] for field in fields], keys)
            values = dict(field.split('=') for field in fields)
            self.assertEqual(values['phase'], 'prepared_through_outputs')
            for key in keys:
                if key not in ('role', 'phase'):
                    self.assertRegex(values[key], r'^(0|[1-9][0-9]*)$')
            self.assertEqual(int(values['pitched_span_bytes']), int(values['pitch_bytes']) * int(values['height']) * int(values['depth']))

    def test_runtime_mutants_rejected(self):
        text = CU.read_text()
        program = self.program(text)
        mutants = [
            ('rand_state_map_.Alternate()->Reinitialize(stream_->Get());', ''),
            ('prev_sel_prob_map_.Flip();', ''),
            ('consistency_mask_.Flip();', ''),
            ('sel_prob_map_->Rotate(prev_sel_prob_map_.Alternate(), stream_->Get());',
             'sel_prob_map_->Rotate(sel_prob_map_.Alternate(), stream_->Get());'),
            ('consistency_mask_.Alternate()->Rotate(consistency_mask_.Current(), stream_->Get());',
             'consistency_mask_.Current()->Rotate(consistency_mask_.Alternate(), stream_->Get());'),
        ]
        for old, new in mutants:
            with self.subTest(mutation=old):
                self.assertEqual(program.count(old), 1)
                run = self.compile_run(program.replace(old, new))
                self.assertNotEqual(run.returncode, 0, 'Mutant survived: ' + old)


if __name__ == '__main__':
    unittest.main()
