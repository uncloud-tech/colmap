# SPDX-License-Identifier: BSD-3-Clause
"Compile actual spare/Rotate/record bodies with host buffer spies, not CUDA."

import re
import resource
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CU = ROOT / "src/colmap/mvs/patch_match_cuda_owned.cu"


def body(text, signature):
    start = text.index(signature)
    pos = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[pos] == "{") - (text[pos] == "}")
        pos += 1
    return text[start:pos]


CPP = (
    "\n"
    '#include "colmap/mvs/owned_rotation_storage.h"\n'
    "#include <algorithm>\n"
    "#include <cassert>\n"
    "#include <cstdint>\n"
    "#include <iostream>\n"
    "#include <locale>\n"
    "#include <memory>\n"
    "#include <sstream>\n"
    "#include <string>\n"
    "#include <vector>\n"
    "using namespace colmap::mvs::owned;\n"
    "int allocations=0, frees=0, initializations=0, normals=0;\n"
    "struct curandState { uint32_t words[12]; };\n"
    "std::vector<std::string> logs;\n"
    "struct Log {\n"
    " std::ostringstream text;\n"
    " template<class T> Log& operator<<(const T& value) { text<<v"
    "alue; return *this; }\n"
    " ~Log() { logs.push_back(text.str()); }\n"
    "};\n"
    "#define LOG(x) Log()\n"
    "#define THROW_CHECK_EQ(a,b) assert((a)==(b))\n"
    "#define THROW_CHECK_GE(a,b) assert((a)>=(b))\n"
    "struct Stream { int Get() const { return 7; } };\n"
    "template<class T> struct GpuMat {\n"
    " size_t w,h,d,pitch; int id; std::vector<int> cells;\n"
    " GpuMat(size_t width,size_t height,size_t depth=1): w(width)"
    ",h(height),d(depth),\n"
    "   pitch(((width*sizeof(T)+63)/64)*64 + 64*(allocations+1)),"
    "id(++allocations),cells(w*h*d,-777) {}\n"
    " ~GpuMat() { ++frees; }\n"
    " size_t GetWidth() const { return w; } size_t GetHeight() co"
    "nst { return h; }\n"
    " size_t GetDepth() const { return d; } size_t GetPitch() con"
    "st { return pitch; }\n"
    " const T* GetPtr() const { return nullptr; } // used only wi"
    "thin unevaluated sizeof\n"
    " GpuMat* View() { return this; }\n"
    " void FillWithScalar(int v, int stream) { assert(stream==7);"
    " std::fill(cells.begin(),cells.end(),v); }\n"
    " void Rotate(GpuMat* target,int stream) {\n"
    "   assert(stream==7 && target!=this && target->id!=id);\n"
    "   assert(target->w==h && target->h==w && target->d==d);\n"
    "   for(size_t z=0;z<d;++z) for(size_t y=0;y<h;++y) for(size_"
    "t x=0;x<w;++x)\n"
    "     target->cells[z*w*h+(w-1-x)*h+y]=cells[z*w*h+y*w+x];\n"
    " }\n"
    "};\n"
    "struct GpuMatPRNG: GpuMat<curandState> {\n"
    " struct UninitializedTag {};\n"
    " GpuMatPRNG(size_t w,size_t h,int stream):GpuMat(w,h) { Rein"
    "itialize(stream); }\n"
    " GpuMatPRNG(size_t w,size_t h,UninitializedTag):GpuMat(w,h) "
    "{}\n"
    " void Reinitialize(int stream) { assert(stream==7); ++initia"
    "lizations; std::fill(cells.begin(),cells.end(),-888); }\n"
    "};\n"
    "struct GpuMatRefImage {\n"
    " std::unique_ptr<GpuMat<uint8_t>> image;\n"
    " std::unique_ptr<GpuMat<float>> sum_image,squared_sum_image;"
    "\n"
    " GpuMatRefImage(size_t w,size_t h):image(new GpuMat<uint8_t>"
    "(w,h)),\n"
    "   sum_image(new GpuMat<float>(w,h)),squared_sum_image(new G"
    "puMat<float>(w,h)) {}\n"
    "};\n"
    "void RotateNormalMap(GpuMat<float>* normal) { assert(normal-"
    ">d==3); ++normals; }\n"
    "class PatchMatchCuda {\n"
    " public:\n"
    " struct Problem { int ref_image_idx=7; std::vector<int> src_"
    "image_idxs; } problem_;\n"
    " size_t ref_width_,ref_height_; int rotation_in_half_pi_=0,b"
    "inds=0,configs=0;\n"
    " std::unique_ptr<Stream> stream_=std::make_unique<Stream>();"
    "\n"
    " OwnedRotationLayout rotation_layout_;\n"
    " OwnedRotationPair<GpuMatPRNG> rand_state_map_;\n"
    " OwnedRotationPair<GpuMatRefImage> ref_image_;\n"
    " OwnedRotationPair<GpuMat<float>> depth_map_,normal_map_,sel"
    "_prob_map_,prev_sel_prob_map_,cost_map_;\n"
    " OwnedRotationPair<GpuMat<uint8_t>> consistency_mask_;\n"
    " std::unique_ptr<GpuMat<float>> global_workspace_;\n"
    " PatchMatchCuda(size_t w,size_t h,size_t s):ref_width_(w),re"
    "f_height_(h),rotation_layout_(BuildOwnedRotationLayout(w,h,s"
    ")) {\n"
    "   problem_.src_image_idxs.resize(s);\n"
    "   rand_state_map_.Set(0,std::make_unique<GpuMatPRNG>(w,h,7)"
    ");\n"
    "   ref_image_.Set(0,std::make_unique<GpuMatRefImage>(w,h));\n"
    "   depth_map_.Set(0,std::make_unique<GpuMat<float>>(w,h));\n"
    "   normal_map_.Set(0,std::make_unique<GpuMat<float>>(w,h,3))"
    ";\n"
    "   sel_prob_map_.Set(0,std::make_unique<GpuMat<float>>(w,h,s"
    "));\n"
    "   prev_sel_prob_map_.Set(0,std::make_unique<GpuMat<float>>("
    "w,h,s));\n"
    "   cost_map_.Set(0,std::make_unique<GpuMat<float>>(w,h,s));\n"
    "   global_workspace_=std::make_unique<GpuMat<float>>(std::ma"
    "x(w,h),s,2);\n"
    "   InitRotationStorage(); EmitRotationAllocations();\n"
    " }\n"
    " void InitRotationStorage(); void EmitRotationAllocations();"
    " void Rotate();\n"
    " void BindRefImageTexture() { ++binds; assert(ref_image_->im"
    "age->w==depth_map_->w); }\n"
    " void ComputeCudaConfig() { ++configs; assert(cost_map_->w=="
    "depth_map_->w); }\n"
    "};\n"
    "// ACTUAL_METHODS\n"
    "std::vector<int> Rotated(const std::vector<int>& cells,size_"
    "t w,size_t h,size_t d) {\n"
    " std::vector<int> result(cells.size());\n"
    " for(size_t z=0;z<d;++z) for(size_t y=0;y<h;++y) for(size_t "
    "x=0;x<w;++x)\n"
    "   result[z*w*h+(w-1-x)*h+y]=cells[z*w*h+y*w+x];\n"
    " return result;\n"
    "}\n"
    "void Check(size_t w,size_t h,size_t sources,bool filter) {\n"
    " allocations=frees=initializations=normals=0; logs.clear();\n"
    " {\n"
    "  PatchMatchCuda p(w,h,sources);\n"
    "  assert(allocations==21 && frees==0 && initializations==1 &"
    "& logs.size()==21);\n"
    "  for(const auto& line:logs) std::cout<<line<<'\\n';\n"
    "  std::vector<int> mask_expected;\n"
    "  for(int step=0;step<20;++step) {\n"
    "   auto* current=p.sel_prob_map_.Current(); auto* previous=p"
    ".prev_sel_prob_map_.Current();\n"
    "   assert(current!=previous && current!=p.prev_sel_prob_map_"
    ".Alternate());\n"
    "   for(size_t i=0;i<current->cells.size();++i) current->cell"
    "s[i]=100000*step+int(i);\n"
    "   const auto expected=Rotated(current->cells,current->w,cur"
    "rent->h,current->d);\n"
    "   if(filter && step==19) {\n"
    "     p.consistency_mask_->FillWithScalar(0,7);\n"
    "     for(size_t i=0;i<p.consistency_mask_->cells.size();++i)"
    " p.consistency_mask_->cells[i]=int(i)%3;\n"
    "     mask_expected=Rotated(p.consistency_mask_->cells,curren"
    "t->w,current->h,current->d);\n"
    "   }\n"
    "   p.Rotate();\n"
    "   assert(p.prev_sel_prob_map_->cells==expected);\n"
    "   assert(p.rotation_in_half_pi_==(step+1)%4);\n"
    "   for(unsigned parity:{p.rand_state_map_.CurrentParity(),p."
    "depth_map_.CurrentParity(),\n"
    "       p.normal_map_.CurrentParity(),p.ref_image_.CurrentPar"
    "ity(),p.cost_map_.CurrentParity(),\n"
    "       p.sel_prob_map_.CurrentParity(),p.prev_sel_prob_map_."
    "CurrentParity(),p.consistency_mask_.CurrentParity()})\n"
    "     assert(parity==unsigned((step+1)%2));\n"
    "   if(filter && step==19) {\n"
    "     auto& consistency_mask_=p.consistency_mask_; auto& stre"
    "am_=p.stream_;\n"
    "     // ACTUAL_FINAL_MASK_ROTATION\n"
    "     assert(p.consistency_mask_->cells==mask_expected);\n"
    "   }\n"
    "   assert(allocations==21 && frees==0);\n"
    "  }\n"
    "  assert(initializations==21 && normals==20 && p.binds==20 &"
    "& p.configs==20);\n"
    "  if(!filter) for(unsigned parity:{0,1}) for(int v:p.consist"
    "ency_mask_.At(parity)->cells) assert(v==-777);\n"
    " }\n"
    " assert(frees==21);\n"
    "}\n"
    "int main() { Check(35,19,2,true); Check(17,17,3,false); }\n"
)


class ActualRotation(unittest.TestCase):
    def program(self, text):
        methods = "\n".join(
            body(text, "void PatchMatchCuda::" + name + "()")
            for name in (
                "InitRotationStorage",
                "EmitRotationAllocations",
                "Rotate",
            )
        )
        methods = re.sub(r"<<<.*?>>>", "", methods, flags=re.S)
        mask = re.search(
            r"consistency_mask_\.Alternate\(\)->Rotate\([^;]+;", text
        ).group(0)
        return CPP.replace("// ACTUAL_METHODS", methods).replace(
            "// ACTUAL_FINAL_MASK_ROTATION", mask
        )

    def compile_run(self, program):
        with tempfile.TemporaryDirectory(prefix="rotation-actual-") as tmp:
            src, exe = Path(tmp) / "test.cc", Path(tmp) / "test"
            src.write_text(program)
            build = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(ROOT / "src"),
                    str(src),
                    "-o",
                    str(exe),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            return subprocess.run(
                [str(exe)],
                capture_output=True,
                text=True,
                timeout=10,
                preexec_fn=lambda: resource.setrlimit(
                    resource.RLIMIT_CORE, (0, 0)
                ),
            )

    def test_actual_bodies_and_literal_records(self):
        run = self.compile_run(self.program(CU.read_text()))
        self.assertEqual(run.returncode, 0, run.stderr)
        lines = run.stdout.splitlines()
        self.assertEqual(len(lines), 42)
        keys = [
            "reference",
            "role",
            "parity",
            "width",
            "height",
            "depth",
            "element_bytes",
            "pitch_bytes",
            "requested_span_bytes",
            "pitched_span_bytes",
            "phase",
        ]
        for line in lines:
            prefix, *fields = line.split(" ")
            self.assertEqual(prefix, "OWNED_ROTATION_ALLOCATION")
            self.assertEqual([field.split("=")[0] for field in fields], keys)
            values = dict(field.split("=") for field in fields)
            self.assertEqual(values["phase"], "prepared_through_outputs")
            for key in keys:
                if key not in ("role", "phase"):
                    self.assertRegex(values[key], r"^(0|[1-9][0-9]*)$")
            self.assertEqual(
                int(values["pitched_span_bytes"]),
                int(values["pitch_bytes"])
                * int(values["height"])
                * int(values["depth"]),
            )

    def test_runtime_mutants_rejected(self):
        text = CU.read_text()
        program = self.program(text)
        mutants = [
            ("rand_state_map_.Alternate()->Reinitialize(stream_->Get());", ""),
            ("prev_sel_prob_map_.Flip();", ""),
            ("consistency_mask_.Flip();", ""),
            (
                (
                    "sel_prob_map_->Rotate(prev_sel_prob_map_.Alternate(), st"
                    "ream_->Get());"
                ),
                (
                    "sel_prob_map_->Rotate(sel_prob_map_.Alternate(), stream_"
                    "->Get());"
                ),
            ),
            (
                (
                    "consistency_mask_.Alternate()->Rotate(consistency_mask_."
                    "Current(), stream_->Get());"
                ),
                (
                    "consistency_mask_.Current()->Rotate(consistency_mask_.Al"
                    "ternate(), stream_->Get());"
                ),
            ),
        ]
        for old, new in mutants:
            with self.subTest(mutation=old):
                self.assertEqual(program.count(old), 1)
                run = self.compile_run(program.replace(old, new))
                self.assertNotEqual(
                    run.returncode, 0, "Mutant survived: " + old
                )


if __name__ == "__main__":
    unittest.main()
