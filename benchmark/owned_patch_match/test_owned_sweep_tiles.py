# SPDX-License-Identifier: BSD-3-Clause
"""CPU runtime dispatch/config checks; not a CUDA codegen or exactness claim."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/colmap/mvs/patch_match_cuda_owned.cu"


class OwnedSweepTiles(unittest.TestCase):
    def test_runtime_dispatch_and_launch_configuration(self):
        text = SOURCE.read_text()
        config = text[text.index("void PatchMatchCuda::ComputeCudaConfig() {"):
                      text.index("void PatchMatchCuda::BindRefImageTexture()")]
        source = r'''
#include "colmap/mvs/owned_tile.h"
#include <cassert>
#include <memory>
using namespace colmap::mvs::owned;
constexpr int kOwnedElementwiseThreads = 32;
struct Dim { int x=0,y=0,z=0; };
struct Map { int width; int GetWidth(){return width;} int GetHeight(){return 65;} };
struct PatchMatchCuda {
 const int owned_tile_threads_;
 Dim sweep_block_size_,sweep_grid_size_,elem_wise_block_size_,elem_wise_grid_size_;
 std::unique_ptr<Map> depth_map_;
 void ComputeCudaConfig();
};
''' + config + r'''
int main() {
 for (int tile : {8,16,32}) for(int width : {1,7,8,9,15,16,17,31,32,33,100}) {
  PatchMatchCuda owner{tile,{},{},{},{},std::make_unique<Map>(Map{width})};
  owner.ComputeCudaConfig();
  int calls=0;
  DispatchOwnedTile(tile,[&](auto selected) {
   constexpr int threads=decltype(selected)::value;
   ++calls; assert(threads==tile);
   assert(owner.sweep_block_size_.x==threads);
   assert(owner.sweep_grid_size_.x==(width-1)/threads+1);
   for(int col=0;col<width;++col) for(int offset=-5;offset<=5;++offset) {
    const int local=threads+col%threads+offset;
    assert(local>=0 && local<3*threads);
    assert(col/threads*threads-threads+local==col+offset);
   }
  });
  assert(calls==1);
  assert(owner.elem_wise_block_size_.x==32 && owner.elem_wise_block_size_.y==32);
  assert(owner.elem_wise_grid_size_.x==(width-1)/32+1);
  assert(owner.elem_wise_grid_size_.y==3);
 }
 for(int tile : {0,4,64,-8}) {
  bool rejected=false; int calls=0;
  try { DispatchOwnedTile(tile,[&](auto){++calls;}); }
  catch(const std::invalid_argument&){rejected=true;}
  assert(rejected && calls==0);
 }
}
'''
        temp = pathlib.Path(tempfile.mkdtemp(prefix="owned-runtime-tile-"))
        program, binary = temp / "test.cc", temp / "test"
        program.write_text(source)
        result = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                                 "-I", str(ROOT / "src"), str(program), "-o", str(binary)],
                                capture_output=True, text=True, timeout=30)
        (temp / "compile.log").write_text(result.stdout + result.stderr)
        print(f"OWNED_RUNTIME_TILE_ARTIFACT {temp}")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
        (temp / "run.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("COLMAP_OWNED_SWEEP_THREADS", text)
        self.assertIn("static_assert(kWindowRadius <= kTileThreads", text)
        self.assertIn("ComputeInitialCost<kWindowSize, kWindowStep, kTileThreads>", text)
        self.assertEqual(text.count("PhotoConsistencyCostComputer<kWindowSize, kWindowStep, kTileThreads>"), 2)
        self.assertIn("kTileThreads,", text[text.index("#define CALL_SWEEP_FUNC"):])
        self.assertIn("RunWithWindowSizeAndStep<11, 1, decltype(tile)::value>()", text)
        self.assertLess(text.index("owned_tile_threads_(ValidateOwnedTileThreads"),
                        text.index("  SetBestCudaDevice"))


if __name__ == "__main__":
    unittest.main()
