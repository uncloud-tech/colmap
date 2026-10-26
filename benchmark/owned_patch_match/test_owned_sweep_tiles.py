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
        config = text[
            text.index(
                "void PatchMatchCuda::ComputeCudaConfig() {"
            ) : text.index("void PatchMatchCuda::BindRefImageTexture()")
        ]
        source = (
            (
                "\n"
                '#include "colmap/mvs/owned_tile.h"\n'
                "#include <cassert>\n"
                "#include <memory>\n"
                "using namespace colmap::mvs::owned;\n"
                "constexpr int kOwnedElementwiseThreads = 32;\n"
                "struct Dim { int x=0,y=0,z=0; };\n"
                "struct Map { int width; int GetWidth(){return width;} int Ge"
                "tHeight(){return 65;} };\n"
                "struct PatchMatchCuda {\n"
                " const int owned_tile_threads_;\n"
                " Dim sweep_block_size_,sweep_grid_size_,elem_wise_block_size"
                "_,elem_wise_grid_size_;\n"
                " std::unique_ptr<Map> depth_map_;\n"
                " void ComputeCudaConfig();\n"
                "};\n"
            )
            + config
            + (
                "\n"
                "int main() {\n"
                " for (int tile : {8,16,32}) for(int width : {1,7,8,9,15,16,1"
                "7,31,32,33,100}) {\n"
                "  PatchMatchCuda owner{tile,{},{},{},{},std::make_unique<Map"
                ">(Map{width})};\n"
                "  owner.ComputeCudaConfig();\n"
                "  int calls=0;\n"
                "  DispatchOwnedTile(tile,[&](auto selected) {\n"
                "   constexpr int threads=decltype(selected)::value;\n"
                "   ++calls; assert(threads==tile);\n"
                "   assert(owner.sweep_block_size_.x==threads);\n"
                "   assert(owner.sweep_grid_size_.x==(width-1)/threads+1);\n"
                "   for(int col=0;col<width;++col) for(int offset=-5;offset<="
                "5;++offset) {\n"
                "    const int local=threads+col%threads+offset;\n"
                "    assert(local>=0 && local<3*threads);\n"
                "    assert(col/threads*threads-threads+local==col+offset);\n"
                "   }\n"
                "  });\n"
                "  assert(calls==1);\n"
                "  assert(owner.elem_wise_block_size_.x==32 && owner.elem_wis"
                "e_block_size_.y==32);\n"
                "  assert(owner.elem_wise_grid_size_.x==(width-1)/32+1);\n"
                "  assert(owner.elem_wise_grid_size_.y==3);\n"
                " }\n"
                " for(int tile : {0,4,64,-8}) {\n"
                "  bool rejected=false; int calls=0;\n"
                "  try { DispatchOwnedTile(tile,[&](auto){++calls;}); }\n"
                "  catch(const std::invalid_argument&){rejected=true;}\n"
                "  assert(rejected && calls==0);\n"
                " }\n"
                "}\n"
            )
        )
        temp = pathlib.Path(tempfile.mkdtemp(prefix="owned-runtime-tile-"))
        program, binary = temp / "test.cc", temp / "test"
        program.write_text(source)
        result = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(ROOT / "src"),
                str(program),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        (temp / "compile.log").write_text(result.stdout + result.stderr)
        print(f"OWNED_RUNTIME_TILE_ARTIFACT {temp}")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [str(binary)], capture_output=True, text=True, timeout=10
        )
        (temp / "run.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("COLMAP_OWNED_SWEEP_THREADS", text)
        self.assertIn("static_assert(kWindowRadius <= kTileThreads", text)
        self.assertIn(
            "ComputeInitialCost<kWindowSize, kWindowStep, kTileThreads>", text
        )
        self.assertEqual(
            text.count(
                "PhotoConsistencyCostComputer<kWindowSize, kWindowStep, k"
                "TileThreads>"
            ),
            2,
        )
        self.assertIn(
            "kTileThreads,", text[text.index("#define CALL_SWEEP_FUNC") :]
        )
        self.assertIn(
            "RunWithWindowSizeAndStep<11, 1, decltype(tile)::value>()", text
        )
        self.assertLess(
            text.index("owned_tile_threads_(ValidateOwnedTileThreads"),
            text.index("  SetBestCudaDevice"),
        )


if __name__ == "__main__":
    unittest.main()
