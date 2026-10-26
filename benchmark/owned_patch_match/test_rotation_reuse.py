# SPDX-License-Identifier: BSD-3-Clause
"""Owned integration source fences. CPU evidence, never CUDA exactness proof."""

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MVS = ROOT / "src/colmap/mvs"
BASE = "a16435ec2bcb024aa466663f6f0c02a3a6895293"


def body(text, signature):
    begin = text.index(signature)
    start = text.index("{", begin)
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[begin:end]


class RotationReuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cu = (MVS / "patch_match_cuda_owned.cu").read_text()
        cls.header = (MVS / "patch_match_cuda_owned.h").read_text()
        cls.base = subprocess.check_output(
            ["git", "show", BASE + ":src/colmap/mvs/patch_match_cuda_owned.cu"],
            cwd=ROOT,
            text=True,
        )

    def test_device_code_and_array_paths_unchanged(self):
        marker = "PatchMatchCuda::PatchMatchCuda("
        # Undo only the tile template plumbing; arithmetic/device bodies must
        # remain byte-identical to the accepted compile-width implementation.
        start = "__device__"
        device = self.cu[self.cu.index(start) : self.cu.index(marker)]
        device = device.replace("kTileThreads", "THREADS_PER_BLOCK")
        device = device.replace(", int THREADS_PER_BLOCK", "")
        device = device.replace(", THREADS_PER_BLOCK>", ">")
        device = device.replace("          int THREADS_PER_BLOCK,\n", "")
        self.assertEqual(
            device, self.base[self.base.index(start) : self.base.index(marker)]
        )
        for name in ("InitSourceImages", "InitTransforms"):
            signature = "void PatchMatchCuda::" + name + "()"
            self.assertEqual(
                body(self.cu, signature), body(self.base, signature)
            )
        stock = "src/colmap/mvs/patch_match_cuda.cu"
        self.assertEqual(
            (ROOT / stock).read_bytes(),
            subprocess.check_output(
                ["git", "show", BASE + ":" + stock], cwd=ROOT
            ),
        )

    def test_storage_header_in_cuda_target(self):
        self.assertIn(
            "            owned_rotation_storage.h\n",
            (MVS / "CMakeLists.txt").read_text(),
        )

    def test_fixed_banks_and_preparation(self):
        for kind, name in [
            ("GpuMatRefImage", "ref_image_"),
            ("GpuMatPRNG", "rand_state_map_"),
            *[
                ("GpuMat<float>", n)
                for n in (
                    "depth_map_",
                    "normal_map_",
                    "sel_prob_map_",
                    "prev_sel_prob_map_",
                    "cost_map_",
                )
            ],
            ("GpuMat<uint8_t>", "consistency_mask_"),
        ]:
            self.assertIn(
                "OwnedRotationPair<" + kind + "> " + name + ";", self.header
            )
        ctor = body(self.cu, "PatchMatchCuda::PatchMatchCuda(")
        self.assertLess(
            ctor.index("stream_->Synchronize();"),
            ctor.index("InitRotationStorage();"),
        )
        self.assertLess(
            ctor.index("InitRotationStorage();"),
            ctor.index("EmitRotationAllocations();"),
        )
        spare = body(self.cu, "void PatchMatchCuda::InitRotationStorage()")
        self.assertIn("static_assert(sizeof(curandState) == 48", spare)
        self.assertIn("GpuMatPRNG::UninitializedTag{}", spare)
        self.assertNotIn("Reinitialize(", spare)
        self.assertNotIn("FillWith", spare)
        for name in (
            "ref_image_",
            "rand_state_map_",
            "depth_map_",
            "normal_map_",
            "sel_prob_map_",
            "prev_sel_prob_map_",
            "cost_map_",
            "consistency_mask_",
        ):
            self.assertIn(name + ".Set(1,", spare)
        self.assertIn("consistency_mask_.Set(0,", spare)

    def test_rotate_and_mask_do_not_allocate(self):
        rotate = body(self.cu, "void PatchMatchCuda::Rotate()")
        run = body(self.cu, "void PatchMatchCuda::RunWithWindowSizeAndStep()")
        for block in (rotate, run):
            self.assertNotRegex(
                re.sub(r"//[^\n]*", "", block),
                r"\bnew\b|make_unique|\.reset\(|\.swap\(",
            )
        self.assertLess(
            rotate.index(
                "rand_state_map_.Alternate()->Reinitialize(stream_->Get())"
            ),
            rotate.index("rand_state_map_->Rotate("),
        )
        self.assertIn(
            (
                "sel_prob_map_->Rotate(prev_sel_prob_map_.Alternate(), stream"
                "_->Get())"
            ),
            rotate,
        )
        for name in (
            "rand_state_map_",
            "depth_map_",
            "normal_map_",
            "ref_image_",
            "prev_sel_prob_map_",
            "sel_prob_map_",
            "cost_map_",
            "consistency_mask_",
        ):
            self.assertEqual(
                len(re.findall(r"\b" + re.escape(name + ".Flip();"), rotate)), 1
            )
        self.assertIn(
            (
                "consistency_mask_.Alternate()->Rotate(consistency_mask_.Curr"
                "ent(), stream_->Get())"
            ),
            run,
        )
        self.assertEqual(
            run.count("consistency_mask_->FillWithScalar(0, stream_->Get())"), 1
        )

    def test_unwritten_masks_are_never_exported(self):
        export = body(
            self.cu,
            "std::vector<int> PatchMatchCuda::GetConsistentImageIdxs() const",
        )
        self.assertIn(
            "if (!options_.filter || options_.num_iterations == 0)", export
        )
        self.assertLess(export.index("return {};"), export.index("CopyToMat("))

    def test_record_emission_precedes_ready_in_source(self):
        controller = (MVS / "patch_match.cc").read_text()
        prepare = body(
            controller,
            "void PatchMatch::PrepareOwned(const int owned_tile_threads)",
        )
        self.assertIn(
            (
                "std::make_unique<owned::PatchMatchCuda>(options_, problem_, "
                "owned_tile_threads)"
            ),
            prepare,
        )
        pair = controller[
            controller.index(
                "void PatchMatchController::ExecutePreparedOwnedPairJob("
            ) :
        ]
        self.assertLess(
            pair.index("patch_match->PrepareOwned(hooks.owned_tile_threads);"),
            pair.index("coordinator.ReadyAndWait("),
        )
        self.assertLess(
            pair.index("coordinator.ReadyAndWait("), pair.index("} catch (...)")
        )
        ctor = body(self.cu, "PatchMatchCuda::PatchMatchCuda(")
        self.assertLess(
            ctor.index("EmitRotationAllocations();"),
            ctor.index("} catch (...)"),
        )
        self.assertEqual(ctor.count("EmitRotationAllocations();"), 1)
        self.assertIn("throw;", ctor[ctor.index("} catch (...)") :])

    def test_allocation_records_use_actual_storage(self):
        records = body(
            self.cu, "void PatchMatchCuda::EmitRotationAllocations()"
        )
        self.assertIn("rotation_layout_.slot_layouts", records)
        self.assertIn(
            "MakeRotationAllocationRecord(slot, mat.GetPitch())", records
        )
        for method in (
            "GetWidth()",
            "GetHeight()",
            "GetDepth()",
            "sizeof(*mat.GetPtr())",
        ):
            self.assertIn(method, records)
        self.assertIn("OWNED_ROTATION_ALLOCATION reference=", records)
        self.assertIn("problem_.ref_image_idx", records)
        self.assertIn("std::locale::classic()", records)
        self.assertNotIn("OWNED_PAIR_EVENT", records)
        for role in (
            "Prng",
            "Depth",
            "Normal",
            "RefImage",
            "RefSum",
            "RefSquaredSum",
            "SelectionCurrent",
            "SelectionPrevious",
            "Cost",
            "Mask",
            "Workspace",
        ):
            self.assertIn("RotationRole::k" + role + ":", records)


if __name__ == "__main__":
    unittest.main()
