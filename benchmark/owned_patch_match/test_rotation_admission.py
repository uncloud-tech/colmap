# SPDX-License-Identifier: BSD-3-Clause
"""Execute the actual CLI device-minimum block, not a reimplemented formula."""

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class RotationAdmission(unittest.TestCase):
    def test_actual_cli_minimum_and_boundary_rejection(self):
        source = (ROOT / "src/colmap/exe/owned_pair.cc").read_text()
        begin = source.index("  uint64_t device=")
        end = source.index("\n}", begin)
        calculation = source[begin:end]
        cpp = (
            "\n"
            '#include "colmap/mvs/owned_pair_validation.h"\n'
            '#include "colmap/mvs/owned_pair_coordinator.h"\n'
            '#include "colmap/mvs/owned_rotation_storage.h"\n'
            "#include <cassert>\n"
            "#include <limits>\n"
            "namespace pair=colmap::mvs::owned;\n"
            "struct Ref {size_t w,h; size_t GetWidth()const{return w;} si"
            "ze_t GetHeight()const{return h;}};\n"
            "uint64_t Check(Ref ref,uint64_t s,uint64_t u,uint64_t suppli"
            "ed) {\n"
            " uint64_t a=pair::CheckedMultiply(ref.w,ref.h),host=1;\n"
            " pair::ByteCharge charge{1,supplied};\n"
            " // ACTUAL_MINIMUM_AND_REJECTION\n"
            " return device;\n"
            "}\n"
            "int main() {\n"
            " constexpr uint64_t required=9331516384ULL; // P+U+2A+688S +"
            " retained32B calibration policy\n"
            " assert(Check({6204,4132},8,205079424,required)==required);\n"
            " bool rejected=false;\n"
            " try {Check({6204,4132},8,205079424,required-1);}catch(const"
            " std::invalid_argument&){rejected=true;}\n"
            " assert(rejected);\n"
            " // Old logical allowance must no longer admit this candidat"
            "e.\n"
            " rejected=false;\n"
            " try {Check({6204,4132},8,205079424,5998975744ULL);}catch(co"
            "nst std::invalid_argument&){rejected=true;}\n"
            " assert(rejected);\n"
            " assert(Check({3,5},2,70,4558)==4558);\n"
            " rejected=false;\n"
            " try {Check({3,5},2,std::numeric_limits<uint64_t>::max(),std"
            "::numeric_limits<uint64_t>::max());}\n"
            " catch(const std::overflow_error&){rejected=true;}\n"
            " assert(rejected);\n"
            "}\n"
        ).replace("// ACTUAL_MINIMUM_AND_REJECTION", calculation)
        with tempfile.TemporaryDirectory(prefix="rotation-admission-") as tmp:
            path = Path(tmp) / "test.cc"
            exe = Path(tmp) / "test"
            path.write_text(cpp)
            result = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-pthread",
                    "-I",
                    str(ROOT / "src"),
                    str(path),
                    "-o",
                    str(exe),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                [str(exe)], capture_output=True, text=True, timeout=5
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
