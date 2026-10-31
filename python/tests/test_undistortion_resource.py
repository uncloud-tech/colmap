# SPDX-License-Identifier: BSD-3-Clause
"""Dependency-free native logical-allocation seam; real I/O runs in CTest."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class UndistortionResourceTest(unittest.TestCase):
    def test_native_payload_and_worker_admission(self):
        root = Path(__file__).resolve().parents[2]
        source = r'''
#include "colmap/controllers/undistortion_resource.h"
#include <cassert>
#include <limits>
int main() {
  using namespace colmap;
  // Source 100x80 RGB, target 20x16: full warp + resize coexist.
  assert(UndistortionJobPayload(100,80,20,16,false,false) == 48960);
  assert(UndistortionJobPayload(100,80,60,48,true,false) == 32640);
  assert(UndistortionJobPayload(100,80,100,80,true,true) == 0);
  UndistortionLogicalBudget budget(60000,1000,1000);
  assert(budget.BatchSize({24000,24000,24000},0,3) == 2);
  assert(budget.BatchSize({48960,960},0,4) == 2);
  assert(budget.BatchSize({48960,960},1,1) == 1);
  bool refused=false;
  try { budget.BatchSize({60000},0,4); } catch(const std::runtime_error&) {refused=true;}
  assert(refused);
  refused=false;
  try { UndistortionLogicalBudget(std::numeric_limits<uint64_t>::max(),1,0)
      .BatchSize({std::numeric_limits<uint64_t>::max()},0,1); }
  catch(const std::overflow_error&) {refused=true;} assert(refused);
  // ceil(1.03 * (100+1)) + 7 == 112, reserve outside the margin.
  assert(UndistortionLogicalBudget(112,1,7).BatchSize({100},0,1)==1);
  refused=false;
  try { UndistortionLogicalBudget(111,1,7).BatchSize({100},0,1); }
  catch(const std::runtime_error&) {refused=true;} assert(refused);
  for (auto dims : {0ULL, std::numeric_limits<unsigned long long>::max()}) {
    refused=false;
    try { UndistortionJobPayload(dims,80,20,16,false,false); }
    catch(const std::exception&) {refused=true;} assert(refused);
  }
}
'''
        with tempfile.TemporaryDirectory(dir=root.parent) as directory:
            cpp = Path(directory) / 'test.cc'
            exe = Path(directory) / 'test'
            cpp.write_text(source)
            built = subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                '-I', str(root / 'src'), str(cpp), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
