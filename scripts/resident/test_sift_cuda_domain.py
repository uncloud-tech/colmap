# SPDX-License-Identifier: BSD-3-Clause
"""CPU executable tests of the same domain/index helpers used by CUDA SIFT."""
import pathlib
import subprocess
import tempfile
import unittest


class SiftCudaDomainTest(unittest.TestCase):
    def test_domain_and_boundary_indices(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        code = r'''
#include "src/thirdparty/SiftGPU/CudaImageDomain.h"
#include <cassert>
int main() {
  using namespace siftgpu;
  assert(LevelWidth(0, true) == 1); // positive first octave on 1px input
  assert(LevelWidth(6201, true) == 6201);
  assert(LevelWidth(6202, true) == 6202);
  assert(LevelWidth(6203, true) == 6203);
  assert(LevelWidth(6204, false) == 6204);
  assert(LevelWidth(3101, false) == 3104); // retained aligned-input octave
  assert(LevelWidth(3100, true) == 3100);
  assert(!IsFeatureColumn(0, 5));
  assert(IsFeatureColumn(3, 5));
  assert(!IsFeatureColumn(4, 5));
  assert(!IsFeatureColumn(5, 5)); // rounded allocation is not content
  assert(!IsFeatureColumn(1, 1));
  assert(HistogramWidth(1) == 1);
  assert(HistogramWidth(2) == 1);
  assert(HistogramWidth(3) == 1);
  assert(HistogramWidth(4) == 1);
  assert(HistogramWidth(5) == 2);
  assert(UpsampleNeighbor(4, 0, 5, 2, true) == 4); // not next row
  assert(UpsampleNeighbor(4, 1, 5, 2, true) == 9);
  assert(UpsampleNeighbor(0, 0, 1, 1, true) == 0);
  assert(UpsampleNeighbor(3, 0, 4, 2, false) == 4); // legacy unchanged
  assert(UpsampleRow(3, 2, true) == 1);
  assert(UpsampleRow(3, 2, false) == 3);
}
'''
        with tempfile.TemporaryDirectory(dir=root.parent) as temp:
            source = pathlib.Path(temp) / 'domain.cc'
            binary = pathlib.Path(temp) / 'domain'
            source.write_text(code)
            subprocess.run(['c++', '-std=c++17', '-I', str(root), str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
