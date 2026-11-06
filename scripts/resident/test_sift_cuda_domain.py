# SPDX-License-Identifier: BSD-3-Clause
"""CPU executable tests of the same domain/index helpers used by CUDA SIFT."""
import pathlib
import subprocess
import tempfile
import unittest


class SiftCudaDomainTest(unittest.TestCase):
    def test_original_aligned_flattened_fetch_trace(self):
        # Literal original U addresses, including the third-row first element
        # and beyond-active-image addresses. This makes no claim about values
        # returned outside the *bound resource*, which can exceed active size.
        root = pathlib.Path(__file__).resolve().parents[2]
        code = r'''
#include "CudaImageDomain.h"
#include <array>
int main() {
  using namespace siftgpu;
  const std::array<int, 4> interior = {3, 4, 7, 8}; // W=4,H=3,y=1,x=3
  const std::array<int, 4> bottom = {11, 12, 15, 16}; // y=5,x=3
  for (int row : {0, 2}) {
    std::array<int, 4> actual = {
        row * 4 + 3, UpsampleNeighbor(3, row, 4, 3, false),
        UpsampleRow(row + 1, 3, false) * 4 + 3,
        UpsampleNeighbor(3, row + 1, 4, 3, false)};
    if (actual != (row == 0 ? interior : bottom)) return 1;
  }
  // Original 6048-wide default has the same flattened-address issue, not
  // a distinct resolution/quality problem. No full-size allocation needed.
  if (UpsampleNeighbor(6047, 2, 6048, 3, false) != 18144) return 2;
  if (UpsampleNeighbor(6047, 3, 6048, 3, false) != 24192) return 3;
}
'''
        candidate = (root / 'src/thirdparty/SiftGPU/CudaImageDomain.h').read_text()
        with tempfile.TemporaryDirectory(dir=root.parent) as temp:
            temp = pathlib.Path(temp)
            source, binary, header = temp / 'trace.cc', temp / 'trace', temp / 'CudaImageDomain.h'
            source.write_text(code)
            def run(text):
                header.write_text(text)
                subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                '-I', str(temp), str(source), '-o', str(binary)], check=True)
                return subprocess.run([str(binary)], capture_output=True).returncode
            # Red: a plausible but prohibited horizontal border repair must
            # disagree with the unchanged original literal trace.
            mutated = candidate.replace('exact_domain && col + 1 >= width',
                                        'col + 1 >= width')
            self.assertNotEqual(mutated, candidate)
            self.assertEqual(run(mutated), 1)
            self.assertEqual(run(candidate), 0)

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
