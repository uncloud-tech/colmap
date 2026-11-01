# SPDX-License-Identifier: BSD-3-Clause
"""Dependency-free mapping admission seam; native loaders run in CTest."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class MappingResourceTest(unittest.TestCase):
    def test_overlap_refusal_and_owner_retirement(self):
        root = Path(__file__).resolve().parents[2]
        source = r'''
#include "colmap/scene/mapping_resource.h"
#include <cassert>
#include <limits>
#include <vector>
int main() {
  using namespace colmap;
  // ceil(1.03*(100+1))+7 = 112: reserve is outside the one margin.
  auto ledger = std::make_shared<MappingLedger>(112, 1, 7);
  auto owner = ledger->Acquire(100);
  assert(ledger->Used() == 100);
  bool refused = false;
  try { auto overlap = ledger->Acquire(1); }
  catch (const std::runtime_error&) { refused = true; }
  assert(refused && ledger->Used() == 100);
  auto alias = owner;
  owner.reset();
  assert(ledger->Used() == 100);
  alias.reset();
  assert(ledger->Used() == 0);
  refused = false;
  try { MappingAdd(std::numeric_limits<uint64_t>::max(), 1); }
  catch (const std::overflow_error&) { refused = true; }
  assert(refused);
  auto growth = std::make_shared<MappingLedger>(213, 0, 0);
  auto credit = growth->Acquire(0);
  std::vector<uint8_t> payload;
  MappingReserve(payload, 100, credit);
  assert(payload.capacity() >= 100 && growth->Used() == payload.capacity());
  refused = false;
  try { MappingReserve(payload, 107, credit); }
  catch (const std::runtime_error&) { refused = true; }
  assert(refused && payload.capacity() == 100 && growth->Used() == 100);
  MappingReserve(payload, 106, credit);
  assert(growth->Used() == 106);
  // Payload must die before its credit; erasing size is not retirement.
  payload.clear();
  assert(growth->Used() == 106);
  std::vector<uint8_t>().swap(payload);
  credit.reset();
  assert(growth->Used() == 0);
}
'''
        with tempfile.TemporaryDirectory(dir=root.parent) as directory:
            cpp = Path(directory) / 'test.cc'
            exe = Path(directory) / 'test'
            cpp.write_text(source)
            built = subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                    '-I', str(root / 'src'), str(cpp), '-o', str(exe)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
