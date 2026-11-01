# SPDX-License-Identifier: BSD-3-Clause
"""Allocation-free shape checks, including malicious full-width SQLite dims."""
import pathlib
import subprocess
import tempfile
import unittest


class DatabaseBlobShapeTest(unittest.TestCase):
    def test_checked_shape_before_allocation(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        source = r'''
#include "colmap/scene/database_blob.h"
#include <cassert>
#include <cstdint>
#include <stdexcept>
int main() {
  using colmap::ValidateDatabaseBlobShape;
  ValidateDatabaseBlobShape(1, 2, 8, 4, INT64_MAX);
  ValidateDatabaseBlobShape(0, 2, 0, 4, INT64_MAX);
  auto refuses = [](int64_t r, int64_t c, uint64_t b) {
    try { ValidateDatabaseBlobShape(r, c, b, 4, INT64_MAX); }
    catch (const std::runtime_error&) { return true; }
    return false;
  };
  assert(refuses(4294967297LL, 2, 8));
  assert(refuses(INT64_MAX, INT64_MAX, 8));
  assert(refuses(-1, 2, 0));
  assert(refuses(1, -2, 0));
  assert(refuses(2, 2, 8));
  assert(refuses(1, 2, 9));
  assert(refuses(INT64_MAX, 0, 0));
  try { ValidateDatabaseBlobShape(1, 2, 8, 4, 1); assert(false); }
  catch (const std::runtime_error&) {}
  try { ValidateDatabaseBlobShape(2147483647, 2147483647, 0, 8, UINT64_MAX); assert(false); }
  catch (const std::runtime_error&) {}
  try { ValidateDatabaseBlobShape(65536, 65536, 0, 1, INT32_MAX); assert(false); }
  catch (const std::runtime_error&) {}
  try { ValidateDatabaseBlobShape(0, 0, 0, 0, INT64_MAX); assert(false); }
  catch (const std::runtime_error&) {}
}
'''
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = pathlib.Path(directory)
            (path / 'test.cc').write_text(source)
            subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I', str(root / 'src'), str(path / 'test.cc'),
                            '-o', str(path / 'test')], check=True)
            subprocess.run([str(path / 'test')], check=True)


if __name__ == '__main__':
    unittest.main()
