# SPDX-License-Identifier: BSD-3-Clause
"""CPU tests for the pair-only checked float Mat output seam.

Compiles the REAL new header (src/colmap/mvs/owned_pair_output.h) against the
actual Mat<float>::Write reference writer (real src/colmap/mvs/mat.cc) and the
real little-endian serializer (src/colmap/util/endian.h / endian.cc). No
product configure/build, no CUDA, no GPU. The reference link additionally uses
the real util/string.cc, util/file.cc, util/logging.cc plus glog/boost -- these
are disclosed host dependencies, not stubs, and they are CPU-only.

Run: python3 -m unittest benchmark.owned_patch_match.test_pair_output -v
"""
import resource
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / 'src/colmap/mvs/owned_pair_output.h'
BASE = 'd41c4915b95e4e31b57c3f01c6ece3fb1ac14401'
EIGEN_INCLUDE = '/usr/include/eigen3'
REFERENCE_SOURCES = [
    'src/colmap/mvs/mat.cc',
    'src/colmap/util/endian.cc',
    'src/colmap/util/string.cc',
    'src/colmap/util/file.cc',
    'src/colmap/util/logging.cc',
]

PROGRAM = r'''
#include "colmap/mvs/owned_pair_output.h"
#include "colmap/mvs/mat.h"
#include "colmap/util/endian.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <iterator>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using colmap::mvs::Mat;
using colmap::mvs::owned::WriteCheckedMat;
using colmap::mvs::owned::WriteCheckedMatWith;

static int failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #cond); \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

#define REQUIRE(cond)                                                      \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "REQUIRE failed at line %d: %s\n", __LINE__, #cond); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

static std::vector<char> ReadAll(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>());
}

struct OkBuf : std::streambuf {
 protected:
  std::streamsize xsputn(const char*, std::streamsize n) override { return n; }
  int overflow(int c) override { return c == EOF ? 0 : c; }
};

// Accepts at most `limit` bytes then reports a short write.
struct ShortWriteBuf : std::streambuf {
  explicit ShortWriteBuf(std::streamsize limit) : remaining_(limit) {}

 protected:
  std::streamsize xsputn(const char*, std::streamsize n) override {
    if (remaining_ <= 0) {
      return 0;
    }
    const std::streamsize take = std::min(n, remaining_);
    remaining_ -= take;
    return take;
  }
  int overflow(int c) override {
    if (c == EOF) {
      return 0;
    }
    if (remaining_ <= 0) {
      return EOF;
    }
    --remaining_;
    return c;
  }

 private:
  std::streamsize remaining_;
};

// Accepts every byte but fails the explicit flush.
struct FlushFailBuf : std::streambuf {
 protected:
  std::streamsize xsputn(const char*, std::streamsize n) override { return n; }
  int overflow(int c) override { return c == EOF ? 0 : c; }
  int sync() override { return -1; }
};

// Writes real bytes to a real file descriptor, up to `limit`, then short-writes.
struct LimitedFdBuf : std::streambuf {
  LimitedFdBuf(int fd, std::streamsize limit) : fd_(fd), remaining_(limit) {}

 protected:
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    if (remaining_ <= 0) {
      return 0;
    }
    const std::streamsize take = std::min(n, remaining_);
    const ssize_t written = ::write(fd_, s, static_cast<size_t>(take));
    if (written <= 0) {
      return 0;
    }
    remaining_ -= written;
    return written;
  }
  int overflow(int c) override {
    if (c == EOF) {
      return 0;
    }
    const char ch = static_cast<char>(c);
    return xsputn(&ch, 1) == 1 ? c : EOF;
  }

 private:
  int fd_;
  std::streamsize remaining_;
};

struct GroupingLocale : std::numpunct<char> {
  char do_thousands_sep() const override { return '_'; }
  std::string do_grouping() const override { return "\1"; }
};

int main(int argc, char** argv) {
  REQUIRE(argc == 2);
  const std::filesystem::path tmp = argv[1];

  Mat<float> mat(2, 3, 2);
  REQUIRE(mat.GetData().size() == 12);
  const uint32_t bits[12] = {0x00000000u, 0x80000000u, 0x3f800000u, 0xbf000000u,
                             0x7f800000u, 0x00000001u, 0x7fc00000u, 0xff800000u,
                             0x4b000001u, 0x3eaaaaabu, 0x7f7fffffu, 0x00800000u};
  for (size_t i = 0; i < 12; ++i) {
    std::memcpy(mat.GetPtr() + i, &bits[i], sizeof(float));
  }

  const auto stock = tmp / "stock.mat";
  const auto checked = tmp / "checked.mat";

  // 1) Success bytes must equal the current stock writer and the real
  //    endian-reconstructed expectation.
  mat.Write(stock);
  WriteCheckedMat(mat, checked);
  CHECK(ReadAll(stock) == ReadAll(checked));
  {
    std::ostringstream oss(std::ios::binary);
    oss << mat.GetWidth() << "&" << mat.GetHeight() << "&" << mat.GetDepth()
        << "&";
    colmap::WriteBinaryLittleEndian<float>(
        &oss, {mat.GetPtr(), mat.GetData().size()});
    const std::string text = oss.str();
    const std::vector<char> expected(text.begin(), text.end());
    CHECK(ReadAll(checked) == expected);
  }

  // Stock resets the stream to classic locale, even under global grouping.
  {
    const auto previous = std::locale::global(
        std::locale(std::locale::classic(), new GroupingLocale));
    Mat<float> grouped(12, 2, 1);
    grouped.Write(tmp / "grouped-stock.mat");
    WriteCheckedMat(grouped, tmp / "grouped-checked.mat");
    std::locale::global(previous);
    CHECK(ReadAll(tmp / "grouped-stock.mat") ==
          ReadAll(tmp / "grouped-checked.mat"));
  }

  // 2) Short write must throw.
  {
    ShortWriteBuf buf(8);
    std::ostream out(&buf);
    bool threw = false;
    try {
      WriteCheckedMatWith(mat, out, {});
    } catch (const std::ios_base::failure&) {
      threw = true;
    }
    CHECK(threw);
  }

  // 3) Flush failure must throw.
  {
    FlushFailBuf buf;
    std::ostream out(&buf);
    bool threw = false;
    try {
      WriteCheckedMatWith(mat, out, {});
    } catch (const std::ios_base::failure&) {
      threw = true;
    }
    CHECK(threw);
  }

  // 4) Close failure must throw and close must actually be invoked.
  {
    OkBuf buf;
    std::ostream out(&buf);
    bool closed = false;
    bool threw = false;
    try {
      WriteCheckedMatWith(mat, out, [&]() {
        closed = true;
        out.setstate(std::ios::failbit);
      });
    } catch (const std::ios_base::failure&) {
      threw = true;
    }
    CHECK(closed);
    CHECK(threw);
  }

  // 5) A partial file must remain after a short write to a real file.
  {
    const auto partial = tmp / "partial.mat";
    const int fd = ::open(partial.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    REQUIRE(fd >= 0);
    bool threw = false;
    {
      LimitedFdBuf buf(fd, 8);
      std::ostream out(&buf);
      try {
        WriteCheckedMatWith(mat, out, {});
      } catch (const std::ios_base::failure&) {
        threw = true;
      }
    }
    ::close(fd);
    CHECK(threw);
    CHECK(std::filesystem::exists(partial));
    const std::vector<char> got = ReadAll(partial);
    const std::vector<char> ref = ReadAll(stock);
    CHECK(got.size() == 8);
    CHECK(got.size() <= ref.size());
    CHECK(std::equal(got.begin(), got.end(), ref.begin()));
  }

  // 6) Concrete defect contrast: stock Mat::Write silently accepts a full
  //    device while the checked seam throws. Best effort (needs /dev/full).
  if (std::filesystem::exists("/dev/full")) {
    bool stock_threw = false;
    bool checked_threw = false;
    try {
      mat.Write("/dev/full");
    } catch (...) {
      stock_threw = true;
    }
    try {
      WriteCheckedMat(mat, "/dev/full");
    } catch (...) {
      checked_threw = true;
    }
    CHECK(!stock_threw);
    CHECK(checked_threw);
  } else {
    std::fprintf(stderr, "note: /dev/full absent; disk-full contrast skipped\n");
  }

  if (failures != 0) {
    std::fprintf(stderr, "%d pair output check(s) failed\n", failures);
    return 1;
  }
  std::printf("all pair output checks passed\n");
  return 0;
}
'''


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class PairOutputTest(unittest.TestCase):
    def test_stock_mat_unchanged(self):
        for relative in ('src/colmap/mvs/mat.cc', 'src/colmap/mvs/mat.h'):
            original = subprocess.check_output(
                ['git', '-C', str(ROOT), 'show', f'{BASE}:{relative}'])
            self.assertEqual((ROOT / relative).read_bytes(), original,
                             f'{relative} must stay byte-identical to base')

    def run_program(self, mutation=None):
        source = HEADER.read_text()
        if mutation is not None:
            source = mutation(source)
        temp = Path(tempfile.mkdtemp(prefix='owned-pair-output-'))
        header_dir = temp / 'colmap' / 'mvs'
        header_dir.mkdir(parents=True)
        (header_dir / 'owned_pair_output.h').write_text(source)
        program = temp / 'program.cc'
        program.write_text(PROGRAM)
        binary = temp / 'program'
        command = ['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O1',
                   '-pthread', '-I', str(temp), '-I', str(ROOT / 'src'),
                   '-I', EIGEN_INCLUDE, str(program)]
        command += [str(ROOT / rel) for rel in REFERENCE_SOURCES]
        command += ['-lglog', '-lboost_filesystem', '-lboost_system',
                    '-o', str(binary)]
        compiled = subprocess.run(command, capture_output=True, text=True,
                                  timeout=120)
        (temp / 'compile.log').write_text(compiled.stdout + compiled.stderr)
        self.assertEqual(compiled.returncode, 0,
                         'compile failed:\n' + compiled.stdout + compiled.stderr)
        run_dir = temp / 'run'
        run_dir.mkdir()
        try:
            result = subprocess.run([str(binary), str(run_dir)],
                                    capture_output=True, text=True, timeout=60,
                                    preexec_fn=no_core)
        except subprocess.TimeoutExpired:
            print(f'PAIR_OUTPUT_TIMEOUT artifact={temp}')
            return None
        (temp / 'run.log').write_text(result.stdout + result.stderr)
        print(f'PAIR_OUTPUT_ARTIFACT {temp} exit={result.returncode}')
        return result

    def test_real_header_behavior(self):
        result = self.run_program()
        self.assertIsNotNone(result, 'real header program hung / timed out')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mutant_missing_close(self):
        def mutation(source):
            old = '  if (close) {\n    close();\n  }'
            self.assertEqual(source.count(old), 1,
                             'close anchor not unique')
            return source.replace(old,
                                  '  (void)close;  // mutant: close dropped')

        result = self.run_program(mutation)
        self.assertTrue(result is None or result.returncode != 0,
                        'missing close/check behavior was not detected')

    def test_mutant_missing_flush(self):
        def mutation(source):
            old = '  stream.flush();\n'
            self.assertEqual(source.count(old), 1,
                             'flush anchor not unique')
            return source.replace(old, '  // mutant: flush dropped\n')

        result = self.run_program(mutation)
        self.assertTrue(result is None or result.returncode != 0,
                        'missing explicit flush was not detected')

    def test_mutant_missing_exception_mask(self):
        def mutation(source):
            old = '  stream.exceptions(std::ios::failbit | std::ios::badbit);\n'
            self.assertEqual(source.count(old), 1,
                             'exceptions anchor not unique')
            return source.replace(old, '  // mutant: exceptions dropped\n')

        result = self.run_program(mutation)
        self.assertTrue(result is None or result.returncode != 0,
                        'missing stream exception mask was not detected')


if __name__ == '__main__':
    unittest.main(verbosity=2)
