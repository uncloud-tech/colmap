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
HEADER = ROOT / "src/colmap/mvs/owned_pair_output.h"
BASE = "d41c4915b95e4e31b57c3f01c6ece3fb1ac14401"
EIGEN_INCLUDE = "/usr/include/eigen3"
REFERENCE_SOURCES = [
    "src/colmap/mvs/mat.cc",
    "src/colmap/util/endian.cc",
    "src/colmap/util/string.cc",
    "src/colmap/util/file.cc",
    "src/colmap/util/logging.cc",
]

PROGRAM = (
    "\n"
    '#include "colmap/mvs/owned_pair_output.h"\n'
    '#include "colmap/mvs/mat.h"\n'
    '#include "colmap/util/endian.h"\n'
    "\n"
    "#include <algorithm>\n"
    "#include <cstdint>\n"
    "#include <cstdio>\n"
    "#include <cstdlib>\n"
    "#include <cstring>\n"
    "#include <filesystem>\n"
    "#include <fstream>\n"
    "#include <functional>\n"
    "#include <ios>\n"
    "#include <iterator>\n"
    "#include <ostream>\n"
    "#include <sstream>\n"
    "#include <stdexcept>\n"
    "#include <streambuf>\n"
    "#include <string>\n"
    "#include <vector>\n"
    "\n"
    "#include <fcntl.h>\n"
    "#include <unistd.h>\n"
    "\n"
    "using colmap::mvs::Mat;\n"
    "using colmap::mvs::owned::WriteCheckedMat;\n"
    "using colmap::mvs::owned::WriteCheckedMatWith;\n"
    "\n"
    "static int failures = 0;\n"
    "\n"
    "#define CHECK(cond)                                         "
    "               \\\n"
    "  do {                                                      "
    "               \\\n"
    "    if (!(cond)) {                                          "
    "               \\\n"
    '      std::fprintf(stderr, "CHECK failed at line %d: %s\\n", '
    "__LINE__, #cond); \\\n"
    "      ++failures;                                           "
    "               \\\n"
    "    }                                                       "
    "               \\\n"
    "  } while (0)\n"
    "\n"
    "#define REQUIRE(cond)                                       "
    "               \\\n"
    "  do {                                                      "
    "               \\\n"
    "    if (!(cond)) {                                          "
    "               \\\n"
    '      std::fprintf(stderr, "REQUIRE failed at line %d: %s\\n"'
    ", __LINE__, #cond); \\\n"
    "      std::exit(1);                                         "
    "               \\\n"
    "    }                                                       "
    "               \\\n"
    "  } while (0)\n"
    "\n"
    "static std::vector<char> ReadAll(const std::filesystem::path"
    "& path) {\n"
    "  std::ifstream file(path, std::ios::binary);\n"
    "  return std::vector<char>(std::istreambuf_iterator<char>(fi"
    "le),\n"
    "                           std::istreambuf_iterator<char>())"
    ";\n"
    "}\n"
    "\n"
    "struct OkBuf : std::streambuf {\n"
    " protected:\n"
    "  std::streamsize xsputn(const char*, std::streamsize n) ove"
    "rride { return n; }\n"
    "  int overflow(int c) override { return c == EOF ? 0 : c; }\n"
    "};\n"
    "\n"
    "// Accepts at most `limit` bytes then reports a short write."
    "\n"
    "struct ShortWriteBuf : std::streambuf {\n"
    "  explicit ShortWriteBuf(std::streamsize limit) : remaining_"
    "(limit) {}\n"
    "\n"
    " protected:\n"
    "  std::streamsize xsputn(const char*, std::streamsize n) ove"
    "rride {\n"
    "    if (remaining_ <= 0) {\n"
    "      return 0;\n"
    "    }\n"
    "    const std::streamsize take = std::min(n, remaining_);\n"
    "    remaining_ -= take;\n"
    "    return take;\n"
    "  }\n"
    "  int overflow(int c) override {\n"
    "    if (c == EOF) {\n"
    "      return 0;\n"
    "    }\n"
    "    if (remaining_ <= 0) {\n"
    "      return EOF;\n"
    "    }\n"
    "    --remaining_;\n"
    "    return c;\n"
    "  }\n"
    "\n"
    " private:\n"
    "  std::streamsize remaining_;\n"
    "};\n"
    "\n"
    "// Accepts every byte but fails the explicit flush.\n"
    "struct FlushFailBuf : std::streambuf {\n"
    " protected:\n"
    "  std::streamsize xsputn(const char*, std::streamsize n) ove"
    "rride { return n; }\n"
    "  int overflow(int c) override { return c == EOF ? 0 : c; }\n"
    "  int sync() override { return -1; }\n"
    "};\n"
    "\n"
    "// Writes real bytes to a real file descriptor, up to `limit"
    "`, then short-writes.\n"
    "struct LimitedFdBuf : std::streambuf {\n"
    "  LimitedFdBuf(int fd, std::streamsize limit) : fd_(fd), rem"
    "aining_(limit) {}\n"
    "\n"
    " protected:\n"
    "  std::streamsize xsputn(const char* s, std::streamsize n) o"
    "verride {\n"
    "    if (remaining_ <= 0) {\n"
    "      return 0;\n"
    "    }\n"
    "    const std::streamsize take = std::min(n, remaining_);\n"
    "    const ssize_t written = ::write(fd_, s, static_cast<size"
    "_t>(take));\n"
    "    if (written <= 0) {\n"
    "      return 0;\n"
    "    }\n"
    "    remaining_ -= written;\n"
    "    return written;\n"
    "  }\n"
    "  int overflow(int c) override {\n"
    "    if (c == EOF) {\n"
    "      return 0;\n"
    "    }\n"
    "    const char ch = static_cast<char>(c);\n"
    "    return xsputn(&ch, 1) == 1 ? c : EOF;\n"
    "  }\n"
    "\n"
    " private:\n"
    "  int fd_;\n"
    "  std::streamsize remaining_;\n"
    "};\n"
    "\n"
    "struct GroupingLocale : std::numpunct<char> {\n"
    "  char do_thousands_sep() const override { return '_'; }\n"
    '  std::string do_grouping() const override { return "\\1"; }\n'
    "};\n"
    "\n"
    "int main(int argc, char** argv) {\n"
    "  REQUIRE(argc == 2);\n"
    "  const std::filesystem::path tmp = argv[1];\n"
    "\n"
    "  Mat<float> mat(2, 3, 2);\n"
    "  REQUIRE(mat.GetData().size() == 12);\n"
    "  const uint32_t bits[12] = {0x00000000u, 0x80000000u, 0x3f8"
    "00000u, 0xbf000000u,\n"
    "                             0x7f800000u, 0x00000001u, 0x7fc"
    "00000u, 0xff800000u,\n"
    "                             0x4b000001u, 0x3eaaaaabu, 0x7f7"
    "fffffu, 0x00800000u};\n"
    "  for (size_t i = 0; i < 12; ++i) {\n"
    "    std::memcpy(mat.GetPtr() + i, &bits[i], sizeof(float));\n"
    "  }\n"
    "\n"
    '  const auto stock = tmp / "stock.mat";\n'
    '  const auto checked = tmp / "checked.mat";\n'
    "\n"
    "  // 1) Success bytes must equal the current stock writer an"
    "d the real\n"
    "  //    endian-reconstructed expectation.\n"
    "  mat.Write(stock);\n"
    "  WriteCheckedMat(mat, checked);\n"
    "  CHECK(ReadAll(stock) == ReadAll(checked));\n"
    "  {\n"
    "    std::ostringstream oss(std::ios::binary);\n"
    '    oss << mat.GetWidth() << "&" << mat.GetHeight() << "&" <'
    "< mat.GetDepth()\n"
    '        << "&";\n'
    "    colmap::WriteBinaryLittleEndian<float>(\n"
    "        &oss, {mat.GetPtr(), mat.GetData().size()});\n"
    "    const std::string text = oss.str();\n"
    "    const std::vector<char> expected(text.begin(), text.end("
    "));\n"
    "    CHECK(ReadAll(checked) == expected);\n"
    "  }\n"
    "\n"
    "  // Stock resets the stream to classic locale, even under g"
    "lobal grouping.\n"
    "  {\n"
    "    const auto previous = std::locale::global(\n"
    "        std::locale(std::locale::classic(), new GroupingLoca"
    "le));\n"
    "    Mat<float> grouped(12, 2, 1);\n"
    '    grouped.Write(tmp / "grouped-stock.mat");\n'
    '    WriteCheckedMat(grouped, tmp / "grouped-checked.mat");\n'
    "    std::locale::global(previous);\n"
    '    CHECK(ReadAll(tmp / "grouped-stock.mat") ==\n'
    '          ReadAll(tmp / "grouped-checked.mat"));\n'
    "  }\n"
    "\n"
    "  // 2) Short write must throw.\n"
    "  {\n"
    "    ShortWriteBuf buf(8);\n"
    "    std::ostream out(&buf);\n"
    "    bool threw = false;\n"
    "    try {\n"
    "      WriteCheckedMatWith(mat, out, {});\n"
    "    } catch (const std::ios_base::failure&) {\n"
    "      threw = true;\n"
    "    }\n"
    "    CHECK(threw);\n"
    "  }\n"
    "\n"
    "  // 3) Flush failure must throw.\n"
    "  {\n"
    "    FlushFailBuf buf;\n"
    "    std::ostream out(&buf);\n"
    "    bool threw = false;\n"
    "    try {\n"
    "      WriteCheckedMatWith(mat, out, {});\n"
    "    } catch (const std::ios_base::failure&) {\n"
    "      threw = true;\n"
    "    }\n"
    "    CHECK(threw);\n"
    "  }\n"
    "\n"
    "  // 4) Close failure must throw and close must actually be "
    "invoked.\n"
    "  {\n"
    "    OkBuf buf;\n"
    "    std::ostream out(&buf);\n"
    "    bool closed = false;\n"
    "    bool threw = false;\n"
    "    try {\n"
    "      WriteCheckedMatWith(mat, out, [&]() {\n"
    "        closed = true;\n"
    "        out.setstate(std::ios::failbit);\n"
    "      });\n"
    "    } catch (const std::ios_base::failure&) {\n"
    "      threw = true;\n"
    "    }\n"
    "    CHECK(closed);\n"
    "    CHECK(threw);\n"
    "  }\n"
    "\n"
    "  // 5) A partial file must remain after a short write to a "
    "real file.\n"
    "  {\n"
    '    const auto partial = tmp / "partial.mat";\n'
    "    const int fd = ::open(partial.c_str(), O_WRONLY | O_CREA"
    "T | O_TRUNC, 0644);\n"
    "    REQUIRE(fd >= 0);\n"
    "    bool threw = false;\n"
    "    {\n"
    "      LimitedFdBuf buf(fd, 8);\n"
    "      std::ostream out(&buf);\n"
    "      try {\n"
    "        WriteCheckedMatWith(mat, out, {});\n"
    "      } catch (const std::ios_base::failure&) {\n"
    "        threw = true;\n"
    "      }\n"
    "    }\n"
    "    ::close(fd);\n"
    "    CHECK(threw);\n"
    "    CHECK(std::filesystem::exists(partial));\n"
    "    const std::vector<char> got = ReadAll(partial);\n"
    "    const std::vector<char> ref = ReadAll(stock);\n"
    "    CHECK(got.size() == 8);\n"
    "    CHECK(got.size() <= ref.size());\n"
    "    CHECK(std::equal(got.begin(), got.end(), ref.begin()));\n"
    "  }\n"
    "\n"
    "  // 6) Concrete defect contrast: stock Mat::Write silently "
    "accepts a full\n"
    "  //    device while the checked seam throws. Best effort (n"
    "eeds /dev/full).\n"
    '  if (std::filesystem::exists("/dev/full")) {\n'
    "    bool stock_threw = false;\n"
    "    bool checked_threw = false;\n"
    "    try {\n"
    '      mat.Write("/dev/full");\n'
    "    } catch (...) {\n"
    "      stock_threw = true;\n"
    "    }\n"
    "    try {\n"
    '      WriteCheckedMat(mat, "/dev/full");\n'
    "    } catch (...) {\n"
    "      checked_threw = true;\n"
    "    }\n"
    "    CHECK(!stock_threw);\n"
    "    CHECK(checked_threw);\n"
    "  } else {\n"
    '    std::fprintf(stderr, "note: /dev/full absent; disk-full '
    'contrast skipped\\n");\n'
    "  }\n"
    "\n"
    "  if (failures != 0) {\n"
    '    std::fprintf(stderr, "%d pair output check(s) failed\\n",'
    " failures);\n"
    "    return 1;\n"
    "  }\n"
    '  std::printf("all pair output checks passed\\n");\n'
    "  return 0;\n"
    "}\n"
)


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class PairOutputTest(unittest.TestCase):
    def test_stock_mat_unchanged(self):
        for relative in ("src/colmap/mvs/mat.cc", "src/colmap/mvs/mat.h"):
            original = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{BASE}:{relative}"]
            )
            self.assertEqual(
                (ROOT / relative).read_bytes(),
                original,
                f"{relative} must stay byte-identical to base",
            )

    def run_program(self, mutation=None):
        source = HEADER.read_text()
        if mutation is not None:
            source = mutation(source)
        temp = Path(tempfile.mkdtemp(prefix="owned-pair-output-"))
        header_dir = temp / "colmap" / "mvs"
        header_dir.mkdir(parents=True)
        (header_dir / "owned_pair_output.h").write_text(source)
        program = temp / "program.cc"
        program.write_text(PROGRAM)
        binary = temp / "program"
        command = [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O1",
            "-pthread",
            "-I",
            str(temp),
            "-I",
            str(ROOT / "src"),
            "-I",
            EIGEN_INCLUDE,
            str(program),
        ]
        command += [str(ROOT / rel) for rel in REFERENCE_SOURCES]
        command += [
            "-lglog",
            "-lboost_filesystem",
            "-lboost_system",
            "-o",
            str(binary),
        ]
        compiled = subprocess.run(
            command, capture_output=True, text=True, timeout=120
        )
        (temp / "compile.log").write_text(compiled.stdout + compiled.stderr)
        self.assertEqual(
            compiled.returncode,
            0,
            "compile failed:\n" + compiled.stdout + compiled.stderr,
        )
        run_dir = temp / "run"
        run_dir.mkdir()
        try:
            result = subprocess.run(
                [str(binary), str(run_dir)],
                capture_output=True,
                text=True,
                timeout=60,
                preexec_fn=no_core,
            )
        except subprocess.TimeoutExpired:
            print(f"PAIR_OUTPUT_TIMEOUT artifact={temp}")
            return None
        (temp / "run.log").write_text(result.stdout + result.stderr)
        print(f"PAIR_OUTPUT_ARTIFACT {temp} exit={result.returncode}")
        return result

    def test_real_header_behavior(self):
        result = self.run_program()
        self.assertIsNotNone(result, "real header program hung / timed out")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mutant_missing_close(self):
        def mutation(source):
            old = "  if (close) {\n    close();\n  }"
            self.assertEqual(source.count(old), 1, "close anchor not unique")
            return source.replace(
                old, "  (void)close;  // mutant: close dropped"
            )

        result = self.run_program(mutation)
        self.assertTrue(
            result is None or result.returncode != 0,
            "missing close/check behavior was not detected",
        )

    def test_mutant_missing_flush(self):
        def mutation(source):
            old = "  stream.flush();\n"
            self.assertEqual(source.count(old), 1, "flush anchor not unique")
            return source.replace(old, "  // mutant: flush dropped\n")

        result = self.run_program(mutation)
        self.assertTrue(
            result is None or result.returncode != 0,
            "missing explicit flush was not detected",
        )

    def test_mutant_missing_exception_mask(self):
        def mutation(source):
            old = "  stream.exceptions(std::ios::failbit | std::ios::badbit);\n"
            self.assertEqual(
                source.count(old), 1, "exceptions anchor not unique"
            )
            return source.replace(old, "  // mutant: exceptions dropped\n")

        result = self.run_program(mutation)
        self.assertTrue(
            result is None or result.returncode != 0,
            "missing stream exception mask was not detected",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
