# SPDX-License-Identifier: BSD-3-Clause
"""Actual host OpenImageIO headers, no image decode or CUDA/product build."""

import pathlib
import shlex
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class PairImage(unittest.TestCase):
    def test_actual_file_dimensions_before_decode(self):
        flags = shlex.split(
            subprocess.check_output(
                ["pkg-config", "--cflags", "--libs", "OpenImageIO"], text=True
            )
        )
        cpp = (
            "\n"
            '#include "colmap/mvs/owned_pair_image.h"\n'
            '#include "colmap/util/cache.h"\n'
            "#include <future>\n"
            "#include <mutex>\n"
            "#include <thread>\n"
            "#include <cassert>\n"
            "#include <fstream>\n"
            "using namespace colmap::mvs::owned;\n"
            "struct Entry {size_t bytes=0;size_t NumBytes()const{return b"
            "ytes;}};\n"
            "void Interleave(size_t actual_bytes,bool expires){\n"
            " colmap::MemoryConstrainedLRUCache<int,Entry> cache(8,[](int"
            "){return std::make_shared<Entry>();});\n"
            " std::mutex mutex;std::promise<void> first,second;auto f=fir"
            "st.get_future();auto s=second.get_future();std::weak_ptr<Ent"
            "ry> weak;\n"
            " std::thread a([&]{std::shared_ptr<Entry> owner;{std::lock_g"
            "uard<std::mutex> l(mutex);owner=cache.Get(0);owner->bytes=ac"
            "tual_bytes;cache.UpdateNumBytes(0);weak=owner;}first.set_val"
            "ue();s.wait();owner.reset();});\n"
            " std::thread b([&]{f.wait();{std::lock_guard<std::mutex> l(m"
            "utex);auto owner=cache.Get(1);owner->bytes=4;cache.UpdateNum"
            "Bytes(1);}second.set_value();});\n"
            " a.join();b.join();assert(weak.expired()==expires);\n"
            "}\n"
            "int main(int argc,char** argv){assert(argc==2);std::filesyst"
            "em::path dir(argv[1]);\n"
            ' {std::ofstream f(dir/"good.ppm",std::ios::binary);f<<"P6\\n4'
            ' 3\\n255\\n";for(int i=0;i<36;++i)f.put(char(i));}\n'
            ' assert(ReadPairBitmapPixels(dir/"good.ppm",4,3)==12);\n'
            " // Real cache policy: underestimated file loses its last ow"
            "ner; true byte\n"
            " // admission retains it under the identical controlled two-"
            "decoder order.\n"
            " Interleave(12,true);Interleave(4,false);\n"
            ' bool rejected=false;try{ReadPairBitmapPixels(dir/"good.ppm"'
            ",2,2);}catch(const std::exception&){rejected=true;}assert(re"
            "jected);\n"
            ' {std::ofstream f(dir/"oversized.ppm");f<<"P6\\n10000 10000\\n'
            '255\\n";}\n'
            ' rejected=false;try{ReadPairBitmapPixels(dir/"oversized.ppm"'
            ",2,2);}catch(const std::exception&){rejected=true;}assert(re"
            "jected);\n"
            ' rejected=false;try{ReadPairBitmapPixels(dir/"missing.ppm",4'
            ",3);}catch(const std::exception&){rejected=true;}assert(reje"
            "cted);\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory(prefix="owned-image-header-") as tmp:
            source = pathlib.Path(tmp) / "test.cc"
            exe = pathlib.Path(tmp) / "test"
            source.write_text(cpp)
            p = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-pthread",
                    "-I",
                    "/usr/include/eigen3",
                    "-I",
                    str(ROOT / "src"),
                    str(source),
                    str(ROOT / "src/colmap/util/logging.cc"),
                    str(ROOT / "src/colmap/util/string.cc"),
                    str(ROOT / "src/colmap/util/file.cc"),
                    "-o",
                    str(exe),
                    "-lglog",
                    "-lboost_filesystem",
                    "-lboost_system",
                    *flags,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            p = subprocess.run(
                [str(exe), tmp], capture_output=True, text=True, timeout=10
            )
            self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    unittest.main()
