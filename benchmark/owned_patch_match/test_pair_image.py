# SPDX-License-Identifier: BSD-3-Clause
"""Actual host OpenImageIO headers, no image decode or CUDA/product build."""
import pathlib
import shlex
import subprocess
import tempfile
import unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]

class PairImage(unittest.TestCase):
    def test_actual_file_dimensions_before_decode(self):
        flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','OpenImageIO'],text=True))
        cpp=r'''
#include "colmap/mvs/owned_pair_image.h"
#include "colmap/util/cache.h"
#include <future>
#include <mutex>
#include <thread>
#include <cassert>
#include <fstream>
using namespace colmap::mvs::owned;
struct Entry {size_t bytes=0;size_t NumBytes()const{return bytes;}};
void Interleave(size_t actual_bytes,bool expires){
 colmap::MemoryConstrainedLRUCache<int,Entry> cache(8,[](int){return std::make_shared<Entry>();});
 std::mutex mutex;std::promise<void> first,second;auto f=first.get_future();auto s=second.get_future();std::weak_ptr<Entry> weak;
 std::thread a([&]{std::shared_ptr<Entry> owner;{std::lock_guard<std::mutex> l(mutex);owner=cache.Get(0);owner->bytes=actual_bytes;cache.UpdateNumBytes(0);weak=owner;}first.set_value();s.wait();owner.reset();});
 std::thread b([&]{f.wait();{std::lock_guard<std::mutex> l(mutex);auto owner=cache.Get(1);owner->bytes=4;cache.UpdateNumBytes(1);}second.set_value();});
 a.join();b.join();assert(weak.expired()==expires);
}
int main(int argc,char** argv){assert(argc==2);std::filesystem::path dir(argv[1]);
 {std::ofstream f(dir/"good.ppm",std::ios::binary);f<<"P6\n4 3\n255\n";for(int i=0;i<36;++i)f.put(char(i));}
 assert(ReadPairBitmapPixels(dir/"good.ppm",4,3)==12);
 // Real cache policy: underestimated file loses its last owner; true byte
 // admission retains it under the identical controlled two-decoder order.
 Interleave(12,true);Interleave(4,false);
 bool rejected=false;try{ReadPairBitmapPixels(dir/"good.ppm",2,2);}catch(const std::exception&){rejected=true;}assert(rejected);
 {std::ofstream f(dir/"oversized.ppm");f<<"P6\n10000 10000\n255\n";}
 rejected=false;try{ReadPairBitmapPixels(dir/"oversized.ppm",2,2);}catch(const std::exception&){rejected=true;}assert(rejected);
 rejected=false;try{ReadPairBitmapPixels(dir/"missing.ppm",4,3);}catch(const std::exception&){rejected=true;}assert(rejected);
}
'''
        with tempfile.TemporaryDirectory(prefix='owned-image-header-') as tmp:
            source=pathlib.Path(tmp)/'test.cc';exe=pathlib.Path(tmp)/'test';source.write_text(cpp)
            p=subprocess.run(['g++','-std=c++17','-pthread','-I','/usr/include/eigen3','-I',str(ROOT/'src'),str(source),str(ROOT/'src/colmap/util/logging.cc'),str(ROOT/'src/colmap/util/string.cc'),str(ROOT/'src/colmap/util/file.cc'),'-o',str(exe),'-lglog','-lboost_filesystem','-lboost_system',*flags],capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([str(exe),tmp],capture_output=True,text=True,timeout=10)
            self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__':unittest.main()
