# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import resource
import subprocess
import tempfile
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[2]
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

class PairValidation(unittest.TestCase):
    def test_actual_header(self):
        source = r'''
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include <future>
#include <thread>
#include <atomic>
#include <cassert>
#include <limits>
using namespace colmap::mvs::owned;
template<class F> bool bad(F f){try{f();return false;}catch(const std::exception&){return true;}}
int main(){
 assert(ParseBytes("0")==0);assert(ParseBytes("18446744073709551615")==UINT64_MAX);
 for(auto s:{"", "-1", "+1", " 1", "1 ", "1e3", "18446744073709551616"})assert(bad([&]{ParseBytes(s);}));
 assert(CheckedAdd(3,4)==7);assert(bad([]{CheckedAdd(UINT64_MAX,1);}));
 assert(CheckedMultiply(3,4)==12);assert(bad([]{CheckedMultiply(UINT64_MAX,2);}));
 assert(CacheBytes(.5)==536870912);assert(bad([]{CacheBytes(0);}));
 assert(bad([]{CacheBytes(std::numeric_limits<double>::infinity());}));
 assert(bad([]{CacheBytes(std::numeric_limits<double>::quiet_NaN());}));
 ValidateIo(2,3,4,8);assert(bad([]{ValidateIo(0,3,4,8);}));
 assert(bad([]{ValidateIo(5,3,4,8);}));assert(bad([]{ValidateIo(2,3,9,8);}));
 assert(PathsOverlap("/a/b","/a/b/c"));assert(!PathsOverlap("/a/b","/a/bc"));
 assert(PathContains("/a/b","/a/b/c"));assert(!PathContains("/a/b","/a"));
 assert(Hex("A =\n")=="41203d0a");
 const char* argv[]={"driver","--dispatch","serial","--host_budget_bytes","10"};
 auto values=ParseStrictArguments(5,argv,{"dispatch","host_budget_bytes"});assert(values.at("dispatch")=="serial");
 const char* dup[]={"driver","--dispatch","serial","--dispatch","pair"};
 assert(bad([&]{ParseStrictArguments(5,dup,{"dispatch"});}));
 assert(bad([&]{ParseStrictArguments(5,argv,{"dispatch"});}));
 assert(bad([]{PairCoordinator c(static_cast<PairDispatch>(99),{9,9},{1,1},{{{1,1},{1,1}}},std::chrono::seconds(1));}));
 PairCoordinator c(PairDispatch::kSerial,{9,9},{1,1},{{{1,1},{1,1}}},std::chrono::seconds(2));
 std::promise<void> cancelled;auto ready=cancelled.get_future();std::atomic<bool> retire_failed{false};
 std::thread a([&]{
   try{c.BeginPreparation(0);c.ReadyAndWait(0);c.SolverDone(0);c.BeginOutput(0);throw std::runtime_error("write failure");}
   catch(...){c.Cancel(0,std::current_exception());cancelled.set_value();}
   try{c.Retire(0);}catch(...){retire_failed=true;}
 });
 std::thread b([&]{
   try{c.BeginPreparation(1);c.ReadyAndWait(1);c.SolverDone(1);ready.wait();c.BeginOutput(1);}
   catch(...){c.Cancel(1,std::current_exception());}
   try{c.Retire(1);}catch(...){retire_failed=true;}
 });
 a.join();b.join();assert(!retire_failed);
 try{c.Finish();assert(false);}catch(const std::runtime_error& e){assert(std::string(e.what())=="write failure");}
 assert(c.ReservedBytes().host==0&&c.ReservedBytes().device==0);
 PairCoordinator incomplete(PairDispatch::kSerial,{9,9},{1,1},{{{1,1},{1,1}}},std::chrono::seconds(1));
 incomplete.Cancel(0,std::make_exception_ptr(std::runtime_error("first failure")));
 try{incomplete.Finish();assert(false);}catch(const std::exception& e){assert(std::string(e.what())=="first failure");}
 assert(incomplete.ReservedBytes().host==3&&incomplete.ReservedBytes().device==3);
}
'''
        with tempfile.TemporaryDirectory(prefix='owned-pair-validation-') as tmp:
            cpp=pathlib.Path(tmp)/'test.cc';exe=pathlib.Path(tmp)/'test';cpp.write_text(source)
            p=subprocess.run(['g++','-std=c++17','-pthread','-Wall','-Wextra','-Werror','-I',str(ROOT/'src'),str(cpp),'-o',str(exe)],capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
            self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__':unittest.main()
