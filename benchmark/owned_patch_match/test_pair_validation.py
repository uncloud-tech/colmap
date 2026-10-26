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
        source = (
            "\n"
            '#include "colmap/mvs/owned_pair_validation.h"\n'
            '#include "colmap/mvs/owned_pair_coordinator.h"\n'
            "#include <future>\n"
            "#include <thread>\n"
            "#include <atomic>\n"
            "#include <cassert>\n"
            "#include <limits>\n"
            "using namespace colmap::mvs::owned;\n"
            "template<class F> bool bad(F f){try{f();return false;}catch("
            "const std::exception&){return true;}}\n"
            "int main(){\n"
            ' assert(ParseBytes("0")==0);assert(ParseBytes("1844674407370'
            '9551615")==UINT64_MAX);\n'
            ' for(auto s:{"", "-1", "+1", " 1", "1 ", "1e3", "18446744073'
            '709551616"})assert(bad([&]{ParseBytes(s);}));\n'
            " assert(CheckedAdd(3,4)==7);assert(bad([]{CheckedAdd(UINT64_"
            "MAX,1);}));\n"
            " assert(CheckedMultiply(3,4)==12);assert(bad([]{CheckedMulti"
            "ply(UINT64_MAX,2);}));\n"
            " assert(CacheBytes(.5)==536870912);assert(bad([]{CacheBytes("
            "0);}));\n"
            " assert(bad([]{CacheBytes(std::numeric_limits<double>::infin"
            "ity());}));\n"
            " assert(bad([]{CacheBytes(std::numeric_limits<double>::quiet"
            "_NaN());}));\n"
            " ValidateIo(2,3,4,8);assert(bad([]{ValidateIo(0,3,4,8);}));\n"
            " assert(bad([]{ValidateIo(5,3,4,8);}));assert(bad([]{Validat"
            "eIo(2,3,9,8);}));\n"
            ' assert(PathsOverlap("/a/b","/a/b/c"));assert(!PathsOverlap('
            '"/a/b","/a/bc"));\n'
            ' assert(PathContains("/a/b","/a/b/c"));assert(!PathContains('
            '"/a/b","/a"));\n'
            ' assert(Hex("A =\\n")=="41203d0a");\n'
            ' const char* argv[]={"driver","--dispatch","serial","--host_'
            'budget_bytes","10"};\n'
            ' auto values=ParseStrictArguments(5,argv,{"dispatch","host_b'
            'udget_bytes"});assert(values.at("dispatch")=="serial");\n'
            ' const char* dup[]={"driver","--dispatch","serial","--dispat'
            'ch","pair"};\n'
            ' assert(bad([&]{ParseStrictArguments(5,dup,{"dispatch"});}))'
            ";\n"
            ' assert(bad([&]{ParseStrictArguments(5,argv,{"dispatch"});})'
            ");\n"
            " assert(bad([]{PairCoordinator c(static_cast<PairDispatch>(9"
            "9),{9,9},{1,1},{{{1,1},{1,1}}},std::chrono::seconds(1));}));"
            "\n"
            " PairCoordinator c(PairDispatch::kSerial,{9,9},{1,1},{{{1,1}"
            ",{1,1}}},std::chrono::seconds(2));\n"
            " std::promise<void> cancelled;auto ready=cancelled.get_futur"
            "e();std::atomic<bool> retire_failed{false};\n"
            " std::thread a([&]{\n"
            "   try{c.BeginPreparation(0);c.ReadyAndWait(0);c.SolverDone("
            '0);c.BeginOutput(0);throw std::runtime_error("write failure"'
            ");}\n"
            "   catch(...){c.Cancel(0,std::current_exception());cancelled"
            ".set_value();}\n"
            "   try{c.Retire(0);}catch(...){retire_failed=true;}\n"
            " });\n"
            " std::thread b([&]{\n"
            "   try{c.BeginPreparation(1);c.ReadyAndWait(1);c.SolverDone("
            "1);ready.wait();c.BeginOutput(1);}\n"
            "   catch(...){c.Cancel(1,std::current_exception());}\n"
            "   try{c.Retire(1);}catch(...){retire_failed=true;}\n"
            " });\n"
            " a.join();b.join();assert(!retire_failed);\n"
            " try{c.Finish();assert(false);}catch(const std::runtime_erro"
            'r& e){assert(std::string(e.what())=="write failure");}\n'
            " assert(c.ReservedBytes().host==0&&c.ReservedBytes().device="
            "=0);\n"
            " PairCoordinator incomplete(PairDispatch::kSerial,{9,9},{1,1"
            "},{{{1,1},{1,1}}},std::chrono::seconds(1));\n"
            " incomplete.Cancel(0,std::make_exception_ptr(std::runtime_er"
            'ror("first failure")));\n'
            " try{incomplete.Finish();assert(false);}catch(const std::exc"
            'eption& e){assert(std::string(e.what())=="first failure");}\n'
            " assert(incomplete.ReservedBytes().host==3&&incomplete.Reser"
            "vedBytes().device==3);\n"
            "}\n"
        )
        with tempfile.TemporaryDirectory(
            prefix="owned-pair-validation-"
        ) as tmp:
            cpp = pathlib.Path(tmp) / "test.cc"
            exe = pathlib.Path(tmp) / "test"
            cpp.write_text(source)
            p = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-pthread",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(ROOT / "src"),
                    str(cpp),
                    "-o",
                    str(exe),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            p = subprocess.run(
                [str(exe)], capture_output=True, text=True, timeout=10
            )
            self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    unittest.main()
