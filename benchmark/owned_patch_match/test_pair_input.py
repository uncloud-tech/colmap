# SPDX-License-Identifier: BSD-3-Clause
"Real extracted ThreadPool implementation; logging/hash aliases are CPU shims."

import pathlib
import resource
import subprocess
import tempfile
import unittest

from test_pair_integration import function

ROOT = pathlib.Path(__file__).resolve().parents[2]
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class PairInput(unittest.TestCase):
    def test_real_pool_exception_queue_is_drained(self):
        header = (ROOT / "src/colmap/util/threading.h").read_text()
        impl = (ROOT / "src/colmap/util/threading.cc").read_text()
        classes = header[
            header.index("class AggregateException") : header.index(
                "// A job queue class"
            )
        ]
        add = header[
            header.index(
                "template <class func_t, class... args_t>\n"
                "auto ThreadPool::AddTask"
            ) : header.index("template <typename T>\nJobQueue<T>::JobQueue()")
        ]
        bodies = "\n".join(
            function(impl, s)
            for s in (
                "ThreadPool::ThreadPool(",
                "ThreadPool::~ThreadPool(",
                "void ThreadPool::Stop(",
                "void ThreadPool::Wait(",
                "void ThreadPool::CheckFinishedTasks(",
                "void ThreadPool::WorkerFunc(",
                "std::thread::id ThreadPool::GetThreadId(",
            )
        )
        source = (
            (
                "\n"
                '#include "colmap/mvs/owned_pair_input.h"\n'
                "#include <atomic>\n"
                "#include <cassert>\n"
                "#include <condition_variable>\n"
                "#include <cstdlib>\n"
                "#include <functional>\n"
                "#include <iostream>\n"
                "#include <mutex>\n"
                "#include <queue>\n"
                "#include <thread>\n"
                "#include <unordered_map>\n"
                "#include <vector>\n"
                "#include <string>\n"
                "template<class K,class V>using NodeHashMap=std::unordered_ma"
                "p<K,V>;\n"
                "struct Fatal {template<class T>Fatal& operator<<(const T& x)"
                "{std::cerr<<x;return *this;}~Fatal(){std::abort();}};\n"
                "#define LOG(x) Fatal()\n"
                "int GetEffectiveNumThreads(int n){assert(n>0);return n;}\n"
                "#define private public\n"
            )
            + classes
            + "\n#undef private\n"
            + add
            + bodies
            + (
                "\n"
                "int main(int argc,char**){\n"
                " auto pool=std::make_unique<ThreadPool>(2);\n"
                " std::atomic<int> cancel_count{0};std::vector<std::shared_fu"
                "ture<void>> futures;\n"
                " futures.push_back(pool->AddTask([]{throw std::runtime_error"
                '("decode failure");}));\n'
                " futures.push_back(pool->AddTask([]{}));\n"
                " // Deterministically prove the checkers were queued before "
                "the destructor.\n"
                " {std::unique_lock<std::mutex> l(pool->mutex_);pool->finishe"
                "d_condition_.wait(l,[&]{return pool->tasks_.empty()&&pool->n"
                "um_active_workers_==0;});}\n"
                " if(argc>1){for(auto& f:futures){try{f.get();}catch(...){}}p"
                "ool.reset();return 99;}\n"
                " bool failed=false;\n"
                " try{colmap::mvs::owned::DrainPairInputs(pool,futures,{},[&]"
                "(std::exception_ptr){++cancel_count;});}\n"
                " catch(const std::runtime_error& e){failed=true;assert(std::"
                'string(e.what())=="decode failure");}\n'
                " assert(failed&&cancel_count>0&&!pool);\n"
                "}\n"
            )
        )
        with tempfile.TemporaryDirectory(prefix="owned-real-pool-") as tmp:
            cpp = pathlib.Path(tmp) / "test.cc"
            exe = pathlib.Path(tmp) / "test"
            cpp.write_text(source)
            p = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-pthread",
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
                [str(exe), "legacy"], capture_output=True, text=True, timeout=10
            )
            self.assertNotEqual(p.returncode, 0)
            self.assertIn(
                "Uncaught exceptions in thread pool destructor", p.stderr
            )
            p = subprocess.run(
                [str(exe)], capture_output=True, text=True, timeout=10
            )
            self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    unittest.main()
