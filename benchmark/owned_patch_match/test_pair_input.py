# SPDX-License-Identifier: BSD-3-Clause
"""Real extracted ThreadPool implementation; logging/hash aliases are CPU shims."""
import pathlib
import resource
import subprocess
import tempfile
import unittest
from test_pair_integration import function
ROOT=pathlib.Path(__file__).resolve().parents[2]
resource.setrlimit(resource.RLIMIT_CORE,(0,0))

class PairInput(unittest.TestCase):
    def test_real_pool_exception_queue_is_drained(self):
        header=(ROOT/'src/colmap/util/threading.h').read_text()
        impl=(ROOT/'src/colmap/util/threading.cc').read_text()
        classes=header[header.index('class AggregateException'):header.index('// A job queue class')]
        add=header[header.index('template <class func_t, class... args_t>\nauto ThreadPool::AddTask'):header.index('template <typename T>\nJobQueue<T>::JobQueue()')]
        bodies='\n'.join(function(impl,s) for s in (
            'ThreadPool::ThreadPool(', 'ThreadPool::~ThreadPool(',
            'void ThreadPool::Stop(', 'void ThreadPool::Wait(',
            'void ThreadPool::CheckFinishedTasks(', 'void ThreadPool::WorkerFunc(',
            'std::thread::id ThreadPool::GetThreadId('))
        source=r'''
#include "colmap/mvs/owned_pair_input.h"
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>
template<class K,class V>using NodeHashMap=std::unordered_map<K,V>;
struct Fatal {template<class T>Fatal& operator<<(const T& x){std::cerr<<x;return *this;}~Fatal(){std::abort();}};
#define LOG(x) Fatal()
int GetEffectiveNumThreads(int n){assert(n>0);return n;}
#define private public
'''+classes+'\n#undef private\n'+add+bodies+r'''
int main(int argc,char**){
 auto pool=std::make_unique<ThreadPool>(2);
 std::atomic<int> cancel_count{0};std::vector<std::shared_future<void>> futures;
 futures.push_back(pool->AddTask([]{throw std::runtime_error("decode failure");}));
 futures.push_back(pool->AddTask([]{}));
 // Deterministically prove the checkers were queued before the destructor.
 {std::unique_lock<std::mutex> l(pool->mutex_);pool->finished_condition_.wait(l,[&]{return pool->tasks_.empty()&&pool->num_active_workers_==0;});}
 if(argc>1){for(auto& f:futures){try{f.get();}catch(...){}}pool.reset();return 99;}
 bool failed=false;
 try{colmap::mvs::owned::DrainPairInputs(pool,futures,{},[&](std::exception_ptr){++cancel_count;});}
 catch(const std::runtime_error& e){failed=true;assert(std::string(e.what())=="decode failure");}
 assert(failed&&cancel_count>0&&!pool);
}
'''
        with tempfile.TemporaryDirectory(prefix='owned-real-pool-') as tmp:
            cpp=pathlib.Path(tmp)/'test.cc';exe=pathlib.Path(tmp)/'test';cpp.write_text(source)
            p=subprocess.run(['g++','-std=c++17','-pthread','-I',str(ROOT/'src'),str(cpp),'-o',str(exe)],capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([str(exe),'legacy'],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Uncaught exceptions in thread pool destructor',p.stderr)
            p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
            self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__':unittest.main()
