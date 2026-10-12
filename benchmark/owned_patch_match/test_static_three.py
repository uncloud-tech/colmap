# SPDX-License-Identifier: BSD-3-Clause
"""Bounded real coordinator/runner/events CPU proof; no CUDA/build claim."""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP = r'''
#include "colmap/mvs/owned_pair_events.h"
#include "colmap/mvs/owned_pair_run.h"
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <map>
using namespace colmap::mvs::owned;
using namespace std::chrono_literals;
struct Gate {
 std::mutex mutex; std::condition_variable cv; bool open=false;
 void Set(){std::lock_guard<std::mutex> l(mutex);open=true;cv.notify_all();}
 void Wait(){std::unique_lock<std::mutex> l(mutex);assert(cv.wait_for(l,3s,[&]{return open;}));}
};
const std::array<ByteCharge,3> charges{{{13,10},{13,10},{10,8}}};
template<class F> bool Reject(F f){try{f();}catch(const std::exception&){return true;}return false;}
void Admission(){
 PairCoordinator exact(PairDispatch::kPair,{38,30},{2,2},charges,3s,{},2);
 assert(exact.JobCount()==3 && exact.ReservedBytes().host==38);
 assert(Reject([&]{exact.BeginPreparation(3);}));
 assert(Reject([&]{exact.ReadyAndWait(0);}));
 assert(Reject([&]{exact.SolverDone(2);}));
 assert(Reject([&]{exact.BeginOutput(1);}));
 assert(Reject([&]{exact.Retire(2);}));
 assert(Reject([&]{exact.Finish();}));
 for(auto cap: {ByteCharge{37,30}, ByteCharge{38,29}})
  assert(Reject([&]{PairCoordinator c(PairDispatch::kPair,cap,{2,2},charges,3s,{},2);}));
 for(size_t j=0;j<3;++j){
  auto bad=charges;bad[j].host=0;
  assert(Reject([&]{PairCoordinator c(PairDispatch::kPair,{38,30},{2,2},bad,3s,{},2);}));
  bad=charges;bad[j].device=UINT64_MAX;
  assert(Reject([&]{PairCoordinator c(PairDispatch::kPair,{UINT64_MAX,UINT64_MAX},{2,2},bad,3s,{},2);}));
 }
 for(size_t permits:{0,1,4}) assert(Reject([&]{PairCoordinator c(PairDispatch::kPair,{38,30},{2,2},charges,3s,{},permits);}));
 assert(Reject([&]{PairCoordinator c(PairDispatch::kSerial,{38,30},{2,2},charges,3s,{},2);}));
}
void Run(size_t permits,int fail_job=-1,int phase=-1,bool spawn_fail=false,bool unknown=false,bool early_done=false){
 PairEvents events("pair",3); Gate c_ready,c_started,a_done;
 std::mutex mutex; std::vector<std::pair<size_t,std::string>> trace;
 std::map<std::pair<size_t,std::string>,uint64_t> sequence;
 std::atomic<int> alive{0};
 auto record=[&](size_t job,const std::string& event){
  std::lock_guard<std::mutex> l(mutex);trace.emplace_back(job,event);
 };
 PairCoordinator c(PairDispatch::kPair,{38,30},{2,2},charges,3s,[&](const PairEvent& e){
  record(e.job,e.event);
  {std::lock_guard<std::mutex> l(mutex);sequence[{e.job,e.event}]=e.sequence;}
  events.Coordinator(e);
  if(e.job==0 && e.event=="solver_end")a_done.Set();
  if(early_done && e.job==2 && e.event=="prep_start")a_done.Wait();
  if(e.job==2 && e.event=="ready")c_ready.Set();
  if(e.job==2 && e.event=="solver_start")c_started.Set();
 },permits);
 size_t starts=0;
 auto factory=[&](std::function<void()> task){
  if(spawn_fail && starts++==2)throw std::runtime_error("third thread failed");
  return std::thread(std::move(task));
 };
 bool failed=false;
 try {
 RunPairOwners(c,[&](size_t j){
  events.Owner(j); events.Identity(j,17+j,0);
  struct Resource{std::atomic<int>& alive;Resource(std::atomic<int>& a):alive(a){++alive;}~Resource(){--alive;}} resource(alive);
  try{
   c.BeginPreparation(j);
   if(int(j)==fail_job && phase==0)throw std::runtime_error("injected");
   c.ReadyAndWait(j);
   if(phase==3 && j<2){c_ready.Wait();throw std::runtime_error("injected");}
   if(fail_job<0 && !spawn_fail && !unknown && !early_done && j<2){
    c_ready.Wait(); if(permits==3)c_started.Wait();
   }
   if(int(j)==fail_job && phase==1)throw std::runtime_error("injected");
   c.SolverDone(j);c.BeginOutput(j);
   if(int(j)==fail_job && phase==2)throw std::runtime_error("injected");
   c.EndOutput(j);c.AwaitOutputs(j);
  }catch(...){c.CancelAndAwaitCleanup(j,std::current_exception(),[]{});throw;}
 },[&](size_t j,const char* event){record(j,event);events.Emit(j,event);},factory,[&](size_t j){
  if(unknown && j==2)throw std::runtime_error("cleanup unproven");
 });
 }catch(const std::runtime_error& error){
  failed=true;
  assert(std::string(error.what())==(spawn_fail?"third thread failed":unknown?"cleanup unproven":"injected"));
 }
 assert(failed==(fail_job>=0 || spawn_fail || unknown)); assert(alive==0);
 assert(c.ReservedBytes().host==(unknown?38:0));
 auto index=[&](size_t j,const char* e){for(size_t i=0;i<trace.size();++i)if(trace[i]==std::make_pair(j,std::string(e)))return i;return trace.size();};
 auto seq=[&](size_t j,const char* e){return sequence.at({j,e});};
 const size_t batch=PairEvents::kBatch;
 size_t retired=0,released=0;
 for(const auto& e:trace){retired+=e.second=="retired";released+=e.second=="ledger_release";}
 assert(retired==3 && released==(unknown?0:1));
 if(!unknown)assert(index(batch,"owners_joined")<index(batch,"ledger_release"));
 if(!failed){
  events.Finished();
  for(size_t j=0;j<3;++j){
   assert(seq(j,"output_end")<seq(0,"retired"));
   assert(seq(j,"output_end")<seq(1,"retired"));
   assert(seq(j,"output_end")<seq(2,"retired"));
   assert(index(j,"cleanup_complete")<index(j,"retired"));
  }
  assert(seq(0,"prep_end")<seq(1,"prep_start"));
  assert(seq(1,"prep_end")<seq(2,"prep_start"));
  assert(seq(0,"output_end")<seq(1,"output_start"));
  assert(seq(1,"output_end")<seq(2,"output_start"));
  if(early_done)assert(seq(0,"solver_end")<seq(2,"ready"));
  if(permits==2)assert(std::min(seq(0,"solver_end"),seq(1,"solver_end"))<seq(2,"solver_release"));
  else if(!early_done)assert(seq(2,"solver_start")<std::min(seq(0,"solver_end"),seq(1,"solver_end")));
  assert(Reject([&]{c.SolverDone(2);})); // stale completion
  assert(Reject([&]{c.Retire(2);}));
  c.Finish(); // no duplicate release
 } else if(phase==3) {
  assert(index(2,"solver_release")==trace.size());
  assert(index(2,"solver_start")==trace.size());
 } else if(fail_job<2 && phase==1) {
  // No successful A/B completion may be fabricated from failure.
  assert(index(size_t(fail_job),"solver_end")==trace.size());
 }
}
int main(){Admission();for(size_t permits:{2,3}){
 Run(permits);Run(permits,-1,-1,false,false,true);
 for(int j=0;j<3;++j)for(int phase=0;phase<3;++phase)Run(permits,j,phase);
 Run(permits,-1,-1,true);Run(permits,-1,-1,false,true);
} Run(2,0,3);std::cout<<"STATIC_THREE_CPU_OK\n";}
'''


class StaticThreeTest(unittest.TestCase):
    def test_actual_driver_epoch(self):
        driver = (ROOT / 'src/colmap/exe/owned_pair.cc').read_text()
        settings = driver[driver.index('struct PairSettings {'):]
        settings = settings[:settings.index('\n};') + 3]
        epoch = driver[driver.index('template <class RunJob>\nvoid RunPairEpoch('):
                       driver.index('using PreparedHostJob =')]
        source = r'''
#include "colmap/mvs/owned_pair_events.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_pair_validation.h"
#include <filesystem>
#include <memory>
#include <cassert>
namespace pair = colmap::mvs::owned;
namespace fs = std::filesystem;
namespace mvs {
struct Options { int num_threads = 1; };
struct Problem {};
struct PatchMatchController {
 struct OwnedPairHooks {
  std::function<void(size_t,const Options&,const Problem&)> check_inputs,metadata;
  std::function<void(size_t,int,int)> sweep;
  std::function<void(size_t,const char*)> output_phase;
 };
};
}
struct Globals { int seed=0,level=0,severity=0; bool color=true; std::string target="stderr",path; };
struct JobOptions { Globals globals; fs::path workspace;
 std::unique_ptr<mvs::Options> patch_match_stereo=std::make_unique<mvs::Options>(); };
template<class T> std::string Value(const T& value){std::ostringstream s;s<<value;return s.str();}
void CheckInputs(const mvs::Options&,const mvs::Problem&,pair::ByteCharge,const fs::path&) {}
void Metadata(pair::PairEvents& e,size_t j,const mvs::Options&,const mvs::Problem&){e.Identity(j,17+j,0);}
''' + settings + epoch + r'''
int main(){
 for(size_t permits:{2,3}){
  JobOptions a,b,c;
  PairSettings settings{"pair",{38,30},{2,2},{{{13,10},{13,10},{10,8}}},3,1,false,permits};
  auto run=[](JobOptions& job,pair::PairCoordinator& coord,size_t j,const auto& hooks){
   coord.BeginPreparation(j);hooks.metadata(j,*job.patch_match_stereo,mvs::Problem{});
   coord.ReadyAndWait(j);coord.SolverDone(j);coord.BeginOutput(j);
   for(auto phase:{"depth_begin","depth_end","normal_begin","normal_end"})hooks.output_phase(j,phase);
   coord.EndOutput(j);coord.AwaitOutputs(j);
  };
  RunPairEpoch(a,b,settings,run,{},{},{},&c);
 }
}
'''
        temp = Path(tempfile.mkdtemp(prefix='static-three-epoch-'))
        program, binary = temp / 'test.cc', temp / 'test'
        program.write_text(source)
        compiled = subprocess.run(
            ['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O1',
             '-pthread', '-I', str(ROOT / 'src'), str(program), '-o', str(binary)],
            capture_output=True, text=True, timeout=60)
        (temp / 'compile.log').write_text(compiled.stdout + compiled.stderr)
        print(f'STATIC_THREE_EPOCH_ARTIFACT {temp} compile_exit={compiled.returncode}')
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
        (temp / 'run.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [dict(field.split('=', 1) for field in line.split()[1:])
                   for line in result.stdout.splitlines() if line.startswith('OWNED_PAIR_EVENT ')]
        validated = [r for r in records if r['event'] == 'validated']
        self.assertEqual([r['static_three_solver_permits'] for r in validated], ['2', '3'])
        for record in validated:
            self.assertEqual(record['job_count'], '3')
            self.assertEqual((record['hc'], record['dc'], record['io_c']), ('10', '8', '1'))
        self.assertEqual(sum(r['event'] == 'batch_finished' for r in records), 2)
        self.assertEqual(sum(r['event'] == 'normal_end' for r in records), 6)

    def test_bounded_coordinator_runner_and_events(self):
        temp = Path(tempfile.mkdtemp(prefix='static-three-cpu-'))
        source, binary = temp / 'test.cc', temp / 'test'
        source.write_text(CPP)
        compile_result = subprocess.run(
            ['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O1',
             '-pthread', '-I', str(ROOT / 'src'), str(source), '-o', str(binary)],
            capture_output=True, text=True, timeout=60)
        (temp / 'compile.log').write_text(compile_result.stdout + compile_result.stderr)
        print(f'STATIC_THREE_ARTIFACT {temp} compile_exit={compile_result.returncode}')
        self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
        (temp / 'run.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('STATIC_THREE_CPU_OK', result.stdout)
        records = [dict(field.split('=', 1) for field in line.split()[1:])
                   for line in result.stdout.splitlines() if line.startswith('OWNED_PAIR_EVENT ')]
        successes = [r for r in records if r['event'] == 'batch_finished']
        self.assertEqual(len(successes), 4)
        for success in successes:
            batch = [r for r in records if r['batch'] == success['batch']]
            self.assertEqual({r['job'] for r in batch if r['event'] == 'ready'}, {'A', 'B', 'C'})
            self.assertEqual(sum(r['event'] == 'all_ready' for r in batch), 1)
            self.assertFalse(any(r['event'] == 'both_ready' for r in batch))


if __name__ == '__main__':
    unittest.main(verbosity=2)
