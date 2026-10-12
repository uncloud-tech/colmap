# SPDX-License-Identifier: BSD-3-Clause
"""Real-thread bounded rolling coordinator, runner and actual CLI CPU seams.

No CUDA construction, native product build or sampled-memory certification.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP = r'''
#include "colmap/mvs/owned_pair_events.h"
#include "colmap/mvs/owned_pair_run.h"
#include <cassert>
#include <atomic>
#include <map>
#include <iostream>
#include <cstdlib>
using namespace colmap::mvs::owned;
using namespace std::chrono_literals;
struct Gate {
 std::mutex mutex; std::condition_variable cv; bool open=false;
 void Set(){std::lock_guard<std::mutex> l(mutex);open=true;cv.notify_all();}
 void Wait(){std::unique_lock<std::mutex> l(mutex);assert(cv.wait_for(l,3s,[&]{return open;}));}
};
const std::array<ByteCharge,3> slots{{{13,10},{13,10},{12,9}}};
const std::vector<ByteCharge> forecasts{{13,10},{13,10},{10,8},{13,10},{13,10},{12,9}};
template<class F> bool Reject(F f){try{f();}catch(const std::exception&){return true;}return false;}
void Admission(){
 for(ByteCharge cap:{ByteCharge{39,31},ByteCharge{40,30}})
  assert(Reject([&]{PairCoordinator c(cap,{2,2},slots,forecasts,3s);}));
 for(ByteCharge bad:{ByteCharge{0,1},ByteCharge{1,0},ByteCharge{14,1},ByteCharge{1,11}}){
  auto f=forecasts;f.back()=bad;
  assert(Reject([&]{PairCoordinator c({40,31},{2,2},slots,f,3s);}));
 }
 // Exact production reviewed fit fixture, including the single 3% allowance.
 constexpr uint64_t g=1ULL<<30;
 const std::array<ByteCharge,3> real{{{13*g,10*g},{13*g,10*g},{12*g,9*g}}};
 for(ByteCharge f:{ByteCharge{13343149594ULL,10447295974ULL},ByteCharge{12338645847ULL,9629641669ULL}}){
  assert(PairCoordinator::Fits(f,real[0]));assert(PairCoordinator::Fits(f,real[1]));
 }
 assert(!PairCoordinator::Fits({13343149594ULL,10447295974ULL},real[2]));
 assert(PairCoordinator::Fits({10438883260ULL,7995452196ULL},real[2]));
 assert(PairCoordinator::Fits({12338645847ULL,9629641669ULL},real[2]));
}
void Turnover(){
 Gate initial,slot2_joined,retired0,residual_release,successor_prepared;
 std::atomic<int> initial_count{0},alive{0},peak{0};
 std::mutex mutex; std::vector<std::pair<size_t,std::string>> trace;
 std::vector<PairEvent> transitions;
 auto record=[&](size_t j,const std::string& e){std::lock_guard<std::mutex> l(mutex);trace.emplace_back(j,e);};
 PairEvents events("pair",6,true); PairCoordinator* ptr=nullptr;
 PairCoordinator c({40,31},{2,2},slots,forecasts,3s,[&](const PairEvent& e){
  if(e.event=="admitted")events.BindSlot(e.job,ptr->Slot(e.job));
  events.Coordinator(e);record(e.job,e.event);
  {std::lock_guard<std::mutex> l(mutex);transitions.push_back(e);}
  if(e.job==2 && e.event=="owner_joined")slot2_joined.Set();
  if(e.job==0 && e.event=="retired")retired0.Set();
 });ptr=&c;
 std::thread delayed([&]{
  retired0.Wait();
  assert(c.AdmittedCount()==3); // C's joined empty slot does not fit FIFO head D.
  assert(c.ReservedBytes().host==40);
  residual_release.Set();
 });
 size_t created=0;
 auto factory=[&](std::function<void()> task){
  size_t job=created++;
  return std::thread([&,job,task=std::move(task)]() mutable {
   task();
   if(job==0){residual_release.Wait();record(job,"residual_destroyed");}
   task={}; // Real per-thread callback destruction before parent join returns.
  });
 };
 RunRollingOwners(c,[&](size_t j){
  events.Owner(j);events.Identity(j,17+j,0);
  struct Resource{std::atomic<int>& a;Resource(std::atomic<int>& a):a(a){++a;}~Resource(){--a;}} resource(alive);
  int now=alive.load(),old=peak.load();while(now>old&&!peak.compare_exchange_weak(old,now)){}
  try{
   c.BeginPreparation(j);
   if(j==3){
    assert(c.Slot(0)==c.Slot(3));
    assert(Reject([&]{c.ReadyAndWait(0);}));assert(Reject([&]{c.SolverDone(0);}));
    assert(Reject([&]{c.EndOutput(0);}));assert(Reject([&]{c.Retire(0);}));
    c.Cancel(0,std::make_exception_ptr(std::runtime_error("stale cancellation")));
    successor_prepared.Set();
   }
   c.ReadyAndWait(j);
   if(j<3 && ++initial_count==3)initial.Set();
   if(j<3)initial.Wait();
   if(j==0)slot2_joined.Wait();
   if(j==1)successor_prepared.Wait();
   c.SolverDone(j);c.BeginOutput(j);c.EndOutput(j);c.AwaitOutputs(j);
  }catch(...){c.CancelAndAwaitCleanup(j,std::current_exception(),[]{});throw;}
 },[&](size_t j,const char* e){record(j,e);events.Emit(j,e);},factory);
 delayed.join();events.Finished();assert(alive==0 && peak<=3);assert(c.ReservedBytes().host==0);
 auto index=[&](size_t j,const char* e){for(size_t i=0;i<trace.size();++i)if(trace[i]==std::make_pair(j,std::string(e)))return i;assert(false);return size_t(0);};
 assert(index(2,"output_end")<index(0,"output_end"));
 assert(index(2,"owner_joined")<index(0,"output_end"));
 assert(index(0,"cleanup_complete")<index(0,"retired"));
 assert(index(0,"retired")<index(0,"residual_destroyed"));
 assert(index(0,"residual_destroyed")<index(0,"owner_joined"));
 assert(index(0,"owner_joined")<index(3,"admitted"));
 assert(index(3,"admitted")<index(3,"prep_start"));
 assert(index(3,"prep_start")<index(1,"output_end"));
 int prepared=0,output=0;size_t next=0,release=0;
 std::sort(transitions.begin(),transitions.end(),[](const PairEvent& a,const PairEvent& b){return a.sequence<b.sequence;});
 for(const auto& event:transitions){
  const auto j=event.job;const auto& e=event.event;
  if(e=="prep_start"){assert(j==next++);assert(++prepared==1);}
  if(e=="prep_end")assert(--prepared==0);
  if(e=="output_start")assert(++output==1);
  if(e=="output_end")assert(--output==0);
  if(e=="ledger_release")++release;
 }
 assert(next==6&&release==1);
}
void Failure(int phase,int failjob){
 std::mutex mutex; std::vector<PairEvent> trace;
 PairCoordinator c({40,31},{2,2},slots,forecasts,3s,[&](const PairEvent& e){
  {std::lock_guard<std::mutex> l(mutex);trace.push_back(e);}
  if(phase==7 && int(e.job)==failjob && e.event=="admitted")throw std::runtime_error("injected");
  if(phase==8 && int(e.job)==failjob && e.event=="retired")throw std::runtime_error("injected");
  if(phase==9 && int(e.job)==failjob && e.event=="owner_joined")throw std::runtime_error("injected");
 });
 std::atomic<int> alive{0};size_t starts=0;
 auto factory=[&](std::function<void()> task){
  if(phase==4 && int(starts++)==failjob)throw std::runtime_error("injected");
  if(phase==10 && int(starts++)==failjob)return std::thread{};
  return std::thread(std::move(task));
 };
 bool failed=false;
 try{RunRollingOwners(c,[&](size_t j){
  if(phase==0 && int(j)==failjob)throw std::runtime_error("injected"); // Controller ctor.
  struct Resource{std::atomic<int>& a;Resource(std::atomic<int>& a):a(a){++a;}~Resource(){--a;}} resource(alive);
  try{
   c.BeginPreparation(j);
   if(phase==1 && int(j)==failjob)throw std::runtime_error("injected");
   c.ReadyAndWait(j);
   if(phase==2 && int(j)==failjob)throw std::runtime_error("injected");
   c.SolverDone(j);c.BeginOutput(j);
   if(phase==3 && int(j)==failjob)throw std::runtime_error("injected");
   c.EndOutput(j);c.AwaitOutputs(j);
  }catch(...){c.CancelAndAwaitCleanup(j,std::current_exception(),[]{});throw;}
 },[&](size_t j,const char*){if(phase==6&&int(j)==failjob)throw std::runtime_error("injected");},factory,
 [&](size_t j){if(phase==5&&int(j)==failjob)throw std::runtime_error("injected");});}
 catch(const std::runtime_error& e){failed=true;assert(std::string(e.what())==(phase==10?"thread factory returned no owner":"injected"));}
 assert(failed && alive==0);assert(c.ReservedBytes().host==(phase==5?40:0));
 size_t retired=0,joined=0,released=0;uint64_t cancel=UINT64_MAX;
 for(const auto& e:trace){if(e.event=="cancelled")cancel=std::min(cancel,e.sequence);}
 for(const auto& e:trace){
  retired+=e.event=="retired";joined+=e.event=="owner_joined";released+=e.event=="ledger_release";
  if(e.event=="admitted")assert(e.sequence<cancel);
 }
 assert(retired==c.AdmittedCount()&&joined==retired&&released==(phase==5?0:1));
}
void ActiveFailure(){
 Gate started,cancelled;std::atomic<int> count{0},active{0},destroyed{0};
 PairCoordinator c({40,31},{2,2},slots,forecasts,3s,[&](const PairEvent& e){
  if(e.event=="cancelled")cancelled.Set();
 });
 bool failed=false;
 try{RunRollingOwners(c,[&](size_t j){
  struct Resource{std::atomic<int>& active;std::atomic<int>& destroyed;
   ~Resource(){assert(active==0);++destroyed;}
  } resource{active,destroyed};
  c.BeginPreparation(j);c.ReadyAndWait(j);++active;
  if(++count==3)started.Set();
  started.Wait();
  try{
   if(j==0)throw std::runtime_error("active failure");
   cancelled.Wait();throw std::runtime_error("peer cancellation");
  }catch(...){c.CancelAndAwaitCleanup(j,std::current_exception(),[&]{--active;});throw;}
 },[](size_t,const char*){});}
 catch(const std::runtime_error& e){failed=true;assert(std::string(e.what())=="active failure");}
 assert(failed&&destroyed==3&&c.AdmittedCount()==3&&c.ReservedBytes().host==0);
}
int main(int argc,char**){
 if(argc>1){
  std::set_terminate([]{std::_Exit(86);});
  PairCoordinator c({40,31},{2,2},slots,forecasts,3s);c.AdmitNext();
  c.CancelAndAwaitCleanup(0,std::make_exception_ptr(std::runtime_error("uncertain")),[]{throw std::runtime_error("drain unproven");});
  assert(false);
 }
 Admission();Turnover();ActiveFailure();for(int phase=0;phase<=10;++phase)for(int job:{0,2,3,5})Failure(phase,job);std::cout<<"ROLLING_THREE_CPU_OK\n";
}
'''


class RollingThreeTest(unittest.TestCase):
    def compile_run(self, source, name, args=(), expected_exit=0):
        temp = Path(tempfile.mkdtemp(prefix=f'rolling-three-{name}-'))
        program, binary = temp / 'test.cc', temp / 'test'
        program.write_text(source)
        result = subprocess.run(
            ['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O1',
             '-pthread', '-I', str(ROOT / 'src'), str(program), '-o', str(binary)],
            capture_output=True, text=True, timeout=60)
        (temp / 'compile.log').write_text(result.stdout + result.stderr)
        print(f'ROLLING_THREE_ARTIFACT {temp} compile_exit={result.returncode}')
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([str(binary), *args], capture_output=True,
                                text=True, timeout=30)
        (temp / 'run.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, expected_exit, result.stdout + result.stderr)
        return result.stdout

    def test_uncertain_drain_is_failstop(self):
        self.compile_run(CPP, 'uncertain-drain', args=('uncertain',), expected_exit=86)

    def test_actual_native_preparation_uses_generation_not_slot(self):
        native = (ROOT / 'src/colmap/mvs/patch_match.cc').read_text()
        prepare = native[native.index('std::unique_ptr<PatchMatchController::OwnedPreparedJob>'):
                         native.index('void PatchMatchController::RunOwnedPairJob(')]
        source = r'''
#include <cassert>
#include <memory>
#include <vector>
#include <string>
#include <cstddef>
#define THROW_CHECK(x) assert(x)
#define THROW_CHECK_EQ(x,y) assert((x)==(y))
#define THROW_CHECK_GE(x,y) assert((x)>=(y))
#define THROW_CHECK_GT(x,y) assert((x)>(y))
#define THROW_CHECK_LE(x,y) assert((x)<=(y))
#define THROW_CHECK_NOTNULL(x) assert((x)!=nullptr)
enum class PatchMatchExecution { kOwnedCuda };
template<class T> std::vector<T> CSVToVector(const std::string&){return {0};}
class PatchMatch { public: void Check() {} };
class ThreadPool { public: explicit ThreadPool(int) {} };
class PatchMatchController {
 public:
 struct Options { std::string gpu_index="0"; int num_threads=1; } options_;
 struct OwnedPreparedJob { std::unique_ptr<PatchMatch> patch_match; };
 struct OwnedPairHooks {};
 PatchMatchExecution execution_=PatchMatchExecution::kOwnedCuda;
 std::unique_ptr<int> workspace_;
 std::unique_ptr<ThreadPool> io_thread_pool_,thread_pool_;
 std::vector<int> problems_,depth_ranges_,gpu_indices_;
 size_t seen=100;
 std::unique_ptr<OwnedPreparedJob> PrepareOwnedPairJob(size_t,const OwnedPairHooks&,int=0);
 void ReadWorkspace(){workspace_=std::make_unique<int>(1);}
 void ReadProblems(){problems_={0};}
 void ReadGpuIndices(){gpu_indices_={0};}
 void ProcessProblemInputs(const Options&,int,OwnedPreparedJob* p,size_t j,const OwnedPairHooks&){
  seen=j;p->patch_match=std::make_unique<PatchMatch>();
 }
};
''' + prepare + r'''
int main(){for(size_t j=0;j<6;++j){
 PatchMatchController controller;
 auto prepared=controller.PrepareOwnedPairJob(j,{});
 assert(prepared&&prepared->patch_match&&controller.seen==j);
 assert(!controller.workspace_&&!controller.io_thread_pool_);
 assert(controller.problems_.empty()&&controller.gpu_indices_.empty());
}}
'''
        self.compile_run(source, 'native-preparation')

    def test_actual_driver_feed_and_forecast_reader(self):
        driver = (ROOT / 'src/colmap/exe/owned_pair.cc').read_text()
        settings = driver[driver.index('struct PairSettings {'):]
        settings = settings[:settings.index('\n};') + 3]
        feed = driver[driver.index('// Literal positive integer pairs:'):
                      driver.index('template <class RunJob, class PrepareJob =')]
        source = r'''
#include "colmap/mvs/owned_pair_events.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_pair_validation.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <cassert>
#include <atomic>
namespace pair = colmap::mvs::owned;
namespace fs = std::filesystem;
namespace mvs {
struct Options { int num_threads=1,reference=17; };
struct Problem {};
struct PatchMatchController {
 struct OwnedPairHooks {
  int owned_tile_threads=32;
  std::function<void(size_t,const Options&,const Problem&)> check_inputs,metadata;
  std::function<void(size_t,int,int)> sweep;
  std::function<void(size_t,const char*)> output_phase;
 };
};
}
struct JobOptions { fs::path workspace;
 std::unique_ptr<mvs::Options> patch_match_stereo=std::make_unique<mvs::Options>(); };
template<class T> std::string Value(const T& value){std::ostringstream s;s<<value;return s.str();}
void CheckInputs(const mvs::Options&,const mvs::Problem&,pair::ByteCharge,const fs::path&) {}
void Metadata(pair::PairEvents& e,size_t j,const mvs::Options& o,const mvs::Problem&,int tile){e.Identity(j,o.reference,0);e.Emit(j,"effective_metadata_begin","owned_tile_threads="+Value(tile));}
''' + settings + feed + r'''
template<class F> bool Reject(F f){try{f();}catch(const std::exception&){return true;}return false;}
int main(){
 auto path=fs::temp_directory_path()/("rolling-forecast-cpu-"+std::to_string(getpid()));
 auto put=[&](const std::string& text){std::ofstream out(path);out<<text;};
 put("13 10\n13 10\n10 8\n13 10\n13 10\n12 9\n");
 auto forecasts=ReadRollingForecasts(path,6);assert(forecasts.size()==6);
 for(auto text:{"", "0 1\n", "1 0\n", "-1 2\n", "1 2 3\n", "1\n", "18446744073709551616 1\n", "1 2\n\n"}){
  put(text);assert(Reject([&]{ReadRollingForecasts(path,1);}));
 }
 put("1 2\n");assert(Reject([&]{ReadRollingForecasts(path,6);}));fs::remove(path);
 assert(Reject([&]{ReadRollingForecasts(path,1);}));
 std::vector<std::unique_ptr<JobOptions>> jobs;
 for(size_t j=0;j<6;++j){jobs.push_back(std::make_unique<JobOptions>());jobs.back()->patch_match_stereo->reference=17+j;}
 PairSettings settings{"pair",{40,31},{2,2},{{{13,10},{13,10},{12,9}}},3,1,false,3,16,true};
 std::atomic<int> calls{0};
 auto run=[&](JobOptions& job,pair::PairCoordinator& c,size_t j,const auto& hooks){
  ++calls;assert(j==size_t(job.patch_match_stereo->reference-17));assert(hooks.owned_tile_threads==16);
  try{
   c.BeginPreparation(j);hooks.check_inputs(j,*job.patch_match_stereo,mvs::Problem{});
   hooks.metadata(j,*job.patch_match_stereo,mvs::Problem{});c.ReadyAndWait(j);
   hooks.sweep(j,0,0);c.SolverDone(j);c.BeginOutput(j);
   for(auto phase:{"depth_begin","depth_end","normal_begin","normal_end"})hooks.output_phase(j,phase);
   c.EndOutput(j);c.AwaitOutputs(j);
  }catch(...){c.CancelAndAwaitCleanup(j,std::current_exception(),[]{});throw;}
 };
 auto impossible=forecasts;impossible.back()={14,8};
 assert(Reject([&]{RunRollingFeed(jobs,settings,impossible,run);}));assert(calls==0);
 RunRollingFeed(jobs,settings,forecasts,run);assert(calls==6);
}
'''
        output = self.compile_run(source, 'driver')
        records = [dict(field.split('=', 1) for field in line.split()[1:])
                   for line in output.splitlines()
                   if line.startswith('OWNED_PAIR_EVENT ')]
        successes = [r for r in records if r['event'] == 'batch_finished']
        self.assertEqual(len(successes), 1)
        own = [r for r in records if r['batch'] == successes[0]['batch']]
        self.assertEqual(sum(r['event'] == 'admitted' for r in own), 6)
        self.assertEqual(sum(r['event'] == 'normal_end' for r in own), 6)
        self.assertEqual(sum(r['event'] == 'owner_joined' for r in own), 6)
        self.assertEqual(sum(r['event'] == 'owners_joined' for r in own), 1)
        self.assertEqual(sum(r['event'] == 'ledger_release' for r in own), 1)
        validated = next(r for r in own if r['event'] == 'validated')
        self.assertEqual(validated['owned_tile_threads'], '16')
        self.assertEqual(validated['job_count'], '6')
        self.assertEqual(validated['hc'], '12')

    def test_real_threads_fit_postjoin_generations_failures(self):
        output = self.compile_run(CPP, 'threads')
        self.assertIn('ROLLING_THREE_CPU_OK', output)
        records = [dict(field.split('=', 1) for field in line.split()[1:])
                   for line in output.splitlines()
                   if line.startswith('OWNED_PAIR_EVENT ')]
        self.assertTrue(records)
        self.assertTrue(all(r['schema'] == '2' and r['feed_mode'] == 'rolling'
                            for r in records))
        for j in range(6):
            own = [r for r in records if r['job'] == f'J{j}']
            self.assertTrue(own)
            self.assertEqual({r['queue_index'] for r in own}, {str(j)})
            self.assertEqual({r['generation'] for r in own}, {str(j + 1)})
            self.assertEqual(len({r['slot'] for r in own}), 1)
            ready = next(r for r in own if r['event'] == 'ready')
            self.assertEqual(ready['reference'], str(17 + j))
            self.assertNotEqual(ready['owner_tid'], '0')
        batch = [r for r in records if r['job'] == 'batch']
        self.assertTrue(all(r['slot'] == '-1' and r['generation'] == '0' for r in batch))
        self.assertEqual(batch[-1]['event'], 'batch_finished')


if __name__ == '__main__':
    unittest.main(verbosity=2)
