# SPDX-License-Identifier: BSD-3-Clause
"""Real-thread bounded rolling coordinator, runner and actual CLI CPU seams.

No CUDA construction, native product build or sampled-memory certification.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP = (
    "\n"
    '#include "colmap/mvs/owned_pair_events.h"\n'
    '#include "colmap/mvs/owned_pair_run.h"\n'
    "#include <cassert>\n"
    "#include <atomic>\n"
    "#include <map>\n"
    "#include <iostream>\n"
    "#include <cstdlib>\n"
    "using namespace colmap::mvs::owned;\n"
    "using namespace std::chrono_literals;\n"
    "struct Gate {\n"
    " std::mutex mutex; std::condition_variable cv; bool open=fal"
    "se;\n"
    " void Set(){std::lock_guard<std::mutex> l(mutex);open=true;c"
    "v.notify_all();}\n"
    " void Wait(){std::unique_lock<std::mutex> l(mutex);assert(cv"
    ".wait_for(l,3s,[&]{return open;}));}\n"
    "};\n"
    "const std::array<ByteCharge,3> slots{{{13,10},{13,10},{12,9}"
    "}};\n"
    "const std::vector<ByteCharge> forecasts{{13,10},{13,10},{10,"
    "8},{13,10},{13,10},{12,9}};\n"
    "template<class F> bool Reject(F f){try{f();}catch(const std:"
    ":exception&){return true;}return false;}\n"
    "void Admission(){\n"
    " for(ByteCharge cap:{ByteCharge{39,31},ByteCharge{40,30}})\n"
    "  assert(Reject([&]{PairCoordinator c(cap,{2,2},slots,foreca"
    "sts,3s);}));\n"
    " for(ByteCharge bad:{ByteCharge{0,1},ByteCharge{1,0},ByteCha"
    "rge{14,1},ByteCharge{1,11}}){\n"
    "  auto f=forecasts;f.back()=bad;\n"
    "  assert(Reject([&]{PairCoordinator c({40,31},{2,2},slots,f,"
    "3s);}));\n"
    " }\n"
    " // Exact production reviewed fit fixture, including the sin"
    "gle 3% allowance.\n"
    " constexpr uint64_t g=1ULL<<30;\n"
    " const std::array<ByteCharge,3> real{{{13*g,10*g},{13*g,10*g"
    "},{12*g,9*g}}};\n"
    " for(ByteCharge f:{ByteCharge{13343149594ULL,10447295974ULL}"
    ",ByteCharge{12338645847ULL,9629641669ULL}}){\n"
    "  assert(PairCoordinator::Fits(f,real[0]));assert(PairCoordi"
    "nator::Fits(f,real[1]));\n"
    " }\n"
    " assert(!PairCoordinator::Fits({13343149594ULL,10447295974UL"
    "L},real[2]));\n"
    " assert(PairCoordinator::Fits({10438883260ULL,7995452196ULL}"
    ",real[2]));\n"
    " assert(PairCoordinator::Fits({12338645847ULL,9629641669ULL}"
    ",real[2]));\n"
    "}\n"
    "void Turnover(){\n"
    " Gate initial,slot2_joined,retired0,residual_release,success"
    "or_prepared;\n"
    " std::atomic<int> initial_count{0},alive{0},peak{0};\n"
    " std::mutex mutex; std::vector<std::pair<size_t,std::string>"
    "> trace;\n"
    " std::vector<PairEvent> transitions;\n"
    " auto record=[&](size_t j,const std::string& e){std::lock_gu"
    "ard<std::mutex> l(mutex);trace.emplace_back(j,e);};\n"
    ' PairEvents events("pair",6,true); PairCoordinator* ptr=null'
    "ptr;\n"
    " PairCoordinator c({40,31},{2,2},slots,forecasts,3s,[&](cons"
    "t PairEvent& e){\n"
    '  if(e.event=="admitted")events.BindSlot(e.job,ptr->Slot(e.j'
    "ob));\n"
    "  events.Coordinator(e);record(e.job,e.event);\n"
    "  {std::lock_guard<std::mutex> l(mutex);transitions.push_bac"
    "k(e);}\n"
    '  if(e.job==2 && e.event=="owner_joined")slot2_joined.Set();'
    "\n"
    '  if(e.job==0 && e.event=="retired")retired0.Set();\n'
    " });ptr=&c;\n"
    " std::thread delayed([&]{\n"
    "  retired0.Wait();\n"
    "  assert(c.AdmittedCount()==3); // C's joined empty slot doe"
    "s not fit FIFO head D.\n"
    "  assert(c.ReservedBytes().host==40);\n"
    "  residual_release.Set();\n"
    " });\n"
    " size_t created=0;\n"
    " auto factory=[&](std::function<void()> task){\n"
    "  size_t job=created++;\n"
    "  return std::thread([&,job,task=std::move(task)]() mutable "
    "{\n"
    "   task();\n"
    '   if(job==0){residual_release.Wait();record(job,"residual_d'
    'estroyed");}\n'
    "   task={}; // Real per-thread callback destruction before p"
    "arent join returns.\n"
    "  });\n"
    " };\n"
    " RunRollingOwners(c,[&](size_t j){\n"
    "  events.Owner(j);events.Identity(j,17+j,0);\n"
    "  struct Resource{std::atomic<int>& a;Resource(std::atomic<i"
    "nt>& a):a(a){++a;}~Resource(){--a;}} resource(alive);\n"
    "  int now=alive.load(),old=peak.load();while(now>old&&!peak."
    "compare_exchange_weak(old,now)){}\n"
    "  try{\n"
    "   c.BeginPreparation(j);\n"
    "   if(j==3){\n"
    "    assert(c.Slot(0)==c.Slot(3));\n"
    "    assert(Reject([&]{c.ReadyAndWait(0);}));assert(Reject([&"
    "]{c.SolverDone(0);}));\n"
    "    assert(Reject([&]{c.EndOutput(0);}));assert(Reject([&]{c"
    ".Retire(0);}));\n"
    '    c.Cancel(0,std::make_exception_ptr(std::runtime_error("s'
    'tale cancellation")));\n'
    "    successor_prepared.Set();\n"
    "   }\n"
    "   c.ReadyAndWait(j);\n"
    "   if(j<3 && ++initial_count==3)initial.Set();\n"
    "   if(j<3)initial.Wait();\n"
    "   if(j==0)slot2_joined.Wait();\n"
    "   if(j==1)successor_prepared.Wait();\n"
    "   c.SolverDone(j);c.BeginOutput(j);c.EndOutput(j);c.AwaitOu"
    "tputs(j);\n"
    "  }catch(...){c.CancelAndAwaitCleanup(j,std::current_excepti"
    "on(),[]{});throw;}\n"
    " },[&](size_t j,const char* e){record(j,e);events.Emit(j,e);"
    "},factory);\n"
    " delayed.join();events.Finished();assert(alive==0 && peak<=3"
    ");assert(c.ReservedBytes().host==0);\n"
    " auto index=[&](size_t j,const char* e){for(size_t i=0;i<tra"
    "ce.size();++i)if(trace[i]==std::make_pair(j,std::string(e)))"
    "return i;assert(false);return size_t(0);};\n"
    ' assert(index(2,"output_end")<index(0,"output_end"));\n'
    ' assert(index(2,"owner_joined")<index(0,"output_end"));\n'
    ' assert(index(0,"cleanup_complete")<index(0,"retired"));\n'
    ' assert(index(0,"retired")<index(0,"residual_destroyed"));\n'
    ' assert(index(0,"residual_destroyed")<index(0,"owner_joined"'
    "));\n"
    ' assert(index(0,"owner_joined")<index(3,"admitted"));\n'
    ' assert(index(3,"admitted")<index(3,"prep_start"));\n'
    ' assert(index(3,"prep_start")<index(1,"output_end"));\n'
    " int prepared=0,output=0;size_t next=0,release=0;\n"
    " std::sort(transitions.begin(),transitions.end(),[](const Pa"
    "irEvent& a,const PairEvent& b){return a.sequence<b.sequence;"
    "});\n"
    " for(const auto& event:transitions){\n"
    "  const auto j=event.job;const auto& e=event.event;\n"
    '  if(e=="prep_start"){assert(j==next++);assert(++prepared==1'
    ");}\n"
    '  if(e=="prep_end")assert(--prepared==0);\n'
    '  if(e=="output_start")assert(++output==1);\n'
    '  if(e=="output_end")assert(--output==0);\n'
    '  if(e=="ledger_release")++release;\n'
    " }\n"
    " assert(next==6&&release==1);\n"
    "}\n"
    "void Failure(int phase,int failjob){\n"
    " std::mutex mutex; std::vector<PairEvent> trace;\n"
    " PairCoordinator c({40,31},{2,2},slots,forecasts,3s,[&](cons"
    "t PairEvent& e){\n"
    "  {std::lock_guard<std::mutex> l(mutex);trace.push_back(e);}"
    "\n"
    '  if(phase==7 && int(e.job)==failjob && e.event=="admitted")'
    'throw std::runtime_error("injected");\n'
    '  if(phase==8 && int(e.job)==failjob && e.event=="retired")t'
    'hrow std::runtime_error("injected");\n'
    '  if(phase==9 && int(e.job)==failjob && e.event=="owner_join'
    'ed")throw std::runtime_error("injected");\n'
    " });\n"
    " std::atomic<int> alive{0};size_t starts=0;\n"
    " auto factory=[&](std::function<void()> task){\n"
    "  if(phase==4 && int(starts++)==failjob)throw std::runtime_e"
    'rror("injected");\n'
    "  if(phase==10 && int(starts++)==failjob)return std::thread{"
    "};\n"
    "  return std::thread(std::move(task));\n"
    " };\n"
    " bool failed=false;\n"
    " try{RunRollingOwners(c,[&](size_t j){\n"
    '  if(phase==0 && int(j)==failjob)throw std::runtime_error("i'
    'njected"); // Controller ctor.\n'
    "  struct Resource{std::atomic<int>& a;Resource(std::atomic<i"
    "nt>& a):a(a){++a;}~Resource(){--a;}} resource(alive);\n"
    "  try{\n"
    "   c.BeginPreparation(j);\n"
    '   if(phase==1 && int(j)==failjob)throw std::runtime_error("'
    'injected");\n'
    "   c.ReadyAndWait(j);\n"
    '   if(phase==2 && int(j)==failjob)throw std::runtime_error("'
    'injected");\n'
    "   c.SolverDone(j);c.BeginOutput(j);\n"
    '   if(phase==3 && int(j)==failjob)throw std::runtime_error("'
    'injected");\n'
    "   c.EndOutput(j);c.AwaitOutputs(j);\n"
    "  }catch(...){c.CancelAndAwaitCleanup(j,std::current_excepti"
    "on(),[]{});throw;}\n"
    " },[&](size_t j,const char*){if(phase==6&&int(j)==failjob)th"
    'row std::runtime_error("injected");},factory,\n'
    " [&](size_t j){if(phase==5&&int(j)==failjob)throw std::runti"
    'me_error("injected");});}\n'
    " catch(const std::runtime_error& e){failed=true;assert(std::"
    'string(e.what())==(phase==10?"thread factory returned no own'
    'er":"injected"));}\n'
    " assert(failed && alive==0);assert(c.ReservedBytes().host==("
    "phase==5?40:0));\n"
    " size_t retired=0,joined=0,released=0;uint64_t cancel=UINT64"
    "_MAX;\n"
    ' for(const auto& e:trace){if(e.event=="cancelled")cancel=std'
    "::min(cancel,e.sequence);}\n"
    " for(const auto& e:trace){\n"
    '  retired+=e.event=="retired";joined+=e.event=="owner_joined'
    '";released+=e.event=="ledger_release";\n'
    '  if(e.event=="admitted")assert(e.sequence<cancel);\n'
    " }\n"
    " assert(retired==c.AdmittedCount()&&joined==retired&&release"
    "d==(phase==5?0:1));\n"
    "}\n"
    "void ActiveFailure(){\n"
    " Gate started,cancelled;std::atomic<int> count{0},active{0},"
    "destroyed{0};\n"
    " PairCoordinator c({40,31},{2,2},slots,forecasts,3s,[&](cons"
    "t PairEvent& e){\n"
    '  if(e.event=="cancelled")cancelled.Set();\n'
    " });\n"
    " bool failed=false;\n"
    " try{RunRollingOwners(c,[&](size_t j){\n"
    "  struct Resource{std::atomic<int>& active;std::atomic<int>&"
    " destroyed;\n"
    "   ~Resource(){assert(active==0);++destroyed;}\n"
    "  } resource{active,destroyed};\n"
    "  c.BeginPreparation(j);c.ReadyAndWait(j);++active;\n"
    "  if(++count==3)started.Set();\n"
    "  started.Wait();\n"
    "  try{\n"
    '   if(j==0)throw std::runtime_error("active failure");\n'
    '   cancelled.Wait();throw std::runtime_error("peer cancellat'
    'ion");\n'
    "  }catch(...){c.CancelAndAwaitCleanup(j,std::current_excepti"
    "on(),[&]{--active;});throw;}\n"
    " },[](size_t,const char*){});}\n"
    " catch(const std::runtime_error& e){failed=true;assert(std::"
    'string(e.what())=="active failure");}\n'
    " assert(failed&&destroyed==3&&c.AdmittedCount()==3&&c.Reserv"
    "edBytes().host==0);\n"
    "}\n"
    "int main(int argc,char**){\n"
    " if(argc>1){\n"
    "  std::set_terminate([]{std::_Exit(86);});\n"
    "  PairCoordinator c({40,31},{2,2},slots,forecasts,3s);c.Admi"
    "tNext();\n"
    "  c.CancelAndAwaitCleanup(0,std::make_exception_ptr(std::run"
    'time_error("uncertain")),[]{throw std::runtime_error("drain '
    'unproven");});\n'
    "  assert(false);\n"
    " }\n"
    " Admission();Turnover();ActiveFailure();for(int phase=0;phas"
    "e<=10;++phase)for(int job:{0,2,3,5})Failure(phase,job);std::"
    'cout<<"ROLLING_THREE_CPU_OK\\n";\n'
    "}\n"
)


class RollingThreeTest(unittest.TestCase):
    def compile_run(self, source, name, args=(), expected_exit=0):
        temp = Path(tempfile.mkdtemp(prefix=f"rolling-three-{name}-"))
        program, binary = temp / "test.cc", temp / "test"
        program.write_text(source)
        result = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-O1",
                "-pthread",
                "-I",
                str(ROOT / "src"),
                str(program),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        (temp / "compile.log").write_text(result.stdout + result.stderr)
        print(f"ROLLING_THREE_ARTIFACT {temp} compile_exit={result.returncode}")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [str(binary), *args], capture_output=True, text=True, timeout=30
        )
        (temp / "run.log").write_text(result.stdout + result.stderr)
        self.assertEqual(
            result.returncode, expected_exit, result.stdout + result.stderr
        )
        return result.stdout

    def test_uncertain_drain_is_failstop(self):
        self.compile_run(
            CPP, "uncertain-drain", args=("uncertain",), expected_exit=86
        )

    def test_actual_native_preparation_uses_generation_not_slot(self):
        native = (ROOT / "src/colmap/mvs/patch_match.cc").read_text()
        prepare = native[
            native.index(
                "std::unique_ptr<PatchMatchController::OwnedPreparedJob>"
            ) : native.index("void PatchMatchController::RunOwnedPairJob(")
        ]
        source = (
            (
                "\n"
                "#include <cassert>\n"
                "#include <memory>\n"
                "#include <vector>\n"
                "#include <string>\n"
                "#include <cstddef>\n"
                "#define THROW_CHECK(x) assert(x)\n"
                "#define THROW_CHECK_EQ(x,y) assert((x)==(y))\n"
                "#define THROW_CHECK_GE(x,y) assert((x)>=(y))\n"
                "#define THROW_CHECK_GT(x,y) assert((x)>(y))\n"
                "#define THROW_CHECK_LE(x,y) assert((x)<=(y))\n"
                "#define THROW_CHECK_NOTNULL(x) assert((x)!=nullptr)\n"
                "enum class PatchMatchExecution { kOwnedCuda };\n"
                "template<class T> std::vector<T> CSVToVector(const std::stri"
                "ng&){return {0};}\n"
                "class PatchMatch { public: void Check() {} };\n"
                "class ThreadPool { public: explicit ThreadPool(int) {} };\n"
                "class PatchMatchController {\n"
                " public:\n"
                ' struct Options { std::string gpu_index="0"; int num_threads'
                "=1; } options_;\n"
                " struct OwnedPreparedJob { std::unique_ptr<PatchMatch> patch"
                "_match; };\n"
                " struct OwnedPairHooks {};\n"
                " PatchMatchExecution execution_=PatchMatchExecution::kOwnedC"
                "uda;\n"
                " std::unique_ptr<int> workspace_;\n"
                " std::unique_ptr<ThreadPool> io_thread_pool_,thread_pool_;\n"
                " std::vector<int> problems_,depth_ranges_,gpu_indices_;\n"
                " size_t seen=100;\n"
                " std::unique_ptr<OwnedPreparedJob> PrepareOwnedPairJob(size_"
                "t,const OwnedPairHooks&,int=0);\n"
                " void ReadWorkspace(){workspace_=std::make_unique<int>(1);}"
                "\n"
                " void ReadProblems(){problems_={0};}\n"
                " void ReadGpuIndices(){gpu_indices_={0};}\n"
                " void ProcessProblemInputs(const Options&,int,OwnedPreparedJ"
                "ob* p,size_t j,const OwnedPairHooks&){\n"
                "  seen=j;p->patch_match=std::make_unique<PatchMatch>();\n"
                " }\n"
                "};\n"
            )
            + prepare
            + r"""
int main(){for(size_t j=0;j<6;++j){
 PatchMatchController controller;
 auto prepared=controller.PrepareOwnedPairJob(j,{});
 assert(prepared&&prepared->patch_match&&controller.seen==j);
 assert(!controller.workspace_&&!controller.io_thread_pool_);
 assert(controller.problems_.empty()&&controller.gpu_indices_.empty());
}}
"""
        )
        self.compile_run(source, "native-preparation")

    def test_actual_driver_feed_and_forecast_reader(self):
        driver = (ROOT / "src/colmap/exe/owned_pair.cc").read_text()
        settings = driver[driver.index("struct PairSettings {") :]
        settings = settings[: settings.index("\n};") + 3]
        feed = driver[
            driver.index("// Literal positive integer pairs:") : driver.index(
                "template <class RunJob, class PrepareJob ="
            )
        ]
        source = (
            (
                "\n"
                '#include "colmap/mvs/owned_pair_events.h"\n'
                '#include "colmap/mvs/owned_pair_run.h"\n'
                '#include "colmap/mvs/owned_pair_validation.h"\n'
                "#include <filesystem>\n"
                "#include <fstream>\n"
                "#include <memory>\n"
                "#include <cassert>\n"
                "#include <atomic>\n"
                "namespace pair = colmap::mvs::owned;\n"
                "namespace fs = std::filesystem;\n"
                "namespace mvs {\n"
                "struct Options { int num_threads=1,reference=17; };\n"
                "struct Problem {};\n"
                "struct PatchMatchController {\n"
                " struct OwnedPairHooks {\n"
                "  int owned_tile_threads=32;\n"
                "  std::function<void(size_t,const Options&,const Problem&)> "
                "check_inputs,metadata;\n"
                "  std::function<void(size_t,int,int)> sweep;\n"
                "  std::function<void(size_t,const char*)> output_phase;\n"
                " };\n"
                "};\n"
                "}\n"
                "struct JobOptions { fs::path workspace;\n"
                " std::unique_ptr<mvs::Options> patch_match_stereo=std::make_"
                "unique<mvs::Options>(); };\n"
                "template<class T> std::string Value(const T& value){std::ost"
                "ringstream s;s<<value;return s.str();}\n"
                "void CheckInputs(const mvs::Options&,const mvs::Problem&,pai"
                "r::ByteCharge,const fs::path&) {}\n"
                "void Metadata(pair::PairEvents& e,size_t j,const mvs::Option"
                "s& o,const mvs::Problem&,int tile){e.Identity(j,o.reference,"
                '0);e.Emit(j,"effective_metadata_begin","owned_tile_threads="'
                "+Value(tile));}\n"
            )
            + settings
            + feed
            + (
                "\n"
                "template<class F> bool Reject(F f){try{f();}catch(const std:"
                ":exception&){return true;}return false;}\n"
                "int main(){\n"
                ' auto path=fs::temp_directory_path()/("rolling-forecast-cpu-'
                '"+std::to_string(getpid()));\n'
                " auto put=[&](const std::string& text){std::ofstream out(pat"
                "h);out<<text;};\n"
                ' put("13 10\\n13 10\\n10 8\\n13 10\\n13 10\\n12 9\\n");\n'
                " auto forecasts=ReadRollingForecasts(path,6);assert(forecast"
                "s.size()==6);\n"
                ' for(auto text:{"", "0 1\\n", "1 0\\n", "-1 2\\n", "1 2 3\\n'
                '", "1\\n", "18446744073709551616 1\\n", "1 2\\n\\n"}){\n'
                "  put(text);assert(Reject([&]{ReadRollingForecasts(path,1);}"
                "));\n"
                " }\n"
                ' put("1 2\\n");assert(Reject([&]{ReadRollingForecasts(path,6'
                ");}));fs::remove(path);\n"
                " assert(Reject([&]{ReadRollingForecasts(path,1);}));\n"
                " std::vector<std::unique_ptr<JobOptions>> jobs;\n"
                " for(size_t j=0;j<6;++j){jobs.push_back(std::make_unique<Job"
                "Options>());jobs.back()->patch_match_stereo->reference=17+j;"
                "}\n"
                ' PairSettings settings{"pair",{40,31},{2,2},{{{13,10},{13,10'
                "},{12,9}}},3,1,false,3,16,true};\n"
                " std::atomic<int> calls{0};\n"
                " auto run=[&](JobOptions& job,pair::PairCoordinator& c,size_"
                "t j,const auto& hooks){\n"
                "  ++calls;assert(j==size_t(job.patch_match_stereo->reference"
                "-17));assert(hooks.owned_tile_threads==16);\n"
                "  try{\n"
                "   c.BeginPreparation(j);hooks.check_inputs(j,*job.patch_mat"
                "ch_stereo,mvs::Problem{});\n"
                "   hooks.metadata(j,*job.patch_match_stereo,mvs::Problem{});"
                "c.ReadyAndWait(j);\n"
                "   hooks.sweep(j,0,0);c.SolverDone(j);c.BeginOutput(j);\n"
                '   for(auto phase:{"depth_begin","depth_end","normal_begin",'
                '"normal_end"})hooks.output_phase(j,phase);\n'
                "   c.EndOutput(j);c.AwaitOutputs(j);\n"
                "  }catch(...){c.CancelAndAwaitCleanup(j,std::current_excepti"
                "on(),[]{});throw;}\n"
                " };\n"
                " auto impossible=forecasts;impossible.back()={14,8};\n"
                " assert(Reject([&]{RunRollingFeed(jobs,settings,impossible,r"
                "un);}));assert(calls==0);\n"
                " RunRollingFeed(jobs,settings,forecasts,run);assert(calls==6"
                ");\n"
                "}\n"
            )
        )
        output = self.compile_run(source, "driver")
        records = [
            dict(field.split("=", 1) for field in line.split()[1:])
            for line in output.splitlines()
            if line.startswith("OWNED_PAIR_EVENT ")
        ]
        successes = [r for r in records if r["event"] == "batch_finished"]
        self.assertEqual(len(successes), 1)
        own = [r for r in records if r["batch"] == successes[0]["batch"]]
        self.assertEqual(sum(r["event"] == "admitted" for r in own), 6)
        self.assertEqual(sum(r["event"] == "normal_end" for r in own), 6)
        self.assertEqual(sum(r["event"] == "owner_joined" for r in own), 6)
        self.assertEqual(sum(r["event"] == "owners_joined" for r in own), 1)
        self.assertEqual(sum(r["event"] == "ledger_release" for r in own), 1)
        validated = next(r for r in own if r["event"] == "validated")
        self.assertEqual(validated["owned_tile_threads"], "16")
        self.assertEqual(validated["job_count"], "6")
        self.assertEqual(validated["hc"], "12")

    def test_real_threads_fit_postjoin_generations_failures(self):
        output = self.compile_run(CPP, "threads")
        self.assertIn("ROLLING_THREE_CPU_OK", output)
        records = [
            dict(field.split("=", 1) for field in line.split()[1:])
            for line in output.splitlines()
            if line.startswith("OWNED_PAIR_EVENT ")
        ]
        self.assertTrue(records)
        self.assertTrue(
            all(
                r["schema"] == "2" and r["feed_mode"] == "rolling"
                for r in records
            )
        )
        for j in range(6):
            own = [r for r in records if r["job"] == f"J{j}"]
            self.assertTrue(own)
            self.assertEqual({r["queue_index"] for r in own}, {str(j)})
            self.assertEqual({r["generation"] for r in own}, {str(j + 1)})
            self.assertEqual(len({r["slot"] for r in own}), 1)
            ready = next(r for r in own if r["event"] == "ready")
            self.assertEqual(ready["reference"], str(17 + j))
            self.assertNotEqual(ready["owner_tid"], "0")
        batch = [r for r in records if r["job"] == "batch"]
        self.assertTrue(
            all(r["slot"] == "-1" and r["generation"] == "0" for r in batch)
        )
        self.assertEqual(batch[-1]["event"], "batch_finished")


if __name__ == "__main__":
    unittest.main(verbosity=2)
