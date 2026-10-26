# SPDX-License-Identifier: BSD-3-Clause
"""Bounded real coordinator/runner/events CPU proof; no CUDA/build claim."""

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
    "#include <condition_variable>\n"
    "#include <iostream>\n"
    "#include <map>\n"
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
    "const std::array<ByteCharge,3> charges{{{13,10},{13,10},{10,"
    "8}}};\n"
    "template<class F> bool Reject(F f){try{f();}catch(const std:"
    ":exception&){return true;}return false;}\n"
    "void Admission(){\n"
    " PairCoordinator exact(PairDispatch::kPair,{38,30},{2,2},cha"
    "rges,3s,{},2);\n"
    " assert(exact.JobCount()==3 && exact.ReservedBytes().host==3"
    "8);\n"
    " assert(Reject([&]{exact.BeginPreparation(3);}));\n"
    " assert(Reject([&]{exact.ReadyAndWait(0);}));\n"
    " assert(Reject([&]{exact.SolverDone(2);}));\n"
    " assert(Reject([&]{exact.BeginOutput(1);}));\n"
    " assert(Reject([&]{exact.Retire(2);}));\n"
    " assert(Reject([&]{exact.Finish();}));\n"
    " for(auto cap: {ByteCharge{37,30}, ByteCharge{38,29}})\n"
    "  assert(Reject([&]{PairCoordinator c(PairDispatch::kPair,ca"
    "p,{2,2},charges,3s,{},2);}));\n"
    " for(size_t j=0;j<3;++j){\n"
    "  auto bad=charges;bad[j].host=0;\n"
    "  assert(Reject([&]{PairCoordinator c(PairDispatch::kPair,{3"
    "8,30},{2,2},bad,3s,{},2);}));\n"
    "  bad=charges;bad[j].device=UINT64_MAX;\n"
    "  assert(Reject([&]{PairCoordinator c(PairDispatch::kPair,{U"
    "INT64_MAX,UINT64_MAX},{2,2},bad,3s,{},2);}));\n"
    " }\n"
    " for(size_t permits:{0,1,4}) assert(Reject([&]{PairCoordinat"
    "or c(PairDispatch::kPair,{38,30},{2,2},charges,3s,{},permits"
    ");}));\n"
    " assert(Reject([&]{PairCoordinator c(PairDispatch::kSerial,{"
    "38,30},{2,2},charges,3s,{},2);}));\n"
    "}\n"
    "void Run(size_t permits,int fail_job=-1,int phase=-1,bool sp"
    "awn_fail=false,bool unknown=false,bool early_done=false){\n"
    ' PairEvents events("pair",3); Gate c_ready,c_started,a_done;'
    "\n"
    " std::mutex mutex; std::vector<std::pair<size_t,std::string>"
    "> trace;\n"
    " std::map<std::pair<size_t,std::string>,uint64_t> sequence;\n"
    " std::atomic<int> alive{0};\n"
    " auto record=[&](size_t job,const std::string& event){\n"
    "  std::lock_guard<std::mutex> l(mutex);trace.emplace_back(jo"
    "b,event);\n"
    " };\n"
    " PairCoordinator c(PairDispatch::kPair,{38,30},{2,2},charges"
    ",3s,[&](const PairEvent& e){\n"
    "  record(e.job,e.event);\n"
    "  {std::lock_guard<std::mutex> l(mutex);sequence[{e.job,e.ev"
    "ent}]=e.sequence;}\n"
    "  events.Coordinator(e);\n"
    '  if(e.job==0 && e.event=="solver_end")a_done.Set();\n'
    '  if(early_done && e.job==2 && e.event=="prep_start")a_done.'
    "Wait();\n"
    '  if(e.job==2 && e.event=="ready")c_ready.Set();\n'
    '  if(e.job==2 && e.event=="solver_start")c_started.Set();\n'
    " },permits);\n"
    " size_t starts=0;\n"
    " auto factory=[&](std::function<void()> task){\n"
    '  if(spawn_fail && starts++==2)throw std::runtime_error("thi'
    'rd thread failed");\n'
    "  return std::thread(std::move(task));\n"
    " };\n"
    " bool failed=false;\n"
    " try {\n"
    " RunPairOwners(c,[&](size_t j){\n"
    "  events.Owner(j); events.Identity(j,17+j,0);\n"
    "  struct Resource{std::atomic<int>& alive;Resource(std::atom"
    "ic<int>& a):alive(a){++alive;}~Resource(){--alive;}} resourc"
    "e(alive);\n"
    "  try{\n"
    "   c.BeginPreparation(j);\n"
    "   if(int(j)==fail_job && phase==0)throw std::runtime_error("
    '"injected");\n'
    "   c.ReadyAndWait(j);\n"
    "   if(phase==3 && j<2){c_ready.Wait();throw std::runtime_err"
    'or("injected");}\n'
    "   if(fail_job<0 && !spawn_fail && !unknown && !early_done &"
    "& j<2){\n"
    "    c_ready.Wait(); if(permits==3)c_started.Wait();\n"
    "   }\n"
    "   if(int(j)==fail_job && phase==1)throw std::runtime_error("
    '"injected");\n'
    "   c.SolverDone(j);c.BeginOutput(j);\n"
    "   if(int(j)==fail_job && phase==2)throw std::runtime_error("
    '"injected");\n'
    "   c.EndOutput(j);c.AwaitOutputs(j);\n"
    "  }catch(...){c.CancelAndAwaitCleanup(j,std::current_excepti"
    "on(),[]{});throw;}\n"
    " },[&](size_t j,const char* event){record(j,event);events.Em"
    "it(j,event);},factory,[&](size_t j){\n"
    '  if(unknown && j==2)throw std::runtime_error("cleanup unpro'
    'ven");\n'
    " });\n"
    " }catch(const std::runtime_error& error){\n"
    "  failed=true;\n"
    '  assert(std::string(error.what())==(spawn_fail?"third threa'
    'd failed":unknown?"cleanup unproven":"injected"));\n'
    " }\n"
    " assert(failed==(fail_job>=0 || spawn_fail || unknown)); ass"
    "ert(alive==0);\n"
    " assert(c.ReservedBytes().host==(unknown?38:0));\n"
    " auto index=[&](size_t j,const char* e){for(size_t i=0;i<tra"
    "ce.size();++i)if(trace[i]==std::make_pair(j,std::string(e)))"
    "return i;return trace.size();};\n"
    " auto seq=[&](size_t j,const char* e){return sequence.at({j,"
    "e});};\n"
    " const size_t batch=PairEvents::kBatch;\n"
    " size_t retired=0,released=0;\n"
    ' for(const auto& e:trace){retired+=e.second=="retired";relea'
    'sed+=e.second=="ledger_release";}\n'
    " assert(retired==3 && released==(unknown?0:1));\n"
    ' if(!unknown)assert(index(batch,"owners_joined")<index(batch'
    ',"ledger_release"));\n'
    " if(!failed){\n"
    "  events.Finished();\n"
    "  for(size_t j=0;j<3;++j){\n"
    '   assert(seq(j,"output_end")<seq(0,"retired"));\n'
    '   assert(seq(j,"output_end")<seq(1,"retired"));\n'
    '   assert(seq(j,"output_end")<seq(2,"retired"));\n'
    '   assert(index(j,"cleanup_complete")<index(j,"retired"));\n'
    "  }\n"
    '  assert(seq(0,"prep_end")<seq(1,"prep_start"));\n'
    '  assert(seq(1,"prep_end")<seq(2,"prep_start"));\n'
    '  assert(seq(0,"output_end")<seq(1,"output_start"));\n'
    '  assert(seq(1,"output_end")<seq(2,"output_start"));\n'
    '  if(early_done)assert(seq(0,"solver_end")<seq(2,"ready"));\n'
    '  if(permits==2)assert(std::min(seq(0,"solver_end"),seq(1,"s'
    'olver_end"))<seq(2,"solver_release"));\n'
    '  else if(!early_done)assert(seq(2,"solver_start")<std::min('
    'seq(0,"solver_end"),seq(1,"solver_end")));\n'
    "  assert(Reject([&]{c.SolverDone(2);})); // stale completion"
    "\n"
    "  assert(Reject([&]{c.Retire(2);}));\n"
    "  c.Finish(); // no duplicate release\n"
    " } else if(phase==3) {\n"
    '  assert(index(2,"solver_release")==trace.size());\n'
    '  assert(index(2,"solver_start")==trace.size());\n'
    " } else if(fail_job<2 && phase==1) {\n"
    "  // No successful A/B completion may be fabricated from fai"
    "lure.\n"
    '  assert(index(size_t(fail_job),"solver_end")==trace.size())'
    ";\n"
    " }\n"
    "}\n"
    "int main(){Admission();for(size_t permits:{2,3}){\n"
    " Run(permits);Run(permits,-1,-1,false,false,true);\n"
    " for(int j=0;j<3;++j)for(int phase=0;phase<3;++phase)Run(per"
    "mits,j,phase);\n"
    " Run(permits,-1,-1,true);Run(permits,-1,-1,false,true);\n"
    '} Run(2,0,3);std::cout<<"STATIC_THREE_CPU_OK\\n";}\n'
)


class StaticThreeTest(unittest.TestCase):
    def test_actual_driver_epoch(self):
        driver = (ROOT / "src/colmap/exe/owned_pair.cc").read_text()
        settings = driver[driver.index("struct PairSettings {") :]
        settings = settings[: settings.index("\n};") + 3]
        epoch = driver[
            driver.index(
                "template <class RunJob>\nvoid RunPairEpoch("
            ) : driver.index("using PreparedHostJob =")
        ]
        loop = driver[
            driver.index(
                "  if (static_three_permits) {\n    auto execute"
            ) : driver.index("  // The full additional transient charge")
        ]
        source = (
            (
                "\n"
                '#include "colmap/mvs/owned_pair_events.h"\n'
                '#include "colmap/mvs/owned_pair_run.h"\n'
                '#include "colmap/mvs/owned_pair_validation.h"\n'
                "#include <filesystem>\n"
                "#include <memory>\n"
                "#include <cassert>\n"
                "namespace pair = colmap::mvs::owned;\n"
                "namespace fs = std::filesystem;\n"
                "namespace mvs {\n"
                "struct Options { int num_threads = 1; int reference = 17; };"
                "\n"
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
                "struct Globals { int seed=0,level=0,severity=0; bool color=t"
                'rue; std::string target="stderr",path; };\n'
                "struct JobOptions { Globals globals; fs::path workspace;\n"
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
            + epoch
            + r"""
using PreparedHostJob = int;
template<class RunJob>
int RunStatic(std::vector<std::unique_ptr<JobOptions>>& jobs,
              const PairSettings& settings, RunJob run_job) {
 const auto static_three_permits = settings.static_three_solver_permits;
"""
            + loop
            + (
                "\n"
                " return EXIT_FAILURE;\n"
                "}\n"
                "int main(int argc,char** argv){\n"
                " assert(argc==2); const int fail=std::stoi(argv[1]);\n"
                " for(int tile:{8,16,32}) for(size_t permits:{2,3}){\n"
                "  std::vector<std::unique_ptr<JobOptions>> jobs;\n"
                "  for(size_t j=0;j<6;++j){\n"
                "   jobs.push_back(std::make_unique<JobOptions>());\n"
                "   jobs.back()->workspace=std::to_string(j);\n"
                "   jobs.back()->patch_match_stereo->reference=17+j;\n"
                "  }\n"
                "  std::atomic<int> alive{0}; std::atomic<size_t> calls{0};\n"
                '  PairSettings settings{"pair",{38,30},{2,2},{{{13,10},{13,1'
                "0},{10,8}}},3,1,false,permits,tile};\n"
                "  auto run=[&](JobOptions& job,pair::PairCoordinator& coord,"
                "size_t j,const auto& hooks){\n"
                "   const int index=std::stoi(job.workspace.string());\n"
                "   assert(j==size_t(index)%3); assert(hooks.owned_tile_threa"
                "ds==tile); ++calls;\n"
                "   struct Resource { std::atomic<int>& alive;\n"
                "    explicit Resource(std::atomic<int>& a):alive(a){++alive;"
                "}\n"
                "    ~Resource(){--alive;}\n"
                "   } resource(alive);\n"
                "   coord.BeginPreparation(j);hooks.metadata(j,*job.patch_mat"
                "ch_stereo,mvs::Problem{});\n"
                '   if(index==fail)throw std::runtime_error("injected epoch f'
                'ailure");\n'
                "   coord.ReadyAndWait(j);coord.SolverDone(j);coord.BeginOutp"
                "ut(j);\n"
                '   for(auto phase:{"depth_begin","depth_end","normal_begin",'
                '"normal_end"})hooks.output_phase(j,phase);\n'
                "   coord.EndOutput(j);coord.AwaitOutputs(j);\n"
                "  };\n"
                "  bool failed=false;\n"
                "  try {assert(RunStatic(jobs,settings,run)==EXIT_SUCCESS);}"
                "\n"
                "  catch(const std::runtime_error& e){failed=true;assert(std:"
                ':string(e.what())=="injected epoch failure");}\n'
                "  assert(failed==(fail>=0));assert(alive==0);\n"
                "  if(fail<0)assert(calls==6);\n"
                "  else if(fail<3)assert(calls==3); // No successor owner may"
                " be launched.\n"
                " }\n"
                "}\n"
            )
        )
        temp = Path(tempfile.mkdtemp(prefix="static-three-epoch-"))
        program, binary = temp / "test.cc", temp / "test"
        program.write_text(source)
        compiled = subprocess.run(
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
        (temp / "compile.log").write_text(compiled.stdout + compiled.stderr)
        print(
            f"STATIC_THREE_EPOCH_ARTIFACT {temp} "
            f"compile_exit={compiled.returncode}"
        )
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        for failure in (-1, 0, 2, 3, 5):
            result = subprocess.run(
                [str(binary), str(failure)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            (temp / f"run-{failure}.log").write_text(
                result.stdout + result.stderr
            )
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr
            )
            records = [
                dict(field.split("=", 1) for field in line.split()[1:])
                for line in result.stdout.splitlines()
                if line.startswith("OWNED_PAIR_EVENT ")
            ]
            validated = [r for r in records if r["event"] == "validated"]
            expected = (
                ["2", "2", "3", "3"]
                if failure < 0 or failure >= 3
                else ["2", "3"]
            )
            self.assertEqual(
                [r["static_three_solver_permits"] for r in validated],
                expected * 3,
            )
            self.assertEqual(
                [r["owned_tile_threads"] for r in validated],
                [str(t) for t in (8, 16, 32) for _ in expected],
            )
            for record in validated:
                self.assertEqual(record["job_count"], "3")
                self.assertEqual(
                    (record["hc"], record["dc"], record["io_c"]),
                    ("10", "8", "1"),
                )
            batches = list(dict.fromkeys(r["batch"] for r in records))
            for ordinal, batch in enumerate(batches):
                epoch = [r for r in records if r["batch"] == batch]
                events = [r["event"] for r in epoch]
                self.assertEqual(sum(e == "retired" for e in events), 3)
                self.assertEqual(sum(e == "owners_joined" for e in events), 1)
                self.assertEqual(sum(e == "ledger_release" for e in events), 1)
                release = next(
                    r for r in epoch if r["event"] == "ledger_release"
                )
                self.assertEqual(
                    (release["reserved_host"], release["reserved_device"]),
                    ("0", "0"),
                )
                self.assertLess(
                    max(i for i, e in enumerate(events) if e == "retired"),
                    events.index("owners_joined"),
                )
                self.assertLess(
                    events.index("owners_joined"),
                    events.index("ledger_release"),
                )
                for job in ("A", "B", "C"):
                    own = [r["event"] for r in epoch if r["job"] == job]
                    self.assertLess(
                        own.index("cleanup_complete"), own.index("retired")
                    )
                if "batch_finished" in events:
                    first = 17 + 3 * (ordinal % 2) if len(batches) == 12 else 17
                    self.assertEqual(
                        {
                            r["job"]: int(r["reference"])
                            for r in epoch
                            if r["event"] == "ready"
                        },
                        {"A": first, "B": first + 1, "C": first + 2},
                    )
                    self.assertLess(
                        events.index("ledger_release"),
                        events.index("batch_finished"),
                    )
                    self.assertEqual(events[-1], "batch_finished")
                    self.assertEqual(sum(e == "normal_end" for e in events), 3)
                    # Coordinator timestamps/sequences are authoritative:
                    # observer writes from different owners can interleave.
                    outputs = [
                        int(r["coord_seq"])
                        for r in epoch
                        if r["event"] == "output_end"
                    ]
                    retirements = [
                        int(r["coord_seq"])
                        for r in epoch
                        if r["event"] == "retired"
                    ]
                    self.assertLess(max(outputs), min(retirements))
                else:
                    self.assertEqual(events[-1], "batch_failed")
            # Stream order proves closure/Finish before the next epoch's admit.
            for previous, following in zip(batches, batches[1:], strict=False):
                prev = [
                    i for i, r in enumerate(records) if r["batch"] == previous
                ]
                nxt = [
                    i for i, r in enumerate(records) if r["batch"] == following
                ]
                self.assertLess(max(prev), min(nxt))

    def test_bounded_coordinator_runner_and_events(self):
        temp = Path(tempfile.mkdtemp(prefix="static-three-cpu-"))
        source, binary = temp / "test.cc", temp / "test"
        source.write_text(CPP)
        compile_result = subprocess.run(
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
                str(source),
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        (temp / "compile.log").write_text(
            compile_result.stdout + compile_result.stderr
        )
        print(
            f"STATIC_THREE_ARTIFACT {temp} "
            f"compile_exit={compile_result.returncode}"
        )
        self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
        result = subprocess.run(
            [str(binary)], capture_output=True, text=True, timeout=30
        )
        (temp / "run.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("STATIC_THREE_CPU_OK", result.stdout)
        records = [
            dict(field.split("=", 1) for field in line.split()[1:])
            for line in result.stdout.splitlines()
            if line.startswith("OWNED_PAIR_EVENT ")
        ]
        successes = [r for r in records if r["event"] == "batch_finished"]
        self.assertEqual(len(successes), 4)
        for success in successes:
            batch = [r for r in records if r["batch"] == success["batch"]]
            self.assertEqual(
                {r["job"] for r in batch if r["event"] == "ready"},
                {"A", "B", "C"},
            )
            self.assertEqual(sum(r["event"] == "all_ready" for r in batch), 1)
            self.assertFalse(any(r["event"] == "both_ready" for r in batch))


if __name__ == "__main__":
    unittest.main(verbosity=2)
