# SPDX-License-Identifier: BSD-3-Clause
"""Bounded CPU compilation of the actual driver parser/admission seam.

Job loading is injected; this does not compile the product or exercise CUDA.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class EarlyGeometryCliTest(unittest.TestCase):
    def test_actual_even_list_and_job_guards(self):
        driver = (ROOT / 'src/colmap/exe/owned_pair.cc').read_text()
        settings = driver[driver.index('struct PairSettings {'):]
        settings = settings[:settings.index('\n};') + 3]
        loader = driver[driver.index('std::vector<fs::path> JobProjectPaths('):
                        driver.index('// RunJob owns the controller')]
        source = r'''
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/owned_host_lookahead.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
namespace fs = std::filesystem;
namespace pair = colmap::mvs::owned;
''' + settings + r'''
struct Options { bool geom_consistency = true; std::string gpu_index = "0"; int num_threads = 1; };
struct Globals { bool operator==(const Globals&) const { return true; } };
struct Spec { Options options; fs::path workspace; std::string reference; std::array<fs::path,2> dirs; };
std::vector<Spec> specs;
struct JobOptions {
  Globals globals;
  std::unique_ptr<Options> patch_match_stereo = std::make_unique<Options>();
  fs::path workspace; std::string reference_name; std::array<fs::path,2> output_dirs;
  bool Read(const std::string& path, bool) {
    const auto& s = specs.at(std::stoul(fs::path(path).stem().string()));
    *patch_match_stereo = s.options; workspace = s.workspace;
    reference_name = s.reference; output_dirs = s.dirs; return true;
  }
};
// Native INI/decode is not part of this bounded CPU build. All list loading,
// cross-job phase/device/alias and per-pair IO guards below are real source.
void ValidateJob(JobOptions&, pair::ByteCharge) {}
''' + loader + r'''
template<class F> bool Reject(F f) {
  try { f(); } catch (const std::exception&) { return true; } return false;
}
int main(int argc, char** argv) {
  assert(argc == 2); const fs::path root(argv[1]); fs::create_directories(root);
  PairSettings settings{"pair", {100,100}, {1,1}, {{{1,1},{1,1}}}, 2, 1, true};
  const auto list = root / "jobs.txt";
  const std::map<std::string,std::string> args{{"job_project_list_path",list.string()}};
  auto write = [&](size_t count) { std::ofstream out(list); for(size_t j=0;j<count;++j) out << j << ".ini\n"; };
  for(size_t j=0;j<4;++j) {
    const auto workspace = root / std::to_string(j); fs::create_directories(workspace);
    specs.push_back({{},workspace,std::to_string(j),{workspace/"depth",workspace/"normal"}});
  }
  auto load = [&] { return LoadJobs(JobProjectPaths(args),settings); };
  write(4); assert(load().size()==4);
  write(3); assert(Reject(load)); write(0); assert(Reject(load)); write(4);
  specs[3].options.geom_consistency=false; assert(Reject(load)); specs[3].options.geom_consistency=true;
  specs[3].options.gpu_index="1"; assert(Reject(load)); specs[3].options.gpu_index="0";
  const auto saved=specs[3]; specs[3].workspace=specs[0].workspace; assert(Reject(load)); specs[3]=saved;
  specs[3].dirs[0]=specs[0].dirs[0]; assert(Reject(load)); specs[3]=saved;
  specs[3].reference=specs[0].reference; assert(Reject(load)); specs[3]=saved;
  assert(load().size()==4);
  for(auto& spec:specs) spec.options.geom_consistency=false;
  assert(load().size()==4); // global phase barrier still permits all-photo lists
  write(2); assert(load().size()==2);
  settings.static_three_solver_permits = 2;
  settings.affinity = 3;
  auto triple_args = args;
  triple_args.emplace("static_three_solver_permits", "2");
  auto triple = [&] { return LoadJobs(JobProjectPaths(triple_args), settings); };
  write(3); assert(triple().size() == 3);
  for (size_t count : {0,1,2,4,6}) { write(count); assert(Reject(triple)); }
  write(3);
  specs[2].options.num_threads = 2; assert(Reject(triple)); specs[2].options.num_threads = 1;
  const auto c = specs[2];
  specs[2].workspace = specs[0].workspace; assert(Reject(triple)); specs[2] = c;
  specs[2].dirs[0] = specs[0].dirs[0]; assert(Reject(triple)); specs[2] = c;
  specs[2].options.geom_consistency = true; assert(Reject(triple)); specs[2] = c;
  // Dev1 full forecast: no active-credit reduction or cap expansion.
  constexpr uint64_t gib=1ULL<<30;
  assert(!pair::CanReserveHostLookahead(32*gib,2*gib,15*gib,15*gib,22777529106ULL));
  assert(!pair::CanReserveHostLookahead(32*gib,2*gib,15*gib,15*gib,6095949420ULL));
}
'''
        temp = Path(tempfile.mkdtemp(prefix='geometry-pair-list-'))
        program = temp / 'test.cc'
        binary = temp / 'test'
        program.write_text(source)
        compiled = subprocess.run(
            ['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O1',
             '-pthread', '-I', str(ROOT / 'src'), str(program), '-o', str(binary)],
            capture_output=True, text=True, timeout=60)
        (temp / 'compile.log').write_text(compiled.stdout + compiled.stderr)
        print(f'GEOMETRY_PAIR_LIST_ARTIFACT {temp} compile_exit={compiled.returncode}')
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        result = subprocess.run([str(binary), str(temp / 'fixtures')],
                                capture_output=True, text=True, timeout=10)
        (temp / 'run.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_parser_and_geometry_gate(self):
        driver = (ROOT / 'src/colmap/exe/owned_pair.cc').read_text()
        settings = driver[driver.index('struct PairSettings {'):]
        settings = settings[:settings.index('\n};') + 3]
        parser = driver[driver.index('  std::set<std::string> keys = {'):]
        parser = parser[:parser.index('  const auto& g = jobs.front()->globals;')]
        source = r'''
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
namespace pair = colmap::mvs::owned;
''' + settings + r'''
static bool geometry = false;
static size_t count = 2;
uint64_t AffinityCount() { return 3; }
struct BaseOptionManager { explicit BaseOptionManager(bool) {} };
struct Options { bool geom_consistency = geometry; };
struct Job { std::unique_ptr<Options> patch_match_stereo = std::make_unique<Options>(); };
std::vector<int> JobProjectPaths(const std::map<std::string, std::string>&) { return {}; }
std::vector<std::unique_ptr<Job>> LoadJobs(const std::vector<int>&, const PairSettings&) {
  std::vector<std::unique_ptr<Job>> jobs;
  for (size_t i = 0; i < count; ++i) jobs.push_back(std::make_unique<Job>());
  return jobs;
}
int Parse(int argc, char** argv) {
''' + parser + r'''
  return settings.static_three_solver_permits ? settings.static_three_solver_permits : settings.geometry_early_permits ? 1 : 0;
}
int Invoke(std::vector<std::string> args) {
  std::vector<char*> argv;
  for (auto& arg : args) argv.push_back(arg.data());
  return Parse(static_cast<int>(argv.size()), argv.data());
}
bool Reject(std::vector<std::string> args) {
  try { Invoke(args); } catch (const std::invalid_argument&) { return true; }
  return false;
}
int main() {
  const std::vector<std::string> base = {
    "driver", "--dispatch", "pair", "--host_budget_bytes", "1000",
    "--device_budget_bytes", "1000", "--job_a_host_charge_bytes", "100",
    "--job_b_host_charge_bytes", "100", "--job_a_device_charge_bytes", "100",
    "--job_b_device_charge_bytes", "100", "--process_host_charge_bytes", "10",
    "--process_device_charge_bytes", "10", "--io_threads_limit", "1",
    "--job_a_project_path", "a.ini", "--job_b_project_path", "b.ini"};
  assert(Invoke(base) == 0);
  auto args = base;
  args.insert(args.end(), {"--geometry_early_permits", "0"});
  assert(Invoke(args) == 0);
  args.back() = "1";
  assert(Reject(args)); // photo rejected before owners
  geometry = true;
  assert(Invoke(args) == 1);
  args[2] = "serial";
  assert(Invoke(args) == 1); // coordinator preserves SERIAL scheduling
  args.back() = "0";
  assert(Invoke(args) == 0); // legacy geometry OFF
  args.back() = "1";
  args[2] = "pair";
  for (auto invalid : {"on", "off", "2", "-1", "", "01"}) {
    args.back() = invalid;
    assert(Reject(args));
  }
  args.back() = "1";
  count = 4;
  assert(Invoke(args) == 1); // finite geometry pairs, not rolling owners
  args.back() = "0";
  assert(Invoke(args) == 0);
  args.back() = "1";
  count = 2;
  args.insert(args.end(), {"--host_lookahead", "1", "--host_lookahead_charge_bytes", "1"});
  assert(Reject(args));
  args[args.size() - 3] = "0";
  assert(Invoke(args) == 1);
  args.insert(args.end(), {"--geometry_early_permits", "1"});
  assert(Reject(args)); // duplicate selection is not silently accepted
  args = base;
  args.insert(args.end(), {"--static_three_solver_permits", "2",
      "--job_c_project_path", "c.ini", "--job_c_host_charge_bytes", "100",
      "--job_c_device_charge_bytes", "100"});
  assert(Reject(args)); // exactly three only
  count = 3;
  assert(Invoke(args) == 2);
  args[args.size()-7] = "3";
  assert(Invoke(args) == 3);
  for (auto invalid : {"1", "4", "03", "-1"}) {
    args[args.size()-7] = invalid; assert(Reject(args));
  }
  args[args.size()-7] = "2";
  geometry = false; assert(Reject(args)); geometry = true;
  args[2] = "serial"; assert(Reject(args)); args[2] = "pair";
  count = 6; assert(Reject(args)); count = 3;
  args.insert(args.end(), {"--host_lookahead", "0"}); assert(Reject(args));
  std::cout << "early geometry CLI checks passed\n";
}
'''
        temp = Path(tempfile.mkdtemp(prefix='early-geometry-cli-'))
        program = temp / 'test.cc'
        binary = temp / 'test'
        program.write_text(source)
        compiled = subprocess.run(
            ['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O1',
             '-pthread', '-I', str(ROOT / 'src'), str(program), '-o', str(binary)],
            capture_output=True, text=True, timeout=60)
        (temp / 'compile.log').write_text(compiled.stdout + compiled.stderr)
        print(f'EARLY_GEOMETRY_CLI_ARTIFACT {temp} compile_exit={compiled.returncode}')
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
        (temp / 'run.log').write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
