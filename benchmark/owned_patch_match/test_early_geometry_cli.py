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
uint64_t AffinityCount() { return 2; }
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
  return settings.geometry_early_permits ? 1 : 0;
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
  args[2] = "pair";
  for (auto invalid : {"on", "off", "2", "-1", "", "01"}) {
    args.back() = invalid;
    assert(Reject(args));
  }
  args.back() = "1";
  count = 4;
  assert(Reject(args));
  count = 2;
  args.insert(args.end(), {"--host_lookahead", "1", "--host_lookahead_charge_bytes", "1"});
  assert(Reject(args));
  args[args.size() - 3] = "0";
  assert(Invoke(args) == 1);
  args.insert(args.end(), {"--geometry_early_permits", "1"});
  assert(Reject(args)); // duplicate selection is not silently accepted
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
