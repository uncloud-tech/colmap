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
        driver = (ROOT / "src/colmap/exe/owned_pair.cc").read_text()
        settings = driver[driver.index("struct PairSettings {") :]
        settings = settings[: settings.index("\n};") + 3]
        loader = driver[
            driver.index(
                "std::vector<fs::path> JobProjectPaths("
            ) : driver.index("// RunJob owns the controller")
        ]
        source = (
            r"""
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/owned_tile.h"
#include "colmap/mvs/owned_host_lookahead.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
namespace fs = std::filesystem;
namespace pair = colmap::mvs::owned;
"""
            + settings
            + (
                "\n"
                "struct Options { bool geom_consistency = true; std::string g"
                'pu_index = "0"; int num_threads = 1; };\n'
                "struct Globals { int seed = 0; bool operator==(const Globals"
                "& other) const { return seed == other.seed; } };\n"
                "struct Spec { Options options; fs::path workspace; std::stri"
                "ng reference; std::array<fs::path,2> dirs; Globals globals; "
                "};\n"
                "std::vector<Spec> specs;\n"
                "struct JobOptions {\n"
                "  Globals globals;\n"
                "  std::unique_ptr<Options> patch_match_stereo = std::make_un"
                "ique<Options>();\n"
                "  fs::path workspace; std::string reference_name; std::array"
                "<fs::path,2> output_dirs;\n"
                "  bool Read(const std::string& path, bool) {\n"
                "    const auto& s = specs.at(std::stoul(fs::path(path).stem("
                ").string()));\n"
                "    *patch_match_stereo = s.options; workspace = s.workspace"
                ";\n"
                "    reference_name = s.reference; output_dirs = s.dirs; glob"
                "als = s.globals; return true;\n"
                "  }\n"
                "};\n"
                "// Native INI/decode is not part of this bounded CPU build. "
                "All list loading,\n"
                "// cross-job phase/device/alias and per-pair IO guards below"
                " are real source.\n"
                "std::vector<pair::ByteCharge> validated_charges;\n"
                "void ValidateJob(JobOptions&, pair::ByteCharge charge) { val"
                "idated_charges.push_back(charge); }\n"
            )
            + loader
            + (
                "\n"
                "template<class F> bool Reject(F f) {\n"
                "  try { f(); } catch (const std::exception&) { return true; "
                "} return false;\n"
                "}\n"
                "int main(int argc, char** argv) {\n"
                "  assert(argc == 2); const fs::path root(argv[1]); fs::creat"
                "e_directories(root);\n"
                '  PairSettings settings{"pair", {100,100}, {1,1}, {{{1,1},{1'
                ",1}}}, 2, 1, true};\n"
                '  const auto list = root / "jobs.txt";\n'
                '  const std::map<std::string,std::string> args{{"job_project'
                '_list_path",list.string()}};\n'
                "  auto write = [&](size_t count) { std::ofstream out(list); "
                'for(size_t j=0;j<count;++j) out << j << ".ini\\n"; };\n'
                "  for(size_t j=0;j<6;++j) {\n"
                "    const auto workspace = root / std::to_string(j); fs::cre"
                "ate_directories(workspace);\n"
                "    specs.push_back({{},workspace,std::to_string(j),{workspa"
                'ce/"depth",workspace/"normal"},{}});\n'
                "  }\n"
                "  auto load = [&] { return LoadJobs(JobProjectPaths(args),se"
                "ttings); };\n"
                "  write(4); assert(load().size()==4);\n"
                "  write(3); assert(Reject(load)); write(0); assert(Reject(lo"
                "ad)); write(4);\n"
                "  specs[3].options.geom_consistency=false; assert(Reject(loa"
                "d)); specs[3].options.geom_consistency=true;\n"
                '  specs[3].options.gpu_index="1"; assert(Reject(load)); spec'
                's[3].options.gpu_index="0";\n'
                "  const auto saved=specs[3]; specs[3].workspace=specs[0].wor"
                "kspace; assert(Reject(load)); specs[3]=saved;\n"
                "  specs[3].dirs[0]=specs[0].dirs[0]; assert(Reject(load)); s"
                "pecs[3]=saved;\n"
                "  specs[3].reference=specs[0].reference; assert(Reject(load)"
                "); specs[3]=saved;\n"
                "  assert(load().size()==4);\n"
                "  for(auto& spec:specs) spec.options.geom_consistency=false;"
                "\n"
                "  assert(load().size()==4); // global phase barrier still pe"
                "rmits all-photo lists\n"
                "  write(2); assert(load().size()==2);\n"
                "  settings.static_three_solver_permits = 2;\n"
                "  settings.affinity = 3;\n"
                "  auto triple_args = args;\n"
                '  triple_args.emplace("static_three_solver_permits", "2");\n'
                "  auto triple = [&] { return LoadJobs(JobProjectPaths(triple"
                "_args), settings); };\n"
                "  write(3); assert(triple().size() == 3);\n"
                "  for (size_t count : {0,1,2,4,5}) { write(count); assert(Re"
                "ject(triple)); }\n"
                "  settings.charges = {{{13,10},{13,10},{12,9}}};\n"
                "  write(6); validated_charges.clear(); assert(triple().size("
                ") == 6);\n"
                "  assert(validated_charges.size() == 6);\n"
                "  for(size_t j=0;j<6;++j) {\n"
                "    assert(validated_charges[j].host == settings.charges[j%3"
                "].host);\n"
                "    assert(validated_charges[j].device == settings.charges[j"
                "%3].device);\n"
                "  }\n"
                "  const auto last = specs[5];\n"
                "  specs[5].globals.seed = 1; assert(Reject(triple)); specs[5"
                "] = last;\n"
                "  specs[5].options.geom_consistency = true; assert(Reject(tr"
                "iple)); specs[5] = last;\n"
                '  specs[5].options.gpu_index = "1"; assert(Reject(triple)); '
                "specs[5] = last;\n"
                "  specs[5].options.num_threads = 2; assert(Reject(triple)); "
                "specs[5] = last;\n"
                "  specs[5].workspace = specs[0].workspace; assert(Reject(tri"
                "ple)); specs[5] = last;\n"
                "  specs[5].dirs[0] = specs[0].dirs[0]; assert(Reject(triple)"
                "); specs[5] = last;\n"
                "  specs[5].reference = specs[0].reference; assert(Reject(tri"
                "ple)); specs[5] = last;\n"
                "  write(3);\n"
                "  specs[2].options.num_threads = 2; assert(Reject(triple)); "
                "specs[2].options.num_threads = 1;\n"
                "  const auto c = specs[2];\n"
                "  specs[2].workspace = specs[0].workspace; assert(Reject(tri"
                "ple)); specs[2] = c;\n"
                "  specs[2].dirs[0] = specs[0].dirs[0]; assert(Reject(triple)"
                "); specs[2] = c;\n"
                "  specs[2].options.geom_consistency = true; assert(Reject(tr"
                "iple)); specs[2] = c;\n"
                "  // Dev1 full forecast: no active-credit reduction or cap e"
                "xpansion.\n"
                "  constexpr uint64_t gib=1ULL<<30;\n"
                "  assert(!pair::CanReserveHostLookahead(32*gib,2*gib,15*gib,"
                "15*gib,22777529106ULL));\n"
                "  assert(!pair::CanReserveHostLookahead(32*gib,2*gib,15*gib,"
                "15*gib,6095949420ULL));\n"
                "}\n"
            )
        )
        temp = Path(tempfile.mkdtemp(prefix="geometry-pair-list-"))
        program = temp / "test.cc"
        binary = temp / "test"
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
            f"GEOMETRY_PAIR_LIST_ARTIFACT {temp} "
            f"compile_exit={compiled.returncode}"
        )
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        result = subprocess.run(
            [str(binary), str(temp / "fixtures")],
            capture_output=True,
            text=True,
            timeout=10,
        )
        (temp / "run.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_parser_and_geometry_gate(self):
        driver = (ROOT / "src/colmap/exe/owned_pair.cc").read_text()
        settings = driver[driver.index("struct PairSettings {") :]
        settings = settings[: settings.index("\n};") + 3]
        parser = driver[driver.index("  std::set<std::string> keys = {") :]
        parser = parser[
            : parser.index("  const auto& g = jobs.front()->globals;")
        ]
        source = (
            r"""
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/owned_tile.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
namespace pair = colmap::mvs::owned;
"""
            + settings
            + (
                "\n"
                "static bool geometry = false;\n"
                "static size_t count = 2;\n"
                "static int selected_tile = 0;\n"
                "uint64_t AffinityCount() { return 3; }\n"
                "struct BaseOptionManager { explicit BaseOptionManager(bool) "
                "{} };\n"
                "struct Options { bool geom_consistency = geometry; };\n"
                "struct Job { std::unique_ptr<Options> patch_match_stereo = s"
                "td::make_unique<Options>(); };\n"
                "std::vector<int> JobProjectPaths(const std::map<std::string,"
                " std::string>&) { return {}; }\n"
                "std::vector<std::unique_ptr<Job>> LoadJobs(const std::vector"
                "<int>&, const PairSettings& settings) {\n"
                "  selected_tile = settings.owned_tile_threads;\n"
                "  std::vector<std::unique_ptr<Job>> jobs;\n"
                "  for (size_t i = 0; i < count; ++i) jobs.push_back(std::mak"
                "e_unique<Job>());\n"
                "  return jobs;\n"
                "}\n"
                "int Parse(int argc, char** argv) {\n"
            )
            + parser
            + (
                "\n"
                "  return settings.static_three_solver_permits ? settings.sta"
                "tic_three_solver_permits : settings.geometry_early_permits ?"
                " 1 : 0;\n"
                "}\n"
                "int Invoke(std::vector<std::string> args) {\n"
                "  std::vector<char*> argv;\n"
                "  for (auto& arg : args) argv.push_back(arg.data());\n"
                "  return Parse(static_cast<int>(argv.size()), argv.data());"
                "\n"
                "}\n"
                "bool Reject(std::vector<std::string> args) {\n"
                "  try { Invoke(args); } catch (const std::invalid_argument&)"
                " { return true; }\n"
                "  return false;\n"
                "}\n"
                "int main() {\n"
                "  const std::vector<std::string> base = {\n"
                '    "driver", "--dispatch", "pair", "--host_budget_bytes", "'
                '1000",\n'
                '    "--device_budget_bytes", "1000", "--job_a_host_charge_by'
                'tes", "100",\n'
                '    "--job_b_host_charge_bytes", "100", "--job_a_device_char'
                'ge_bytes", "100",\n'
                '    "--job_b_device_charge_bytes", "100", "--process_host_ch'
                'arge_bytes", "10",\n'
                '    "--process_device_charge_bytes", "10", "--io_threads_lim'
                'it", "1",\n'
                '    "--job_a_project_path", "a.ini", "--job_b_project_path",'
                ' "b.ini"};\n'
                "  assert(Invoke(base) == 0); assert(selected_tile == 32);\n"
                '  for (auto width : {"8", "16", "32"}) {\n'
                "    auto tile_args = base;\n"
                '    tile_args.insert(tile_args.end(), {"--owned_tile_threads'
                '", width});\n'
                "    assert(Invoke(tile_args) == 0); assert(selected_tile == "
                "std::stoi(width));\n"
                '    tile_args[2] = "serial";\n'
                "    assert(Invoke(tile_args) == 0); assert(selected_tile == "
                "std::stoi(width));\n"
                '    tile_args.insert(tile_args.end(), {"--owned_tile_threads'
                '", width});\n'
                "    assert(Reject(tile_args));\n"
                "  }\n"
                '  for (auto invalid : {"0", "4", "64", "-8", "+8", "08", "16'
                ' ", "", "auto"}) {\n'
                "    auto tile_args = base;\n"
                '    tile_args.insert(tile_args.end(), {"--owned_tile_threads'
                '", invalid});\n'
                "    selected_tile = 0;\n"
                "    assert(Reject(tile_args)); assert(selected_tile == 0); /"
                "/ before even job loading\n"
                "  }\n"
                "  auto args = base;\n"
                '  args.insert(args.end(), {"--geometry_early_permits", "0"})'
                ";\n"
                "  assert(Invoke(args) == 0);\n"
                '  args.back() = "1";\n'
                "  assert(Reject(args)); // photo rejected before owners\n"
                "  geometry = true;\n"
                "  assert(Invoke(args) == 1);\n"
                '  args[2] = "serial";\n'
                "  assert(Invoke(args) == 1); // coordinator preserves SERIAL"
                " scheduling\n"
                '  args.back() = "0";\n'
                "  assert(Invoke(args) == 0); // legacy geometry OFF\n"
                '  args.back() = "1";\n'
                '  args[2] = "pair";\n'
                '  for (auto invalid : {"on", "off", "2", "-1", "", "01"}) {'
                "\n"
                "    args.back() = invalid;\n"
                "    assert(Reject(args));\n"
                "  }\n"
                '  args.back() = "1";\n'
                "  count = 4;\n"
                "  assert(Invoke(args) == 1); // finite geometry pairs, not r"
                "olling owners\n"
                '  args.back() = "0";\n'
                "  assert(Invoke(args) == 0);\n"
                '  args.back() = "1";\n'
                "  count = 2;\n"
                '  args.insert(args.end(), {"--host_lookahead", "1", "--host_'
                'lookahead_charge_bytes", "1"});\n'
                "  assert(Reject(args));\n"
                '  args[args.size() - 3] = "0";\n'
                "  assert(Invoke(args) == 1);\n"
                '  args.insert(args.end(), {"--geometry_early_permits", "1"})'
                ";\n"
                "  assert(Reject(args)); // duplicate selection is not silent"
                "ly accepted\n"
                "  args = base;\n"
                '  args.insert(args.end(), {"--static_three_solver_permits", '
                '"2",\n'
                '      "--job_c_project_path", "c.ini", "--job_c_host_charge_'
                'bytes", "100",\n'
                '      "--job_c_device_charge_bytes", "100"});\n'
                "  assert(Reject(args)); // exactly three only\n"
                "  count = 3;\n"
                "  assert(Invoke(args) == 2);\n"
                '  args[args.size()-7] = "3";\n'
                "  assert(Invoke(args) == 3);\n"
                '  for (auto invalid : {"1", "4", "03", "-1"}) {\n'
                "    args[args.size()-7] = invalid; assert(Reject(args));\n"
                "  }\n"
                '  args[args.size()-7] = "2";\n'
                "  geometry = false; assert(Reject(args)); geometry = true;\n"
                '  args[2] = "serial"; assert(Reject(args)); args[2] = "pair"'
                ";\n"
                "  count = 6; assert(Invoke(args) == 2);\n"
                "  for(size_t invalid : {0,1,2,4,5,7}) { count = invalid; ass"
                "ert(Reject(args)); }\n"
                "  count = 3;\n"
                '  args.insert(args.end(), {"--host_lookahead", "0"}); assert'
                "(Reject(args));\n"
                "  auto rolling = base;\n"
                '  rolling.insert(rolling.end(), {"--static_three_solver_perm'
                'its", "3",\n'
                '      "--job_c_project_path", "c.ini", "--job_c_host_charge_'
                'bytes", "100",\n'
                '      "--job_c_device_charge_bytes", "100", "--owned_feed_mo'
                'de", "rolling",\n'
                '      "--job_forecast_list_path", "forecasts.txt"});\n'
                "  count = 6;\n"
                "  assert(Invoke(rolling) == 3);\n"
                '  auto invalid = rolling; invalid[2] = "serial"; assert(Reje'
                "ct(invalid));\n"
                "  geometry = false; assert(Reject(rolling)); geometry = true"
                ";\n"
                '  invalid = rolling; invalid[invalid.size()-3] = "auto"; ass'
                "ert(Reject(invalid));\n"
                '  invalid[invalid.size()-3] = "epochs"; assert(Reject(invali'
                "d)); // No epoch forecasts.\n"
                "  invalid = rolling; invalid.resize(invalid.size()-2); asser"
                "t(Reject(invalid));\n"
                '  invalid = rolling; invalid[base.size()+1] = "2"; assert(Re'
                "ject(invalid));\n"
                '  for (auto flag : {"--host_lookahead", "--geometry_early_pe'
                'rmits"}) {\n'
                "    invalid = rolling; invalid.insert(invalid.end(), {flag, "
                '"0"}); assert(Reject(invalid));\n'
                "  }\n"
                '  invalid = base; invalid.insert(invalid.end(), {"--owned_fe'
                'ed_mode", "epochs"});\n'
                "  count = 2; assert(Invoke(invalid) == 0); // Explicit defau"
                "lt does not select feed.\n"
                '  std::cout << "early geometry CLI checks passed\\n";\n'
                "}\n"
            )
        )
        temp = Path(tempfile.mkdtemp(prefix="early-geometry-cli-"))
        program = temp / "test.cc"
        binary = temp / "test"
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
            f"EARLY_GEOMETRY_CLI_ARTIFACT {temp} "
            f"compile_exit={compiled.returncode}"
        )
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        result = subprocess.run(
            [str(binary)], capture_output=True, text=True, timeout=10
        )
        (temp / "run.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
