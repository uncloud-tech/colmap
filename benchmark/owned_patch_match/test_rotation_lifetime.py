# SPDX-License-Identifier: BSD-3-Clause
"""Actual coordinator fault tests; fake drains stand in for CUDA completion."""

import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CPP = (
    "\n"
    '#include "colmap/mvs/owned_pair_coordinator.h"\n'
    "#include <atomic>\n"
    "#include <cassert>\n"
    "#include <cstdlib>\n"
    "#include <future>\n"
    "#include <thread>\n"
    "using namespace colmap::mvs::owned;\n"
    "using namespace std::chrono_literals;\n"
    "std::exception_ptr Error(const char* s) { return std::make_e"
    "xception_ptr(std::runtime_error(s)); }\n"
    "void FinishFailed(PairCoordinator& c, const std::string& fir"
    "st) {\n"
    "  assert(c.ReservedBytes().device == 3);\n"
    "  try { c.Finish(); assert(false); }\n"
    "  catch (const std::runtime_error& e) { assert(e.what() == f"
    "irst); }\n"
    "  assert(c.ReservedBytes().device == 0);\n"
    "}\n"
    "int main(int argc, char** argv) {\n"
    "  std::set_terminate([] { std::_Exit(91); });\n"
    '  const std::string mode = argc > 1 ? argv[1] : "solver";\n'
    "  PairCoordinator c(PairDispatch::kPair, {3,3}, {1,1}, {{{1,"
    "1},{1,1}}},\n"
    '                    mode == "timeout" ? 80ms : 2s);\n'
    '  if (mode == "unknown") {\n'
    "    c.BeginPreparation(0);\n"
    '    c.CancelAndAwaitCleanup(0, Error("first"), [] { throw st'
    'd::runtime_error("drain unknown"); });\n'
    "    return 0;\n"
    "  }\n"
    '  if (mode == "unstarted") {\n'
    "    c.BeginPreparation(0);\n"
    "    bool drained = false;\n"
    '    c.CancelAndAwaitCleanup(0, Error("first"), [&] { drained'
    " = true; });\n"
    "    assert(drained);\n"
    '    c.Retire(0); c.Retire(1); FinishFailed(c, "first"); retu'
    "rn 0;\n"
    "  }\n"
    '  if (mode == "preparation") {\n'
    "    std::atomic<int> solver_entries{0}, drains{0};\n"
    "    std::thread a([&] {\n"
    "      try { c.BeginPreparation(0); c.ReadyAndWait(0); ++solv"
    "er_entries; }\n"
    "      catch (...) { c.CancelAndAwaitCleanup(0, std::current_"
    "exception(), [&] { ++drains; }); }\n"
    "      c.Retire(0);\n"
    "    });\n"
    "    std::thread b([&] {\n"
    "      c.BeginPreparation(1);\n"
    '      c.CancelAndAwaitCleanup(1, Error("preparation"), [&] {'
    " ++drains; });\n"
    "      c.Retire(1);\n"
    "    });\n"
    "    a.join(); b.join(); assert(solver_entries==0 && drains=="
    "2);\n"
    '    FinishFailed(c, "preparation"); return 0;\n'
    "  }\n"
    "  std::promise<void> peer_started, release_peer, drain_a, do"
    "ne_a;\n"
    "  auto peer_ready = peer_started.get_future();\n"
    "  auto peer_release = release_peer.get_future();\n"
    "  auto a_drained = drain_a.get_future();\n"
    "  auto a_done = done_a.get_future();\n"
    "  std::atomic<bool> peer_active{false}, freed_a{false}, free"
    "d_b{false};\n"
    "  std::thread b([&] {\n"
    "    c.BeginPreparation(1); c.ReadyAndWait(1);\n"
    "    peer_active=true; peer_started.set_value(); peer_release"
    ".wait();\n"
    "    peer_active=false; c.SolverDone(1);\n"
    "    try { c.BeginOutput(1); assert(false); }\n"
    '    catch (...) { c.CancelAndAwaitCleanup(1, Error("secondar'
    'y"), [] {}); }\n'
    "    freed_b=true; c.Retire(1);\n"
    "  });\n"
    "  std::thread a([&] {\n"
    "    c.BeginPreparation(0); c.ReadyAndWait(0); peer_ready.wai"
    "t();\n"
    '    if (mode == "output") { c.SolverDone(0); c.BeginOutput(0'
    "); }\n"
    "    c.CancelAndAwaitCleanup(0, Error(mode.c_str()), [&] { dr"
    "ain_a.set_value(); });\n"
    "    assert(!peer_active); // no owning-buffer destruction be"
    "fore peer completion\n"
    "    freed_a=true; c.Retire(0); done_a.set_value();\n"
    "  });\n"
    "  a_drained.wait();\n"
    '  if (mode == "timeout") {\n'
    "    // Unknown peer completion must terminate, not return an"
    "d release owners.\n"
    "    if (a_done.wait_for(500ms)==std::future_status::ready) {"
    "\n"
    "      release_peer.set_value(); a.join(); b.join(); return 0"
    ";\n"
    "    }\n"
    "    std::_Exit(92);\n"
    "  }\n"
    "  assert(a_done.wait_for(25ms)==std::future_status::timeout)"
    ";\n"
    "  assert(!freed_a && !freed_b && c.ReservedBytes().host==3);"
    "\n"
    "  release_peer.set_value(); a.join(); b.join();\n"
    "  assert(freed_a && freed_b); FinishFailed(c, mode);\n"
    "}\n"
)


class RotationLifetime(unittest.TestCase):
    def test_real_owner_wiring(self):
        cc = (ROOT / "src/colmap/mvs/patch_match.cc").read_text()
        header = (ROOT / "src/colmap/mvs/patch_match.h").read_text()
        cuda = (ROOT / "src/colmap/mvs/patch_match_cuda_owned.h").read_text()
        self.assertIn("void DrainOwned() noexcept;", header)
        self.assertIn("void PatchMatch::DrainOwned() noexcept", cc)
        self.assertIn("owned_patch_match_cuda_->DrainForCleanup();", cc)
        self.assertIn("void DrainForCleanup() noexcept", cuda)
        self.assertIn("stream_->DrainBeforeDestruction();", cuda)
        pair = cc[cc.index("std::unique_ptr<PatchMatch> patch_match;") :]
        catch = pair[
            pair.index("} catch (...) {") : pair.index(
                "\n  PatchMatch patch_match("
            )
        ]
        self.assertLess(
            catch.index("CancelAndAwaitCleanup"),
            catch.index("patch_match->DrainOwned()"),
        )
        self.assertLess(
            catch.index("patch_match->DrainOwned()"), catch.index("throw;")
        )

    def test_real_coordinator_fault_boundaries(self):
        with tempfile.TemporaryDirectory(prefix="rotation-lifetime-") as tmp:
            src = pathlib.Path(tmp) / "test.cc"
            exe = pathlib.Path(tmp) / "test"
            src.write_text(CPP)
            build = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-pthread",
                    "-I",
                    str(ROOT / "src"),
                    str(src),
                    "-o",
                    str(exe),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            for mode in (
                "preparation",
                "unstarted",
                "solver",
                "output",
                "unknown",
                "timeout",
            ):
                with self.subTest(mode=mode):
                    run = subprocess.run(
                        [str(exe), mode],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    self.assertEqual(
                        run.returncode,
                        91 if mode in ("unknown", "timeout") else 0,
                        run.stdout + run.stderr,
                    )


if __name__ == "__main__":
    unittest.main()
