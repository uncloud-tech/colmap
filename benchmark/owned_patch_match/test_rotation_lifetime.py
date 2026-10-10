# SPDX-License-Identifier: BSD-3-Clause
"""Actual coordinator fault tests; fake drains stand in for CUDA completion."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CPP = r'''
#include "colmap/mvs/owned_pair_coordinator.h"
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <future>
#include <thread>
using namespace colmap::mvs::owned;
using namespace std::chrono_literals;
std::exception_ptr Error(const char* s) { return std::make_exception_ptr(std::runtime_error(s)); }
void FinishFailed(PairCoordinator& c, const std::string& first) {
  assert(c.ReservedBytes().device == 3);
  try { c.Finish(); assert(false); }
  catch (const std::runtime_error& e) { assert(e.what() == first); }
  assert(c.ReservedBytes().device == 0);
}
int main(int argc, char** argv) {
  std::set_terminate([] { std::_Exit(91); });
  const std::string mode = argc > 1 ? argv[1] : "solver";
  PairCoordinator c(PairDispatch::kPair, {3,3}, {1,1}, {{{1,1},{1,1}}},
                    mode == "timeout" ? 80ms : 2s);
  if (mode == "unknown") {
    c.BeginPreparation(0);
    c.CancelAndAwaitCleanup(0, Error("first"), [] { throw std::runtime_error("drain unknown"); });
    return 0;
  }
  if (mode == "unstarted") {
    c.BeginPreparation(0);
    bool drained = false;
    c.CancelAndAwaitCleanup(0, Error("first"), [&] { drained = true; });
    assert(drained);
    c.Retire(0); c.Retire(1); FinishFailed(c, "first"); return 0;
  }
  if (mode == "preparation") {
    std::atomic<int> solver_entries{0}, drains{0};
    std::thread a([&] {
      try { c.BeginPreparation(0); c.ReadyAndWait(0); ++solver_entries; }
      catch (...) { c.CancelAndAwaitCleanup(0, std::current_exception(), [&] { ++drains; }); }
      c.Retire(0);
    });
    std::thread b([&] {
      c.BeginPreparation(1);
      c.CancelAndAwaitCleanup(1, Error("preparation"), [&] { ++drains; });
      c.Retire(1);
    });
    a.join(); b.join(); assert(solver_entries==0 && drains==2);
    FinishFailed(c, "preparation"); return 0;
  }
  std::promise<void> peer_started, release_peer, drain_a, done_a;
  auto peer_ready = peer_started.get_future();
  auto peer_release = release_peer.get_future();
  auto a_drained = drain_a.get_future();
  auto a_done = done_a.get_future();
  std::atomic<bool> peer_active{false}, freed_a{false}, freed_b{false};
  std::thread b([&] {
    c.BeginPreparation(1); c.ReadyAndWait(1);
    peer_active=true; peer_started.set_value(); peer_release.wait();
    peer_active=false; c.SolverDone(1);
    try { c.BeginOutput(1); assert(false); }
    catch (...) { c.CancelAndAwaitCleanup(1, Error("secondary"), [] {}); }
    freed_b=true; c.Retire(1);
  });
  std::thread a([&] {
    c.BeginPreparation(0); c.ReadyAndWait(0); peer_ready.wait();
    if (mode == "output") { c.SolverDone(0); c.BeginOutput(0); }
    c.CancelAndAwaitCleanup(0, Error(mode.c_str()), [&] { drain_a.set_value(); });
    assert(!peer_active); // no owning-buffer destruction before peer completion
    freed_a=true; c.Retire(0); done_a.set_value();
  });
  a_drained.wait();
  if (mode == "timeout") {
    // Unknown peer completion must terminate, not return and release owners.
    if (a_done.wait_for(500ms)==std::future_status::ready) {
      release_peer.set_value(); a.join(); b.join(); return 0;
    }
    std::_Exit(92);
  }
  assert(a_done.wait_for(25ms)==std::future_status::timeout);
  assert(!freed_a && !freed_b && c.ReservedBytes().host==3);
  release_peer.set_value(); a.join(); b.join();
  assert(freed_a && freed_b); FinishFailed(c, mode);
}
'''


class RotationLifetime(unittest.TestCase):
    def test_real_owner_wiring(self):
        cc = (ROOT / 'src/colmap/mvs/patch_match.cc').read_text()
        header = (ROOT / 'src/colmap/mvs/patch_match.h').read_text()
        cuda = (ROOT / 'src/colmap/mvs/patch_match_cuda_owned.h').read_text()
        self.assertIn('void DrainOwned() noexcept;', header)
        self.assertIn('void PatchMatch::DrainOwned() noexcept', cc)
        self.assertIn('owned_patch_match_cuda_->DrainForCleanup();', cc)
        self.assertIn('void DrainForCleanup() noexcept', cuda)
        self.assertIn('stream_->DrainBeforeDestruction();', cuda)
        pair = cc[cc.index('std::unique_ptr<PatchMatch> patch_match;'):]
        catch = pair[pair.index('} catch (...) {'):pair.index('\n  PatchMatch patch_match(')]
        self.assertLess(catch.index('CancelAndAwaitCleanup'), catch.index('patch_match->DrainOwned()'))
        self.assertLess(catch.index('patch_match->DrainOwned()'), catch.index('throw;'))

    def test_real_coordinator_fault_boundaries(self):
        with tempfile.TemporaryDirectory(prefix='rotation-lifetime-') as tmp:
            src = pathlib.Path(tmp) / 'test.cc'
            exe = pathlib.Path(tmp) / 'test'
            src.write_text(CPP)
            build = subprocess.run(['g++', '-std=c++17', '-pthread', '-I', str(ROOT / 'src'),
                                    str(src), '-o', str(exe)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            for mode in ('preparation', 'unstarted', 'solver', 'output', 'unknown', 'timeout'):
                with self.subTest(mode=mode):
                    run = subprocess.run([str(exe), mode], capture_output=True, text=True, timeout=5)
                    self.assertEqual(run.returncode, 91 if mode in ('unknown', 'timeout') else 0,
                                     run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
