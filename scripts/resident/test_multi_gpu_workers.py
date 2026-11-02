# SPDX-License-Identifier: BSD-3-Clause
"""Standalone CPU proof of the production device-worker ownership seam."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class MultiGpuWorkersTest(unittest.TestCase):
    def run_cpp(self, body):
        with tempfile.TemporaryDirectory(prefix='.workers-test-', dir=ROOT) as directory:
            source = pathlib.Path(directory) / 'test.cc'
            binary = pathlib.Path(directory) / 'test'
            source.write_text('''#include "colmap/mvs/multi_gpu_workers.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_graph_coordinator.h"
#include <cassert>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
using namespace colmap::mvs::owned;
int main() {''' + body + '}')
            subprocess.run(['g++', '-std=c++17', '-pthread', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT / 'src'), str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=10)

    def test_partial_launch_cancels_registered_controls_and_joins(self):
        self.run_cpp('''
MultiGpuWorkers workers;
std::mutex mutex;
std::condition_variable ready;
bool cancelled = false;
bool started = false;
std::atomic<int> drained{0};
workers.Register([&](std::exception_ptr error) {
  assert(error);
  std::lock_guard<std::mutex> lock(mutex);
  cancelled = true;
  ready.notify_all();
});
workers.Register([&](std::exception_ptr) { ++drained; });
size_t launched = 0;
try {
  workers.Run(2, [&](size_t) {
    std::unique_lock<std::mutex> lock(mutex);
    started = true;
    ready.notify_all();
    ready.wait(lock, [&] { return cancelled; });
    ++drained;
  }, [&](std::function<void()> task) {
    if (launched++ == 1) {
      std::unique_lock<std::mutex> lock(mutex);
      ready.wait(lock, [&] { return started; });
      throw std::runtime_error("launch failure");
    }
    return std::thread(std::move(task));
  });
  assert(false);
} catch (const std::runtime_error& error) {
  assert(std::string(error.what()) == "launch failure");
}
assert(drained == 2);
assert(workers.CancellationToken()->load());
''')

    def test_partial_owner_launch_failure_cancels_peer_before_local_drain(self):
        body = '''
MultiGpuWorkers workers;
const auto no_job = std::numeric_limits<size_t>::max();
PairCoordinator local({100, 100}, {1, 1}, 2,
    std::vector<ByteCharge>(3, {10, 10}), std::chrono::seconds(5));
PairCoordinator peer({100, 100}, {1, 1}, 1,
    std::vector<ByteCharge>(3, {10, 10}), std::chrono::seconds(5));
OwnerGraphLaunchState local_graph(3), peer_graph(3);
std::mutex mutex;
std::condition_variable wake;
bool local_running = false, peer_running = false, release_drain = false;
bool launch_thrown = false, immediate = false, cancelled_before_drain = false;
int controls_cancelled = 0;
std::atomic<int> cleaned{0};
std::atomic<int> local_starts{0}, peer_starts{0};
workers.Register([&](std::exception_ptr error) {
  local.Cancel(no_job, error);
  local_graph.Cancel(error);
  std::lock_guard<std::mutex> lock(mutex);
  cancelled_before_drain = cleaned == 0;
  ++controls_cancelled;
  wake.notify_all();
});
workers.Register([&](std::exception_ptr error) {
  peer.Cancel(no_job, error);
  peer_graph.Cancel(error);
  std::lock_guard<std::mutex> lock(mutex);
  ++controls_cancelled;
  wake.notify_all();
});
std::string reported;
try {
  workers.Run(2, [&](size_t device) {
    if (device == 0) {
      size_t launches = 0;
      auto factory = workers.OwnerThreadFactory([&](auto task) {
        if (launches++ == 1) {
          std::unique_lock<std::mutex> lock(mutex);
          wake.wait(lock, [&] { return local_running && peer_running; });
          launch_thrown = true;
          wake.notify_all();
          throw std::runtime_error("original owner launch failure");
        }
        return std::thread(std::move(task));
      });
      RunRollingOwners(local, [&](size_t job) {
        ++local_starts;
        local.BeginPreparation(job);
        local.ReadyAndWait(job);
        std::unique_lock<std::mutex> lock(mutex);
        local_running = true;
        wake.notify_all();
        wake.wait(lock, [&] { return release_drain; });
      }, [](size_t, const char*) {}, factory, [&](size_t) { ++cleaned; });
    } else {
      RunRollingOwners(peer, [&](size_t job) {
        ++peer_starts;
        peer.BeginPreparation(job);
        peer.ReadyAndWait(job);
        {
          std::unique_lock<std::mutex> lock(mutex);
          peer_running = true;
          wake.notify_all();
          wake.wait(lock, [&] { return launch_thrown; });
          immediate = wake.wait_for(lock, std::chrono::seconds(1), [&] {
            return controls_cancelled == 2;
          });
        }
        if (immediate) {
          try { local_graph.Check(0); assert(false); }
          catch (const std::runtime_error&) {}
          try { peer_graph.Check(0); assert(false); }
          catch (const std::runtime_error&) {}
          assert(local.AdmittedCount() == 2 && peer.AdmittedCount() == 1);
        }
        {
          std::lock_guard<std::mutex> lock(mutex);
          release_drain = true;
          wake.notify_all();
        }
        // A later device error must not replace the failed owner's launch.
        workers.Cancel(std::make_exception_ptr(std::runtime_error("later peer failure")));
        throw std::runtime_error("later peer failure");
      }, [](size_t, const char*) {}, workers.OwnerThreadFactory(),
         [&](size_t) { ++cleaned; });
    }
  });
} catch (const std::runtime_error& error) { reported = error.what(); }
// Use a normal failure exit rather than abort so the RED run leaves no core.
if (!immediate || !cancelled_before_drain || controls_cancelled != 2 ||
    reported != "original owner launch failure") return 1;
assert(local_starts == 1 && peer_starts == 1);
assert(local.AdmittedCount() == 2 && peer.AdmittedCount() == 1);
// Device controllers are joined; the parent can check refusal without racing
// RunRollingOwners' parent-only FIFO admission interface.
assert(local.AdmitNext() == no_job && peer.AdmitNext() == no_job);
assert(cleaned == 3);  // Running local, failed launch's admitted slot, peer.
'''
        for empty_owner in (False, True):
            with self.subTest(empty_owner=empty_owner):
                case = body
                if empty_owner:
                    case = case.replace('throw std::runtime_error("original owner launch failure");',
                                        'return std::thread{};')
                    case = case.replace('original owner launch failure', 'owner factory returned no worker')
                self.run_cpp(case)

    def test_worker_failure_wakes_sibling_and_preserves_first_error(self):
        self.run_cpp('''
MultiGpuWorkers workers;
std::mutex mutex;
std::condition_variable wake;
bool cancelled = false;
bool started = false;
std::atomic<int> drained{0};
workers.Register([&](std::exception_ptr) {
  std::lock_guard<std::mutex> lock(mutex);
  cancelled = true;
  wake.notify_all();
});
try {
  workers.Run(2, [&](size_t device) {
    if (device == 0) {
      std::unique_lock<std::mutex> lock(mutex);
      started = true;
      wake.notify_all();
      wake.wait(lock, [&] { return cancelled; });
      ++drained;
    } else {
      std::unique_lock<std::mutex> lock(mutex);
      wake.wait(lock, [&] { return started; });
      throw std::runtime_error("device failed");
    }
  });
  assert(false);
} catch (const std::runtime_error& error) {
  assert(std::string(error.what()) == "device failed");
}
workers.Cancel(std::make_exception_ptr(std::runtime_error("later")));
try { workers.Check(); assert(false); }
catch (const std::runtime_error& error) {
  assert(std::string(error.what()) == "device failed");
}
assert(drained == 1);
''')

    def test_error_fanout_wakes_real_coordinator_ready_waiter(self):
        self.run_cpp('''
MultiGpuWorkers workers;
PairCoordinator first(PairDispatch::kPair, {100, 100}, {1, 1},
    std::array<ByteCharge, 2>{{{10, 10}, {10, 10}}}, std::chrono::seconds(5));
PairCoordinator second(PairDispatch::kPair, {100, 100}, {1, 1},
    std::array<ByteCharge, 2>{{{10, 10}, {10, 10}}}, std::chrono::seconds(5));
workers.Register([&](std::exception_ptr e) { first.Cancel(std::numeric_limits<size_t>::max(), e); });
workers.Register([&](std::exception_ptr e) { second.Cancel(std::numeric_limits<size_t>::max(), e); });
std::mutex mutex;
std::condition_variable ready;
bool preparing = false;
std::atomic<bool> woke{false};
try {
  workers.Run(2, [&](size_t device) {
    if (device == 0) {
      first.BeginPreparation(0);
      { std::lock_guard<std::mutex> lock(mutex); preparing = true; ready.notify_all(); }
      try { first.ReadyAndWait(0); assert(false); }
      catch (const std::runtime_error&) { woke = true; throw; }
    } else {
      std::unique_lock<std::mutex> lock(mutex);
      ready.wait(lock, [&] { return preparing; });
      throw std::runtime_error("other device failed");
    }
  });
  assert(false);
} catch (const std::runtime_error& error) {
  assert(std::string(error.what()) == "other device failed");
}
assert(woke);
''')

    def test_external_cancel_wakes_blocked_device_waiter(self):
        self.run_cpp('''
MultiGpuWorkers workers;
std::mutex mutex;
std::condition_variable wake;
bool cancelled = false;
std::atomic<bool> request{false};
std::atomic<int> drained{0};
workers.Register([&](std::exception_ptr) {
  std::lock_guard<std::mutex> lock(mutex);
  cancelled = true;
  wake.notify_all();
});
try {
  workers.Run(1, [&](size_t) {
    request = true;
    std::unique_lock<std::mutex> lock(mutex);
    wake.wait(lock, [&] { return cancelled; });
    ++drained;
  }, MultiGpuWorkers::ThreadFactory{}, [&] {
    if (request) throw std::runtime_error("external cancel");
  });
  assert(false);
} catch (const std::runtime_error& error) {
  assert(std::string(error.what()) == "external cancel");
}
assert(drained == 1);
''')

    def test_null_cancellation_never_returns_success(self):
        self.run_cpp('''
MultiGpuWorkers workers;
workers.Cancel({});
try { workers.Run(1, [](size_t) {}); }
catch (const std::runtime_error&) { return 0; }
return 1;
''')

    def test_complete_phase_barrier_and_unique_outputs(self):
        self.run_cpp('''
std::atomic<int> photo_done{0};
std::atomic<int> writes[4]{};
MultiGpuWorkers photo;
photo.Run(2, [&](size_t device) {
  ++writes[device];
  ++photo_done;
});
MultiGpuWorkers geo;
geo.Run(2, [&](size_t device) {
  assert(photo_done == 2);
  ++writes[device + 2];
});
for (const auto& count : writes) assert(count == 1);
try { photo.Register([](std::exception_ptr) {}); assert(false); }
catch (const std::logic_error&) {}
''')


if __name__ == '__main__':
    unittest.main()
