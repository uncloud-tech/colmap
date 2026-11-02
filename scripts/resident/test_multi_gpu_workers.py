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
