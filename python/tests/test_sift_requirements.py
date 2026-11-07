# SPDX-License-Identifier: BSD-3-Clause
"""CPU-native public SIFT sequence admission contract; no CUDA execution claim."""
import os
import shlex
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SiftRequirementsTest(unittest.TestCase):
    def test_sequence_selection_and_actual_growth(self):
        source = r'''
#include "thirdparty/SiftGPU/SiftRequirements.h"
#include <cassert>
#include <limits>
int main() {
  using namespace siftgpu;
  auto tiny = SiftSequenceRequirements({{1,1}});
  assert(SiftSelect(tiny,1ULL<<30,1ULL<<30,0,0,0,0,0).mode=="resident");
  assert(!tiny.stream_supported);
  auto one = SiftSequenceRequirements({{256,128}});
  auto crossed = SiftSequenceRequirements({{256,128},{128,256}});
  assert(crossed.resident_device > one.resident_device);
  assert(crossed.input_history_bytes == one.input_history_bytes);
  auto history = SiftSequenceRequirements({{264,129},{256,129},{264,129},
      {257,129},{256,129},{264,130},{256,129}});
  assert(history.input_history_bytes == 137280);
  assert(history.retained_width == 528 && history.retained_height == 260);
  auto exact = SiftSelect(one, SiftRequired(one.resident_host,10,7),
                         SiftRequired(one.resident_device,20,9),
                         10,20,7,9,0);
  assert(exact.mode == "resident");
  auto streamed = SiftSelect(one, 1ULL<<30,
                            SiftRequired(one.stream_device,20,9),
                            10,20,7,9,one.scratch_bytes);
  assert(streamed.mode == "streamed");
  auto disk = SiftSelect(one, 1ULL<<30,
                        SiftRequired(one.stream_device,20,9),
                        10,20,7,9,one.scratch_bytes-1);
  assert(disk.mode == "refused");
  auto neither = SiftSelect(one,1,1,0,0,0,0,0);
  assert(neither.mode == "refused");
  assert(SiftRequired(100,10,7) == 121);
  bool overflow=false;
  try { SiftRequired(std::numeric_limits<uint64_t>::max(),1,0); }
  catch (const std::overflow_error&) { overflow=true; }
  assert(overflow);
  SiftDeviceBudget budget(103,0,0);
  budget.Replace(0,100);
  bool refused=false;
  try { budget.Replace(0,1); } catch(const std::runtime_error&) {refused=true;}
  assert(refused && budget.Used()==100);
  budget.Replace(100,99);
  assert(budget.Used()==99);
  bool unsupported=false;
  try { SiftSequenceRequirements({{0,128}}); }
  catch (const std::invalid_argument&) { unsupported=true; }
  assert(unsupported);
  std::vector<float> values(10);
  const auto old_capacity=values.capacity();
  {
    SiftHostBudgetScope scope([](uint64_t) { throw std::runtime_error("host no fit"); });
    bool host_refused=false;
    try { SiftResizeHostVector(values,100); }
    catch(const std::runtime_error&) { host_refused=true; }
    assert(host_refused && values.capacity()==old_capacity && values.size()==10);
  }
  bool window_refused=false;
  try { DescriptorPlan(256,100,1000,1000); }
  catch(const std::runtime_error&) { window_refused=true; }
  assert(window_refused);
  const auto raw=SiftPayloadCapacity(121,10,7);
  assert(raw==100 && SiftRequired(raw,10,7)==121);
  assert(SiftSelect(one,1ULL<<30,SiftRequired(one.stream_device,20,9)-1,
                    10,20,7,9,one.scratch_bytes).mode=="refused");
}
'''
        with tempfile.TemporaryDirectory(dir=ROOT / 'python/tests') as directory:
            cc = Path(directory) / 'test.cc'
            binary = Path(directory) / 'test'
            cc.write_text(source)
            subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17',
                            *shlex.split(os.environ.get('CXXFLAGS', '')),
                            '-I' + str(ROOT / 'src'), str(cc), '-o', str(binary)],
                           check=True, text=True)
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    def test_retained_host_credits_outlive_writer_and_cover_relocation(self):
        source = r'''
#include "thirdparty/SiftGPU/SiftRequirements.h"
#include <cassert>
#include <future>
int main() {
  colmap::ExtractionBudget budget(2060,0,0); // 2000 raw bytes, one 3% margin.
  auto image = budget.Acquire(100, [] { return false; });
  std::shared_ptr<colmap::ExtractionBudget::Ticket> retained;
  std::vector<float> values;
  std::shared_ptr<colmap::ExtractionBudget::Ticket> converted;
  {
    siftgpu::SiftHostBudgetScope scope([&](uint64_t bytes) {image->Grow(bytes);}, [&](uint64_t bytes) {
      return image->Fork(bytes);
    });
    siftgpu::SiftResizeHostVector(values,100,&retained);
    converted=siftgpu::SiftAcquireHost(300);
  }
  assert(budget.Used()==800);
  image.reset(); // Writer retires its image; native buffers remain live.
  assert(budget.Used()==700);
  std::promise<void> waiting;
  std::once_flag notified;
  auto reader=std::async(std::launch::async,[&] {
    return budget.Acquire(1800, [] {return false;}, [&] {
      std::call_once(notified,[&] {waiting.set_value();});
    });
  });
  assert(waiting.get_future().wait_for(std::chrono::seconds(2))==std::future_status::ready);
  {
    // The next input has a distinct image credit, not an old writer credit.
    auto next=budget.Acquire(0,[] {return false;});
    siftgpu::SiftHostBudgetScope scope([&](uint64_t bytes) {next->Grow(bytes);}, [&](uint64_t bytes) {
      return next->Fork(bytes);
    });
    bool refused=false;
    try {siftgpu::SiftResizeHostVector(values,500,&retained);}
    catch(const std::runtime_error&) {refused=true;}
    assert(refused && values.size()==100 && budget.Used()==700);
    converted.reset(); // Existing converted-input replacement/free boundary.
    siftgpu::SiftResizeHostVector(values,200,&retained);
    assert(budget.Used()==values.capacity()*sizeof(float));
    assert(budget.Peak()>=1200); // Old and new vector capacities coexisted.
  }
  std::vector<float>().swap(values); // Actual buffer is freed BEFORE its credit.
  retained.reset();
  auto admitted=reader.get();
  assert(budget.Used()==1800);
  admitted.reset();
  assert(budget.Used()==0);
}
'''
        with tempfile.TemporaryDirectory(dir=ROOT / 'python/tests') as directory:
            cc = Path(directory) / 'test.cc'
            binary = Path(directory) / 'test'
            cc.write_text(source)
            subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17', '-pthread',
                            *shlex.split(os.environ.get('CXXFLAGS', '')),
                            '-I' + str(ROOT / 'src'), str(cc), '-o', str(binary)],
                           check=True, text=True)
            subprocess.run([str(binary)], check=True, text=True, timeout=10)
