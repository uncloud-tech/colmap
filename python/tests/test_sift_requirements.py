# SPDX-License-Identifier: BSD-3-Clause
"CPU-native public SIFT sequence admission contract; no CUDA execution claim."

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class SiftRequirementsTest(unittest.TestCase):
    def test_sequence_selection_and_actual_growth(self):
        source = """
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
  try { budget.Replace(0,1); } catch(const std::runtime_error&) \
{refused=true;}
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
    SiftHostBudgetScope scope([](uint64_t) { throw std::runtime_error("host \
no fit"); });
    bool host_refused=false;
    try { SiftResizeHostVector(values,100); }
    catch(const std::runtime_error&) { host_refused=true; }
    assert(host_refused && values.capacity()==old_capacity && \
values.size()==10);
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
"""
        with tempfile.TemporaryDirectory(
            dir=ROOT / "python/tests"
        ) as directory:
            cc = Path(directory) / "test.cc"
            binary = Path(directory) / "test"
            cc.write_text(source)
            subprocess.run(
                [
                    os.environ.get("CXX", "g++"),
                    "-std=c++17",
                    *shlex.split(os.environ.get("CXXFLAGS", "")),
                    "-I" + str(ROOT / "src"),
                    str(cc),
                    "-o",
                    str(binary),
                ],
                check=True,
                text=True,
            )
            subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True
            )

    def test_retained_host_credits_outlive_writer_and_cover_relocation(self):
        source = """
#include "thirdparty/SiftGPU/SiftRequirements.h"
#include <cassert>
#include <future>
int main() {
  colmap::ExtractionBudget budget(2060,0,0); // 2000 raw bytes, one 3% \
margin.
  auto image = budget.Acquire(100, [] { return false; });
  std::shared_ptr<colmap::ExtractionBudget::Ticket> retained;
  std::vector<float> values;
  std::shared_ptr<colmap::ExtractionBudget::Ticket> converted;
  {
    siftgpu::SiftHostBudgetScope scope([&](uint64_t bytes) \
{image->Grow(bytes);}, [&](uint64_t bytes) {
      return image->Fork(bytes);
    });
    siftgpu::SiftResizeHostVector(values,100,&retained);
    converted=siftgpu::SiftAcquireHost(300);
  }
  assert(budget.Used()==800);
  image.reset(); // Writer retires its image; native buffers remain live.
  assert(budget.Used()==700);
  auto reader=std::async(std::launch::async,[&] {
    try {budget.Acquire(1800, [] {return false;});return false;}
    catch(const std::runtime_error& e) {
      return std::string(e.what()).find("persistent")!=std::string::npos;
    }
  });
  \
assert(reader.wait_for(std::chrono::seconds(2))==std::future_status::ready);
  assert(reader.get());
  {
    // The next input has a distinct image credit, not an old writer credit.
    auto next=budget.Acquire(0,[] {return false;});
    siftgpu::SiftHostBudgetScope scope([&](uint64_t bytes) \
{next->Grow(bytes);}, [&](uint64_t bytes) {
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
  std::vector<float>().swap(values); // Actual buffer is freed BEFORE its \
credit.
  retained.reset();
  assert(budget.Used()==0);
}
"""
        with tempfile.TemporaryDirectory(
            dir=ROOT / "python/tests"
        ) as directory:
            cc = Path(directory) / "test.cc"
            binary = Path(directory) / "test"
            cc.write_text(source)
            subprocess.run(
                [
                    os.environ.get("CXX", "g++"),
                    "-std=c++17",
                    "-pthread",
                    *shlex.split(os.environ.get("CXXFLAGS", "")),
                    "-I" + str(ROOT / "src"),
                    str(cc),
                    "-o",
                    str(binary),
                ],
                check=True,
                text=True,
            )
            subprocess.run([str(binary)], check=True, text=True, timeout=10)

    def test_reader_refuses_nondrainable_payload_and_rechecks_waiters(self):
        source = """
#include "thirdparty/SiftGPU/SiftRequirements.h"
#include <future>
#include <string>
using namespace std::chrono_literals;
using Credit=std::shared_ptr<colmap::ExtractionBudget::Ticket>;
// Same ownership order as the native extractor: actual buffer dies before \
credit.
struct ExtractorPayload {
  Credit credit;
  std::vector<float> buffer;
  void Allocate(const Credit& image,int floats) {
    siftgpu::SiftHostBudgetScope scope({},[&](uint64_t bytes) {return \
image->Fork(bytes);});
    siftgpu::SiftResizeHostVector(buffer,floats,&credit);
  }
};
int main() {
  {
    colmap::ExtractionBudget budget(1030,0,0);
    auto writer=budget.Acquire(100,[] {return false;});
    ExtractorPayload extractor;
    extractor.Allocate(writer,175); // 700 bytes remain owned between \
extractions.
    writer.reset();
    auto reader=std::async(std::launch::async,[&] {
      try {budget.Acquire(400,[] {return false;});return false;}
      catch(const std::runtime_error& e) {
        return std::string(e.what()).find("persistent")!=std::string::npos;
      }
    });
    const bool ready=reader.wait_for(2s)==std::future_status::ready;
    if(!ready) budget.Cancel(); // Bound a broken implementation, not its \
owners.
    const bool refused=reader.get();
    if(!ready || !refused || budget.Used()!=700 || \
extractor.buffer.size()!=175) return 21;
    auto exact=budget.Acquire(300,[] {return false;}); // \
ceil((700+300)*1.03)==1030
    if(budget.Used()!=1000) return 29;
    bool boundary_refused=false;
    try {budget.Acquire(301,[] {return false;});}
    catch(const std::runtime_error& e) {
      \
boundary_refused=std::string(e.what()).find("persistent")!=std::string::npos;
    }
    if(!boundary_refused) return 30;
  } // No manual owner release is needed to unblock admission.
  {
    colmap::ExtractionBudget budget(1030,0,0);
    auto writer=budget.Acquire(400,[] {return false;});
    auto retained=writer->Fork(100);
    std::promise<void> waiting;std::once_flag notified;
    auto reader=std::async(std::launch::async,[&] {
      try {
        budget.Acquire(600,[] {return false;},[&] {
          std::call_once(notified,[&] {waiting.set_value();});
        });return false;
      } catch(const std::runtime_error& e) {
        return std::string(e.what()).find("persistent")!=std::string::npos;
      }
    });
    if(waiting.get_future().wait_for(2s)!=std::future_status::ready) \
{budget.Cancel();reader.get();return 22;}
    retained->Grow(500); // Existing extraction grows while reader already \
waits.
    const bool ready=reader.wait_for(2s)==std::future_status::ready;
    if(!ready) budget.Cancel();
    if(!reader.get() || !ready || budget.Used()!=1000) return 23;
  }
  {
    colmap::ExtractionBudget budget(1030,0,0);
    auto writer=budget.Acquire(400,[] {return false;});
    auto retained=writer->Fork(100);
    std::promise<void> waiting;std::once_flag notified;
    auto reader=std::async(std::launch::async,[&] {
      return budget.Acquire(600,[] {return false;},[&] {
        std::call_once(notified,[&] {waiting.set_value();});
      });
    });
    if(waiting.get_future().wait_for(2s)!=std::future_status::ready) \
{budget.Cancel();return 24;}
    writer.reset(); // Drainable writer retires; persistent owner stays live.
    if(reader.wait_for(2s)!=std::future_status::ready) \
{budget.Cancel();return 25;}
    auto image=reader.get();
    if(budget.Used()!=700) return 26;
  }
  {
    colmap::ExtractionBudget budget(1030,0,0);
    auto writer=budget.Acquire(400,[] {return false;});
    auto retained=writer->Fork(100);
    std::promise<void> waiting;std::once_flag notified;
    auto reader=std::async(std::launch::async,[&] {
      try {
        budget.Acquire(600,[] {return false;},[&] {
          std::call_once(notified,[&] {waiting.set_value();});
        });return false;
      } catch(const std::runtime_error& e) {
        return std::string(e.what()).find("cancelled")!=std::string::npos;
      }
    });
    if(waiting.get_future().wait_for(2s)!=std::future_status::ready) \
{budget.Cancel();reader.get();return 27;}
    budget.Cancel();
    if(reader.wait_for(2s)!=std::future_status::ready || !reader.get() || \
budget.Used()!=500) return 28;
  }
}
"""
        with tempfile.TemporaryDirectory(
            dir=ROOT / "python/tests"
        ) as directory:
            cc = Path(directory) / "test.cc"
            binary = Path(directory) / "test"
            cc.write_text(source)
            subprocess.run(
                [
                    os.environ.get("CXX", "g++"),
                    "-std=c++17",
                    "-pthread",
                    *shlex.split(os.environ.get("CXXFLAGS", "")),
                    "-I" + str(ROOT / "src"),
                    str(cc),
                    "-o",
                    str(binary),
                ],
                check=True,
                text=True,
            )
            subprocess.run([str(binary)], check=True, text=True, timeout=15)

    def test_streamed_working_credit_retires_before_previous_writer(self):
        source = r"""
#include "colmap/feature/sift_streaming.h"
#include "thirdparty/SiftGPU/SiftRequirements.h"
#include <future>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;
using Ticket = std::shared_ptr<colmap::ExtractionBudget::Ticket>;
auto Scratch(const Ticket& image, uint64_t staging, uint64_t metadata) {
  return colmap::ReserveSiftStreamingScratch(
      staging, metadata,
      [&](uint64_t bytes) { return image->Fork(bytes); },
      [&](uint64_t bytes) { image->Grow(bytes); });
}
int main() {
  // Recorded limits: reserve payload without multi-GB physical allocations.
  colmap::ExtractionBudget budget(7398268928, 2147483648, 536870912);
  const uint64_t stage = 4512840863, metadata = 266000;
  auto writer = budget.Acquire(
      colmap::ExtractionBitmapPayload(264,129,1,264,false),
      [] {return false;});
  auto second = budget.Acquire(
      colmap::ExtractionBitmapPayload(256,129,1,264,false),
      [] {return false;});
  // Actual retained native vector owner shares the ledger but not image life.
  Ticket retained;
  std::vector<float> persistent;
  {
    siftgpu::SiftHostBudgetScope scope({},[&](uint64_t bytes) {
      return writer->Fork(bytes);
    });
    siftgpu::SiftResizeHostVector(persistent,256,&retained);
  }
  std::vector<uint8_t> public_result(24*152);
  writer->Grow(public_result.size());
  const auto live_outputs = budget.Used();
  {
    auto credit = Scratch(writer, stage, metadata);
    std::vector<float> conversion_row(264);
    if(budget.Used()!=live_outputs+stage+metadata) return 20;
  } // All streamed working owners end; writer/result/retained owners stay live.
  try {
    auto credit = Scratch(second, stage, metadata);
    std::vector<float> conversion_row(256);
    if(budget.Used()!=live_outputs+stage+metadata) return 21;
  } catch(const std::exception& e) {
    std::cerr << "second-image reservation: " << e.what() << '\n';
    return 22;
  }
  if(budget.Used()!=live_outputs || persistent.size()!=256) return 23;
  {
    auto credit = Scratch(second, stage, metadata);
    bool refused=false;
    try { auto concurrent = Scratch(writer, stage, metadata); }
    catch(const std::runtime_error&) {refused=true;}
    if(!refused || budget.Used()!=live_outputs+stage+metadata) return 24;
  }
  try {
    auto credit = Scratch(second, stage, metadata);
    std::vector<float> conversion_row(256);
    throw std::runtime_error("failure after allocating working owners");
  } catch(const std::runtime_error&) {}
  if(budget.Used()!=live_outputs) return 25;
  // Temporary scratch must be drainable while extraction is still running.
  colmap::ExtractionBudget small(2060,0,0);
  auto image = small.Acquire(100,[] {return false;});
  std::promise<void> waiting;
  std::once_flag notified;
  std::future<Ticket> reader;
  {
    auto credit = Scratch(image,1500,0);
    std::vector<float> working(100);
    reader=std::async(std::launch::async,[&] {
      return small.Acquire(600,[] {return false;},[&] {
        std::call_once(notified,[&] {waiting.set_value();});
      });
    });
    if(waiting.get_future().wait_for(2s)!=std::future_status::ready) {
      small.Cancel();try {reader.get();}catch(...){}return 26;
    }
  }
  if(reader.wait_for(2s)!=std::future_status::ready) {
    small.Cancel();try {reader.get();}catch(...){}return 27;
  }
  auto admitted=reader.get();
  if(small.Used()!=700) return 28; // Writer is intentionally still alive.
}
"""
        with tempfile.TemporaryDirectory(
            dir=ROOT / "python/tests"
        ) as directory:
            cc = Path(directory) / "test.cc"
            binary = Path(directory) / "test"
            cc.write_text(source)
            subprocess.run(
                [
                    os.environ.get("CXX", "g++"),
                    "-std=c++17",
                    "-pthread",
                    *shlex.split(os.environ.get("CXXFLAGS", "")),
                    "-I" + str(ROOT / "src"),
                    str(cc),
                    "-o",
                    str(binary),
                ],
                check=True,
                text=True,
            )
            subprocess.run([str(binary)], check=True, text=True, timeout=15)
