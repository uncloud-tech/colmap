// SPDX-License-Identifier: BSD-3-Clause
// Standalone CPU seam: g++ -std=c++17 -pthread -Isrc optimized_dispatch_test.cc
// -o /tmp/optimized_dispatch_test
#include "colmap/mvs/optimized_dispatch.h"

#include "colmap/mvs/optimized_receipt.h"
#include "colmap/mvs/owned_pair_coordinator.h"

#include <cassert>
#include <fstream>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

int main() {
  using namespace colmap::mvs;
  auto rejects = [](const std::function<void()>& f) {
    bool rejected = false;
    try {
      f();
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  };
  // The production CLI uses this exact bounded policy before dispatch.
  ValidateOptimizedStereoConcurrency(false, "1", false, false);
  ValidateOptimizedStereoConcurrency(true, "1", true, false);
  ValidateOptimizedStereoConcurrency(true, "1", false, true);
  ValidateOptimizedStereoConcurrency(true, "2", true, false);
  rejects([] { ValidateOptimizedStereoConcurrency(false, "2", false, false); });
  rejects([] { ValidateOptimizedStereoConcurrency(false, "2", true, false); });
  rejects([] { ValidateOptimizedStereoConcurrency(true, "2", false, false); });
  rejects([] { ValidateOptimizedStereoConcurrency(true, "2", false, true); });
  rejects([] { ValidateOptimizedStereoConcurrency(true, "2", true, true); });
  for (const auto* concurrency : {"3", "4", "5"}) {
    ValidateOptimizedStereoConcurrency(true, concurrency, true, false);
    rejects([&] {
      ValidateOptimizedStereoConcurrency(false, concurrency, true, false);
    });
    rejects([&] {
      ValidateOptimizedStereoConcurrency(true, concurrency, false, false);
    });
    rejects([&] {
      ValidateOptimizedStereoConcurrency(true, concurrency, false, true);
    });
    rejects([&] {
      ValidateOptimizedStereoConcurrency(true, concurrency, true, true);
    });
  }
  for (const auto* unsupported :
       {"", "0", "6", "02", "03", "04", "05", " 3", "+4", "4 ", "-1", "2x"}) {
    rejects([&] {
      ValidateOptimizedStereoConcurrency(true, unsupported, true, false);
    });
  }
  // Exercise the existing dynamic admission seam at each extended bound. The
  // owner must complete output/retirement AND join before reusing any credit.
  for (const size_t concurrency : {3u, 4u, 5u}) {
    using namespace colmap::mvs::owned;
    const std::vector<ByteCharge> forecasts(concurrency + 1, {10, 20});
    PairCoordinator coordinator({2 + concurrency * 10, 3 + concurrency * 20},
                                {2, 3},
                                concurrency,
                                forecasts,
                                std::chrono::milliseconds(1000));
    for (size_t job = 0; job < concurrency; ++job) {
      assert(coordinator.AdmitNext() == job);
    }
    assert(coordinator.AdmitNext() == std::numeric_limits<size_t>::max());
    const auto reserved = coordinator.ReservedBytes();
    assert(reserved.host == 2 + concurrency * 10);
    assert(reserved.device == 3 + concurrency * 20);
    for (size_t job = 0; job <= concurrency; ++job) {
      coordinator.BeginPreparation(job);
      coordinator.ReadyAndWait(job);
      coordinator.SolverDone(job);
      coordinator.BeginOutput(job);
      coordinator.EndOutput(job);
      coordinator.AwaitOutputs(job);
      coordinator.Retire(job, true);
      if (job == 0) {
        // Even proven retirement is not yet a joined owner.
        assert(coordinator.AdmitNext() == std::numeric_limits<size_t>::max());
        assert(coordinator.ReservedBytes().host == reserved.host);
        assert(coordinator.ReservedBytes().device == reserved.device);
      }
      coordinator.Joined(job);
      if (job == 0) assert(coordinator.AdmitNext() == concurrency);
    }
    coordinator.Finish();
    assert(coordinator.ReservedBytes().host == 0);
    assert(coordinator.ReservedBytes().device == 0);
  }
  // Five active owners must all drain and join after cancellation. No new
  // work or private-credit reuse is allowed between cancellation and join.
  {
    using namespace colmap::mvs::owned;
    PairCoordinator coordinator({52, 103},
                                {2, 3},
                                size_t{5},
                                std::vector<ByteCharge>(6, {10, 20}),
                                std::chrono::milliseconds(1000));
    for (size_t job = 0; job < 5; ++job) {
      assert(coordinator.AdmitNext() == job);
      coordinator.BeginPreparation(job);
      coordinator.ReadyAndWait(job);
    }
    std::vector<std::thread> owners;
    for (size_t job = 0; job < 5; ++job) {
      owners.emplace_back([&, job] {
        coordinator.CancelAndAwaitCleanup(job, {}, [] {});
        coordinator.Retire(job);
      });
    }
    for (auto& owner : owners) owner.join();
    assert(coordinator.ReservedBytes().host == 52);
    assert(coordinator.ReservedBytes().device == 103);
    assert(coordinator.AdmitNext() == std::numeric_limits<size_t>::max());
    for (size_t job = 0; job < 5; ++job) {
      coordinator.Joined(job);
      assert(coordinator.ReservedBytes().host == 52 - (job + 1) * 10);
      assert(coordinator.ReservedBytes().device == 103 - (job + 1) * 20);
      assert(coordinator.AdmitNext() == std::numeric_limits<size_t>::max());
    }
    bool failed = false;
    try {
      coordinator.Finish();
    } catch (const std::runtime_error&) {
      failed = true;
    }
    assert(failed);  // Proven cleanup must not turn cancellation into success.
    assert(coordinator.ReservedBytes().host == 0);
    assert(coordinator.ReservedBytes().device == 0);
  }
  // N5 admission retains checked process-plus-private charges in both spaces.
  for (const bool host : {true, false}) {
    using namespace colmap::mvs::owned;
    bool overflow = false;
    try {
      PairCoordinator coordinator(
          {UINT64_MAX, UINT64_MAX},
          {2, 3},
          size_t{5},
          std::vector<ByteCharge>(
              5,
              host ? ByteCharge{UINT64_MAX, 20} : ByteCharge{10, UINT64_MAX}),
          std::chrono::milliseconds(1000));
    } catch (const std::overflow_error&) {
      overflow = true;
    }
    assert(overflow);
  }
  // N2 receipts name the admitted maximum, not observed physical overlap.
  const auto n2 = OptimizedDispatchReceipt("geo", 32, 7, true, false, 2);
  assert(n2.find("\"backend\":\"owned_cuda\"") != std::string::npos);
  assert(n2.find("\"max_concurrent_solvers\":2") != std::string::npos);
  assert(n2.find("\"lookahead\":false") != std::string::npos);
  rejects([] { OptimizedDispatchReceipt("photo", 32, 7, true, false, 2); });
  for (int concurrency : {3, 4, 5}) {
    assert(OptimizedDispatchReceipt("geo", 32, 7, true, false, concurrency) ==
           "{\"schema_version\":1,\"phase\":\"geo\",\"backend\":\"owned_cuda\","
           "\"tile\":32,\"cache\":true,\"lookahead\":false,\"lifetime\":"
           "\"preallocated_rotation\",\"fusion_mode\":null,"
           "\"max_concurrent_solvers\":" +
               std::to_string(concurrency) + "}\n");
    rejects([&] {
      OptimizedDispatchReceipt("photo", 32, 7, true, false, concurrency);
    });
    rejects([&] {
      OptimizedDispatchReceipt("fusion", 0, 0, false, false, concurrency);
    });
  }
  for (int unsupported : {-1, 0, 6}) {
    rejects([&] {
      OptimizedDispatchReceipt("geo", 32, 7, true, false, unsupported);
    });
  }
  // Keep the existing N1/N2 schema and bytes unchanged.
  assert(OptimizedDispatchReceipt("geo", 32, 7, true, false, 1) ==
         "{\"schema_version\":1,\"phase\":\"geo\",\"backend\":\"owned_cuda\","
         "\"tile\":32,\"cache\":true,\"lookahead\":false,\"lifetime\":"
         "\"preallocated_rotation\",\"fusion_mode\":null}\n");
  assert(n2 ==
         "{\"schema_version\":1,\"phase\":\"geo\",\"backend\":\"owned_cuda\","
         "\"tile\":32,\"cache\":true,\"lookahead\":false,\"lifetime\":"
         "\"preallocated_rotation\",\"fusion_mode\":null,"
         "\"max_concurrent_solvers\":2}\n");
  rejects([] { OptimizedDispatchReceipt("geo", 8, 7, true, false, 2); });
  rejects([] { OptimizedDispatchReceipt("geo", 32, 16, true, false, 2); });
  rejects([] { OptimizedDispatchReceipt("geo", 32, 7, false, false, 2); });
  ValidateOptimizedFusion("COLMAP", true, false, 1, -1);
  rejects([] { ValidateOptimizedFusion("COLMAP", false, false, 1, -1); });
  rejects([] { ValidateOptimizedFusion("COLMAP", true, true, 1, -1); });
  rejects([] { ValidateOptimizedFusion("COLMAP", true, false, 2, -1); });
  rejects([] { ValidateOptimizedFusion("PMVS", true, false, 1, -1); });
  ValidateOptimizedStereo(
      true, 16, 15, true, true, 5, 1, true, false, false, true);
  // Preserve the actual PHOTO-unfiltered -> GEO-filtered workflow.
  ValidateOptimizedStereo(
      true, 16, 15, true, true, 5, 1, false, false, false, false);
  rejects([] {
    ValidateOptimizedStereo(
        true, 16, 15, true, true, 5, 1, false, false, false, true);
  });
  for (int tile : {16, 32}) {
    for (int samples : {1, 7, 15}) {
      ValidateOptimizedStereo(
          true, tile, samples, true, true, 5, 1, false, false, false, false);
    }
    for (int samples : {-1, 0, 16}) {
      rejects([&] {
        ValidateOptimizedStereo(
            true, tile, samples, true, true, 5, 1, true, false, false, true);
      });
    }
  }
  rejects([] {
    ValidateOptimizedStereo(
        true, 8, 15, true, true, 5, 1, true, false, false, true);
  });
  for (int tile : {16, 32}) {
    for (int samples : {1, 15}) {
      for (bool geo : {false, true}) {
        ValidateOptimizedStereo(
            true, tile, samples, true, false, 5, 1, geo, false, false, geo);
      }
    }
  }
  // S20 is a source count; dispatch is bounded by samples, not sources.
  ValidateOptimizedStereo(
      true, 32, 15, true, true, 5, 1, true, false, false, true);
  rejects([] {
    ValidateOptimizedStereo(
        true, 16, 16, true, true, 5, 1, true, false, false, true);
  });
  rejects([] {
    ValidateOptimizedStereo(
        true, 16, 15, false, true, 5, 1, true, false, false, true);
  });
  rejects([] {
    ValidateOptimizedStereo(
        false, 16, 15, true, true, 5, 1, true, false, false, true);
  });
  assert(OptimizedDispatchReceipt("photo", 16, 15, true) ==
         "{\"schema_version\":1,\"phase\":\"photo\",\"backend\":\"owned_cuda\","
         "\"tile\":16,\"cache\":true,\"lookahead\":true,\"lifetime\":"
         "\"preallocated_rotation\",\"fusion_mode\":null}\n");
  assert(
      OptimizedDispatchReceipt("geo", 32, 15, true).find("\"phase\":\"geo\"") !=
      std::string::npos);
  for (const auto& phase : {"photo", "geo"}) {
    for (int tile : {16, 32}) {
      assert(OptimizedDispatchReceipt(phase, tile, 15, true, false)
                 .find("\"lookahead\":false") != std::string::npos);
      assert(OptimizedDispatchReceipt(phase, tile, 15, true, true)
                 .find("\"lookahead\":true") != std::string::npos);
    }
  }
  assert(OptimizedDispatchReceipt("fusion") ==
         "{\"schema_version\":1,\"phase\":\"fusion\",\"backend\":\"cpu\","
         "\"tile\":null,\"cache\":false,\"lookahead\":null,\"lifetime\":"
         "\"mapped_inputs\",\"fusion_mode\":\"mmap\"}\n");
  assert(OptimizedDispatchReceipt("photo", 32, 7, true).find("\"tile\":32") !=
         std::string::npos);
  rejects([] { OptimizedDispatchReceipt("photo", 8, 15, true); });
  rejects([] { OptimizedDispatchReceipt("geo", 32, 16, true); });
  rejects([] { OptimizedDispatchReceipt("photo", 32, 15, false); });
  rejects([] { OptimizedDispatchReceipt("unknown"); });
#ifndef _WIN32
  const auto path = std::filesystem::path("optimized-dispatch-receipt-" +
                                          std::to_string(getpid()) + ".json");
  ValidateOptimizedReceiptPath(path);
  PublishOptimizedReceipt(path,
                          OptimizedDispatchReceipt("photo", 16, 15, true));
  std::ifstream input(path);
  const std::string actual((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
  assert(actual == OptimizedDispatchReceipt("photo", 16, 15, true));
  rejects([&] {
    PublishOptimizedReceipt(path, OptimizedDispatchReceipt("fusion"));
  });
  std::filesystem::remove(path);
#endif
}
