// SPDX-License-Identifier: BSD-3-Clause
// Standalone CPU seam: g++ -std=c++17 -Isrc optimized_dispatch_test.cc -o
// /tmp/optimized_dispatch_test
#include "colmap/mvs/optimized_dispatch.h"

#include "colmap/mvs/optimized_receipt.h"

#include <cassert>
#include <fstream>
#include <functional>

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
