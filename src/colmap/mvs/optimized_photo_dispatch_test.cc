// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/optimized_photo_dispatch.h"

#include <cassert>
#include <functional>

int main() {
  using namespace colmap::mvs;
  const auto rejects = [](const std::function<void()>& body) {
    bool refused = false;
    try {
      body();
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused);
  };
  // Test evidence only: deliberately below the historical private/cache floor.
  const std::string digest(64, 'a');
  const auto v2 = "COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                  "\n310000 100 200000 900000\n0 2 4 5\n1\n" + digest +
                  " 2f61\n";
  const auto bytes = "COLMAP_OPTIMIZED_PHOTO_V3\n1 1 1\n" + owned::Hex("/c0") +
                     "\n" + owned::Hex(v2) + "\n";
  const owned::ByteCharge caps{510000, 900100};
  const PhotoPhaseV1Binding evidence{"photo-phase-v1",
                                     std::string(40, 'b'),
                                     digest,
                                     digest,
                                     {310000, 100},
                                     {200000, 900000},
                                     0,
                                     {4, 5}};
  int delivered = 0;
  TrustedPhotoPhaseV1Delivery delivery = [&](const std::string& exact_bytes) {
    assert(exact_bytes == bytes);
    ++delivered;
    return std::vector<PhotoPhaseV1Binding>{evidence};
  };
  // Same adapter-aware parser and input dispatch used by RunOptimizedPhoto.
  auto dispatch = ParseOptimizedPhotoLaunch(bytes, "/c0", "1", caps, delivery);
  assert(delivered == 1 && dispatch.launch.admitted_n == 1);
  bool executed = false;
  dispatch.WithCheckedInputs(
      0, 0, {4, 5}, {300000, 30000, 20000, 100, 100, 2}, [&] {
        executed = true;
      });
  assert(executed);
  executed = false;
  rejects([&] {
    dispatch.WithCheckedInputs(
        0, 0, {4, 5}, {310001, 30000, 20000, 100, 100, 2}, [&] {
          executed = true;
        });
  });
  assert(!executed);  // Floor really ran, not merely adapter/admission.
  rejects([&] {
    dispatch.WithCheckedInputs(
        0, 0, {5, 4}, {300000, 30000, 20000, 100, 100, 2}, [] {});
  });
  rejects([&] { ParseOptimizedPhotoLaunch(bytes, "/c0", "1", caps, {}); });
  rejects([&] {
    ParseOptimizedPhotoLaunch(bytes, "/c0", "1", caps, [](const std::string&) {
      return std::vector<PhotoPhaseV1Binding>{};
    });
  });
  // Legacy dispatch without delivery must still enforce its historical floor.
  auto legacy = ParseOptimizedPhotoLaunch(v2, "/c0", "1", caps, {});
  rejects([&] {
    legacy.WithCheckedInputs(
        0, 0, {4, 5}, {300000, 30000, 20000, 100, 100, 2}, [] {});
  });
}
