// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/optimized_photo_forecast.h"

#include "colmap/mvs/owned_pair_run.h"

#include <cassert>
int main() {
  using namespace colmap::mvs;
  const std::string digest(64, 'a');
  const auto bytes = "COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                     "\n1 2 3 4\n0 2 4 2\n1\n" + digest + " 2f61\n";
  auto wire = ParseOptimizedPhotoForecast(bytes);
  assert(wire.process.host == 1 && wire.private_charge.device == 4);
  assert(wire.files.at("/a") == digest);
  assert(wire.reference_image_idx == 0);
  assert((wire.source_image_idxs == std::vector<int>{4, 2}));
  ValidateOptimizedPhotoOrder(wire, 0, {4, 2});
  for (const auto& sources : {std::vector<int>{2, 4},
                              std::vector<int>{4},
                              std::vector<int>{4, 2, 3}}) {
    bool refused = false;
    try {
      ValidateOptimizedPhotoOrder(wire, 0, sources);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused);
  }
  bool wrong_reference = false;
  try {
    ValidateOptimizedPhotoOrder(wire, 1, {4, 2});
  } catch (const std::invalid_argument&) {
    wrong_reference = true;
  }
  assert(wrong_reference);
  for (const auto* order : {"0 2 4 4",
                            "0 2 4 0",
                            "0 2 4",
                            "0 0",
                            "00 1 4",
                            "0 1 2147483648",
                            "0 1 4 ",
                            "0 1 -1"}) {
    bool refused = false;
    try {
      ParseOptimizedPhotoForecast("COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                                  "\n1 2 3 4\n" + order + "\n1\n" + digest +
                                  " 2f61\n");
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused);
  }
  bool old_wire = false;
  try {
    ParseOptimizedPhotoForecast("COLMAP_OPTIMIZED_PHOTO_V1\n" + digest +
                                "\n1 2 3 4\n1\n" + digest + " 2f61\n");
  } catch (const std::invalid_argument&) {
    old_wire = true;
  }
  assert(old_wire);
  // The production PHOTO route uses the existing one-permit rolling owner
  // protocol, including postjoin retirement; it is not an unconstrained Run().
  owned::PairCoordinator coordinator(
      {10, 10},
      wire.process,
      size_t{1},
      std::vector<owned::ByteCharge>{wire.private_charge},
      std::chrono::milliseconds{500});
  owned::RunRollingOwners(
      coordinator,
      [&](size_t job) {
        coordinator.BeginPreparation(job);
        coordinator.ReadyAndWait(job);
        coordinator.SolverDone(job);
        coordinator.BeginOutput(job);
        coordinator.EndOutput(job);
        coordinator.AwaitOutputs(job);
      },
      [](size_t, const char*) {});
  assert(coordinator.ReservedBytes().host == 0);
  assert(coordinator.ReservedBytes().device == 0);
  for (const auto& bad : {bytes + "extra\n",
                          std::string("bad\n"),
                          "COLMAP_OPTIMIZED_PHOTO_V2\n" + digest +
                              "\n0 2 3 4\n0 2 4 2\n1\n" + digest + " 2f61\n"}) {
    bool rejected = false;
    try {
      ParseOptimizedPhotoForecast(bad);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
}
