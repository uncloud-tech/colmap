// SPDX-License-Identifier: BSD-3-Clause
// CPU-only real native parser probe; no CUDA execution/qualification.
#include "colmap/mvs/native_run_deadline.h"
#include "colmap/mvs/optimized_photo_forecast.h"

#include <fstream>
#include <iostream>
#include <iterator>

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "timeout") {
      colmap::mvs::ResolveOwnedRunDeadline(
          argv[2], true, true, colmap::mvs::NativeRunDeadline::Clock::now());
    } else if (argc == 6 && std::string(argv[1]) == "wire") {
      std::ifstream stream(argv[2]);
      const std::string bytes((std::istreambuf_iterator<char>(stream)), {});
      auto launch = colmap::mvs::ParseOptimizedPhotoLaunch(
          bytes, "/unused", argv[3], {std::stoull(argv[4]), std::stoull(argv[5])});
      auto bound = colmap::mvs::AdmitBoundOptimizedPhotoLaunch(
          bytes, "/unused", argv[3], {std::stoull(argv[4]), std::stoull(argv[5])}, launch.bindings);
      std::cout << bound.admitted_n << " " << bound.jobs.size() << " " << bound.bindings.size();
    } else {
      return 2;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what();
    return 1;
  }
}
