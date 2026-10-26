// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/optimized_photo_prepared.h"

#include <cassert>
#include <memory>
#include <string>

struct Engine {
  std::string options = "effective PHOTO options";
  const std::string& GetOptions() const { return options; }
};
struct Prepared {
  const void* prepared_by = nullptr;
  size_t prepared_job = 0;
  std::unique_ptr<Engine> patch_match;
};
int main() {
  int controller, other;
  auto prepared = std::make_unique<Prepared>();
  const auto check = [&] {
    colmap::mvs::CheckOptimizedPhotoPreparedJob(
        prepared,
        &controller,
        3,
        "effective PHOTO options",
        [](const auto& options) { return options; });
  };
  const auto refuse = [&](const char* name) {
    try {
      check();
      assert(false);
    } catch (const std::invalid_argument& e) {
      assert(e.what() == std::string(name));
    }
  };
  refuse("PHOTO prepared engine missing");
  prepared->patch_match = std::make_unique<Engine>();
  prepared->prepared_by = &other;
  prepared->prepared_job = 3;
  refuse("PHOTO prepared job provenance mismatch");
  prepared->prepared_by = &controller;
  prepared->prepared_job = 2;
  refuse("PHOTO prepared job provenance mismatch");
  prepared->prepared_job = 3;
  check();
  prepared->patch_match->options = "controller options, not effective options";
  refuse("PHOTO prepared effective options mismatch");
  prepared->patch_match->options = "effective GEO options";
  refuse("PHOTO prepared effective options mismatch");
  prepared.reset();
  refuse("PHOTO prepared engine missing");
}
