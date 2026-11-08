// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <optional>
#include <ostream>
#include <string_view>

namespace colmap::mvs::owned {
// Target-local metadata mode. No test initialization, CUDA calls, filesystem
// reads, or runtime Git lookup. nullopt means ordinary GTest argument handling.
inline std::optional<int> StripTestIdentity(int argc,
                                            const char* const* argv,
                                            std::string_view commit,
                                            std::string_view fingerprint,
                                            std::ostream& output,
                                            std::ostream& error) {
  constexpr std::string_view flag = "--stereo-test-identity";
  if (argc == 2 && argv[1] == flag) {
    const auto hex = [](std::string_view value, size_t length) {
      return value.size() == length &&
             std::all_of(value.begin(), value.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
             });
    };
    if (!hex(commit, 40) || !hex(fingerprint, 64)) {
      error << "compiled strip harness identity unavailable\n";
      return 1;
    }
    output << "{\"schema_version\":1,\"source_commit\":\"" << commit
           << "\",\"allocation_fingerprint\":\"" << fingerprint << "\"}\n";
    return output ? 0 : 1;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument.rfind("--gtest_", 0) == 0 ||
        argument.rfind("--gmock_", 0) == 0 || argument == "--help" ||
        argument == "-h" || argument == "-?")
      continue;
    error << "unexpected strip harness argument: " << argument << '\n';
    return 1;
  }
  return std::nullopt;
}
}  // namespace colmap::mvs::owned
