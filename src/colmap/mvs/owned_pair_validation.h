// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace colmap::mvs::owned {

inline uint64_t CheckedAdd(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a) {
    throw std::overflow_error("owned pair byte addition overflow");
  }
  return a + b;
}
inline uint64_t CheckedMultiply(uint64_t a, uint64_t b) {
  if (a && b > std::numeric_limits<uint64_t>::max() / a) {
    throw std::overflow_error("owned pair byte multiplication overflow");
  }
  return a * b;
}
inline uint64_t ParseBytes(const std::string& text) {
  if (text.empty()) throw std::invalid_argument("empty unsigned byte value");
  uint64_t value = 0;
  for (const unsigned char c : text) {
    if (c < '0' || c > '9') {
      throw std::invalid_argument("unsigned values require ASCII digits only");
    }
    value = CheckedAdd(CheckedMultiply(value, 10), c - '0');
  }
  return value;
}
inline uint64_t CacheBytes(double gib) {
  const long double bytes = static_cast<long double>(gib) * 1073741824.0L;
  if (!std::isfinite(gib) || bytes < 1 ||
      bytes >= static_cast<long double>(std::numeric_limits<size_t>::max())) {
    throw std::invalid_argument("invalid or unrepresentable cache byte cap");
  }
  return static_cast<uint64_t>(bytes);
}
inline void ValidateIo(int a, int b, uint64_t cap, uint64_t affinity) {
  if (a <= 0 || b <= 0 || !cap || cap > affinity ||
      static_cast<uint64_t>(a) > cap || static_cast<uint64_t>(b) > cap) {
    throw std::invalid_argument("explicit I/O workers/cap exceed CPU affinity");
  }
}
inline bool PathContains(const std::filesystem::path& root,
                         const std::filesystem::path& path) {
  auto it = path.begin();
  for (const auto& component : root) {
    if (it == path.end() || *it != component) return false;
    ++it;
  }
  return true;
}
inline bool PathsOverlap(const std::filesystem::path& a,
                         const std::filesystem::path& b) {
  auto ai = a.begin();
  auto bi = b.begin();
  while (ai != a.end() && bi != b.end()) {
    if (*ai != *bi) return false;
    ++ai;
    ++bi;
  }
  return true;
}
inline bool PathsAlias(const std::filesystem::path& a,
                       const std::filesystem::path& b) {
  return PathsOverlap(a, b) ||
         (std::filesystem::exists(a) && std::filesystem::exists(b) &&
          std::filesystem::equivalent(a, b));
}
inline std::string Hex(const std::string& value) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() * 2);
  for (unsigned char c : value) {
    result.push_back(digits[c >> 4]);
    result.push_back(digits[c & 15]);
  }
  return result;
}
// Exact long names only. No abbreviations, duplicate options, positional args,
// implicit overrides, --key=value variants or sign/whitespace integer syntax.
inline std::map<std::string, std::string> ParseStrictArguments(
    int argc, const char* const* argv, const std::set<std::string>& allowed) {
  std::map<std::string, std::string> values;
  if (argc % 2 != 1) throw std::invalid_argument("expected --key value pairs");
  for (int i = 1; i < argc; i += 2) {
    const std::string arg = argv[i];
    if (arg.size() < 3 || arg.substr(0, 2) != "--") {
      throw std::invalid_argument("expected exact long option");
    }
    const std::string key = arg.substr(2);
    if (!allowed.count(key) || !values.emplace(key, argv[i + 1]).second) {
      throw std::invalid_argument("unknown or duplicate option: " + key);
    }
  }
  for (const auto& key : allowed) {
    if (!values.count(key) || values.at(key).empty()) {
      throw std::invalid_argument("missing required option: " + key);
    }
  }
  return values;
}

}  // namespace colmap::mvs::owned
