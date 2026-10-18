// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_validation.h"

#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace colmap::mvs {
struct OptimizedPhotoForecast {
  std::string options_sha256;
  owned::ByteCharge process;
  owned::ByteCharge private_charge;
  int reference_image_idx = -1;
  std::vector<int> source_image_idxs;
  std::map<std::string, std::string> files;
};
inline bool OptimizedDigest(const std::string& s) {
  return s.size() == 64 &&
         s.find_first_not_of("0123456789abcdef") == std::string::npos;
}
inline OptimizedPhotoForecast ParseOptimizedPhotoForecast(
    const std::string& bytes) {
  auto fail = [] {
    throw std::invalid_argument("invalid optimized PHOTO forecast v2");
  };
  std::istringstream input(bytes);
  std::string line;
  auto next = [&] {
    if (!std::getline(input, line) || input.eof()) fail();
  };
  next();
  if (line != "COLMAP_OPTIMIZED_PHOTO_V2") fail();
  OptimizedPhotoForecast result;
  next();
  result.options_sha256 = line;
  if (!OptimizedDigest(line)) fail();
  next();
  std::istringstream charges(line);
  std::string h, d, ph, pd, extra;
  if (!(charges >> h >> d >> ph >> pd) || charges >> extra) fail();
  if (h + " " + d + " " + ph + " " + pd != line) fail();
  auto number = [&](const std::string& s) {
    if (s.empty() || s[0] == '0') fail();
    return owned::ParseBytes(s);
  };
  result.process = {number(h), number(d)};
  result.private_charge = {number(ph), number(pd)};
  next();
  std::istringstream order(line);
  std::string ref_token, count_token;
  if (!(order >> ref_token >> count_token)) fail();
  const auto index = [&](const std::string& token) {
    const uint64_t value = token == "0" ? 0 : number(token);
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) fail();
    return static_cast<int>(value);
  };
  result.reference_image_idx = index(ref_token);
  const uint64_t source_count = number(count_token);
  if (source_count > line.size() / 2) fail();
  std::set<int> seen{result.reference_image_idx};
  std::string canonical_order = ref_token + " " + count_token;
  for (uint64_t i = 0; i < source_count; ++i) {
    std::string token;
    if (!(order >> token)) fail();
    const int source = index(token);
    if (!seen.insert(source).second) fail();
    result.source_image_idxs.push_back(source);
    canonical_order += " " + token;
  }
  if (order >> extra || canonical_order != line) fail();
  next();
  const uint64_t count = number(line);
  // Bounded by actual wire bytes; never reserve based on an untrusted count.
  if (count > bytes.size() / 67) fail();
  for (uint64_t i = 0; i < count; ++i) {
    next();
    const auto split = line.find(' ');
    if (split != 64 || !OptimizedDigest(line.substr(0, split))) fail();
    const auto hex = line.substr(split + 1);
    if (hex.empty() || hex.size() % 2 ||
        hex.find_first_not_of("0123456789abcdef") != std::string::npos)
      fail();
    std::string path;
    for (size_t j = 0; j < hex.size(); j += 2) {
      const auto c =
          static_cast<char>(std::stoi(hex.substr(j, 2), nullptr, 16));
      if (!c) fail();
      path.push_back(c);
    }
    const std::filesystem::path p(path);
    if (!p.is_absolute() || p.lexically_normal().string() != path ||
        !result.files.emplace(path, line.substr(0, split)).second)
      fail();
  }
  if (std::getline(input, line)) fail();
  return result;
}
// Never sort or compare membership only: native effective order is part of
// approval. Model/config bytes and global indices are bound by this same wire.
inline void ValidateOptimizedPhotoOrder(
    const OptimizedPhotoForecast& forecast,
    int reference_image_idx,
    const std::vector<int>& source_image_idxs) {
  if (forecast.reference_image_idx < 0 || forecast.source_image_idxs.empty() ||
      forecast.reference_image_idx != reference_image_idx ||
      forecast.source_image_idxs != source_image_idxs) {
    throw std::invalid_argument(
        "PHOTO effective reference/source order mismatch; reapproval required");
  }
}
}  // namespace colmap::mvs
