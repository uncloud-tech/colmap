// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/mvs/optimized_dispatch.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_validation.h"

#include <limits>
#include <map>
#include <optional>
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
struct PhotoPhaseV1Binding {
  std::string profile;
  std::string lineage_sha;
  std::string fingerprint;
  std::string options_sha256;
  owned::ByteCharge process;
  owned::ByteCharge private_charge;
  int reference_image_idx = -1;
  std::vector<int> source_image_idxs;
  // Exact committed closure revision, distinct from the required ancestry base.
  std::string source_commit;
};
// Each section is the typed object itself, not an encoded secondary payload.
inline PhotoPhaseV1Binding ParsePhotoPhaseV1WireBinding(std::istream& input) {
  const auto fail = [](const std::string& field) {
    throw std::invalid_argument("PHOTO malformed binding section: " + field);
  };
  const std::set<std::string> keys = {"profile",
                                      "lineage_sha",
                                      "fingerprint",
                                      "options_sha256",
                                      "process",
                                      "private_charge",
                                      "reference_image_idx",
                                      "source_image_idxs",
                                      "source_commit"};
  std::map<std::string, std::string> values;
  std::string line;
  bool ended = false;
  while (std::getline(input, line)) {
    if (input.eof()) fail("unterminated");
    if (line == "[END]") {
      ended = true;
      break;
    }
    const auto split = line.find('=');
    if (split == std::string::npos) fail("expected key=value");
    const auto key = line.substr(0, split);
    if (!keys.count(key)) fail("unknown key " + key);
    if (!values.emplace(key, line.substr(split + 1)).second)
      fail("duplicate key " + key);
  }
  if (!ended) fail("missing [END]");
  for (const auto& key : keys)
    if (!values.count(key) || values.at(key).empty()) fail("missing " + key);
  const auto numbers =
      [&](const std::string& key, size_t count, uint64_t maximum) {
        std::istringstream fields(values.at(key));
        std::vector<uint64_t> result;
        std::string token, canonical;
        while (fields >> token) {
          uint64_t value = 0;
          try {
            value = owned::ParseBytes(token);
          } catch (const std::exception&) {
            fail(key);
          }
          if (token != std::to_string(value) || value > maximum) fail(key);
          if (!canonical.empty()) canonical += " ";
          canonical += token;
          result.push_back(value);
        }
        if (canonical != values.at(key) || result.empty() ||
            (count && result.size() != count))
          fail(key);
        return result;
      };
  PhotoPhaseV1Binding b;
  b.profile = values.at("profile");
  b.lineage_sha = values.at("lineage_sha");
  b.fingerprint = values.at("fingerprint");
  b.options_sha256 = values.at("options_sha256");
  b.source_commit = values.at("source_commit");
  const auto max = std::numeric_limits<uint64_t>::max();
  const auto process = numbers("process", 2, max);
  const auto private_charge = numbers("private_charge", 2, max);
  b.process = {process[0], process[1]};
  b.private_charge = {private_charge[0], private_charge[1]};
  b.reference_image_idx = static_cast<int>(
      numbers("reference_image_idx", 1, std::numeric_limits<int>::max())[0]);
  for (const auto index :
       numbers("source_image_idxs", 0, std::numeric_limits<int>::max()))
    b.source_image_idxs.push_back(static_cast<int>(index));
  return b;
}
// V3 is an explicit launch envelope: `k rolling_active_jobs job_count`, then
// pairs of hex-encoded absolute config paths and complete V2 forecast bytes,
// optionally followed by one named binding section per job in launch order.
// V2 remains the single-job default, with its historical options/hash binding.
struct OptimizedPhotoLaunch {
  struct Job {
    std::filesystem::path config;
    OptimizedPhotoForecast forecast;
  };
  int admitted_n = 1;
  std::vector<Job> jobs;
  std::vector<PhotoPhaseV1Binding> bindings;
};
inline OptimizedPhotoLaunch ParseOptimizedPhotoLaunch(
    const std::string& bytes,
    const std::filesystem::path& config,
    const std::string& requested,
    owned::ByteCharge capacity) {
  OptimizedPhotoLaunch result;
  OptimizedConcurrencyLimits limits;
  limits.concurrency = requested;
  limits.photo_ceiling = 1;
  if (bytes.rfind("COLMAP_OPTIMIZED_PHOTO_V3\n", 0) != 0) {
    result.jobs.push_back({config, ParseOptimizedPhotoForecast(bytes)});
  } else {
    std::istringstream input(bytes);
    std::string line;
    std::getline(input, line);
    if (!std::getline(input, line))
      throw std::invalid_argument("missing PHOTO V3 envelope");
    std::istringstream fields(line);
    std::string k, active, count, extra;
    if (!(fields >> k >> active >> count) || fields >> extra ||
        line != k + " " + active + " " + count || active != requested)
      throw std::invalid_argument("PHOTO rolling_active_jobs must equal CLI N");
    limits.photo_ceiling = ParseOptimizedConcurrency(k);
    limits.full_bank_forecast = true;
    const auto jobs = owned::ParseBytes(count);
    if (!jobs || count != std::to_string(jobs) || jobs > bytes.size() / 4)
      throw std::invalid_argument("invalid PHOTO V3 job count");
    const auto decode = [&]() {
      if (!std::getline(input, line) || line.empty() || line.size() % 2 ||
          line.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::invalid_argument("invalid PHOTO V3 hex field");
      std::string value;
      for (size_t i = 0; i < line.size(); i += 2)
        value.push_back(
            static_cast<char>(std::stoi(line.substr(i, 2), nullptr, 16)));
      return value;
    };
    std::set<int> references;
    for (uint64_t i = 0; i < jobs; ++i) {
      const auto path = decode();
      if (path.find('\0') != std::string::npos ||
          !std::filesystem::path(path).is_absolute() ||
          std::filesystem::path(path).lexically_normal().string() != path)
        throw std::invalid_argument("invalid PHOTO V3 config path");
      auto forecast = ParseOptimizedPhotoForecast(decode());
      if (!references.insert(forecast.reference_image_idx).second)
        throw std::invalid_argument("duplicate PHOTO V3 reference");
      result.jobs.push_back({path, std::move(forecast)});
    }
    while (std::getline(input, line)) {
      if (line != "[BINDING]")
        throw std::invalid_argument("trailing PHOTO V3 bytes");
      result.bindings.push_back(ParsePhotoPhaseV1WireBinding(input));
    }
    if (!result.bindings.empty() && result.bindings.size() != jobs)
      throw std::invalid_argument("PHOTO binding section count mismatch");
  }
  limits.job_count = result.jobs.size();
  result.admitted_n = AdmitOptimizedConcurrency(limits);
  const auto process = result.jobs.front().forecast.process;
  std::vector<uint64_t> host, device;
  for (const auto& job : result.jobs) {
    if (job.forecast.process.host != process.host ||
        job.forecast.process.device != process.device)
      throw std::invalid_argument("PHOTO V3 process charge mismatch");
    host.push_back(job.forecast.private_charge.host);
    device.push_back(job.forecast.private_charge.device);
  }
  std::sort(host.rbegin(), host.rend());
  std::sort(device.rbegin(), device.rend());
  auto total = process;
  for (int i = 0; i < result.admitted_n; ++i) {
    total.host = owned::CheckedAdd(total.host, host.at(i));
    total.device = owned::CheckedAdd(total.device, device.at(i));
  }
  if (!owned::PairCoordinator::Fits(total, capacity))
    throw std::invalid_argument("PHOTO top-N private charges exceed caps");
  return result;
}
// Production admission is separate from the synthetic wire qualification seam.
// Only the trusted photo-phase-v1 delivery adapter may attest this evidence;
// it is not an authorization token and does not replace input/security checks.
inline void ValidatePhotoPhaseV1Binding(
    const PhotoPhaseV1Binding& b, const OptimizedPhotoForecast& forecast) {
  if (b.profile != "photo-phase-v1" || b.lineage_sha.size() != 40 ||
      b.lineage_sha.find_first_not_of("0123456789abcdef") !=
          std::string::npos ||
      !OptimizedDigest(b.fingerprint) || !OptimizedDigest(b.options_sha256) ||
      b.options_sha256 != forecast.options_sha256 ||
      b.process.host != forecast.process.host ||
      b.process.device != forecast.process.device ||
      b.private_charge.host != forecast.private_charge.host ||
      b.private_charge.device != forecast.private_charge.device ||
      b.reference_image_idx != forecast.reference_image_idx ||
      b.source_image_idxs != forecast.source_image_idxs)
    throw std::invalid_argument("PHOTO photo-phase-v1 binding mismatch");
}
inline OptimizedPhotoLaunch AdmitBoundOptimizedPhotoLaunch(
    const std::string& bytes,
    const std::filesystem::path& config,
    const std::string& requested,
    owned::ByteCharge capacity,
    const std::vector<PhotoPhaseV1Binding>& bindings = {}) {
  if (bytes.rfind("COLMAP_OPTIMIZED_PHOTO_V3\n", 0) == 0 && bindings.empty())
    throw std::invalid_argument(
        "PHOTO V3 requires delivered photo-phase-v1 binding; forecast "
        "unavailable");
  auto launch = ParseOptimizedPhotoLaunch(bytes, config, requested, capacity);
  if (!bindings.empty()) {
    if (bindings.size() != launch.jobs.size())
      throw std::invalid_argument(
          "PHOTO photo-phase-v1 binding count mismatch");
    for (size_t i = 0; i < bindings.size(); ++i)
      ValidatePhotoPhaseV1Binding(bindings[i], launch.jobs[i].forecast);
  }
  return launch;
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
