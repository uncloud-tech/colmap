// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/reconstruction_input_summary.h"
#include "colmap/mvs/stock_prng_state.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace colmap {

int RunPhotoDirectorySummary(int argc, char** argv);

// Compiled-only build identity serialized into the photo directory summary
// JSON contract (see docs/designs/photo-cli-interface.md, V1 shape). The
// `revision` is expected to come from the compiled accessor
// GetCommitRevision(); an empty revision serializes as JSON null. The
// stock_memory_fingerprint field is always null in V1 and honestly states
// that calibration qualification is unavailable; the serializer never
// invents a digest or registry entry for it.
struct PhotoDirectorySummaryJsonBuild {
  std::string revision;
  std::string version;
  bool cuda_compiled = false;
  // Unavailable unless supplied by the CUDA+MVS compile-time accessor. This
  // logical layout is not a memory-model fingerprint or physical-fit claim.
  std::optional<mvs::StockPrngStateLayout> stock_prng_state;
};

// Appends the given text to `json` as one quoted JSON string byte for byte:
// double quotes, backslashes and the required control characters are
// escaped, and other control characters (< 0x20) use the \u00XX form. Bytes
// >= 0x20, including multi-byte UTF-8 sequences, pass through unchanged.
// Kept as an internal helper only because paths reaching the V1 photo summary
// contract can contain characters that require escaping; the function itself
// is dependency-free so that the emitted document is testable without
// linking the CLI dependencies.
inline void AppendQuotedJsonString(std::string& json, std::string_view text) {
  static constexpr char kHexDigits[] = "0123456789abcdef";
  json.push_back('"');
  for (const char c : text) {
    switch (c) {
      case '"':
        json += "\\\"";
        break;
      case '\\':
        json += "\\\\";
        break;
      case '\b':
        json += "\\b";
        break;
      case '\f':
        json += "\\f";
        break;
      case '\n':
        json += "\\n";
        break;
      case '\r':
        json += "\\r";
        break;
      case '\t':
        json += "\\t";
        break;
      default: {
        const auto uc = static_cast<unsigned char>(c);
        if (uc < 0x20) {
          json += "\\u00";
          json.push_back(kHexDigits[(uc >> 4) & 0xF]);
          json.push_back(kHexDigits[uc & 0xF]);
        } else {
          json.push_back(c);
        }
      }
    }
  }
  json.push_back('"');
}

// Serializes a photo directory summary into the frozen V1 JSON contract as
// one compact single-line document. All scalars are serialized as actual
// JSON types (true/false/null/decimal integers) and empty vectors as empty
// arrays; nothing goes through PropertyTree's stringified scalar
// approximations. `image_path` is the directory exactly as given on the
// command line. The caller owns the `build` identity; the summary's problem
// kinds are mapped to their frozen `corrupt_image`/`unknown_dimensions`
// strings and each receives a fixed explanatory message, since the accepted
// collector itself carries no per-problem message.
inline std::string WritePhotoDirectorySummaryJson(
    const PhotoDirectorySummary& summary,
    const std::string_view image_path,
    const PhotoDirectorySummaryJsonBuild& build) {
  static constexpr int kSchemaVersion = 1;
  std::string json;
  json.reserve(1024);

  json += "{\"schema_version\":";
  json += std::to_string(kSchemaVersion);
  json += ",\"image_path\":";
  AppendQuotedJsonString(json, image_path);
  json += ",\"photo_count\":";
  json += std::to_string(summary.photos.size());
  json += ",\"photos\":[";
  for (size_t i = 0; i < summary.photos.size(); ++i) {
    if (i != 0) {
      json += ',';
    }
    const PhotoScanEntry& photo = summary.photos[i];
    json += "{\"path\":";
    AppendQuotedJsonString(json, photo.relative_path);
    json += ",\"width\":";
    json += std::to_string(photo.encoded_width);
    json += ",\"height\":";
    json += std::to_string(photo.encoded_height);
    json += ",\"orientation\":";
    if (photo.exif_orientation.has_value()) {
      json += std::to_string(*photo.exif_orientation);
    } else {
      json += "null";
    }
    json += '}';
  }
  json += "],\"largest_photo\":";
  if (summary.largest_photo.has_value()) {
    json += "{\"path\":";
    AppendQuotedJsonString(json, summary.largest_photo->relative_path);
    json += ",\"width\":";
    json += std::to_string(summary.largest_photo->encoded_width);
    json += ",\"height\":";
    json += std::to_string(summary.largest_photo->encoded_height);
    json += '}';
  } else {
    json += "null";
  }
  json += ",\"dimension_envelope\":";
  if (summary.dimension_envelope.IsValid()) {
    json += "{\"max_width\":";
    json += std::to_string(summary.dimension_envelope.max_encoded_width);
    json += ",\"max_height\":";
    json += std::to_string(summary.dimension_envelope.max_encoded_height);
    json += '}';
  } else {
    json += "null";
  }
  json += ",\"problems\":[";
  for (size_t i = 0; i < summary.problems.size(); ++i) {
    if (i != 0) {
      json += ',';
    }
    const PhotoScanProblem& problem = summary.problems[i];
    json += "{\"path\":";
    AppendQuotedJsonString(json, problem.relative_path);
    if (problem.kind == PhotoScanProblem::Kind::kCorruptImage) {
      json +=
          ",\"kind\":\"corrupt_image\",\"message\":\"Image header could "
          "not be parsed; the file appears corrupt or its format is "
          "unavailable.\"}";
    } else if (problem.kind == PhotoScanProblem::Kind::kUnknownDimensions) {
      json +=
          ",\"kind\":\"unknown_dimensions\",\"message\":\"Image header "
          "reports missing or invalid encoded dimensions.\"}";
    }
  }
  json += "],\"non_image_files\":[";
  for (size_t i = 0; i < summary.non_image_files.size(); ++i) {
    if (i != 0) {
      json += ',';
    }
    AppendQuotedJsonString(json, summary.non_image_files[i].relative_path);
  }
  json += "],\"build\":{\"revision\":";
  if (build.revision.empty()) {
    json += "null";
  } else {
    AppendQuotedJsonString(json, build.revision);
  }
  json += ",\"version\":";
  AppendQuotedJsonString(json, build.version);
  json += ",\"cuda_compiled\":";
  json += build.cuda_compiled ? "true" : "false";
  json += ",\"stock_prng_state\":";
  if (build.cuda_compiled && build.stock_prng_state.has_value() &&
      build.stock_prng_state->bytes > 0 &&
      build.stock_prng_state->alignment > 0) {
    json += "{\"bytes\":";
    json += std::to_string(build.stock_prng_state->bytes);
    json += ",\"alignment\":";
    json += std::to_string(build.stock_prng_state->alignment);
    json +=
        ",\"type\":\"curandState\","
        "\"provenance\":\"cuda_translation_unit_compile_time\"}";
  } else {
    json += "null";
  }
  json += ",\"stock_memory_fingerprint\":null";
  json +=
      "},\"fit_claim\":false,\"header_only\":true,\"orientation_applied\":"
      "false}";
  return json;
}

}  // namespace colmap