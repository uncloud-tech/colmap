// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/resident_contract.h"

#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(COLMAP_CUDA_ENABLED)
constexpr bool kBuildCudaEnabled = true;
#else
constexpr bool kBuildCudaEnabled = false;
#endif
#if defined(COLMAP_HIP_ENABLED)
constexpr bool kBuildHipEnabled = true;
#else
constexpr bool kBuildHipEnabled = false;
#endif

namespace colmap {
namespace mvs {
namespace {

// ---------------------------------------------------------------------------
// Canonical value encoding. Every string/path is a decimal byte length and
// lowercase hex bytes. Every float/double is its exact IEEE-754 bit pattern
// in lowercase hex. Integers are decimal. Lines are '\n' terminated with
// single spaces between tokens.
// ---------------------------------------------------------------------------

void AppendLine(std::string* out, const std::string& line) {
  out->append(line);
  out->push_back('\n');
}

std::string HexEncode(const std::string& bytes) {
  static constexpr char kHexDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (const char byte : bytes) {
    out.push_back(kHexDigits[static_cast<unsigned char>(byte) >> 4]);
    out.push_back(kHexDigits[static_cast<unsigned char>(byte) & 0x0F]);
  }
  return out;
}

void AppendString(std::string* out,
                  const std::string& key,
                  const std::string& value) {
  AppendLine(out,
             StringPrintf("%s %zu %s",
                          key.c_str(),
                          value.size(),
                          HexEncode(value).c_str()));
}

// Strict hex decode; returns false so the parser can produce line context.
// Canonical-lowercase decode only: the serializer never emits uppercase
// hex, so the parser rejects it outright and parse implies canonical form
// (in addition to the CLI canonical-byte equality check).
bool HexDecode(const std::string& hex, std::string* out) {
  if (hex.size() % 2 != 0) return false;
  const auto digit = [](const char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  out->clear();
  out->reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    const int hi = digit(hex[i]);
    const int lo = digit(hex[i + 1]);
    if (hi < 0 || lo < 0) return false;
    out->push_back(static_cast<char>((static_cast<unsigned char>(hi) << 4) |
                                     static_cast<unsigned char>(lo)));
  }
  return true;
}

std::string EncodeDouble(const double value) {
  // Exact IEEE-754 bit pattern as a conventional numeric hex string, e.g.
  // 1.5 -> "3ff8000000000000", independent of stream formatting or locale.
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return StringPrintf("%016llx",
                      static_cast<unsigned long long>(bits));
}

std::string EncodeFloat(const float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return StringPrintf("%08x", bits);
}

double DecodeDouble(const std::string& hex) {
  THROW_CHECK_EQ(hex.size(), 16)
      << "Invalid double bit representation: " << hex;
  uint64_t bits = 0;
  const auto result = std::from_chars(
      hex.data(), hex.data() + hex.size(), bits, 16);
  THROW_CHECK(result.ec == std::errc() && result.ptr == hex.data() + hex.size())
      << "Invalid double bit representation: " << hex;
  double value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float DecodeFloat(const std::string& hex) {
  THROW_CHECK_EQ(hex.size(), 8)
      << "Invalid float bit representation: " << hex;
  uint32_t bits = 0;
  const auto result = std::from_chars(
      hex.data(), hex.data() + hex.size(), bits, 16);
  THROW_CHECK(result.ec == std::errc() && result.ptr == hex.data() + hex.size())
      << "Invalid float bit representation: " << hex;
  float value = 0;
  std::memcpy(&value, &bits, sizeof(bits));
  return value;
}

const char* EncodeBool(const bool value) { return value ? "1" : "0"; }

// Fixed, exhaustive, version-bound field order. Adding a PatchMatchOptions
// field requires a contract version bump; this list must stay exact.
std::string SerializePatchMatchOptions(const PatchMatchOptions& options) {
  std::string out;
  const auto integer = [&out](const char* key, const int value) {
    AppendLine(&out, StringPrintf("opt_int %s %d", key, value));
  };
  const auto boolean = [&out](const char* key, const bool value) {
    AppendLine(&out, StringPrintf("opt_bool %s %s", key, EncodeBool(value)));
  };
  const auto real = [&out](const char* key, const double value) {
    AppendLine(&out,
               StringPrintf("opt_double %s %s",
                            key,
                            EncodeDouble(value).c_str()));
  };
  const auto string = [&out](const char* key, const std::string& value) {
    AppendLine(&out,
               StringPrintf("opt_string %s %zu %s",
                            key,
                            value.size(),
                            HexEncode(value).c_str()));
  };

  integer("max_image_size", options.max_image_size);
  string("gpu_index", options.gpu_index);
  real("depth_min", options.depth_min);
  real("depth_max", options.depth_max);
  integer("random_seed", options.random_seed);
  integer("window_radius", options.window_radius);
  integer("window_step", options.window_step);
  real("sigma_spatial", options.sigma_spatial);
  real("sigma_color", options.sigma_color);
  integer("num_samples", options.num_samples);
  boolean("owned_distinct_source_cache", options.owned_distinct_source_cache);
  boolean("owned_tex_lookahead", options.owned_tex_lookahead);
  real("ncc_sigma", options.ncc_sigma);
  real("min_triangulation_angle", options.min_triangulation_angle);
  real("incident_angle_sigma", options.incident_angle_sigma);
  integer("num_iterations", options.num_iterations);
  boolean("geom_consistency", options.geom_consistency);
  real("geom_consistency_regularizer", options.geom_consistency_regularizer);
  real("geom_consistency_max_cost", options.geom_consistency_max_cost);
  boolean("filter", options.filter);
  real("filter_min_ncc", options.filter_min_ncc);
  real("filter_min_triangulation_angle",
       options.filter_min_triangulation_angle);
  integer("filter_min_num_consistent", options.filter_min_num_consistent);
  real("filter_geom_consistency_max_cost",
       options.filter_geom_consistency_max_cost);
  boolean("write_consistency_graph", options.write_consistency_graph);
  boolean("allow_missing_files", options.allow_missing_files);
  integer("num_threads", options.num_threads);
  real("cache_size", options.cache_size);
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Float map header reader (metadata only). Mirrors Mat<float>'s text header:
// "width&height&depth&" followed by a binary payload that is NOT read here.
// ---------------------------------------------------------------------------

namespace {

struct FloatMapHeader {
  size_t width = 0;
  size_t height = 0;
  size_t depth = 0;
};

uintmax_t CheckedFloatMapPayloadBytes(const size_t width,
                                     const size_t height,
                                     const size_t depth,
                                     const std::filesystem::path& path) {
  THROW_CHECK_LE(width, std::numeric_limits<size_t>::max() / height)
      << "Float map dims overflow: " << path;
  const size_t area = width * height;
  THROW_CHECK_LE(area, std::numeric_limits<size_t>::max() / depth)
      << "Float map dims overflow: " << path;
  const size_t count = area * depth;
  THROW_CHECK_LE(count, std::numeric_limits<size_t>::max() / sizeof(float))
      << "Float map payload overflow: " << path;
  return static_cast<uintmax_t>(count * sizeof(float));
}

// Mirrors the PRODUCTION shape handling exactly (DepthMap/NormalMap Downsize
// plus the owned bank's dimension checks): float factors, std::round rounding
// (std::round(float) -> float), and downsize only when max_image_size > 0.
// Maps with both dims within the target are left untouched by Downsize's
// early return and must already match the effective image dims exactly.
bool MapShapeAdmitted(const std::filesystem::path& path,
                      const size_t map_width,
                      const size_t map_height,
                      const size_t image_width,
                      const size_t image_height,
                      const int max_image_size) {
  if (map_width == image_width && map_height == image_height) return true;
  if (map_width <= image_width && map_height <= image_height) {
    // Downsize early-returns; the bank then requires exact equality.
    return false;
  }
  if (max_image_size <= 0) {
    // The bank performs no float-map downsize without a positive effective
    // max_image_size; the map must already equal the image dims.
    return false;
  }
  const float factor_x =
      static_cast<float>(image_width) / static_cast<float>(map_width);
  const float factor_y =
      static_cast<float>(image_height) / static_cast<float>(map_height);
  const float factor = std::min(factor_x, factor_y);
  const size_t expected_width =
      static_cast<size_t>(std::round(static_cast<float>(map_width) * factor));
  const size_t expected_height =
      static_cast<size_t>(std::round(static_cast<float>(map_height) * factor));
  if (expected_width != image_width || expected_height != image_height) {
    LOG(WARNING) << "Float map header shape mismatch: " << path;
    return false;
  }
  return true;
}

// Reads and validates ONLY the float-map text header ("W&H&D&"), mirroring
// Mat<float>'s format exactly, with no payload reads. Validates the three
// separators, positive dims, the channel count binding (depth maps carry 1
// channel, normal maps carry 3), the production-exact shape handling above
// (float factors and std::round, downsize only with max_image_size > 0), and
// the checked payload extent against the actual file.
FloatMapHeader ReadFloatMapHeader(const std::filesystem::path& path,
                                  const size_t image_width,
                                  const size_t image_height,
                                  const int max_image_size,
                                  const size_t expected_channels) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  size_t width = 0, height = 0, depth = 0;
  char separator1, separator2, separator3;
  // No payload reads: extraction stops right after the third '&' separator.
  file >> width >> separator1 >> height >> separator2 >> depth >> separator3;
  THROW_CHECK(file) << "Invalid float map header: " << path;
  THROW_CHECK_EQ(separator1, '&')
      << "Invalid float map header separator: " << path;
  THROW_CHECK_EQ(separator2, '&')
      << "Invalid float map header separator: " << path;
  THROW_CHECK_EQ(separator3, '&')
      << "Invalid float map header separator: " << path;
  THROW_CHECK_EQ(depth, expected_channels)
      << "Unsupported float map channel count: " << path;
  THROW_CHECK_GT(width, 0) << path;
  THROW_CHECK_GT(height, 0) << path;
  // Production-exact shape admission, header-only (see MapShapeAdmitted).
  THROW_CHECK(MapShapeAdmitted(path,
                               width,
                               height,
                               image_width,
                               image_height,
                               max_image_size))
      << "Float map header does not match the production shape handling for "
         "the effective image dims: "
      << path;
  // Checked extent without a payload read: header text plus the declared
  // payload must fit within the actual file size. Extra trailing bytes stay
  // accepted, consistent with the production Mat reader.
  const auto payload_start = file.tellg();
  THROW_CHECK(payload_start != std::streampos(-1)) << path;
  file.seekg(0, std::ios::end);
  const auto file_end = file.tellg();
  THROW_CHECK(file_end != std::streampos(-1) && file_end >= payload_start)
      << path;
  const uintmax_t header_bytes = static_cast<uintmax_t>(payload_start);
  const uintmax_t file_size = static_cast<uintmax_t>(file_end);
  const uintmax_t payload_bytes =
      CheckedFloatMapPayloadBytes(width, height, depth, path);
  THROW_CHECK_GE(file_size - header_bytes, payload_bytes)
      << "Truncated float map extent: " << path;
  file.close();
  return {width, height, depth};
}

// Strict line scanner: '\n'-terminated lines, exactly-splitted tokens.
class ContractLexer {
 public:
  explicit ContractLexer(const std::string& bytes) : bytes_(bytes) {}

  // Splits the next line into single-space-separated tokens. Empty lines are
  // represented by one empty token and rejected by the grammar. Returns
  // false at end of input.
  bool Next() {
    if (pos_ >= bytes_.size()) {
      tokens_.clear();
      line_.clear();
      return false;
    }
    const size_t eol = bytes_.find('\n', pos_);
    if (eol == std::string::npos) {
      line_.clear();
      tokens_.clear();
      tokens_.push_back("unterminated");
      line_no_ += 1;
      return true;
    }
    line_ = bytes_.substr(pos_, eol - pos_);
    pos_ = eol + 1;
    line_no_ += 1;
    SplitTokens();
    return true;
  }

  // Marks an unterminated final line as the parse error it is.
  bool AtUnterminatedLine() const { return Key() == "unterminated"; }

  // Bytes not yet consumed; used to bound declared record counts by the
  // bytes that could actually contain their records before any reserve().
  size_t RemainingBytes() const { return bytes_.size() - pos_; }

  // Rejects declared record counts that cannot possibly be backed by the
  // remaining bytes (each record needs at least `min_record_bytes`). This
  // fails closed before any huge reserve() on truncated or hostile bytes.
  size_t BoundCount(const std::string& key,
                    const size_t count,
                    const size_t min_record_bytes) const {
    if (count > RemainingBytes() / min_record_bytes) {
      Fail("declared " + key + " count exceeds the remaining contract size");
    }
    return count;
  }

  bool AtEnd() const { return pos_ != 0 && pos_ >= bytes_.size(); }
  size_t LineNo() const { return line_no_; }
  size_t TokenCount() const { return tokens_.size(); }
  const std::string& Token(const size_t index) const {
    if (index >= tokens_.size()) {
      ThrowParseError("missing token");
    }
    return tokens_[index];
  }
  const std::string& LineText() const { return line_; }
  const std::string& Key() const {
    return tokens_.empty() ? empty_ : tokens_.front();
  }

  void Fail(const std::string& message) const {
    ThrowParseError(message);
  }

 private:
  void ThrowParseError(const std::string& message) const {
    throw std::runtime_error(StringPrintf(
        "Malformed owned bank contract at line %zu (%s): %s",
        line_no_,
        line_.c_str(),
        message.c_str()));
  }

  void SplitTokens() {
    tokens_.clear();
    size_t cursor = 0;
    while (true) {
      const size_t end = line_.find(' ', cursor);
      tokens_.push_back(line_.substr(
          cursor, end == std::string::npos ? end : end - cursor));
      if (end == std::string::npos) break;
      cursor = end + 1;
    }
    if (tokens_.size() == 1 && tokens_.front().empty()) {
      tokens_.push_back("empty");
    }
  }

  const std::string& bytes_;
  size_t pos_ = 0;
  size_t line_no_ = 0;
  std::vector<std::string> tokens_;
  std::string line_;
  std::string empty_;
};

// Canonical signed decimal: optional '-', digits only, no '+', no leading
// zeros (the serializer emits exactly this spelling).
int64_t ParseInt64(const ContractLexer& lexer,
                   const std::string& token,
                   const std::string& key) {
  if (token.empty() || token == "-" ||
      token.find('+') != std::string::npos ||
      (token.size() > 1 && token.front() == '0') ||
      (token.size() > 2 && token[0] == '-' && token[1] == '0')) {
    lexer.Fail("non-canonical integer spelling for " + key);
  }
  int64_t value = 0;
  const auto result =
      std::from_chars(token.data(), token.data() + token.size(), value, 10);
  if (result.ec != std::errc() ||
      result.ptr != token.data() + token.size()) {
    lexer.Fail("invalid integer for " + key);
  }
  return value;
}

int ParseInt32(const ContractLexer& lexer,
               const std::string& token,
               const std::string& key) {
  const int64_t value = ParseInt64(lexer, token, key);
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    lexer.Fail("integer out of range for " + key);
  }
  return static_cast<int>(value);
}

// Canonical unsigned decimal: digits only, no '+', no '-', no leading zeros
// (the serializer emits exactly this spelling). Parses into the destination
// unsigned width directly with NO signed intermediate, so every value the
// serializer can emit round-trips.
uint64_t ParseUint64(const ContractLexer& lexer,
                     const std::string& token,
                     const std::string& key) {
  if (token.empty() || token.front() == '+' || token.front() == '-' ||
      (token.size() > 1 && token.front() == '0')) {
    lexer.Fail("non-canonical unsigned spelling for " + key);
  }
  uint64_t value = 0;
  const auto result =
      std::from_chars(token.data(), token.data() + token.size(), value, 10);
  if (result.ec != std::errc() ||
      result.ptr != token.data() + token.size()) {
    lexer.Fail("invalid unsigned integer for " + key);
  }
  return value;
}

uintmax_t ParseUintMax(const ContractLexer& lexer,
                       const std::string& token,
                       const std::string& key) {
  const uint64_t value = ParseUint64(lexer, token, key);
  THROW_CHECK_LE(value, std::numeric_limits<uintmax_t>::max())
      << "Unsigned value out of range for " << key;
  return static_cast<uintmax_t>(value);
}

size_t ParseUnsigned(const ContractLexer& lexer,
                     const std::string& token,
                     const std::string& key,
                     const bool positive_only) {
  const uint64_t value = ParseUint64(lexer, token, key);
  THROW_CHECK_LE(value, std::numeric_limits<size_t>::max())
      << "Unsigned value out of size_t range for " << key;
  if (positive_only && value == 0) {
    lexer.Fail("expected positive value for " + key);
  }
  return static_cast<size_t>(value);
}

// Positive-only unsigned count (non-zero).
size_t ParseSize(const ContractLexer& lexer,
                 const std::string& token,
                 const std::string& key) {
  return ParseUnsigned(lexer, token, key, true);
}

std::string ParseBool(const ContractLexer& lexer,
                      const std::string& token,
                      const std::string& key) {
  if (token != "0" && token != "1") {
    lexer.Fail("invalid boolean for " + key);
  }
  return token;
}

void ParseBoolInto(const ContractLexer& lexer,
                   const std::string& token,
                   const std::string& key,
                   bool* out) {
  *out = ParseBool(lexer, token, key) == "1";
}

// Length-prefixed hex value at the given token index; following tokens on
// the line are not checked (callers validate record lengths).
std::string DecodeHexValue(const ContractLexer& lexer,
                           const std::string& key,
                           const size_t token_index) {
  const size_t length = ParseUnsigned(
      lexer, lexer.Token(token_index), key + " length", false);
  if (length == 0) {
    lexer.Fail("empty string for " + key);
  }
  if (length > std::numeric_limits<std::string::size_type>::max() / 2) {
    lexer.Fail("string length overflow for " + key);
  }
  if (lexer.TokenCount() < token_index + 2) {
    lexer.Fail("malformed string encoding for " + key);
  }
  const std::string& hex = lexer.Token(token_index + 1);
  if (hex.size() != length * 2) {
    lexer.Fail("hex length does not match declared length for " + key);
  }
  std::string decoded;
  if (!HexDecode(hex, &decoded)) {
    lexer.Fail("invalid hex encoding for " + key);
  }
  return decoded;
}

// String value that must be the WHOLE remaining line.
std::string ParseHexString(const ContractLexer& lexer,
                           const std::string& key,
                           const size_t token_index) {
  if (lexer.TokenCount() != token_index + 2) {
    lexer.Fail("malformed string encoding for " + key);
  }
  return DecodeHexValue(lexer, key, token_index);
}

void ParsePathInto(const ContractLexer& lexer,
                   const std::string& key,
                   const size_t token_index,
                   std::filesystem::path* out) {
  const std::string value = ParseHexString(lexer, key, token_index);
  if (value.empty()) {
    lexer.Fail("empty path for " + key);
  }
  *out = std::filesystem::path(value);
}

void ParseIdentityInto(const ContractLexer& lexer,
                       const std::string& key,
                       const size_t token_index,
                       owned::BankFileIdentity* identity) {
  const std::string path = DecodeHexValue(lexer, key, token_index);
  if (path.empty()) {
    lexer.Fail("empty identity path for " + key);
  }
  if (lexer.TokenCount() != token_index + 9) {
    lexer.Fail("identity record field count for " + key);
  }
  identity->path = std::filesystem::path(path);
  identity->size = ParseUintMax(
      lexer, lexer.Token(token_index + 2), key + " size");
  identity->device = ParseUintMax(
      lexer, lexer.Token(token_index + 3), key + " device");
  identity->inode = ParseUintMax(
      lexer, lexer.Token(token_index + 4), key + " inode");
  identity->modified_seconds = ParseInt64(
      lexer, lexer.Token(token_index + 5), key + " modified seconds");
  identity->modified_nanoseconds = ParseInt64(
      lexer, lexer.Token(token_index + 6), key + " modified nanoseconds");
  identity->changed_seconds = ParseInt64(
      lexer, lexer.Token(token_index + 7), key + " changed seconds");
  identity->changed_nanoseconds = ParseInt64(
      lexer, lexer.Token(token_index + 8), key + " changed nanoseconds");
}

void ParseOptionsBlock(ContractLexer& lexer, PatchMatchOptions* options) {
  int fields_seen = 0;
  const auto expect = [&lexer, &fields_seen](const std::string& type,
                                             const std::string& key) {
    if (!lexer.Next()) lexer.Fail("unexpected end of contract options block");
    if (lexer.AtUnterminatedLine()) lexer.Fail("unterminated final line");
    if (lexer.TokenCount() < 2 || lexer.Token(0) != type ||
        lexer.Token(1) != key) {
      lexer.Fail("expected " + type + " " + key + ", got \"" +
                 lexer.LineText() + "\"");
    }
    fields_seen += 1;
  };
  const auto integer = [&](const char* key, int* value) {
    expect("opt_int", key);
    if (lexer.TokenCount() != 3) lexer.Fail("opt_int field count");
    *value = ParseInt32(lexer, lexer.Token(2), key);
  };
  const auto boolean = [&](const char* key, bool* value) {
    expect("opt_bool", key);
    if (lexer.TokenCount() != 3) lexer.Fail("opt_bool field count");
    ParseBoolInto(lexer, lexer.Token(2), key, value);
  };
  const auto real = [&](const char* key, double* value) {
    expect("opt_double", key);
    if (lexer.TokenCount() != 3) lexer.Fail("opt_double field count");
    *value = DecodeDouble(lexer.Token(2));
  };
  const auto string = [&](const char* key, std::string* value) {
    expect("opt_string", key);
    if (lexer.TokenCount() != 4) lexer.Fail("opt_string field count");
    *value = ParseHexString(lexer, key, 2);
  };

  integer("max_image_size", &options->max_image_size);
  string("gpu_index", &options->gpu_index);
  real("depth_min", &options->depth_min);
  real("depth_max", &options->depth_max);
  integer("random_seed", &options->random_seed);
  integer("window_radius", &options->window_radius);
  integer("window_step", &options->window_step);
  real("sigma_spatial", &options->sigma_spatial);
  real("sigma_color", &options->sigma_color);
  integer("num_samples", &options->num_samples);
  boolean("owned_distinct_source_cache", &options->owned_distinct_source_cache);
  boolean("owned_tex_lookahead", &options->owned_tex_lookahead);
  real("ncc_sigma", &options->ncc_sigma);
  real("min_triangulation_angle", &options->min_triangulation_angle);
  real("incident_angle_sigma", &options->incident_angle_sigma);
  integer("num_iterations", &options->num_iterations);
  boolean("geom_consistency", &options->geom_consistency);
  real("geom_consistency_regularizer", &options->geom_consistency_regularizer);
  real("geom_consistency_max_cost", &options->geom_consistency_max_cost);
  boolean("filter", &options->filter);
  real("filter_min_ncc", &options->filter_min_ncc);
  real("filter_min_triangulation_angle",
       &options->filter_min_triangulation_angle);
  integer("filter_min_num_consistent", &options->filter_min_num_consistent);
  real("filter_geom_consistency_max_cost",
       &options->filter_geom_consistency_max_cost);
  boolean("write_consistency_graph", &options->write_consistency_graph);
  boolean("allow_missing_files", &options->allow_missing_files);
  integer("num_threads", &options->num_threads);
  real("cache_size", &options->cache_size);
  THROW_CHECK_EQ(fields_seen, 28);
}

}  // namespace

OwnedBankContract MakeOwnedBankContract(
    const PatchMatchOptions& invocation_options,
    const std::string& stereo_folder) {
  OwnedBankContract contract;
  contract.version = kOwnedBankContractVersion;
  contract.build_cuda_enabled = kBuildCudaEnabled;
  contract.build_hip_enabled = kBuildHipEnabled;
  // Honest unqualified statement only; a generated allocation fingerprint is
  // a separate future lane and is never fabricated here.
  contract.allocation_compatibility = "unqualified-cpu-only";
  contract.invocation_options = invocation_options;
  contract.stereo_folder = stereo_folder;
  return contract;
}

OwnedBankContractProblem CaptureOwnedBankContractProblem(
    const owned::BankJob& job,
    const std::vector<std::string>& image_names) {
  THROW_CHECK_LT(job.reference_image_idx, static_cast<int>(image_names.size()));
  THROW_CHECK(!job.inputs.empty());
  OwnedBankContractProblem problem;
  problem.reference_name = image_names.at(job.reference_image_idx);
  problem.reference_image_idx = job.reference_image_idx;
  problem.effective_options = job.options;
  problem.output_depth_path = job.depth_map_path;
  problem.output_normal_path = job.normal_map_path;
  problem.inputs.reserve(job.inputs.size());
  for (const auto& input : job.inputs) {
    OwnedBankContractInput record;
    record.is_reference = input.image_idx == job.reference_image_idx;
    record.image_idx = input.image_idx;
    record.image_name = image_names.at(input.image_idx);
    record.width = input.image.GetWidth();
    record.height = input.image.GetHeight();
    std::memcpy(record.K.data(), input.image.GetK(), 9 * sizeof(float));
    std::memcpy(record.R.data(), input.image.GetR(), 9 * sizeof(float));
    std::memcpy(record.T.data(), input.image.GetT(), 3 * sizeof(float));
    record.max_image_size = input.max_image_size;
    record.image_path = input.image.GetPath();
    record.depth_path = input.depth_path;
    record.normal_path = input.normal_path;
    record.image_identity = owned::CaptureFileIdentity(record.image_path);
    record.depth_identity = owned::CaptureFileIdentity(record.depth_path);
    record.normal_identity = owned::CaptureFileIdentity(record.normal_path);
    // Header-only reads with layout/extent/shape validation. Depth maps
    // carry 1 channel, normal maps carry 3; shape handling mirrors the
    // production DepthMap/NormalMap Downsize exactly (float factors,
    // std::round, downsize only with max_image_size > 0, exact original
    // dims otherwise), against the effective image dims captured above.
    const auto depth_header = ReadFloatMapHeader(record.depth_path,
                                                 record.width,
                                                 record.height,
                                                 input.max_image_size,
                                                 /*expected_channels=*/1);
    record.depth_header_width = depth_header.width;
    record.depth_header_height = depth_header.height;
    record.depth_header_depth = depth_header.depth;
    const auto normal_header = ReadFloatMapHeader(record.normal_path,
                                                  record.width,
                                                  record.height,
                                                  input.max_image_size,
                                                  /*expected_channels=*/3);
    record.normal_header_width = normal_header.width;
    record.normal_header_height = normal_header.height;
    record.normal_header_depth = normal_header.depth;
    // The invocation snapshot must describe ONE stable file state: after all
    // header reads, every captured identity is revalidated against the
    // current file state. A change within the capture window fails closed.
    owned::CheckFileIdentityUnchanged(record.image_identity);
    owned::CheckFileIdentityUnchanged(record.depth_identity);
    owned::CheckFileIdentityUnchanged(record.normal_identity);
    problem.inputs.push_back(std::move(record));
  }
  THROW_CHECK_EQ(std::count_if(problem.inputs.begin(),
                               problem.inputs.end(),
                               [](const OwnedBankContractInput& input) {
                                 return input.is_reference;
                               }),
                 1)
      << "Ordered inputs must contain the reference image exactly once";
  return problem;
}

std::string SerializeOwnedBankContract(const OwnedBankContract& contract) {
  THROW_CHECK_EQ(contract.version, kOwnedBankContractVersion);
  THROW_CHECK(!contract.problems.empty())
      << "Contract must capture at least one problem";
  THROW_CHECK(!contract.allocation_compatibility.empty());
  THROW_CHECK(!contract.invocation_options.gpu_index.empty());
  THROW_CHECK(!contract.stereo_folder.empty());
  std::string out;
  AppendLine(&out, "owned-bank-contract");
  AppendLine(&out, StringPrintf("version %d", contract.version));
  AppendLine(
      &out,
      StringPrintf("build_cuda_enabled %s",
                   EncodeBool(contract.build_cuda_enabled)));
  AppendLine(
      &out,
      StringPrintf("build_hip_enabled %s",
                   EncodeBool(contract.build_hip_enabled)));
  AppendString(&out,
               "allocation_compatibility",
               contract.allocation_compatibility);
  out.append("invocation_options\n");
  out.append(SerializePatchMatchOptions(contract.invocation_options));
  AppendString(&out, "stereo_folder", contract.stereo_folder);
  AppendLine(&out, StringPrintf("problem_count %zu", contract.problems.size()));
  for (size_t i = 0; i < contract.problems.size(); ++i) {
    const auto& problem = contract.problems[i];
    THROW_CHECK(!problem.reference_name.empty());
    THROW_CHECK(!problem.inputs.empty());
    AppendLine(&out, StringPrintf("problem %zu", i));
    AppendLine(
        &out,
        StringPrintf("reference_image_idx %d", problem.reference_image_idx));
    AppendString(&out, "reference_name", problem.reference_name);
    AppendString(&out, "output_depth_path", problem.output_depth_path.string());
    AppendString(
        &out, "output_normal_path", problem.output_normal_path.string());
    out.append("problem_options\n");
    out.append(SerializePatchMatchOptions(problem.effective_options));
    AppendLine(&out, StringPrintf("input_count %zu", problem.inputs.size()));
    for (size_t k = 0; k < problem.inputs.size(); ++k) {
      const auto& input = problem.inputs[k];
      THROW_CHECK(!input.image_name.empty());
      AppendLine(&out, StringPrintf("input %zu", k));
      AppendLine(
          &out,
          StringPrintf("is_reference %s", EncodeBool(input.is_reference)));
      AppendLine(&out, StringPrintf("image_idx %d", input.image_idx));
      AppendString(&out, "image_name", input.image_name);
      AppendLine(
          &out,
          StringPrintf("image_dims %zu %zu", input.width, input.height));
      AppendLine(&out,
                 StringPrintf("max_image_size %d", input.max_image_size));
      std::string parts;
      for (const float value : input.K) {
        parts += " " + EncodeFloat(value);
      }
      AppendLine(&out, "calib_k" + parts);
      parts.clear();
      for (const float value : input.R) {
        parts += " " + EncodeFloat(value);
      }
      AppendLine(&out, "calib_r" + parts);
      parts.clear();
      for (const float value : input.T) {
        parts += " " + EncodeFloat(value);
      }
      AppendLine(&out, "calib_t" + parts);
      AppendString(&out, "image_path", input.image_path.string());
      const auto identity =
          [&out](const char* key, const owned::BankFileIdentity& identity) {
            out.append(StringPrintf(
                "%s %zu %s %ju %ju %ju %" PRIi64 " %" PRIi64 " %" PRIi64
                " %" PRIi64 "\n",
                key,
                identity.path.string().size(),
                HexEncode(identity.path.string()).c_str(),
                identity.size,
                identity.device,
                identity.inode,
                identity.modified_seconds,
                identity.modified_nanoseconds,
                identity.changed_seconds,
                identity.changed_nanoseconds));
          };
      identity("image_identity", input.image_identity);
      AppendString(&out, "depth_path", input.depth_path.string());
      identity("depth_identity", input.depth_identity);
      AppendLine(&out,
                 StringPrintf("depth_header %zu %zu %zu",
                              input.depth_header_width,
                              input.depth_header_height,
                              input.depth_header_depth));
      AppendString(&out, "normal_path", input.normal_path.string());
      identity("normal_identity", input.normal_identity);
      AppendLine(&out,
                 StringPrintf("normal_header %zu %zu %zu",
                              input.normal_header_width,
                              input.normal_header_height,
                              input.normal_header_depth));
    }
  }
  return out;
}

OwnedBankContract ParseOwnedBankContract(const std::string& bytes) {
  THROW_CHECK(!bytes.empty());
  ContractLexer lexer(bytes);
  OwnedBankContract contract;

  const auto next_line = [&lexer](const std::string& expected_key) {
    if (!lexer.Next() || lexer.AtUnterminatedLine()) {
      lexer.Fail("expected " + expected_key + ", got end of contract");
    }
    if (lexer.Key() != expected_key) {
      lexer.Fail("expected " + expected_key + ", got \"" + lexer.Key() + "\"");
    }
  };

  next_line("owned-bank-contract");
  if (lexer.TokenCount() != 1) lexer.Fail("header field count");
  next_line("version");
  if (lexer.TokenCount() != 2) lexer.Fail("version field count");
  const int version = ParseInt32(lexer, lexer.Token(1), "version");
  if (version != kOwnedBankContractVersion) {
    lexer.Fail("unsupported owned bank contract version " +
               std::to_string(version));
  }
  contract.version = version;

  next_line("build_cuda_enabled");
  if (lexer.TokenCount() != 2) lexer.Fail("build_cuda_enabled field count");
  ParseBoolInto(lexer, lexer.Token(1), "build_cuda_enabled",
                &contract.build_cuda_enabled);
  next_line("build_hip_enabled");
  if (lexer.TokenCount() != 2) lexer.Fail("build_hip_enabled field count");
  ParseBoolInto(lexer, lexer.Token(1), "build_hip_enabled",
                &contract.build_hip_enabled);

  next_line("allocation_compatibility");
  if (lexer.TokenCount() != 3) {
    lexer.Fail("allocation_compatibility field count");
  }
  contract.allocation_compatibility =
      ParseHexString(lexer, "allocation_compatibility", 1);

  next_line("invocation_options");
  if (lexer.TokenCount() != 1) lexer.Fail("invocation_options field count");
  ParseOptionsBlock(lexer, &contract.invocation_options);

  next_line("stereo_folder");
  contract.stereo_folder = ParseHexString(lexer, "stereo_folder", 1);

  next_line("problem_count");
  if (lexer.TokenCount() != 2) lexer.Fail("problem_count field count");
  // Bounded before any reserve(): a declared count must be backed by the
  // remaining bytes (each problem record needs far more than 256 bytes).
  const size_t problem_count = lexer.BoundCount(
      "problem_count",
      ParseSize(lexer, lexer.Token(1), "problem_count"),
      /*min_record_bytes=*/256);
  contract.problems.reserve(problem_count);

  for (size_t i = 0; i < problem_count; ++i) {
    next_line("problem");
    if (lexer.TokenCount() != 2 ||
        ParseUnsigned(lexer, lexer.Token(1), "problem ordinal", false) != i) {
      lexer.Fail("problem ordinals must be sequential");
    }
    OwnedBankContractProblem problem;
    next_line("reference_image_idx");
    if (lexer.TokenCount() != 2) lexer.Fail("reference_image_idx field count");
    problem.reference_image_idx =
        ParseInt32(lexer, lexer.Token(1), "reference_image_idx");
    if (problem.reference_image_idx < 0) {
      lexer.Fail("negative reference_image_idx");
    }
    next_line("reference_name");
    if (lexer.TokenCount() != 3) lexer.Fail("reference_name field count");
    problem.reference_name = ParseHexString(lexer, "reference_name", 1);
    next_line("output_depth_path");
    if (lexer.TokenCount() != 3) lexer.Fail("output_depth_path field count");
    ParsePathInto(lexer, "output_depth_path", 1, &problem.output_depth_path);
    next_line("output_normal_path");
    if (lexer.TokenCount() != 3) lexer.Fail("output_normal_path field count");
    ParsePathInto(lexer, "output_normal_path", 1, &problem.output_normal_path);
    next_line("problem_options");
    if (lexer.TokenCount() != 1) lexer.Fail("problem_options field count");
    ParseOptionsBlock(lexer, &problem.effective_options);

    next_line("input_count");
    if (lexer.TokenCount() != 2) lexer.Fail("input_count field count");
    // Bounded before any reserve(): each input record needs far more than
    // 128 bytes.
    const size_t input_count = lexer.BoundCount(
        "input_count",
        ParseSize(lexer, lexer.Token(1), "input_count"),
        /*min_record_bytes=*/128);
    problem.inputs.reserve(input_count);
    for (size_t k = 0; k < input_count; ++k) {
      next_line("input");
      if (lexer.TokenCount() != 2 ||
          ParseUnsigned(lexer, lexer.Token(1), "input ordinal", false) != k) {
        lexer.Fail("input ordinals must be sequential");
      }
      OwnedBankContractInput input;
      next_line("is_reference");
      if (lexer.TokenCount() != 2) lexer.Fail("is_reference field count");
      ParseBoolInto(lexer, lexer.Token(1), "is_reference", &input.is_reference);
      next_line("image_idx");
      if (lexer.TokenCount() != 2) lexer.Fail("image_idx field count");
      input.image_idx = ParseInt32(lexer, lexer.Token(1), "image_idx");
      if (input.image_idx < 0) lexer.Fail("negative image_idx");
      next_line("image_name");
      if (lexer.TokenCount() != 3) lexer.Fail("image_name field count");
      input.image_name = ParseHexString(lexer, "image_name", 1);
      next_line("image_dims");
      if (lexer.TokenCount() != 3) lexer.Fail("image_dims field count");
      input.width =
          ParseUnsigned(lexer, lexer.Token(1), "image_dims width", true);
      input.height =
          ParseUnsigned(lexer, lexer.Token(2), "image_dims height", true);
      next_line("max_image_size");
      if (lexer.TokenCount() != 2) lexer.Fail("max_image_size field count");
      input.max_image_size =
          ParseInt32(lexer, lexer.Token(1), "max_image_size");
      const auto parse_floats = [&lexer, &next_line](
                                    const std::string& key,
                                    const size_t count,
                                    float* out) {
        next_line(key);
        if (lexer.TokenCount() != 1 + count) lexer.Fail(key + " field count");
        std::vector<float> values;
        values.reserve(count);
        for (size_t v = 0; v < count; ++v) {
          const std::string token = lexer.Token(1 + v);
          std::string hex_bytes;
          if (!HexDecode(token, &hex_bytes) ||
              hex_bytes.size() != sizeof(uint32_t)) {
            lexer.Fail("invalid float bit representation for " + key);
          }
          values.push_back(DecodeFloat(token));
        }
        std::memcpy(out, values.data(), count * sizeof(float));
      };
      parse_floats("calib_k", 9, input.K.data());
      parse_floats("calib_r", 9, input.R.data());
      parse_floats("calib_t", 3, input.T.data());
      next_line("image_path");
      ParsePathInto(lexer, "image_path", 1, &input.image_path);
      if (!lexer.Next() || lexer.AtUnterminatedLine() ||
          lexer.Key() != "image_identity") {
        lexer.Fail("expected image_identity");
      }
      ParseIdentityInto(lexer, "image_identity", 1, &input.image_identity);
      next_line("depth_path");
      ParsePathInto(lexer, "depth_path", 1, &input.depth_path);
      if (!lexer.Next() || lexer.AtUnterminatedLine() ||
          lexer.Key() != "depth_identity") {
        lexer.Fail("expected depth_identity");
      }
      ParseIdentityInto(lexer, "depth_identity", 1, &input.depth_identity);
      next_line("depth_header");
      if (lexer.TokenCount() != 4) lexer.Fail("depth_header field count");
      input.depth_header_width =
          ParseUnsigned(lexer, lexer.Token(1), "depth_header width", true);
      input.depth_header_height =
          ParseUnsigned(lexer, lexer.Token(2), "depth_header height", true);
      input.depth_header_depth =
          ParseUnsigned(lexer, lexer.Token(3), "depth_header depth", true);
      next_line("normal_path");
      ParsePathInto(lexer, "normal_path", 1, &input.normal_path);
      if (!lexer.Next() || lexer.AtUnterminatedLine() ||
          lexer.Key() != "normal_identity") {
        lexer.Fail("expected normal_identity");
      }
      ParseIdentityInto(
          lexer, "normal_identity", 1, &input.normal_identity);
      next_line("normal_header");
      if (lexer.TokenCount() != 4) lexer.Fail("normal_header field count");
      input.normal_header_width =
          ParseUnsigned(lexer, lexer.Token(1), "normal_header width", true);
      input.normal_header_height =
          ParseUnsigned(lexer, lexer.Token(2), "normal_header height", true);
      input.normal_header_depth =
          ParseUnsigned(lexer, lexer.Token(3), "normal_header depth", true);
      problem.inputs.push_back(std::move(input));
    }
    contract.problems.push_back(std::move(problem));
  }

  if (lexer.Next()) {
    lexer.Fail("trailing content after contract");
  }
  return contract;
}

size_t FirstOwnedBankContractByteDifference(const std::string& a,
                                            const std::string& b) {
  const size_t common = std::min(a.size(), b.size());
  for (size_t i = 0; i < common; ++i) {
    if (a[i] != b[i]) return i;
  }
  return a.size() == b.size() ? std::string::npos : common;
}

}  // namespace mvs
}  // namespace colmap