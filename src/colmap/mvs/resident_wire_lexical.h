// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Strict forecast-wire lexical primitives shared by the native forecast
// parsers (the full-bank resident-forecast wire and the cohort wire). The
// contents were MOVED from resident_forecast.cc's internal block: the
// USED parsing paths behave identically (fullbank fixtures re-verified).
// One previously-DEAD helper (ValidateFingerprintText) was repaired for
// the cohort scope's exact 64-lowercase-hex digest-TEXT rule; the
// full-bank parser still does not call it. Not a wire framework: two
// strict parsers of one shared grammar style; each wire keeps its own
// magic, layout and gates.

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <glog/logging.h>

namespace colmap {
namespace mvs {
namespace resident_lexical {

// ---------------------------------------------------------------------------
// Strict line lexer mirroring the canonical Python wire parser: ASCII
// bytes, '\n'-terminated final line, no empty/padded lines, exact
// single-space tokenization, canonical decimal integers, lowercase hex.
// ---------------------------------------------------------------------------

class ForecastLexer {
 public:
  explicit ForecastLexer(std::string_view bytes) : bytes_(bytes) {}

  // Splits the next line into single-space-separated tokens. Returns false
  // at end of input. Unlike the native contract lexer, empty and padded
  // lines fail immediately (mirroring the Python `_Lexer.next`).
  inline bool Next() {
    if (pos_ >= bytes_.size()) {
      tokens_.clear();
      line_ = std::string_view();
      return false;
    }
    const size_t eol = bytes_.find('\n', pos_);
    if (eol == std::string_view::npos) {
      tokens_.clear();
      line_ = bytes_.substr(bytes_.size() > 0 ? bytes_.size() - 1 : 0, 0);
      ThrowAt(line_no_ + 1, "unterminated final line");
    }
    line_ = bytes_.substr(pos_, eol - pos_);
    line_start_ = pos_;
    pos_ = eol + 1;
    ++line_no_;
    if (line_.empty() || line_.front() == ' ' || line_.back() == ' ') {
      ThrowAt(line_no_, "empty or padded line");
    }
    tokens_.clear();
    size_t cursor = 0;
    while (true) {
      const size_t end = line_.find(' ', cursor);
      const size_t count = end == std::string_view::npos ? end : end - cursor;
      tokens_.push_back(std::string(line_.substr(cursor, count)));
      if (end == std::string_view::npos) break;
      cursor = end + 1;
    }
    return true;
  }

  inline size_t LineNo() const { return line_no_; }
  inline const std::string& LineText() const { return line_buffer_; }
  inline const std::string& Token(const size_t index) const {
    if (index >= tokens_.size()) {
      ThrowMissingToken(LineNo(), LineText());
    }
    return tokens_[index];
  }
  inline size_t TokenCount() const { return tokens_.size(); }
  inline const std::string& Key() const {
    return tokens_.empty() ? empty_ : tokens_.front();
  }

  // Bytes not yet consumed INCLUDING the current line: a declared value on
  // the consumed line (e.g. the raw hex block) is backed by those bytes.
  inline size_t RemainingBytes() const {
    return bytes_.size() - line_start_ + (pos_ < line_start_ ? pos_ : 0);
  }

  // Declared byte/string counts bounded by the remaining bytes before any
  // large decode: the raw block needs 2 hex chars per byte.
  inline uint64_t BoundByteCount(const std::string& key,
                                 const uint64_t count) const {
    if (count > RemainingBytes() / 2) {
      ThrowAt(
          LineNo(),
          "declared " + key + " size exceeds the remaining " + "forecast size");
    }
    return count;
  }

  [[noreturn]] inline void Fail(const std::string& message) const {
    ThrowAt(line_no_, message);
  }

 private:
  inline static std::string RenderLine(const std::string_view& line) {
    return std::string(line.substr(0, std::min<size_t>(line.size(), 80)));
  }

  [[noreturn]] inline static void ThrowMissingToken(size_t line,
                                                    const std::string& text) {
    throw std::runtime_error("Malformed resident forecast at line " +
                             std::to_string(line) + " (" + text +
                             "): missing token");
  }

  [[noreturn]] void ThrowAt(size_t line, const std::string& message) const {
    throw std::runtime_error("Malformed resident forecast at line " +
                             std::to_string(line) + " (" + RenderLine(line_) +
                             "): " + message);
  }

  std::string_view bytes_;
  size_t pos_ = 0;
  size_t line_no_ = 0;
  std::string_view line_;
  size_t line_start_ = 0;
  std::vector<std::string> tokens_;
  std::string line_buffer_;
  std::string empty_;
};

// Canonical unsigned decimal: ASCII digits only, no sign, no leading
// zeros; into uint64 without lossy intermediates.
inline uint64_t ParseUint(const ForecastLexer& lexer,
                          const std::string& token,
                          const std::string& key) {
  if (token.empty() || token.front() == '+' || token.front() == '-' ||
      (token.size() > 1 && token.front() == '0')) {
    lexer.Fail("non-canonical unsigned decimal for " + key);
  }
  uint64_t value = 0;
  const auto result =
      std::from_chars(token.data(), token.data() + token.size(), value, 10);
  if (result.ec != std::errc() || result.ptr != token.data() + token.size()) {
    lexer.Fail("invalid unsigned decimal for " + key);
  }
  return value;
}

inline uint64_t ParseUintMinimum(const ForecastLexer& lexer,
                                 const std::string& token,
                                 const std::string& key,
                                 const uint64_t minimum) {
  const uint64_t value = ParseUint(lexer, token, key);
  if (value < minimum) {
    lexer.Fail("value below the minimum for " + key);
  }
  return value;
}

inline uint64_t ParseUintMax(const ForecastLexer& lexer,
                             const std::string& token,
                             const std::string& key,
                             const uint64_t maximum) {
  const uint64_t value = ParseUint(lexer, token, key);
  if (value > maximum) {
    lexer.Fail("value above the maximum for " + key);
  }
  return value;
}

// Strict lowercase-only hex decode with a declared byte length.
inline bool TryHexDecode(const std::string& hex, std::string* out) {
  if (hex.size() % 2 != 0) return false;
  out->clear();
  out->reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); ++i) {
    const char c = hex[i];
    int value;
    if (c >= '0' && c <= '9')
      value = c - '0';
    else if (c >= 'a' && c <= 'f')
      value = c - 'a' + 10;
    else
      return false;
    if (i % 2 == 0) {
      out->push_back(static_cast<char>(value << 4));
    } else {
      out->back() =
          static_cast<char>(static_cast<unsigned char>(out->back()) | value);
    }
  }
  return true;
}

// Length-prefixed byte-string value. `line_final` marks strings that must
// be the WHOLE remaining line (field-count enforced); embedded strings are
// length/hex-checked in place and the line's remaining layout is validated
// by the caller (mirroring the Python parser's exact record layouts).
inline std::string ParseByteString(const ForecastLexer& lexer,
                                   const std::string& key,
                                   const size_t length_token,
                                   const size_t hex_token,
                                   const bool line_final) {
  const uint64_t length =
      ParseUint(lexer, lexer.Token(length_token), key + " length");
  lexer.BoundByteCount(key, length);
  if (line_final && lexer.TokenCount() != hex_token + 1) {
    lexer.Fail("malformed byte string for " + key);
  }
  std::string decoded;
  if (!TryHexDecode(lexer.Token(hex_token), &decoded) ||
      decoded.size() != length) {
    lexer.Fail("invalid hex encoding for " + key);
  }
  return decoded;
}

// 64 lowercase hex chars as a bare token (contract_sha256).
inline std::string ParseShaToken(const ForecastLexer& lexer,
                                 const std::string& token) {
  if (token.size() != 64) {
    lexer.Fail("contract_sha256 must carry 64 lowercase hex characters");
  }
  for (const char c : token) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
      lexer.Fail("contract_sha256 must carry 64 lowercase hex characters");
    }
  }
  return token;
}

// The fingerprint fields carry the digest TEXT (64 lowercase hex chars)
// as a length-prefixed byte string, or the literal token `absent`.
inline void ParseFingerprint(const ForecastLexer& lexer,
                             const std::string& key,
                             const uint64_t line_no,
                             bool* present,
                             std::string* value) {
  if (lexer.TokenCount() == 2 && lexer.Token(1) == "absent") {
    *present = false;
    value->clear();
    return;
  }
  if (lexer.TokenCount() != 3) {
    lexer.Fail("malformed " + key + " fingerprint field layout");
  }
  const std::string decoded =
      ParseByteString(lexer, key, 1, 2, /*line_final=*/true);
  if (decoded.size() != 64) {
    lexer.Fail(key + " must carry 64 hex digits or 'absent'");
  }
  // Python's parser checks only the length here; the digest comparison to
  // the compiled build fingerprint happens in the parameter gate. Keep the
  // same strictness (no extra constraints) to avoid divergent parsers.
  *present = true;
  *value = decoded;
}

// The 64-byte fingerprint payload must be 64 lowercase hex DIGIT characters
// when compared as text; the parameter gate compares to the compiled digest.
// The fingerprint fields carry the digest TEXT as a length-prefixed byte
// string: exactly 64 lowercase hex DIGIT characters (the cohort wire's
// execution qualification compares to the compiled digest; a non-hex or
// uppercase payload refuses).
inline void ValidateFingerprintText(const ForecastLexer& lexer,
                                    const std::string& key,
                                    const std::string& payload) {
  if (payload.size() != 64) {
    lexer.Fail(key + " must be a 64-character digest");
  }
  for (const char c : payload) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
      lexer.Fail(key + " payload is not a 64-lowercase-hex digest");
    }
  }
}

inline int64_t ParseSignedInt32(const ForecastLexer& lexer,
                                const std::string& token,
                                const std::string& key) {
  std::string_view body(token);
  const bool negative = !body.empty() && body.front() == '-';
  if (negative) body.remove_prefix(1);
  if (body.empty() || (body.front() == '0' && body.size() > 1)) {
    lexer.Fail("non-canonical signed decimal for " + key);
  }
  for (const char c : body) {
    if (c < '0' || c > '9') lexer.Fail("invalid signed decimal for " + key);
  }
  // Magnitude bound at the int32 range; int64 keeps the negation exact.
  uint64_t magnitude = 0;
  const auto result =
      std::from_chars(body.data(), body.data() + body.size(), magnitude, 10);
  if (result.ec != std::errc() || result.ptr != body.data() + body.size()) {
    lexer.Fail("invalid signed decimal for " + key);
  }
  if (!negative && magnitude > 2147483647ull) {
    lexer.Fail("integer out of range for " + key);
  }
  if (negative && magnitude > 2147483648ull) {
    lexer.Fail("integer out of range for " + key);
  }
  if (negative) {
    return -1 * static_cast<int64_t>(magnitude);
  }
  return static_cast<int64_t>(magnitude);
}

constexpr int64_t kExpectedContractVersion = 1;

}  // namespace resident_lexical
}  // namespace mvs
}  // namespace colmap
