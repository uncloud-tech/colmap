// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace colmap::dense_plan {
// Small typed JSON emitter. No raw-JSON/string-typing escape hatch.
class Json {
 public:
  Json() : bytes_("null") {}
  explicit Json(bool value) : bytes_(value ? "true" : "false") {}
  explicit Json(int value) : bytes_(std::to_string(value)) {}
  explicit Json(double value) {
    if (!std::isfinite(value))
      throw std::invalid_argument("nonfinite JSON number");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << value;
    bytes_ = out.str();
  }
  explicit Json(const char* value) : Json(std::string(value)) {}
  explicit Json(const std::string& value) : bytes_(Quote(value)) {}
  static Json Array(const std::vector<Json>& values) {
    Json result;
    result.bytes_ = "[";
    for (const auto& value : values) {
      if (result.bytes_.size() > 1) result.bytes_ += ',';
      result.bytes_ += value.bytes_;
    }
    result.bytes_ += ']';
    return result;
  }
  static Json Object(const std::vector<std::pair<std::string, Json>>& values) {
    Json result;
    result.bytes_ = "{";
    std::set<std::string> keys;
    for (const auto& [key, value] : values) {
      if (!keys.insert(key).second)
        throw std::invalid_argument("duplicate JSON object key");
      if (result.bytes_.size() > 1) result.bytes_ += ',';
      result.bytes_ += Quote(key) + ':' + value.bytes_;
    }
    result.bytes_ += '}';
    return result;
  }
  const std::string& Dump() const { return bytes_; }

 private:
  static std::string Quote(const std::string& text) {
    // JSON strings must be valid UTF-8, including paths from a Unix filesystem.
    // Reject malformed names rather than silently corrupting their identity.
    for (size_t i = 0; i < text.size();) {
      const auto c = static_cast<unsigned char>(text[i++]);
      if (c < 0x80) continue;
      int count;
      uint32_t code;
      if (c >= 0xc2 && c <= 0xdf) {
        count = 1;
        code = c & 0x1f;
      } else if (c >= 0xe0 && c <= 0xef) {
        count = 2;
        code = c & 0x0f;
      } else if (c >= 0xf0 && c <= 0xf4) {
        count = 3;
        code = c & 7;
      } else
        throw std::invalid_argument("invalid UTF-8 in JSON string");
      const int total = count;
      while (count--) {
        if (i == text.size()) throw std::invalid_argument("truncated UTF-8");
        const auto next = static_cast<unsigned char>(text[i++]);
        if ((next & 0xc0) != 0x80) throw std::invalid_argument("invalid UTF-8");
        code = (code << 6) | (next & 0x3f);
      }
      if ((total == 2 && code < 0x800) || (total == 3 && code < 0x10000) ||
          (code >= 0xd800 && code <= 0xdfff) || code > 0x10ffff)
        throw std::invalid_argument("invalid UTF-8 code point");
    }
    static constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : text) {
      switch (c) {
        case '\"':
          out += "\\\"";
          break;
        case '\\':
          out += "\\\\";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        case '\t':
          out += "\\t";
          break;
        default:
          if (c < 32) {
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
          } else
            out += static_cast<char>(c);
      }
    }
    return out + '\"';
  }
  std::string bytes_;
};

// Publish a complete sibling temporary inode with link(2)'s atomic no-replace
// semantics. No mkdir, no replacing symlinks, no partial destination on error.
// Caller owns an immutable output parent; this is not a security sandbox.
inline void WriteNewAtomic(const std::filesystem::path& path,
                           const std::string& bytes) {
#if defined(__linux__)
  auto pattern =
      (path.parent_path() / ("." + path.filename().string() + ".tmp-XXXXXX"))
          .string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  int fd = mkstemp(name.data());
  if (fd < 0) throw std::runtime_error("cannot create output temporary file");
  try {
    size_t done = 0;
    while (done < bytes.size()) {
      const auto written = write(fd, bytes.data() + done, bytes.size() - done);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) throw std::runtime_error("cannot write output");
      done += static_cast<size_t>(written);
    }
    if (fsync(fd)) throw std::runtime_error("cannot sync output");
    const int closed = close(fd);
    fd = -1;
    if (closed) throw std::runtime_error("cannot close output");
    if (link(name.data(), path.c_str()))
      throw std::runtime_error("output must be a NEW file: " + path.string());
    unlink(name.data());
  } catch (...) {
    if (fd >= 0) close(fd);
    unlink(name.data());
    throw;
  }
#else
  throw std::runtime_error("atomic exclusive plan output requires Linux");
#endif
}
}  // namespace colmap::dense_plan
