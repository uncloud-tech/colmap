// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Pair-only checked binary writer for a float Mat.
//
// The stock Mat<float>::Write (src/colmap/mvs/mat.cc) checks only that the
// output file opened and then ignores payload, flush and close failures, so a
// disk-full write can still be reported as success. This owned seam keeps the
// existing default writer untouched and instead provides an additive,
// fail-closed path for the opt-in owned pair driver.
//
// The emitted bytes are EXACTLY the stock layout: the ASCII header
// "<width>&<height>&<depth>&" followed by the data as little-endian IEEE-754
// floats (reusing colmap::WriteBinaryLittleEndian). failbit|badbit exceptions
// are enabled before any write; the stream is explicitly flushed and then
// closed through an injected close callable. Any short write, flush or close
// failure propagates and the partial file is left in place. The injectable
// helper performs no buffering, no global hook and no allocator override, so
// short-write / flush / close failures can be driven deterministically from
// CPU tests.

#include "colmap/mvs/mat.h"
#include "colmap/util/endian.h"
#include "colmap/util/string.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <ostream>

namespace colmap {
namespace mvs {
namespace owned {

// Writes the exact Mat<float>::Write byte layout to `stream`, flushes it, and
// then invokes `close`. Enables failbit|badbit exceptions before the first
// write so failures surface as exceptions. `close` may be empty when the
// stream is not owned by this call. This is the single seam shared by the real
// file writer and by the bounded CPU tests.
inline void WriteCheckedMatWith(const Mat<float>& mat,
                                std::ostream& stream,
                                const std::function<void()>& close) {
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  SetFullPrecTextStream(stream);
  stream << mat.GetWidth() << "&" << mat.GetHeight() << "&" << mat.GetDepth()
         << "&";
  WriteBinaryLittleEndian<float>(&stream,
                                 {mat.GetPtr(), mat.GetData().size()});
  stream.flush();
  if (close) {
    close();
  }
}

// Pair-only checked float Mat writer. Unlike the stock writer, exceptions are
// enabled BEFORE open, so an open failure throws as well. Flush and close are
// explicit; the first failure propagates and any partial file remains.
inline void WriteCheckedMat(const Mat<float>& mat,
                            const std::filesystem::path& path) {
  std::ofstream file;
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.open(path, std::ios::binary);
  WriteCheckedMatWith(mat, file, [&file]() { file.close(); });
}

}  // namespace owned
}  // namespace mvs
}  // namespace colmap
