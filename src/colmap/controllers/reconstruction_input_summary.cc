// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/reconstruction_input_summary.h"

#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/oiio_utils.h"
#include "colmap/util/string.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <OpenImageIO/imageio.h>

namespace colmap {
namespace {

// Common raster image file extensions handled by COLMAP's OpenImageIO based
// photo I/O. Used only to classify files that fail header parsing as corrupt
// image candidates (recognized extension) versus ordinary sidecar files
// (any other extension).
//
// The comparison must normalize the file's actual suffix: the generic
// HasFileExtension() helper lowercases only the requested extension, not the
// filename's extension, so an unreadable uppercase (e.g. ".JPG") suffix
// would silently fall through to the sidecar classification. Keep this
// normalization local; the global HasFileExtension() helper is intentionally
// left unchanged.
bool HasCommonPhotoExtension(const std::filesystem::path& path) {
  static constexpr std::array<std::string_view, 11> kCommonPhotoExtensions = {
      ".jpg",
      ".jpeg",
      ".png",
      ".tif",
      ".tiff",
      ".bmp",
      ".pgm",
      ".ppm",
      ".pnm",
      ".webp",
      ".exr"};
  std::string extension = path.extension().string();
  StringToLower(&extension);
  return std::find(kCommonPhotoExtensions.begin(),
                   kCommonPhotoExtensions.end(),
                   extension) != kCommonPhotoExtensions.end();
}

// Convert a filesystem path to a UTF-8 std::string. On Windows,
// path.string() returns a locale-dependent narrow string, which mangles
// non-ASCII characters before they reach OIIO. path.u8string() always yields
// UTF-8 bytes, but its return type changes from std::string in C++17 to
// std::u8string in C++20; the reinterpret_cast keeps this portable.
std::string PathToUtf8(const std::filesystem::path& path) {
  const auto u8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

}  // namespace

PhotoDirectorySummary SummarizePhotoDirectory(
    const std::filesystem::path& image_path) {
  THROW_CHECK_DIR_EXISTS(image_path);

  PhotoDirectorySummary summary;

  // Follow the established OpenImageIO initialization convention (see
  // Bitmap and the CLI entry point) so reusable callers need no prior
  // initialization.
  EnsureOpenImageIOInitialized();

  // Enumerate recursively, then sort the native file paths under
  // std::filesystem::path ordering. This is the same deterministic
  // convention as ImageReader; note that this ordering is by path components
  // and is not guaranteed to match a plain lexicographic ordering of the
  // relative-path strings. Directory symbolic links are not followed by
  // GetRecursiveFileList(): photo directories referenced through symlinks
  // are omitted, while symlinks to regular files count as files.
  std::vector<std::filesystem::path> file_paths =
      GetRecursiveFileList(image_path);
  std::sort(file_paths.begin(), file_paths.end());

  for (const auto& file_path : file_paths) {
    const std::string relative_path =
        GetNormalizedRelativePath(file_path, image_path);

    OIIO::ImageSpec config;
    // Never apply EXIF reorientation when probing: consumers receive the
    // raw encoded dimensions and the orientation tag separately.
    config["oiio:reorient"] = 0;

    const auto input = OIIO::ImageInput::open(PathToUtf8(file_path), &config);
    if (!input) {
      // Always retrieve the error to clear OIIO's pending error state.
      const std::string error = OIIO::geterror();
      VLOG(3) << "Failed to read photo metadata from " << relative_path << ": "
              << error;
      if (HasCommonPhotoExtension(file_path)) {
        summary.problems.push_back(PhotoScanProblem{
            PhotoScanProblem::Kind::kCorruptImage, relative_path});
      } else {
        summary.non_image_files.push_back(PhotoScanNonImageFile{
            PhotoScanNonImageFile::Kind::kSidecar, relative_path});
      }
      continue;
    }

    const OIIO::ImageSpec image_spec = input->spec();
    input->close();

    if (image_spec.width <= 0 || image_spec.height <= 0) {
      VLOG(3) << "Photo metadata reports invalid dimensions for "
              << relative_path;
      summary.problems.push_back(PhotoScanProblem{
          PhotoScanProblem::Kind::kUnknownDimensions, relative_path});
      continue;
    }

    PhotoScanEntry entry;
    entry.relative_path = relative_path;
    entry.encoded_width = image_spec.width;
    entry.encoded_height = image_spec.height;
    const int orientation = image_spec.get_int_attribute("Orientation", 0);
    if (orientation > 0) {
      // Report the raw tag verbatim. The EXIF standard defines meaningful
      // orientations 1-8; any other positive value is passed through
      // unvalidated for consumers to reject or interpret. No orientation
      // transform is applied here.
      entry.exif_orientation = orientation;
    }

    summary.photos.push_back(entry);
    if (!summary.largest_photo.has_value() ||
        entry.NumEncodedPixels() > summary.largest_photo->NumEncodedPixels()) {
      // If several photos share the largest encoded pixel count, the entry
      // first in the scan order above is kept; distinct files always have
      // distinct paths, so this only pins the outcome deterministically.
      summary.largest_photo = entry;
    }
    if (entry.encoded_width > summary.dimension_envelope.max_encoded_width) {
      summary.dimension_envelope.max_encoded_width = entry.encoded_width;
      summary.dimension_envelope.max_width_path = relative_path;
    }
    if (entry.encoded_height > summary.dimension_envelope.max_encoded_height) {
      summary.dimension_envelope.max_encoded_height = entry.encoded_height;
      summary.dimension_envelope.max_height_path = relative_path;
    }
  }

  return summary;
}

}  // namespace colmap