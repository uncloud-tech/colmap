// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace colmap {

// A single photo file that could be enumerated through its image metadata
// alone, without decoding any pixel data.
struct PhotoScanEntry {
  // Path relative to and normalized against the scanned root directory
  // (e.g., "sub/dir/img1.jpg"). Entries are in deterministic scan order:
  // recursive enumeration sorted under std::filesystem::path ordering, the
  // same convention as ImageReader. Note that this component-based ordering
  // is not guaranteed to match a plain lexicographic ordering of these
  // strings (e.g., on GNU libstdc++, "a/x.png" sorts before "a-b.png",
  // although the relative-path strings compare in the opposite order).
  std::string relative_path;

  // Encoded ("as stored in the file") dimensions as reported by the image
  // header. Zero when the header does not carry usable dimensions; such
  // files do not become photo entries.
  int encoded_width = 0;
  int encoded_height = 0;

  // EXIF orientation tag exactly as stored in the metadata, when a strictly
  // positive value is present. The encoded dimensions above are NOT
  // compensated for this orientation (orientation tags 5-8 transpose the
  // effective pixel grid), and no image transformation is applied here;
  // consumers must interpret dimensions together with this tag. The EXIF
  // standard defines meaningful orientations 1-8: consumers must validate
  // that range themselves, since any other positive raw value is passed
  // through verbatim rather than rejected or normalized.
  std::optional<int> exif_orientation;

  // Pixel count of the encoded dimensions.
  int64_t NumEncodedPixels() const {
    return static_cast<int64_t>(encoded_width) *
           static_cast<int64_t>(encoded_height);
  }

  bool operator==(const PhotoScanEntry& other) const {
    return relative_path == other.relative_path &&
           encoded_width == other.encoded_width &&
           encoded_height == other.encoded_height &&
           exif_orientation == other.exif_orientation;
  }
};

// A file that appears to be a photo but could not be enumerated completely.
// A planning layer must not silently treat these as valid reconstruction
// inputs; the inventory is incomplete as long as this list is non-empty.
struct PhotoScanProblem {
  enum class Kind {
    // A file with a common image file extension whose header could not be
    // parsed by any OpenImageIO format plugin (typically truncated or
    // corrupt data). Note the limitations documented on
    // SummarizePhotoDirectory(): corruption after a parseable header can
    // remain undetected, and a file that a format plugin refuses due to a
    // missing/corrupt payload beyond the header is classified here as well.
    // Distinguishing corruption from a missing format plugin would require
    // format-specific probing and is intentionally out of scope.
    kCorruptImage,
    // A file that was recognized by a format plugin but reports missing or
    // invalid encoded dimensions.
    kUnknownDimensions,
  };
  Kind kind;
  std::string relative_path;
};

// A regular file in the scanned directory that is not recognized as an
// image. Ordinary sidecar files (e.g., readme.txt, database.db) do not
// invalidate the directory, but are reported for completeness.
struct PhotoScanNonImageFile {
  enum class Kind {
    // Any regular file that no OpenImageIO format plugin can open. Note: a
    // file with an unrecognized image extension or without an extension
    // whose content is not detectably an image is classified here as well.
    kSidecar,
  };
  Kind kind;
  std::string relative_path;
};

// Conservative upper bounds over all photos with known encoded dimensions.
// The width and height maxima may be attained by different photos: this is
// a bounding envelope for planning, not a description of a single real
// photo. See also PhotoDirectorySummary::largest_photo for the largest
// actual photo.
struct PhotoDimensionEnvelope {
  int max_encoded_width = 0;
  // First photo attaining max_encoded_width in scan order, for provenance.
  std::string max_width_path;
  int max_encoded_height = 0;
  // First photo attaining max_encoded_height in scan order, for provenance.
  std::string max_height_path;

  bool IsValid() const {
    return max_encoded_width > 0 && max_encoded_height > 0;
  }
};

struct PhotoDirectorySummary {
  // Photos usable as reconstruction inputs, in the deterministic scan order
  // documented on PhotoScanEntry::relative_path. An empty list means no
  // photo was found.
  std::vector<PhotoScanEntry> photos;
  // Image candidates that could not be enumerated; see PhotoScanProblem.
  std::vector<PhotoScanProblem> problems;
  // Regular files not recognized as images; see PhotoScanNonImageFile.
  std::vector<PhotoScanNonImageFile> non_image_files;
  // Largest actual photo by encoded pixel count; the envelope maxima may
  // come from other photos. Equal counts are broken deterministically by
  // scan order (see PhotoScanEntry::relative_path), not by a re-sorted
  // relative-path string ordering.
  std::optional<PhotoScanEntry> largest_photo;
  PhotoDimensionEnvelope dimension_envelope;
};

// Recursively scans a photo directory and summarizes reconstruction-relevant
// inventory from image metadata alone: photo count, per-photo encoded
// dimensions, the largest actual photo, and the maximum per-image width /
// height envelope. Reads only image headers/metadata through OpenImageIO
// (after performing OpenImageIO's one-time initialization, following the
// Bitmap/CLI convention); no full pixel decode is performed.
//
// Throws an explicit error when the directory does not exist. Empty
// directories produce an empty summary; corrupt, unreadable or
// dimension-less image candidates are exposed as problems rather than being
// silently skipped, and ordinary non-image sidecar files are reported but do
// not invalidate the directory. Extension classification normalizes the
// actual file suffix case, so unreadable ".JPG"/".Jpeg" files are treated
// the same as their lowercase variants.
//
// Known limitations, by design:
// - Only headers are inspected: corruption located after a parseable header
//   (e.g., truncated JPEG/PNG payloads) can remain undetected at this stage.
//   Later reconstruction stages surface such failures on their own.
// - Encoded dimensions are as stored in the file; the EXIF orientation tag
//   (meaningful range 1-8) is reported verbatim alongside but never
//   applied.
// - Files are enumerated recursively and sorted under
//   std::filesystem::path ordering (the deterministic convention
//   documented on PhotoScanEntry::relative_path); directory symbolic links
//   are not descended into, so photo directories referenced through
//   symlinks are omitted, while symlinks to regular files count as files.
// - Format support follows the OpenImageIO plugins available at runtime.
PhotoDirectorySummary SummarizePhotoDirectory(
    const std::filesystem::path& image_path);

}  // namespace colmap