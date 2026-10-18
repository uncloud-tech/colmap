// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/reconstruction_input_summary.h"

#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <fstream>

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Creates a tiny fixture photo at the given path through the public Bitmap
// (OpenImageIO based) write API. The summary under test only inspects image
// headers, so fixture pixel content is arbitrary.
void WriteTestImage(const std::filesystem::path& path,
                    const int width,
                    const int height) {
  Bitmap bitmap(width, height, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(128));
  ASSERT_TRUE(bitmap.Write(path));
}

// Creates a plain text file, used both for sidecar fixtures and for files
// that carry a recognized image extension but no valid image content.
void WriteTextFile(const std::filesystem::path& path, const std::string& text) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file << text;
}

}  // namespace

TEST(PhotoDirectorySummary, HeterogeneousPhotos) {
  const auto image_path = CreateTestDir() / "photos";
  CreateDirIfNotExists(image_path, /*recursive=*/true);
  WriteTestImage(image_path / "alpha.png", /*width=*/4, /*height=*/3);  // 12 px
  WriteTestImage(image_path / "beta.jpg", /*width=*/2, /*height=*/5);   // 10 px
  CreateDirIfNotExists(image_path / "sub", /*recursive=*/true);
  WriteTestImage(image_path / "sub" / "gamma.png",
                 /*width=*/5,
                 /*height=*/4);  // 20 px

  const PhotoDirectorySummary summary = SummarizePhotoDirectory(image_path);

  ASSERT_EQ(summary.photos.size(), 3);
  EXPECT_EQ(summary.problems.size(), 0);
  EXPECT_EQ(summary.non_image_files.size(), 0);

  // Deterministic enumeration: sorted lexicographically by relative path.
  EXPECT_EQ(summary.photos[0].relative_path, "alpha.png");
  EXPECT_EQ(summary.photos[1].relative_path, "beta.jpg");
  EXPECT_EQ(summary.photos[2].relative_path, "sub/gamma.png");
  // Encoded header dimensions, i.e. not compensated for EXIF orientation.
  EXPECT_EQ(summary.photos[0].encoded_width, 4);
  EXPECT_EQ(summary.photos[0].encoded_height, 3);
  EXPECT_EQ(summary.photos[1].encoded_width, 2);
  EXPECT_EQ(summary.photos[1].encoded_height, 5);
  EXPECT_EQ(summary.photos[2].encoded_width, 5);
  EXPECT_EQ(summary.photos[2].encoded_height, 4);

  // Largest actual photo by pixel count: a real single photo, not a maximum.
  ASSERT_TRUE(summary.largest_photo.has_value());
  EXPECT_EQ(summary.largest_photo->relative_path, "sub/gamma.png");
  EXPECT_EQ(summary.largest_photo->encoded_width, 5);
  EXPECT_EQ(summary.largest_photo->encoded_height, 4);

  // Separate named envelope: the width and height maxima are attained by
  // different photos and describe an upper bound, not one real photo.
  EXPECT_EQ(summary.dimension_envelope.max_encoded_width, 5);
  EXPECT_EQ(summary.dimension_envelope.max_width_path, "sub/gamma.png");
  EXPECT_EQ(summary.dimension_envelope.max_encoded_height, 5);
  EXPECT_EQ(summary.dimension_envelope.max_height_path, "beta.jpg");
}

TEST(PhotoDirectorySummary, DeterministicEnumeration) {
  const auto image_path = CreateTestDir() / "photos";
  CreateDirIfNotExists(image_path, /*recursive=*/true);
  CreateDirIfNotExists(image_path / "b_sub", /*recursive=*/true);
  CreateDirIfNotExists(image_path / "a_sub", /*recursive=*/true);
  // Create in scrambled order relative to the expected sorted output.
  WriteTestImage(image_path / "b_sub" / "delta.png", 2, 2);
  WriteTestImage(image_path / "alpha.png", 2, 2);
  WriteTestImage(image_path / "a_sub" / "charlie.png", 2, 2);
  WriteTestImage(image_path / "bravo.jpg", 2, 2);

  const PhotoDirectorySummary first = SummarizePhotoDirectory(image_path);
  const PhotoDirectorySummary second = SummarizePhotoDirectory(image_path);

  EXPECT_EQ(first.photos, second.photos);

  ASSERT_EQ(first.photos.size(), 4);
  EXPECT_EQ(first.photos[0].relative_path, "a_sub/charlie.png");
  EXPECT_EQ(first.photos[1].relative_path, "alpha.png");
  EXPECT_EQ(first.photos[2].relative_path, "b_sub/delta.png");
  EXPECT_EQ(first.photos[3].relative_path, "bravo.jpg");
}

TEST(PhotoDirectorySummary, NonImageSidecarsAreReportedButDoNotInvalidate) {
  const auto image_path = CreateTestDir() / "photos";
  CreateDirIfNotExists(image_path, /*recursive=*/true);
  WriteTestImage(image_path / "photo.png", 3, 2);
  WriteTextFile(image_path / "notes.txt", "sidecar note\n");
  WriteTextFile(image_path / "project.db", "not an image\n");

  const PhotoDirectorySummary summary = SummarizePhotoDirectory(image_path);

  EXPECT_EQ(summary.photos.size(), 1);
  EXPECT_TRUE(summary.largest_photo.has_value());
  EXPECT_EQ(summary.largest_photo->relative_path, "photo.png");
  // Sidecars must not appear as photos and must not surface as problems.
  EXPECT_EQ(summary.problems.size(), 0);
  ASSERT_EQ(summary.non_image_files.size(), 2);
  EXPECT_EQ(summary.non_image_files[0].relative_path, "notes.txt");
  EXPECT_EQ(summary.non_image_files[0].kind,
            PhotoScanNonImageFile::Kind::kSidecar);
  EXPECT_EQ(summary.non_image_files[1].relative_path, "project.db");
  EXPECT_EQ(summary.non_image_files[1].kind,
            PhotoScanNonImageFile::Kind::kSidecar);
}

TEST(PhotoDirectorySummary, CorruptImagesAreExposedExplicitly) {
  const auto image_path = CreateTestDir() / "photos";
  CreateDirIfNotExists(image_path, /*recursive=*/true);
  WriteTestImage(image_path / "good.png", 3, 2);
  WriteTextFile(image_path / "corrupt.jpg", "garbage garbage garbage");
  WriteTextFile(image_path / "corrupt.png", "garbage garbage garbage");

  const PhotoDirectorySummary summary = SummarizePhotoDirectory(image_path);

  // Corrupt files are not silently counted as fit inputs.
  EXPECT_EQ(summary.photos.size(), 1);
  ASSERT_TRUE(summary.largest_photo.has_value());
  EXPECT_EQ(summary.largest_photo->relative_path, "good.png");
  ASSERT_EQ(summary.problems.size(), 2);
  EXPECT_EQ(summary.problems[0].relative_path, "corrupt.jpg");
  EXPECT_EQ(summary.problems[0].kind, PhotoScanProblem::Kind::kCorruptImage);
  EXPECT_EQ(summary.problems[1].relative_path, "corrupt.png");
  EXPECT_EQ(summary.problems[1].kind, PhotoScanProblem::Kind::kCorruptImage);
  EXPECT_EQ(summary.non_image_files.size(), 0);
}

TEST(PhotoDirectorySummary, LargestPhotoTieKeepsScanOrderEntry) {
  const auto image_path = CreateTestDir() / "photos";
  CreateDirIfNotExists(image_path, /*recursive=*/true);
  // Equal encoded pixel counts across a nested directory and the root.
  // The tie winner must be the entry first in the documented scan order;
  // this pair agrees between path-component and plain string ordering so
  // the expectation stays portable across standard libraries.
  CreateDirIfNotExists(image_path / "outer", /*recursive=*/true);
  WriteTestImage(image_path / "outer" / "first.png",
                 /*width=*/3,
                 /*height=*/2);  // 6 px
  WriteTestImage(image_path / "second.png",
                 /*width=*/2,
                 /*height=*/3);  // 6 px

  const PhotoDirectorySummary summary = SummarizePhotoDirectory(image_path);

  ASSERT_EQ(summary.photos.size(), 2);
  EXPECT_EQ(summary.photos[0].relative_path, "outer/first.png");
  EXPECT_EQ(summary.photos[1].relative_path, "second.png");
  // Tie: the first entry in scan order wins.
  ASSERT_TRUE(summary.largest_photo.has_value());
  EXPECT_EQ(summary.largest_photo->relative_path,
            summary.photos[0].relative_path);
  EXPECT_EQ(summary.largest_photo->encoded_width, 3);
  EXPECT_EQ(summary.largest_photo->encoded_height, 2);
  // Strictly greater maxima still move their provenance paths.
  EXPECT_EQ(summary.dimension_envelope.max_width_path, "outer/first.png");
  EXPECT_EQ(summary.dimension_envelope.max_height_path, "second.png");
}

TEST(PhotoDirectorySummary, EmptyDirectory) {
  const auto image_path = CreateTestDir() / "photos";
  CreateDirIfNotExists(image_path, /*recursive=*/true);

  const PhotoDirectorySummary summary = SummarizePhotoDirectory(image_path);

  EXPECT_TRUE(summary.photos.empty());
  EXPECT_TRUE(summary.problems.empty());
  EXPECT_TRUE(summary.non_image_files.empty());
  EXPECT_FALSE(summary.largest_photo.has_value());
  EXPECT_FALSE(summary.dimension_envelope.IsValid());
}

TEST(PhotoDirectorySummary, UppercaseSuffixClassification) {
  const auto image_path = CreateTestDir() / "photos";
  CreateDirIfNotExists(image_path, /*recursive=*/true);
  WriteTestImage(image_path / "photo.png", 3, 2);
  // Unreadable files must be classified by their actual suffix case:
  // with only a lowercase extension comparison they would silently become
  // sidecars instead of explicit problems.
  WriteTextFile(image_path / "IMG_0001.JPG", "garbage");
  WriteTextFile(image_path / "corrupt.Jpeg", "garbage");
  // A readable photo with an uppercase suffix stays a photo.
  WriteTestImage(image_path / "readable.TIFF", 2, 2);

  const PhotoDirectorySummary summary = SummarizePhotoDirectory(image_path);

  // Sorted (path) order: "IMG_0001.JPG" < "corrupt.Jpeg" < "photo.png"
  // < "readable.TIFF" (uppercase letters sort before lowercase ones).
  ASSERT_EQ(summary.problems.size(), 2);
  EXPECT_EQ(summary.problems[0].relative_path, "IMG_0001.JPG");
  EXPECT_EQ(summary.problems[0].kind, PhotoScanProblem::Kind::kCorruptImage);
  EXPECT_EQ(summary.problems[1].relative_path, "corrupt.Jpeg");
  EXPECT_EQ(summary.problems[1].kind, PhotoScanProblem::Kind::kCorruptImage);
  EXPECT_EQ(summary.non_image_files.size(), 0);
  ASSERT_EQ(summary.photos.size(), 2);
  EXPECT_EQ(summary.photos[0].relative_path, "photo.png");
  EXPECT_EQ(summary.photos[1].relative_path, "readable.TIFF");
}

TEST(PhotoDirectorySummary, MissingDirectoryIsExplicitError) {
  const auto missing_path =
      CreateTestDir() / "photos" / "does" / "not" / "exist";
  EXPECT_ANY_THROW(SummarizePhotoDirectory(missing_path));
}

TEST(PhotoDirectorySummary, ExifOrientationIsReportedWithoutApplyingIt) {
  const auto image_path = CreateTestDir() / "photos";
  CreateDirIfNotExists(image_path, /*recursive=*/true);
  Bitmap bitmap(2, 3, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(10));
  const int orientation = 6;  // 90 degrees rotation request.
  bitmap.SetMetaData("Orientation", "int", &orientation);
  // JPEG is used for the fixture because OpenImageIO round-trips EXIF
  // orientation tags reliably through its JPEG EXIF support.
  ASSERT_TRUE(bitmap.Write(image_path / "oriented.jpg"));

  const PhotoDirectorySummary summary = SummarizePhotoDirectory(image_path);

  ASSERT_EQ(summary.photos.size(), 1);
  // Encoded dimensions stay in stored orientation; the caller is responsible
  // for interpreting them together with the reported EXIF orientation tag.
  EXPECT_EQ(summary.photos[0].encoded_width, 2);
  EXPECT_EQ(summary.photos[0].encoded_height, 3);
  ASSERT_TRUE(summary.photos[0].exif_orientation.has_value());
  EXPECT_EQ(summary.photos[0].exif_orientation.value(), 6);
}

}  // namespace colmap