// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/feature_extraction.h"

#include "colmap/scene/database.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <atomic>
#include <fstream>
#include <future>

#include <gtest/gtest.h>

namespace colmap {
namespace {

Bitmap CreateTestBitmap() {
  Bitmap bitmap(100, 100, /*as_rgb=*/false);
  bitmap.Fill(BitmapColor<uint8_t>(0));
  for (int y = 30; y < 70; ++y) {
    for (int x = 30; x < 70; ++x) {
      bitmap.SetPixel(x, y, BitmapColor<uint8_t>(255));
    }
  }
  return bitmap;
}

#if defined(COLMAP_MVS_ENABLED)
// Exercise actual reader -> extractor -> writer ownership. The first writer
// cannot retire its ticket until the reader has demonstrably hit backpressure.
TEST(CreateFeatureExtractorController,
     BudgetedBackpressureWritesAndWriterFailureDrains) {
  for (const bool fail_writer : {false, true}) {
    const auto root = CreateTestDir();
    const auto photos = root / "photos";
    CreateDirIfNotExists(photos);
    constexpr size_t kNumImages = 12;
    for (size_t i = 0; i < kNumImages; ++i) {
      ASSERT_TRUE(
          CreateTestBitmap().Write(photos / (std::to_string(i) + ".png")));
    }
    ImageReaderOptions reader;
    reader.image_path = photos;
    FeatureExtractionOptions options;
    options.use_gpu = false;
    options.num_threads = 2;
    // One 100x100 grayscale image costs 50,000 bitmap/staging bytes.
    // Leave room for its real tiny SIFT output, but not a second bitmap.
    options.host_budget_bytes = 90000;
    auto result = std::make_shared<FeatureExtractionResult>();
    auto controller =
        CreateFeatureExtractorController(root / "db", reader, options, result);
    std::promise<void> blocked;
    auto admission_wait = blocked.get_future().share();
    std::atomic<size_t> waits{0};
    std::atomic<size_t> writes{0};
    controller->AddCallback(kFeatureExtractionAdmissionWait, [&] {
      if (waits.fetch_add(1) == 0) blocked.set_value();
    });
    controller->AddCallback(kFeatureExtractionBeforeWrite, [&] {
      // A bounded event wait, never a scheduler-dependent sleep. If admission
      // does not block this fails rather than leaving the writer hung.
      if (admission_wait.wait_for(std::chrono::seconds(10)) !=
          std::future_status::ready) {
        throw std::runtime_error("reader did not reach admission wait");
      }
      ++writes;
      if (fail_writer) throw std::runtime_error("injected writer failure");
    });
    controller->Start();
    controller->Wait();
    EXPECT_GT(waits.load(), 0);
    EXPECT_GT(writes.load(), 0);
    EXPECT_EQ(result->success, !fail_writer);
    EXPECT_EQ(static_cast<bool>(result->error), fail_writer);
    EXPECT_EQ(result->live_bytes, 0);
    auto db = Database::Open(root / "db");
    if (fail_writer) {
      ASSERT_TRUE(result->error);
      try {
        std::rethrow_exception(result->error);
      } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), "injected writer failure");
      }
      EXPECT_TRUE(db->ReadAllImages().empty());
    } else {
      ASSERT_EQ(db->ReadAllImages().size(), kNumImages);
      for (const auto& image : db->ReadAllImages()) {
        const auto keypoints = db->ReadKeypoints(image.ImageId());
        const auto descriptors = db->ReadDescriptors(image.ImageId());
        EXPECT_GT(keypoints.size(), 0);
        EXPECT_EQ(descriptors.data.rows(), keypoints.size());
        EXPECT_EQ(descriptors.data.cols(), 128);
        EXPECT_GT(descriptors.data.size(), 0);
      }
    }
  }
}

TEST(CreateFeatureExtractorController, BudgetedHeterogeneousMatchesLegacy) {
  const auto root = CreateTestDir();
  const auto photos = root / "photos";
  CreateDirIfNotExists(photos);
  ASSERT_TRUE(CreateTestBitmap().Write(photos / "a.png"));
  Bitmap small(1, 7, true);
  small.Fill(BitmapColor<uint8_t>(40, 80, 120));
  ASSERT_TRUE(small.Write(photos / "b.png"));
  ImageReaderOptions reader;
  reader.image_path = photos;
  FeatureExtractionOptions options;
  options.use_gpu = false;
  options.num_threads = 2;
  options.max_image_size = 64;
  auto control = CreateFeatureExtractorController(root / "control.db", reader, options);
  control->Start();
  control->Wait();
  options.host_budget_bytes = 32 * 1024 * 1024;
  auto result = std::make_shared<FeatureExtractionResult>();
  auto candidate = CreateFeatureExtractorController(root / "candidate.db", reader, options, result);
  candidate->Start();
  candidate->Wait();
  ASSERT_TRUE(result->success);
  ASSERT_FALSE(result->error);
  auto a = Database::Open(root / "control.db");
  auto b = Database::Open(root / "candidate.db");
  ASSERT_EQ(a->ReadAllImages().size(), 2);
  ASSERT_EQ(b->ReadAllImages().size(), 2);
  for (const auto& image : a->ReadAllImages()) {
    const auto other = b->ReadImageWithName(image.Name()).value();
    EXPECT_EQ(a->ReadKeypoints(image.ImageId()), b->ReadKeypoints(other.ImageId()));
    const auto desc_a = a->ReadDescriptors(image.ImageId());
    const auto desc_b = b->ReadDescriptors(other.ImageId());
    ASSERT_EQ(desc_a.data.rows(), desc_b.data.rows());
    EXPECT_TRUE((desc_a.data.array() == desc_b.data.array()).all());
  }
}
TEST(CreateFeatureExtractorController, BudgetedSingletonGrowthAndDecodeFailuresDrain) {
  for (const uint64_t cap : {uint64_t{1}, uint64_t{51500}, uint64_t{32 * 1024 * 1024}}) {
    const auto root = CreateTestDir();
    const auto photos = root / "photos";
    CreateDirIfNotExists(photos);
    if (cap == 32 * 1024 * 1024) {
      ASSERT_TRUE(CreateTestBitmap().Write(photos / "a.png"));
      std::ofstream(photos / "z.png") << "not an image";
    } else {
      ASSERT_TRUE(CreateTestBitmap().Write(photos / "a.png"));
    }
    ImageReaderOptions reader;
    reader.image_path = photos;
    FeatureExtractionOptions options;
    options.use_gpu = false;
    options.num_threads = 2;
    options.host_budget_bytes = cap;
    auto result = std::make_shared<FeatureExtractionResult>();
    auto controller = CreateFeatureExtractorController(root / "db", reader, options, result);
    controller->Start();
    controller->Wait();
    EXPECT_FALSE(result->success);
    EXPECT_TRUE(result->error);
    auto db = Database::Open(root / "db");
    if (cap == 32 * 1024 * 1024)
      EXPECT_LE(db->ReadAllImages().size(), 1);  // Honest partial DB allowed.
    else
      EXPECT_TRUE(db->ReadAllImages().empty());
  }
}
TEST(CreateFeatureExtractorController, BudgetedUnsupportedModesRefuse) {
  const auto root = CreateTestDir();
  ImageReaderOptions reader;
  reader.image_path = root;
  reader.mask_path = root;
  FeatureExtractionOptions options;
  options.use_gpu = false;
  options.host_budget_bytes = 1024 * 1024;
  EXPECT_THROW(CreateFeatureExtractorController(root / "db", reader, options), std::invalid_argument);
  reader.mask_path.clear();
  options.type = FeatureExtractorType::ALIKED_N16ROT;
  EXPECT_THROW(CreateFeatureExtractorController(root / "db", reader, options), std::invalid_argument);
}
#endif

TEST(ExtractionBitmap, SameHeaderAdmissionPrecedesPixelAllocation) {
  const auto root = CreateTestDir();
  Bitmap rgb(7, 1, true);
  rgb.Fill(BitmapColor<uint8_t>(10, 20, 30));
  ASSERT_TRUE(rgb.Write(root / "rgb.png"));
  Bitmap bitmap;
  bool admitted = false;
  EXPECT_THROW(bitmap.Read(root / "rgb.png", false, false, 0, 0, 0,
      [&](int w, int h, int channels) {
        EXPECT_EQ(w, 7);
        EXPECT_EQ(h, 1);
        EXPECT_EQ(channels, 3);
        EXPECT_TRUE(bitmap.IsEmpty());
        admitted = true;
        throw std::runtime_error("refuse before pixels");
      }), std::runtime_error);
  EXPECT_TRUE(admitted);
  EXPECT_TRUE(bitmap.IsEmpty());
}

TEST(CreateFeatureExtractorController, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  CreateDirIfNotExists(image_path);

  // Create test images
  const int kNumImages = 2;
  const Bitmap test_bitmap = CreateTestBitmap();
  for (int i = 0; i < kNumImages; ++i) {
    test_bitmap.Write(image_path / (std::to_string(i) + ".png"));
  }

  // Set up options
  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = kNumImages;

  // Create and run the controller
  auto controller = CreateFeatureExtractorController(
      database_path, reader_options, extraction_options);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  // Verify results in database
  auto database = Database::Open(database_path);
  const std::vector<Image> images = database->ReadAllImages();
  EXPECT_EQ(images.size(), kNumImages);

  for (const auto& image : images) {
    EXPECT_TRUE(database->ExistsKeypoints(image.ImageId()));
    EXPECT_TRUE(database->ExistsDescriptors(image.ImageId()));

    const FeatureKeypoints keypoints = database->ReadKeypoints(image.ImageId());
    const FeatureDescriptors descriptors =
        database->ReadDescriptors(image.ImageId());

    // Check that features were extracted
    EXPECT_GT(keypoints.size(), 0);
    EXPECT_EQ(keypoints.size(), descriptors.data.rows());
    EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
    EXPECT_EQ(descriptors.data.cols(), 128);
  }
}

TEST(CreateFeatureExtractorController, WithCameraMask) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  const auto mask_path = test_dir / "mask.png";
  CreateDirIfNotExists(image_path);

  // Create test image with features
  const Bitmap test_bitmap = CreateTestBitmap();
  test_bitmap.Write(image_path / "test.png");

  // Create a mask that only allows the center region (white = keep, black =
  // mask) The test bitmap has a white square from (30,30) to (70,70) We'll
  // create a mask that only keeps a smaller region
  Bitmap mask_bitmap(100, 100, /*as_rgb=*/false);
  mask_bitmap.Fill(BitmapColor<uint8_t>(0));  // Start with all black (masked)

  // Only keep center region (40,40) to (60,60)
  for (int y = 40; y < 60; ++y) {
    for (int x = 40; x < 60; ++x) {
      mask_bitmap.SetPixel(x, y, BitmapColor<uint8_t>(255));  // White = keep
    }
  }
  mask_bitmap.Write(mask_path);

  // Extract features without mask first to get baseline
  ImageReaderOptions reader_options_no_mask;
  reader_options_no_mask.image_path = image_path;

  FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.num_threads = 1;

  auto controller = CreateFeatureExtractorController(
      database_path, reader_options_no_mask, extraction_options);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  auto database = Database::Open(database_path);
  std::vector<Image> images = database->ReadAllImages();
  ASSERT_EQ(images.size(), 1);

  const size_t num_features_no_mask =
      database->ReadKeypoints(images[0].ImageId()).size();
  EXPECT_GT(num_features_no_mask, 0);

  // Now extract with mask
  const auto database_path_masked = test_dir / "database_masked.db";
  ImageReaderOptions reader_options_masked;
  reader_options_masked.image_path = image_path;
  reader_options_masked.camera_mask_path = mask_path;

  controller = CreateFeatureExtractorController(
      database_path_masked, reader_options_masked, extraction_options);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  auto database_masked = Database::Open(database_path_masked);
  images = database_masked->ReadAllImages();
  ASSERT_EQ(images.size(), 1);

  const FeatureKeypoints keypoints_masked =
      database_masked->ReadKeypoints(images[0].ImageId());
  const FeatureDescriptors descriptors_masked =
      database_masked->ReadDescriptors(images[0].ImageId());
  const size_t num_features_masked = keypoints_masked.size();

  // With mask, should have fewer features
  EXPECT_LT(num_features_masked, num_features_no_mask);
  EXPECT_GT(num_features_masked, 0);  // But should still have some features

  // All remaining keypoints should be within the unmasked region (40-60, 40-60)
  for (const auto& kp : keypoints_masked) {
    EXPECT_GE(kp.x, 40.0f);
    EXPECT_LT(kp.x, 60.0f);
    EXPECT_GE(kp.y, 40.0f);
    EXPECT_LT(kp.y, 60.0f);
  }

  // Descriptors should match keypoints count
  EXPECT_EQ(descriptors_masked.data.rows(), keypoints_masked.size());
  EXPECT_EQ(descriptors_masked.type, FeatureExtractorType::SIFT);
  EXPECT_EQ(descriptors_masked.data.cols(), 128);
}

TEST(CreateFeatureImporterController, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto image_path = test_dir / "images";
  const auto import_path = test_dir / "features";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(import_path);

  const int kNumImages = 2;
  const int kNumFeatures = 3;

  // Create test images
  const Bitmap test_bitmap = CreateTestBitmap();
  for (int i = 0; i < kNumImages; ++i) {
    test_bitmap.Write(image_path / (std::to_string(i) + ".png"));
  }

  // Create feature text files for each image
  for (int i = 0; i < kNumImages; ++i) {
    const auto feature_file = import_path / (std::to_string(i) + ".png.txt");
    std::ofstream file(feature_file);
    ASSERT_TRUE(file.is_open());

    // Write header: num_features dimension
    const int kDimension = 128;
    file << kNumFeatures << " " << kDimension << "\n";

    // Write features: x y scale orientation descriptor[0..127]
    for (int j = 0; j < kNumFeatures; ++j) {
      // Keypoint data
      file << (10.0f + j * 5.0f) << " "  // x
           << (20.0f + j * 5.0f) << " "  // y
           << (1.5f + j * 0.1f) << " "   // scale
           << (0.5f + j * 0.2f);         // orientation

      // Descriptor data (128 values)
      for (int k = 0; k < kDimension; ++k) {
        file << " " << ((j * kDimension + k) % 256);
      }
      file << "\n";
    }
  }

  // Set up options
  ImageReaderOptions reader_options;
  reader_options.image_path = image_path;

  // Create and run the controller
  auto controller = CreateFeatureImporterController(
      database_path, reader_options, import_path);
  ASSERT_NE(controller, nullptr);
  controller->Start();
  controller->Wait();

  // Verify results in database
  auto database = Database::Open(database_path);
  const std::vector<Image> images = database->ReadAllImages();
  EXPECT_EQ(images.size(), kNumImages);

  for (const auto& image : images) {
    EXPECT_TRUE(database->ExistsKeypoints(image.ImageId()));
    EXPECT_TRUE(database->ExistsDescriptors(image.ImageId()));

    const FeatureKeypoints keypoints = database->ReadKeypoints(image.ImageId());
    const FeatureDescriptors descriptors =
        database->ReadDescriptors(image.ImageId());

    // Check that features were imported correctly
    EXPECT_EQ(keypoints.size(), kNumFeatures);
    EXPECT_EQ(descriptors.type, FeatureExtractorType::SIFT);
    EXPECT_EQ(descriptors.data.rows(), kNumFeatures);
    EXPECT_EQ(descriptors.data.cols(), 128);

    // Verify some keypoint values
    EXPECT_FLOAT_EQ(keypoints[0].x, 10.0f);
    EXPECT_FLOAT_EQ(keypoints[0].y, 20.0f);
  }
}

}  // namespace
}  // namespace colmap
