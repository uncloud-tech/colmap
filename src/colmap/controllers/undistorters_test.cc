// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/undistorters.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/string.h"
#include "colmap/util/testing.h"

#include <filesystem>
#include <fstream>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

Reconstruction CreateSyntheticReconstructionWithBitmaps(
    const std::filesystem::path& image_path,
    int num_images = 2,
    int image_width = 100,
    int image_height = 100,
    const std::string& image_extension = ".png") {
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = num_images;
  synthetic_dataset_options.camera_width = image_width;
  synthetic_dataset_options.camera_height = image_height;
  synthetic_dataset_options.image_extension = image_extension;

  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);

  // Create dummy images.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    Bitmap bitmap(image_width, image_height, true);
    bitmap.Fill(BitmapColor<uint8_t>(128, 128, 128));
    bitmap.Write(image_path / image.Name());
  }

  return reconstruction;
}

#if defined(__linux__) && defined(COLMAP_MVS_ENABLED)
TEST(COLMAPUndistorter, BudgetedSingletonRefusesBeforeDecodeAndPublication) {
  const auto dir = CreateTestDir();
  CreateDirIfNotExists(dir / "inputs");
  auto reconstruction = CreateSyntheticReconstructionWithBitmaps(dir / "inputs");
  UndistortCameraOptions camera_options;
  camera_options.max_image_size = 20;
  COLMAPUndistorter::Options options;
  options.host_budget_bytes = 1;
  options.num_threads = 4;
  COLMAPUndistorter undistorter(options, camera_options, reconstruction,
                                dir / "inputs", dir / "output");
  try {
    undistorter.Run();
    FAIL() << "expected singleton no-fit";
  } catch (const std::runtime_error& error) {
    EXPECT_THAT(error.what(), testing::HasSubstr("before decode"));
  }
  EXPECT_FALSE(ExistsDir(dir / "output"));
}

TEST(COLMAPUndistorter, BudgetedWorkersPreserveNativeBytes) {
  const auto dir = CreateTestDir();
  CreateDirIfNotExists(dir / "inputs");
  auto reconstruction = CreateSyntheticReconstructionWithBitmaps(
      dir / "inputs", 4, 100, 100);
  // Nonuniform pixels make the full-size antialiased warp path observable.
  for (const auto& [id, image] : reconstruction.Images()) {
    Bitmap bitmap(100, 100, true);
    for (int y = 0; y < 100; ++y)
      for (int x = 0; x < 100; ++x)
        bitmap.SetPixel(x, y, BitmapColor<uint8_t>(x * 2, y * 2, (x + y) % 256));
    ASSERT_TRUE(bitmap.Write(dir / "inputs" / image.Name()));
  }
  UndistortCameraOptions camera_options;
  camera_options.max_image_size = 20;
  COLMAPUndistorter::Options options;
  options.num_threads = 4;
  COLMAPUndistorter control(options, camera_options, reconstruction,
                            dir / "inputs", dir / "control");
  control.Run();
  // 61,200 known bytes per indirect RGB job. Two fit, three do not.
  options.host_budget_bytes = 150000;
  COLMAPUndistorter bounded(options, camera_options, reconstruction,
                            dir / "inputs", dir / "bounded");
  bounded.Run();
  options.host_budget_bytes = 70000;  // automatically reduced to one worker
  COLMAPUndistorter serial(options, camera_options, reconstruction,
                           dir / "inputs", dir / "serial");
  serial.Run();
  for (const auto& path : GetRecursiveFileList(dir / "control")) {
    const auto name = std::filesystem::relative(path, dir / "control");
    std::vector<char> expected, actual, one;
    ReadBinaryBlob(path, &expected);
    ReadBinaryBlob(dir / "bounded" / name, &actual);
    ReadBinaryBlob(dir / "serial" / name, &one);
    EXPECT_EQ(expected, actual) << name;
    EXPECT_EQ(expected, one) << name;
  }
}

TEST(COLMAPUndistorter, BudgetedMalformedUnsupportedAndHeaderMismatch) {
  const auto dir = CreateTestDir();
  CreateDirIfNotExists(dir / "inputs");
  auto reconstruction = CreateSyntheticReconstructionWithBitmaps(dir / "inputs");
  COLMAPUndistorter::Options options;
  options.host_budget_bytes = uint64_t{1} << 30;
  UndistortCameraOptions camera_options;
  camera_options.max_image_size = 20;
  camera_options.roi_min_x = 0.1;
  COLMAPUndistorter roi(options, camera_options, reconstruction,
                        dir / "inputs", dir / "roi");
  EXPECT_THROW(roi.Run(), std::invalid_argument);
  camera_options.roi_min_x = 0;
  camera_options.max_scale = std::numeric_limits<double>::quiet_NaN();
  COLMAPUndistorter malformed(options, camera_options, reconstruction,
                              dir / "inputs", dir / "nan");
  EXPECT_THROW(malformed.Run(), std::invalid_argument);
  camera_options.max_scale = 2;
  const auto id = reconstruction.RegImageIds().front();
  Bitmap wrong_shape(101, 100, true);
  ASSERT_TRUE(wrong_shape.Write(dir / "inputs" / reconstruction.Image(id).Name()));
  COLMAPUndistorter mismatched(options, camera_options, reconstruction,
                               dir / "inputs", dir / "mismatch");
  EXPECT_THROW(mismatched.Run(), std::invalid_argument);
  EXPECT_FALSE(ExistsDir(dir / "mismatch"));
}

TEST(COLMAPUndistorter, BudgetedPerspectiveModelsSphericalAndCopyThrough) {
  const auto dir = CreateTestDir();
  CreateDirIfNotExists(dir / "inputs");
  auto reconstruction = CreateSyntheticReconstructionWithBitmaps(dir / "inputs");
  const auto camera_id = reconstruction.Cameras().begin()->first;
  for (const auto model : {CameraModelId::kSimpleRadial, CameraModelId::kRadial,
                           CameraModelId::kOpenCV, CameraModelId::kPinhole,
                           CameraModelId::kEquirectangular}) {
    reconstruction.Camera(camera_id) = Camera::CreateFromModelId(
        camera_id, model, 100, 100, 100);
    UndistortCameraOptions camera_options;
    camera_options.max_image_size = 20;
    COLMAPUndistorter::Options options;
    options.num_threads = 2;
    const auto label = std::to_string(static_cast<int>(model));
    COLMAPUndistorter control(options, camera_options, reconstruction,
                              dir / "inputs", dir / (label + "control"));
    control.Run();
    options.host_budget_bytes = uint64_t{1} << 30;
    COLMAPUndistorter bounded(options, camera_options, reconstruction,
                              dir / "inputs", dir / (label + "bounded"));
    bounded.Run();
    for (const auto& [id, image] : reconstruction.Images()) {
      std::vector<char> expected, actual;
      ReadBinaryBlob(dir / (label + "control") / "images" / image.Name(), &expected);
      ReadBinaryBlob(dir / (label + "bounded") / "images" / image.Name(), &actual);
      EXPECT_EQ(expected, actual);
    }
    if (model == CameraModelId::kPinhole || model == CameraModelId::kEquirectangular) {
      camera_options.max_image_size = -1;
      options.host_budget_bytes = 1;  // No native pixel vector on copy-through.
      COLMAPUndistorter copied(options, camera_options, reconstruction,
                               dir / "inputs", dir / (label + "copy"));
      copied.Run();
      for (const auto& [id, image] : reconstruction.Images()) {
        std::vector<char> expected, actual;
        ReadBinaryBlob(dir / "inputs" / image.Name(), &expected);
        ReadBinaryBlob(dir / (label + "copy") / "images" / image.Name(), &actual);
        EXPECT_EQ(expected, actual);
      }
    }
  }
}

TEST(COLMAPUndistorter, BudgetedHeterogeneousDimensions) {
  const auto dir = CreateTestDir();
  CreateDirIfNotExists(dir / "inputs");
  SyntheticDatasetOptions synthetic;
  synthetic.num_rigs = 1;
  synthetic.num_cameras_per_rig = 2;
  synthetic.num_frames_per_rig = 2;
  synthetic.camera_width = 100;
  synthetic.camera_height = 100;
  synthetic.image_extension = ".png";
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic, &reconstruction);
  auto& smaller = reconstruction.Camera(reconstruction.Cameras().begin()->first);
  smaller.width = 50;
  smaller.height = 40;
  for (const auto& [id, image] : reconstruction.Images()) {
    const auto& camera = *image.CameraPtr();
    Bitmap bitmap(camera.width, camera.height, true);
    bitmap.Fill(BitmapColor<uint8_t>(17, 41, 83));
    ASSERT_TRUE(bitmap.Write(dir / "inputs" / image.Name()));
  }
  UndistortCameraOptions camera_options;
  camera_options.max_image_size = 20;
  COLMAPUndistorter::Options options;
  options.host_budget_bytes = 150000;
  options.num_threads = 4;
  COLMAPUndistorter bounded(options, camera_options, reconstruction,
                            dir / "inputs", dir / "bounded");
  bounded.Run();
  for (const auto& [id, image] : reconstruction.Images()) {
    Bitmap actual;
    ASSERT_TRUE(actual.Read(dir / "bounded/images" / image.Name()));
    const auto expected = UndistortCamera(camera_options, *image.CameraPtr());
    EXPECT_EQ(actual.Width(), expected.width);
    EXPECT_EQ(actual.Height(), expected.height);
  }
}

TEST(COLMAPUndistorter, BudgetedWorkerHeaderExceptionDrains) {
  const auto dir = CreateTestDir();
  CreateDirIfNotExists(dir / "inputs");
  auto reconstruction = CreateSyntheticReconstructionWithBitmaps(dir / "inputs", 4);
  UndistortCameraOptions camera_options;
  camera_options.max_image_size = 20;
  COLMAPUndistorter::Options options;
  options.num_threads = 4;
  options.host_budget_bytes = uint64_t{1} << 30;
  COLMAPUndistorter undistorter(options, camera_options, reconstruction,
                                dir / "inputs", dir / "output");
  bool replaced = false;
  undistorter.SetCheckIfStoppedFunc([&] {
    if (!replaced) {
      // After inventory but before task submission: the SAME decoder must
      // reject a changed header before payload. Exercise exception, not false.
      Bitmap changed(101, 100, true);
      EXPECT_TRUE(changed.Write(dir / "inputs" /
          reconstruction.Image(reconstruction.RegImageIds().front()).Name()));
      replaced = true;
    }
    return false;
  });
  EXPECT_ANY_THROW(undistorter.Run());
  EXPECT_TRUE(replaced);
  EXPECT_FALSE(ExistsFile(dir / "output/stereo/fusion.cfg"));
  EXPECT_FALSE(ExistsFile(dir / "output/sparse/cameras.bin"));
}

TEST(COLMAPUndistorter, BudgetedFailureAndCancellationDrainWithoutCompletion) {
  const auto dir = CreateTestDir();
  CreateDirIfNotExists(dir / "inputs");
  auto reconstruction = CreateSyntheticReconstructionWithBitmaps(dir / "inputs", 4);
  COLMAPUndistorter::Options options;
  options.host_budget_bytes = uint64_t{1} << 30;
  options.num_threads = 2;
  UndistortCameraOptions camera_options;
  camera_options.max_image_size = 20;
  // Header preflight succeeds, then output Write fails on a directory.
  const auto name = reconstruction.Image(reconstruction.RegImageIds().front()).Name();
  CreateDirIfNotExists(dir / "failed");
  CreateDirIfNotExists(dir / "failed/images");
  CreateDirIfNotExists(dir / "failed/images" / name);
  COLMAPUndistorter failed(options, camera_options, reconstruction,
                           dir / "inputs", dir / "failed");
  EXPECT_THROW(failed.Run(), std::runtime_error);
  EXPECT_FALSE(ExistsFile(dir / "failed/stereo/fusion.cfg"));
  EXPECT_FALSE(ExistsFile(dir / "failed/sparse/cameras.bin"));
  COLMAPUndistorter cancelled(options, camera_options, reconstruction,
                              dir / "inputs", dir / "cancelled");
  int checks = 0;
  cancelled.SetCheckIfStoppedFunc([&checks] { return ++checks >= 2; });
  EXPECT_THROW(cancelled.Run(), std::runtime_error);
  EXPECT_FALSE(ExistsFile(dir / "cancelled/stereo/fusion.cfg"));
  EXPECT_FALSE(ExistsFile(dir / "cancelled/sparse/cameras.bin"));
  // Completed output bodies are stable after Run returns: no owned tasks left.
  for (const auto& path : GetRecursiveFileList(dir / "cancelled/images")) {
    Bitmap bitmap;
    EXPECT_TRUE(bitmap.Read(path));
  }
}
#endif

TEST(COLMAPUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run COLMAP undistorter.
  COLMAPUndistorter undistorter(COLMAPUndistorter::Options(),
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify output directories were created.
  EXPECT_TRUE(ExistsDir(output_path / "images"));
  EXPECT_TRUE(ExistsDir(output_path / "sparse"));
  EXPECT_TRUE(ExistsDir(output_path / "stereo"));

  // Verify undistorted images were written.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    EXPECT_TRUE(ExistsFile(output_path / "images" / image.Name()));
  }

  // Expect dense reconstruction files to be written.
  EXPECT_TRUE(ExistsFile(output_path / "stereo/patch-match.cfg"));
  EXPECT_TRUE(ExistsFile(output_path / "stereo/fusion.cfg"));
}

TEST(COLMAPUndistorter, StopsPendingWork) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path, /*num_images=*/10);
  COLMAPUndistorter undistorter(COLMAPUndistorter::Options(),
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  bool stop_checked = false;
  undistorter.SetCheckIfStoppedFunc([&stop_checked]() {
    stop_checked = true;
    return true;
  });
  undistorter.Run();

  EXPECT_TRUE(stop_checked);
  EXPECT_FALSE(ExistsFile(output_path / "stereo/patch-match.cfg"));
  EXPECT_FALSE(ExistsFile(output_path / "stereo/fusion.cfg"));
}

TEST(COLMAPUndistorter, SpecificImages) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path,
                                               /*num_images=*/2,
                                               /*image_width=*/100,
                                               /*image_height=*/100,
                                               /*image_extension=*/".jpg");

  const Image& image = reconstruction.Image(reconstruction.RegImageIds()[0]);

  // Run COLMAP undistorter.
  COLMAPUndistorter::Options options;
  options.image_ids = {image.ImageId()};
  COLMAPUndistorter undistorter(options,
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify that only the specified image was written.
  EXPECT_THAT(
      GetRecursiveFileList(output_path / "images"),
      testing::UnorderedElementsAre(output_path / "images" / image.Name()));
}

TEST(COLMAPUndistorter, JpegQuality) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path,
                                               /*num_images=*/1,
                                               /*image_width=*/100,
                                               /*image_height=*/100,
                                               /*image_extension=*/".jpg");

  // Run COLMAP undistorter.
  COLMAPUndistorter::Options options;
  options.jpeg_quality = 50;
  COLMAPUndistorter undistorter(options,
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify undistorted images were written.
  for (const auto& [image_id, image] : reconstruction.Images()) {
    EXPECT_TRUE(ExistsFile(output_path / "images" / image.Name()));
  }
}

TEST(PMVSUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "pmvs_output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run PMVS undistorter.
  PMVSUndistorter undistorter(PMVSUndistorter::Options(),
                              UndistortCameraOptions(),
                              reconstruction,
                              image_path,
                              output_path);
  undistorter.Run();

  // Verify PMVS output structure was created (under pmvs/ subdirectory).
  EXPECT_TRUE(ExistsDir(output_path / "pmvs"));
  EXPECT_TRUE(ExistsDir(output_path / "pmvs" / "models"));
  EXPECT_TRUE(ExistsDir(output_path / "pmvs" / "txt"));
  EXPECT_TRUE(ExistsDir(output_path / "pmvs" / "visualize"));

  // Verify undistorted images were written with numbered names.
  // PMVS writes images as 00000000.jpg, 00000001.jpg, etc.
  const size_t num_images = reconstruction.NumRegImages();
  for (size_t i = 0; i < num_images; ++i) {
    const std::string image_name = StringPrintf("%08zu.jpg", i);
    EXPECT_TRUE(ExistsFile(output_path / "pmvs" / "visualize" / image_name));
  }
}

TEST(CMPMVSUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "cmpmvs_output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Run CMP-MVS undistorter.
  CMPMVSUndistorter undistorter(CMPMVSUndistorter::Options(),
                                UndistortCameraOptions(),
                                reconstruction,
                                image_path,
                                output_path);
  undistorter.Run();

  // Verify CMP-MVS output structure was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify undistorted images were written with sequential numbering.
  // CMP-MVS writes images as 00001.jpg, 00002.jpg, etc.
  const size_t num_images = reconstruction.NumRegImages();
  for (size_t i = 1; i <= num_images; ++i) {
    const std::string image_name = StringPrintf("%05zu.jpg", i);
    EXPECT_TRUE(ExistsFile(output_path / image_name));
  }
}

TEST(StandaloneImageUndistorter, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "pure_output";
  CreateDirIfNotExists(image_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  StandaloneImageUndistorter::Options options;
  for (const auto& [_, image] : reconstruction.Images()) {
    options.image_names_and_cameras.emplace_back(image.Name(),
                                                 *image.CameraPtr());
  }

  // Run standalone image undistorter.
  StandaloneImageUndistorter undistorter(
      options, UndistortCameraOptions(), image_path, output_path);
  undistorter.Run();

  // Verify output directory was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify undistorted images were written.
  for (const auto& [image_name, camera] : options.image_names_and_cameras) {
    EXPECT_TRUE(ExistsFile(output_path / image_name));
  }
}

TEST(StereoImageRectifier, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_path = temp_dir / "input_images";
  const auto output_path = temp_dir / "stereo_output";
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(output_path);

  // Create synthetic reconstruction with dummy images.
  const Reconstruction reconstruction =
      CreateSyntheticReconstructionWithBitmaps(image_path);

  // Create stereo pair from first two images.
  StereoImageRectifier::Options options;
  const std::vector<image_t> image_ids = reconstruction.RegImageIds();
  ASSERT_GE(image_ids.size(), 2);
  options.stereo_pairs.emplace_back(image_ids[0], image_ids[1]);

  // Run stereo image rectifier.
  StereoImageRectifier rectifier(options,
                                 UndistortCameraOptions(),
                                 reconstruction,
                                 image_path,
                                 output_path);
  rectifier.Run();

  // Verify output directory was created.
  EXPECT_TRUE(ExistsDir(output_path));

  // Verify rectified images were written.
  // StereoImageRectifier creates a subdirectory for each stereo pair.
  const auto& image1 = reconstruction.Image(options.stereo_pairs[0].first);
  const auto& image2 = reconstruction.Image(options.stereo_pairs[0].second);
  const std::string stereo_pair_name =
      StringPrintf("%s-%s", image1.Name().c_str(), image2.Name().c_str());
  EXPECT_TRUE(ExistsDir(output_path / stereo_pair_name));
  EXPECT_TRUE(ExistsFile(output_path / stereo_pair_name / image1.Name()));
  EXPECT_TRUE(ExistsFile(output_path / stereo_pair_name / image2.Name()));
}

}  // namespace
}  // namespace colmap
