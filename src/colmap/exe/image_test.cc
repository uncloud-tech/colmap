// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/image.h"

#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <boost/property_tree/json_parser.hpp>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Exercise the real parsed command, not a duplicate projection implementation.
int Undistort(std::vector<std::string> args) {
  args.insert(args.begin(), "image_undistorter");
  std::vector<char*> argv;
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);
  return RunImageUndistorter(static_cast<int>(args.size()), argv.data());
}

class PreviewStdout {
 public:
  PreviewStdout() : saved_(std::cout.rdbuf(stream_.rdbuf())) {}
  ~PreviewStdout() { std::cout.rdbuf(saved_); }
  std::string Text() const { return stream_.str(); }

 private:
  std::ostringstream stream_;
  std::streambuf* saved_;
};

std::map<std::string, std::string> ImageTreeSnapshot(
    const std::filesystem::path& root) {
  std::map<std::string, std::string> result;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(root)) {
    std::string content;
    if (entry.is_regular_file()) {
      std::ifstream file(entry.path(), std::ios::binary);
      content.assign(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
    }
    content +=
        "\nmtime=" +
        std::to_string(entry.last_write_time().time_since_epoch().count());
    result.emplace(entry.path().lexically_relative(root).string(), content);
  }
  return result;
}

class UndistortionPreview : public ::testing::Test {
 protected:
  void SetUp() override {
    root = CreateTestDir();
    input = root / "model";
    photos = root / "photos";
    planned = root / "absent-parent" / "planned";
    CreateDirIfNotExists(input);
    CreateDirIfNotExists(photos);
    SyntheticDatasetOptions options;
    options.num_rigs = 1;
    options.num_cameras_per_rig = 2;
    options.num_frames_per_rig = 2;
    options.num_points3D = 4;
    options.camera_width = 16;
    options.camera_height = 12;
    options.camera_params = {20, 8, 6, 0.05};
    SynthesizeDataset(options, &reconstruction);
    reconstruction.Image(reconstruction.RegImageIds().front())
        .SetName("quoted\"image.png");
    reconstruction.Write(input);
  }

  std::vector<std::string> Args() const {
    return {"--image_path",
            photos.string(),
            "--input_path",
            input.string(),
            "--output_path",
            planned.string()};
  }

  std::filesystem::path root, input, photos, planned;
  Reconstruction reconstruction;
};

TEST_F(UndistortionPreview, GeometryJsonWithoutReadingPhotosOrCreatingOutput) {
  // Invalid encoded bytes must be irrelevant to a model-only preview.
  reconstruction.CreateImageDirs(photos);
  for (const auto& [id, image] : reconstruction.Images()) {
    std::ofstream(photos / image.Name()) << "not an image";
  }
  const auto before = ImageTreeSnapshot(root);
  auto args = Args();
  args.insert(args.end(), {"--dry_run", "1"});
  PreviewStdout capture;
  ASSERT_EQ(Undistort(args), EXIT_SUCCESS);
  std::istringstream stream(capture.Text());
  boost::property_tree::ptree json;
  ASSERT_NO_THROW(boost::property_tree::read_json(stream, json));
  EXPECT_EQ(json.get<int>("schema_version"), 1);
  EXPECT_EQ(json.get<std::string>("kind"), "undistortion_target_preview");
  EXPECT_EQ(json.get<std::string>("source_basis"), "native_camera_geometry");
  EXPECT_EQ(json.get<std::string>("serialization_order"),
            "image_id_ascending_not_execution_or_source_order");
  EXPECT_FALSE(json.get<bool>("decoded_images_verified"));
  EXPECT_FALSE(json.get<bool>("target_output_verified"));
  EXPECT_FALSE(json.get<bool>("execution_authorized"));
  EXPECT_EQ(json.get<std::string>("physical_fit"), "null");
  EXPECT_NE(capture.Text().find("\"physical_fit\":null"), std::string::npos);
  EXPECT_NE(capture.Text().find("\"decoded_images_verified\":false"),
            std::string::npos);
  EXPECT_EQ(json.get<int>("options.max_image_size"), -1);
  EXPECT_DOUBLE_EQ(json.get<double>("options.min_scale"), 0.2);
  EXPECT_DOUBLE_EQ(json.get<double>("options.max_scale"), 2.0);
  EXPECT_DOUBLE_EQ(json.get<double>("options.blank_pixels"), 0.0);
  EXPECT_EQ(json.get_child("images").size(), 4);
  image_t previous = 0;
  for (const auto& item : json.get_child("images")) {
    const auto& row = item.second;
    const auto id = row.get<image_t>("image_id");
    EXPECT_GT(id, previous);
    previous = id;
    EXPECT_EQ(row.get<std::string>("name"), reconstruction.Image(id).Name());
    EXPECT_EQ(row.get<camera_t>("camera_id"),
              reconstruction.Image(id).CameraId());
    EXPECT_EQ(row.get<int>("model_width"), 16);
    EXPECT_EQ(row.get<int>("model_height"), 12);
    EXPECT_GT(row.get<int>("target_width"), 0);
    EXPECT_GT(row.get<int>("target_height"), 0);
    EXPECT_EQ(row.get<std::string>("target_camera_model"), "PINHOLE");
    EXPECT_EQ(row.get_child("target_camera_params").size(), 4);
  }
  EXPECT_EQ(ImageTreeSnapshot(root), before);
}

TEST_F(UndistortionPreview, PreservesJsonEscapingForImageNames) {
  // Native binary image names are NUL-terminated; all other control bytes
  // can reach the preview without ever being used as filesystem paths.
  std::string name = "quote\"backslash\\/";
  for (char c = 1; c < 0x20; ++c) {
    name.push_back(c);
  }
  name += "\x7f\xc3\xa9\xe9\x9b\xaa.png";
  reconstruction.Image(reconstruction.RegImageIds().front()).SetName(name);
  reconstruction.Write(input);
  const auto before = ImageTreeSnapshot(root);
  auto args = Args();
  args.insert(args.end(), {"--dry_run", "1"});
  PreviewStdout capture;
  ASSERT_EQ(Undistort(args), EXIT_SUCCESS);
  const std::string expected =
      "\"name\":\"quote\\\"backslash\\\\/"
      "\\u0001\\u0002\\u0003\\u0004\\u0005\\u0006\\u0007\\b\\t\\n"
      "\\u000b\\f\\r\\u000e\\u000f\\u0010\\u0011\\u0012\\u0013"
      "\\u0014\\u0015\\u0016\\u0017\\u0018\\u0019\\u001a\\u001b"
      "\\u001c\\u001d\\u001e\\u001f\x7f\xc3\xa9\xe9\x9b\xaa.png\"";
  EXPECT_NE(capture.Text().find(expected), std::string::npos);
  std::istringstream stream(capture.Text());
  boost::property_tree::ptree json;
  ASSERT_NO_THROW(boost::property_tree::read_json(stream, json));
  bool found = false;
  for (const auto& item : json.get_child("images")) {
    found |= item.second.get<std::string>("name") == name;
  }
  EXPECT_TRUE(found);
  EXPECT_EQ(ImageTreeSnapshot(root), before);
}

TEST_F(UndistortionPreview, MatchesOrdinaryCpuOutputForDefaultAndCappedRoi) {
  reconstruction.CreateImageDirs(photos);
  Bitmap bitmap(16, 12, true);
  bitmap.Fill(BitmapColor<uint8_t>(20, 40, 60));
  for (const auto& [id, image] : reconstruction.Images()) {
    ASSERT_TRUE(bitmap.Write(photos / image.Name()));
  }
  for (const bool capped : {false, true}) {
    auto args = Args();
    if (capped) {
      args.insert(args.end(),
                  {"--max_image_size",
                   "9",
                   "--roi_min_x",
                   "0.1",
                   "--roi_max_y",
                   "0.9",
                   "--blank_pixels",
                   "0.5",
                   "--min_scale",
                   "0.5",
                   "--max_scale",
                   "1.5"});
    }
    auto preview_args = args;
    preview_args.insert(preview_args.end(), {"--dry_run", "1"});
    const auto before = ImageTreeSnapshot(root);
    boost::property_tree::ptree json;
    {
      PreviewStdout capture;
      ASSERT_EQ(Undistort(preview_args), EXIT_SUCCESS);
      std::istringstream stream(capture.Text());
      ASSERT_NO_THROW(boost::property_tree::read_json(stream, json));
    }
    EXPECT_EQ(ImageTreeSnapshot(root), before);
    EXPECT_EQ(json.get<int>("options.max_image_size"), capped ? 9 : -1);
    EXPECT_DOUBLE_EQ(json.get<double>("options.roi_min_x"), capped ? 0.1 : 0.0);
    EXPECT_DOUBLE_EQ(json.get<double>("options.roi_max_y"), capped ? 0.9 : 1.0);
    // Separate ordinary execution, omitting --dry_run to retain default
    // control.
    const auto actual = root / (capped ? "actual-cap" : "actual-default");
    args[5] = actual.string();
    args.insert(args.end(), {"--num_threads", "1"});
    ASSERT_EQ(Undistort(args), EXIT_SUCCESS);
    for (const auto& item : json.get_child("images")) {
      const auto& row = item.second;
      Bitmap output;
      ASSERT_TRUE(
          output.Read(actual / "images" / row.get<std::string>("name")));
      EXPECT_EQ(output.Width(), row.get<int>("target_width"));
      EXPECT_EQ(output.Height(), row.get<int>("target_height"));
    }
  }
}

TEST_F(UndistortionPreview, IncludesOnlyRegisteredImages) {
  reconstruction.DeRegisterFrame(reconstruction.RegFrameIds().back());
  reconstruction.Write(input);
  auto args = Args();
  args.insert(args.end(), {"--dry_run", "1"});
  PreviewStdout capture;
  ASSERT_EQ(Undistort(args), EXIT_SUCCESS);
  std::istringstream stream(capture.Text());
  boost::property_tree::ptree json;
  ASSERT_NO_THROW(boost::property_tree::read_json(stream, json));
  EXPECT_EQ(json.get_child("images").size(), reconstruction.NumRegImages());
  for (const auto& item : json.get_child("images")) {
    EXPECT_TRUE(
        reconstruction.Image(item.second.get<image_t>("image_id")).HasPose());
  }
}

TEST_F(UndistortionPreview, RefusesUnsupportedModesSubsetsAndInvalidOptions) {
  const auto before = ImageTreeSnapshot(root);
  const std::vector<std::vector<std::string>> cases = {
      {"--output_type", "PMVS"},
      {"--output_type", "CMP-MVS"},
      {"--output_type", "unknown"},
      {"--image_list_path", "absent-list"},
      {"--max_image_size", "0"},
      {"--blank_pixels", "2"},
      {"--min_scale", "0"},
      {"--max_scale", "nan"},
      {"--blank_pixels", "inf"},
      {"--roi_min_x", "0.9", "--roi_max_x", "0.1"}};
  for (const auto& extra : cases) {
    auto args = Args();
    args.insert(args.end(), {"--dry_run", "1"});
    args.insert(args.end(), extra.begin(), extra.end());
    PreviewStdout capture;
    testing::internal::CaptureStderr();
    EXPECT_EQ(Undistort(args), EXIT_FAILURE);
    EXPECT_FALSE(testing::internal::GetCapturedStderr().empty());
    EXPECT_TRUE(capture.Text().empty());
    EXPECT_EQ(ImageTreeSnapshot(root), before);
  }
}

TEST_F(UndistortionPreview,
       RefusesMissingMalformedEmptyAndInvalidGeometryModels) {
  for (const std::string problem : {"missing",
                                    "malformed",
                                    "empty",
                                    "zero-width",
                                    "zero-focal",
                                    "nonfinite"}) {
    const auto bad = root / problem;
    if (problem != "missing") {
      CreateDirIfNotExists(bad);
      if (problem == "malformed") {
        std::ofstream(bad / "cameras.txt") << "broken camera row\n";
        std::ofstream(bad / "images.txt");
        std::ofstream(bad / "points3D.txt");
      } else if (problem == "empty") {
        Reconstruction().Write(bad);
      } else {
        auto invalid = reconstruction;
        auto& camera = invalid.Camera(invalid.Cameras().begin()->first);
        if (problem == "zero-width") camera.width = 0;
        if (problem == "zero-focal") camera.params[0] = 0;
        if (problem == "nonfinite")
          camera.params[0] = std::numeric_limits<double>::infinity();
        invalid.Write(bad);
      }
    }
    auto args = Args();
    args[3] = bad.string();
    args.insert(args.end(), {"--dry_run", "1"});
    const auto before = ImageTreeSnapshot(root);
    PreviewStdout capture;
    testing::internal::CaptureStderr();
    EXPECT_EQ(Undistort(args), EXIT_FAILURE) << problem;
    EXPECT_FALSE(testing::internal::GetCapturedStderr().empty()) << problem;
    EXPECT_TRUE(capture.Text().empty()) << problem;
    EXPECT_EQ(ImageTreeSnapshot(root), before) << problem;
  }
}

TEST(ImageRegistrator, Nominal) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());
  database.reset();

  ASSERT_EQ(reconstruction.NumRegImages(), 3);
  reconstruction.DeRegisterFrame(reconstruction.RegFrameIds().back());
  ASSERT_EQ(reconstruction.NumRegImages(), 2);

  const auto input_path = test_path / "input";
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(input_path);
  CreateDirIfNotExists(output_path);
  reconstruction.Write(input_path);

  std::vector<std::string> args = {
      "image_registrator",
      "--database_path",
      database_path.string(),
      "--input_path",
      input_path.string(),
      "--output_path",
      output_path.string(),
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }

  EXPECT_EQ(RunImageRegistrator(argv.size(), argv.data()), EXIT_SUCCESS);

  Reconstruction output_reconstruction;
  output_reconstruction.Read(output_path);
  EXPECT_EQ(output_reconstruction.NumRegImages(), 3);
}

TEST(ImageRegistrator, RegistersImageWithDisconnectedRegisteredImage) {
  const auto test_path = CreateTestDir();
  const auto database_path = test_path / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_options;
  synthetic_options.num_rigs = 1;
  synthetic_options.num_cameras_per_rig = 1;
  synthetic_options.num_frames_per_rig = 3;
  synthetic_options.num_points3D = 100;
  SynthesizeDataset(synthetic_options, &reconstruction, database.get());

  ASSERT_EQ(reconstruction.NumRegImages(), 3);
  const auto image_ids = reconstruction.RegImageIds();
  const image_t disconnected_image_id = image_ids.front();
  for (const image_t image_id : image_ids) {
    if (image_id != disconnected_image_id) {
      database->DeleteTwoViewGeometry(disconnected_image_id, image_id);
    }
  }
  database.reset();

  const image_t image_to_register_id = image_ids.back();
  reconstruction.DeRegisterFrame(
      reconstruction.Image(image_to_register_id).FrameId());
  ASSERT_EQ(reconstruction.NumRegImages(), 2);

  const auto input_path = test_path / "input";
  const auto output_path = test_path / "output";
  CreateDirIfNotExists(input_path);
  CreateDirIfNotExists(output_path);
  reconstruction.Write(input_path);

  const auto image_list_path = test_path / "image-list.txt";
  {
    std::ofstream image_list_file(image_list_path);
    image_list_file << reconstruction.Image(image_to_register_id).Name()
                    << '\n';
  }

  std::vector<std::string> args = {
      "image_registrator",
      "--database_path",
      database_path.string(),
      "--input_path",
      input_path.string(),
      "--output_path",
      output_path.string(),
      "--Mapper.image_list_path",
      image_list_path.string(),
  };
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }

  EXPECT_EQ(RunImageRegistrator(argv.size(), argv.data()), EXIT_SUCCESS);

  Reconstruction output_reconstruction;
  output_reconstruction.Read(output_path);
  EXPECT_EQ(output_reconstruction.NumRegImages(), 3);
  EXPECT_TRUE(output_reconstruction.Image(image_to_register_id).HasPose());
}

}  // namespace
}  // namespace colmap
