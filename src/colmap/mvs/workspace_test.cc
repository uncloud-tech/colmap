// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/workspace.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <exception>
#include <filesystem>
#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

class ParameterizedWorkspaceTests
    : public ::testing::TestWithParam<std::function<std::unique_ptr<Workspace>(
          const Workspace::Options&)>> {
 protected:
  void SetUp() override {
    temp_dir_ = CreateTestDir();
    CreateDirIfNotExists(temp_dir_ / "sparse");
    CreateDirIfNotExists(temp_dir_ / "images");
    CreateDirIfNotExists(temp_dir_ / "stereo");
    CreateDirIfNotExists(temp_dir_ / "stereo" / "depth_maps");
    CreateDirIfNotExists(temp_dir_ / "stereo" / "normal_maps");

    SyntheticDatasetOptions options;
    options.num_rigs = 1;
    options.num_cameras_per_rig = 1;
    options.num_frames_per_rig = 1;
    options.camera_width = 10;
    options.camera_height = 5;
    Reconstruction reconstruction;
    SynthesizeDataset(options, &reconstruction);
    reconstruction.Write(temp_dir_ / "sparse");

    image_name_ = reconstruction.Image(1).Name();

    Mat<float> depth_map(options.camera_width, options.camera_height, 1);
    depth_map.Fill(1.0f);
    depth_map.Write(temp_dir_ / "stereo" / "depth_maps" /
                    (image_name_ + ".geometric.bin"));

    Mat<float> normal_map(options.camera_width, options.camera_height, 3);
    normal_map.Fill(1.0f);
    normal_map.Write(temp_dir_ / "stereo" / "normal_maps" /
                     (image_name_ + ".geometric.bin"));

    Bitmap bitmap(options.camera_width, options.camera_height, true);
    bitmap.Fill(BitmapColor<uint8_t>(0, 0, 0));
    bitmap.Write(temp_dir_ / "images" / image_name_);
  }

  Workspace::Options GetOptions() {
    Workspace::Options options;
    options.workspace_path = temp_dir_;
    options.workspace_format = "COLMAP";
    options.input_type = "geometric";
    return options;
  }

  std::filesystem::path temp_dir_;
  std::string image_name_;
};

TEST_P(ParameterizedWorkspaceTests, GetData) {
  auto workspace = GetParam()(GetOptions());
  const auto& model = workspace->GetModel();
  EXPECT_EQ(model.images.size(), 1);
  workspace->Load({image_name_});
  EXPECT_TRUE(workspace->HasBitmap(0));
  EXPECT_THAT(workspace->GetBitmapPath(0).string(),
              testing::HasSubstr(image_name_));
  EXPECT_FALSE(workspace->GetBitmap(0).IsEmpty());
  EXPECT_TRUE(workspace->HasDepthMap(0));
  EXPECT_THAT(workspace->GetDepthMapPath(0).string(),
              testing::HasSubstr(image_name_));
  EXPECT_GT(workspace->GetDepthMap(0).GetNumBytes(), 0);
  EXPECT_TRUE(workspace->HasNormalMap(0));
  EXPECT_THAT(workspace->GetNormalMapPath(0).string(),
              testing::HasSubstr(image_name_));
  EXPECT_GT(workspace->GetNormalMap(0).GetNumBytes(), 0);
}

TEST_P(ParameterizedWorkspaceTests, MaxImageSize) {
  Workspace::Options options = GetOptions();
  options.max_image_size = 4;
  auto workspace = GetParam()(options);
  workspace->Load({image_name_});
  ASSERT_EQ(workspace->GetModel().images.size(), 1);
  EXPECT_EQ(workspace->GetModel().images[0].GetWidth(), 4);
  EXPECT_EQ(workspace->GetModel().images[0].GetHeight(), 2);
  EXPECT_EQ(workspace->GetBitmap(0).Width(), 4);
  EXPECT_EQ(workspace->GetBitmap(0).Height(), 2);
  EXPECT_EQ(workspace->GetDepthMap(0).GetWidth(), 4);
  EXPECT_EQ(workspace->GetDepthMap(0).GetHeight(), 2);
  EXPECT_EQ(workspace->GetNormalMap(0).GetWidth(), 4);
  EXPECT_EQ(workspace->GetNormalMap(0).GetHeight(), 2);
}

TEST_P(ParameterizedWorkspaceTests, BitmapReadFailureBeforeRescaleAndRetry) {
  Workspace::Options options = GetOptions();
  options.max_image_size = 4;
  options.num_threads = 1;
  auto workspace = GetParam()(options);
  const auto image_path = workspace->GetBitmapPath(0);

  Bitmap original;
  ASSERT_TRUE(original.Read(image_path));
  ASSERT_EQ(original.Width(), 10);
  ASSERT_EQ(original.Height(), 5);
  ASSERT_THAT(original.RowMajorData(), testing::Each(uint8_t{0}));

  {
    // Keep the file present so Workspace::Load does not skip the failed read.
    // An invalid header fails ImageInput::open, not pixel payload decoding.
    std::ofstream corrupt(image_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(corrupt.is_open());
    corrupt << "not an image";
    corrupt.close();
    ASSERT_TRUE(corrupt.good());
  }
  ASSERT_TRUE(workspace->HasBitmap(0));
  EXPECT_THROW(
      {
        // Load reads eagerly for Workspace and is a no-op for CachedWorkspace.
        workspace->Load({image_name_});
        workspace->GetBitmap(0);
      },
      std::exception);

  ASSERT_TRUE(original.Write(image_path));
  // Retry on the same workspace without clearing its cache or rebuilding it.
  ASSERT_NO_THROW(workspace->Load({image_name_}));
  const Bitmap* bitmap = nullptr;
  ASSERT_NO_THROW(bitmap = &workspace->GetBitmap(0));
  ASSERT_FALSE(bitmap->IsEmpty());
  EXPECT_EQ(bitmap->Width(), 4);
  EXPECT_EQ(bitmap->Height(), 2);
  ASSERT_EQ(bitmap->RowMajorData().size(), 4 * 2 * 3);
  EXPECT_THAT(bitmap->RowMajorData(), testing::Each(uint8_t{0}));

  // Rescaling the workspace bitmap must not resize the repaired input file.
  Bitmap input;
  ASSERT_TRUE(input.Read(image_path));
  EXPECT_EQ(input.Width(), 10);
  EXPECT_EQ(input.Height(), 5);
  EXPECT_EQ(input.RowMajorData(), original.RowMajorData());
}

TEST_P(ParameterizedWorkspaceTests, Load) {
  auto workspace = GetParam()(GetOptions());
  workspace->Load({image_name_});
}

INSTANTIATE_TEST_SUITE_P(WorkspaceTests,
                         ParameterizedWorkspaceTests,
                         ::testing::Values(
                             [](const Workspace::Options& options) {
                               return std::make_unique<Workspace>(options);
                             },
                             [](const Workspace::Options& options) {
                               return std::make_unique<CachedWorkspace>(
                                   options);
                             }));

}  // namespace
}  // namespace mvs
}  // namespace colmap
