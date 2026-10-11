// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace colmap::mvs {
namespace {

class OwnedHostSnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    workspace_path_ = CreateTestDir();
    CreateDirIfNotExists(workspace_path_ / "sparse");
    CreateDirIfNotExists(workspace_path_ / "images");
    CreateDirIfNotExists(workspace_path_ / "stereo" / "depth_maps");
    CreateDirIfNotExists(workspace_path_ / "stereo" / "normal_maps");

    SyntheticDatasetOptions synthetic;
    synthetic.num_rigs = 1;
    synthetic.num_cameras_per_rig = 1;
    synthetic.num_frames_per_rig = 2;
    synthetic.camera_width = kWidth;
    synthetic.camera_height = kHeight;
    synthetic.camera_model_id = SimplePinholeCameraModel::model_id;
    synthetic.camera_params = {12, 5, 4};
    Reconstruction reconstruction;
    SynthesizeDataset(synthetic, &reconstruction);
    reconstruction.Write(workspace_path_ / "sparse");
    image_names_[0] = reconstruction.Image(1).Name();
    image_names_[1] = reconstruction.Image(2).Name();
    for (size_t image = 0; image != 2; ++image) WriteImage(image);

    config_path_ = workspace_path_ / "stereo" / "patch-match.cfg";
    std::ofstream config(config_path_);
    ASSERT_TRUE(config.is_open());
    config << image_names_[0] << '\n' << image_names_[1] << '\n';
    config.close();
    ASSERT_TRUE(config.good());
  }

  void TearDown() override { std::filesystem::remove_all(workspace_path_); }

  void WriteImage(size_t image) {
    Bitmap bitmap(kWidth, kHeight, false);
    for (int y = 0; y != kHeight; ++y) {
      for (int x = 0; x != kWidth; ++x) {
        ASSERT_TRUE(
            bitmap.SetPixel(x, y, BitmapColor<uint8_t>(Pixel(image, x, y))));
      }
    }
    ASSERT_TRUE(bitmap.Write(workspace_path_ / "images" / image_names_[image]));
  }

  static uint8_t Pixel(size_t image, int x, int y) {
    return static_cast<uint8_t>(20 + 80 * image + x + 2 * y);
  }

  PatchMatchOptions Options() const {
    PatchMatchOptions options;
    options.gpu_index = "0";
    options.num_threads = 1;
    options.geom_consistency = false;
    options.depth_min = 1.25;
    options.depth_max = 8.5;
    options.window_radius = 5;
    options.window_step = 1;
    options.num_iterations = 2;
    options.num_samples = 3;
    options.sigma_spatial = -1;
    options.sigma_color = 0.25;
    options.cache_size = 0.01;
    return options;
  }

  std::unique_ptr<PatchMatchController> Controller(
      const PatchMatchOptions& options) const {
    return std::make_unique<PatchMatchController>(
        options,
        workspace_path_,
        "COLMAP",
        "",
        config_path_,
        PatchMatchExecution::kOwnedCuda);
  }

  void ExpectSnapshot(const PatchMatchController::OwnedPreparedJob& prepared) {
    ASSERT_NE(prepared.patch_match, nullptr);
    const auto& engine = *prepared.patch_match;
    const auto& options = engine.GetOptions();
    EXPECT_EQ(options.gpu_index, "0");
    EXPECT_EQ(options.num_threads, 1);
    EXPECT_FALSE(options.geom_consistency);
    EXPECT_DOUBLE_EQ(options.depth_min, 1.25);
    EXPECT_DOUBLE_EQ(options.depth_max, 8.5);
    EXPECT_DOUBLE_EQ(options.sigma_spatial, 5);
    EXPECT_DOUBLE_EQ(options.sigma_color, 0.25);
    EXPECT_EQ(options.window_radius, 5);
    EXPECT_EQ(options.window_step, 1);
    EXPECT_EQ(options.num_iterations, 2);
    EXPECT_EQ(options.num_samples, 3);
    EXPECT_EQ(options.filter_min_num_consistent, 1);
    const auto& problem = engine.GetProblem();
    ASSERT_NE(problem.images, nullptr);
    EXPECT_EQ(problem.depth_maps, nullptr);
    EXPECT_EQ(problem.normal_maps, nullptr);
    ASSERT_EQ(problem.images->size(), 2);
    ASSERT_EQ(problem.src_image_idxs.size(), 1);
    const std::array<int, 2> indices = {problem.ref_image_idx,
                                      problem.src_image_idxs[0]};
    EXPECT_NE(indices[0], indices[1]);
    for (size_t image = 0; image != 2; ++image) {
      ASSERT_GE(indices[image], 0);
      ASSERT_LT(static_cast<size_t>(indices[image]), problem.images->size());
      const auto& input = problem.images->at(indices[image]);
      EXPECT_EQ(input.GetPath().filename(), image_names_[image]);
      EXPECT_EQ(input.GetWidth(), kWidth);
      EXPECT_EQ(input.GetHeight(), kHeight);
      const auto& bitmap = input.GetBitmap();
      ASSERT_EQ(bitmap.Width(), kWidth);
      ASSERT_EQ(bitmap.Height(), kHeight);
      EXPECT_TRUE(bitmap.IsGrey());
      for (int y = 0; y != kHeight; ++y) {
        for (int x = 0; x != kWidth; ++x) {
          const auto pixel = bitmap.GetPixel(x, y);
          ASSERT_TRUE(pixel.has_value());
          EXPECT_EQ(pixel->r, Pixel(image, x, y));
        }
      }
    }
    EXPECT_EQ(prepared.depth_map_path,
              workspace_path_ / "stereo" / "depth_maps" /
                  (image_names_[0] + ".photometric.bin"));
    EXPECT_EQ(prepared.normal_map_path,
              workspace_path_ / "stereo" / "normal_maps" /
                  (image_names_[0] + ".photometric.bin"));
    // Check is CPU validation only; do not construct or run a CUDA backend.
    EXPECT_NO_THROW(engine.Check());
  }

  static constexpr int kWidth = 10;
  static constexpr int kHeight = 8;
  std::filesystem::path workspace_path_;
  std::filesystem::path config_path_;
  std::array<std::string, 2> image_names_;
};

TEST_F(OwnedHostSnapshotTest, InputsSurviveControllersAndWorkspaceDeletion) {
  std::array<std::unique_ptr<PatchMatchController::OwnedPreparedJob>, 2>
      prepared;
  int checks = 0;
  int metadata_calls = 0;
  PatchMatchController::OwnedPairHooks hooks;
  hooks.check_inputs = [&](size_t job,
                           const PatchMatchOptions& options,
                           const PatchMatch::Problem& problem) {
    EXPECT_EQ(job, static_cast<size_t>(checks));
    ++checks;
    EXPECT_FALSE(options.geom_consistency);
    EXPECT_EQ(problem.src_image_idxs.size(), 1);
  };
  hooks.metadata =
      [&](size_t, const PatchMatchOptions&, const PatchMatch::Problem&) {
        ++metadata_calls;
      };
  for (size_t job = 0; job != 2; ++job) {
    auto controller = Controller(Options());
    prepared[job] = controller->PrepareOwnedPairJob(job, hooks);
    // Controller (including its workspace and I/O pool) dies each iteration.
  }
  EXPECT_EQ(checks, 2);
  EXPECT_EQ(metadata_calls, 0);
  std::filesystem::remove_all(workspace_path_);
  EXPECT_FALSE(std::filesystem::exists(workspace_path_));
  for (const auto& job : prepared) {
    ASSERT_NE(job, nullptr);
    ExpectSnapshot(*job);
  }
  const auto& a = prepared[0]->patch_match->GetProblem();
  const auto& b = prepared[1]->patch_match->GetProblem();
  EXPECT_NE(a.images, b.images);
  EXPECT_EQ(a.ref_image_idx, b.ref_image_idx);
  EXPECT_EQ(a.src_image_idxs, b.src_image_idxs);
  for (size_t image = 0; image != a.images->size(); ++image) {
    const auto& left = a.images->at(image);
    const auto& right = b.images->at(image);
    EXPECT_EQ(left.GetBitmap().RowMajorData(),
              right.GetBitmap().RowMajorData());
    for (int element = 0; element != 9; ++element) {
      EXPECT_FLOAT_EQ(left.GetK()[element], right.GetK()[element]);
      EXPECT_FLOAT_EQ(left.GetR()[element], right.GetR()[element]);
    }
    for (int element = 0; element != 3; ++element) {
      EXPECT_FLOAT_EQ(left.GetT()[element], right.GetT()[element]);
    }
  }
}

TEST_F(OwnedHostSnapshotTest, UnsupportedModesRejectBeforeDecodeOrDeviceWork) {
  // With the workspace absent, these failures must be eligibility failures,
  // not successful preparation. No Run/PrepareOwned entrypoint is used.
  std::filesystem::remove_all(workspace_path_);
  for (const std::string gpu_index : {"-1", "0,1"}) {
    SCOPED_TRACE(gpu_index);
    auto options = Options();
    options.gpu_index = gpu_index;
    auto controller = Controller(options);
    try {
      controller->PrepareOwnedPairJob(0, {});
      ADD_FAILURE() << "Unsupported GPU selection was accepted";
    } catch (const std::invalid_argument& error) {
      const std::string message = error.what();
      EXPECT_NE(message.find(gpu_index == "-1" ? "explicit_gpu_indices.front()"
                                               : "explicit_gpu_indices.size()"),
                std::string::npos);
    }
  }
  auto options = Options();
  options.geom_consistency = true;
  try {
    auto controller = Controller(options);
    controller->PrepareOwnedPairJob(0, {});
    ADD_FAILURE() << "Geometric preparation was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("photo window"),
              std::string::npos);
  }
}

TEST_F(OwnedHostSnapshotTest,
       DecodeFailureDrainsInputsAndAllowsFreshPreparation) {
  {
    std::ofstream corrupt(workspace_path_ / "images" / image_names_[0],
                          std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(corrupt.is_open());
    corrupt << "not a PNG image";
    corrupt.close();
    ASSERT_TRUE(corrupt.good());
  }
  auto controller = Controller(Options());
  EXPECT_THROW(controller->PrepareOwnedPairJob(0, {}), std::exception);
  // The same controller must have relinquished workspace/pool owners on error.
  WriteImage(0);
  auto prepared = controller->PrepareOwnedPairJob(1, {});
  controller.reset();
  std::filesystem::remove_all(workspace_path_);
  ASSERT_NE(prepared, nullptr);
  ExpectSnapshot(*prepared);
}

}  // namespace
}  // namespace colmap::mvs
