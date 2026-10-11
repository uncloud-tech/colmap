// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/workspace.h"
#include "colmap/mvs/model.h"
#include "colmap/util/hash_containers.h"

#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"
#include "colmap/util/threading.h"

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
    CreateDirIfNotExists(workspace_path_ / "stereo");
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

TEST_F(OwnedHostSnapshotTest, IoOneOverrideKeepsMultiSourceOrderAndAutomaticDepth) {
  SyntheticDatasetOptions synthetic;
  synthetic.num_rigs = 1;
  synthetic.num_cameras_per_rig = 1;
  synthetic.num_frames_per_rig = 3;
  synthetic.camera_width = kWidth;
  synthetic.camera_height = kHeight;
  synthetic.camera_model_id = SimplePinholeCameraModel::model_id;
  synthetic.camera_params = {12, 5, 4};
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic, &reconstruction);
  reconstruction.Write(workspace_path_ / "sparse");
  const auto third_name = reconstruction.Image(3).Name();
  Bitmap bitmap(kWidth, kHeight, false);
  bitmap.Fill(BitmapColor<uint8_t>(120));
  ASSERT_TRUE(bitmap.Write(workspace_path_ / "images" / third_name));
  {
    std::ofstream config(config_path_);
    config << image_names_[0] << '\n' << third_name << ',' << image_names_[1] << '\n';
    ASSERT_TRUE(config.good());
  }
  Model model;
  model.ReadFromCOLMAP(workspace_path_, "sparse", "images");
  const int reference = model.GetImageIdx(image_names_[0]);
  const auto depth = model.ComputeDepthRanges().at(reference);
  ASSERT_GT(depth.first, 0);
  ASSERT_GT(depth.second, depth.first);
  const std::vector<int> configured{model.GetImageIdx(third_name),
                                    model.GetImageIdx(image_names_[1])};
  FlatHashSet<int> used(configured.begin(), configured.end());
  used.insert(reference);
  std::vector<int> effective;
  for (const int index : used) if (index != reference) effective.push_back(index);
  auto options = Options();
  options.num_threads = 2;
  options.depth_min = options.depth_max = -1;
  auto synchronous = Controller(options)->PrepareOwnedPairJob(0, {});
  auto lookahead = Controller(options)->PrepareOwnedPairJob(1, {}, 1);
  for (const auto* prepared : {synchronous.get(), lookahead.get()}) {
    const auto& engine = *prepared->patch_match;
    EXPECT_EQ(engine.GetOptions().num_threads, 2);
    EXPECT_EQ(engine.GetOptions().depth_min, depth.first);
    EXPECT_EQ(engine.GetOptions().depth_max, depth.second);
    EXPECT_EQ(engine.GetProblem().ref_image_idx, reference);
    EXPECT_EQ(engine.GetProblem().src_image_idxs, effective);
  }
  std::filesystem::remove_all(workspace_path_);
  for (size_t i = 0; i < model.images.size(); ++i) {
    EXPECT_EQ(synchronous->patch_match->GetProblem().images->at(i).GetBitmap().RowMajorData(),
              lookahead->patch_match->GetProblem().images->at(i).GetBitmap().RowMajorData());
  }
  EXPECT_NO_THROW(synchronous->patch_match->Check());
  EXPECT_NO_THROW(lookahead->patch_match->Check());
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
}

TEST_F(OwnedHostSnapshotTest, PreparedGeometrySurvivesInputRetirement) {
  SyntheticDatasetOptions synthetic;
  synthetic.num_rigs = 1;
  synthetic.num_cameras_per_rig = 1;
  synthetic.num_frames_per_rig = 4;
  synthetic.camera_width = kWidth;
  synthetic.camera_height = kHeight;
  synthetic.camera_model_id = SimplePinholeCameraModel::model_id;
  synthetic.camera_params = {12, 5, 4};
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic, &reconstruction);
  reconstruction.Write(workspace_path_ / "sparse");
  Model model;
  model.ReadFromCOLMAP(workspace_path_, "sparse", "images");
  const auto ref_name = reconstruction.Image(1).Name();
  const auto first_name = reconstruction.Image(4).Name();
  const auto second_name = reconstruction.Image(2).Name();
  const int reference = model.GetImageIdx(ref_name);
  const int unused = model.GetImageIdx(reconstruction.Image(3).Name());
  const std::vector<int> configured{model.GetImageIdx(first_name),
                                    model.GetImageIdx(second_name)};
  FlatHashSet<int> used(configured.begin(), configured.end());
  used.insert(reference);
  std::vector<int> effective;
  for (const int idx : used) {
    if (idx != reference) effective.push_back(idx);
    Bitmap bitmap(kWidth, kHeight, false);
    bitmap.Fill(BitmapColor<uint8_t>(20 + idx));
    ASSERT_TRUE(bitmap.Write(model.images.at(idx).GetPath()));
    DepthMap depth(kWidth, kHeight, 1.25, 8.5);
    NormalMap normal(kWidth, kHeight);
    depth.Fill(2 + idx);
    normal.Fill(0.25 + idx);
    const auto name = model.GetImageName(idx) + ".photometric.bin";
    depth.Write(workspace_path_ / "stereo" / "depth_maps" / name);
    normal.Write(workspace_path_ / "stereo" / "normal_maps" / name);
  }
  {
    std::ofstream config(config_path_);
    config << ref_name << '\n' << first_name << ',' << second_name << '\n';
    ASSERT_TRUE(config.good());
  }
  auto options = Options();
  options.geom_consistency = true;
  int checks = 0;
  PatchMatchController::OwnedPairHooks hooks;
  hooks.check_inputs = [&](size_t, const auto& o, const auto& p) {
    ++checks;
    EXPECT_TRUE(o.geom_consistency);
    EXPECT_EQ(p.src_image_idxs, effective);
  };
  std::array<std::unique_ptr<PatchMatchController::OwnedPreparedJob>, 2> prepared;
  for (size_t job = 0; job < 2; ++job) {
    auto controller = Controller(options);
    prepared[job] = controller->PrepareOwnedPairJob(job, hooks);
    // Shared preparation releases Workspace/local inputs before returning;
    // controller retirement must not invalidate the maps or bitmaps either.
  }
  EXPECT_EQ(checks, 2);
  std::filesystem::remove_all(workspace_path_);
  for (const auto& job : prepared) {
    const auto& engine = *job->patch_match;
    const auto& problem = engine.GetProblem();
    EXPECT_TRUE(engine.GetOptions().geom_consistency);
    EXPECT_EQ(problem.ref_image_idx, reference);
    EXPECT_EQ(problem.src_image_idxs, effective);  // Native hash order, not sorted.
    ASSERT_EQ(problem.depth_maps->size(), 4);
    ASSERT_EQ(problem.normal_maps->size(), 4);
    EXPECT_EQ(problem.depth_maps->at(unused).GetWidth(), 0);
    EXPECT_EQ(problem.normal_maps->at(unused).GetWidth(), 0);
    EXPECT_EQ(problem.images->at(unused).GetBitmap().Width(), 0);
    for (const int idx : used) {
      EXPECT_FLOAT_EQ(problem.depth_maps->at(idx).Get(0, 0), 2 + idx);
      for (size_t channel = 0; channel < 3; ++channel) {
        EXPECT_FLOAT_EQ(problem.normal_maps->at(idx).Get(0, 0, channel),
                        0.25 + idx);
      }
      EXPECT_EQ(problem.images->at(idx).GetBitmap().GetPixel(0, 0)->r, 20 + idx);
    }
    EXPECT_EQ(job->depth_map_path.filename(), ref_name + ".geometric.bin");
    EXPECT_EQ(job->normal_map_path.filename(), ref_name + ".geometric.bin");
    EXPECT_NO_THROW(engine.Check());
  }
}

TEST_F(OwnedHostSnapshotTest, GeometryPreparationRejectsUnsupportedFlags) {
  std::filesystem::remove_all(workspace_path_);
  for (int flag = 0; flag < 3; ++flag) {
    auto options = Options();
    options.geom_consistency = true;
    options.filter = flag != 0;
    options.write_consistency_graph = flag == 1;
    options.allow_missing_files = flag == 2;
    EXPECT_THROW(Controller(options), std::invalid_argument);
  }
}

TEST_F(OwnedHostSnapshotTest, PreparedGeometryRejectsMissingSeedsAndFreshOutputs) {
  auto options = Options();
  options.geom_consistency = true;
  EXPECT_THROW(Controller(options)->PrepareOwnedPairJob(0, {}),
               std::invalid_argument);
  for (const auto* folder : {"depth_maps", "normal_maps"}) {
    const auto path = workspace_path_ / "stereo" / folder /
                      (image_names_[0] + ".geometric.bin");
    {
      std::ofstream output(path);
      output << "partial";
    }
    EXPECT_THROW(Controller(options)->PrepareOwnedPairJob(0, {}),
               std::invalid_argument);
    std::filesystem::remove(path);
  }
}

TEST_F(OwnedHostSnapshotTest, GeometrySnapshotsOwnMapsAndPreserveGlobalIndices) {
  auto prepared = Controller(Options())->PrepareOwnedPairJob(0, {});
  auto options = Options();
  options.geom_consistency = true;
  std::unique_ptr<PatchMatch> engine;
  {
    auto problem = prepared->patch_match->GetProblem();
    std::vector<Image> images = *problem.images;
    std::vector<DepthMap> depths(images.size() + 1);
    std::vector<NormalMap> normals(images.size() + 1);
    images.emplace_back();  // Unused global index stays empty.
    for (size_t i = 0; i != images.size() - 1; ++i) {
      depths[i] = DepthMap(kWidth, kHeight, 1.25, 8.5);
      normals[i] = NormalMap(kWidth, kHeight);
      depths[i].Fill(2 + i);
      normals[i].Fill(0.25 + i);
    }
    problem.images = &images;
    problem.depth_maps = &depths;
    problem.normal_maps = &normals;
    PatchMatch stock(options, problem);
    EXPECT_EQ(stock.GetProblem().depth_maps, &depths);
    EXPECT_EQ(stock.GetProblem().normal_maps, &normals);
    engine = std::make_unique<PatchMatch>(
        options, problem, PatchMatchExecution::kOwnedCuda);
    EXPECT_NE(engine->GetProblem().depth_maps, &depths);
    EXPECT_NE(engine->GetProblem().normal_maps, &normals);
    for (auto& depth : depths) depth.Fill(99);
    for (auto& normal : normals) normal.Fill(99);
    EXPECT_FLOAT_EQ(stock.GetProblem().depth_maps->at(0).Get(0, 0), 99);
  }
  prepared.reset();
  std::filesystem::remove_all(workspace_path_);
  const auto& problem = engine->GetProblem();
  ASSERT_EQ(problem.depth_maps->size(), 3);
  ASSERT_EQ(problem.normal_maps->size(), 3);
  EXPECT_EQ(problem.depth_maps->back().GetWidth(), 0);
  EXPECT_EQ(problem.normal_maps->back().GetWidth(), 0);
  for (int i : {problem.ref_image_idx, problem.src_image_idxs.at(0)}) {
    EXPECT_FLOAT_EQ(problem.depth_maps->at(i).Get(0, 0), 2 + i);
    EXPECT_FLOAT_EQ(problem.normal_maps->at(i).Get(0, 0), 0.25 + i);
  }
  EXPECT_NO_THROW(engine->Check());
}

TEST_F(OwnedHostSnapshotTest, GeometryRejectsMalformedMapInputs) {
  auto prepared = Controller(Options())->PrepareOwnedPairJob(0, {});
  auto problem = prepared->patch_match->GetProblem();
  auto options = Options();
  options.geom_consistency = true;
  std::vector<DepthMap> depths(2, DepthMap(kWidth, kHeight, 1.25, 8.5));
  std::vector<NormalMap> normals(2, NormalMap(kWidth, kHeight));
  problem.depth_maps = &depths;
  problem.normal_maps = &normals;
  const auto rejects = [&] {
    PatchMatch engine(options, problem, PatchMatchExecution::kOwnedCuda);
    EXPECT_THROW(engine.Check(), std::invalid_argument);
  };
  problem.depth_maps = nullptr;
  rejects();
  problem.depth_maps = &depths;
  problem.normal_maps = nullptr;
  rejects();
  problem.normal_maps = &normals;
  depths.pop_back();
  rejects();
  depths.emplace_back(kWidth, kHeight, 1.25, 8.5);
  normals.pop_back();
  rejects();
  normals.emplace_back(kWidth, kHeight);
  depths.at(problem.src_image_idxs.at(0)) = DepthMap(1, 1, 1.25, 8.5);
  rejects();
  depths.at(problem.src_image_idxs.at(0)) = DepthMap(kWidth, kHeight, 1.25, 8.5);
  normals.at(problem.src_image_idxs.at(0)) = NormalMap(1, 1);
  rejects();
}

TEST_F(OwnedHostSnapshotTest, GeometryRequiresExistingSeedsWithoutPhotoPrepass) {
  auto options = Options();
  options.geom_consistency = true;
  // No seeds: fail before any backend can be constructed. A photo prepass
  // would try CUDA instead and/or create photometric output here.
  try {
    Controller(options)->Run();
    ADD_FAILURE() << "Missing photometric seeds accepted";
  } catch (const AggregateException& error) {
    ASSERT_EQ(error.exceptions().size(), 1);
    try {
      std::rethrow_exception(error.exceptions().front());
    } catch (const std::invalid_argument& cause) {
      EXPECT_NE(std::string(cause.what()).find("Missing image or map dependency"),
                std::string::npos);
    }
  }
  for (const auto& name : image_names_) {
    EXPECT_FALSE(ExistsFile(workspace_path_ / "stereo" / "depth_maps" /
                           (name + ".photometric.bin")));
    EXPECT_FALSE(ExistsFile(workspace_path_ / "stereo" / "normal_maps" /
                           (name + ".photometric.bin")));
  }
}

TEST_F(OwnedHostSnapshotTest, GeometryRequiresBothOutputPathsFresh) {
  auto options = Options();
  options.geom_consistency = true;
  for (const auto* folder : {"depth_maps", "normal_maps"}) {
    const auto path = workspace_path_ / "stereo" / folder /
                      (image_names_[0] + ".geometric.bin");
    {
      std::ofstream output(path);
      output << "existing output";
      ASSERT_TRUE(output.good());
    }
    try {
      Controller(options)->Run();
      ADD_FAILURE() << "Partial existing geometric output accepted";
    } catch (const AggregateException& error) {
      ASSERT_EQ(error.exceptions().size(), 1);
      try {
        std::rethrow_exception(error.exceptions().front());
      } catch (const std::invalid_argument& cause) {
        EXPECT_NE(std::string(cause.what()).find("fresh output paths"),
                  std::string::npos);
      }
    }
    std::filesystem::remove(path);
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
