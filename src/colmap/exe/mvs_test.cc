// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/mvs.h"

#include "colmap/mvs/mat.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

class StereoFuserOutput : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = CreateTestDir();
    for (const auto* directory :
         {"sparse", "images", "stereo/depth_maps", "stereo/normal_maps"}) {
      std::filesystem::create_directories(root_ / directory);
    }
    SyntheticDatasetOptions synthetic;
    synthetic.num_rigs = 1;
    synthetic.num_cameras_per_rig = 1;
    synthetic.num_frames_per_rig = 2;
    synthetic.camera_width = 8;
    synthetic.camera_height = 40;
    Reconstruction reconstruction;
    SynthesizeDataset(synthetic, &reconstruction);
    for (const auto& [id, frame] : reconstruction.Frames()) {
      reconstruction.Frame(id).SetRigFromWorld(Rigid3d());
    }
    for (const auto& [id, camera] : reconstruction.Cameras()) {
      auto& pinhole = reconstruction.Camera(id);
      pinhole.model_id = CameraModelId::kPinhole;
      pinhole.params = {8.0, 8.0, 0.5, 0.5};
    }
    reconstruction.DeleteAllPoints2DAndPoints3D();
    Track track;
    Point2D observation;
    observation.xy = Eigen::Vector2d(2.5, 1.5);
    std::vector<std::string> names;
    for (const auto& [id, image] : reconstruction.Images()) {
      reconstruction.Image(id).SetPoints2D(std::vector<Point2D>{observation});
      track.AddElement(id, 0);
      names.push_back(image.Name());
    }
    reconstruction.AddPoint3D(Eigen::Vector3d(1, 0.5, 4), std::move(track));
    reconstruction.Write(root_ / "sparse");
    std::sort(names.begin(), names.end());
    std::ofstream config(root_ / "stereo/fusion.cfg");
    for (const auto& name : names) {
      config << name << '\n';
      mvs::Mat<float> depth(8, 40, 1);
      mvs::Mat<float> normal(8, 40, 3);
      depth.Fill(0);
      normal.Fill(0);
      Bitmap bitmap(8, 40, true);
      bitmap.Fill(BitmapColor<uint8_t>(0));
      // Three shared components plus a singleton: four fused points, not the
      // single sparse point. Identical poses make this independent of RNG.
      for (int component = 0; component < 4; ++component) {
        if (component == 3 && name != names.front()) continue;
        const int row = 1 + 10 * component;
        const int col = component == 3 ? 3 : 2;
        depth.Set(row, col, 0, 4);
        normal.Set(row, col, 2, 1);
        ASSERT_TRUE(
            bitmap.SetPixel(col, row, BitmapColor<uint8_t>(row, col, 40)));
      }
      depth.Write(root_ / "stereo/depth_maps" / (name + ".geometric.bin"));
      normal.Write(root_ / "stereo/normal_maps" / (name + ".geometric.bin"));
      ASSERT_TRUE(bitmap.Write(root_ / "images" / name));
    }
    options_.use_cache = false;
    options_.num_threads = 1;
    options_.min_num_pixels = 1;
    options_.max_num_pixels = 10;
    options_.max_traversal_depth = 10;
  }

  Reconstruction RunAPI(const std::filesystem::path& output,
                        const std::string& type = "PLY",
                        std::function<bool()> stopped = {}) {
    return RunStereoFuserImpl(output,
                              root_,
                              "COLMAP",
                              "option-all",
                              "geometric",
                              options_,
                              type,
                              std::move(stopped));
  }

  int RunCLI(const std::filesystem::path& output, const std::string& type) {
    std::vector<std::string> args = {"stereo_fusion",
                                     "--workspace_path",
                                     root_.string(),
                                     "--output_path",
                                     output.string(),
                                     "--output_type",
                                     type,
                                     "--StereoFusion.use_cache",
                                     "0",
                                     "--StereoFusion.use_mmap",
                                     options_.use_mmap ? "1" : "0",
                                     "--StereoFusion.num_threads",
                                     "1",
                                     "--StereoFusion.min_num_pixels",
                                     "1",
                                     "--StereoFusion.max_num_pixels",
                                     "10",
                                     "--StereoFusion.max_traversal_depth",
                                     "10"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return RunStereoFuser(static_cast<int>(argv.size()), argv.data());
  }

  std::string Bytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.is_open()) << path;
    return {std::istreambuf_iterator<char>(stream), {}};
  }

  void ExpectSameModel(const Reconstruction& actual,
                       const Reconstruction& expected) {
    EXPECT_EQ(actual.NumCameras(), expected.NumCameras());
    EXPECT_EQ(actual.NumRegImages(), expected.NumRegImages());
    ASSERT_EQ(actual.NumPoints3D(), expected.NumPoints3D());
    for (const auto& [id, point] : expected.Points3D()) {
      ASSERT_TRUE(actual.ExistsPoint3D(id));
      EXPECT_EQ(actual.Point3D(id).xyz, point.xyz);
      EXPECT_EQ(actual.Point3D(id).color, point.color);
    }
  }

  std::filesystem::path root_;
  mvs::StereoFusionOptions options_;
};

TEST_F(StereoFuserOutput, CliPlyMatchesMaterializingAPIBytes) {
  const auto api_path = root_ / "api.ply";
  const auto cli_path = root_ / "cli.ply";
  // Exercise the unchanged public C++ signature used by Python, including its
  // default cancellation argument. The API must still return dense points.
  const auto reconstruction = RunStereoFuserImpl(
      api_path, root_, "COLMAP", "option-all", "geometric", options_, "PLY");
  ASSERT_EQ(reconstruction.NumPoints3D(), 4);
  EXPECT_EQ(reconstruction.NumRegImages(), 2);
  EXPECT_EQ(reconstruction.NumCameras(), 1);
  const auto points = ReadPly(api_path);
  ASSERT_EQ(points.size(), reconstruction.NumPoints3D());
  std::vector<point3D_t> ids;
  for (const auto& [id, point] : reconstruction.Points3D()) ids.push_back(id);
  std::sort(ids.begin(), ids.end());
  for (size_t i = 0; i < points.size(); ++i) {
    const auto& point = reconstruction.Point3D(ids[i]);
    EXPECT_EQ(point.xyz,
              Eigen::Vector3d(points[i].x, points[i].y, points[i].z));
    EXPECT_EQ(point.color,
              Eigen::Vector3ub(points[i].r, points[i].g, points[i].b));
  }
  ASSERT_EQ(RunCLI(cli_path, "pLy"), EXIT_SUCCESS);
  for (const auto* suffix : {"", ".vis"}) {
    const auto expected = Bytes(api_path.string() + suffix);
    ASSERT_FALSE(expected.empty());
    EXPECT_EQ(Bytes(cli_path.string() + suffix), expected);
  }
}

#if defined(__unix__) || defined(__APPLE__)
TEST_F(StereoFuserOutput, MappedCliPlyMatchesMaterializingAPIBytes) {
  options_.use_mmap = true;
  const auto api_path = root_ / "api.ply";
  const auto cli_path = root_ / "cli.ply";
  ASSERT_EQ(RunAPI(api_path).NumPoints3D(), 4);
  ASSERT_EQ(RunCLI(cli_path, "PLY"), EXIT_SUCCESS);
  EXPECT_EQ(Bytes(cli_path), Bytes(api_path));
  EXPECT_EQ(Bytes(cli_path.string() + ".vis"),
            Bytes(api_path.string() + ".vis"));
}
#endif

TEST_F(StereoFuserOutput, BinAndTxtStillMaterializeDenseModels) {
  for (const std::string type : {"BIN", "TXT"}) {
    SCOPED_TRACE(type);
    const auto api_path = root_ / (type + "-api");
    const auto cli_path = root_ / (type + "-cli");
    std::filesystem::create_directory(api_path);
    std::filesystem::create_directory(cli_path);
    const auto expected = RunAPI(api_path, type);
    ASSERT_EQ(expected.NumPoints3D(), 4);
    ASSERT_EQ(RunCLI(cli_path, type), EXIT_SUCCESS);
    Reconstruction actual;
    actual.Read(cli_path);
    ExpectSameModel(actual, expected);
    for (const auto& entry : std::filesystem::directory_iterator(api_path)) {
      EXPECT_EQ(Bytes(cli_path / entry.path().filename()), Bytes(entry.path()));
    }
  }
}

TEST_F(StereoFuserOutput, StoppedEmptyDoesNotWriteOutput) {
  for (const std::string type : {"PLY", "BIN", "TXT"}) {
    const auto output = root_ / ("empty-" + type);
    EXPECT_EQ(RunAPI(output, type, [] { return true; }).NumPoints3D(), 0);
    EXPECT_FALSE(std::filesystem::exists(output));
    EXPECT_FALSE(std::filesystem::exists(output.string() + ".partial"));
    EXPECT_FALSE(std::filesystem::exists(output.string() + ".vis"));
  }
}

TEST_F(StereoFuserOutput, StoppedAfterFirstImageWritesPartialOutput) {
  for (const std::string type : {"PLY", "BIN", "TXT"}) {
    const auto output = root_ / ("stopped-" + type + ".ply");
    int polls = 0;
    // Serial stock fusion checks before initialization completes and before
    // each image. Allow the first image, then cancel at the next checkpoint.
    const auto reconstruction =
        RunAPI(output, type, [&] { return ++polls >= 3; });
    ASSERT_GT(reconstruction.NumPoints3D(), 0);
    EXPECT_FALSE(std::filesystem::exists(output));
    const auto partial =
        type == "PLY" ? root_ / ("stopped-" + type + ".partial.ply")
                      : std::filesystem::path(output.string() + ".partial");
    if (type == "PLY") {
      EXPECT_EQ(ReadPly(partial).size(), reconstruction.NumPoints3D());
      EXPECT_FALSE(Bytes(partial.string() + ".vis").empty());
    } else {
      Reconstruction actual;
      actual.Read(partial);
      ExpectSameModel(actual, reconstruction);
    }
  }
}

TEST_F(StereoFuserOutput, SparseValidationIsRetainedAfterFusion) {
  const auto output = root_ / "invalid.ply";
  bool removed = false;
  // Workspace initialization has already read the model at the first stop
  // checkpoint. Removing it now isolates the post-fusion sparse Read check.
  EXPECT_THROW(RunAPI(output,
                      "PLY",
                      [&] {
                        if (!removed) {
                          std::filesystem::remove_all(root_ / "sparse");
                          removed = true;
                        }
                        return false;
                      }),
               std::exception);
  EXPECT_TRUE(removed);
  EXPECT_FALSE(std::filesystem::exists(output));
}

TEST_F(StereoFuserOutput, CliRejectsMissingSparseModelWithoutWriting) {
  std::filesystem::remove_all(root_ / "sparse");
  const auto output = root_ / "invalid.ply";
  EXPECT_THROW(RunCLI(output, "PLY"), std::exception);
  EXPECT_FALSE(std::filesystem::exists(output));
  EXPECT_FALSE(std::filesystem::exists(output.string() + ".vis"));
}

}  // namespace
}  // namespace colmap
