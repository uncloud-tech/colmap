// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/fusion.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/math/math.h"
#include "colmap/mvs/fusion_mapped_mat.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"
#include "colmap/util/testing.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iterator>
#include <sstream>
#include <stdexcept>

#if defined(__unix__) || defined(__APPLE__)
#include <cerrno>

#include <sys/wait.h>
#include <unistd.h>
#endif

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

// 28 ten-row tasks, but only 2240 pixels per image.
constexpr size_t kWidth = 8;
constexpr size_t kHeight = 280;

std::vector<std::string> CreateFusionWorkspace(
    const std::filesystem::path& temp_dir,
    const bool default_quality = false,
    const bool heterogeneous = false) {
  CreateDirIfNotExists(temp_dir / "sparse");
  CreateDirIfNotExists(temp_dir / "images");
  CreateDirIfNotExists(temp_dir / "stereo");
  CreateDirIfNotExists(temp_dir / "stereo" / "depth_maps");
  CreateDirIfNotExists(temp_dir / "stereo" / "normal_maps");

  // Retain the two-view diagnostic; default quality needs five observations.
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = heterogeneous ? 2 : 1;
  synthetic_dataset_options.num_frames_per_rig =
      heterogeneous ? 1 : default_quality ? 5 : 2;
  synthetic_dataset_options.camera_width = kWidth;
  synthetic_dataset_options.camera_height = kHeight;
  if (default_quality) {
    synthetic_dataset_options.camera_model_id = CameraModelId::kPinhole;
    synthetic_dataset_options.camera_params = {
        100, 100, kWidth / 2.0, kHeight / 2.0};
  }
  Reconstruction reconstruction;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction);
  if (heterogeneous) {
    EXPECT_GE(reconstruction.NumCameras(), 2);
    reconstruction.Camera(reconstruction.Cameras().begin()->first).Rescale(4, 4);
  }
  if (default_quality) {
    for (const auto& [image_id, image] : reconstruction.Images()) {
      // A fronto-parallel plane at z=4 shifts by exactly one pixel between
      // adjacent views: f * baseline / depth = 100 * 0.04 / 4 = 1.
      // Four columns remain visible in all five registered views.
      reconstruction.Frame(image.FrameId())
          .SetCamFromWorld(
              image.CameraId(),
              Rigid3d(Eigen::Quaterniond::Identity(),
                      Eigen::Vector3d(0.04 * (image_id - 1), 0, 0)));
    }
  }
  reconstruction.Write(temp_dir / "sparse");

  // Create depth maps and normal maps for every image.
  std::vector<std::string> image_names;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    image_names.push_back(image.Name());

    const auto& camera = reconstruction.Camera(image.CameraId());
    const size_t width = camera.width;
    const size_t height = camera.height;
    Mat<float> depth_map(width, height, 1);
    Mat<float> normal_map(width, height, 3);
    Bitmap bitmap(width, height, true);
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        depth_map.Set(row,
                      col,
                      0,
                      default_quality ? 4.0f
                      : row == 0 && col < 2
                          ? -static_cast<float>(col)
                          : 4.0f + 0.001f * row + 0.01f * col);
        const Eigen::Vector3f normal =
            default_quality
                ? Eigen::Vector3f(0, 0, 1)
                : Eigen::Vector3f(0.001f * row, 0.01f * col, 1.0f).normalized();
        for (size_t channel = 0; channel < 3; ++channel) {
          normal_map.Set(row, col, channel, normal[channel]);
        }
        EXPECT_TRUE(bitmap.SetPixel(
            col,
            row,
            BitmapColor<uint8_t>(row % 251, 32 + col, 128 + row % 97)));
      }
    }
    depth_map.Write(temp_dir / "stereo" / "depth_maps" /
                    (image.Name() + ".geometric.bin"));
    normal_map.Write(temp_dir / "stereo" / "normal_maps" /
                     (image.Name() + ".geometric.bin"));
    EXPECT_TRUE(bitmap.Write(temp_dir / "images" / image.Name()));
  }

  // Write fusion config
  std::ofstream fusion_cfg(temp_dir / "stereo" / "fusion.cfg");
  for (const auto& name : image_names) {
    fusion_cfg << name << "\n";
  }
  fusion_cfg.close();

  return image_names;
}

#if defined(__unix__) || defined(__APPLE__)
// Reuse the workspace writer, but replace random camera geometry and dense
// maps with a worked example: three two-view components and one singleton.
// An explicit sparse track establishes the two-image overlap.
std::string CreateCardinalityFusionWorkspace(
    const std::filesystem::path& temp_dir) {
  const auto names = CreateFusionWorkspace(temp_dir);
  Reconstruction reconstruction;
  reconstruction.Read(temp_dir / "sparse");
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    reconstruction.Frame(frame_id).SetRigFromWorld(Rigid3d());
  }
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    auto& pinhole = reconstruction.Camera(camera_id);
    pinhole.model_id = CameraModelId::kPinhole;
    pinhole.params = {8.0, 8.0, 0.5, 0.5};
  }
  // The reused generator's default calibration is for a much larger image;
  // its in-frame filtering does not guarantee shared tracks at 8x280. Replace
  // those observations with a real two-view track under the new calibration.
  // Model accepts zero triangulation angle at fusion's minimum angle of zero.
  reconstruction.DeleteAllPoints2DAndPoints3D();
  Track track;
  Point2D point2D;
  point2D.xy = Eigen::Vector2d(2.5, 1.5);
  for (const auto& [image_id, image] : reconstruction.Images()) {
    reconstruction.Image(image_id).SetPoints2D(std::vector<Point2D>{point2D});
    track.AddElement(image_id, 0);
  }
  reconstruction.AddPoint3D(Eigen::Vector3d(1.0, 0.5, 4.0), std::move(track));
  reconstruction.Write(temp_dir / "sparse");

  for (const auto& name : names) {
    Mat<float> depth(kWidth, kHeight, 1);
    depth.Fill(0.0f);
    depth.Set(0, 0, 0, -1.0f);
    Mat<float> normal(kWidth, kHeight, 3);
    normal.Fill(0.0f);
    Bitmap bitmap(kWidth, kHeight, true);
    bitmap.Fill(BitmapColor<uint8_t>(0));
    for (size_t component = 0; component < 4; ++component) {
      if (component == 3 && name != names.front()) {
        continue;
      }
      const size_t row = 1 + 10 * component;
      const size_t col = component == 3 ? 3 : 2;
      depth.Set(row, col, 0, 4.0f);
      normal.Set(row, col, 2, 1.0f);
      EXPECT_TRUE(
          bitmap.SetPixel(col, row, BitmapColor<uint8_t>(row, col, 40)));
    }
    depth.Write(temp_dir / "stereo" / "depth_maps" / (name + ".geometric.bin"));
    normal.Write(temp_dir / "stereo" / "normal_maps" /
                 (name + ".geometric.bin"));
    EXPECT_TRUE(bitmap.Write(temp_dir / "images" / name));
  }
  return names.front();
}

TEST(StereoFusion, MappedCardinalityVisibilityAndMerge) {
  const auto dir = CreateTestDir();
  const auto singleton_name = CreateCardinalityFusionWorkspace(dir);
  Model model;
  model.Read(dir, "COLMAP");
  ASSERT_EQ(model.images.size(), 2);
  ASSERT_EQ(model.GetMaxOverlappingImages(1, 0),
            (std::vector<std::vector<int>>{{1}, {0}}));
  const int singleton_idx = model.GetImageIdx(singleton_name);

  // P counts the seven positive-depth input pixels, not all allocated raster
  // slots. Run/GetFusedPoints expose no visited-mask or task-capacity
  // telemetry. The known components let us infer consumption in the serial case
  // only.
  constexpr size_t kPositivePixels = 7;
  for (const int num_threads : {1, 4}) {
    // Parallel masks are load/store, not an exclusive claim. Do not promote the
    // serial unique-consumption bound to a general multithreaded guarantee.
    for (const int min_pixels : {1, 2, 3}) {
      if (num_threads != 1 && min_pixels != 1) {
        continue;
      }
      SCOPED_TRACE(::testing::Message() << "threads=" << num_threads
                                        << " min_pixels=" << min_pixels);
      StereoFusionOptions options;
      options.use_mmap = true;
      options.num_threads = num_threads;
      options.min_num_pixels = min_pixels;
      options.max_num_pixels = 10;
      options.max_traversal_depth = 10;
      StereoFusion fusion(options, dir, "COLMAP", "", "geometric");
      fusion.Run();
      const auto& points = fusion.GetFusedPoints();
      const auto& visibility = fusion.GetFusedPointsVisibility();
      ASSERT_EQ(points.size(), visibility.size());
      std::array<size_t, 4> component_counts{};
      size_t total_visibility = 0;
      for (size_t i = 0; i < points.size(); ++i) {
        const auto& point = points[i];
        ASSERT_TRUE(point.r == 1 || point.r == 11 || point.r == 21 ||
                    point.r == 31);
        const size_t component = (point.r - 1) / 10;
        ++component_counts[component];
        const int col = component == 3 ? 3 : 2;
        // K's half-pixel correction gives x=col*depth/8, y=row*depth/8.
        // Distinct colors/positions catch point/visibility merge misalignment.
        EXPECT_FLOAT_EQ(point.x, 0.5f * col);
        EXPECT_FLOAT_EQ(point.y, 0.5f * point.r);
        EXPECT_FLOAT_EQ(point.z, 4.0f);
        EXPECT_FLOAT_EQ(point.nx, 0.0f);
        EXPECT_FLOAT_EQ(point.ny, 0.0f);
        EXPECT_FLOAT_EQ(point.nz, 1.0f);
        EXPECT_EQ(point.g, col);
        EXPECT_EQ(point.b, 40);
        auto indices = visibility[i];
        ASSERT_FALSE(indices.empty());
        std::sort(indices.begin(), indices.end());
        EXPECT_EQ(std::adjacent_find(indices.begin(), indices.end()),
                  indices.end());
        for (const int idx : indices) {
          EXPECT_GE(idx, 0);
          EXPECT_LT(idx, static_cast<int>(model.images.size()));
        }
        if (component == 3) {
          EXPECT_EQ(indices, std::vector<int>({singleton_idx}));
        } else if (num_threads == 1) {
          EXPECT_EQ(indices, std::vector<int>({0, 1}));
        }
        total_visibility += indices.size();
      }
      if (num_threads == 1) {
        // A rejected component still consumes its pixels; revisiting a cycle
        // must not turn either pair into a spurious three-pixel component.
        const size_t pairs = min_pixels <= 2 ? 1 : 0;
        const size_t singleton = min_pixels == 1 ? 1 : 0;
        EXPECT_EQ(component_counts,
                  (std::array<size_t, 4>{pairs, pairs, pairs, singleton}));
        EXPECT_EQ(total_visibility, 6 * pairs + singleton);
        EXPECT_LE(total_visibility, kPositivePixels);  // V <= P
        EXPECT_LE(points.size(), kPositivePixels / min_pixels);
      } else {
        // Several ten-row tasks feed the final merge. Only output pairing,
        // valid image indices, and coverage are claimed, not ordering or a
        // bound on parallel consumption/output capacity.
        for (const size_t count : component_counts) {
          EXPECT_GT(count, 0);
        }
      }
    }
  }
}

#endif

// Fit one complete image entry (RGB + depth + normal), but not two. The
// initialization touches both depths and fusion then loads normals/colors,
// forcing eviction without ever making a single entry exceed the cache cap.
size_t FusionCacheBytes() {
  return kWidth * kHeight * 4 * sizeof(float) +
         Bitmap(kWidth, kHeight, true).NumBytes();
}

double FusionCacheGiB() {
  return static_cast<double>(FusionCacheBytes()) / (1024.0 * 1024.0 * 1024.0);
}

#if defined(__linux__)
TEST(StereoFusion, BudgetedCliFailureDoesNotPublishOutputOrReceipt) {
  const char* binary = std::getenv("COLMAP_BINARY");
  if (binary == nullptr || binary[0] == '\0') GTEST_SKIP();
  const auto dir = CreateTestDir();
  CreateFusionWorkspace(dir);
  const std::array<const char*, 5> caps{"1", "20765", "1073741824", "-1",
                                       "1073741824"};
  for (size_t i = 0; i < caps.size(); ++i) {
    const auto output = dir / (std::to_string(i) + ".ply");
    const auto receipt = dir / (std::to_string(i) + ".receipt");
    const pid_t pid = fork();
    ASSERT_NE(pid, -1);
    if (pid == 0) {
      execl(binary, binary, "stereo_fusion", "--workspace_path", dir.c_str(),
            "--output_path", output.c_str(), "--optimized_only", "true",
            "--optimized_dispatch_receipt", receipt.c_str(),
            "--StereoFusion.num_threads", i == 2 ? "2" : "1",
            "--StereoFusion.min_num_pixels", "1",
            "--StereoFusion.host_budget_bytes", caps[i],
            static_cast<char*>(nullptr));
      _exit(127);
    }
    int status = 0;
    pid_t waited;
    do {
      waited = waitpid(pid, &status, 0);
    } while (waited == -1 && errno == EINTR);
    ASSERT_EQ(waited, pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), i < 2 ? 3 : i < 4 ? 1 : 0);
    EXPECT_EQ(std::filesystem::exists(output), i == 4);
    EXPECT_EQ(std::filesystem::exists(output.string() + ".vis"), i == 4);
    EXPECT_EQ(std::filesystem::exists(receipt), i == 4);
  }
}

TEST(StereoFusion, BudgetRefusesBeforeBitmapDecode) {
  const auto temp_dir = CreateTestDir();
  const auto names = CreateFusionWorkspace(temp_dir);
  // Existing but undecodable RGB: a tiny budget must refuse before Read.
  std::ofstream(temp_dir / "images" / names.front()) << "not a bitmap";
  StereoFusionOptions options;
  options.num_threads = 1;
  options.host_budget_bytes = 1;
  StereoFusion fusion(options, temp_dir, "COLMAP", "", "geometric");
  EXPECT_THROW(fusion.Run(), FusionResourceError);
  EXPECT_TRUE(fusion.GetFusedPoints().empty());
}

TEST(StereoFusion, BudgetedHeterogeneousFullResolutionInputs) {
  const auto dir = CreateTestDir();
  CreateFusionWorkspace(dir, false, true);
  StereoFusionOptions options;
  options.num_threads = 1;
  options.min_num_pixels = options.max_num_pixels;
  // 2240 + 16 pixels. Regardless of configured order, known peak <=11264,
  // ceil(11264 * 1.03)=11602. A homogeneous-max-size sum would refuse.
  options.host_budget_bytes = 11602;
  StereoFusion fusion(options, dir, "COLMAP", "", "geometric");
  EXPECT_NO_THROW(fusion.Run());
  EXPECT_TRUE(fusion.GetFusedPoints().empty());
}

TEST(StereoFusion, BudgetDeduplicatesAndValidatesMapsBeforeDecode) {
  const auto dir = CreateTestDir();
  const auto names = CreateFusionWorkspace(dir);
  std::ofstream cfg(dir / "stereo" / "fusion.cfg", std::ios::app);
  cfg << names.front() << "\n" << names.back() << "\n";
  cfg.close();
  StereoFusionOptions options;
  options.num_threads = 1;
  options.min_num_pixels = options.max_num_pixels;
  options.host_budget_bytes = 20765;  // Exact 2-image input peak, not 4 entries.
  StereoFusion fusion(options, dir, "COLMAP", "", "geometric");
  EXPECT_NO_THROW(fusion.Run());
  EXPECT_TRUE(fusion.GetFusedPoints().empty());
  // Complete malformed map plus invalid bitmap: map validation must occur
  // before RGB decode even when admission has generous space.
  std::ofstream(dir / "stereo" / "depth_maps" /
                (names.front() + ".geometric.bin")) << "malformed";
  std::ofstream(dir / "images" / names.front()) << "bad bitmap";
  options.host_budget_bytes = uint64_t{1} << 30;
  StereoFusion malformed(options, dir, "COLMAP", "", "geometric");
  try {
    malformed.Run();
    FAIL() << "malformed map accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("mapped"), std::string::npos);
  }
}

TEST(StereoFusion, BudgetedMergeRefusesWithoutPublishingPartialPoints) {
  const auto dir = CreateTestDir();
  CreateFusionWorkspace(dir);
  StereoFusionOptions options;
  options.num_threads = 1;
  options.min_num_pixels = 1;
  // Locate the exact executable boundary using the public Run interface.
  // Above the fixed input peak, runtime-growth failures say workers drained;
  // a later generic resource error comes from the post-join final merge.
  uint64_t low = 20765;
  uint64_t high = 1024 * 1024;
  bool merge_refused = false;
  while (low < high) {
    options.host_budget_bytes = low + (high - low) / 2;
    StereoFusion fusion(options, dir, "COLMAP", "", "geometric");
    try {
      fusion.Run();
      high = options.host_budget_bytes;
    } catch (const FusionResourceError& error) {
      if (std::string(error.what()).find("workers drained") == std::string::npos) {
        merge_refused = true;
        EXPECT_TRUE(fusion.GetFusedPoints().empty());
      }
      low = options.host_budget_bytes + 1;
    }
  }
  EXPECT_TRUE(merge_refused);
}

TEST(StereoFusion, BudgetedMappedOutputMatchesControlAndDrainsFailure) {
  const auto temp_dir = CreateTestDir();
  CreateCardinalityFusionWorkspace(temp_dir);
  StereoFusionOptions options;
  options.num_threads = 1;
  options.min_num_pixels = 1;
  StereoFusion control(options, temp_dir, "COLMAP", "", "geometric");
  control.Run();
  options.host_budget_bytes = uint64_t{1} << 30;
  StereoFusion budgeted(options, temp_dir, "COLMAP", "", "geometric");
  budgeted.Run();
  WriteBinaryPlyPoints(temp_dir / "control.ply", control.GetFusedPoints());
  WriteBinaryPlyPoints(temp_dir / "budgeted.ply", budgeted.GetFusedPoints());
  auto read_bytes = [](const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), {});
  };
  EXPECT_EQ(read_bytes(temp_dir / "control.ply"),
            read_bytes(temp_dir / "budgeted.ply"));
  EXPECT_EQ(control.GetFusedPointsVisibility(),
            budgeted.GetFusedPointsVisibility());
  // Input peak is 9 * 2240 = 20160; ceil(20160 * 1.03)=20765.
  // At this exact input boundary there is insufficient output growth space.
  const auto failure_dir = CreateTestDir();
  CreateFusionWorkspace(failure_dir);
  options.host_budget_bytes = 20765;
  StereoFusion exhausted(options, failure_dir, "COLMAP", "", "geometric");
  EXPECT_THROW(exhausted.Run(), FusionResourceError);
  EXPECT_TRUE(exhausted.GetFusedPoints().empty());
  // Return only occurs after queued tasks drain; inputs can be removed now.
  std::filesystem::remove_all(failure_dir / "stereo" / "depth_maps");
}
#endif

TEST(StereoFusion, BudgetRejectsUnsupportedModes) {
  StereoFusionOptions options;
  options.host_budget_bytes = 1000000;
  options.num_threads = 1;
  EXPECT_TRUE(options.Check());
  options.num_threads = 2;
  EXPECT_FALSE(options.Check());
  options.num_threads = 1;
  options.mask_path = "external-masks";
  EXPECT_FALSE(options.Check());
  options.mask_path.clear();
  options.use_mmap = false;
  EXPECT_FALSE(options.Check());
}

TEST(StereoFusion, Integration) {
  const auto temp_dir = CreateTestDir();
  const auto image_names = CreateFusionWorkspace(temp_dir);

  // Run fusion
  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.max_traversal_depth = 10;
  options.check_num_images = 10;
  options.use_cache = false;
  options.use_mmap = false;
  options.num_threads = 1;

  StereoFusion fusion(options, temp_dir, "COLMAP", "", "geometric");
  fusion.Run();

  // Verify that some points were fused
  const auto& fused_points = fusion.GetFusedPoints();
  const auto& visibility = fusion.GetFusedPointsVisibility();

  ASSERT_FALSE(fused_points.empty());
  EXPECT_EQ(fused_points.size(), visibility.size());

  for (const auto& point : fused_points) {
    EXPECT_GT(point.x, -10.0f);
    EXPECT_LT(point.x, 10.0f);
    EXPECT_GT(point.y, -10.0f);
    EXPECT_LT(point.y, 10.0f);
    EXPECT_GT(point.z, -10.0f);
    EXPECT_LT(point.z, 10.0f);
    EXPECT_GE(point.g, 32);
    EXPECT_LT(point.g, 32 + kWidth);
    EXPECT_GE(point.b, 128);
    EXPECT_FLOAT_EQ(
        point.nx * point.nx + point.ny * point.ny + point.nz * point.nz, 1.0f);
  }

  for (const auto& vis : visibility) {
    EXPECT_GT(vis.size(), 0);
  }

  bool stock_stop_checked = false;
  StereoFusion cancelled_stock(options, temp_dir, "COLMAP", "", "geometric");
  cancelled_stock.SetCheckIfStoppedFunc([&stock_stop_checked]() {
    stock_stop_checked = true;
    return true;
  });
  cancelled_stock.Run();
  EXPECT_TRUE(stock_stop_checked);
  EXPECT_TRUE(cancelled_stock.GetFusedPoints().empty());

#if defined(__unix__) || defined(__APPLE__)
  // Full-resolution serial mapped fusion must preserve every output byte.
  options.use_mmap = true;
  StereoFusion mapped(options, temp_dir, "COLMAP", "", "geometric");
  mapped.Run();
  ASSERT_EQ(mapped.GetFusedPoints().size(), fused_points.size());
  EXPECT_EQ(mapped.GetFusedPointsVisibility(), visibility);
  const auto ReadBytes = [](const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.is_open()) << path;
    return std::string(std::istreambuf_iterator<char>(stream),
                       std::istreambuf_iterator<char>());
  };
  WriteBinaryPlyPoints(temp_dir / "stock.ply", fused_points);
  WriteBinaryPlyPoints(temp_dir / "mapped.ply", mapped.GetFusedPoints());
  EXPECT_EQ(ReadBytes(temp_dir / "stock.ply"),
            ReadBytes(temp_dir / "mapped.ply"));
  WritePointsVisibility(temp_dir / "stock.vis", visibility);
  WritePointsVisibility(temp_dir / "mapped.vis",
                        mapped.GetFusedPointsVisibility());
  EXPECT_EQ(ReadBytes(temp_dir / "stock.vis"),
            ReadBytes(temp_dir / "mapped.vis"));

  options.use_mmap = false;
  options.use_cache = true;
  options.cache_size = FusionCacheGiB();
  StereoFusion cached(options, temp_dir, "COLMAP", "", "geometric");
  cached.Run();
  ASSERT_FALSE(cached.GetFusedPoints().empty());
  WriteBinaryPlyPoints(temp_dir / "cached.ply", cached.GetFusedPoints());
  WritePointsVisibility(temp_dir / "cached.vis",
                        cached.GetFusedPointsVisibility());
  EXPECT_EQ(ReadBytes(temp_dir / "cached.ply"),
            ReadBytes(temp_dir / "mapped.ply"));
  EXPECT_EQ(ReadBytes(temp_dir / "cached.vis"),
            ReadBytes(temp_dir / "mapped.vis"));

  options.use_cache = false;
  options.use_mmap = true;
  options.num_threads = 28;
  StereoFusion parallel(options, temp_dir, "COLMAP", "", "geometric");
  parallel.Run();
  ASSERT_FALSE(parallel.GetFusedPoints().empty());
  EXPECT_EQ(parallel.GetFusedPoints().size(),
            parallel.GetFusedPointsVisibility().size());
  for (const auto& point : parallel.GetFusedPoints()) {
    EXPECT_TRUE(std::isfinite(point.x));
    EXPECT_TRUE(std::isfinite(point.y));
    EXPECT_TRUE(std::isfinite(point.z));
    EXPECT_TRUE(std::isfinite(point.nx));
    EXPECT_TRUE(std::isfinite(point.ny));
    EXPECT_TRUE(std::isfinite(point.nz));
  }

  // Cancellation at the next image boundary occurs only after row tasks join.
  int checks = 0;
  StereoFusion joined(options, temp_dir, "COLMAP", "", "geometric");
  joined.SetCheckIfStoppedFunc([&checks]() { return ++checks == 5; });
  joined.Run();
  EXPECT_EQ(checks, 5);
  EXPECT_FALSE(joined.GetFusedPoints().empty());
  joined.SetCheckIfStoppedFunc([]() { return false; });
  joined.Run();  // Reusing owners must not retain masks or task output.
  EXPECT_FALSE(joined.GetFusedPoints().empty());

  options.mask_path = temp_dir / "masks";
  CreateDirIfNotExists(options.mask_path);
  for (const auto& name : image_names) {
    Bitmap mask(kWidth, kHeight, false);
    mask.Fill(BitmapColor<uint8_t>(0));
    mask.Write(options.mask_path / (name + ".png"));
  }
  StereoFusion masked(options, temp_dir, "COLMAP", "", "geometric");
  masked.Run();
  EXPECT_TRUE(masked.GetFusedPoints().empty());
  options.mask_path.clear();

  bool stop_checked = false;
  StereoFusion cancelled_fusion(options, temp_dir, "COLMAP", "", "geometric");
  cancelled_fusion.SetCheckIfStoppedFunc([&stop_checked]() {
    stop_checked = true;
    return true;
  });
  cancelled_fusion.Run();
  EXPECT_TRUE(stop_checked);
  EXPECT_TRUE(cancelled_fusion.GetFusedPoints().empty());

  // Bitmap dimensions are also strict: the mapped path cannot resize RGB.
  Bitmap small_bitmap(1, 1, true);
  small_bitmap.Fill(BitmapColor<uint8_t>(0));
  small_bitmap.Write(temp_dir / "images" / image_names.front());
  StereoFusion wrong_bitmap(options, temp_dir, "COLMAP", "", "geometric");
  EXPECT_THROW(wrong_bitmap.Run(), std::exception);
  Bitmap restored_bitmap(kWidth, kHeight, true);
  restored_bitmap.Fill(BitmapColor<uint8_t>(0, 64, 128));
  restored_bitmap.Write(temp_dir / "images" / image_names.front());

  // Dimensions must be rejected, never silently downsized.
  Mat<float> wrong_size(1, 1, 1);
  wrong_size.Fill(5.0f);
  const auto depth_path = temp_dir / "stereo" / "depth_maps" /
                          (image_names.front() + ".geometric.bin");
  wrong_size.Write(depth_path);
  StereoFusion wrong_resolution(options, temp_dir, "COLMAP", "", "geometric");
  EXPECT_THROW(wrong_resolution.Run(), std::exception);
  std::ofstream(depth_path, std::ios::binary | std::ios::trunc) << "bad";
  StereoFusion malformed(options, temp_dir, "COLMAP", "", "geometric");
  EXPECT_THROW(malformed.Run(), std::exception);
#else
  options.use_mmap = true;
  StereoFusion unsupported(options, temp_dir, "COLMAP", "", "geometric");
  EXPECT_THROW(unsupported.Run(), std::runtime_error);
  EXPECT_THROW((FusionMappedMat(temp_dir / "unused", 1, 1, 1)),
               std::runtime_error);
#endif
}

#if defined(__unix__) || defined(__APPLE__)
TEST(StereoFusion, MappedWorkerFailureDrainsReaders) {
  const auto path = CreateTestDir() / "depth.bin";
  Mat<float> depth(8, 280, 1);
  depth.Fill(4.25f);
  depth.Write(path);
  {
    FusionMappedMat mapped(path, 8, 280, 1);
    std::promise<void> reader_started;
    const auto started = reader_started.get_future().share();
    // Declare the pool last so even exceptional unwinding joins before unmap.
    ThreadPool pool(2);
    const auto failed = pool.AddTask([started]() {
      started.wait();
      throw std::runtime_error("controlled worker failure");
    });
    const auto reader = pool.AddTask([&mapped, &reader_started, failed]() {
      reader_started.set_value();
      // This reader remains outstanding until the other worker has thrown.
      failed.wait();
      float sum = 0;
      for (size_t row = 0; row < mapped.GetHeight(); ++row) {
        for (size_t col = 0; col < mapped.GetWidth(); ++col) {
          sum += mapped.Get(row, col);
        }
      }
      return sum;
    });
    EXPECT_THROW(pool.Wait(), AggregateException);
    EXPECT_EQ(reader.wait_for(std::chrono::seconds(0)),
              std::future_status::ready);
    EXPECT_FLOAT_EQ(reader.get(), 8 * 280 * 4.25f);
    mapped.VerifyUnchanged();
    EXPECT_NO_THROW(pool.Wait());
  }
  // Removal is only after both pool and mapped owner have been destroyed.
  EXPECT_TRUE(std::filesystem::remove(path));
}

// Invoke the real binary: a missing dependency must not produce a receipt
// (or an output), while the legacy mapped command remains permissive.
TEST(StereoFusion, CliStrictRequiresEveryConfiguredDependency) {
  const char* binary = std::getenv("COLMAP_BINARY");
  if (binary == nullptr || binary[0] == '\0') {
    GTEST_SKIP() << "Set COLMAP_BINARY to the colmap executable";
  }
  ASSERT_TRUE(std::filesystem::is_regular_file(binary));
  const auto root = CreateTestDir();
  for (const std::string missing : {"bitmap", "depth", "normal", "none"}) {
    const auto dir = root / missing;
    ASSERT_TRUE(std::filesystem::create_directory(dir));
    const auto names = CreateFusionWorkspace(dir);
    ASSERT_EQ(names.size(), 2);
    // Remove the LAST configured input, so a per-image late check alone is
    // not enough to establish completeness before any processing.
    if (missing == "bitmap") {
      ASSERT_TRUE(std::filesystem::remove(dir / "images" / names.back()));
    } else if (missing != "none") {
      ASSERT_TRUE(std::filesystem::remove(
          dir / "stereo" / (missing == "depth" ? "depth_maps" : "normal_maps") /
          (names.back() + ".geometric.bin")));
    }
    for (const bool strict : {true, false}) {
      const auto output = dir / (strict ? "strict.ply" : "legacy.ply");
      const auto receipt = dir / "receipt.json";
      const std::string workspace_arg = dir.string();
      const std::string output_arg = output.string();
      const std::string receipt_arg = receipt.string();
      const pid_t pid = fork();
      ASSERT_NE(pid, -1);
      if (pid == 0) {
        std::vector<std::string> args = {binary,
                                         "stereo_fusion",
                                         "--workspace_path",
                                         workspace_arg,
                                         "--output_path",
                                         output_arg,
                                         "--StereoFusion.use_mmap",
                                         "true",
                                         "--StereoFusion.use_cache",
                                         "false",
                                         "--StereoFusion.num_threads",
                                         "1",
                                         "--StereoFusion.min_num_pixels",
                                         "1"};
        if (strict)
          args.insert(args.end(),
                      {"--optimized_only",
                       "true",
                       "--optimized_dispatch_receipt",
                       receipt_arg});
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        execv(binary, argv.data());
        _exit(127);
      }
      int status = 0;
      pid_t waited;
      do {
        waited = waitpid(pid, &status, 0);
      } while (waited == -1 && errno == EINTR);
      ASSERT_EQ(waited, pid);
      const bool success = WIFEXITED(status) && WEXITSTATUS(status) == 0;
      const bool expected_success = !strict || missing == "none";
      EXPECT_EQ(success, expected_success) << missing << " strict=" << strict;
      EXPECT_EQ(std::filesystem::exists(output), expected_success);
      EXPECT_EQ(std::filesystem::exists(receipt), strict && missing == "none");
      if (strict && missing == "none") {
        ASSERT_TRUE(std::filesystem::remove(receipt));
      }
    }
    std::filesystem::remove_all(dir);
  }
}

void RunCliSerialExactness(const bool default_quality) {
  // CMake supplies the actual colmap CLI. No PATH lookup or shell is involved.
  const char* binary = std::getenv("COLMAP_BINARY");
  if (binary == nullptr || binary[0] == '\0') {
    GTEST_SKIP() << "Set COLMAP_BINARY to the colmap executable";
  }
  ASSERT_TRUE(std::filesystem::is_regular_file(binary));
  if (default_quality) {
    const StereoFusionOptions defaults;
    ASSERT_EQ(defaults.min_num_pixels, 5);
    ASSERT_EQ(defaults.max_num_pixels, 10000);
    ASSERT_EQ(defaults.max_traversal_depth, 100);
    ASSERT_EQ(defaults.max_reproj_error, 2);
    ASSERT_EQ(defaults.max_depth_error, 0.01f);
    ASSERT_EQ(defaults.max_normal_error, 10);
    ASSERT_EQ(defaults.check_num_images, 50);
    ASSERT_EQ(defaults.max_image_size, -1);
  }
  const auto dir = CreateTestDir();
  const auto image_names = CreateFusionWorkspace(dir, default_quality);
  ASSERT_EQ(image_names.size(), default_quality ? 5 : 2);
  // The cap fits one largest entry but cannot retain two, so both fixtures
  // force eviction while loading multiple images. Do not change stock defaults.
  for (const std::string mode : {"stock", "cached", "mapped", "default"}) {
    const auto output = dir / (mode + ".ply");
    const std::string workspace_arg = dir.string();
    const std::string output_arg = output.string();
    std::ostringstream cache_size_stream;
    cache_size_stream.precision(17);
    cache_size_stream << FusionCacheGiB();
    const std::string cache_size_arg = cache_size_stream.str();
    const pid_t pid = fork();
    ASSERT_NE(pid, -1);
    if (pid == 0) {
      if (mode == "default") {
        // Exercise omitted backend flags with the same qualified serial
        // schedule and numerical options as the explicit mapped and legacy
        // controls.
        execl(binary,
              binary,
              "stereo_fusion",
              "--workspace_path",
              workspace_arg.c_str(),
              "--output_path",
              output_arg.c_str(),
              "--StereoFusion.num_threads",
              "1",
              default_quality ? nullptr : "--StereoFusion.min_num_pixels",
              default_quality ? nullptr : "1",
              static_cast<char*>(nullptr));
        _exit(127);
      }
      execl(binary,
            binary,
            "stereo_fusion",
            "--workspace_path",
            workspace_arg.c_str(),
            "--output_path",
            output_arg.c_str(),
            "--StereoFusion.use_mmap",
            mode == "mapped" ? "1" : "0",
            "--StereoFusion.use_cache",
            mode == "cached" ? "1" : "0",
            "--StereoFusion.cache_size",
            cache_size_arg.c_str(),
            "--StereoFusion.num_threads",
            "1",
            // Omit the override entirely for full default-quality coverage.
            default_quality ? nullptr : "--StereoFusion.min_num_pixels",
            default_quality ? nullptr : "1",
            static_cast<char*>(nullptr));
      _exit(127);
    }
    int status = 0;
    pid_t waited;
    do {
      waited = waitpid(pid, &status, 0);
    } while (waited == -1 && errno == EINTR);
    ASSERT_EQ(waited, pid);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);
    const auto points = ReadPly(output);
    ASSERT_FALSE(points.empty());
    const auto visibility =
        ReadPointsVisibility(output.string() + ".vis", points.size());
    ASSERT_EQ(visibility.size(), points.size());
    for (const auto& vis : visibility) {
      ASSERT_GE(vis.size(), default_quality ? 5 : 1);
    }
  }
  for (const std::string suffix : {".ply", ".ply.vis"}) {
    std::ifstream mapped(dir / ("mapped" + suffix), std::ios::binary);
    ASSERT_TRUE(mapped.is_open());
    const std::string mapped_bytes((std::istreambuf_iterator<char>(mapped)),
                                   {});
    ASSERT_FALSE(mapped_bytes.empty());
    for (const std::string mode : {"stock", "cached", "default"}) {
      std::ifstream stock(dir / (mode + suffix), std::ios::binary);
      ASSERT_TRUE(stock.is_open());
      const std::string stock_bytes((std::istreambuf_iterator<char>(stock)),
                                    {});
      EXPECT_EQ(stock_bytes, mapped_bytes) << mode << suffix;
    }
  }
}

TEST(StereoFusion, CliSerialExactness) { RunCliSerialExactness(false); }

TEST(StereoFusion, CliSerialDefaultQualityExactness) {
  RunCliSerialExactness(true);
}
#endif

#if defined(__unix__) || defined(__APPLE__)
TEST(StereoFusion, DefaultBackendSerialDefaultQualityExactness) {
  const auto dir = CreateTestDir();
  CreateFusionWorkspace(dir, true);
  StereoFusionOptions options;
  // Keep the qualified serial schedule explicit: the default thread count is
  // intentionally unchanged and parallel fusion is not claimed byte-exact.
  options.num_threads = 1;
  const auto Run = [&](const std::string& name) {
    StereoFusion fusion(options, dir, "COLMAP", "", "geometric");
    fusion.Run();
    EXPECT_FALSE(fusion.GetFusedPoints().empty());
    WriteBinaryPlyPoints(dir / (name + ".ply"), fusion.GetFusedPoints());
    WritePointsVisibility(dir / (name + ".vis"),
                          fusion.GetFusedPointsVisibility());
  };
  Run("default");
  options.use_mmap = true;
  Run("mapped");
  options.use_mmap = false;
  Run("legacy");
  for (const std::string suffix : {".ply", ".vis"}) {
    const auto ReadBytes = [&](const std::string& name) {
      std::ifstream stream(dir / (name + suffix), std::ios::binary);
      EXPECT_TRUE(stream.is_open());
      return std::string(std::istreambuf_iterator<char>(stream), {});
    };
    const auto default_bytes = ReadBytes("default");
    EXPECT_FALSE(default_bytes.empty());
    EXPECT_EQ(default_bytes, ReadBytes("mapped"));
    EXPECT_EQ(default_bytes, ReadBytes("legacy"));
  }
}
#endif

TEST(StereoFusion, ResizingQualityPresetsRunWithLegacyBytes) {
  const auto dir = CreateTestDir();
  CreateFusionWorkspace(dir, true);
  struct Preset {
    void (OptionManager::*modify)();
    int max_image_size;
    int check_num_images;
  };
  for (const auto& preset :
       {Preset{&OptionManager::ModifyForLowQuality, 1000, 25},
        Preset{&OptionManager::ModifyForMediumQuality, 1600, 33},
        Preset{&OptionManager::ModifyForHighQuality, 2400, 50}}) {
    SCOPED_TRACE(preset.max_image_size);
    OptionManager manager;
    (manager.*preset.modify)();
    auto options = *manager.stereo_fusion;
    ASSERT_FALSE(options.use_mmap);
    ASSERT_EQ(options.num_threads, -1);
    // Serial is required only for byte comparison, not for preset admission.
    options.num_threads = 1;
    const auto Run = [&](const StereoFusionOptions& run_options,
                         const std::string& name) {
      StereoFusion fusion(run_options, dir, "COLMAP", "", "geometric");
      fusion.Run();
      EXPECT_FALSE(fusion.GetFusedPoints().empty());
      WriteBinaryPlyPoints(dir / (name + ".ply"), fusion.GetFusedPoints());
      WritePointsVisibility(dir / (name + ".vis"),
                            fusion.GetFusedPointsVisibility());
    };
    Run(options, "preset");

    // Independent legacy control retains the pre-mapped-default preset values.
    StereoFusionOptions legacy;
    legacy.use_mmap = false;
    legacy.num_threads = 1;
    legacy.max_image_size = preset.max_image_size;
    legacy.check_num_images = preset.check_num_images;
    Run(legacy, "legacy");
    for (const std::string suffix : {".ply", ".vis"}) {
      const auto ReadBytes = [&](const std::string& name) {
        std::ifstream stream(dir / (name + suffix), std::ios::binary);
        EXPECT_TRUE(stream.is_open());
        return std::string(std::istreambuf_iterator<char>(stream), {});
      };
      const auto preset_bytes = ReadBytes("preset");
      EXPECT_FALSE(preset_bytes.empty());
      EXPECT_EQ(preset_bytes, ReadBytes("legacy"));
    }
  }
}

TEST(StereoFusion, StrictPreflightPrecedesDecoding) {
  const auto dir = CreateTestDir();
  const auto names = CreateFusionWorkspace(dir);
  ASSERT_EQ(names.size(), 2);
  // A decoder would fail on the first configured image. Strict preflight
  // instead must diagnose the last configured dependency before decoding.
  std::ofstream(dir / "stereo/depth_maps" / (names.front() + ".geometric.bin"),
                std::ios::trunc)
      << "bad";
  const auto missing =
      dir / "stereo/normal_maps" / (names.back() + ".geometric.bin");
  ASSERT_TRUE(std::filesystem::remove(missing));
  StereoFusionOptions options;
  options.use_mmap = true;
  options.num_threads = 1;
  options.require_complete_inputs = true;
  StereoFusion fusion(options, dir, "COLMAP", "", "geometric");
  try {
    fusion.Run();
    FAIL() << "Strict fusion accepted missing dependency";
  } catch (const std::invalid_argument& exc) {
    EXPECT_NE(std::string(exc.what()).find(missing.string()),
              std::string::npos);
  }
  EXPECT_TRUE(fusion.GetFusedPoints().empty());
  std::filesystem::remove_all(dir);
}

TEST(StereoFusion, MappedOptions) {
  StereoFusionOptions options;
#if defined(__unix__) || defined(__APPLE__)
  EXPECT_TRUE(options.use_mmap);
#else
  EXPECT_FALSE(options.use_mmap);
#endif
  EXPECT_EQ(options.num_threads, -1);
  EXPECT_EQ(options.max_image_size, -1);
  EXPECT_FALSE(options.use_cache);
  EXPECT_TRUE(options.Check());
  options.use_mmap = true;
  EXPECT_TRUE(options.Check());
  options.max_image_size = 100;
  EXPECT_FALSE(options.Check());
  options.max_image_size = -1;
  options.use_cache = true;
  EXPECT_FALSE(options.Check());
  options.use_cache = false;
  StereoFusion fusion(options, "", "PMVS", "", "geometric");
  EXPECT_THROW(fusion.Run(), std::exception);

  // Legacy/cache/resize remain explicit controls, not silent fallback.
  options.use_mmap = false;
  options.max_image_size = 100;
  options.use_cache = true;
  EXPECT_TRUE(options.Check());
}

TEST(ReadPointsVisibility, RoundTrip) {
  const auto test_dir = CreateTestDir();
  const auto vis_path = test_dir / "test.vis";

  std::vector<std::vector<int>> expected = {
      {0, 1, 2},
      {1, 3},
      {},
      {0, 2, 3, 4},
  };
  WritePointsVisibility(vis_path, expected);

  const auto actual = ReadPointsVisibility(vis_path, expected.size());

  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(actual[i].size(), expected[i].size());
    for (size_t j = 0; j < expected[i].size(); ++j) {
      EXPECT_EQ(actual[i][j], expected[i][j]);
    }
  }
}

TEST(ReadPointsVisibility, SizeMismatch) {
  const auto test_dir = CreateTestDir();
  const auto vis_path = test_dir / "test.vis";

  std::vector<std::vector<int>> data = {{0, 1}, {2}};
  WritePointsVisibility(vis_path, data);

  EXPECT_THROW(ReadPointsVisibility(vis_path, 5), std::invalid_argument);
}

TEST(StereoFusion, Rotation180Invariance) {
  const size_t width = 40;
  const size_t height = 30;
  const double focal_length = 100.0;
  const double cx = 18.0;
  const double cy = 12.0;

  const int test_col = 12;
  const int test_row = 8;
  const float test_depth = 5.0f;

  auto CreateWorkspace = [&](const std::filesystem::path& dir,
                             const double cam_cx,
                             const double cam_cy,
                             const Rigid3d& cam_from_world,
                             const int col,
                             const int row) {
    CreateDirIfNotExists(dir / "sparse");
    CreateDirIfNotExists(dir / "images");
    CreateDirIfNotExists(dir / "stereo");
    CreateDirIfNotExists(dir / "stereo" / "depth_maps");
    CreateDirIfNotExists(dir / "stereo" / "normal_maps");

    Reconstruction reconstruction;
    colmap::Camera camera;
    camera.camera_id = 1;
    camera.model_id = CameraModelId::kPinhole;
    camera.width = width;
    camera.height = height;
    camera.params = {focal_length, focal_length, cam_cx, cam_cy};
    reconstruction.AddCameraWithTrivialRig(camera);

    colmap::Image image;
    image.SetImageId(1);
    image.SetName("image.png");
    image.SetCameraId(1);
    reconstruction.AddImageWithTrivialFrame(image, cam_from_world);
    reconstruction.Write(dir / "sparse");

    Mat<float> depth_map(width, height, 1);
    depth_map.Fill(0.0f);
    depth_map.Set(row, col, 0, test_depth);
    depth_map.Write(dir / "stereo" / "depth_maps" / "image.png.geometric.bin");

    Mat<float> normal_map(width, height, 3);
    normal_map.Fill(0.0f);
    normal_map.Set(row, col, 2, 1.0f);
    normal_map.Write(dir / "stereo" / "normal_maps" /
                     "image.png.geometric.bin");

    Bitmap bitmap(width, height, true);
    bitmap.Fill(BitmapColor<uint8_t>(100, 100, 100));
    bitmap.Write(dir / "images" / "image.png");

    std::ofstream fusion_cfg(dir / "stereo" / "fusion.cfg");
    fusion_cfg << "image.png\n";
    fusion_cfg.close();
  };

  const auto temp_dir1 = CreateTestDir();
  CreateWorkspace(temp_dir1, cx, cy, Rigid3d(), test_col, test_row);

  const auto temp_dir2 = CreateTestDir();
  const Eigen::Quaterniond q_z180(
      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
  CreateWorkspace(temp_dir2,
                  width - cx,
                  height - cy,
                  Rigid3d(q_z180, Eigen::Vector3d::Zero()),
                  width - 1 - test_col,
                  height - 1 - test_row);

  StereoFusionOptions options;
  options.min_num_pixels = 1;
  options.max_num_pixels = 100;
  options.use_cache = false;

  StereoFusion fusion1(options, temp_dir1, "COLMAP", "", "geometric");
  fusion1.Run();

  StereoFusion fusion2(options, temp_dir2, "COLMAP", "", "geometric");
  fusion2.Run();

  const auto& points1 = fusion1.GetFusedPoints();
  const auto& points2 = fusion2.GetFusedPoints();

  ASSERT_EQ(points1.size(), 1);
  ASSERT_EQ(points2.size(), 1);

  // Expected 3D point using pixel center at (col + 0.5, row + 0.5).
  const float expected_x = test_depth * (test_col + 0.5f - cx) / focal_length;
  const float expected_y = test_depth * (test_row + 0.5f - cy) / focal_length;
  const float expected_z = test_depth;

  EXPECT_NEAR(points1[0].x, expected_x, 1e-5f);
  EXPECT_NEAR(points1[0].y, expected_y, 1e-5f);
  EXPECT_NEAR(points1[0].z, expected_z, 1e-5f);

  EXPECT_NEAR(points2[0].x, expected_x, 1e-5f);
  EXPECT_NEAR(points2[0].y, expected_y, 1e-5f);
  EXPECT_NEAR(points2[0].z, expected_z, 1e-5f);

  EXPECT_NEAR(points1[0].x, points2[0].x, 1e-5f);
  EXPECT_NEAR(points1[0].y, points2[0].y, 1e-5f);
  EXPECT_NEAR(points1[0].z, points2[0].z, 1e-5f);
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
