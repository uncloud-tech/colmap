// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/undistorters.h"

#include "colmap/controllers/undistortion_resource.h"
#if defined(COLMAP_MVS_ENABLED)
#include "colmap/mvs/owned_fixed_cohort_validation.h"
#endif
#include "colmap/scene/reconstruction_io.h"
#include "colmap/sensor/models.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/oiio_utils.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <thread>

namespace colmap {
namespace {

// Inventory uses the same camera geometry and branch predicates as execution.
// Headers are metadata only, never guessed RGB stride/decoder physical floors.
std::vector<uint64_t> UndistortionPayloads(
    const Reconstruction& reconstruction, const std::vector<image_t>& ids,
    const std::filesystem::path& image_path,
    const UndistortCameraOptions& options, std::vector<int>* channels) {
  if (options.roi_min_x != 0 || options.roi_min_y != 0 ||
      options.roi_max_x != 1 || options.roi_max_y != 1)
    throw std::invalid_argument("budgeted undistortion does not support ROI");
  if (!std::isfinite(options.blank_pixels) ||
      !std::isfinite(options.min_scale) || !std::isfinite(options.max_scale) ||
      !std::isfinite(options.max_cam_point_norm) ||
      !std::isfinite(options.warp_options.direct_warp_min_scale) ||
      options.warp_options.direct_warp_min_scale < 0 ||
      (options.warp_options.interpolation != WarpImageOptions::Interpolation::kBilinear &&
       options.warp_options.interpolation != WarpImageOptions::Interpolation::kNearestNeighbor) ||
      options.max_image_size == 0 || options.blank_pixels < 0 ||
      options.blank_pixels > 1 || options.min_scale <= 0 ||
      options.max_scale < options.min_scale || options.max_cam_point_norm == 0)
    throw std::invalid_argument("invalid budgeted undistortion options");
  EnsureOpenImageIOInitialized();
  std::vector<uint64_t> payloads;
  payloads.reserve(ids.size());
  FlatHashSet<std::string> names;
  for (const auto id : ids) {
    const auto& image = reconstruction.Image(id);
    const auto& camera = *image.CameraPtr();
    const std::filesystem::path name(image.Name());
    if (name.empty() || name.is_absolute() ||
        std::find(name.begin(), name.end(), "..") != name.end() ||
        !names.insert(name.lexically_normal().string()).second)
      throw std::invalid_argument("budgeted undistortion requires distinct relative image paths");
    if (!camera.VerifyParams() ||
        (!camera.IsPerspective() && !camera.IsSpherical()) ||
        !std::all_of(camera.params.begin(), camera.params.end(),
                     [](double v) { return std::isfinite(v); }))
      throw std::invalid_argument("unsupported/malformed budgeted undistortion camera");
    if (camera.IsPerspective() &&
        (!(camera.FocalLengthX() > 0) || !(camera.FocalLengthY() > 0)))
      throw std::invalid_argument("invalid budgeted undistortion focal length");
    // Validate int allocation arithmetic BEFORE geometry calculation/Rescale.
    UndistortionJobPayload(camera.width, camera.height,
                           camera.width, camera.height, true, false);
    if (options.max_scale > static_cast<double>(std::numeric_limits<int>::max()) /
                                std::max(camera.width, camera.height))
      throw std::invalid_argument("budgeted undistortion scale exceeds int range");
    const bool copy = (!camera.IsPerspective() || camera.IsUndistorted()) &&
                      options.max_image_size < 0;
    Camera target = camera;
    if (!copy) {
      if (camera.IsSpherical()) {
        if (options.max_image_size > 0) {
          const double scale = std::min(
              options.max_image_size / static_cast<double>(camera.width),
              options.max_image_size / static_cast<double>(camera.height));
          if (scale < 1) target.Rescale(scale);
        }
      } else {
        target = UndistortCamera(options, camera);
      }
    }
    const bool equal = target.width == camera.width &&
                       target.height == camera.height;
    const bool direct = equal || (!camera.IsSpherical() &&
        std::min(static_cast<double>(target.width) / camera.width,
                 static_cast<double>(target.height) / camera.height) >=
            options.warp_options.direct_warp_min_scale);
    auto payload = UndistortionJobPayload(camera.width, camera.height,
                                          target.width, target.height, direct, copy);
    OIIO::ImageSpec config;
    config["oiio:reorient"] = 0;
    config["oiio:UnassociatedAlpha"] = 1;
    const auto input = OIIO::ImageInput::open(
        (image_path / image.Name()).string(), &config);
    if (!input) {
      OIIO::geterror();
      throw std::invalid_argument("budgeted undistortion invalid image header");
    }
    const auto& spec = input->spec();
    const bool valid = spec.width == static_cast<int>(camera.width) &&
                       spec.height == static_cast<int>(camera.height) &&
                       spec.format == OIIO::TypeDesc::UINT8 &&
                       spec.depth == 1 && !spec.deep &&
                       spec.nchannels >= 1 && spec.nchannels <= 4;
    const bool grey = spec.nchannels <= 2;
    channels->push_back(spec.nchannels);
    input->close();
    if (!valid)
      throw std::invalid_argument("budgeted undistortion header/model mismatch or unsupported format (uint8, 1-4 channels required)");
    // CloneAsRGB holds grey and RGB at once before the warp. Alpha is
    // discarded by Read without allocating a second native pixel vector.
    if (!copy && grey) payload = std::max(payload,
        UndistortionCheckedMultiply(UndistortionCheckedMultiply(
            camera.width, camera.height), 4));
    LOG(INFO) << "Undistortion job " << image.Name() << ": source "
              << camera.width << "x" << camera.height << ", encoded channels "
              << channels->back() << ", target " << target.width << "x"
              << target.height << ", branch " << (copy ? "copy" : direct ? "direct/clone" : "full-source+resize")
              << ", known bitmap payload bytes " << payload;
    payloads.push_back(payload);
  }
  return payloads;
}

void MaybeSetJpegQuality(const std::filesystem::path& path,
                         Bitmap& bitmap,
                         int jpeg_quality) {
  if ((HasFileExtension(path, ".jpg") || HasFileExtension(path, ".jpeg")) &&
      jpeg_quality > 0) {
    bitmap.SetMetaData("Compression", "jpeg:" + std::to_string(jpeg_quality));
  }
}

template <typename Derived>
void WriteMatrix(const Eigen::MatrixBase<Derived>& matrix,
                 std::ofstream* file) {
  using index_t = typename Eigen::MatrixBase<Derived>::Index;
  for (index_t r = 0; r < matrix.rows(); ++r) {
    for (index_t c = 0; c < matrix.cols() - 1; ++c) {
      *file << matrix(r, c) << " ";
    }
    *file << matrix(r, matrix.cols() - 1) << '\n';
  }
}

// Write projection matrix P = K * [R t] to file and prepend given header.
void WriteProjectionMatrix(const std::filesystem::path& path,
                           const Camera& camera,
                           const Image& image,
                           const std::string& header) {
  THROW_CHECK(camera.model_id == PinholeCameraModel::model_id);

  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);

  Eigen::Matrix3d calib_matrix = Eigen::Matrix3d::Identity();
  calib_matrix(0, 0) = camera.FocalLengthX();
  calib_matrix(1, 1) = camera.FocalLengthY();
  calib_matrix(0, 2) = camera.PrincipalPointX();
  calib_matrix(1, 2) = camera.PrincipalPointY();

  const Eigen::Matrix3x4d img_from_world =
      calib_matrix * image.CamFromWorld().ToMatrix();

  if (!header.empty()) {
    file << header << '\n';
  }

  WriteMatrix(img_from_world, &file);
}

void WriteCOLMAPCommands(const bool geometric,
                         const std::filesystem::path& workspace_path,
                         const std::string& workspace_format,
                         const std::string& pmvs_option_name,
                         const std::string& output_prefix,
                         const std::string& indent,
                         std::ofstream* file) {
  if (geometric) {
    *file << indent << "$COLMAP_EXE_PATH/colmap patch_match_stereo \\\n";
    *file << indent << "  --workspace_path " << workspace_path << " \\\n";
    *file << indent << "  --workspace_format " << workspace_format << " \\\n";
    if (workspace_format == "PMVS") {
      *file << indent << "  --pmvs_option_name " << pmvs_option_name << " \\\n";
    }
    *file << indent << "  --PatchMatchStereo.max_image_size 2000 \\\n";
    *file << indent << "  --PatchMatchStereo.geom_consistency true\n";
  } else {
    *file << indent << "$COLMAP_EXE_PATH/colmap patch_match_stereo \\\n";
    *file << indent << "  --workspace_path " << workspace_path << " \\\n";
    *file << indent << "  --workspace_format " << workspace_format << " \\\n";
    if (workspace_format == "PMVS") {
      *file << indent << "  --pmvs_option_name " << pmvs_option_name << " \\\n";
    }
    *file << indent << "  --PatchMatchStereo.max_image_size 2000 \\\n";
    *file << indent << "  --PatchMatchStereo.geom_consistency false\n";
  }

  *file << indent << "$COLMAP_EXE_PATH/colmap stereo_fusion \\\n";
  *file << indent << "  --workspace_path " << workspace_path << " \\\n";
  *file << indent << "  --workspace_format " << workspace_format << " \\\n";
  if (workspace_format == "PMVS") {
    *file << indent << "  --pmvs_option_name " << pmvs_option_name << " \\\n";
  }
  if (geometric) {
    *file << indent << "  --input_type geometric \\\n";
  } else {
    *file << indent << "  --input_type photometric \\\n";
  }
  *file << indent << "  --output_path "
        << workspace_path / (output_prefix + "fused.ply") << " \\\n";

  *file << indent << "$COLMAP_EXE_PATH/colmap poisson_mesher \\\n";
  *file << indent << "  --input_path "
        << workspace_path / (output_prefix + "fused.ply") << " \\\n";
  *file << indent << "  --output_path "
        << workspace_path / (output_prefix + "meshed-poisson.ply") << " \\\n";

  *file << indent << "$COLMAP_EXE_PATH/colmap delaunay_mesher \\\n";
  *file << indent << "  --input_path " << workspace_path / output_prefix
        << " \\\n";
  *file << indent << "  --input_type dense \\\n";
  *file << indent << "  --output_path "
        << workspace_path / (output_prefix + "meshed-delaunay.ply") << " \\\n";
}

}  // namespace

COLMAPUndistorter::COLMAPUndistorter(
    Options options,
    const UndistortCameraOptions& camera_options,
    const Reconstruction& reconstruction,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path)
    : options_(std::move(options)),
      camera_options_(camera_options),
      reconstruction_(reconstruction),
      image_path_(image_path),
      output_path_(output_path) {
  THROW_CHECK_GE(options_.num_patch_match_src_images, 1);
  THROW_CHECK_GE(options_.jpeg_quality, -1);
  THROW_CHECK_LE(options_.jpeg_quality, 100);
  if (!options_.host_budget_bytes) {
    THROW_CHECK_EQ(options_.policy_host_bytes, 0);
    THROW_CHECK_EQ(options_.reserve_host_bytes, 0);
  }
}

void COLMAPUndistorter::Run() {
  LOG_HEADING1("Image undistortion");

  Timer run_timer;
  run_timer.Start();

  const std::vector<image_t> image_ids = options_.image_ids.empty()
                                             ? reconstruction_.RegImageIds()
                                             : options_.image_ids;
  const size_t num_images = image_ids.size();
  std::optional<UndistortionLogicalBudget> budget;
  std::vector<uint64_t> payloads;
  std::vector<int> channels;
  if (options_.host_budget_bytes) {
    if (ExistsFile(output_path_ / "stereo/patch-match.cfg") ||
        ExistsFile(output_path_ / "stereo/fusion.cfg") ||
        ExistsFile(output_path_ / "sparse/cameras.bin") ||
        ExistsFile(output_path_ / "sparse/cameras.txt"))
      throw std::invalid_argument("budgeted undistortion refuses an existing completed workspace");
#if defined(COLMAP_MVS_ENABLED)
    const auto capacity = std::min(options_.host_budget_bytes,
        mvs::owned::ObserveFixedCohortHostAvailable());
    budget.emplace(capacity, options_.policy_host_bytes,
                   options_.reserve_host_bytes);
    payloads = UndistortionPayloads(reconstruction_, image_ids, image_path_,
                                    camera_options_, &channels);
    if (!budget->Fits(0))
      throw std::runtime_error("undistortion logical policy/reserve does not fit");
    // Preflight all singletons before ANY pixel decode or output creation.
    for (size_t i = 0; i < payloads.size(); ++i)
      budget->BatchSize(payloads, i, 1);
    LOG(INFO) << "Undistortion logical known bitmap payload inventory, jobs: "
              << payloads.size() << ", unknown-cost policy: "
              << options_.policy_host_bytes << ", reserve: "
              << options_.reserve_host_bytes << ", effective host capacity: "
              << capacity << "; full physical fit UNKNOWN";
#else
    throw std::invalid_argument("budgeted undistortion requires MVS host observation support");
#endif
  }

  CreateDirIfNotExists(output_path_ / "images");
  CreateDirIfNotExists(output_path_ / "sparse");
  CreateDirIfNotExists(output_path_ / "stereo");
  CreateDirIfNotExists(output_path_ / "stereo" / "depth_maps");
  CreateDirIfNotExists(output_path_ / "stereo" / "normal_maps");
  CreateDirIfNotExists(output_path_ / "stereo" / "consistency_graphs");
  reconstruction_.CreateImageDirs(output_path_ / "images");
  reconstruction_.CreateImageDirs(output_path_ / "stereo" / "depth_maps");
  reconstruction_.CreateImageDirs(output_path_ / "stereo" / "normal_maps");
  reconstruction_.CreateImageDirs(output_path_ / "stereo" /
                                  "consistency_graphs");

  std::vector<std::string> image_names;
  image_names.reserve(num_images);
  if (budget) {
    const size_t ceiling = GetEffectiveNumThreads(options_.num_threads);
    for (size_t begin = 0; begin < num_images;) {
      if (CheckIfStopped())
        throw std::runtime_error("undistortion cancelled; workers drained");
      const size_t count = budget->BatchSize(payloads, begin, ceiling);
      LOG(INFO) << "Undistortion admitted workers: " << count
                << " (thread ceiling " << ceiling << ")";
      ThreadPool pool(static_cast<int>(count));
      std::vector<std::shared_future<bool>> futures;
      std::exception_ptr failure;
      bool cancelled = false;
      try {
        for (size_t i = 0; i < count; ++i)
          futures.push_back(pool.AddTask(&COLMAPUndistorter::Undistort, this,
                                         image_ids[begin + i], channels[begin + i]));
        for (auto& future : futures) {
          if (CheckIfStopped()) cancelled = true;
          if (!future.get())
            throw std::runtime_error("budgeted undistortion read/write failed");
        }
      } catch (...) {
        failure = std::current_exception();
      }
      // Consume task exceptions before Stop: ThreadPool's destructor otherwise
      // treats unhandled task failures as fatal. Wait drains ALL owned workers.
      try {
        pool.Wait();
      } catch (...) {
        if (!failure) failure = std::current_exception();
      }
      pool.Stop();
      if (failure) std::rethrow_exception(failure);
      if (cancelled || CheckIfStopped())
        throw std::runtime_error("undistortion cancelled; workers drained");
      for (size_t i = 0; i < count; ++i)
        image_names.push_back(reconstruction_.Image(image_ids[begin + i]).Name());
      begin += count;
    }
  } else {
    ThreadPool thread_pool(options_.num_threads);
    std::vector<std::shared_future<bool>> futures;
    futures.reserve(num_images);
    for (const image_t image_id : image_ids) {
      futures.push_back(
          thread_pool.AddTask(&COLMAPUndistorter::Undistort, this, image_id, 0));
    }

    // Only use the image names for the successfully undistorted images
    // when writing the MVS config files
    bool stopped = false;
    for (size_t i = 0; i < futures.size(); ++i) {
      if (CheckIfStopped()) {
        thread_pool.Stop();
        stopped = true;
        break;
      }

      LOG(INFO) << StringPrintf(
          "Undistorting image [%d/%d]", i + 1, futures.size());

      if (futures[i].get()) {
        image_names.push_back(reconstruction_.Image(image_ids[i]).Name());
      }
    }
    if (stopped) {
      LOG(WARNING) << "Stopped image undistortion before writing the sparse "
                      "model and stereo configuration files.";
      run_timer.PrintMinutes();
      return;
    }
  }

  LOG(INFO) << "Writing reconstruction...";
  Reconstruction undistorted_reconstruction = reconstruction_;
  UndistortReconstruction(camera_options_, &undistorted_reconstruction);
  undistorted_reconstruction.Write(output_path_ / "sparse");

  LOG(INFO) << "Writing configuration...";
  WritePatchMatchConfig(image_names);
  WriteFusionConfig(image_names);

  LOG(INFO) << "Writing scripts...";
  WriteScript(/*geometric=*/false);
  WriteScript(/*geometric=*/true);

  run_timer.PrintMinutes();
}

bool COLMAPUndistorter::Undistort(
    const image_t image_id, const int expected_file_channels) const {
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = *image.CameraPtr();

  const auto input_image_path = image_path_ / image.Name();
  const auto output_image_path = output_path_ / "images" / image.Name();

  // Non-perspective cameras (e.g. EQUIRECTANGULAR) have no pinhole image plane
  // to undistort to. Without a size limit they are copied through unchanged
  // (they cannot be rescaled to a pinhole image for MVS); with a max_image_size
  // they still go through UndistortImage below, which resizes them to a smaller
  // image of the same model.
  if (!camera.IsPerspective() && camera_options_.max_image_size < 0 &&
      ExistsFile(input_image_path)) {
    LOG(WARNING) << "Cannot undistort image " << image.Name()
                 << " with non-perspective camera model " << camera.ModelName()
                 << "; copying the original image.";
    FileCopy(input_image_path, output_image_path, options_.copy_type);
    return true;
  }

  // Already-undistorted perspective images are copied through only when no
  // rescaling is requested; with a max_image_size they still go through
  // UndistortImage below so the size limit is applied.
  if (camera.IsUndistorted() && camera_options_.max_image_size < 0 &&
      ExistsFile(input_image_path)) {
    LOG(INFO) << "Copying already undistorted image to location: "
              << output_image_path;
    FileCopy(input_image_path, output_image_path, options_.copy_type);
    return true;
  }

  Bitmap distorted_bitmap;
  if (!distorted_bitmap.Read(input_image_path, true, false,
          options_.host_budget_bytes ? static_cast<int>(camera.width) : 0,
          options_.host_budget_bytes ? static_cast<int>(camera.height) : 0,
          expected_file_channels)) {
    LOG(ERROR) << "Cannot read image at path: " << input_image_path;
    return false;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(camera_options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);

  MaybeSetJpegQuality(
      output_image_path, undistorted_bitmap, options_.jpeg_quality);

  return undistorted_bitmap.Write(output_image_path);
}

void COLMAPUndistorter::WritePatchMatchConfig(
    const std::vector<std::string>& image_names) const {
  const auto path = output_path_ / "stereo" / "patch-match.cfg";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);
  for (const auto& image_name : image_names) {
    file << image_name << '\n';

    file << "__auto__, " << options_.num_patch_match_src_images << '\n';
  }
}

void COLMAPUndistorter::WriteFusionConfig(
    const std::vector<std::string>& image_names) const {
  const auto path = output_path_ / "stereo" / "fusion.cfg";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  for (const auto& image_name : image_names) {
    file << image_name << '\n';
  }
}

void COLMAPUndistorter::WriteScript(const bool geometric) const {
  const auto path = output_path_ / (geometric ? "run-colmap-geometric.sh"
                                              : "run-colmap-photometric.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $COLMAP_EXE_PATH to \n"
       << "# the directory containing the COLMAP executables.\n";
  WriteCOLMAPCommands(geometric, ".", "COLMAP", "option-all", "", "", &file);
}

PMVSUndistorter::PMVSUndistorter(const Options& options,
                                 const UndistortCameraOptions& camera_options,
                                 const Reconstruction& reconstruction,
                                 const std::filesystem::path& image_path,
                                 const std::filesystem::path& output_path)
    : options_(options),
      camera_options_(camera_options),
      reconstruction_(reconstruction),
      image_path_(image_path),
      output_path_(output_path) {
  THROW_CHECK_GE(options_.jpeg_quality, -1);
  THROW_CHECK_LE(options_.jpeg_quality, 100);
}

void PMVSUndistorter::Run() {
  LOG_HEADING1("Image undistortion (CMVS/PMVS)");

  Timer run_timer;
  run_timer.Start();

  CreateDirIfNotExists(output_path_ / "pmvs");
  CreateDirIfNotExists(output_path_ / "pmvs" / "txt");
  CreateDirIfNotExists(output_path_ / "pmvs" / "visualize");
  CreateDirIfNotExists(output_path_ / "pmvs" / "models");

  ThreadPool thread_pool(options_.num_threads);
  std::vector<std::shared_future<bool>> futures;
  futures.reserve(reconstruction_.NumRegImages());
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    futures.push_back(
        thread_pool.AddTask(&PMVSUndistorter::Undistort, this, i));
  }

  bool stopped = false;
  for (size_t i = 0; i < futures.size(); ++i) {
    if (CheckIfStopped()) {
      thread_pool.Stop();
      stopped = true;
      break;
    }

    LOG(INFO) << StringPrintf(
        "Undistorting image [%d/%d]", i + 1, futures.size());

    (void)futures[i].get();
  }
  if (stopped) {
    LOG(WARNING) << "Stopped image undistortion before writing the bundle and "
                    "configuration files.";
    run_timer.PrintMinutes();
    return;
  }

  LOG(INFO) << "Writing bundle file...";
  Reconstruction undistorted_reconstruction = reconstruction_;
  UndistortReconstruction(camera_options_, &undistorted_reconstruction);
  const auto bundle_path = output_path_ / "pmvs" / "bundle.rd.out";
  ExportBundler(undistorted_reconstruction,
                bundle_path,
                AddFileExtension(bundle_path, ".list.txt"));

  LOG(INFO) << "Writing visibility file...";
  WriteVisibilityData();

  LOG(INFO) << "Writing option file...";
  WriteOptionFile();

  LOG(INFO) << "Writing scripts...";
  WritePMVSScript();
  WriteCMVSPMVSScript();
  WriteCOLMAPScript(false);
  WriteCOLMAPScript(true);
  WriteCMVSCOLMAPScript(false);
  WriteCMVSCOLMAPScript(true);

  run_timer.PrintMinutes();
}

bool PMVSUndistorter::Undistort(const size_t reg_image_idx) const {
  const auto output_image_path =
      output_path_ / StringPrintf("pmvs/visualize/%08d.jpg", reg_image_idx);
  const auto proj_matrix_path =
      output_path_ / StringPrintf("pmvs/txt/%08d.txt", reg_image_idx);

  const image_t image_id =
      *std::next(reconstruction_.RegImageIds().begin(), reg_image_idx);
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = *image.CameraPtr();

  Bitmap distorted_bitmap;
  const auto input_image_path = image_path_ / image.Name();
  if (!distorted_bitmap.Read(input_image_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image_path;
    return false;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(camera_options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);

  WriteProjectionMatrix(proj_matrix_path, undistorted_camera, image, "CONTOUR");
  MaybeSetJpegQuality(
      output_image_path, undistorted_bitmap, options_.jpeg_quality);
  return undistorted_bitmap.Write(output_image_path);
}

void PMVSUndistorter::WriteVisibilityData() const {
  const auto path = output_path_ / "pmvs" / "vis.dat";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);

  file << "VISDATA\n";
  file << reconstruction_.NumRegImages() << '\n';

  size_t image_idx = 0;
  for (const image_t image_id : reconstruction_.RegImageIds()) {
    const Image& image = reconstruction_.Image(image_id);
    FlatHashSet<image_t> visible_image_ids;
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (point2D.HasPoint3D()) {
        const Point3D& point3D = reconstruction_.Point3D(point2D.point3D_id);
        for (const TrackElement& track_el : point3D.track.Elements()) {
          if (track_el.image_id != image_id) {
            visible_image_ids.insert(track_el.image_id);
          }
        }
      }
    }

    std::vector<image_t> sorted_visible_image_ids(visible_image_ids.begin(),
                                                  visible_image_ids.end());
    std::sort(sorted_visible_image_ids.begin(), sorted_visible_image_ids.end());

    file << image_idx++ << " " << visible_image_ids.size();
    for (const image_t visible_image_id : sorted_visible_image_ids) {
      file << " " << visible_image_id;
    }
    file << '\n';
  }
}

void PMVSUndistorter::WritePMVSScript() const {
  const auto path = output_path_ / "run-pmvs.sh";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $PMVS_EXE_PATH to \n"
       << "# the directory containing the CMVS-PMVS executables.\n";
  file << "$PMVS_EXE_PATH/pmvs2 pmvs/ option-all\n";
}

void PMVSUndistorter::WriteCMVSPMVSScript() const {
  const auto path = output_path_ / "run-cmvs-pmvs.sh";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $PMVS_EXE_PATH to \n"
       << "# the directory containing the CMVS-PMVS executables.\n";
  file << "$PMVS_EXE_PATH/cmvs pmvs/\n";
  file << "$PMVS_EXE_PATH/genOption pmvs/\n";
  file << "find pmvs/ -iname \"option-*\" | sort | while read file_name\n";
  file << "do\n";
  file << "    option_name=$(basename \"$file_name\")\n";
  file << "    if [ \"$option_name\" = \"option-all\" ]; then\n";
  file << "        continue\n";
  file << "    fi\n";
  file << "    $PMVS_EXE_PATH/pmvs2 pmvs/ $option_name\n";
  file << "done\n";
}

void PMVSUndistorter::WriteCOLMAPScript(const bool geometric) const {
  const auto path = output_path_ / (geometric ? "run-colmap-geometric.sh"
                                              : "run-colmap-photometric.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $COLMAP_EXE_PATH to \n"
       << "# the directory containing the COLMAP executables.\n";
  WriteCOLMAPCommands(
      geometric, "pmvs", "PMVS", "option-all", "option-all-", "", &file);
}

void PMVSUndistorter::WriteCMVSCOLMAPScript(const bool geometric) const {
  const auto path =
      output_path_ / (geometric ? "run-cmvs-colmap-geometric.sh"
                                : "run-cmvs-colmap-photometric.sh");
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);

  file << "# You must set $PMVS_EXE_PATH to \n"
       << "# the directory containing the CMVS-PMVS executables\n";
  file << "# and you must set $COLMAP_EXE_PATH to \n"
       << "# the directory containing the COLMAP executables.\n";
  file << "$PMVS_EXE_PATH/cmvs pmvs/\n";
  file << "$PMVS_EXE_PATH/genOption pmvs/\n";
  file << "find pmvs/ -iname \"option-*\" | sort | while read file_name\n";
  file << "do\n";
  file << "    workspace_path=$(dirname \"$file_name\")\n";
  file << "    option_name=$(basename \"$file_name\")\n";
  file << "    if [ \"$option_name\" = \"option-all\" ]; then\n";
  file << "        continue\n";
  file << "    fi\n";
  file << "    rm -rf \"$workspace_path/stereo\"\n";
  WriteCOLMAPCommands(geometric,
                      "pmvs",
                      "PMVS",
                      "$option_name",
                      "$option_name-",
                      "    ",
                      &file);
  file << "done\n";
}

void PMVSUndistorter::WriteOptionFile() const {
  const auto path = output_path_ / "pmvs" / "option-all";
  std::ofstream file(path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(file, path);
  SetFullPrecTextStream(file);

  file << "# Generated by COLMAP - all images, no clustering.\n";

  file << "level 1\n";
  file << "csize 2\n";
  file << "threshold 0.7\n";
  file << "wsize 7\n";
  file << "minImageNum 3\n";
  file << "CPU " << std::thread::hardware_concurrency() << '\n';
  file << "setEdge 0\n";
  file << "useBound 0\n";
  file << "useVisData 1\n";
  file << "sequence -1\n";
  file << "maxAngle 10\n";
  file << "quad 2.0\n";

  file << "timages " << reconstruction_.NumRegImages();
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    file << " " << i;
  }
  file << '\n';

  file << "oimages 0\n";
}

CMPMVSUndistorter::CMPMVSUndistorter(
    const Options& options,
    const UndistortCameraOptions& camera_options,
    const Reconstruction& reconstruction,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path)
    : options_(options),
      camera_options_(camera_options),
      image_path_(image_path),
      output_path_(output_path),
      reconstruction_(reconstruction) {
  THROW_CHECK_GE(options_.jpeg_quality, -1);
  THROW_CHECK_LE(options_.jpeg_quality, 100);
}

void CMPMVSUndistorter::Run() {
  LOG_HEADING1("Image undistortion (CMP-MVS)");

  Timer run_timer;
  run_timer.Start();

  ThreadPool thread_pool(options_.num_threads);
  std::vector<std::shared_future<bool>> futures;
  futures.reserve(reconstruction_.NumRegImages());
  for (size_t i = 0; i < reconstruction_.NumRegImages(); ++i) {
    futures.push_back(
        thread_pool.AddTask(&CMPMVSUndistorter::Undistort, this, i));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    if (CheckIfStopped()) {
      thread_pool.Stop();
      break;
    }

    LOG(INFO) << StringPrintf(
        "Undistorting image [%d/%d]", i + 1, futures.size());

    (void)futures[i].get();
  }

  run_timer.PrintMinutes();
}

bool CMPMVSUndistorter::Undistort(const size_t reg_image_idx) const {
  const auto output_image_path =
      output_path_ / StringPrintf("%05d.jpg", reg_image_idx + 1);
  const auto proj_matrix_path =
      output_path_ / StringPrintf("%05d_P.txt", reg_image_idx + 1);

  const image_t image_id =
      *std::next(reconstruction_.RegImageIds().begin(), reg_image_idx);
  const Image& image = reconstruction_.Image(image_id);
  const Camera& camera = *image.CameraPtr();

  Bitmap distorted_bitmap;
  const auto input_image_path = image_path_ / image.Name();
  if (!distorted_bitmap.Read(input_image_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image_path;
    return false;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(camera_options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);

  WriteProjectionMatrix(proj_matrix_path, undistorted_camera, image, "CONTOUR");
  MaybeSetJpegQuality(
      output_image_path, undistorted_bitmap, options_.jpeg_quality);
  return undistorted_bitmap.Write(output_image_path);
}

StandaloneImageUndistorter::StandaloneImageUndistorter(
    Options options,
    const UndistortCameraOptions& camera_options,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path)
    : options_(std::move(options)),
      camera_options_(camera_options),
      image_path_(image_path),
      output_path_(output_path) {
  THROW_CHECK_GE(options_.jpeg_quality, -1);
  THROW_CHECK_LE(options_.jpeg_quality, 100);
}

void StandaloneImageUndistorter::Run() {
  LOG_HEADING1("Image undistortion");

  Timer run_timer;
  run_timer.Start();

  CreateDirIfNotExists(output_path_);

  ThreadPool thread_pool(options_.num_threads);
  std::vector<std::shared_future<bool>> futures;
  const size_t num_images = options_.image_names_and_cameras.size();
  futures.reserve(num_images);
  for (size_t i = 0; i < num_images; ++i) {
    futures.push_back(
        thread_pool.AddTask(&StandaloneImageUndistorter::Undistort, this, i));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    if (CheckIfStopped()) {
      thread_pool.Stop();
      break;
    }

    LOG(INFO) << StringPrintf(
        "Undistorting image [%d/%d]", i + 1, futures.size());

    (void)futures[i].get();
  }

  run_timer.PrintMinutes();
}

bool StandaloneImageUndistorter::Undistort(const size_t image_idx) const {
  const auto& [image_name, camera] =
      options_.image_names_and_cameras[image_idx];

  const auto output_image_path = output_path_ / image_name;
  const auto input_image_path = image_path_ / image_name;

  // Check if the image is already undistorted and copy from source if no
  // scaling is needed
  if (camera.IsUndistorted() && camera_options_.max_image_size < 0 &&
      ExistsFile(input_image_path)) {
    FileCopy(input_image_path, output_image_path, options_.copy_type);
    return true;
  }

  Bitmap distorted_bitmap;
  if (!distorted_bitmap.Read(input_image_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image_path;
    return false;
  }

  Bitmap undistorted_bitmap;
  Camera undistorted_camera;
  UndistortImage(camera_options_,
                 distorted_bitmap,
                 camera,
                 &undistorted_bitmap,
                 &undistorted_camera);

  MaybeSetJpegQuality(
      output_image_path, undistorted_bitmap, options_.jpeg_quality);

  return undistorted_bitmap.Write(output_image_path);
}

StereoImageRectifier::StereoImageRectifier(
    Options options,
    const UndistortCameraOptions& camera_options,
    const Reconstruction& reconstruction,
    const std::filesystem::path& image_path,
    const std::filesystem::path& output_path)
    : options_(std::move(options)),
      camera_options_(camera_options),
      reconstruction_(reconstruction),
      image_path_(image_path),
      output_path_(output_path) {
  THROW_CHECK_GE(options_.jpeg_quality, -1);
  THROW_CHECK_LE(options_.jpeg_quality, 100);
}

void StereoImageRectifier::Run() {
  LOG_HEADING1("Stereo rectification");

  Timer run_timer;
  run_timer.Start();

  ThreadPool thread_pool(options_.num_threads);
  std::vector<std::shared_future<void>> futures;
  futures.reserve(options_.stereo_pairs.size());
  for (const auto& stereo_pair : options_.stereo_pairs) {
    futures.push_back(thread_pool.AddTask(&StereoImageRectifier::Rectify,
                                          this,
                                          stereo_pair.first,
                                          stereo_pair.second));
  }

  for (size_t i = 0; i < futures.size(); ++i) {
    if (CheckIfStopped()) {
      thread_pool.Stop();
      break;
    }

    LOG(INFO) << StringPrintf(
        "Rectifying image pair [%d/%d]", i + 1, futures.size());

    futures[i].get();
  }

  run_timer.PrintMinutes();
}

void StereoImageRectifier::Rectify(const image_t image_id1,
                                   const image_t image_id2) const {
  const Image& image1 = reconstruction_.Image(image_id1);
  const Image& image2 = reconstruction_.Image(image_id2);
  const Camera& camera1 = reconstruction_.Camera(image1.CameraId());
  const Camera& camera2 = reconstruction_.Camera(image2.CameraId());

  const std::string image_name1 = StringReplace(image1.Name(), "/", "-");
  const std::string image_name2 = StringReplace(image2.Name(), "/", "-");

  const std::string stereo_pair_name =
      StringPrintf("%s-%s", image_name1.c_str(), image_name2.c_str());

  CreateDirIfNotExists(output_path_ / stereo_pair_name);

  const auto output_image_path1 = output_path_ / stereo_pair_name / image_name1;
  const auto output_image_path2 = output_path_ / stereo_pair_name / image_name2;

  Bitmap distorted_bitmap1;
  const auto input_image1_path = image_path_ / image1.Name();
  if (!distorted_bitmap1.Read(input_image1_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image1_path;
    return;
  }

  Bitmap distorted_bitmap2;
  const auto input_image2_path = image_path_ / image2.Name();
  if (!distorted_bitmap2.Read(input_image2_path)) {
    LOG(ERROR) << "Cannot read image at path " << input_image2_path;
    return;
  }

  const Rigid3d cam2_from_cam1 =
      image2.CamFromWorld() * Inverse(image1.CamFromWorld());

  Bitmap undistorted_bitmap1;
  Bitmap undistorted_bitmap2;
  Camera undistorted_camera;
  Eigen::Matrix4d Q;
  RectifyAndUndistortStereoImages(camera_options_,
                                  distorted_bitmap1,
                                  distorted_bitmap2,
                                  camera1,
                                  camera2,
                                  cam2_from_cam1,
                                  &undistorted_bitmap1,
                                  &undistorted_bitmap2,
                                  &undistorted_camera,
                                  &Q);

  MaybeSetJpegQuality(
      output_image_path1, undistorted_bitmap1, options_.jpeg_quality);
  MaybeSetJpegQuality(
      output_image_path2, undistorted_bitmap2, options_.jpeg_quality);

  undistorted_bitmap1.Write(output_image_path1);
  undistorted_bitmap2.Write(output_image_path2);

  const auto Q_path = output_path_ / stereo_pair_name / "Q.txt";
  std::ofstream Q_file(Q_path, std::ios::trunc);
  THROW_CHECK_FILE_OPEN(Q_file, Q_path);
  SetFullPrecTextStream(Q_file);
  WriteMatrix(Q, &Q_file);
}

}  // namespace colmap
