// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/image.h"

#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/controllers/undistorters.h"
#include "colmap/image/undistortion.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/base_controller.h"
#include "colmap/util/cancellation.h"
#include "colmap/util/misc.h"
#include "colmap/util/string.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cmath>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace colmap {
namespace {

uint64_t ParseUndistortionBytes(const std::string& text) {
  uint64_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || (text.size() > 1 && text.front() == '0') ||
      parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("invalid canonical undistortion byte allowance");
  return value;
}

// This predicts UndistortCamera's model geometry only. In particular it does
// not inspect decoded dimensions, copy-through paths, or existing outputs.
int PreviewUndistortionTargets(const std::filesystem::path& input_path,
                               const std::string& output_type,
                               const std::filesystem::path& image_list_path,
                               const UndistortCameraOptions& options) {
  try {
    if (output_type != "COLMAP") {
      throw std::invalid_argument("dry_run supports only COLMAP output_type");
    }
    if (!image_list_path.empty()) {
      throw std::invalid_argument(
          "dry_run does not support image_list_path subsets");
    }
    for (const double value : {options.blank_pixels,
                               options.min_scale,
                               options.max_scale,
                               options.roi_min_x,
                               options.roi_min_y,
                               options.roi_max_x,
                               options.roi_max_y}) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("Non-finite undistortion option");
      }
    }
    // Avoid the native reader's fatal-throw diagnostic for absent models.
    const bool binary = ExistsFile(input_path / "cameras.bin") &&
                        ExistsFile(input_path / "images.bin") &&
                        ExistsFile(input_path / "points3D.bin");
    const bool text = ExistsFile(input_path / "cameras.txt") &&
                      ExistsFile(input_path / "images.txt") &&
                      ExistsFile(input_path / "points3D.txt");
    if (!binary && !text) {
      throw std::invalid_argument(
          "No readable native reconstruction at input_path");
    }
    Reconstruction reconstruction;
    reconstruction.Read(input_path);
    auto image_ids = reconstruction.RegImageIds();
    if (image_ids.empty()) {
      throw std::invalid_argument("No registered images to preview");
    }
    std::sort(image_ids.begin(), image_ids.end());

    const auto validate_camera = [](const Camera& camera) {
      if (!camera.VerifyParams() || !camera.IsPerspective() ||
          camera.width == 0 || camera.height == 0 ||
          !std::all_of(camera.params.begin(),
                       camera.params.end(),
                       [](double value) { return std::isfinite(value); }) ||
          camera.FocalLengthX() <= 0 || camera.FocalLengthY() <= 0) {
        throw std::invalid_argument("Invalid or unsupported camera geometry");
      }
    };
    std::ostringstream json;
    json.imbue(std::locale::classic());
    json << std::setprecision(std::numeric_limits<double>::max_digits10);
    const auto quoted = [](const std::string& value) {
      // Preserve UTF-8 bytes, escaping only JSON syntax and control bytes.
      static constexpr char kHexDigits[] = "0123456789abcdef";
      std::string result;
      result.push_back('"');
      for (const char c : value) {
        switch (c) {
          case '"':
            result += "\\\"";
            break;
          case '\\':
            result += "\\\\";
            break;
          case '\b':
            result += "\\b";
            break;
          case '\f':
            result += "\\f";
            break;
          case '\n':
            result += "\\n";
            break;
          case '\r':
            result += "\\r";
            break;
          case '\t':
            result += "\\t";
            break;
          default: {
            const auto uc = static_cast<unsigned char>(c);
            if (uc < 0x20) {
              result += "\\u00";
              result.push_back(kHexDigits[(uc >> 4) & 0xF]);
              result.push_back(kHexDigits[uc & 0xF]);
            } else {
              result.push_back(c);
            }
          }
        }
      }
      result.push_back('"');
      return result;
    };
    json
        << "{\"schema_version\":1,\"kind\":\"undistortion_target_preview\","
           "\"source_basis\":\"native_camera_geometry\","
           "\"prediction_scope\":\"UndistortCamera_model_geometry_not_decoded_"
           "or_copy_through_output\","
           "\"serialization_order\":"
           "\"image_id_ascending_not_execution_or_source_order\","
           "\"decoded_images_verified\":false,\"target_output_verified\":false,"
           "\"physical_fit\":null,\"execution_authorized\":false,\"options\":{"
        << "\"blank_pixels\":" << options.blank_pixels
        << ",\"min_scale\":" << options.min_scale
        << ",\"max_scale\":" << options.max_scale
        << ",\"max_image_size\":" << options.max_image_size
        << ",\"roi_min_x\":" << options.roi_min_x
        << ",\"roi_min_y\":" << options.roi_min_y
        << ",\"roi_max_x\":" << options.roi_max_x
        << ",\"roi_max_y\":" << options.roi_max_y << "},\"images\":[";
    bool first = true;
    for (const image_t image_id : image_ids) {
      const auto& image = reconstruction.Image(image_id);
      const auto& camera = reconstruction.Camera(image.CameraId());
      validate_camera(camera);
      // The public native geometry routine is the sole projection authority.
      const Camera target = UndistortCamera(options, camera);
      validate_camera(target);
      if (!first) json << ',';
      first = false;
      json << "{\"image_id\":" << image_id
           << ",\"name\":" << quoted(image.Name())
           << ",\"camera_id\":" << image.CameraId()
           << ",\"model_width\":" << camera.width
           << ",\"model_height\":" << camera.height
           << ",\"target_width\":" << target.width
           << ",\"target_height\":" << target.height
           << ",\"target_camera_model\":" << quoted(target.ModelName())
           << ",\"target_camera_params\":[";
      for (size_t i = 0; i < target.params.size(); ++i) {
        if (i != 0) json << ',';
        json << target.params[i];
      }
      json << "]}";
    }
    json << "]}\n";
    // No partial success document if reading or projecting any row failed.
    std::cout << json.str();
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    LOG(ERROR) << "Undistortion dry_run failed: " << error.what();
    return EXIT_FAILURE;
  }
}

// Read stereo image pair names from a text file. The text file is expected to
// have one image pair per line, e.g.:
//
//      image_name1.jpg image_name2.jpg
//      image_name3.jpg image_name4.jpg
//      image_name5.jpg image_name6.jpg
//      ...
//
std::vector<std::pair<image_t, image_t>> ReadStereoImagePairs(
    const std::filesystem::path& path, const Reconstruction& reconstruction) {
  const std::vector<std::string> stereo_pair_lines = ReadTextFileLines(path);

  std::vector<std::pair<image_t, image_t>> stereo_pairs;
  stereo_pairs.reserve(stereo_pair_lines.size());

  for (const auto& line : stereo_pair_lines) {
    const std::vector<std::string> names = StringSplit(line, " ");
    THROW_CHECK_EQ(names.size(), 2);

    const Image* image1 = reconstruction.FindImageWithName(names[0]);
    const Image* image2 = reconstruction.FindImageWithName(names[1]);

    THROW_CHECK_NOTNULL(image1);
    THROW_CHECK_NOTNULL(image2);

    stereo_pairs.emplace_back(image1->ImageId(), image2->ImageId());
  }

  return stereo_pairs;
}

}  // namespace

int RunImageDeleter(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::filesystem::path image_ids_path;
  std::filesystem::path image_names_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "image_ids_path",
      &image_ids_path,
      "Path to text file containing one image_id to delete per line");
  options.AddDefaultOption(
      "image_names_path",
      &image_names_path,
      "Path to text file containing one image name to delete per line");
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  if (!image_ids_path.empty()) {
    const auto image_ids = ReadTextFileLines(image_ids_path);

    for (const auto& image_id_str : image_ids) {
      if (image_id_str.empty()) {
        continue;
      }

      const image_t image_id = std::stoi(image_id_str);
      if (reconstruction.ExistsImage(image_id)) {
        const Image& image = reconstruction.Image(image_id);
        LOG(INFO) << StringPrintf(
            "Deleting image_id=%d, image_name=%s, frame_id=%d from "
            "reconstruction",
            image.ImageId(),
            image.Name().c_str(),
            image.FrameId());
        reconstruction.DeRegisterFrame(image.FrameId());
      } else {
        LOG(WARNING) << StringPrintf(
            "Skipping image_id=%s, because it does not "
            "exist in the reconstruction",
            image_id_str.c_str());
      }
    }
  }

  if (!image_names_path.empty()) {
    for (const std::string& image_name : ReadTextFileLines(image_names_path)) {
      if (image_name.empty()) {
        continue;
      }

      const Image* image = reconstruction.FindImageWithName(image_name);
      if (image != nullptr) {
        LOG(INFO) << StringPrintf(
            "Deleting image_id=%d, image_name=%s, frame_id=%d from "
            "reconstruction",
            image->ImageId(),
            image->Name().c_str(),
            image->FrameId());
        reconstruction.DeRegisterFrame(image->FrameId());
      } else {
        LOG(WARNING) << StringPrintf(
            "Skipping image_name=%s, because it does not "
            "exist in the reconstruction",
            image_name.c_str());
      }
    }
  }

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunImageFilterer(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10.0;
  double max_extra_param = 100.0;
  int min_num_observations = 10;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("min_focal_length_ratio", &min_focal_length_ratio);
  options.AddDefaultOption("max_focal_length_ratio", &max_focal_length_ratio);
  options.AddDefaultOption("max_extra_param", &max_extra_param);
  options.AddDefaultOption("min_num_observations", &min_num_observations);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  const size_t num_reg_images = reconstruction.NumRegImages();

  std::vector<frame_t> filtered_frame_ids =
      ObservationManager(reconstruction)
          .FindFramesToFilter(
              /*min_focal_length_ratio=*/min_focal_length_ratio,
              /*max_focal_length_ratio=*/max_focal_length_ratio,
              /*max_extra_param=*/max_extra_param,
              /*min_num_observations=*/min_num_observations);

  for (const auto frame_id : filtered_frame_ids) {
    reconstruction.DeRegisterFrame(frame_id);
  }

  const size_t num_filtered_images =
      num_reg_images - reconstruction.NumRegImages();

  LOG(INFO) << StringPrintf("Filtered %d images from a total of %d images",
                            num_filtered_images,
                            num_reg_images);

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunImageRectifier(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::filesystem::path stereo_pairs_list;

  StereoImageRectifier::Options undistorter_options;
  UndistortCameraOptions undistort_camera_options;
  int num_threads = -1;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddRequiredOption("stereo_pairs_list", &stereo_pairs_list);
  options.AddDefaultOption("blank_pixels",
                           &undistort_camera_options.blank_pixels);
  options.AddDefaultOption("min_scale", &undistort_camera_options.min_scale);
  options.AddDefaultOption("max_scale", &undistort_camera_options.max_scale);
  options.AddDefaultOption("max_image_size",
                           &undistort_camera_options.max_image_size);
  options.AddDefaultOption("num_threads", &num_threads);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  Reconstruction reconstruction;
  reconstruction.Read(input_path);

  undistorter_options.stereo_pairs =
      ReadStereoImagePairs(stereo_pairs_list, reconstruction);
  undistorter_options.num_threads = num_threads;

  StereoImageRectifier rectifier(undistorter_options,
                                 undistort_camera_options,
                                 reconstruction,
                                 *options.image_path,
                                 output_path);
  rectifier.Run();

  return EXIT_SUCCESS;
}

int RunImageRegistrator(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddMapperOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  if (!ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (!ExistsDir(output_path)) {
    LOG(ERROR) << "`output_path` is not a directory";
    return EXIT_FAILURE;
  }

  auto reconstruction = std::make_shared<Reconstruction>();
  reconstruction->Read(input_path);

  LOG_HEADING1("Loading database");

  std::shared_ptr<DatabaseCache> database_cache;

  {
    Timer timer;
    timer.Start();
    DatabaseCache::Options database_cache_options;
    database_cache_options.min_num_matches =
        static_cast<size_t>(options.mapper->min_num_matches);
    database_cache_options.ignore_watermarks =
        options.mapper->ignore_watermarks;
    database_cache_options.image_names = {options.mapper->image_names.begin(),
                                          options.mapper->image_names.end()};
    if (!database_cache_options.image_names.empty()) {
      for (const image_t image_id : reconstruction->RegImageIds()) {
        database_cache_options.image_names.insert(
            reconstruction->Image(image_id).Name());
      }
    }
    database_cache_options.load_all_images = true;
    database_cache = DatabaseCache::Create(
        *Database::Open(*options.database_path), database_cache_options);
    timer.PrintMinutes();
  }

  IncrementalMapper mapper(database_cache);
  mapper.BeginReconstruction(reconstruction);

  const auto mapper_options = options.mapper->Mapper();
  const auto& obs_manager = mapper.ObservationManager();

  for (const auto& [image_id, image] : reconstruction->Images()) {
    if (ScopedSignalHandler::IsInterruptRequested()) {
      break;
    }
    if (image.HasPose()) {
      continue;
    }

    LOG_HEADING1("Registering image #" + std::to_string(image_id) + " (" +
                 std::to_string(reconstruction->NumRegImages() + 1) + ")");

    LOG(INFO) << "\n=> Image sees " << obs_manager.NumVisiblePoints3D(image_id)
              << " / " << obs_manager.NumObservations(image_id) << " points";

    mapper.RegisterNextImage(mapper_options, image_id);
  }

  mapper.EndReconstruction(/*discard=*/false);

  reconstruction->Write(output_path);

  return EXIT_SUCCESS;
}

int RunImageUndistorter(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::string output_type = "COLMAP";
  std::filesystem::path image_list_path;
  std::string copy_policy = "copy";
  bool dry_run = false;
  std::string host_budget = "0", host_policy = "0", host_reserve = "0";

  COLMAPUndistorter::Options undistorter_options;
  UndistortCameraOptions undistort_camera_options;
  int jpeg_quality = -1;
  int num_threads = -1;

  OptionManager options;
  options.AddImageOptions();
  options.AddDefaultOption(
      "dry_run",
      &dry_run,
      "Preview model target geometry only, without image I/O");
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "output_type", &output_type, "{COLMAP, PMVS, CMP-MVS}");
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddDefaultOption(
      "copy_policy", &copy_policy, "{COPY, SOFT_LINK, HARD_LINK}");
  options.AddDefaultOption("blank_pixels",
                           &undistort_camera_options.blank_pixels);
  options.AddDefaultOption("min_scale", &undistort_camera_options.min_scale);
  options.AddDefaultOption("max_scale", &undistort_camera_options.max_scale);
  options.AddDefaultOption("max_image_size",
                           &undistort_camera_options.max_image_size);
  options.AddDefaultOption("roi_min_x", &undistort_camera_options.roi_min_x);
  options.AddDefaultOption("roi_min_y", &undistort_camera_options.roi_min_y);
  options.AddDefaultOption("roi_max_x", &undistort_camera_options.roi_max_x);
  options.AddDefaultOption("roi_max_y", &undistort_camera_options.roi_max_y);
  options.AddDefaultOption("num_patch_match_src_images",
                           &undistorter_options.num_patch_match_src_images);
  options.AddDefaultOption("jpeg_quality", &jpeg_quality);
  options.AddDefaultOption("num_threads", &num_threads);
  options.AddDefaultOption("host_budget_bytes", &host_budget);
  options.AddDefaultOption("policy_host_bytes", &host_policy);
  options.AddDefaultOption("reserve_host_bytes", &host_reserve);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  try {
    undistorter_options.host_budget_bytes = ParseUndistortionBytes(host_budget);
    undistorter_options.policy_host_bytes = ParseUndistortionBytes(host_policy);
    undistorter_options.reserve_host_bytes = ParseUndistortionBytes(host_reserve);
    if ((!undistorter_options.host_budget_bytes &&
         (undistorter_options.policy_host_bytes || undistorter_options.reserve_host_bytes)) ||
        (undistorter_options.host_budget_bytes && (output_type != "COLMAP" || dry_run)))
      throw std::invalid_argument("bounded undistortion requires COLMAP execution and a positive host budget");
  } catch (const std::exception& error) {
    LOG(ERROR) << error.what();
    return EXIT_FAILURE;
  }

  if (dry_run) {
    return PreviewUndistortionTargets(
        input_path, output_type, image_list_path, undistort_camera_options);
  }

  CreateDirIfNotExists(output_path);

  LOG_HEADING1("Reading reconstruction");
  Reconstruction reconstruction;
  reconstruction.Read(input_path);
  LOG(INFO) << StringPrintf("=> Reconstruction with %d images and %d points",
                            reconstruction.NumImages(),
                            reconstruction.NumPoints3D());

  if (!image_list_path.empty()) {
    for (const std::string& image_name : ReadTextFileLines(image_list_path)) {
      const Image* image = reconstruction.FindImageWithName(image_name);
      if (image != nullptr) {
        undistorter_options.image_ids.push_back(image->ImageId());
      } else {
        LOG(WARNING) << "Cannot find image " << image_name;
      }
    }
  }

  StringToUpper(&copy_policy);
  undistorter_options.copy_type = FileCopyTypeFromString(copy_policy);
  undistorter_options.jpeg_quality = jpeg_quality;
  undistorter_options.num_threads = num_threads;

  std::unique_ptr<BaseController> undistorter;
  if (output_type == "COLMAP") {
    undistorter = std::make_unique<COLMAPUndistorter>(undistorter_options,
                                                      undistort_camera_options,
                                                      reconstruction,
                                                      *options.image_path,
                                                      output_path);
  } else if (output_type == "PMVS") {
    PMVSUndistorter::Options pmvs_options;
    pmvs_options.jpeg_quality = jpeg_quality;
    pmvs_options.num_threads = num_threads;
    undistorter = std::make_unique<PMVSUndistorter>(pmvs_options,
                                                    undistort_camera_options,
                                                    reconstruction,
                                                    *options.image_path,
                                                    output_path);
  } else if (output_type == "CMP-MVS") {
    CMPMVSUndistorter::Options cmpmvs_options;
    cmpmvs_options.jpeg_quality = jpeg_quality;
    cmpmvs_options.num_threads = num_threads;
    undistorter = std::make_unique<CMPMVSUndistorter>(cmpmvs_options,
                                                      undistort_camera_options,
                                                      reconstruction,
                                                      *options.image_path,
                                                      output_path);
  } else {
    LOG(ERROR) << "Invalid `output_type` - supported values are "
                  "{'COLMAP', 'PMVS', 'CMP-MVS'}.";
    return EXIT_FAILURE;
  }

  if (undistorter_options.host_budget_bytes) {
    // main owns the process signal handler and translates the received signal
    // into an exit code after this command has drained its workers.
    undistorter->SetCheckIfStoppedFunc(
        [] { return ScopedSignalHandler::IsInterruptRequested(); });
    try {
      undistorter->Run();
    } catch (const std::exception& error) {
      LOG(ERROR) << "Budgeted undistortion failed; workers drained: " << error.what();
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }
  undistorter->Run();

  return EXIT_SUCCESS;
}

int RunImageUndistorterStandalone(int argc, char** argv) {
  std::filesystem::path input_file;
  std::filesystem::path output_path;

  StandaloneImageUndistorter::Options undistorter_options;
  UndistortCameraOptions undistort_camera_options;
  std::string copy_policy = "copy";
  int num_threads = -1;

  OptionManager options;
  options.AddImageOptions();
  options.AddRequiredOption("input_file", &input_file);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption(
      "copy_policy", &copy_policy, "{COPY, SOFT_LINK, HARD_LINK}");
  options.AddDefaultOption("blank_pixels",
                           &undistort_camera_options.blank_pixels);
  options.AddDefaultOption("min_scale", &undistort_camera_options.min_scale);
  options.AddDefaultOption("max_scale", &undistort_camera_options.max_scale);
  options.AddDefaultOption("max_image_size",
                           &undistort_camera_options.max_image_size);
  options.AddDefaultOption("roi_min_x", &undistort_camera_options.roi_min_x);
  options.AddDefaultOption("roi_min_y", &undistort_camera_options.roi_min_y);
  options.AddDefaultOption("roi_max_x", &undistort_camera_options.roi_max_x);
  options.AddDefaultOption("roi_max_y", &undistort_camera_options.roi_max_y);
  options.AddDefaultOption("jpeg_quality", &undistorter_options.jpeg_quality);
  options.AddDefaultOption("num_threads", &num_threads);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  undistorter_options.num_threads = num_threads;

  StringToUpper(&copy_policy);
  undistorter_options.copy_type = FileCopyTypeFromString(copy_policy);

  CreateDirIfNotExists(output_path);

  // Loads a text file containing the image names and camera information.
  // The format of the text file is
  //   image_name CAMERA_MODEL camera_params

  {
    std::ifstream file(input_file);
    THROW_CHECK_FILE_OPEN(file, input_file);

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
      StringTrim(&line);

      if (line.empty()) {
        continue;
      }

      std::string item;
      std::istringstream line_stream(line);
      SetFullPrecTextStream(line_stream);

      // Loads the image name.
      std::string image_name;
      THROW_CHECK(line_stream >> image_name);

      // Loads the camera and its parameters
      struct Camera camera;

      THROW_CHECK(line_stream >> item);
      camera.model_id = CameraModelNameToId(item);
      if (camera.model_id == CameraModelId::kInvalid) {
        LOG(ERROR) << "Camera model " << item << " does not exist";
        return EXIT_FAILURE;
      }

      THROW_CHECK(line_stream >> camera.width >> camera.height);

      camera.params.reserve(CameraModelNumParams(camera.model_id));
      double param;
      while (line_stream >> param) {
        camera.params.push_back(param);
      }

      THROW_CHECK(camera.VerifyParams());

      undistorter_options.image_names_and_cameras.emplace_back(image_name,
                                                               camera);
    }
  }

  auto undistorter =
      std::make_unique<StandaloneImageUndistorter>(undistorter_options,
                                                   undistort_camera_options,
                                                   *options.image_path,
                                                   output_path);
  undistorter->Run();

  return EXIT_SUCCESS;
}

}  // namespace colmap
