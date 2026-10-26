// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/mvs.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/mvs/advancing_front_meshing.h"
#include "colmap/mvs/delaunay_meshing.h"
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/mesh_simplification.h"
#include "colmap/mvs/model.h"
#include "colmap/mvs/native_run_deadline.h"
#include "colmap/mvs/optimized_dispatch.h"
#include "colmap/mvs/optimized_photo_runner.h"
#include "colmap/mvs/optimized_receipt.h"
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/mvs/poisson_meshing.h"
#include "colmap/mvs/resident_cohort_runner.h"
#include "colmap/mvs/resident_contract.h"
#include "colmap/mvs/resident_forecast.h"
#include "colmap/mvs/resident_runner.h"
#include "colmap/mvs/texture_mapping.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/file.h"
#include "colmap/util/ply.h"

#include <limits>
#include <stdexcept>
#include <utility>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace colmap {

int RunAdvancingFrontMesher(int argc, char** argv) {
#if !defined(COLMAP_CGAL_ENABLED)
  LOG(ERROR) << "Advancing front meshing requires CGAL, which is not "
                "available on your system.";
  return EXIT_FAILURE;
#else   // COLMAP_CGAL_ENABLED
  std::filesystem::path input_path;
  std::filesystem::path output_path;

  OptionManager options;
  options.AddRequiredOption(
      "input_path", &input_path, "Path to the dense workspace folder");
  options.AddRequiredOption("output_path", &output_path);
  options.AddAdvancingFrontMeshingOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  mvs::AdvancingFrontMeshing(
      *options.advancing_front_meshing, input_path, output_path);

  return EXIT_SUCCESS;
#endif  // COLMAP_CGAL_ENABLED
}

int RunDelaunayMesher(int argc, char** argv) {
#if !defined(COLMAP_CGAL_ENABLED)
  LOG(ERROR) << "Delaunay meshing requires CGAL, which is not "
                "available on your system.";
  return EXIT_FAILURE;
#else   // COLMAP_CGAL_ENABLED
  std::filesystem::path input_path;
  std::string input_type = "dense";
  std::filesystem::path output_path;

  OptionManager options;
  options.AddRequiredOption(
      "input_path",
      &input_path,
      "Path to either the dense workspace folder or the sparse reconstruction");
  options.AddDefaultOption("input_type", &input_type, "{dense, sparse}");
  options.AddRequiredOption("output_path", &output_path);
  options.AddDelaunayMeshingOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  StringToLower(&input_type);
  if (input_type == "sparse") {
    mvs::SparseDelaunayMeshing(
        *options.delaunay_meshing, input_path, output_path);
  } else if (input_type == "dense") {
    mvs::DenseDelaunayMeshing(
        *options.delaunay_meshing, input_path, output_path);
  } else {
    LOG(ERROR) << "Invalid input type - "
                  "supported values are 'sparse' and 'dense'.";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
#endif  // COLMAP_CGAL_ENABLED
}

int RunMeshSimplifier(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;

  OptionManager options;
  options.AddRequiredOption(
      "input_path", &input_path, "Path to input PLY mesh");
  options.AddRequiredOption(
      "output_path", &output_path, "Path to output PLY mesh");
  options.AddMeshSimplificationOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  THROW_CHECK_HAS_FILE_EXTENSION(input_path, ".ply");
  THROW_CHECK_FILE_EXISTS(input_path);
  THROW_CHECK_HAS_FILE_EXTENSION(output_path, ".ply");

  LOG(INFO) << "Reading mesh from " << input_path;
  const PlyMesh mesh = ReadPlyMesh(input_path).mesh;
  LOG(INFO) << "Input mesh: " << mesh.vertices.size() << " vertices, "
            << mesh.faces.size() << " faces";

  const PlyMesh simplified =
      mvs::SimplifyMesh(mesh, *options.mesh_simplification);

  LOG(INFO) << "Writing simplified mesh to " << output_path;
  WriteBinaryPlyMesh(output_path, PlyTexturedMesh{simplified});

  return EXIT_SUCCESS;
}

int RunMeshTexturer(int argc, char** argv) {
  std::filesystem::path workspace_path;
  std::filesystem::path input_path;
  std::filesystem::path output_path;
  std::string output_type = "BIN";

  OptionManager options;
  options.AddRequiredOption(
      "workspace_path",
      &workspace_path,
      "Path to the workspace folder containing undistorted images and "
      "sparse reconstruction");
  options.AddRequiredOption(
      "input_path", &input_path, "Path to the input PLY mesh file");
  options.AddRequiredOption(
      "output_path",
      &output_path,
      "Path to the output directory. The textured mesh PLY and texture "
      "atlas image will be written here");
  options.AddDefaultOption("output_type", &output_type, "{BIN, TXT}");
  options.AddMeshTextureMappingOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  StringToLower(&output_type);
  THROW_CHECK(output_type == "bin" || output_type == "txt")
      << "Invalid `output_type` " << output_type
      << " - supported values are 'BIN' and 'TXT'.";

  LOG(INFO) << "Reading model...";
  mvs::Model model;
  model.ReadFromCOLMAP(workspace_path);

  LOG(INFO) << "Loading " << model.images.size() << " images...";
  for (auto& image : model.images) {
    Bitmap bitmap;
    THROW_CHECK(bitmap.Read(image.GetPath(), /*as_rgb=*/true))
        << "Failed to read image: " << image.GetPath();
    if (bitmap.Width() != static_cast<int>(image.GetWidth()) ||
        bitmap.Height() != static_cast<int>(image.GetHeight())) {
      bitmap.Rescale(static_cast<int>(image.GetWidth()),
                     static_cast<int>(image.GetHeight()));
    }
    image.SetBitmap(std::move(bitmap));
  }

  LOG(INFO) << "Reading input mesh from " << input_path << "...";
  const PlyMesh mesh = ReadPlyMesh(input_path).mesh;
  LOG(INFO) << "Mesh has " << mesh.vertices.size() << " vertices and "
            << mesh.faces.size() << " faces";

  options.mesh_texture_mapping->Print();

  LOG(INFO) << "Running surface texture mapping...";
  const mvs::MeshTextureMappingResult result = mvs::MeshTextureMapping(
      mesh, model.images, *options.mesh_texture_mapping);

  CreateDirIfNotExists(output_path);

  const std::filesystem::path texture_filename = "texture.png";
  const std::filesystem::path texture_path = output_path / texture_filename;
  LOG(INFO) << "Writing texture atlas to " << texture_path << "...";
  result.texture_atlas.Write(texture_path);

  PlyTexturedMesh textured_mesh;
  textured_mesh.mesh = mesh;
  textured_mesh.face_uvs = result.face_uvs;
  textured_mesh.texture_file = texture_filename.string();

  const std::filesystem::path mesh_path = output_path / "mesh.ply";
  LOG(INFO) << "Writing textured mesh to " << mesh_path << "...";
  if (output_type == "bin") {
    WriteBinaryPlyMesh(mesh_path, textured_mesh);
  } else {
    WriteTextPlyMesh(mesh_path, textured_mesh);
  }

  LOG(INFO) << "Mesh texture mapping complete";
  return EXIT_SUCCESS;
}

// Canonical unsigned CLI literal for the resident caps: exactly like the
// wire's canonical decimals -- digits only, nonempty, no sign, no leading
// zeros. Mirrors the parser-side canonicality the forecast wire requires;
// the shared driver ParseBytes stays unchanged for owned-driver use.
std::string CanonicalCliUnsigned(const std::string& value, const char* name) {
  if (value.empty() || value.front() == '+' || value.front() == '-' ||
      value.front() == ' ' || value.back() == ' ' ||
      (value.size() > 1 && value.front() == '0')) {
    throw std::invalid_argument(
        std::string("resident ") + name +
        " cap requires a canonical unsigned decimal literal");
  }
  for (const char c : value) {
    if (c < '0' || c > '9') {
      throw std::invalid_argument(
          std::string("resident ") + name +
          " cap requires a canonical unsigned decimal literal");
    }
  }
  return value;
}

int RunPatchMatchStereo(int argc, char** argv) {
  return RunPatchMatchStereoWithExecution(
      argc, argv, mvs::PatchMatchExecution::kStock);
}

#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
namespace {

// Exclusive-creation write for the contract export: fails closed if the
// target exists by ANY name at open time (no truncation race with
// WriteBinaryBlob), completes the full byte count with a checked explicit
// close (descriptor ownership relinquished before the close result is
// checked, never retried), and verifies the resulting file size. On ANY
// failure the pathname is NOT removed: an actor may have replaced it after
// exclusive creation, so a possibly partial file created by this call is
// left in place, reported with a manual-cleanup requirement, and future
// exports refuse to overwrite it. Pre-existing targets were refused before
// any write and are never touched. POSIX only; unsupported platforms fail
// closed.
void WriteContractFileExclusively(const std::filesystem::path& path,
                                  const std::string& bytes) {
#if defined(_WIN32)
  THROW_CHECK(false)
      << "Contract export uses exclusive creation, unavailable on Windows";
#else
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  THROW_CHECK_GE(fd, 0)
      << "Refusing to create the contract file (exists or cannot create): "
      << path;
  bool descriptor_released = false;
  try {
    size_t written = 0;
    while (written < bytes.size()) {
      const ssize_t chunk =
          ::write(fd, bytes.data() + written, bytes.size() - written);
      THROW_CHECK_GT(chunk, 0) << "Contract export write failed: " << path;
      written += static_cast<size_t>(chunk);
    }
    // Relinquish descriptor ownership BEFORE checking the close result:
    // close() may report an error after already releasing the descriptor,
    // and retrying it risks closing a reused descriptor. It is never
    // closed or retried again after this point.
    const int close_result = ::close(fd);
    descriptor_released = true;
    THROW_CHECK_EQ(close_result, 0) << "Contract export close failed: " << path;
    std::error_code error;
    const auto actual = std::filesystem::file_size(path, error);
    THROW_CHECK(!error && actual == bytes.size())
        << "Contract export incomplete: " << path;
    LOG(INFO) << "Contract file written exclusively and completely: " << path;
  } catch (...) {
    if (!descriptor_released) {
      ::close(fd);
    }
    // No automatic pathname removal: another actor may have replaced the
    // path after this exclusive creation, and path-ownership races are not
    // handled here. A partial or closed-but-unverified file, if this call
    // created it, is left in place and must be cleaned up manually; future
    // exports refuse to overwrite the existing path.
    throw;
  }
#endif
}

// CPU-only metadata-only mode: export or verify the canonical owned-bank
// contract. No payload decode, no CUDA, no solver and no workspace writes,
// except the new contract export file. Verification re-resolves the SAME
// invocation once natively and compares exact canonical contract bytes;
// equality is a successful binding, never a certification.
int RunOwnedBankMetadataMode(const std::filesystem::path& workspace_path,
                             const std::string& workspace_format,
                             const std::string& pmvs_option_name,
                             const mvs::PatchMatchOptions& options,
                             const std::filesystem::path& config_path,
                             const std::filesystem::path& export_path,
                             const std::filesystem::path& verify_path) {
  const bool verify = !verify_path.empty();
  const std::filesystem::path contract_path =
      verify ? verify_path : export_path;

  std::string workspace_format_lower = workspace_format;
  StringToLower(&workspace_format_lower);
  THROW_CHECK(workspace_format_lower == "colmap" ||
              workspace_format_lower == "pmvs")
      << "Invalid `workspace_format` " << workspace_format_lower
      << " - supported values are 'COLMAP' or 'PMVS'.";

  mvs::PatchMatchController controller(options,
                                       workspace_path,
                                       workspace_format_lower,
                                       pmvs_option_name,
                                       config_path,
                                       mvs::PatchMatchExecution::kStock);

  if (!verify) {
    THROW_CHECK(!ExistsFile(contract_path))
        << "Refusing to overwrite an existing contract file: " << contract_path;
    const auto contract = controller.ResolveOwnedBankContract();
    // The export target must never be any planned solver output of this
    // invocation (depth or normal, compared on resolved paths).
    for (const auto& problem : contract.problems) {
      for (const auto& output :
           {problem.output_depth_path, problem.output_normal_path}) {
        THROW_CHECK(std::filesystem::weakly_canonical(output) !=
                    std::filesystem::weakly_canonical(contract_path))
            << "Contract export target collides with a planned solver "
               "output: "
            << contract_path;
      }
    }
    const auto bytes = mvs::SerializeOwnedBankContract(contract);
    WriteContractFileExclusively(contract_path, bytes);
    const auto digest = [] {
#if defined(COLMAP_DOWNLOAD_ENABLED)
      return &ComputeSHA256;
#else
      return static_cast<std::string (*)(const std::string_view&)>(nullptr);
#endif
    }();
    LOG(INFO) << "Exported owned bank contract to " << contract_path << " ("
              << bytes.size() << " bytes"
              << (digest == nullptr
                      ? std::string(
                            ": sha256 unavailable without download support)")
                      : std::string(", sha256 " + digest(bytes) +
                                    "; binding snapshot, not certification)"));
  } else {
    THROW_CHECK(ExistsFile(contract_path));
    std::vector<char> prior_data;
    ReadBinaryBlob(contract_path, &prior_data);
    const std::string prior_bytes(prior_data.begin(), prior_data.end());
    const auto parsed = mvs::ParseOwnedBankContract(prior_bytes);
    THROW_CHECK_EQ(mvs::SerializeOwnedBankContract(parsed), prior_bytes)
        << "Stored contract is not in canonical form: " << contract_path;
    const auto fresh = controller.ResolveOwnedBankContract();
    const auto fresh_bytes = mvs::SerializeOwnedBankContract(fresh);
    if (fresh_bytes != prior_bytes) {
      const auto difference =
          mvs::FirstOwnedBankContractByteDifference(prior_bytes, fresh_bytes);
      LOG(ERROR) << "Fresh resolution differs from the prior contract at byte "
                 << (difference == std::string::npos ? prior_bytes.size()
                                                     : difference);
      THROW_CHECK(false)
          << "Owned bank contract verification failed (changed order, "
             "options, files, layout or source identity before payload)";
    }
    LOG(INFO) << "Owned bank contract verified: fresh native resolution "
              << "matches the stored invocation snapshot exactly ("
              << prior_bytes.size()
              << " bytes; successful binding, not certification)";
  }
  return EXIT_SUCCESS;
}

}  // namespace
#endif

int RunPatchMatchStereoWithExecution(
    int argc,
    char** argv,
    const mvs::PatchMatchExecution execution,
    const mvs::TrustedPhotoPhaseV1Delivery& phase_delivery,
    const mvs::OptimizedPhotoInputProbe& input_probe,
    const mvs::PhotoPhaseV1RunnerDelivery& runner_delivery) {
  const auto phase_entry = mvs::NativeRunDeadline::Clock::now();
  std::optional<mvs::NativeRunDeadline> execution_deadline;
  std::unique_ptr<mvs::NativeRunDeadlineGuard> deadline_guard;
  std::filesystem::path workspace_path;
  std::string workspace_format = "COLMAP";
  std::string pmvs_option_name = "option-all";
  std::filesystem::path config_path;
  // Explicit metadata-only mode controls, separate from PatchMatchOptions.
  // Stock solver execution remains the default when neither flag is set.
  std::filesystem::path owned_metadata_export_path;
  std::filesystem::path owned_metadata_verify_path;
  // Explicit bounded resident execution backend (opt-in only; stock default
  // unchanged). Caps are separate CLI values; charges come from the wire.
  std::filesystem::path owned_run_forecast_path;
  // Explicit bounded COHORT execution backend (opt-in only; stock default
  // unchanged; distinct cohort wire; strictly sequential FIFO cohorts).
  std::filesystem::path owned_run_cohort_forecast_path;
  std::string owned_run_host_budget_bytes;
  std::string owned_run_device_budget_bytes;
  std::string owned_run_max_concurrent_solvers;
  std::string owned_run_timeout_seconds;
  bool optimized_only = false;
  int owned_tile_threads = 32;
  std::filesystem::path optimized_dispatch_receipt;
  std::filesystem::path owned_photo_forecast;
  std::filesystem::path owned_photo_options_export;

  OptionManager options;
  options.AddRequiredOption(
      "workspace_path",
      &workspace_path,
      "Path to the folder containing the undistorted images");
  options.AddDefaultOption(
      "workspace_format", &workspace_format, "{COLMAP, PMVS}");
  options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
  options.AddDefaultOption("config_path", &config_path);
  options.AddPatchMatchStereoOptions();
  options.AddDefaultOption(
      "optimized_only",
      &optimized_only,
      "Require supported owned dispatch; never ordinary fallback");
  options.AddDefaultOption(
      "optimized_dispatch_receipt",
      &optimized_dispatch_receipt,
      "NEW JSON completion receipt; required with optimized_only");
  options.AddDefaultOption("owned_tile_threads", &owned_tile_threads);
  options.AddDefaultOption(
      "owned_photo_forecast",
      &owned_photo_forecast,
      "Reviewed SHA-bound single-reference PHOTO admission wire v2");
  options.AddDefaultOption(
      "owned_photo_options_export",
      &owned_photo_options_export,
      "Metadata-only: export native PHOTO option binding bytes to a NEW file");
  options.AddDefaultOption("owned_metadata_export",
                           &owned_metadata_export_path,
                           "Metadata-only mode: export the canonical owned "
                           "bank contract to this NEW file; no solver runs");
  options.AddDefaultOption("owned_metadata_verify",
                           &owned_metadata_verify_path,
                           "Metadata-only mode: re-resolve once and verify "
                           "against this prior contract file; no solver runs");
  options.AddDefaultOption(
      "owned_run_forecast",
      &owned_run_forecast_path,
      "Opt-in bounded resident execution: run the full bank under this "
      "verified resident-forecast wire (bounded normals; single explicit "
      "PatchMatchStereo.gpu_index); stock remains the default");
  options.AddDefaultOption(
      "owned_run_cohort_forecast",
      &owned_run_cohort_forecast_path,
      "Opt-in bounded COHORT resident execution: run strictly sequential FIFO "
      "cohorts under this verified resident-cohort-forecast wire (distinct "
      "magic; per-cohort admission ledgers; bounded normals; single explicit "
      "PatchMatchStereo.gpu_index); stock remains the default");
  options.AddDefaultOption(
      "owned_run_timeout_seconds",
      &owned_run_timeout_seconds,
      "Positive canonical integer: absolute owned phase execution budget "
      "including setup/drain/join; required with optimized_only");
  options.AddDefaultOption(
      "owned_run_host_budget_bytes",
      &owned_run_host_budget_bytes,
      "Resident execution host budget cap (required with owned_run_forecast)");
  options.AddDefaultOption("owned_run_device_budget_bytes",
                           &owned_run_device_budget_bytes,
                           "Resident execution device budget cap (required "
                           "with owned_run_forecast)");
  options.AddDefaultOption(
      "owned_run_max_concurrent_solvers",
      &owned_run_max_concurrent_solvers,
      "Resident execution admitted rolling concurrency; must equal the "
      "forecast's declared active_jobs (required with owned_run_forecast)");
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }
  // Fail-closed raw-argv intent check, BEFORE any controller/workspace/
  // payload work: an explicit metadata-mode request must never silently
  // fall back to stock execution. When `--project_path` is given, the
  // project-file parse path can discard CLI-provided option values, so a
  // metadata mode flag could be lost silently; combining those flags with
  // a project file is therefore refused outright at this CPU stage. No
  // generic OptionManager semantics are changed; stock remains the default.
  bool raw_export_request = false;
  bool raw_verify_request = false;
  bool raw_run_request = false;
  bool raw_project_path = false;
  bool raw_timeout_request = false;
  bool raw_optimized_request = false;
  const auto raw_flag = [](const std::string& arg, const char* name) {
    return arg == name ||
           arg.compare(
               0, std::string(name).size() + 1, std::string(name) + "=") == 0;
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i] != nullptr ? argv[i] : "");
    raw_export_request =
        raw_export_request || raw_flag(arg, "--owned_metadata_export");
    raw_verify_request =
        raw_verify_request || raw_flag(arg, "--owned_metadata_verify");
    raw_timeout_request =
        raw_timeout_request || raw_flag(arg, "--owned_run_timeout_seconds");
    raw_run_request = raw_run_request || raw_timeout_request ||
                      raw_flag(arg, "--owned_run_forecast") ||
                      raw_flag(arg, "--owned_run_cohort_forecast") ||
                      raw_flag(arg, "--owned_run_host_budget_bytes") ||
                      raw_flag(arg, "--owned_run_device_budget_bytes") ||
                      raw_flag(arg, "--owned_run_max_concurrent_solvers");
    raw_optimized_request = raw_optimized_request ||
                            raw_flag(arg, "--optimized_only") ||
                            raw_flag(arg, "--optimized_dispatch_receipt") ||
                            raw_flag(arg, "--owned_tile_threads") ||
                            raw_flag(arg, "--owned_photo_forecast") ||
                            raw_flag(arg, "--owned_photo_options_export");
    raw_project_path = raw_project_path || raw_flag(arg, "--project_path");
  }
  if (raw_project_path && (raw_export_request || raw_verify_request ||
                           raw_run_request || raw_optimized_request)) {
    LOG(ERROR) << "--project_path combined with owned bank metadata or "
                  "resident execution modes is unsupported; the "
                  "project-file reread can discard explicit CLI mode "
                  "requests. Refusing to run; re-run with the mode flags "
                  "directly.";
    return EXIT_FAILURE;
  }
  if ((raw_export_request || raw_verify_request || raw_run_request) &&
      owned_metadata_export_path.empty() &&
      owned_metadata_verify_path.empty() && owned_run_forecast_path.empty() &&
      owned_run_cohort_forecast_path.empty() && owned_photo_forecast.empty()) {
    LOG(ERROR) << "An explicit owned bank metadata or resident execution "
                  "mode was requested on the command line but was not "
                  "applied (e.g., discarded by the --project_path "
                  "project-file reread). Refusing to silently run the "
                  "stock solver; re-run without --project_path.";
    return EXIT_FAILURE;
  }
  // Resolve strict dispatch before any controller, decoder or solver exists.
  const bool photo_run_mode = !owned_photo_forecast.empty();
  const bool photo_export_mode = !owned_photo_options_export.empty();
  try {
    if (raw_timeout_request && owned_run_timeout_seconds.empty())
      throw std::invalid_argument("owned_run_timeout_seconds cannot be empty");
    const bool owned_execution =
        (photo_run_mode || !owned_run_forecast_path.empty() ||
         !owned_run_cohort_forecast_path.empty()) &&
        !photo_export_mode && owned_metadata_export_path.empty() &&
        owned_metadata_verify_path.empty();
    execution_deadline = mvs::ResolveOwnedRunDeadline(owned_run_timeout_seconds,
                                                      optimized_only,
                                                      owned_execution,
                                                      phase_entry);
    deadline_guard =
        std::make_unique<mvs::NativeRunDeadlineGuard>(execution_deadline);
    deadline_guard->Check();
    if (!optimized_only &&
        (!optimized_dispatch_receipt.empty() || photo_run_mode))
      throw std::invalid_argument(
          "receipt/PHOTO forecast requires optimized_only=true");
    if (optimized_only) {
#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
      constexpr bool available = true;
#else
      constexpr bool available = false;
#endif
      const auto& pm = *options.patch_match_stereo;
      mvs::ValidateOptimizedStereo(available,
                                   owned_tile_threads,
                                   pm.num_samples,
                                   pm.owned_distinct_source_cache,
                                   pm.owned_tex_lookahead,
                                   pm.window_radius,
                                   pm.window_step,
                                   pm.filter,
                                   pm.write_consistency_graph,
                                   pm.allow_missing_files,
                                   pm.geom_consistency);
      // GEO binds here; PHOTO binds supplied k, exact CLI N, job count and
      // private top-N caps together when consuming its V2/V3 launch forecast.
      mvs::OptimizedConcurrencyLimits concurrency;
      concurrency.geom_consistency = pm.geom_consistency;
      concurrency.concurrency = owned_run_max_concurrent_solvers;
      concurrency.full_bank_forecast = !owned_run_forecast_path.empty();
      concurrency.cohort_forecast = !owned_run_cohort_forecast_path.empty();
      concurrency.photo_ceiling = mvs::kOptimizedPhotoQualifiedConcurrency;
      if (pm.geom_consistency)
        mvs::ValidateOptimizedStereoConcurrency(concurrency);
      else
        (void)mvs::ParseOptimizedConcurrency(concurrency.concurrency);
      mvs::ValidateOptimizedReceiptPath(optimized_dispatch_receipt);
      if (execution != mvs::PatchMatchExecution::kStock ||
          workspace_format != "COLMAP" || pm.max_image_size != -1 ||
          pm.num_threads < 1 || !owned_metadata_export_path.empty() ||
          !owned_metadata_verify_path.empty() || photo_export_mode)
        throw std::invalid_argument(
            "optimized_only requires native COLMAP, explicit threads, "
            "supported explicit concurrency, execution not metadata");
      if (pm.geom_consistency
              ? photo_run_mode || (owned_run_forecast_path.empty() ==
                                   owned_run_cohort_forecast_path.empty())
              : !photo_run_mode || !owned_run_forecast_path.empty() ||
                    !owned_run_cohort_forecast_path.empty())
        throw std::invalid_argument(
            "optimized GEO requires exactly one resident forecast; PHOTO "
            "requires owned_photo_forecast only");
    }
    if (photo_export_mode) {
      if (optimized_only || photo_run_mode ||
          !owned_run_forecast_path.empty() ||
          !owned_run_cohort_forecast_path.empty() ||
          !owned_metadata_export_path.empty() ||
          !owned_metadata_verify_path.empty() ||
          options.patch_match_stereo->geom_consistency)
        throw std::invalid_argument(
            "PHOTO option export is metadata-only and mutually exclusive with "
            "execution");
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
      WriteContractFileExclusively(
          owned_photo_options_export,
          mvs::OptimizedPhotoOptions(
              *options.patch_match_stereo, workspace_path, config_path));
      return EXIT_SUCCESS;
#else
      throw std::invalid_argument(
          "PHOTO metadata export requires dense-stereo library");
#endif
    }
    if (photo_run_mode) {
#if defined(COLMAP_CUDA_ENABLED) && defined(COLMAP_DOWNLOAD_ENABLED) && \
    !defined(COLMAP_HIP_ENABLED)
      if (config_path.empty())
        throw std::invalid_argument(
            "PHOTO requires a single-reference config_path");
      if (phase_delivery && runner_delivery)
        throw std::invalid_argument("PHOTO conflicting delivery sources");
      // Explicit in-process sources remain available to native callers. The
      // ordinary CLI consumes the producer's typed binding sections in V3.
      const mvs::owned::ByteCharge photo_capacity{
          mvs::owned::ParseBytes(
              CanonicalCliUnsigned(owned_run_host_budget_bytes, "host")),
          mvs::owned::ParseBytes(
              CanonicalCliUnsigned(owned_run_device_budget_bytes, "device"))};
      const auto trusted_delivery =
          runner_delivery
              ? mvs::MakeTrustedPhotoPhaseV1Delivery(runner_delivery)
          : phase_delivery ? phase_delivery
                           : mvs::MakeWirePhotoPhaseV1Delivery(
                                 config_path,
                                 owned_run_max_concurrent_solvers,
                                 photo_capacity,
                                 [&](const std::filesystem::path& job_config) {
                                   return mvs::OptimizedPhotoOptions(
                                       *options.patch_match_stereo,
                                       workspace_path,
                                       job_config);
                                 },
                                 [](const std::string_view& bytes) {
                                   return ComputeSHA256(bytes);
                                 });
      const auto photo_observation =
          mvs::RunOptimizedPhoto(*options.patch_match_stereo,
                                 workspace_path,
                                 config_path,
                                 owned_photo_forecast,
                                 photo_capacity,
                                 owned_tile_threads,
                                 execution_deadline,
                                 owned_run_max_concurrent_solvers,
                                 trusted_delivery,
                                 input_probe);
      deadline_guard->Check();
      // N1 preserves historical receipt bytes; N>1 uses real admitted/joined
      // coordinator events, never requested concurrency as observed telemetry.
      mvs::PublishOptimizedReceipt(
          optimized_dispatch_receipt,
          mvs::OptimizedDispatchReceipt(
              "photo",
              owned_tile_threads,
              options.patch_match_stereo->num_samples,
              true,
              options.patch_match_stereo->owned_tex_lookahead,
              photo_observation.admitted_n,
              photo_observation.admitted_n > 1 ? &photo_observation : nullptr));
      return EXIT_SUCCESS;
#else
      throw std::invalid_argument(
          "optimized PHOTO requires CUDA and SHA256 support");
#endif
    }
  } catch (const std::exception& exc) {
    LOG(ERROR) << exc.what();
    return EXIT_FAILURE;
  }
  const bool export_mode = !owned_metadata_export_path.empty();
  const bool verify_mode = !owned_metadata_verify_path.empty();
  const bool run_mode = !owned_run_forecast_path.empty();
  const bool cohort_run_mode = !owned_run_cohort_forecast_path.empty();
  if (run_mode && cohort_run_mode) {
    LOG(ERROR) << "owned_run_forecast and owned_run_cohort_forecast are "
                  "mutually exclusive (distinct wires, distinct magics); "
                  "pass exactly one";
    return EXIT_FAILURE;
  }
  if ((export_mode || verify_mode) && (run_mode || cohort_run_mode)) {
    LOG(ERROR) << "The owned bank metadata modes and either resident "
                  "execution mode (owned_run_forecast or "
                  "owned_run_cohort_forecast) are mutually exclusive; "
                  "pass exactly one";
    return EXIT_FAILURE;
  }
  if (export_mode && verify_mode) {
    LOG(ERROR) << "owned_metadata_export and owned_metadata_verify are "
                  "mutually exclusive; pass exactly one";
    return EXIT_FAILURE;
  }
  if (raw_run_request && !run_mode && !cohort_run_mode) {
    LOG(ERROR) << "Resident execution was requested on the command line "
                  "but was not applied (forecast flags missing or "
                  "discarded); refusing to silently run the stock solver";
    return EXIT_FAILURE;
  }
  if (cohort_run_mode) {
    if (owned_run_host_budget_bytes.empty() ||
        owned_run_device_budget_bytes.empty() ||
        owned_run_max_concurrent_solvers.empty()) {
      LOG(ERROR) << "Cohort resident execution requires "
                    "owned_run_host_budget_bytes, "
                    "owned_run_device_budget_bytes and "
                    "owned_run_max_concurrent_solvers together with "
                    "owned_run_cohort_forecast";
      return EXIT_FAILURE;
    }
    if (execution != mvs::PatchMatchExecution::kStock) {
      LOG(ERROR) << "The cohort runner is dispatched with the standard "
                    "CLI only; no extra execution flag is accepted";
      return EXIT_FAILURE;
    }
    uint64_t host_budget_bytes = 0;
    uint64_t device_budget_bytes = 0;
    try {
      host_budget_bytes = mvs::owned::ParseBytes(
          CanonicalCliUnsigned(owned_run_host_budget_bytes, "host budget"));
      device_budget_bytes = mvs::owned::ParseBytes(
          CanonicalCliUnsigned(owned_run_device_budget_bytes, "device budget"));
      const uint64_t concurrency = mvs::owned::ParseBytes(CanonicalCliUnsigned(
          owned_run_max_concurrent_solvers, "concurrency"));
      if (concurrency > 2147483647ull) {
        throw std::invalid_argument(
            "resident concurrency exceeds the admitted range");
      }
      if (!std::filesystem::is_regular_file(owned_run_cohort_forecast_path)) {
        throw std::invalid_argument(
            "cohort forecast file is missing or not a regular file: " +
            owned_run_cohort_forecast_path.string());
      }
      mvs::resident::ResidentCohortLimits limits;
      limits.host_budget_bytes = host_budget_bytes;
      limits.device_budget_bytes = device_budget_bytes;
      limits.max_concurrent_solvers = static_cast<size_t>(concurrency);
      limits.owned_tile_threads = owned_tile_threads;
      limits.execution_deadline = execution_deadline;
#if defined(COLMAP_CUDA_ENABLED)
      mvs::ResidentSha256Fn sha = nullptr;
#if defined(COLMAP_DOWNLOAD_ENABLED)
      sha = &ComputeSHA256;
#endif
      if (sha == nullptr) {
        LOG(ERROR) << "Cohort execution requires SHA-256 verification; "
                      "this build lacks the crypto backend and refuses";
        return EXIT_FAILURE;
      }
      const int result =
          mvs::resident::RunResidentCohortBank(*options.patch_match_stereo,
                                               workspace_path,
                                               workspace_format,
                                               pmvs_option_name,
                                               config_path,
                                               owned_run_cohort_forecast_path,
                                               limits,
                                               sha);
      deadline_guard->Check();
      if (optimized_only && result == EXIT_SUCCESS)
        mvs::PublishOptimizedReceipt(
            optimized_dispatch_receipt,
            mvs::OptimizedDispatchReceipt(
                "geo",
                owned_tile_threads,
                options.patch_match_stereo->num_samples,
                true,
                options.patch_match_stereo->owned_tex_lookahead));
      return result;
#else
      LOG(ERROR) << "The cohort runner requires the dense-stereo "
                    "library, which is built with CUDA support";
      return EXIT_FAILURE;
#endif
    } catch (const std::exception& exc) {
      LOG(ERROR) << "Cohort execution failed: " << exc.what();
      return EXIT_FAILURE;
    }
  }
  if (run_mode) {
    if (owned_run_host_budget_bytes.empty() ||
        owned_run_device_budget_bytes.empty() ||
        owned_run_max_concurrent_solvers.empty()) {
      LOG(ERROR) << "Resident execution requires owned_run_host_budget_bytes, "
                    "owned_run_device_budget_bytes and "
                    "owned_run_max_concurrent_solvers together with "
                    "owned_run_forecast";
      return EXIT_FAILURE;
    }
    if (execution != mvs::PatchMatchExecution::kStock) {
      LOG(ERROR) << "The resident runner is dispatched with the standard "
                    "CLI only; no extra execution flag is accepted";
      return EXIT_FAILURE;
    }
    uint64_t host_budget_bytes = 0;
    uint64_t device_budget_bytes = 0;
    try {
      host_budget_bytes = mvs::owned::ParseBytes(
          CanonicalCliUnsigned(owned_run_host_budget_bytes, "host budget"));
      device_budget_bytes = mvs::owned::ParseBytes(
          CanonicalCliUnsigned(owned_run_device_budget_bytes, "device budget"));
      const uint64_t concurrency = mvs::owned::ParseBytes(CanonicalCliUnsigned(
          owned_run_max_concurrent_solvers, "concurrency"));
      if (concurrency > 2147483647ull) {
        throw std::invalid_argument(
            "resident concurrency exceeds the admitted range");
      }
      if (!std::filesystem::is_regular_file(owned_run_forecast_path)) {
        throw std::invalid_argument(
            "resident forecast file is missing or not a regular file: " +
            owned_run_forecast_path.string());
      }
      mvs::resident::ResidentRunnerLimits limits;
      limits.host_budget_bytes = host_budget_bytes;
      limits.device_budget_bytes = device_budget_bytes;
      limits.max_concurrent_solvers = static_cast<size_t>(concurrency);
      limits.owned_tile_threads = owned_tile_threads;
      limits.execution_deadline = execution_deadline;
#if defined(COLMAP_CUDA_ENABLED)
      // Execution requires REAL SHA-256 verification: builds without the
      // crypto backend refuse here, before any resolution or payload.
      mvs::ResidentSha256Fn sha = nullptr;
#if defined(COLMAP_DOWNLOAD_ENABLED)
      sha = &ComputeSHA256;
#endif
      if (sha == nullptr) {
        LOG(ERROR) << "Resident execution requires SHA-256 verification; "
                      "this build lacks the crypto backend and refuses";
        return EXIT_FAILURE;
      }
      const int result =
          mvs::resident::RunResidentBank(*options.patch_match_stereo,
                                         workspace_path,
                                         workspace_format,
                                         pmvs_option_name,
                                         config_path,
                                         owned_run_forecast_path,
                                         limits,
                                         sha);
      deadline_guard->Check();
      if (optimized_only && result == EXIT_SUCCESS)
        mvs::PublishOptimizedReceipt(
            optimized_dispatch_receipt,
            mvs::OptimizedDispatchReceipt(
                "geo",
                owned_tile_threads,
                options.patch_match_stereo->num_samples,
                true,
                options.patch_match_stereo->owned_tex_lookahead,
                static_cast<int>(limits.max_concurrent_solvers)));
      return result;
#else
      LOG(ERROR) << "The resident runner requires the dense-stereo "
                    "library, which is built with CUDA support";
      return EXIT_FAILURE;
#endif
    } catch (const std::exception& exc) {
      LOG(ERROR) << "Resident execution failed: " << exc.what();
      return EXIT_FAILURE;
    }
  }
  if (export_mode || verify_mode) {
    if (execution != mvs::PatchMatchExecution::kStock) {
      LOG(ERROR) << "Owned bank metadata modes are CPU-only stock dispatch; "
                    "no execution flag is accepted";
      return EXIT_FAILURE;
    }
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
    try {
      return RunOwnedBankMetadataMode(workspace_path,
                                      workspace_format,
                                      pmvs_option_name,
                                      *options.patch_match_stereo,
                                      config_path,
                                      owned_metadata_export_path,
                                      owned_metadata_verify_path);
    } catch (const std::exception& exc) {
      LOG(ERROR) << "Owned bank metadata mode failed: " << exc.what();
      return EXIT_FAILURE;
    }
#else
    LOG(ERROR) << "The owned bank metadata modes require the dense-stereo "
                  "library, which is built with CUDA or HIP support";
    return EXIT_FAILURE;
#endif
  }
  RunPatchMatchStereoImplWithExecution(workspace_path,
                                       workspace_format,
                                       pmvs_option_name,
                                       *options.patch_match_stereo,
                                       config_path,
                                       execution);
  return EXIT_SUCCESS;
}

void RunPatchMatchStereoImpl(const std::filesystem::path& workspace_path,
                             const std::string& workspace_format,
                             const std::string& pmvs_option_name,
                             const mvs::PatchMatchOptions& options,
                             const std::filesystem::path& config_path) {
  RunPatchMatchStereoImplWithExecution(workspace_path,
                                       workspace_format,
                                       pmvs_option_name,
                                       options,
                                       config_path,
                                       mvs::PatchMatchExecution::kStock);
}

void RunPatchMatchStereoImplWithExecution(
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const mvs::PatchMatchOptions& options,
    const std::filesystem::path& config_path,
    const mvs::PatchMatchExecution execution) {
#if !defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
  LOG(FATAL_THROW) << "Dense stereo reconstruction requires CUDA or HIP, "
                      "neither of which is available on your system.";
#else   // COLMAP_CUDA_ENABLED || COLMAP_HIP_ENABLED
  std::string workspace_format_lower = workspace_format;
  StringToLower(&workspace_format_lower);
  THROW_CHECK(workspace_format_lower == "colmap" ||
              workspace_format_lower == "pmvs")
      << "Invalid `workspace_format` " << workspace_format_lower
      << " - supported values are 'COLMAP' or 'PMVS'.";

  mvs::PatchMatchController controller(options,
                                       workspace_path,
                                       workspace_format_lower,
                                       pmvs_option_name,
                                       config_path,
                                       execution);

  controller.Run();
#endif  // COLMAP_CUDA_ENABLED || COLMAP_HIP_ENABLED
}

int RunPoissonMesher(int argc, char** argv) {
  std::filesystem::path input_path;
  std::filesystem::path output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddPoissonMeshingOptions();
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  THROW_CHECK(
      mvs::PoissonMeshing(*options.poisson_meshing, input_path, output_path));

  return EXIT_SUCCESS;
}

namespace {

Reconstruction RunStereoFuserWithMaterialization(
    const std::filesystem::path& output_path,
    const std::filesystem::path& workspace_path,
    std::string workspace_format,
    const std::string& pmvs_option_name,
    std::string input_type,
    const mvs::StereoFusionOptions& options,
    std::string output_type,
    std::function<bool()> check_if_stopped,
    bool materialize_ply);

}  // namespace

int RunStereoFuser(int argc, char** argv) {
  std::filesystem::path workspace_path;
  std::string input_type = "geometric";
  std::string workspace_format = "COLMAP";
  std::string pmvs_option_name = "option-all";
  std::string output_type = "PLY";
  std::filesystem::path output_path;

  bool optimized_only = false;
  std::filesystem::path optimized_dispatch_receipt;
  OptionManager options;
  options.AddRequiredOption("workspace_path", &workspace_path);
  options.AddDefaultOption(
      "workspace_format", &workspace_format, "{COLMAP, PMVS}");
  options.AddDefaultOption("pmvs_option_name", &pmvs_option_name);
  options.AddDefaultOption(
      "input_type", &input_type, "{photometric, geometric}");
  options.AddDefaultOption("output_type", &output_type, "{BIN, TXT, PLY}");
  options.AddRequiredOption("output_path", &output_path);
  options.AddStereoFusionOptions();
  options.AddDefaultOption("optimized_only", &optimized_only);
  options.AddDefaultOption("optimized_dispatch_receipt",
                           &optimized_dispatch_receipt);
  if (!options.Parse(argc, argv)) {
    return EXIT_FAILURE;
  }

  try {
    bool project = false, strict_request = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      project = project || arg == "--project_path" ||
                arg.rfind("--project_path=", 0) == 0;
      strict_request = strict_request || arg.rfind("--optimized_", 0) == 0;
    }
    if (project && strict_request)
      throw std::invalid_argument(
          "optimized fusion forbids project_path overrides");
    if (!optimized_only && !optimized_dispatch_receipt.empty())
      throw std::invalid_argument("receipt requires optimized_only=true");
    if (optimized_only) {
      const auto& f = *options.stereo_fusion;
      mvs::ValidateOptimizedFusion(workspace_format,
                                   f.use_mmap,
                                   f.use_cache,
                                   f.num_threads,
                                   f.max_image_size);
      if (!f.Check())
        throw std::invalid_argument("invalid optimized fusion options");
      mvs::ValidateOptimizedReceiptPath(optimized_dispatch_receipt);
    }
  } catch (const std::exception& exc) {
    LOG(ERROR) << exc.what();
    return EXIT_FAILURE;
  }

  options.stereo_fusion->require_complete_inputs = optimized_only;
  // The CLI discards the returned model. PLY/VIS are written directly from
  // the fuser, so do not duplicate the dense points in a Reconstruction.
  RunStereoFuserWithMaterialization(output_path,
                                    workspace_path,
                                    workspace_format,
                                    pmvs_option_name,
                                    input_type,
                                    *options.stereo_fusion,
                                    output_type,
                                    {},
                                    /*materialize_ply=*/false);

  if (optimized_only)
    mvs::PublishOptimizedReceipt(optimized_dispatch_receipt,
                                 mvs::OptimizedDispatchReceipt("fusion"));
  return EXIT_SUCCESS;
}

Reconstruction RunStereoFuserImpl(const std::filesystem::path& output_path,
                                  const std::filesystem::path& workspace_path,
                                  std::string workspace_format,
                                  const std::string& pmvs_option_name,
                                  std::string input_type,
                                  const mvs::StereoFusionOptions& options,
                                  std::string output_type,
                                  std::function<bool()> check_if_stopped) {
  return RunStereoFuserWithMaterialization(output_path,
                                           workspace_path,
                                           std::move(workspace_format),
                                           pmvs_option_name,
                                           std::move(input_type),
                                           options,
                                           std::move(output_type),
                                           std::move(check_if_stopped),
                                           /*materialize_ply=*/true);
}

namespace {

Reconstruction RunStereoFuserWithMaterialization(
    const std::filesystem::path& output_path,
    const std::filesystem::path& workspace_path,
    std::string workspace_format,
    const std::string& pmvs_option_name,
    std::string input_type,
    const mvs::StereoFusionOptions& options,
    std::string output_type,
    std::function<bool()> check_if_stopped,
    bool materialize_ply) {
  StringToLower(&workspace_format);
  THROW_CHECK(workspace_format == "colmap" || workspace_format == "pmvs")
      << "Invalid `workspace_format` " << workspace_format
      << " - supported values are 'COLMAP' or 'PMVS'.";

  StringToLower(&input_type);
  THROW_CHECK(input_type == "photometric" || input_type == "geometric")
      << "Invalid `input_type` " << input_type
      << " - supported values are 'photometric' and 'geometric'.";

  StringToLower(&output_type);
  THROW_CHECK(output_type == "bin" || output_type == "ply" ||
              output_type == "txt")
      << "Invalid `output_type` " << output_type
      << " - supported values are 'bin', 'ply' and 'txt'.";

  mvs::StereoFusion fuser(
      options, workspace_path, workspace_format, pmvs_option_name, input_type);
  fuser.SetCheckIfStoppedFunc(std::move(check_if_stopped));

  fuser.Run();

  const bool stopped = fuser.CheckIfStopped();
  if (stopped && fuser.GetFusedPoints().empty()) {
    return {};
  }

  Reconstruction reconstruction;

  // Keep the post-fusion sparse model read/validation even for CLI PLY. The
  // public API and BIN/TXT also need its cameras and registered images.
  if (workspace_format == "colmap") {
    reconstruction.Read(workspace_path / "sparse");
  }

  if (materialize_ply || output_type != "ply") {
    // Overwrite sparse points only when the model is returned or serialized.
    reconstruction.ImportPLY(fuser.GetFusedPoints());
  }

  std::filesystem::path write_path = output_path;
  if (stopped) {
    if (output_type == "ply") {
      write_path = output_path.parent_path() /
                   (output_path.stem().string() + ".partial" +
                    output_path.extension().string());
    } else {
      std::filesystem::path normalized_output_path =
          output_path.lexically_normal();
      if (normalized_output_path.filename().empty()) {
        normalized_output_path = normalized_output_path.parent_path();
      }
      write_path = normalized_output_path.parent_path() /
                   (normalized_output_path.filename().string() + ".partial");
      CreateDirIfNotExists(write_path);
    }
    LOG(WARNING) << "Writing partial fusion output to " << write_path;
  } else {
    LOG(INFO) << "Writing output: " << write_path;
  }

  // write output
  if (output_type == "bin") {
    reconstruction.WriteBinary(write_path);
  } else if (output_type == "txt") {
    reconstruction.WriteText(write_path);
  } else if (output_type == "ply") {
    WriteBinaryPlyPoints(write_path, fuser.GetFusedPoints());
    mvs::WritePointsVisibility(AddFileExtension(write_path, ".vis"),
                               fuser.GetFusedPointsVisibility());
  } else {
    LOG(FATAL_THROW) << "Invalid output_type: " << output_type;
  }

  return reconstruction;
}

}  // namespace
}  // namespace colmap
