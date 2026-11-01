// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/exe/optimized_dense_plan.h"

#include "colmap/exe/optimized_dense_plan_io.h"
#include "colmap/exe/optimized_dense_plan_options.h"
#include "colmap/util/logging.h"
#include "colmap/util/version.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>

#if defined(COLMAP_MVS_ENABLED)
#include "colmap/mvs/generated/photo_phase_fingerprint.h"
#include "colmap/mvs/generated/resident_allocation_fingerprint.h"
#include "colmap/mvs/owned_fixed_cohort_validation.h"
#endif

#if defined(COLMAP_CUDA_ENABLED) && defined(COLMAP_MVS_ENABLED) &&      \
    defined(COLMAP_DOWNLOAD_ENABLED) && !defined(COLMAP_HIP_ENABLED) && \
    defined(__linux__)
#define COLMAP_DENSE_GRAPH_SUPPORTED
#include "colmap/mvs/optimized_photo_options.h"
#include "colmap/mvs/owned_pair_image.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/file.h"

#include <sys/mman.h>
#include <sys/stat.h>
#endif

namespace colmap {
int RunReconstructionInfo(int argc, char** argv) {
#if defined(COLMAP_MVS_ENABLED)
  try {
#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
    constexpr bool cuda = true;
#else
    constexpr bool cuda = false;
#endif
    using dense_plan::Json;
    std::vector<std::pair<std::string, Json>> fields = {
        {"schema_version", Json(1)},
        {"cuda_compiled", Json(cuda)},
        {"source_commit", Json(std::string(mvs::kPhotoPhaseV1SourceCommit))},
        {"lineage_sha", Json(std::string(mvs::kPhotoPhaseV1Lineage))},
        {"fingerprint", Json(std::string(mvs::kPhotoPhaseV1Fingerprint))},
        {"allocation_fingerprint",
         Json(std::string(mvs::kResidentAllocationFingerprint))}};
    if (argc != 1) {
      const bool inventory = argc == 3 &&
                             std::string(argv[1]) == "--deadline_seconds";
      if (!inventory &&
          (argc != 5 || std::string(argv[1]) != "--gpu_index" ||
           std::string(argv[3]) != "--deadline_seconds"))
        throw std::invalid_argument(
            "reconstruction_info accepts no arguments or "
            "--deadline_seconds N (visible inventory) or "
            "--gpu_index N --deadline_seconds N (exact device)");
      const auto integer = [](const char* text) {
        int value;
        const auto end = text + std::char_traits<char>::length(text);
        const auto parsed = std::from_chars(text, end, value);
        if (parsed.ec != std::errc() || parsed.ptr != end || value < 0)
          throw std::invalid_argument("invalid resource observation integer");
        return value;
      };
      const int seconds = integer(argv[inventory ? 2 : 4]);
      const auto deadline = mvs::NativeRunDeadline::FromBudget(
          std::chrono::seconds(seconds));
      mvs::NativeRunDeadlineGuard guard(deadline);
      // Same native FREE host/cgroup and CUDA observer as fixed-cohort
      // admission. This is only a pre-stage reserve sanity check, NOT an
      // SfM/fusion memory model, reservation or inferred process credit.
      if (inventory) {
        std::vector<Json> devices;
        for (const auto& device : mvs::owned::ObserveFixedCohortDevices(guard)) {
          const auto& capacity = device.capacity;
          devices.push_back(Json::Object({
              {"ordinal", Json(capacity.device)},
              {"device_uuid", Json(capacity.device_uuid)},
              {"name", Json(capacity.device_name)},
              {"compute_capability_major", Json(capacity.compute_capability_major)},
              {"compute_capability_minor", Json(capacity.compute_capability_minor)},
              {"total_device_bytes", Json(std::to_string(capacity.device_total))},
              {"available_device_bytes",
               device.usable ? Json(std::to_string(capacity.available.device))
                             : Json()},
              {"usable", Json(device.usable)},
              {"reason", Json(device.reason)}}));
        }
        fields.emplace_back("device_inventory", Json::Array(devices));
      } else {
        const int device = integer(argv[2]);
        const auto observed = mvs::owned::ObserveFixedCohortCapacities(device);
        fields.emplace_back("available_host_bytes",
                            Json(std::to_string(observed.available.host)));
        fields.emplace_back("available_device_bytes",
                            Json(std::to_string(observed.available.device)));
        fields.emplace_back("device_uuid", Json(observed.device_uuid));
        fields.emplace_back("device_ordinal", Json(observed.device));
      }
      guard.Check();
    }
    std::cout << Json::Object(fields).Dump() << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "reconstruction_info: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
#else
  (void)argc;
  (void)argv;
  std::cerr << "reconstruction requires MVS support\n";
  return EXIT_FAILURE;
#endif
}

namespace {
using dense_plan::Json;
namespace fs = std::filesystem;
#ifdef COLMAP_DENSE_GRAPH_SUPPORTED
// Hash encoded bytes, never decode image/map payloads. Mapping avoids a whole
// RGB-file heap buffer. This requires immutable inputs, not a sandbox against
// concurrent hostile truncation. Fresh identities are rechecked before publish.
struct InputSnapshot {
  fs::path source_path;
  fs::path path;
  struct stat identity;
  std::string sha256;
};
bool SameStat(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino &&
         a.st_size == b.st_size && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
         a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
         a.st_ctim.tv_sec == b.st_ctim.tv_sec &&
         a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
InputSnapshot Snapshot(const fs::path& path) {
  InputSnapshot result;
  result.source_path = path;
  result.path = fs::canonical(path);
  const int fd = open(result.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) throw std::runtime_error("cannot open input: " + path.string());
  void* mapping = MAP_FAILED;
  size_t size = 0;
  try {
    if (fstat(fd, &result.identity) || !S_ISREG(result.identity.st_mode) ||
        result.identity.st_size < 0 ||
        static_cast<uintmax_t>(result.identity.st_size) >
            std::numeric_limits<size_t>::max())
      throw std::runtime_error("invalid regular input: " + path.string());
    size = static_cast<size_t>(result.identity.st_size);
    if (size) {
      mapping = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (mapping == MAP_FAILED) throw std::runtime_error("cannot map input");
    }
    result.sha256 = ComputeSHA256(
        size ? std::string_view(static_cast<const char*>(mapping), size)
             : std::string_view());
    struct stat after;
    if (fstat(fd, &after) || !SameStat(result.identity, after))
      throw std::runtime_error("input changed during hash: " + path.string());
    if (size) munmap(mapping, size);
    close(fd);
    return result;
  } catch (...) {
    if (mapping != MAP_FAILED) munmap(mapping, size);
    close(fd);
    throw;
  }
}
std::string ReadConfig(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read patch-match.cfg");
  std::string text((std::istreambuf_iterator<char>(input)), {});
  if (input.bad()) throw std::runtime_error("cannot read patch-match.cfg");
  return text;
}
std::vector<fs::path> SparseInputs(const fs::path& workspace) {
  // SAME precedence as Reconstruction::Read, including optional rig/frame data.
  for (const char* ext : {".bin", ".txt"}) {
    std::vector<fs::path> paths;
    for (const char* stem : {"cameras", "images", "points3D"})
      paths.push_back(workspace / "sparse" / (std::string(stem) + ext));
    if (!std::all_of(paths.begin(), paths.end(), [](const auto& p) {
          return fs::is_regular_file(p);
        }))
      continue;
    for (const char* stem : {"rigs", "frames"}) {
      const auto path = workspace / "sparse" / (std::string(stem) + ext);
      if (fs::exists(path)) paths.push_back(path);
    }
    return paths;
  }
  throw std::invalid_argument(
      "missing complete native sparse model in workspace/sparse");
}
Json PhaseJson(const std::vector<mvs::owned::BankJob>& jobs,
               const mvs::PatchMatchOptions& requested_options,
               const fs::path& workspace,
               const fs::path& config,
               const char* history) {
  std::vector<Json> records;
  for (const auto& job : jobs) {
    std::vector<Json> sources;
    for (int source : job.source_image_idxs) sources.emplace_back(source);
    records.push_back(Json::Object(
        {{"reference", Json(job.reference_image_idx)},
         {"sources", Json::Array(sources)},
         {"depth_min", Json(job.options.depth_min)},
         {"depth_max", Json(job.options.depth_max)},
         {"options", dense_plan::OptionsJson(job.options)},
         {"effective_options_wire_v1",
          Json(mvs::OptimizedPhotoOptions(job.options, workspace, config))},
         {"outputs",
          Json::Object({{"depth", Json(job.depth_map_path.string())},
                        {"normal", Json(job.normal_map_path.string())}})}}));
  }
  return Json::Object(
      {{"source_resolution_history", Json(history)},
       {"requested_options", dense_plan::OptionsJson(requested_options)},
       {"jobs", Json::Array(records)}});
}
std::string Produce(const OptimizedDensePlanArguments& args) {
  const auto workspace = fs::canonical(args.workspace_path);
  const auto config = workspace / "stereo/patch-match.cfg";
  std::vector<InputSnapshot> inputs;
  inputs.push_back(Snapshot(config));
  ValidateOptimizedDenseSourceCap(ReadConfig(config), args.source_cap);
  const auto sparse_paths = SparseInputs(workspace);
  for (const auto& path : sparse_paths) inputs.push_back(Snapshot(path));

  mvs::PatchMatchOptions options;
  dense_plan::ApplyLaunchPolicy(args, options);
  options.gpu_index = std::to_string(args.gpu_index);
  options.num_samples = args.num_samples;
  options.num_threads = 1;
  options.owned_distinct_source_cache = true;
  auto photo_requested = options;
  photo_requested.geom_consistency = false;
  photo_requested.filter = false;
  mvs::PatchMatchController controller(options, workspace, "COLMAP", "");
  const auto graph = controller.ResolveDensePlanGraph();
  std::vector<Json> images;
  for (size_t i = 0; i < graph.images.size(); ++i) {
    const auto& image = graph.images[i];
    // Native encoded header check, same configuration as actual Bitmap::Read.
    inputs.push_back(Snapshot(image.GetPath()));
    mvs::owned::ReadPairBitmapPixels(
        image.GetPath(), image.GetWidth(), image.GetHeight());
    images.push_back(
        Json::Object({{"index", Json(static_cast<int>(i))},
                      {"name", Json(graph.names[i])},
                      {"path", Json(fs::canonical(image.GetPath()).string())},
                      {"width", Json(static_cast<int>(image.GetWidth()))},
                      {"height", Json(static_cast<int>(image.GetHeight()))}}));
  }
  std::vector<Json> dependencies;
  for (const auto& job : graph.photo) {
    if (job.source_image_idxs.empty() ||
        job.source_image_idxs.size() > static_cast<size_t>(args.source_cap))
      throw std::invalid_argument(
          "native source count inconsistent with configured maximum");
    const auto& ref = graph.images.at(job.reference_image_idx);
    dependencies.push_back(
        Json::Object({{"reference", Json(job.reference_image_idx)},
                      {"kind", Json("depth")},
                      {"width", Json(static_cast<int>(ref.GetWidth()))},
                      {"height", Json(static_cast<int>(ref.GetHeight()))},
                      {"channels", Json(1)},
                      {"path", Json(job.depth_map_path.string())}}));
    dependencies.push_back(
        Json::Object({{"reference", Json(job.reference_image_idx)},
                      {"kind", Json("normal")},
                      {"width", Json(static_cast<int>(ref.GetWidth()))},
                      {"height", Json(static_cast<int>(ref.GetHeight()))},
                      {"channels", Json(3)},
                      {"path", Json(job.normal_map_path.string())}}));
  }
  std::vector<Json> files;
  std::map<std::string, std::string> hashes;
  for (const auto& input : inputs) {
    const auto inserted = hashes.emplace(input.path.string(), input.sha256);
    if (!inserted.second && inserted.first->second != input.sha256)
      throw std::runtime_error("shared input changed during graph resolution");
    if (inserted.second)
      files.push_back(Json::Object({{"path", Json(input.path.string())},
                                    {"sha256", Json(input.sha256)}}));
  }
  const auto revision = GetCommitRevision();
  if (revision.empty())
    throw std::runtime_error(
        "missing compiled revision: rebuild with native Git metadata");
  const auto document =
      Json::Object(
          {{"schema_version", Json(1)},
           {"workspace_path", Json(workspace.string())},
           {"source_cap", Json(args.source_cap)},
           {"num_samples", Json(args.num_samples)},
           {"dispatch_support",
            dense_plan::DispatchSupportJson(true, args.num_samples)},
           {"concurrency_support",
            dense_plan::ConcurrencySupportJson(true, args.num_samples)},
           {"build",
            Json::Object({{"revision", Json(revision)},
                          {"version", Json(GetVersionInfo())},
                          {"cuda_compiled", Json(true)},
                          {"native_graph_supported", Json(true)}})},
           {"images", Json::Array(images)},
           {"phases",
            Json::Object({{"photo",
                           PhaseJson(graph.photo,
                                     photo_requested,
                                     workspace,
                                     config,
                                     "fresh_controller_photo")},
                          {"geo",
                           PhaseJson(graph.geo,
                                     options,
                                     workspace,
                                     config,
                                     "fresh_controller_after_photo")}})},
           {"input_files", Json::Array(files)},
           {"future_photo_dependencies", Json::Array(dependencies)}})
          .Dump() +
      '\n';
  // Recheck model-format selection too: a newly appearing .bin model must not
  // make this graph claim it came from the retained .txt bytes.
  if (SparseInputs(workspace) != sparse_paths)
    throw std::runtime_error("native sparse model selection changed");
  for (const auto& input : inputs) {
    struct stat current;
    if (fs::canonical(input.source_path) != input.path ||
        stat(input.path.c_str(), &current) ||
        !SameStat(current, input.identity))
      throw std::runtime_error("input changed during graph resolution: " +
                               input.path.string());
  }
  return document;
}
#endif
}  // namespace

int RunOptimizedDensePlan(int argc, char** argv) {
  // No OptionManager/logging overrides: reject unknown argv before any writes.
  FLAGS_logtostderr = true;
  FLAGS_alsologtostderr = false;
#if COLMAP_GLOG_HAS_STDOUT_SUPPORT
  FLAGS_logtostdout = false;
#endif
  FLAGS_log_dir = "";
  try {
    const auto args = ParseOptimizedDensePlanArguments(argc, argv);
#ifdef COLMAP_DENSE_GRAPH_SUPPORTED
    dense_plan::WriteNewAtomic(args.output_path, Produce(args));
    return EXIT_SUCCESS;
#else
    (void)args;
    throw std::runtime_error(
        "optimized_dense_plan native graph unavailable: requires Linux CUDA "
        "MVS "
        "(not HIP), SHA256/download support, and compiled source revision. "
        "Use an approved matching native build; no GPU context is needed for "
        "this query. "
        "No legacy fallback or dense execution was performed.");
#endif
  } catch (const std::exception& error) {
    std::cerr << "optimized_dense_plan: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
}  // namespace colmap
