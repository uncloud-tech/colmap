// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/exe/fixed_cohort.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/mvs/model.h"
#include "colmap/mvs/owned_fixed_cohort_inventory.h"
#include "colmap/mvs/owned_fixed_cohort_validation.h"
#include "colmap/mvs/owned_fixed_cohort_workspace.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/string.h"

#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>

namespace colmap {
namespace {
namespace owned = mvs::owned;
namespace fs = std::filesystem;
void Require(bool valid, const char* message) {
  if (!valid) throw std::invalid_argument(message);
}
void CheckName(const std::string& name) {
  const fs::path path(name);
  Require(!path.empty() && !path.is_absolute() && !path.has_root_path(),
          "image names must be relative paths");
  for (const auto& component : path)
    Require(component != ".." && component != "." && !component.empty(),
            "unsafe image name for output");
}
using Config = std::vector<std::pair<std::string, std::vector<std::string>>>;
Config ReadConfig(const fs::path& path) {
  Config result;
  std::vector<std::string> lines;
  for (auto line : ReadTextFileLines(path)) {
    StringTrim(&line);
    if (!line.empty() && line[0] != '#') lines.push_back(std::move(line));
  }
  Require(!lines.empty() && lines.size() % 2 == 0,
          "config needs reference/explicit ordered source line pairs");
  std::set<std::string> seen;
  for (size_t i = 0; i < lines.size(); i += 2) {
    CheckName(lines[i]);
    Require(seen.insert(lines[i]).second, "duplicate config reference");
    auto sources = CSVToVector<std::string>(lines[i + 1]);
    Require(!sources.empty(), "empty sources");
    for (const auto& source : sources) {
      CheckName(source);
      Require(source != "__auto__" && source != "__all__",
              "only explicit ordered sources supported; no auto/all expansion");
    }
    result.emplace_back(lines[i], std::move(sources));
  }
  return result;
}
}  // namespace

int RunExperimentalFixedCohort(int argc, char** argv) {
  const auto entry = mvs::NativeRunDeadline::Clock::now();
  try {
    fs::path workspace_path, output_path, photo_config_path, geo_config_path;
    bool prepare_only = false;
    int max_load_image_size = -1, active_owners = 1, tile_threads = 32;
    int deadline_seconds = 0;
    uint64_t loader_scratch_host = 0;
    owned::ByteCharge capacity{};
    owned::FixedCohortBudget declarations;
    std::string output_policy = "final_geo_only";
    OptionManager options;
    options.AddRequiredOption("workspace_path", &workspace_path);
    options.AddRequiredOption(
        "output_path",
        &output_path,
        "Fresh isolated output root; unused in prepare_only");
    options.AddDefaultOption("photo_config_path", &photo_config_path);
    options.AddDefaultOption("geo_config_path", &geo_config_path);
    options.AddDefaultOption(
        "prepare_only",
        &prepare_only,
        "CPU preparation only: NO reconstruction or outputs");
    options.AddDefaultOption("max_load_image_size", &max_load_image_size);
    options.AddDefaultOption(
        "active_owners", &active_owners, "1 serial or 2 concurrent");
    options.AddDefaultOption("tile_threads", &tile_threads);
    options.AddDefaultOption(
        "output_policy", &output_policy, "final_geo_only or photo_and_geo");
    options.AddRequiredOption("deadline_seconds", &deadline_seconds);
    // OptionManager supports strings, but not uint64_t. Parse byte quantities
    // losslessly after its normal required-option handling.
    std::map<std::string, std::pair<std::string, uint64_t*>> byte_options;
    const auto add_bytes = [&](const std::string& name, uint64_t* target) {
      auto& value = byte_options[name];
      value.second = target;
      options.AddRequiredOption(name, &value.first);
    };
    add_bytes("host_capacity_bytes", &capacity.host);
    add_bytes("device_capacity_bytes", &capacity.device);
    add_bytes("loader_scratch_host_bytes", &loader_scratch_host);
    add_bytes("process_host_bytes", &declarations.process.host);
    add_bytes("process_device_bytes", &declarations.process.device);
    add_bytes("unknown_host_bytes", &declarations.unknown.host);
    add_bytes("unknown_device_bytes", &declarations.unknown.device);
    options.AddRequiredOption("unknown_provenance",
                              &declarations.unknown_provenance);
    add_bytes("reserve_host_bytes", &declarations.headroom_reserve.host);
    add_bytes("reserve_device_bytes", &declarations.headroom_reserve.device);
    options.AddPatchMatchStereoOptions();
    if (!options.Parse(argc, argv)) return EXIT_FAILURE;
    for (const auto& [name, value] : byte_options) {
      const auto* begin = value.first.data();
      const auto* end = begin + value.first.size();
      const auto parsed = std::from_chars(begin, end, *value.second);
      if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::invalid_argument("invalid unsigned byte quantity: " + name);
      }
    }
    Require(active_owners == 1 || active_owners == 2,
            "active_owners must be 1 or 2");
    Require(deadline_seconds > 0, "positive finite deadline_seconds required");
    const auto deadline = mvs::NativeRunDeadline::FromBudget(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::seconds(deadline_seconds)),
        entry);
    const auto check_deadline = [&] {
      Require(deadline.Remaining().count() > 0,
              "fixed-cohort command deadline expired");
    };
    check_deadline();
    declarations.max_active_owners = active_owners;
    if (output_policy == "final_geo_only")
      declarations.output_policy =
          owned::FixedCohortOutputPolicy::kFinalGeoOnly;
    else {
      Require(output_policy == "photo_and_geo", "invalid output_policy");
      declarations.output_policy = owned::FixedCohortOutputPolicy::kPhotoAndGeo;
    }
    auto photo_options = *options.patch_match_stereo;
    Require(photo_options.geom_consistency && photo_options.filter,
            "phase contract requires geom_consistency=1/filter=1: PHOTO is "
            "unfiltered, GEO filtered");
    auto geo_options = photo_options;
    photo_options.geom_consistency = false;
    photo_options.filter = false;
    // The native validator requires explicit positive sigma, depth range,
    // thread count and one device; no default sigma/depth inference here.
    if (photo_config_path.empty())
      photo_config_path = workspace_path / "stereo/patch-match.cfg";
    if (geo_config_path.empty()) geo_config_path = photo_config_path;
    const auto photo_config = ReadConfig(photo_config_path);
    const auto geo_config = ReadConfig(geo_config_path);
    Require(photo_config.size() == geo_config.size(),
            "PHOTO/GEO reference set differs");
    std::map<std::string, std::vector<std::string>> geo_sources;
    for (const auto& [name, sources] : geo_config)
      geo_sources.emplace(name, sources);
    mvs::Model model;
    model.Read(workspace_path, "COLMAP");
    std::vector<owned::FixedCohortImageShape> shapes;
    for (auto image : model.images) {
      if (max_load_image_size > 0)
        image.Downsize(max_load_image_size, max_load_image_size);
      shapes.push_back({image.GetWidth(), image.GetHeight()});
    }
    std::vector<owned::FixedCohortWorkspaceReference> named;
    std::vector<owned::FixedCohortJobShape> jobs;
    for (const auto& [name, sources] : photo_config) {
      Require(geo_sources.count(name), "PHOTO/GEO reference set differs");
      named.push_back(
          {name, sources, geo_sources.at(name), photo_options, geo_options});
      owned::FixedCohortReference ref;
      ref.reference_image_idx = model.GetImageIdx(name);
      for (const auto& source : sources)
        ref.photo_sources.push_back(model.GetImageIdx(source));
      for (const auto& source : geo_sources.at(name))
        ref.geo_sources.push_back(model.GetImageIdx(source));
      ref.photo_options = photo_options;
      ref.geo_options = geo_options;
      jobs.push_back(
          {ref.reference_image_idx, ref.photo_sources, ref.geo_sources});
    }
    model = mvs::Model();  // Release the extra sparse model before decoding.
    const auto budget =
        owned::BuildFixedCohortLogicalInventory(shapes, jobs, declarations);
    const auto dispatch = active_owners == 1 ? owned::PairDispatch::kSerial
                                             : owned::PairDispatch::kPair;
    // Preparation validates options, closed GEO producers, loader peak and
    // real host/cgroup headroom before native GPU execution can begin.
    auto prepared = owned::PrepareFixedCohortWorkspace(workspace_path,
                                                       max_load_image_size,
                                                       named,
                                                       budget,
                                                       capacity,
                                                       loader_scratch_host,
                                                       deadline,
                                                       dispatch,
                                                       tile_threads);
    check_deadline();
    const auto admitted = owned::AdmitFixedCohortBudget(budget, capacity);
    std::cout << "experimental unqualified logical envelope; admitted host="
              << admitted.host << " device=" << admitted.device
              << " bytes (single 3% margin); no physical-fit guarantee\n";
    if (prepare_only) {
      std::cout << "PREPARE ONLY: NO GPU execution, reconstruction, results or "
                   "output files\n";
      return EXIT_SUCCESS;
    }
#if !defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
    throw std::runtime_error(
        "experimental_fixed_cohort execution requires native CUDA; no stock "
        "fallback");
#else
    output_path = fs::absolute(output_path).lexically_normal();
    Require(output_path.filename() != "." && output_path.filename() != ".." &&
                !output_path.filename().empty(),
            "invalid output root");
    Require(
        fs::canonical(output_path.parent_path()) == output_path.parent_path(),
        "output parent must exist without symlink components");
    // Atomic no-clobber ownership: never remove/reuse a preexisting path.
    Require(fs::create_directory(output_path), "fresh output_path required");
    const auto result = prepared.Run();
    for (size_t i = 0; i < named.size(); ++i) {
      const auto write_maps = [&](const owned::FixedCohortResult::Maps& maps,
                                  const char* suffix) {
        for (const bool normal : {false, true}) {
          check_deadline();
          const auto path = output_path / "stereo" /
                            (normal ? "normal_maps" : "depth_maps") /
                            (named[i].reference + suffix);
          fs::create_directories(path.parent_path());
          Require(
              !fs::exists(path) && !fs::is_symlink(fs::symlink_status(path)),
              "output map already exists");
          const mvs::Mat<float>& map =
              normal ? static_cast<const mvs::Mat<float>&>(maps.normal)
                     : static_cast<const mvs::Mat<float>&>(maps.depth);
          map.Write(path);
          // The existing writer checks open, but not close; detect short
          // writes rather than claiming success on a full output filesystem.
          const auto header = std::to_string(map.GetWidth()) + "&" +
                              std::to_string(map.GetHeight()) + "&" +
                              std::to_string(map.GetDepth()) + "&";
          Require(fs::file_size(path) ==
                      owned::CheckedAdd(header.size(), map.GetNumBytes()),
                  "incomplete map output");
          check_deadline();
        }
      };
      if (result.output_policy == owned::FixedCohortOutputPolicy::kPhotoAndGeo)
        write_maps(result.photo.at(i), ".photometric.bin");
      write_maps(result.geo.at(i), ".geometric.bin");
    }
    std::cout
        << "Completed experimental PHOTO->GEO; standard depth/normal maps: "
        << output_path << '\n';
    return EXIT_SUCCESS;
#endif
  } catch (const std::exception& error) {
    std::cerr << "experimental_fixed_cohort: " << error.what()
              << "; no success claimed; any fresh partial output is retained\n";
    return EXIT_FAILURE;
  }
}
}  // namespace colmap
