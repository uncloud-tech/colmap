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
// Preserve absence separately from an explicit (possibly invalid) byte string.
// A notifier also works with the existing project-file option parser.
class FixedCohortOptionManager : public OptionManager {
 public:
  void AddOptionalBytes(const std::string& name,
                        std::string* value,
                        std::set<std::string>* supplied) {
    desc_->add_options()(
        name.c_str(),
        boost::program_options::value<std::string>()->notifier(
            [name, value, supplied](const std::string& text) {
              *value = text;
              supplied->insert(name);
            }),
        "Required manually; optional upper bound on observed FREE capacity");
    RegisterOption(name, value);
  }
};
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
    bool prepare_only = false, auto_active_owners = false;
    bool observe_capacities = false, auto_phase_active_owners = false;
    int active_owners_ceiling = 0;
    int max_load_image_size = -1, active_owners = 1, tile_threads = 32;
    int deadline_seconds = 0;
    uint64_t loader_scratch_host = 0;
    owned::ByteCharge capacity{};
    owned::FixedCohortBudget declarations;
    std::string output_policy = "final_geo_only";
    std::string photo_source_storage = "private";
    std::set<std::string> supplied_caps;
    FixedCohortOptionManager options;
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
        "NO solver execution or outputs; CPU-only unless observe_capacities=1");
    options.AddDefaultOption("max_load_image_size", &max_load_image_size);
    options.AddDefaultOption(
        "active_owners", &active_owners, "Positive manual count (default 1)");
    options.AddDefaultOption(
        "auto_active_owners",
        &auto_active_owners,
        "Select largest memory-admissible count, not throughput optimum");
    options.AddDefaultOption("auto_phase_active_owners",
                             &auto_phase_active_owners,
                             "Independent PHOTO/GEO memory-admissible counts; "
                             "requires auto_active_owners");
    options.AddDefaultOption(
        "observe_capacities",
        &observe_capacities,
        "Opt-in FREE host/cgroup and selected CUDA device snapshot for automatic "
        "owners; optional capacity caps only lower free; queries CUDA even in "
        "prepare_only, never solves there");
    options.AddDefaultOption("active_owners_ceiling",
                             &active_owners_ceiling,
                             "Automatic count ceiling; 0 means job count");
    options.AddDefaultOption("tile_threads", &tile_threads);
    options.AddDefaultOption(
        "output_policy", &output_policy, "final_geo_only or photo_and_geo");
    options.AddDefaultOption("photo_source_storage",
                             &photo_source_storage,
                             "private (default) or shared PHOTO grayscale union");
    options.AddRequiredOption("deadline_seconds", &deadline_seconds);
    // OptionManager supports strings, but not uint64_t. Parse byte quantities
    // losslessly after its normal required-option handling.
    std::map<std::string, std::pair<std::string, uint64_t*>> byte_options;
    const auto add_bytes = [&](const std::string& name,
                               uint64_t* target,
                               bool required = true) {
      auto& value = byte_options[name];
      value.second = target;
      if (required)
        options.AddRequiredOption(name, &value.first);
      else
        options.AddOptionalBytes(name, &value.first, &supplied_caps);
    };
    add_bytes("host_capacity_bytes", &capacity.host, false);
    add_bytes("device_capacity_bytes", &capacity.device, false);
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
    const auto cap_present = [&](const char* name) {
      return supplied_caps.count(name) != 0;
    };
    const bool host_cap = cap_present("host_capacity_bytes");
    const bool device_cap = cap_present("device_capacity_bytes");
    Require(observe_capacities || (host_cap && device_cap),
            "manual mode requires host_capacity_bytes and device_capacity_bytes");
    Require(!auto_phase_active_owners || auto_active_owners,
            "auto_phase_active_owners requires auto_active_owners");
    Require(!observe_capacities || auto_active_owners,
            "observe_capacities requires auto_active_owners");
    for (const auto& [name, value] : byte_options) {
      if ((name == "host_capacity_bytes" || name == "device_capacity_bytes") &&
          !cap_present(name.c_str()))
        continue;
      const auto* begin = value.first.data();
      const auto* end = begin + value.first.size();
      const auto parsed = std::from_chars(begin, end, *value.second);
      if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::invalid_argument("invalid unsigned byte quantity: " + name);
      }
    }
    Require(active_owners > 0, "active_owners must be positive");
    Require(active_owners_ceiling >= 0 &&
                (auto_active_owners || active_owners_ceiling == 0),
            "active_owners_ceiling requires auto_active_owners and must be "
            "nonnegative");
    Require(!auto_active_owners || active_owners == 1,
            "auto_active_owners conflicts with manual active_owners; use "
            "active_owners_ceiling");
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
    if (photo_source_storage == "private")
      declarations.photo_source_storage =
          owned::FixedCohortPhotoSourceStorage::kPrivate;
    else {
      Require(photo_source_storage == "shared",
              "invalid photo_source_storage: expected private or shared");
      declarations.photo_source_storage =
          owned::FixedCohortPhotoSourceStorage::kShared;
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
    auto budget =
        owned::BuildFixedCohortLogicalInventory(shapes, jobs, declarations);
    if (observe_capacities) {
      check_deadline();
      // A blocking CUDA initialization/query shares the original deadline;
      // never reset its budget or wait forever inside the driver.
      mvs::NativeRunDeadlineGuard observation_guard(deadline);
      const auto observed = owned::ObserveFixedCohortCapacities(
          owned::FixedCohortDevice(photo_options));
      capacity = owned::LimitFixedCohortAvailable(
          observed.available,
          host_cap ? std::optional<uint64_t>(capacity.host) : std::nullopt,
          device_cap ? std::optional<uint64_t>(capacity.device) : std::nullopt);
      check_deadline();
      std::cout << "observed_device=" << observed.device
                << "; device_name=" << observed.device_name
                << "; device_uuid=" << observed.device_uuid << '\n'
                << "observed_free_host_bytes=" << observed.available.host
                << "; observed_free_device_bytes=" << observed.available.device
                << "; observed_total_device_bytes=" << observed.device_total
                << "; effective_host_capacity_bytes=" << capacity.host
                << "; effective_device_capacity_bytes=" << capacity.device << '\n'
                << "unqualified logical FREE snapshot, racy and not reserved; "
                   "already-live process/context excluded from free; declared "
                   "process/unknown remain additional conservative charges, "
                   "no inferred live credit; may reject feasible work\n";
    }
    std::string count_reason = "manual";
    std::array<owned::FixedCohortActiveSelection, 2> phase_selection;
    if (auto_phase_active_owners) {
      phase_selection = owned::SelectFixedCohortPhaseActiveOwners(
          budget, capacity, active_owners_ceiling);
      budget.phase_active_owners = {phase_selection[0].count,
                                    phase_selection[1].count};
      budget.max_active_owners =
          std::max(phase_selection[0].count, phase_selection[1].count);
      count_reason = "phase-specific (see PHOTO/GEO)";
    } else if (auto_active_owners) {
      const auto selection = owned::SelectFixedCohortActiveOwners(
          budget, capacity, active_owners_ceiling);
      budget.max_active_owners = selection.count;
      count_reason = selection.reason;
    }
    // Selection/refusal precedes workspace decode, frozen pixels, banks and
    // private solver allocation. Never probe fit by attempting allocations.
    owned::AdmitFixedCohortBudget(budget, capacity);
    std::cout << "active_owners=" << budget.max_active_owners
              << "; selection=" << (auto_active_owners ? "automatic" : "manual")
              << "; limiting_reason=" << count_reason
              << "; memory admissibility only, not throughput optimum\n";
    if (auto_phase_active_owners) {
      for (size_t i = 0; i < 2; ++i)
        std::cout << (i ? "GEO" : "PHOTO")
                  << " active_owners=" << phase_selection[i].count
                  << "; limiting_reason=" << phase_selection[i].reason << '\n';
    }
    const auto dispatch = budget.max_active_owners == 1
                              ? owned::PairDispatch::kSerial
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
    std::cout << "photo_source_storage=" << photo_source_storage
              << "; logical PHOTO union host=" << budget.photo_shared.host
              << " device=" << budget.photo_shared.device << " bytes\n";
    std::cout << "experimental unqualified logical envelope; admitted host="
              << admitted.host << " device=" << admitted.device
              << " bytes (single 3% margin); no physical-fit guarantee\n";
    if (prepare_only) {
      if (observe_capacities)
        std::cout << "PREPARE ONLY: CUDA capacity query only; NO solver execution, "
                     "reconstruction, results or output files\n";
      else
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
