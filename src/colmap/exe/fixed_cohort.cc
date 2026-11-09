// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/exe/fixed_cohort.h"

#include "colmap/controllers/option_manager.h"
#include "colmap/exe/multi_gpu_plan_json.h"
#include "colmap/exe/optimized_dense_plan_options.h"
#include "colmap/mvs/generated/resident_allocation_fingerprint.h"
#include "colmap/mvs/model.h"
#include "colmap/mvs/owned_fixed_cohort_inventory.h"
#include "colmap/mvs/owned_fixed_cohort_validation.h"
#include "colmap/mvs/owned_fixed_cohort_workspace.h"
#include "colmap/mvs/owned_tile.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/string.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace colmap {
namespace {
namespace owned = mvs::owned;
namespace fs = std::filesystem;
void Require(bool valid, const char* message) {
  if (!valid) throw std::invalid_argument(message);
}
// Lock-free atomic stores are async-signal-safe. Keep the token process-static
// so a signal can never address an object being destroyed by stack unwinding.
std::atomic<bool> multi_gpu_cancelled{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void CancelMultiGpu(int) { multi_gpu_cancelled.store(true); }
class MultiGpuSignalGuard {
 public:
  MultiGpuSignalGuard() {
    multi_gpu_cancelled.store(false);
    old_int_ = std::signal(SIGINT, CancelMultiGpu);
    if (old_int_ == SIG_ERR)
      throw std::runtime_error("cannot install multi-GPU SIGINT handler");
    old_term_ = std::signal(SIGTERM, CancelMultiGpu);
    if (old_term_ == SIG_ERR) {
      std::signal(SIGINT, old_int_);
      throw std::runtime_error("cannot install multi-GPU SIGTERM handler");
    }
  }
  ~MultiGpuSignalGuard() {
    std::signal(SIGTERM, old_term_);
    std::signal(SIGINT, old_int_);
  }

 private:
  using Handler = void (*)(int);
  Handler old_int_, old_term_;
};
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

std::vector<owned::FixedCohortWorkspaceReference>
ResolveFixedCohortWorkspacePlan(const std::vector<std::string>& names,
                                const std::vector<owned::BankJob>& photo,
                                const std::vector<owned::BankJob>& geo) {
  Require(photo.size() == geo.size(), "native PHOTO/GEO graph differs");
  std::vector<owned::FixedCohortWorkspaceReference> named;
  for (size_t i = 0; i < photo.size(); ++i) {
    Require(photo[i].reference_image_idx == geo[i].reference_image_idx,
            "native PHOTO/GEO reference differs");
    owned::FixedCohortWorkspaceReference ref;
    ref.reference = names.at(photo[i].reference_image_idx);
    for (int source : photo[i].source_image_idxs)
      ref.photo_sources.push_back(names.at(source));
    for (int source : geo[i].source_image_idxs)
      ref.geo_sources.push_back(names.at(source));
    ref.photo_options = photo[i].options;
    ref.geo_options = geo[i].options;
    named.push_back(std::move(ref));
  }
  return named;
}

int RunExperimentalFixedCohort(int argc, char** argv) {
  const auto entry = mvs::NativeRunDeadline::Clock::now();
  try {
    fs::path workspace_path, output_path, photo_config_path, geo_config_path;
    fs::path report_path, approved_multi_gpu_plan;
    std::string stereo_devices;
    std::optional<owned::MultiGpuFixedCohortPlan> multi_gpu_plan;
    std::optional<dense_plan::Json> multi_gpu_placement;
    std::optional<MultiGpuSignalGuard> multi_gpu_signals;
    bool resolve_workspace_plan = false;
    bool automatic_stereo_storage = false;
    bool prepare_only = false, auto_active_owners = false;
    bool observe_capacities = false, auto_phase_active_owners = false;
    int active_owners_ceiling = 0;
    int max_load_image_size = -1, active_owners = 1, tile_threads = 32;
    int deadline_seconds = 0;
    uint64_t loader_scratch_host = 0, stereo_disk_capacity = 0;
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
    options.AddDefaultOption(
        "resolve_workspace_plan",
        &resolve_workspace_plan,
        "Resolve native per-reference depths/options and effective ordered "
        "sources; no global depth inference");
    options.AddDefaultOption(
        "report_path",
        &report_path,
        "Optional fresh typed execution report; multi-device prepare_only "
        "writes the approval plan");
    options.AddDefaultOption(
        "stereo_devices",
        &stereo_devices,
        "Explicit approved ordinal:UUID:capacity-bytes CSV; at least two "
        "devices; host-map PHOTO/GEO exchange");
    options.AddDefaultOption(
        "automatic_stereo_storage",
        &automatic_stereo_storage,
        "Native snapshot-preferred one-device admission with HOST strip "
        "fallback; requires one approved stereo_devices entry");
    options.AddDefaultOption("approved_multi_gpu_plan",
                             &approved_multi_gpu_plan,
                             "Required exact plan JSON from multi-device "
                             "prepare_only before execution");
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
        "Opt-in FREE host/cgroup and selected CUDA device snapshot for "
        "automatic "
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
                             "private (default), shared PHOTO grayscale union, "
                             "or auto selection (requires auto_active_owners)");
    options.AddRequiredOption("deadline_seconds", &deadline_seconds);
    // OptionManager supports strings, but not uint64_t. Parse byte quantities
    // losslessly after its normal required-option handling.
    std::map<std::string, std::pair<std::string, uint64_t*>> byte_options;
    const auto add_bytes =
        [&](const std::string& name, uint64_t* target, bool required = true) {
          auto& value = byte_options[name];
          value.second = target;
          if (required)
            options.AddRequiredOption(name, &value.first);
          else
            options.AddOptionalBytes(name, &value.first, &supplied_caps);
        };
    add_bytes("host_capacity_bytes", &capacity.host, false);
    add_bytes("device_capacity_bytes", &capacity.device, false);
    add_bytes("stereo_disk_capacity_bytes", &stereo_disk_capacity, false);
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
    if (!stereo_devices.empty()) multi_gpu_signals.emplace();
    Require(!automatic_stereo_storage || !stereo_devices.empty(),
            "automatic stereo storage requires one approved device");
    const auto cap_present = [&](const char* name) {
      return supplied_caps.count(name) != 0;
    };
    const bool host_cap = cap_present("host_capacity_bytes");
    const bool device_cap = cap_present("device_capacity_bytes");
    Require(
        observe_capacities || (host_cap && device_cap),
        "manual mode requires host_capacity_bytes and device_capacity_bytes");
    Require(!auto_phase_active_owners || auto_active_owners,
            "auto_phase_active_owners requires auto_active_owners");
    Require(!observe_capacities || auto_active_owners,
            "observe_capacities requires auto_active_owners");
    for (const auto& [name, value] : byte_options) {
      if ((name == "host_capacity_bytes" || name == "device_capacity_bytes" ||
           name == "stereo_disk_capacity_bytes") &&
          !cap_present(name.c_str()))
        continue;
      const auto* begin = value.first.data();
      const auto* end = begin + value.first.size();
      const auto parsed = std::from_chars(begin, end, *value.second);
      if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::invalid_argument("invalid unsigned byte quantity: " + name);
      }
    }
    Require(
        !automatic_stereo_storage || cap_present("stereo_disk_capacity_bytes"),
        "automatic stereo storage requires frozen stereo_disk_capacity_bytes");
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
      Require(!multi_gpu_signals || !multi_gpu_cancelled.load(),
              "multi-GPU command cancelled");
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
    const bool auto_storage = photo_source_storage == "auto";
    Require(!auto_storage || auto_active_owners,
            "auto storage requires auto_active_owners");
    if (photo_source_storage == "private" || auto_storage)
      declarations.photo_source_storage =
          owned::FixedCohortPhotoSourceStorage::kPrivate;
    else {
      Require(photo_source_storage == "shared",
              "invalid photo_source_storage: expected private, shared or auto");
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
    // The original explicit-config mode retains its existing invocation-global
    // options. The opt-in workspace adapter instead consumes native typed jobs;
    // NEVER infer a global depth envelope or re-resolve serialized sources.
    Require(!resolve_workspace_plan ||
                (photo_config_path.empty() && geo_config_path.empty() &&
                 max_load_image_size == -1),
            "workspace plan conflicts with explicit configs/load resize");
    Require(report_path.empty() ||
                (!fs::exists(report_path) &&
                 !fs::is_symlink(fs::symlink_status(report_path))),
            "fresh report_path required");
    if (!prepare_only) {
      // Refuse existing output roots before metadata/pixel preparation; retain
      // the later atomic create_directory as the actual ownership boundary.
      Require(!fs::exists(output_path) &&
                  !fs::is_symlink(fs::symlink_status(output_path)),
              "fresh output_path required");
    }
    std::vector<owned::FixedCohortWorkspaceReference> named;
    if (resolve_workspace_plan) {
#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
      mvs::NativeRunDeadlineGuard resolution_guard(deadline);
      mvs::PatchMatchController controller(
          geo_options, workspace_path.string(), "COLMAP", "");
      const auto graph = controller.ResolveDensePlanGraph();
      named =
          ResolveFixedCohortWorkspacePlan(graph.names, graph.photo, graph.geo);
#else
      throw std::runtime_error(
          "workspace plan resolution requires CUDA or HIP stereo support; "
          "no stock fallback");
#endif
    } else {
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
      for (const auto& [name, sources] : photo_config) {
        Require(geo_sources.count(name), "PHOTO/GEO reference set differs");
        named.push_back(
            {name, sources, geo_sources.at(name), photo_options, geo_options});
      }
    }
    if (auto_storage || automatic_stereo_storage) {
      Require(max_load_image_size == -1 || max_load_image_size > 0,
              "fixed-cohort invalid load image size");
      owned::ValidateOwnedTileThreads(tile_threads);
      owned::ValidateFixedCohortWorkspaceCameras(workspace_path);
    }
    mvs::Model model;
    model.Read(workspace_path, "COLMAP");
    std::vector<owned::FixedCohortImageShape> shapes;
    for (auto& image : model.images) {
      if (max_load_image_size > 0)
        image.Downsize(max_load_image_size, max_load_image_size);
      shapes.push_back({image.GetWidth(), image.GetHeight()});
    }
    std::vector<owned::FixedCohortJobShape> jobs;
    std::vector<owned::FixedCohortReference> references;
    for (const auto& spec : named) {
      CheckName(spec.reference);
      owned::FixedCohortReference ref;
      ref.reference_image_idx = model.GetImageIdx(spec.reference);
      for (const auto& source : spec.photo_sources) {
        CheckName(source);
        ref.photo_sources.push_back(model.GetImageIdx(source));
      }
      for (const auto& source : spec.geo_sources) {
        CheckName(source);
        ref.geo_sources.push_back(model.GetImageIdx(source));
      }
      ref.photo_options = spec.photo_options;
      ref.geo_options = spec.geo_options;
      jobs.push_back(
          {ref.reference_image_idx, ref.photo_sources, ref.geo_sources});
      references.push_back(std::move(ref));
    }
    auto budget =
        owned::BuildFixedCohortLogicalInventory(shapes, jobs, declarations);
    // Common invocation validation must not become a capacity refusal or a
    // reason to try another storage mode. This checks metadata only.
    if (auto_storage || !stereo_devices.empty())
      owned::ValidateFixedCohortShape(model.images, references, budget);
    if (!stereo_devices.empty()) {
      Require((observe_capacities || automatic_stereo_storage) &&
                  auto_active_owners && host_cap && max_load_image_size == -1,
              "multi-GPU requires explicit host cap, observed capacities, "
              "automatic owners and full resolution");
      Require(!report_path.empty(), "multi-GPU requires report_path");
      Require(automatic_stereo_storage || auto_storage ||
                  photo_source_storage == "shared",
              "multi-GPU PHOTO requires shared or auto source storage");
      std::vector<owned::PhasePlanDevice> devices;
      for (const auto& token : CSVToVector<std::string>(stereo_devices)) {
        const auto first = token.find(':');
        const auto second =
            token.find(':', first == std::string::npos ? 0 : first + 1);
        Require(first != std::string::npos && second != std::string::npos &&
                    token.find(':', second + 1) == std::string::npos,
                "invalid approved stereo device token");
        const auto ordinal = owned::ParseBytes(token.substr(0, first));
        Require(
            ordinal <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
            "invalid approved stereo ordinal");
        const auto uuid = token.substr(first + 1, second - first - 1);
        Require(
            uuid.size() == 32 &&
                uuid.find_first_not_of("0123456789abcdef") == std::string::npos,
            "invalid approved stereo UUID");
        owned::PhasePlanDevice device;
        device.ordinal = static_cast<int>(ordinal);
        device.uuid = uuid;
        device.capacity_bytes = owned::ParseBytes(token.substr(second + 1));
        devices.push_back(std::move(device));
      }
      if (automatic_stereo_storage) {
        Require(
            devices.size() == 1,
            "automatic stereo storage requires exactly one approved device");
        if (device_cap)
          devices[0].capacity_bytes =
              std::min(devices[0].capacity_bytes, capacity.device);
        capacity.device = devices[0].capacity_bytes;
        multi_gpu_plan = owned::PlanSingleDeviceStereoFixedCohort(
            shapes,
            references,
            declarations,
            devices[0],
            capacity.host,
            stereo_disk_capacity,
            fs::canonical(workspace_path),
            photo_source_storage,
            active_owners_ceiling ? active_owners_ceiling : references.size(),
            loader_scratch_host,
            tile_threads);
        if (multi_gpu_plan->snapshot_budget)
          budget = *multi_gpu_plan->snapshot_budget;
      } else {
        multi_gpu_plan = owned::PlanMultiGpuFixedCohort(
            shapes,
            references,
            declarations,
            devices,
            capacity.host,
            active_owners_ceiling ? active_owners_ceiling : references.size(),
            loader_scratch_host);
      }
      owned::ValidateMultiGpuFixedCohortPlan(
          model.images, references, budget, *multi_gpu_plan);
      check_deadline();
      multi_gpu_placement =
          dense_plan::MultiGpuPlanJson(*multi_gpu_plan,
                                       references,
                                       model.images,
                                       mvs::kResidentAllocationFingerprint);
      const auto placement = multi_gpu_placement->Dump() + "\n";
      std::cout << "Approved-device multi-GPU phase placement (before payload "
                   "allocations): "
                << placement;
      if (prepare_only) {
        // Manual one-device preparation is metadata-only like the original
        // CPU prepare interface. Execution ALWAYS verifies UUID/FREE/disk.
        if (observe_capacities) {
          mvs::NativeRunDeadlineGuard observation_guard(deadline);
          owned::CheckMultiGpuFixedCohortAvailable(*multi_gpu_plan);
        }
        dense_plan::WriteNewAtomic(report_path, placement);
        return EXIT_SUCCESS;
      }
      Require(!approved_multi_gpu_plan.empty(),
              "multi-GPU execution requires approved phase plan");
      std::ifstream approved(approved_multi_gpu_plan, std::ios::binary);
      Require(approved.is_open(), "cannot read approved multi-GPU plan");
      std::string approved_bytes((std::istreambuf_iterator<char>(approved)),
                                 std::istreambuf_iterator<char>());
      Require(approved_bytes == placement,
              "multi-GPU placement/scientific inputs changed since approval");
      {
        mvs::NativeRunDeadlineGuard observation_guard(deadline);
        owned::CheckMultiGpuFixedCohortAvailable(*multi_gpu_plan);
      }
    } else {
      Require(approved_multi_gpu_plan.empty(),
              "approved multi-GPU plan requires stereo_devices");
    }
    model = mvs::Model();  // Release the extra sparse model before decoding.
    if (observe_capacities && !multi_gpu_plan) {
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
                << "; effective_device_capacity_bytes=" << capacity.device
                << '\n'
                << "unqualified logical FREE snapshot, racy and not reserved; "
                   "already-live process/context excluded from free; declared "
                   "process/unknown remain additional conservative charges, "
                   "no inferred live credit; may reject feasible work\n";
    }
    std::string count_reason = "manual";
    std::array<owned::FixedCohortActiveSelection, 2> phase_selection;
    std::string storage_reason = "explicit storage";
    if (auto_storage && !multi_gpu_plan) {
      auto selected =
          owned::SelectFixedCohortPhotoSourceStorage(shapes,
                                                     jobs,
                                                     declarations,
                                                     capacity,
                                                     active_owners_ceiling,
                                                     auto_phase_active_owners);
      budget = std::move(selected.budget);
      storage_reason = std::move(selected.reason);
    }
    if (multi_gpu_plan) {
      if (!multi_gpu_plan->snapshot_budget) budget.max_active_owners = 1;
      count_reason =
          multi_gpu_plan->snapshot_budget
              ? "complete resident snapshot preferred"
              : "per-device phase plan; immutable host PHOTO exchange";
      storage_reason =
          multi_gpu_plan->snapshot_budget ? "complete snapshot envelope fits"
          : multi_gpu_plan->single_device_host
              ? "snapshot no-fit; native per-reference private/strip HOST "
                "exchange"
              : "admitted per-cohort PHOTO unions; private host-map GEO";
    } else if (auto_phase_active_owners) {
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
    const char* effective_storage =
        multi_gpu_plan && multi_gpu_plan->single_device_host ? "private"
        : multi_gpu_plan && !multi_gpu_plan->snapshot_budget
            ? "shared_per_cohort"
        : budget.photo_source_storage ==
                owned::FixedCohortPhotoSourceStorage::kShared
            ? "shared"
            : "private";
    if (auto_storage && !multi_gpu_plan)
      std::cout << "requested_photo_source_storage=auto; "
                   "effective_photo_source_storage="
                << effective_storage << "; storage_reason=" << storage_reason
                << '\n'
                << "PHOTO active_owners="
                << owned::FixedCohortPhaseActiveOwners(budget, false)
                << "; GEO active_owners="
                << owned::FixedCohortPhaseActiveOwners(budget, true) << '\n';
    // Selection/refusal precedes workspace decode, frozen pixels, banks and
    // private solver allocation. Never probe fit by attempting allocations.
    if (!multi_gpu_plan) owned::AdmitFixedCohortBudget(budget, capacity);
    if (multi_gpu_plan) {
      std::cout << "multi-GPU owner limits: see approved per-device/per-phase "
                   "cohorts; "
                   "no pooled VRAM or measured overlap claim\n";
    } else {
      std::cout << "active_owners=" << budget.max_active_owners
                << "; selection="
                << (auto_active_owners ? "automatic" : "manual")
                << "; limiting_reason=" << count_reason
                << "; memory admissibility only, not throughput optimum\n";
    }
    if (auto_phase_active_owners && !multi_gpu_plan) {
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
    auto prepared = owned::PrepareFixedCohortWorkspace(
        workspace_path,
        max_load_image_size,
        named,
        budget,
        capacity,
        loader_scratch_host,
        deadline,
        dispatch,
        tile_threads,
        multi_gpu_plan ? &*multi_gpu_plan : nullptr);
    check_deadline();
    if (multi_gpu_plan) {
      const auto prepared_binding =
          dense_plan::MultiGpuPlanJson(*multi_gpu_plan,
                                       prepared.References(),
                                       prepared.Images(),
                                       mvs::kResidentAllocationFingerprint);
      Require(
          prepared_binding.Dump() == multi_gpu_placement->Dump(),
          "multi-GPU image identity/calibration changed during preparation");
    }
    const auto admitted =
        multi_gpu_plan ? owned::ByteCharge{owned::MarginBytes(
                                               multi_gpu_plan->host_peak_raw),
                                           0}
                       : owned::AdmitFixedCohortBudget(budget, capacity);
    if (multi_gpu_plan) {
      std::cout
          << "experimental multi-GPU aggregate admitted host=" << admitted.host
          << " bytes (single 3% margin); per-device VRAM and PHOTO unions "
             "are reported separately in the approved plan; no physical-fit "
             "guarantee\n";
    } else {
      std::cout << "photo_source_storage=" << effective_storage
                << "; logical PHOTO union host=" << budget.photo_shared.host
                << " device=" << budget.photo_shared.device << " bytes\n";
      std::cout << "experimental unqualified logical envelope; admitted host="
                << admitted.host << " device=" << admitted.device
                << " bytes (single 3% margin); no physical-fit guarantee\n";
    }
    if (prepare_only) {
      if (observe_capacities)
        std::cout
            << "PREPARE ONLY: CUDA capacity query only; NO solver execution, "
               "reconstruction, results or output files\n";
      else
        std::cout
            << "PREPARE ONLY: NO GPU execution, reconstruction, results or "
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
    const auto result =
        prepared.Run(multi_gpu_signals ? &multi_gpu_cancelled : nullptr);
    if (auto_storage && !multi_gpu_plan)
      std::cout << "executed PHOTO owner_limit=" << result.phase_owner_limits[0]
                << "; GEO owner_limit=" << result.phase_owner_limits[1]
                << "; actual PHOTO union host="
                << result.actual_photo_shared.host
                << " device=" << result.actual_photo_shared.device
                << " bytes; coordinator limits, not observed overlap\n";
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
    if (!report_path.empty()) {
      using dense_plan::Json;
      std::vector<std::pair<std::string, Json>> phases;
      for (const bool geo : {false, true}) {
        std::vector<Json> records;
        for (size_t job = 0; job < references.size(); ++job) {
          const auto& ref = references[job];
          auto executed_options = geo ? ref.geo_options : ref.photo_options;
          // Snapshot execution consumes these original bound options. Only
          // HOST workers perform an explicit per-job placement rewrite.
          if (multi_gpu_plan && !multi_gpu_plan->snapshot_budget) {
            for (const auto& group : multi_gpu_plan->phases[geo].groups)
              for (const auto& cohort : group.cohorts)
                if (std::find(cohort.jobs.begin(), cohort.jobs.end(), job) !=
                    cohort.jobs.end())
                  executed_options.gpu_index =
                      std::to_string(group.device.ordinal);
          }
          std::vector<Json> sources;
          for (int idx : geo ? ref.geo_sources : ref.photo_sources)
            sources.emplace_back(idx);
          records.push_back(Json::Object(
              {{"reference", Json(ref.reference_image_idx)},
               {"sources", Json::Array(sources)},
               {"options", dense_plan::OptionsJson(executed_options)}}));
        }
        phases.emplace_back(geo ? "geo" : "photo", Json::Array(records));
      }
      dense_plan::WriteNewAtomic(
          report_path,
          Json::Object(
              {{"schema_version", Json(1)},
               {"status", Json("completed")},
               {"output_policy",
                Json(owned::FixedCohortOutputPolicyName(result.output_policy))},
               {"requested_backend", Json("experimental_fixed_cohort")},
               {"effective_backend", Json("experimental_fixed_cohort")},
               {"allocation_fingerprint",
                Json(std::string(mvs::kResidentAllocationFingerprint))},
               {"requested_storage", Json(photo_source_storage)},
               {"storage_reason", Json(storage_reason)},
               {"execution_kind",
                Json(multi_gpu_plan && multi_gpu_plan->snapshot_budget
                         ? "resident_snapshot"
                     : multi_gpu_plan && multi_gpu_plan->single_device_host
                         ? "single_device_host_exchange"
                     : multi_gpu_plan ? "multi_gpu_host_exchange"
                                      : "resident_snapshot")},
               {"selection", Json(auto_active_owners ? "automatic" : "manual")},
               {"host_capacity_bytes", Json(std::to_string(capacity.host))},
               {"device_capacity_bytes",
                multi_gpu_plan && !automatic_stereo_storage
                    ? Json()
                    : Json(std::to_string(capacity.device))},
               {"effective_storage", Json(effective_storage)},
               {"photo_active_owners",
                multi_gpu_plan && !automatic_stereo_storage
                    ? Json()
                    : Json(static_cast<int>(result.phase_owner_limits[0]))},
               {"geo_active_owners",
                multi_gpu_plan && !automatic_stereo_storage
                    ? Json()
                    : Json(static_cast<int>(result.phase_owner_limits[1]))},
               {"multi_gpu_plan",
                multi_gpu_placement ? *multi_gpu_placement : Json()},
               {"phases", Json::Object(phases)}})
                  .Dump() +
              "\n");
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
