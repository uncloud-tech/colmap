// SPDX-License-Identifier: BSD-3-Clause
// Optional bounded rolling geometry feed; default finite epochs unchanged.
#include "colmap/controllers/option_manager.h"
#include "colmap/mvs/owned_cohort.h"
#include "colmap/mvs/owned_host_lookahead.h"
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/owned_normal_lookahead.h"
#include "colmap/mvs/owned_pair_events.h"
#include "colmap/mvs/owned_pair_image.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/owned_resident_minima.h"
#include "colmap/mvs/owned_tile.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/workspace.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/string.h"

#include <algorithm>
#include <array>
#include <climits>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif

namespace colmap {
namespace {
namespace pair = mvs::owned;
namespace fs = std::filesystem;

struct Globals {
  int seed = 0;
  std::string target = "stderr_and_file";
  std::string path;
  int level = 0;
  int severity = 0;
  bool color = true;
  bool operator==(const Globals& other) const {
    return std::tie(seed, target, path, level, severity, color) ==
           std::tie(other.seed,
                    other.target,
                    other.path,
                    other.level,
                    other.severity,
                    other.color);
  }
};

// Drop the global-pointer registrations before reading either INI. All base
// constructors run on the main thread before any file is parsed. Job Read()
// therefore changes ONLY job-local storage, not seed/log process globals.
struct JobOptions : OptionManager {
  fs::path workspace;
  fs::path config;
  std::string format = "COLMAP";
  std::string pmvs = "option-all";
  Globals globals;
  std::string reference_name;
  std::array<fs::path, 2> outputs;
  std::array<fs::path, 2> output_dirs;
  explicit JobOptions() : OptionManager(false) {
    BaseOptionManager::Reset(false);
    AddRequiredOption("workspace_path", &workspace);
    AddDefaultOption("workspace_format", &format);
    AddDefaultOption("pmvs_option_name", &pmvs);
    AddDefaultOption("config_path", &config);
    AddDefaultOption("default_random_seed", &globals.seed);
    AddDefaultOption("log_target", &globals.target);
    AddDefaultOption("log_path", &globals.path);
    AddDefaultOption("log_level", &globals.level);
    AddDefaultOption("log_severity", &globals.severity);
    AddDefaultOption("log_color", &globals.color);
    AddPatchMatchStereoOptions();
  }
};

uint64_t AffinityCount() {
#if defined(__linux__)
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
    throw std::runtime_error("cannot establish CPU affinity");
  }
  return CPU_COUNT(&mask);
#else
  throw std::runtime_error("owned pair is currently Linux-only");
#endif
}

void ValidateJob(JobOptions& job,
                 pair::ByteCharge charge,
                 bool resident = false) {
  auto& o = *job.patch_match_stereo;
  StringToLower(&job.format);
  if (job.format != "colmap" || (o.geom_consistency && !o.filter) ||
      o.window_radius != 5 || o.window_step != 1 || o.num_iterations != 5 ||
      o.max_image_size != -1 || o.allow_missing_files ||
      o.write_consistency_graph) {
    throw std::invalid_argument(
        "TWO prototype requires native COLMAP 11/1, five iterations, complete "
        "inputs, no graph; geometry requires filter=1");
  }
  const uint64_t device = pair::ParseBytes(o.gpu_index);
  if (device > INT_MAX) throw std::invalid_argument("device index overflow");
  if (!resident && pair::CacheBytes(o.cache_size) > charge.host) {
    throw std::invalid_argument(
        "cache cap exceeds entire job host reservation");
  }
  job.workspace = fs::canonical(job.workspace);
  if (!fs::is_directory(job.workspace / "images") ||
      !fs::is_directory(job.workspace / "sparse")) {
    throw std::invalid_argument("workspace inputs missing");
  }
  bool complete_sparse = false;
  for (const auto* extension : {".bin", ".txt"}) {
    bool complete = true;
    for (const auto* stem : {"cameras", "images", "points3D"}) {
      complete =
          complete && fs::is_regular_file(job.workspace / "sparse" /
                                          (std::string(stem) + extension));
    }
    complete_sparse = complete_sparse || complete;
  }
  if (!complete_sparse)
    throw std::invalid_argument("sparse model inputs missing");
  job.config = fs::canonical(job.config.empty()
                                 ? job.workspace / "stereo/patch-match.cfg"
                                 : job.config);
  std::vector<std::string> lines;
  for (auto line : ReadTextFileLines(job.config)) {
    StringTrim(&line);
    if (!line.empty() && line[0] != '#') lines.push_back(line);
  }
  if (lines.size() != 2)
    throw std::invalid_argument("exactly ONE reference per job required");
  job.reference_name = lines[0];
  fs::path reference(job.reference_name);
  if (reference.empty() || reference.is_absolute())
    throw std::invalid_argument("invalid reference path");
  for (const auto& component : reference) {
    if (component == ".." || component == ".")
      throw std::invalid_argument("unsafe reference path");
  }
  if (!fs::is_regular_file(job.workspace / "images" / reference)) {
    throw std::invalid_argument("reference image missing");
  }
  const auto stereo = fs::weakly_canonical(job.workspace / "stereo");
  if (!pair::PathContains(job.workspace, stereo) || stereo == job.workspace) {
    throw std::invalid_argument("stereo output root escapes workspace");
  }
  for (size_t k = 0; k < 2; ++k) {
    const auto dir = fs::weakly_canonical(
        job.workspace / "stereo" / (k == 0 ? "depth_maps" : "normal_maps"));
    job.output_dirs[k] = dir;
    if (!pair::PathContains(stereo, dir) || dir == stereo) {
      throw std::invalid_argument(
          "output directory escapes workspace stereo root");
    }
    const auto output =
        dir / (job.reference_name +
               (o.geom_consistency ? ".geometric.bin" : ".photometric.bin"));
    if (fs::exists(output) || fs::is_symlink(fs::symlink_status(output))) {
      throw std::invalid_argument(
          "fresh outputs required (including partial results/symlinks)");
    }
    job.outputs[k] = fs::weakly_canonical(output);
    if (!pair::PathContains(dir, job.outputs[k]))
      throw std::invalid_argument("output path escapes map directory");
  }
  if (pair::PathsAlias(job.output_dirs[0], job.output_dirs[1])) {
    throw std::invalid_argument("depth and normal output directories alias");
  }
}

// Validate seed dimensions before parallel cache access: oversized maps could
// otherwise evict a cache entry while another decoder copies its bitmap/map.
// Native geometry retains one float depth and three float normal channels.
void CheckSeedMap(const fs::path& path,
                  uint64_t width,
                  uint64_t height,
                  uint64_t channels) {
  std::ifstream input(path, std::ios::binary);
  std::array<uint64_t, 3> dimensions{};
  for (auto& dimension : dimensions) {
    std::string field;
    char c = 0;
    while (input.get(c) && c != '&') {
      if (field.size() == 20 || c < '0' || c > '9') {
        throw std::invalid_argument("invalid seed map header: " +
                                    path.string());
      }
      field.push_back(c);
    }
    if (!input || c != '&' || field.empty()) {
      throw std::invalid_argument("missing/invalid seed map header: " +
                                  path.string());
    }
    dimension = pair::ParseBytes(field);
  }
  if (dimensions != std::array<uint64_t, 3>{width, height, channels}) {
    throw std::invalid_argument("seed map/model dimensions mismatch: " +
                                path.string());
  }
  const uint64_t payload = pair::CheckedMultiply(
      4, pair::CheckedMultiply(channels, pair::CheckedMultiply(width, height)));
  if (fs::file_size(path) !=
      pair::CheckedAdd(static_cast<uint64_t>(input.tellg()), payload)) {
    throw std::invalid_argument("seed map payload size mismatch: " +
                                path.string());
  }
}

// File-header/model-checked logical minima, not physical peak guarantees.
// Inputs must remain immutable after admission. Requiring the
// cache to retain all used grayscale inputs and geometry maps also avoids
// returning references after a concurrent cache eviction. No cache
// rewrite/pool.
void CheckInputs(const mvs::PatchMatchOptions& options,
                 const mvs::PatchMatch::Problem& problem,
                 pair::ByteCharge charge,
                 const fs::path& workspace) {
  const auto& images = *problem.images;
  const auto& ref = images.at(problem.ref_image_idx);
  const uint64_t a = pair::ReadPairBitmapPixels(
      ref.GetPath(), ref.GetWidth(), ref.GetHeight());
  const uint64_t s = problem.src_image_idxs.size();
  if (!a || !s)
    throw std::invalid_argument("empty reference/source dimensions");
  uint64_t q = a, max_w = 0, max_h = 0;
  std::set<int> used{problem.ref_image_idx};
  for (int idx : problem.src_image_idxs) {
    if (!used.insert(idx).second)
      throw std::invalid_argument("duplicate effective source");
    const auto& image = images.at(idx);
    if (!image.GetWidth() || !image.GetHeight())
      throw std::invalid_argument("empty source dimensions");
    q = pair::CheckedAdd(
        q,
        pair::ReadPairBitmapPixels(
            image.GetPath(), image.GetWidth(), image.GetHeight()));
    max_w = std::max(max_w, static_cast<uint64_t>(image.GetWidth()));
    max_h = std::max(max_h, static_cast<uint64_t>(image.GetHeight()));
    if (!fs::is_regular_file(image.GetPath()))
      throw std::invalid_argument("source image missing");
  }
  const uint64_t cache = pair::CacheBytes(options.cache_size);
  if (options.geom_consistency) {
    for (const int idx : used) {
      const auto& image = images.at(idx);
      const auto name =
          image.GetPath().lexically_relative(workspace / "images").string();
      for (const auto* folder : {"depth_maps", "normal_maps"}) {
        CheckSeedMap(
            workspace / "stereo" / folder / (name + ".photometric.bin"),
            image.GetWidth(),
            image.GetHeight(),
            std::string(folder) == "depth_maps" ? 1 : 3);
      }
    }
  }
  const uint64_t retained =
      options.geom_consistency ? pair::CheckedMultiply(17, q) : q;
  if (cache < retained)
    throw std::invalid_argument(
        "cache must retain complete TWO-job private input set per owner");
  const uint64_t u =
      pair::CheckedMultiply(s, pair::CheckedMultiply(max_w, max_h));
  // Local decoded inputs and the PatchMatch owned snapshot coexist during
  // construction: each retains grayscale q + depth 4q + normals 12q.
  // Source grayscale staging u and float-depth staging 4u are sequential;
  // preserve the existing 24a result/reference scratch allowance.
  const uint64_t staging =
      options.geom_consistency ? pair::CheckedMultiply(4, u) : u;
  const uint64_t host = pair::CheckedAdd(
      cache,
      pair::CheckedAdd(pair::CheckedMultiply(2, retained),
                       std::max(staging, pair::CheckedMultiply(24, a))));
  // Current-owned initialization/steady envelope, plus private source arrays
  // and both live reference arrays. Logical only, NOT a physical bound.
  uint64_t device = pair::CheckedAdd(
      pair::BuildOwnedRotationMemoryPhases(ref.GetWidth(), ref.GetHeight(), s)
          .peak,
      u);
  if (options.geom_consistency) {
    device = pair::CheckedAdd(device, pair::CheckedMultiply(4, u));
  }
  device = pair::CheckedAdd(device, pair::CheckedMultiply(2, a));
  device = pair::CheckedAdd(device, pair::CheckedMultiply(688, s));
  device =
      pair::CheckedAdd(device, 32);  // Retained calibration policy allowance.
  if (charge.host < host || charge.device < device)
    throw std::invalid_argument(
        "job reservation below logical workload envelope");
}

// Resident private logical minima deliberately exclude the eliminated input
// cache, two host snapshots, source staging and private source CUDA arrays.
// Reviewed forecasts must cover allocation overhead and other private peaks.
// `include_owner_normal` (bounded reference-normal mode): the owner retains
// one 3-float-channel reference normal (12*width*height) until retirement.
// The existing 24a allowance covers result/reference scratch, NOT that input
// normal: distinct holder, no double count, no subtraction from 24a. Bank
// retention is excluded from the bounded bank forecast, so this is the only
// charge for the active normal's holder.
void CheckResidentInputs(const mvs::PatchMatch::Problem& problem,
                         pair::ByteCharge charge,
                         bool include_owner_normal = false) {
  const auto& ref = problem.images->at(problem.ref_image_idx);
  const auto minimum =
      pair::ResidentPrivateLogicalMinimum(ref.GetWidth(),
                                          ref.GetHeight(),
                                          problem.src_image_idxs.size(),
                                          include_owner_normal);
  if (charge.host < minimum.host || charge.device < minimum.device)
    throw std::invalid_argument(
        "resident private reservation below logical minimum");
}

template <class T>
std::string Value(const T& value) {
  std::ostringstream s;
  s << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return s.str();
}
void Metadata(pair::PairEvents& events,
              size_t job,
              const mvs::PatchMatchOptions& o,
              const mvs::PatchMatch::Problem& problem,
              int owned_tile_threads) {
  events.Identity(job,
                  problem.ref_image_idx,
                  static_cast<int>(pair::ParseBytes(o.gpu_index)));
  const auto& ref = problem.images->at(problem.ref_image_idx);
  events.Emit(job,
              "effective_metadata_begin",
              "width=" + Value(ref.GetWidth()) +
                  " height=" + Value(ref.GetHeight()) +
                  " owned_tile_threads=" + Value(owned_tile_threads) +
                  " ref_path_hex=" + pair::Hex(ref.GetPath().string()) +
                  " sources=" + Value(problem.src_image_idxs.size()));
  for (size_t i = 0; i < problem.src_image_idxs.size(); ++i) {
    const int idx = problem.src_image_idxs[i];
    events.Emit(job,
                "effective_source",
                "ordinal=" + Value(i) + " index=" + Value(idx) + " path_hex=" +
                    pair::Hex(problem.images->at(idx).GetPath().string()));
  }
  size_t count = 0;
#define OPTION(name)                                                     \
  events.Emit(job,                                                       \
              "effective_option",                                        \
              "option=" #name " value_hex=" + pair::Hex(Value(o.name))); \
  ++count
  OPTION(random_seed);
  OPTION(depth_min);
  OPTION(depth_max);
  OPTION(sigma_spatial);
  OPTION(sigma_color);
  OPTION(ncc_sigma);
  OPTION(min_triangulation_angle);
  OPTION(incident_angle_sigma);
  OPTION(geom_consistency_regularizer);
  OPTION(geom_consistency_max_cost);
  OPTION(filter_min_ncc);
  OPTION(filter_min_triangulation_angle);
  OPTION(filter_geom_consistency_max_cost);
  OPTION(cache_size);
  OPTION(gpu_index);
  OPTION(max_image_size);
  OPTION(window_radius);
  OPTION(window_step);
  OPTION(owned_distinct_source_cache);
  OPTION(owned_tex_lookahead);
  OPTION(num_samples);
  OPTION(num_iterations);
  OPTION(filter_min_num_consistent);
  OPTION(num_threads);
  OPTION(geom_consistency);
  OPTION(filter);
  OPTION(allow_missing_files);
  OPTION(write_consistency_graph);
#undef OPTION
  events.Emit(job, "effective_metadata_end", "option_count=" + Value(count));
}

struct PairSettings {
  std::string dispatch;
  pair::ByteCharge capacity;
  pair::ByteCharge process;
  std::array<pair::ByteCharge, 3> charges;
  uint64_t affinity;
  uint64_t io_cap;
  bool geometry_early_permits = false;
  size_t static_three_solver_permits = 0;
  int owned_tile_threads = 32;
  bool rolling_feed = false;
  bool resident_inputs = false;
  pair::ByteCharge bank_charge{};
  size_t max_concurrent_solvers = 0;
  bool benchmark_p08 = false;
  // Bounded reference-normal lookahead (experimental, default off): eager
  // banks retain validated reference normals; bounded banks load on demand
  // through a one-to-two-buffer FIFO producer reserved separately.
  bool normal_bounded = false;
  size_t normal_future = 0;
  uint64_t normal_charge = 0;
  // Opt-in contiguous FIFO resident cohorts (default off): per-cohort bank,
  // fresh coordinator/ledger, explicit mapping; see
  // RunCohortResidentProduction.
  bool cohort_feed = false;
  pair::ByteCharge cohort_queue{};
  pair::ByteCharge cohort_unclassified{};
};

// Immutable compact-generation -> original-bank mapping. Never truncate the
// bank.
std::vector<size_t> ResidentJobIndices(const PairSettings& settings,
                                       size_t count) {
  if (settings.benchmark_p08) {
    if (!settings.resident_inputs || !settings.rolling_feed ||
        !settings.max_concurrent_solvers || settings.dispatch != "pair" ||
        count != 76)
      throw std::invalid_argument(
          "benchmark P08 requires dynamic resident pair rolling and full 76 "
          "jobs");
    return {16, 17};
  }
  std::vector<size_t> indices;
  for (size_t j = 0; j < count; ++j) indices.push_back(j);
  return indices;
}

std::vector<fs::path> JobProjectPaths(
    const std::map<std::string, std::string>& args) {
  const auto list = args.find("job_project_list_path");
  if (list == args.end()) {
    std::vector<fs::path> paths{args.at("job_a_project_path"),
                                args.at("job_b_project_path")};
    if (args.count("job_c_project_path"))
      paths.push_back(args.at("job_c_project_path"));
    return paths;
  }
  const auto path = fs::absolute(list->second);
  std::ifstream stream(path);
  if (!stream) throw std::invalid_argument("cannot read job project list");
  std::vector<fs::path> paths;
  std::string line;
  while (std::getline(stream, line)) {
    // A line is a literal path, not shell syntax. Keep spaces, including at
    // either end; only remove the CR in a CRLF line ending.
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.find('\0') != std::string::npos) {
      throw std::invalid_argument("empty or invalid job project list entry");
    }
    fs::path entry(line);
    paths.push_back(entry.is_absolute() ? entry : path.parent_path() / entry);
  }
  if (!stream.eof() || paths.empty() ||
      (!args.count("max_concurrent_solvers") &&
       (args.count("static_three_solver_permits") ? paths.size() % 3 != 0
                                                  : paths.size() % 2 != 0))) {
    throw std::invalid_argument(
        args.count("static_three_solver_permits")
            ? "static three list requires a positive multiple of three jobs"
            : "job project list requires an even count >= 2");
  }
  return paths;
}

std::vector<std::unique_ptr<JobOptions>> LoadJobs(
    const std::vector<fs::path>& paths, const PairSettings& settings) {
  std::vector<std::unique_ptr<JobOptions>> jobs;
  jobs.reserve(paths.size());
  // All registrations are constructed on the main thread, before any INI is
  // read and before process globals are set. No owner starts during validation.
  for (size_t j = 0; j < paths.size(); ++j) {
    jobs.push_back(std::make_unique<JobOptions>());
  }
  std::set<std::string> references;
  for (size_t j = 0; j < jobs.size(); ++j) {
    auto& job = *jobs[j];
    if (!job.Read(paths[j].string(), false)) {
      throw std::invalid_argument("invalid job INI: " + paths[j].string());
    }
    if (!(job.globals == jobs.front()->globals)) {
      throw std::invalid_argument("conflicting process-global settings");
    }
    auto charge =
        settings.charges[j % (settings.static_three_solver_permits ? 3 : 2)];
    if (settings.rolling_feed) {
      // Validation only: actual admission and CheckInputs use a fitting slot.
      for (const auto& slot : settings.charges) {
        charge.host = std::max(charge.host, slot.host);
        charge.device = std::max(charge.device, slot.device);
      }
    }
    ValidateJob(job, charge, settings.resident_inputs);
    if (job.patch_match_stereo->geom_consistency !=
        jobs.front()->patch_match_stereo->geom_consistency) {
      throw std::invalid_argument("mixed photometric/geometric phases");
    }
    if (!references.insert(job.reference_name).second ||
        pair::ParseBytes(job.patch_match_stereo->gpu_index) !=
            pair::ParseBytes(jobs.front()->patch_match_stereo->gpu_index)) {
      throw std::invalid_argument("duplicate reference or mixed device");
    }
    for (size_t previous = 0; previous < j; ++previous) {
      if (pair::PathsAlias(job.workspace, jobs[previous]->workspace)) {
        throw std::invalid_argument("workspace/output roots alias");
      }
      for (const auto& a : job.output_dirs) {
        for (const auto& b : jobs[previous]->output_dirs) {
          if (pair::PathsAlias(a, b)) {
            throw std::invalid_argument("resolved output directories alias");
          }
        }
      }
    }
    if (settings.static_three_solver_permits ||
        settings.max_concurrent_solvers) {
      pair::ValidateIo(job.patch_match_stereo->num_threads,
                       job.patch_match_stereo->num_threads,
                       settings.io_cap,
                       settings.affinity);
    } else if (j % 2 == 1) {
      pair::ValidateIo(jobs[j - 1]->patch_match_stereo->num_threads,
                       job.patch_match_stereo->num_threads,
                       settings.io_cap,
                       settings.affinity);
    }
  }
  return jobs;
}

// RunJob owns the controller and all per-job resources in its call scope.
// The injected CPU test owner follows the SAME coordinator/owner orchestration.
template <class RunJob>
void RunPairEpoch(JobOptions& a,
                  JobOptions& b,
                  const PairSettings& settings,
                  RunJob& run_job,
                  const std::function<void()>& admitted = {},
                  const std::function<void()>& solver_released = {},
                  const std::function<void(size_t)>& cleanup_inputs = {},
                  JobOptions* c = nullptr) {
  const std::array<JobOptions*, 3> jobs{{&a, &b, c}};
  const auto& capacity = settings.capacity;
  const auto& process = settings.process;
  const auto& charges = settings.charges;
  const auto affinity = settings.affinity;
  const auto io_cap = settings.io_cap;
  const auto& g = a.globals;
  pair::PairEvents events(settings.dispatch, c ? 3 : 2);
  try {
    events.Emit(
        pair::PairEvents::kBatch,
        "validated",
        "driver_executable=colmap_owned_pair owned_tile_threads=" +
            Value(settings.owned_tile_threads) +
            (c ? " job_count=3 static_three_solver_permits=" +
                     Value(settings.static_three_solver_permits) +
                     " io_c=" + Value(c->patch_match_stereo->num_threads) +
                     " hc=" + Value(charges[2].host) +
                     " dc=" + Value(charges[2].device)
               : std::string{}) +
            std::string(settings.geometry_early_permits
                            ? " geometry_early_permits=1"
                            : "") +
            " io_threads_limit=" + Value(io_cap) +
            " affinity=" + Value(affinity) +
            " io_a=" + Value(jobs[0]->patch_match_stereo->num_threads) +
            " io_b=" + Value(jobs[1]->patch_match_stereo->num_threads) +
            " host_budget=" + Value(capacity.host) + " device_budget=" +
            Value(capacity.device) + " hp=" + Value(process.host) +
            " ha=" + Value(charges[0].host) + " hb=" + Value(charges[1].host) +
            " dp=" + Value(process.device) + " da=" + Value(charges[0].device) +
            " db=" + Value(charges[1].device) + " default_random_seed=" +
            Value(g.seed) + " log_target_hex=" + pair::Hex(g.target) +
            " log_path_hex=" + pair::Hex(g.path) + " log_level=" +
            Value(g.level) + " log_severity=" + Value(g.severity) +
            " log_color=" + Value(g.color));
    std::unique_ptr<pair::PairCoordinator> coordinator;
    try {
      auto sink = [&](const pair::PairEvent& e) {
        events.Coordinator(e);
        if (e.event == "solver_release" && e.job == 0 && solver_released) {
          solver_released();
        }
      };
      const auto dispatch = settings.dispatch == "serial"
                                ? pair::PairDispatch::kSerial
                                : pair::PairDispatch::kPair;
      if (c) {
        coordinator = std::make_unique<pair::PairCoordinator>(
            dispatch,
            capacity,
            process,
            charges,
            std::chrono::milliseconds(300000),
            sink,
            settings.static_three_solver_permits);
      } else {
        const std::array<pair::ByteCharge, 2> two{{charges[0], charges[1]}};
        coordinator = std::make_unique<pair::PairCoordinator>(
            dispatch,
            capacity,
            process,
            two,
            std::chrono::milliseconds(300000),
            sink,
            settings.geometry_early_permits ? pair::EarlyGeometryPermits::kOn
                                            : pair::EarlyGeometryPermits::kOff);
      }
    } catch (...) {
      events.Emit(pair::PairEvents::kBatch,
                  "admission_denied",
                  "reserved_host=0 reserved_device=0");
      throw;
    }
    const auto reserved = coordinator->ReservedBytes();
    events.Emit(pair::PairEvents::kBatch,
                "ledger_reserved",
                "reserved_host=" + Value(reserved.host) +
                    " reserved_device=" + Value(reserved.device));
    if (admitted) admitted();
    mvs::PatchMatchController::OwnedPairHooks hooks;
    hooks.check_inputs = [&](size_t j, const auto& o, const auto& p) {
      CheckInputs(o, p, charges[j], jobs[j]->workspace);
    };
    hooks.owned_tile_threads = settings.owned_tile_threads;
    hooks.metadata = [&](size_t j, const auto& o, const auto& p) {
      Metadata(events, j, o, p, settings.owned_tile_threads);
    };
    hooks.sweep = [&](size_t j, int iteration, int sweep) {
      events.Emit(j,
                  "sweep_complete",
                  "iteration=" + Value(iteration) + " sweep=" + Value(sweep));
    };
    hooks.output_phase = [&](size_t j, const char* phase) {
      events.Emit(j, phase);
    };
    pair::RunPairOwners(
        *coordinator,
        [&](size_t job) {
          events.Owner(job);
          run_job(*jobs[job], *coordinator, job, hooks);
        },
        [&](size_t job, const char* event) { events.Emit(job, event); },
        pair::PairThreadFactory{},
        [&](size_t job) {
          if (cleanup_inputs) cleanup_inputs(job);
        });
    const auto remaining = coordinator->ReservedBytes();
    if (remaining.host != 0 || remaining.device != 0) {
      throw std::logic_error("epoch finished with unreleased ledger credits");
    }
    events.Finished();
  } catch (const std::exception& e) {
    try {
      events.Emit(pair::PairEvents::kBatch,
                  "batch_failed",
                  "status=failure error_hex=" + pair::Hex(e.what()));
    } catch (...) {
    }
    throw;
  } catch (...) {
    try {
      events.Emit(pair::PairEvents::kBatch,
                  "batch_failed",
                  "status=failure error_hex=" + pair::Hex("unknown exception"));
    } catch (...) {
    }
    throw;
  }
}

using PreparedHostJob = mvs::PatchMatchController::OwnedPreparedJob;
using HostLookahead = pair::HostLookahead<PreparedHostJob>;

void HostLookaheadEvent(size_t target,
                        bool enabled,
                        const char* event,
                        int job,
                        uint64_t reserved,
                        const char* reason = nullptr) {
  std::ostringstream line;
  line << "\nOWNED_HOST_LOOKAHEAD pid=" << getpid()
       << " monotonic_ns=" << pair::PairEvents::NowNs()
       << " target_epoch=" << target << " event=" << event
       << " enabled=" << (enabled ? 1 : 0) << " reserved_host=" << reserved;
  if (job >= 0) line << " job=" << (job == 0 ? "A" : "B");
  if (reason) line << " reason=" << reason;
  line << " io_threads=" << (enabled ? 1 : 0) << '\n';
  const auto text = line.str();
  if (text.size() > 4096) throw std::length_error("lookahead event too long");
  ssize_t written;
  do {
    written = write(STDOUT_FILENO, text.data(), text.size());
  } while (written < 0 && errno == EINTR);
  if (written < 0 || static_cast<size_t>(written) != text.size()) {
    throw std::runtime_error("lookahead event write failed");
  }
}

std::unique_ptr<PreparedHostJob> PrepareFutureHostJob(
    JobOptions& job,
    size_t slot,
    pair::ByteCharge charge,
    const std::atomic<bool>& cancelled) {
  if (job.patch_match_stereo->geom_consistency) {
    throw std::invalid_argument("host lookahead is photometric-only");
  }
  mvs::PatchMatchController controller(*job.patch_match_stereo,
                                       job.workspace,
                                       job.format,
                                       job.pmvs,
                                       job.config,
                                       mvs::PatchMatchExecution::kOwnedCuda);
  mvs::PatchMatchController::OwnedPairHooks hooks;
  hooks.check_inputs = [&](size_t, const auto& options, const auto& problem) {
    if (cancelled.load(std::memory_order_relaxed)) {
      throw std::runtime_error("host preparation cancelled");
    }
    CheckInputs(options, problem, charge, job.workspace);
  };
  return controller.PrepareOwnedPairJob(slot, hooks, 1);
}

// Validate the complete fixed reservation and every reviewed forecast BEFORE
// host payload allocation or any CUDA construction. The bank remains globally
// charged separately from rolling credits; it cannot be borrowed by slots.
PairSettings AdmitResidentBank(PairSettings settings,
                               const std::vector<pair::ByteCharge>& forecasts,
                               pair::BankLimits estimate) {
  if (!settings.bank_charge.host || !settings.bank_charge.device ||
      settings.bank_charge.host < estimate.host_bytes ||
      settings.bank_charge.device < estimate.device_bytes)
    throw std::invalid_argument("resident bank charge below estimated minimum");
  settings.process.host =
      pair::CheckedAdd(settings.process.host, settings.bank_charge.host);
  settings.process.device =
      pair::CheckedAdd(settings.process.device, settings.bank_charge.device);
  // Validate bank+process and every full private forecast before allocation.
  // Legacy rolling also validates the complete three-slot reservation.
  if (settings.max_concurrent_solvers) {
    pair::PairCoordinator admission(settings.capacity,
                                    settings.process,
                                    settings.max_concurrent_solvers,
                                    forecasts,
                                    std::chrono::milliseconds(300000));
  } else {
    pair::PairCoordinator admission(settings.capacity,
                                    settings.process,
                                    settings.charges,
                                    forecasts,
                                    std::chrono::milliseconds(300000));
  }
  return settings;
}

void RunResidentProduction(std::vector<std::unique_ptr<JobOptions>>& jobs,
                           PairSettings settings,
                           std::vector<pair::ByteCharge> forecasts);
void RunCohortResidentProduction(std::vector<std::unique_ptr<JobOptions>>& jobs,
                                 PairSettings settings,
                                 std::vector<pair::ByteCharge> forecasts);

// Literal positive integer pairs: no inferred forecasts or added headroom.
std::vector<pair::ByteCharge> ReadRollingForecasts(const fs::path& path,
                                                   size_t count) {
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("cannot read rolling forecasts");
  std::vector<pair::ByteCharge> forecasts;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line);
    std::string host, device, extra;
    if (!(row >> host >> device) || (row >> extra))
      throw std::invalid_argument(
          "rolling forecast needs exactly two integer fields");
    pair::ByteCharge charge{pair::ParseBytes(host), pair::ParseBytes(device)};
    if (!charge.host || !charge.device)
      throw std::invalid_argument("rolling forecast must be positive");
    forecasts.push_back(charge);
  }
  if (!input.eof() || forecasts.size() != count)
    throw std::invalid_argument("rolling forecast count mismatch");
  return forecasts;
}

struct PrivateInputCheck {
  template <class Options, class Problem>
  void operator()(const Options& o,
                  const Problem& p,
                  pair::ByteCharge c,
                  const fs::path& w) const {
    CheckInputs(o, p, c, w);
  }
};

template <class RunJob, class InputCheck = PrivateInputCheck>
void RunRollingFeed(
    std::vector<std::unique_ptr<JobOptions>>& jobs,
    const PairSettings& settings,
    std::vector<pair::ByteCharge> forecasts,
    RunJob& run_job,
    InputCheck check_inputs = {},
    pair::PairEvents* invocation_events = nullptr,
    const std::vector<size_t>& bank_indices = {},
    std::function<void(const pair::PairEvent&)> coordinator_hook = {},
    size_t event_offset = 0) {
  const auto indices = bank_indices.empty()
                           ? ResidentJobIndices(settings, jobs.size())
                           : bank_indices;
  if (indices.size() != forecasts.size())
    throw std::invalid_argument("mapped forecast count mismatch");
  for (const auto index : indices)
    if (index >= jobs.size())
      throw std::invalid_argument("bank job index out of bounds");
  // Telemetry offsets map coordinator-local generation ids to invocation job
  // ids (cohort feeds hand out fresh ledgers per cohort). Indexing below stays
  // cohort-local; only event ids are remapped.
  const size_t batch = pair::PairEvents::kBatch;
  const auto mapped = [&](size_t job) {
    return job == batch ? batch : job + event_offset;
  };
  pair::PairEvents local_events(
      settings.dispatch,
      indices.size(),
      true,
      settings.max_concurrent_solvers ? settings.max_concurrent_solvers : 3);
  auto& events = invocation_events ? *invocation_events : local_events;
  std::unique_ptr<pair::PairCoordinator> coordinator;
  try {
    events.Emit(
        pair::PairEvents::kBatch,
        "validated",
        "driver_executable=colmap_owned_pair input_mode=" +
            std::string(settings.resident_inputs ? "resident" : "private") +
            " bank_host=" + Value(settings.bank_charge.host) +
            " bank_device=" + Value(settings.bank_charge.device) +
            " io_threads_limit=" + Value(settings.io_cap) + " affinity=" +
            Value(settings.affinity) + " job_count=" + Value(indices.size()) +
            " owned_tile_threads=" + Value(settings.owned_tile_threads) +
            (settings.max_concurrent_solvers
                 ? " max_concurrent_solvers=" +
                       Value(settings.max_concurrent_solvers) +
                       " credit_mode=actual_forecast"
                 : " static_three_solver_permits=3") +
            " host_budget=" + Value(settings.capacity.host) +
            " device_budget=" + Value(settings.capacity.device) +
            " hp=" + Value(settings.process.host) +
            " dp=" + Value(settings.process.device) +
            " ha=" + Value(settings.charges[0].host) +
            " da=" + Value(settings.charges[0].device) +
            " hb=" + Value(settings.charges[1].host) +
            " db=" + Value(settings.charges[1].device) +
            " hc=" + Value(settings.charges[2].host) +
            " dc=" + Value(settings.charges[2].device));
    try {
      auto sink = [&](const pair::PairEvent& e) {
        // Coordinator/owner failure hook runs FIRST: prompt non-joining
        // pipeline cancellation so blocked consumers/producers wake before
        // owners drain and join (which must not happen concurrently).
        if (coordinator_hook) coordinator_hook(e);
        if (e.event == "admitted") {
          events.BindSlot(mapped(e.job), coordinator->Slot(e.job));
        }
        pair::PairEvent forwarded = e;
        forwarded.job = mapped(e.job);
        events.Coordinator(forwarded);
        if (e.event == "admitted") {
          const auto slot = settings.max_concurrent_solvers
                                ? forecasts.at(e.job)
                                : settings.charges.at(coordinator->Slot(e.job));
          events.Emit(
              mapped(e.job),
              "forecast_bound",
              "forecast_host=" + Value(forecasts.at(e.job).host) +
                  " forecast_device=" + Value(forecasts.at(e.job).device) +
                  " slot_host=" + Value(slot.host) +
                  " slot_device=" + Value(slot.device));
        }
        if (settings.max_concurrent_solvers &&
            (e.event == "admitted" || e.event == "owner_joined")) {
          const auto credit = coordinator->ReservedBytes();
          events.Emit(mapped(e.job),
                      "dynamic_credit",
                      "reserved_host=" + Value(credit.host) +
                          " reserved_device=" + Value(credit.device));
        }
      };
      if (settings.max_concurrent_solvers) {
        coordinator = std::make_unique<pair::PairCoordinator>(
            settings.capacity,
            settings.process,
            settings.max_concurrent_solvers,
            forecasts,
            std::chrono::milliseconds(300000),
            sink);
      } else {
        coordinator = std::make_unique<pair::PairCoordinator>(
            settings.capacity,
            settings.process,
            settings.charges,
            forecasts,
            std::chrono::milliseconds(300000),
            sink);
      }
    } catch (...) {
      events.Emit(pair::PairEvents::kBatch,
                  "admission_denied",
                  "reserved_host=0 reserved_device=0");
      throw;
    }
    const auto reserved = coordinator->ReservedBytes();
    events.Emit(pair::PairEvents::kBatch,
                "ledger_reserved",
                "reserved_host=" + Value(reserved.host) +
                    " reserved_device=" + Value(reserved.device));
    mvs::PatchMatchController::OwnedPairHooks hooks;
    hooks.owned_tile_threads = settings.owned_tile_threads;
    hooks.check_inputs = [&](size_t j, const auto& o, const auto& p) {
      // Check both the reviewed forecast and the assigned immutable full slot.
      check_inputs(o, p, forecasts.at(j), jobs.at(indices.at(j))->workspace);
      if (!settings.max_concurrent_solvers)
        check_inputs(o,
                     p,
                     settings.charges.at(coordinator->Slot(j)),
                     jobs.at(indices.at(j))->workspace);
    };
    hooks.metadata = [&](size_t j, const auto& o, const auto& p) {
      Metadata(events, mapped(j), o, p, settings.owned_tile_threads);
    };
    hooks.sweep = [&](size_t j, int iteration, int sweep) {
      events.Emit(mapped(j),
                  "sweep_complete",
                  "iteration=" + Value(iteration) + " sweep=" + Value(sweep));
    };
    hooks.output_phase = [&](size_t j, const char* phase) {
      events.Emit(mapped(j), phase);
    };
    pair::RunRollingOwners(
        *coordinator,
        [&](size_t job) {
          events.Owner(mapped(job));
          run_job(*jobs.at(indices.at(job)), *coordinator, job, hooks);
        },
        [&](size_t job, const char* event) {
          events.Emit(mapped(job), event);
        });
    if (coordinator->ReservedBytes().host ||
        coordinator->ReservedBytes().device)
      throw std::logic_error("rolling feed retained ledger on success");
    if (!invocation_events) events.Finished();
  } catch (...) {
    if (!invocation_events) {
      try {
        events.Emit(pair::PairEvents::kBatch, "batch_failed", "status=failure");
      } catch (...) {
      }
    }
    throw;
  }
}

template <class RunJob,
          class PrepareJob = decltype(&PrepareFutureHostJob),
          class ResidentRun = decltype(&RunResidentProduction),
          class CohortRun = decltype(&RunCohortResidentProduction)>
int RunPairs(int argc,
             char** argv,
             RunJob run_job,
             PrepareJob prepare_job = &PrepareFutureHostJob,
             ResidentRun resident_run = &RunResidentProduction,
             CohortRun cohort_run = &RunCohortResidentProduction) {
  std::set<std::string> keys = {"dispatch",
                                "host_budget_bytes",
                                "device_budget_bytes",
                                "job_a_host_charge_bytes",
                                "job_a_device_charge_bytes",
                                "job_b_host_charge_bytes",
                                "job_b_device_charge_bytes",
                                "process_host_charge_bytes",
                                "process_device_charge_bytes",
                                "io_threads_limit"};
  if (argc == 2 &&
      (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    std::cout << "Finite even owned references; --dispatch serial|pair "
                 "(SOLVER serial only). Required:\n";
    for (const auto& key : keys) std::cout << "  --" << key << " <value>\n";
    std::cout
        << "  --job_a_project_path <INI> --job_b_project_path <INI>\n"
           "OR --job_project_list_path <list: one INI path per line, "
           "relative to list directory or absolute>\n"
           "Optional: --benchmark_selection p08 (dynamic resident only; full "
           "76 bank, dispatch 16/17)\n"
           "Optional: --owned_input_mode private|resident (default private)\n"
           "  Resident requires rolling geometry, reviewed private forecasts,\n"
           "  --bank_host_charge_bytes and --bank_device_charge_bytes; "
           "estimates are not physical bounds.\n"
           "Optional: --owned_feed_mode epochs|rolling (default epochs)\n"
           "  --job_forecast_list_path <FIFO host_bytes device_bytes lines; "
           "rolling only; final reviewed bounds including one headroom "
           "allowance>\n"
           "  Rolling requires geometry pair, static three or dynamic N, no "
           "lookahead.\n"
           "Optional: --max_concurrent_solvers <positive N> (resident rolling "
           "list only; replaces static permits and a/b/c charges)\n"
           "Optional: --owned_tile_threads 8|16|32 (default 32)\n"
           "  --owned_normal_lookahead 0|1 (resident rolling only; default 0 "
           "=\n"
           "  eager retained normals; 1 = experimental bounded on-demand\n"
           "  reference normal producer) --owned_normal_lookahead_depth 1|2\n"
           "  (default 2 future ready buffers) "
           "--normal_lookahead_charge_bytes\n"
           "  <separate host charge covering future+transient allowance;\n"
           "  reviewed owner forecasts must additionally cover the active\n"
           "  owned normal (24a+12a minimum)>\n"
           "  --owned_feed_mode cohort (resident rolling with an explicit\n"
           "  --max_concurrent_solvers only; partitions the resolved jobs\n"
           "  into contiguous FIFO cohorts with a fresh per-cohort ledger\n"
           "  and explicit original->cohort->bank-layer mapping; requires\n"
           "  --cohort_queue_host_bytes --cohort_queue_device_bytes\n"
           "  --cohort_unclassified_host_bytes\n"
           "  --cohort_unclassified_device_bytes (explicit, zero allowed);\n"
           "  reviewed bank charges must cover every cohort estimate)\n"
           "  --host_lookahead 0|1 (default 0) "
           "--host_lookahead_charge_bytes <separate host reservation>\n"
           "  --geometry_early_permits 0|1 (experimental, default 0; "
           "geometry-only, two owners per closed epoch, lookahead OFF; "
           "SERIAL unchanged)\n"
           "  --static_three_solver_permits 2|3 (closed geometry-only triples; "
           "pair dispatch; no lookahead)\n"
           "  --job_c_project_path <INI> (or positive multiple-of-three list)\n"
           "  --job_c_host_charge_bytes <bytes> --job_c_device_charge_bytes "
           "<bytes>\n";
    return EXIT_SUCCESS;
  }
  bool list_mode = false;
  for (int i = 1; i < argc; i += 2) {
    list_mode = list_mode || std::string(argv[i]) == "--job_project_list_path";
    if (std::string(argv[i]) == "--max_concurrent_solvers") {
      keys.insert("max_concurrent_solvers");
      for (const auto* key : {"job_a_host_charge_bytes",
                              "job_a_device_charge_bytes",
                              "job_b_host_charge_bytes",
                              "job_b_device_charge_bytes"})
        keys.erase(key);
    }
    if (std::string(argv[i]) == "--static_three_solver_permits") {
      keys.insert("static_three_solver_permits");
      keys.insert("job_c_host_charge_bytes");
      keys.insert("job_c_device_charge_bytes");
    }
    if (std::string(argv[i]) == "--geometry_early_permits") {
      keys.insert("geometry_early_permits");
    }
    for (const auto* key : {"owned_input_mode",
                            "bank_host_charge_bytes",
                            "bank_device_charge_bytes"})
      if (std::string(argv[i]) == std::string("--") + key) keys.insert(key);
    for (const auto* key : {"cohort_queue_host_bytes",
                            "cohort_queue_device_bytes",
                            "cohort_unclassified_host_bytes",
                            "cohort_unclassified_device_bytes"})
      if (std::string(argv[i]) == std::string("--") + key) keys.insert(key);
    for (const auto* key : {"owned_normal_lookahead",
                            "owned_normal_lookahead_depth",
                            "normal_lookahead_charge_bytes"}) {
      if (std::string(argv[i]) == std::string("--") + key) keys.insert(key);
    }
    if (std::string(argv[i]) == "--benchmark_selection")
      keys.insert("benchmark_selection");
    if (std::string(argv[i]) == "--owned_feed_mode")
      keys.insert("owned_feed_mode");
    if (std::string(argv[i]) == "--job_forecast_list_path")
      keys.insert("job_forecast_list_path");
    if (std::string(argv[i]) == "--owned_tile_threads")
      keys.insert("owned_tile_threads");
    if (std::string(argv[i]) == "--host_lookahead")
      keys.insert("host_lookahead");
    if (std::string(argv[i]) == "--host_lookahead_charge_bytes") {
      keys.insert("host_lookahead_charge_bytes");
    }
  }
  if (list_mode) {
    keys.insert("job_project_list_path");
  } else {
    keys.insert("job_a_project_path");
    keys.insert("job_b_project_path");
    if (keys.count("static_three_solver_permits"))
      keys.insert("job_c_project_path");
  }
  // Selecting one exact key set also rejects mixed legacy/list arguments.
  const auto args = pair::ParseStrictArguments(argc, argv, keys);
  const int owned_tile_threads =
      args.count("owned_tile_threads")
          ? pair::ParseOwnedTileThreads(args.at("owned_tile_threads"))
          : 32;
  if (args.at("dispatch") != "serial" && args.at("dispatch") != "pair") {
    throw std::invalid_argument("invalid dispatch");
  }
  if (args.count("geometry_early_permits") &&
      args.at("geometry_early_permits") != "0" &&
      args.at("geometry_early_permits") != "1") {
    throw std::invalid_argument("geometry_early_permits must be 0 or 1");
  }
  const uint64_t dynamic_limit =
      args.count("max_concurrent_solvers")
          ? pair::ParseBytes(args.at("max_concurrent_solvers"))
          : 0;
  if (args.count("max_concurrent_solvers") &&
      (!dynamic_limit || dynamic_limit > std::numeric_limits<size_t>::max() ||
       !list_mode || args.count("static_three_solver_permits") ||
       args.count("geometry_early_permits") || args.count("host_lookahead") ||
       args.count("host_lookahead_charge_bytes")))
    throw std::invalid_argument(
        "dynamic admission requires positive N, list mode and no "
        "static/early/lookahead options");
  size_t static_three_permits = 0;
  if (args.count("static_three_solver_permits")) {
    if (args.at("static_three_solver_permits") != "2" &&
        args.at("static_three_solver_permits") != "3") {
      throw std::invalid_argument("static_three_solver_permits must be 2 or 3");
    }
    static_three_permits =
        pair::ParseBytes(args.at("static_three_solver_permits"));
    if (args.at("dispatch") != "pair" || args.count("host_lookahead") ||
        args.count("host_lookahead_charge_bytes") ||
        args.count("geometry_early_permits")) {
      throw std::invalid_argument(
          "static three requires pair, no lookahead/other permit flags");
    }
  }
  const std::string feed_mode =
      args.count("owned_feed_mode") ? args.at("owned_feed_mode") : "epochs";
  if (feed_mode != "epochs" && feed_mode != "rolling" && feed_mode != "cohort")
    throw std::invalid_argument(
        "owned_feed_mode must be epochs, rolling, or cohort");
  const bool rolling_feed = feed_mode != "epochs";
  const bool cohort_feed = feed_mode == "cohort";
  if (rolling_feed &&
      ((!dynamic_limit && static_three_permits != 3) ||
       args.at("dispatch") != "pair" || !args.count("job_forecast_list_path")))
    throw std::invalid_argument(
        "rolling requires pair, static three or dynamic N, and reviewed "
        "forecasts");
  if (!rolling_feed && args.count("job_forecast_list_path"))
    throw std::invalid_argument("forecasts are rolling-only");
  const bool early_permits_requested = args.count("geometry_early_permits") &&
                                       args.at("geometry_early_permits") == "1";
  const bool lookahead_requested =
      args.count("host_lookahead") && args.at("host_lookahead") == "1";
  if (args.count("host_lookahead") && args.at("host_lookahead") != "0" &&
      args.at("host_lookahead") != "1") {
    throw std::invalid_argument("host_lookahead must be 0 or 1");
  }
  const uint64_t lookahead_charge =
      args.count("host_lookahead_charge_bytes")
          ? pair::ParseBytes(args.at("host_lookahead_charge_bytes"))
          : 0;
  if (lookahead_requested && lookahead_charge == 0) {
    throw std::invalid_argument(
        "lookahead requires separate positive host charge");
  }
  const std::string input_mode =
      args.count("owned_input_mode") ? args.at("owned_input_mode") : "private";
  if (input_mode != "private" && input_mode != "resident")
    throw std::invalid_argument("owned_input_mode must be private or resident");
  const bool resident = input_mode == "resident";
  if (dynamic_limit && (!resident || !rolling_feed))
    throw std::invalid_argument(
        "max_concurrent_solvers requires resident rolling mode");
  if (resident && (!rolling_feed || lookahead_requested ||
                   args.count("host_lookahead_charge_bytes") ||
                   !args.count("bank_host_charge_bytes") ||
                   !args.count("bank_device_charge_bytes")))
    throw std::invalid_argument(
        "resident requires rolling geometry, no lookahead and explicit bank "
        "charges");
  if (!resident && (args.count("bank_host_charge_bytes") ||
                    args.count("bank_device_charge_bytes")))
    throw std::invalid_argument("bank charges are resident-only");
  size_t normal_future = 0;
  uint64_t normal_charge = 0;
  if (args.count("owned_normal_lookahead")) {
    if (!resident) {
      throw std::invalid_argument(
          "bounded normal lookahead requires resident input mode");
    }
    const auto& value = args.at("owned_normal_lookahead");
    if (value != "0" && value != "1") {
      throw std::invalid_argument("owned_normal_lookahead must be 0 or 1");
    }
    if (value == "1") {
      if (!args.count("normal_lookahead_charge_bytes")) {
        throw std::invalid_argument(
            "bounded normal lookahead requires normal_lookahead_charge_bytes");
      }
      normal_charge =
          pair::ParseBytes(args.at("normal_lookahead_charge_bytes"));
      if (!normal_charge) {
        throw std::invalid_argument("normal lookahead charge must be positive");
      }
      normal_future =
          args.count("owned_normal_lookahead_depth")
              ? pair::ParseBytes(args.at("owned_normal_lookahead_depth"))
              : 2;
      if (normal_future == 0 || normal_future > 2) {
        throw std::invalid_argument(
            "owned_normal_lookahead_depth must be 1 or 2");
      }
      if (lookahead_requested || args.count("host_lookahead_charge_bytes")) {
        throw std::invalid_argument(
            "bounded normal lookahead and host lookahead are mutually "
            "exclusive");
      }
    } else if (args.count("normal_lookahead_charge_bytes") ||
               args.count("owned_normal_lookahead_depth")) {
      throw std::invalid_argument(
          "normal lookahead depth/charge require owned_normal_lookahead=1");
    }
  } else if (args.count("normal_lookahead_charge_bytes") ||
             args.count("owned_normal_lookahead_depth")) {
    throw std::invalid_argument(
        "normal lookahead depth/charge require owned_normal_lookahead");
  }
  const auto bytes = [&](const char* key) {
    return pair::ParseBytes(args.at(key));
  };
  PairSettings settings{
      args.at("dispatch"),
      {bytes("host_budget_bytes"), bytes("device_budget_bytes")},
      {bytes("process_host_charge_bytes"),
       bytes("process_device_charge_bytes")},
      {{{dynamic_limit ? 0 : bytes("job_a_host_charge_bytes"),
         dynamic_limit ? 0 : bytes("job_a_device_charge_bytes")},
        {dynamic_limit ? 0 : bytes("job_b_host_charge_bytes"),
         dynamic_limit ? 0 : bytes("job_b_device_charge_bytes")},
        {static_three_permits ? bytes("job_c_host_charge_bytes") : 0,
         static_three_permits ? bytes("job_c_device_charge_bytes") : 0}}},
      AffinityCount(),
      bytes("io_threads_limit"),
      early_permits_requested,
      static_three_permits,
      owned_tile_threads,
      rolling_feed,
      resident,
      {resident ? bytes("bank_host_charge_bytes") : 0,
       resident ? bytes("bank_device_charge_bytes") : 0},
      static_cast<size_t>(dynamic_limit),
      /*benchmark_p08=*/false,
      normal_future != 0,
      normal_future,
      normal_charge,
      /*cohort_feed=*/cohort_feed};
  if (resident && (!settings.bank_charge.host || !settings.bank_charge.device))
    throw std::invalid_argument("resident bank charges must be positive");
  if (cohort_feed) {
    if (!resident || dynamic_limit == 0)
      throw std::invalid_argument(
          "cohort feed requires resident rolling mode with "
          "max_concurrent_solvers");
    if (args.count("benchmark_selection"))
      throw std::invalid_argument(
          "cohort feed uses full identity selection, not benchmark selections");
    if (!args.count("cohort_queue_host_bytes") ||
        !args.count("cohort_queue_device_bytes") ||
        !args.count("cohort_unclassified_host_bytes") ||
        !args.count("cohort_unclassified_device_bytes"))
      throw std::invalid_argument(
          "cohort feed requires explicit cohort_queue and cohort_unclassified "
          "allowance terms (explicit zero allowed)");
    settings.cohort_queue = {bytes("cohort_queue_host_bytes"),
                             bytes("cohort_queue_device_bytes")};
    settings.cohort_unclassified = {bytes("cohort_unclassified_host_bytes"),
                                    bytes("cohort_unclassified_device_bytes")};
  } else if (args.count("cohort_queue_host_bytes") ||
             args.count("cohort_unclassified_host_bytes")) {
    throw std::invalid_argument(
        "cohort allowance terms require owned_feed_mode cohort");
  }
  if (args.count("benchmark_selection")) {
    if (args.at("benchmark_selection") != "p08")
      throw std::invalid_argument("benchmark_selection must be p08");
    settings.benchmark_p08 = true;
  }
  const auto project_paths = JobProjectPaths(args);
  ResidentJobIndices(settings, project_paths.size());  // Before loading inputs.
  BaseOptionManager globals(false);
  auto jobs = LoadJobs(project_paths, settings);
  if (dynamic_limit && !jobs.front()->patch_match_stereo->geom_consistency)
    throw std::invalid_argument(
        "dynamic resident admission requires geometry jobs");
  if (static_three_permits &&
      (jobs.empty() || jobs.size() % 3 != 0 ||
       !jobs.front()->patch_match_stereo->geom_consistency ||
       settings.affinity < 3)) {
    throw std::invalid_argument(
        "static three requires complete geometry triples and three CPU "
        "equivalents");
  }
  if (early_permits_requested &&
      (!jobs.front()->patch_match_stereo->geom_consistency ||
       lookahead_requested)) {
    throw std::invalid_argument(
        "early geometry permits require geometry jobs and lookahead OFF");
  }
  if (lookahead_requested &&
      jobs.front()->patch_match_stereo->geom_consistency) {
    throw std::invalid_argument("geometry host lookahead is unsupported");
  }
  const auto& g = jobs.front()->globals;
  std::vector<std::string> common = {"colmap_owned_pair",
                                     "--default_random_seed",
                                     Value(g.seed),
                                     "--log_target",
                                     g.target,
                                     "--log_path",
                                     g.path,
                                     "--log_level",
                                     Value(g.level),
                                     "--log_severity",
                                     Value(g.severity)};
#if COLMAP_GLOG_HAS_COLOR_SUPPORT
  common.insert(common.end(), {"--log_color", Value(g.color)});
#endif
  std::vector<char*> common_argv;
  for (auto& value : common) common_argv.push_back(value.data());
  if (!globals.Parse(static_cast<int>(common_argv.size()),
                     common_argv.data())) {
    throw std::invalid_argument("invalid global settings");
  }
  if (rolling_feed) {
    auto forecasts =
        ReadRollingForecasts(args.at("job_forecast_list_path"), jobs.size());
    if (resident) {
      if (settings.cohort_feed) {
        cohort_run(jobs, settings, std::move(forecasts));
        return EXIT_SUCCESS;
      }
      resident_run(jobs, settings, std::move(forecasts));
      return EXIT_SUCCESS;
    }
    auto execute = [&](JobOptions& job,
                       pair::PairCoordinator& coordinator,
                       size_t generation,
                       const auto& hooks) {
      if constexpr (std::is_invocable_v<RunJob&,
                                        JobOptions&,
                                        pair::PairCoordinator&,
                                        size_t,
                                        decltype(hooks),
                                        std::unique_ptr<PreparedHostJob>>) {
        run_job(job, coordinator, generation, hooks, nullptr);
      } else {
        run_job(job, coordinator, generation, hooks);
      }
    };
    RunRollingFeed(jobs, settings, std::move(forecasts), execute);
    return EXIT_SUCCESS;
  }
  if (static_three_permits) {
    auto execute = [&](JobOptions& job,
                       pair::PairCoordinator& coordinator,
                       size_t slot,
                       const auto& hooks) {
      if constexpr (std::is_invocable_v<RunJob&,
                                        JobOptions&,
                                        pair::PairCoordinator&,
                                        size_t,
                                        decltype(hooks),
                                        std::unique_ptr<PreparedHostJob>>) {
        run_job(job, coordinator, slot, hooks, nullptr);
      } else {
        run_job(job, coordinator, slot, hooks);
      }
    };
    for (size_t first = 0; first < jobs.size(); first += 3) {
      // Each call joins and retires all owners, checks Finish credits, and
      // destroys the coordinator before the next independent admission.
      RunPairEpoch(*jobs[first],
                   *jobs[first + 1],
                   settings,
                   execute,
                   {},
                   {},
                   {},
                   jobs[first + 2].get());
    }
    return EXIT_SUCCESS;
  }
  // The full additional transient charge is held even after CPU preparation
  // completes. Never lend current owner/process credits to future snapshots.
  const bool host_room = pair::CanReserveHostLookahead(settings.capacity.host,
                                                       settings.process.host,
                                                       settings.charges[0].host,
                                                       settings.charges[1].host,
                                                       lookahead_charge);
  std::unique_ptr<HostLookahead> next;
  if (lookahead_requested && jobs.size() == 2) {
    HostLookaheadEvent(1, false, "fallback", -1, 0, "no_next_epoch");
  }
  for (size_t first = 0; first < jobs.size(); first += 2) {
    // Prior owners/controllers/coordinator have been destroyed. Join and check
    // CPU errors BEFORE next GPU admission, then transfer only after admission.
    auto pending = std::move(next);
    if (pending) pending->Join();
    HostLookahead::Pair prepared;
    auto admitted = [&] {
      if (pending) prepared = pending->Consume();
    };
    auto execute = [&](JobOptions& job,
                       pair::PairCoordinator& coordinator,
                       size_t slot,
                       const auto& hooks) {
      if constexpr (std::is_invocable_v<RunJob&,
                                        JobOptions&,
                                        pair::PairCoordinator&,
                                        size_t,
                                        decltype(hooks),
                                        std::unique_ptr<PreparedHostJob>>) {
        run_job(job, coordinator, slot, hooks, std::move(prepared[slot]));
      } else {
        if (prepared[slot])
          throw std::logic_error("caller cannot consume snapshot");
        run_job(job, coordinator, slot, hooks);
      }
    };
    auto solver_released = [&] {
      if (first + 2 == jobs.size()) return;
      const size_t target = first / 2 + 1;
      if (!lookahead_requested) {
        HostLookaheadEvent(target, false, "disabled", -1, 0);
        return;
      }
      if (!host_room || settings.affinity < 2) {
        HostLookaheadEvent(target,
                           false,
                           "fallback",
                           -1,
                           0,
                           host_room ? "cpu_io_budget" : "host_budget");
        return;
      }
      next = std::make_unique<HostLookahead>(
          lookahead_charge,
          [target](const char* event, int slot, uint64_t reserved) {
            HostLookaheadEvent(target, true, event, slot, reserved);
          });
      next->Start([&, first](size_t slot, const std::atomic<bool>& cancelled) {
        return prepare_job(
            *jobs[first + 2 + slot], slot, settings.charges[slot], cancelled);
      });
    };
    RunPairEpoch(*jobs[first],
                 *jobs[first + 1],
                 settings,
                 execute,
                 admitted,
                 solver_released,
                 [&](size_t slot) { prepared[slot].reset(); });
  }
  return EXIT_SUCCESS;
}

void RunResidentProduction(std::vector<std::unique_ptr<JobOptions>>& jobs,
                           PairSettings settings,
                           std::vector<pair::ByteCharge> forecasts) {
  const auto bank_indices = ResidentJobIndices(settings, jobs.size());
  if (forecasts.size() != jobs.size())
    throw std::invalid_argument("full-bank forecast count mismatch");
  std::vector<pair::ByteCharge> selected_forecasts;
  for (const auto index : bank_indices)
    selected_forecasts.push_back(forecasts.at(index));
  pair::PairEvents events(
      settings.dispatch,
      bank_indices.size(),
      true,
      settings.max_concurrent_solvers ? settings.max_concurrent_solvers : 3);
  const char* stage = "resolve";
  bool bank_reserved = false;
  bool bank_constructed = false;
  try {
    events.Emit(
        pair::PairEvents::kBatch, "bank_init_begin", "input_mode=resident");
    std::vector<pair::BankJob> resolved;
    resolved.reserve(jobs.size());
    for (const auto& job : jobs) {
      mvs::PatchMatchController controller(
          *job->patch_match_stereo,
          job->workspace,
          job->format,
          job->pmvs,
          job->config,
          mvs::PatchMatchExecution::kOwnedCuda);
      resolved.push_back(controller.ResolveOwnedBankJob());
    }
    stage = "admission";
    if (settings.normal_bounded) {
      // Bounded reference-normal admission: the bank estimate excludes ALL
      // reference-normal retention; explicit disjoint allowances and the
      // operator charge are reserved separately. The pipeline charge is
      // folded into the process reservation BEFORE coordinator admission so
      // the whole invocation (bank + process + pipeline + each private
      // forecast) is validated against capacity with no borrowed credits and
      // no double release; the rolling ledger releases it after owners drain.
      const auto allowance = pair::HostInputBank::EstimateNormalAllowance(
          resolved, settings.normal_future);
      const uint64_t required =
          pair::CheckedAdd(allowance.future_bytes, allowance.transient_bytes);
      if (settings.normal_charge < required) {
        throw std::invalid_argument(
            "normal lookahead charge below explicit future+transient "
            "allowance");
      }
      events.Emit(pair::PairEvents::kBatch,
                  "normal_allowance",
                  "owner_bytes=" + Value(allowance.owner_bytes) +
                      " future_bytes=" + Value(allowance.future_bytes) +
                      " transient_bytes=" + Value(allowance.transient_bytes) +
                      " reference_layers=" + Value(allowance.reference_layers) +
                      " max_future=" + Value(settings.normal_future) +
                      " charge_bytes=" + Value(settings.normal_charge));
      settings.process.host =
          pair::CheckedAdd(settings.process.host, settings.normal_charge);
      settings = AdmitResidentBank(
          settings, forecasts, pair::HostInputBank::EstimateBounded(resolved));
    } else {
      settings = AdmitResidentBank(
          settings, forecasts, pair::HostInputBank::Estimate(resolved));
    }
    // The admission above checks the full invocation. Dedicate the bank slice
    // outside the rolling ledger: Finish releases only private process/slots,
    // while this invocation retains bank credit until GPU/host destruction.
    settings.process.host -= settings.bank_charge.host;
    settings.process.device -= settings.bank_charge.device;
    settings.capacity.host -= settings.bank_charge.host;
    settings.capacity.device -= settings.bank_charge.device;
    bank_reserved = true;
    events.Emit(pair::PairEvents::kBatch,
                "bank_reserved",
                "reserved_host=" + Value(settings.bank_charge.host) +
                    " reserved_device=" + Value(settings.bank_charge.device));
    // Validate private minima with resolved metadata, not eliminated input IO.
    stage = "private_validation";
    for (size_t j = 0; j < resolved.size(); ++j) {
      mvs::PatchMatch::Problem problem;
      problem.images = &resolved[j].images;
      problem.ref_image_idx = resolved[j].reference_image_idx;
      problem.src_image_idxs = resolved[j].source_image_idxs;
      CheckResidentInputs(problem, forecasts.at(j), settings.normal_bounded);
    }
    {
      stage = "host_load";
      pair::HostInputBank host(
          std::move(resolved),
          {settings.bank_charge.host, settings.bank_charge.device},
          settings.normal_bounded ? pair::BankNormals::kBounded
                                  : pair::BankNormals::kEager);
      stage = "gpu_upload";
      pair::ResidentInputBank gpu(
          host,
          static_cast<int>(
              pair::ParseBytes(jobs.front()->patch_match_stereo->gpu_index)));
      bank_constructed = true;
      stage = "bank_ready";
      host.ReleaseUploadPayloads();
      const auto counts = host.WorkCounts();
      events.Emit(pair::PairEvents::kBatch,
                  "bank_ready",
                  "layers=" + Value(host.Layers().size()) + " width=" +
                      Value(host.Width()) + " height=" + Value(host.Height()) +
                      " grayscale_reads=" + Value(counts.grayscale_reads) +
                      " depth_reads=" + Value(counts.depth_reads) +
                      " normal_reads=" + Value(counts.normal_reads) +
                      " normal_mode=" +
                      std::string(settings.normal_bounded ? "bounded_lookahead"
                                                          : "eager_retained") +
                      " bank_host=" + Value(settings.bank_charge.host) +
                      " bank_device=" + Value(settings.bank_charge.device));
      for (size_t j = 0; j < jobs.size(); ++j) {
        events.Emit(
            settings.benchmark_p08 ? pair::PairEvents::kBatch : j,
            "bank_reference",
            (settings.benchmark_p08 ? "bank_job_index=" + Value(j) + " " : "") +
                "layer=" + Value(host.ReferenceLayer(j)));
        const auto& layers = host.SourceLayers(j);
        for (size_t ordinal = 0; ordinal < layers.size(); ++ordinal)
          events.Emit(
              settings.benchmark_p08 ? pair::PairEvents::kBatch : j,
              "bank_source",
              (settings.benchmark_p08 ? "bank_job_index=" + Value(j) + " "
                                      : "") +
                  "ordinal=" + Value(ordinal) +
                  " layer=" + Value(layers[ordinal]));
      }
      // Bounded pipeline declared after the banks: its destructor joins the
      // CPU producer (in-flight reads complete first) before bank teardown,
      // and the loader lambda may safely borrow the host bank.
      std::unique_ptr<pair::NormalLookahead> normal_pipeline;
      if (settings.normal_bounded) {
        std::vector<size_t> pipeline_layers;
        pipeline_layers.reserve(bank_indices.size());
        for (const auto bank_job : bank_indices) {
          pipeline_layers.push_back(host.ReferenceLayer(bank_job));
        }
        normal_pipeline = std::make_unique<pair::NormalLookahead>(
            std::move(pipeline_layers),
            settings.normal_future,
            settings.normal_charge,
            [&](const char* event, size_t generation, uint64_t reserved) {
              const auto name = std::string("normal_pipeline_") + event;
              if (generation == pair::NormalLookahead::kBatch) {
                events.Emit(pair::PairEvents::kBatch,
                            name.c_str(),
                            "reserved_host=" + Value(reserved));
              } else {
                events.Emit(
                    generation,
                    name.c_str(),
                    "bank_job_index=" + Value(bank_indices.at(generation)) +
                        " reserved_host=" + Value(reserved));
              }
            });
        normal_pipeline->Start([&host](size_t layer, const std::atomic<bool>&) {
          return host.LoadReferenceNormal(layer);
        });
      }
      if (settings.benchmark_p08)
        for (size_t generation = 0; generation < bank_indices.size();
             ++generation)
          events.Emit(generation,
                      "bank_selection",
                      "bank_job_index=" + Value(bank_indices.at(generation)));
      auto execute = [&](JobOptions& job,
                         pair::PairCoordinator& coordinator,
                         size_t generation,
                         const auto& hooks) {
        mvs::NormalMap owned_normal;
        const mvs::NormalMap* owned_ref_normal = nullptr;
        if (normal_pipeline) {
          // Handoff of the unique owned normal for exactly this FIFO
          // generation to the admitted owner; the owner frame keeps it alive
          // until upload and owner retirement.
          owned_normal = normal_pipeline->Consume(generation);
          owned_ref_normal = &owned_normal;
        }
        mvs::PatchMatchController controller(
            *job.patch_match_stereo,
            job.workspace,
            job.format,
            job.pmvs,
            job.config,
            mvs::PatchMatchExecution::kOwnedCuda);
        controller.RunResidentOwnedPairJob(host,
                                           gpu,
                                           coordinator,
                                           generation,
                                           bank_indices.at(generation),
                                           hooks,
                                           owned_ref_normal);
      };
      // This returns/throws only after all owners have drained and joined.
      stage = "owners";
      RunRollingFeed(
          jobs,
          settings,
          std::move(selected_forecasts),
          execute,
          [bounded = settings.normal_bounded](const auto&,
                                              const auto& problem,
                                              pair::ByteCharge charge,
                                              const auto&) {
            CheckResidentInputs(problem, charge, bounded);
          },
          &events,
          bank_indices,
          [&](const pair::PairEvent& e) {
            // Wire any coordinator/owner failure to prompt, non-joining
            // pipeline cancellation; the producer joins exactly once at the
            // existing parent cleanup (Join on success or destructor unwind).
            if (normal_pipeline && e.event == "cancelled") {
              normal_pipeline->Cancel();
            }
          });
      if (normal_pipeline) normal_pipeline->Join();  // propagate loader errors
    }
    bank_constructed = false;
    events.Emit(pair::PairEvents::kBatch, "bank_destroyed");
    bank_reserved = false;
    events.Emit(pair::PairEvents::kBatch,
                "bank_release",
                "reserved_host=0 reserved_device=0");
    events.Finished();
  } catch (...) {
    // Inner scopes have unwound, and RunRollingFeed joined any owner threads.
    // A constructor failure never claims destruction of an unconstructed bank.
    try {
      if (bank_constructed)
        events.Emit(pair::PairEvents::kBatch, "bank_destroyed");
      if (bank_reserved)
        events.Emit(pair::PairEvents::kBatch,
                    "bank_release",
                    "reserved_host=0 reserved_device=0");
      events.Emit(pair::PairEvents::kBatch,
                  "batch_failed",
                  std::string("status=failure stage=") + stage);
    } catch (...) {
    }
    throw;
  }
}

// Opt-in contiguous FIFO resident cohorts (backlog story 04, experiment
// only; defaults unchanged). Resolve once per invocation; plan per-cohort
// contiguous slices with pure metadata (explicit terms, checked overflow);
// construct Host/ResidentInputBank per cohort, run its members on a fresh
// rolling ledger, drain owners and uploads, destroy the bank, then — and
// only then — admit the next cohort. Any failure/cancellation after the
// owners drain stops the whole invocation; no later cohort is started.
// Benchmark selections (p08) are rejected so full-bank global identity is
// never accidentally shrunk.
void RunCohortResidentProduction(std::vector<std::unique_ptr<JobOptions>>& jobs,
                                 PairSettings settings,
                                 std::vector<pair::ByteCharge> forecasts) {
  if (settings.benchmark_p08) {
    throw std::invalid_argument(
        "cohort feed uses full identity selection, not benchmark selections");
  }
  const auto bank_indices = ResidentJobIndices(settings, jobs.size());
  if (forecasts.size() != jobs.size())
    throw std::invalid_argument("full forecast count mismatch");
  pair::PairEvents events(
      settings.dispatch,
      bank_indices.size(),
      true,
      settings.max_concurrent_solvers ? settings.max_concurrent_solvers : 3);
  const char* stage = "resolve";
  bool bank_constructed = false;
  bool bank_reserved = false;
  try {
    events.Emit(pair::PairEvents::kBatch,
                "bank_init_begin",
                "input_mode=resident feed_mode=cohort");
    // Invocation-wide resolution: metadata only, no payload, no CUDA.
    std::vector<pair::BankJob> resolved;
    resolved.reserve(jobs.size());
    for (const auto& job : jobs) {
      mvs::PatchMatchController controller(
          *job->patch_match_stereo,
          job->workspace,
          job->format,
          job->pmvs,
          job->config,
          mvs::PatchMatchExecution::kOwnedCuda);
      resolved.push_back(controller.ResolveOwnedBankJob());
    }
    stage = "plan";
    const int device_index = static_cast<int>(
        pair::ParseBytes(jobs.front()->patch_match_stereo->gpu_index));
    const bool bounded = settings.normal_bounded;
    // Capture the invocation-wide immutable identity ONCE, per job/input,
    // before any planning read; every later estimator call, bank reload and
    // on-demand load validates against THIS snapshot — recapture is never
    // adopted as proof.
    stage = "identity_snapshot";
    const auto identity_snapshot =
        pair::HostInputBank::CaptureJobIdentities(resolved);
    events.Emit(pair::PairEvents::kBatch,
                "cohort_identity_snapshot",
                "jobs=" + Value(identity_snapshot.size()));
    const auto validate_slice_identity =
        [&](const std::vector<pair::BankJob>& slice, size_t first) {
          const auto current = pair::HostInputBank::CaptureJobIdentities(slice);
          if (first + current.size() > identity_snapshot.size()) {
            throw std::runtime_error("identity snapshot/job mapping mismatch");
          }
          for (size_t k = 0; k < current.size(); ++k) {
            const auto& expected = identity_snapshot.at(first + k);
            if (expected.size() != current[k].size()) {
              throw std::runtime_error(
                  "resident identity snapshot/job mapping mismatch");
            }
            for (size_t m = 0; m < current[k].size(); ++m) {
              if (!pair::SameIdentity(current[k].at(m).image,
                                      expected.at(m).image) ||
                  !pair::SameIdentity(current[k].at(m).depth,
                                      expected.at(m).depth) ||
                  !pair::SameIdentity(current[k].at(m).normal,
                                      expected.at(m).normal)) {
                throw std::runtime_error(
                    "resident input changed: does not match the captured "
                    "invocation identity");
              }
            }
          }
        };
    const auto estimate_slice = [&](size_t first,
                                    size_t count) -> pair::ByteCharge {
      const std::vector<pair::BankJob> slice(
          resolved.begin() + static_cast<long>(first),
          resolved.begin() + static_cast<long>(first + count));
      // Validate the snapshot against this planner read BEFORE and AFTER:
      // a same-shaped replacement is rejected, never re-admitted.
      validate_slice_identity(slice, first);
      const auto limits = bounded ? pair::HostInputBank::EstimateBounded(slice)
                                  : pair::HostInputBank::Estimate(slice);
      validate_slice_identity(slice, first);
      return {limits.host_bytes, limits.device_bytes};
    };
    // Pure metadata planning with explicit terms: total-envelope preflight of
    // every singleton (NOT a bank-only device cap), contiguous FIFO packing,
    // checked overflow; no payload/CUDA allocation happens before this.
    // The queue/transient term carries the EXPLICIT bounded normal pipeline
    // charge (when enabled) plus the caller's explicit queue allowance, so
    // the plan arithmetic equals the per-cohort ledger admission arithmetic
    // exactly; the private forecasts and unclassified allowance stay
    // separate, and nothing is invented here.
    pair::CohortTerms terms;
    terms.process = settings.process;
    terms.queue_transient = {
        pair::CheckedAdd(settings.cohort_queue.host, settings.normal_charge),
        settings.cohort_queue.device};
    terms.unclassified = settings.cohort_unclassified;
    // The reviewed per-cohort bank reservation: planning packs under the
    // ACTUAL reservation policy (bank estimate must fit inside the charge,
    // charge + terms + private must fit capacity), so later admissions can
    // never fail what planning already claimed fit.
    terms.reviewed_bank = settings.bank_charge;
    stage = "plan";
    const auto plan = pair::PlanCohorts(
        bank_indices, forecasts, terms, settings.capacity, estimate_slice);
    std::string cohort_jobs;
    for (size_t c = 0; c < plan.cohorts.size(); ++c) {
      for (const auto id : plan.cohorts.at(c).original_jobs) {
        if (!cohort_jobs.empty()) cohort_jobs += "+";
        cohort_jobs += std::to_string(id);
      }
    }
    events.Emit(
        pair::PairEvents::kBatch,
        "cohort_plan",
        "cohorts=" + Value(plan.cohorts.size()) + " job_count=" +
            Value(plan.job_count) + " jobs=" + cohort_jobs + " normal_mode=" +
            std::string(bounded ? "bounded_lookahead" : "eager_retained") +
            " budget_host=" + Value(settings.capacity.host) +
            " budget_device=" + Value(settings.capacity.device) +
            " process_host=" + Value(terms.process.host) +
            " process_device=" + Value(terms.process.device) +
            " queue_host=" + Value(terms.queue_transient.host) +
            " queue_device=" + Value(terms.queue_transient.device) +
            " unclassified_host=" + Value(terms.unclassified.host) +
            " unclassified_device=" + Value(terms.unclassified.device));
    // Exact per-cohort subset + budget terms for the runtime memory model
    // adapter: bank estimate (native, single 3% margin), explicit terms, per-
    // side maxima of the member private forecasts, admitted totals, and the
    // explicit original->cohort->ordinal mapping per job.
    for (size_t c = 0; c < plan.cohorts.size(); ++c) {
      const auto& entry = plan.cohorts.at(c);
      std::string jobs_field;
      for (const auto id : entry.original_jobs) {
        if (!jobs_field.empty()) jobs_field += "+";
        jobs_field += std::to_string(id);
      }
      events.Emit(
          pair::PairEvents::kBatch,
          "cohort_planned",
          "cohort=" + Value(c) + " bank_jobs=" + jobs_field +
              " count=" + Value(entry.count) +
              " bank_host=" + Value(entry.bank_estimate.host) +
              " bank_device=" + Value(entry.bank_estimate.device) +
              " process_host=" + Value(terms.process.host) +
              " process_device=" + Value(terms.process.device) +
              " queue_host=" + Value(terms.queue_transient.host) +
              " queue_device=" + Value(terms.queue_transient.device) +
              " unclassified_host=" + Value(terms.unclassified.host) +
              " unclassified_device=" + Value(terms.unclassified.device) +
              " max_private_host=" + Value(entry.member_private.host) +
              " max_private_device=" + Value(entry.member_private.device) +
              " terms_total_host=" + Value(entry.terms_total.host) +
              " terms_total_device=" + Value(entry.terms_total.device) +
              " required_host=" + Value(entry.required.host) +
              " required_device=" + Value(entry.required.device) +
              " budget_host=" + Value(settings.capacity.host) +
              " budget_device=" + Value(settings.capacity.device));
      for (size_t k = 0; k < entry.original_jobs.size(); ++k)
        events.Emit(pair::PairEvents::kBatch,
                    "cohort_mapping",
                    "cohort=" + Value(c) +
                        " bank_job_index=" + Value(entry.original_jobs.at(k)) +
                        " cohort_ordinal=" + Value(k));
    }
    // Reserve the explicit outside-formula allowances once for the whole
    // invocation; per-cohort ledger admission validates against the total.
    settings.process.host =
        pair::CheckedAdd(settings.process.host,
                         pair::CheckedAdd(settings.cohort_queue.host,
                                          settings.cohort_unclassified.host));
    settings.process.device =
        pair::CheckedAdd(settings.process.device,
                         pair::CheckedAdd(settings.cohort_queue.device,
                                          settings.cohort_unclassified.device));
    // Full pre-payload preflight of EVERY planned cohort's ACTUAL
    // reservation policy + every singleton's complete private minimum and
    // (bounded mode) normal allowance minimum. Nothing is built, uploaded
    // or admitted before all of these pass, so no later cohort admission
    // can fail an earlier cohort's already-allocated payload.
    for (size_t c = 0; c < plan.cohorts.size(); ++c) {
      stage = "cohort_admission";
      const size_t first = plan.cohorts.at(c).first;
      const size_t count = plan.cohorts.at(c).count;
      if (settings.bank_charge.host < plan.cohorts.at(c).bank_estimate.host ||
          settings.bank_charge.device < plan.cohorts.at(c).bank_estimate.device)
        throw std::invalid_argument("cohort bank charge below native estimate");
      std::vector<pair::ByteCharge> member_forecasts(
          forecasts.begin() + static_cast<long>(first),
          forecasts.begin() + static_cast<long>(first + count));
      if (bounded) {
        const std::vector<pair::BankJob> slice(
            resolved.begin() + static_cast<long>(first),
            resolved.begin() + static_cast<long>(first + count));
        const auto allowance = pair::HostInputBank::EstimateNormalAllowance(
            slice, settings.normal_future);
        const uint64_t required =
            pair::CheckedAdd(allowance.future_bytes, allowance.transient_bytes);
        if (settings.normal_charge < required)
          throw std::invalid_argument(
              "normal lookahead charge below explicit future+transient "
              "allowance");
      }
      // Same admission validation execution performs, on a throwaway copy.
      PairSettings admission_settings = settings;
      admission_settings =
          AdmitResidentBank(admission_settings,
                            member_forecasts,
                            {plan.cohorts.at(c).bank_estimate.host,
                             plan.cohorts.at(c).bank_estimate.device});
      (void)admission_settings;
    }
    // Invocation-wide complete-private-job minima (metadata only) for ALL
    // members against their reviewed forecasts, before any payload build.
    for (size_t j = 0; j < jobs.size(); ++j) {
      mvs::PatchMatch::Problem problem;
      problem.images = &resolved[j].images;
      problem.ref_image_idx = resolved[j].reference_image_idx;
      problem.src_image_idxs = resolved[j].source_image_idxs;
      CheckResidentInputs(problem, forecasts.at(j), bounded);
    }
    // Per-cohort execution: identical actual reservation policy validated
    // in the pre-payload preflight above is re-validated here as each fresh
    // ledger is constructed, then the bank is built, run, drained and
    // destroyed strictly sequentially.
    for (size_t c = 0; c < plan.cohorts.size(); ++c) {
      stage = "cohort_admission";
      PairSettings cohort_settings = settings;
      const size_t first = plan.cohorts.at(c).first;
      const size_t count = plan.cohorts.at(c).count;
      std::vector<pair::ByteCharge> member_forecasts(
          forecasts.begin() + static_cast<long>(first),
          forecasts.begin() + static_cast<long>(first + count));
      if (bounded) {
        const std::vector<pair::BankJob> slice(
            resolved.begin() + static_cast<long>(first),
            resolved.begin() + static_cast<long>(first + count));
        const auto allowance = pair::HostInputBank::EstimateNormalAllowance(
            slice, settings.normal_future);
        events.Emit(
            pair::PairEvents::kBatch,
            "cohort_normal_allowance",
            "cohort=" + Value(c) +
                " owner_bytes=" + Value(allowance.owner_bytes) +
                " future_bytes=" + Value(allowance.future_bytes) +
                " transient_bytes=" + Value(allowance.transient_bytes) +
                " reference_layers=" + Value(allowance.reference_layers) +
                " max_future=" + Value(settings.normal_future) +
                " charge_bytes=" + Value(settings.normal_charge));
        cohort_settings.process.host = pair::CheckedAdd(
            cohort_settings.process.host, settings.normal_charge);
      }
      cohort_settings =
          AdmitResidentBank(cohort_settings,
                            member_forecasts,
                            {plan.cohorts.at(c).bank_estimate.host,
                             plan.cohorts.at(c).bank_estimate.device});
      // Dedicate the cohort bank outside the rolling ledger until teardown.
      cohort_settings.process.host -= settings.bank_charge.host;
      cohort_settings.process.device -= settings.bank_charge.device;
      cohort_settings.capacity.host -= settings.bank_charge.host;
      cohort_settings.capacity.device -= settings.bank_charge.device;
      bank_reserved = true;
      events.Emit(pair::PairEvents::kBatch,
                  "cohort_bank_reserved",
                  "cohort=" + Value(c) +
                      " reserved_host=" + Value(settings.bank_charge.host) +
                      " reserved_device=" + Value(settings.bank_charge.device));
      stage = "host_load";
      // The bank-owning scope ends BEFORE the destroy/release telemetry: the
      // events must claim nothing about still-live resources. The scope's
      // bank reload validates every admitted member's captured invocation
      // identity before and after the payload reads; it never recaptures.
      {
        std::vector<pair::BankJob> slice(
            resolved.begin() + static_cast<long>(first),
            resolved.begin() + static_cast<long>(first + count));
        pair::HostInputBank host(
            std::move(slice),
            {settings.bank_charge.host, settings.bank_charge.device},
            bounded ? pair::BankNormals::kBounded : pair::BankNormals::kEager,
            identity_snapshot.data() + first,
            count);
        // Bounded pipeline declared after the host bank; its destructor joins
        // the producer before the bank teardown (loader borrows the bank).
        std::unique_ptr<pair::NormalLookahead> normal_pipeline;
        if (bounded) {
          std::vector<size_t> pipeline_layers;
          pipeline_layers.reserve(count);
          for (size_t local = 0; local < count; ++local)
            pipeline_layers.push_back(host.ReferenceLayer(local));
          normal_pipeline = std::make_unique<pair::NormalLookahead>(
              std::move(pipeline_layers),
              settings.normal_future,
              settings.normal_charge,
              [&, c](const char* event, size_t generation, uint64_t reserved) {
                const auto name = std::string("normal_pipeline_") + event;
                if (generation == pair::NormalLookahead::kBatch) {
                  events.Emit(pair::PairEvents::kBatch,
                              name.c_str(),
                              "cohort=" + Value(c) +
                                  " reserved_host=" + Value(reserved));
                } else {
                  events.Emit(pair::PairEvents::kBatch,
                              name.c_str(),
                              "cohort=" + Value(c) + " bank_job_index=" +
                                  Value(plan.cohorts.at(c).original_jobs.at(
                                      generation)) +
                                  " bank_layer=" +
                                  Value(static_cast<size_t>(
                                      host.ReferenceLayer(generation))) +
                                  " reserved_host=" + Value(reserved));
                }
              });
          normal_pipeline->Start(
              [&host](size_t layer, const std::atomic<bool>&) {
                return host.LoadReferenceNormal(layer);
              });
        }
        stage = "gpu_upload";
        pair::ResidentInputBank gpu(host, device_index);
        bank_constructed = true;
        stage = "bank_ready";
        host.ReleaseUploadPayloads();
        const auto counts = host.WorkCounts();
        events.Emit(
            pair::PairEvents::kBatch,
            "cohort_bank_ready",
            "cohort=" + Value(c) + " layers=" + Value(host.Layers().size()) +
                " width=" + Value(host.Width()) +
                " height=" + Value(host.Height()) +
                " grayscale_reads=" + Value(counts.grayscale_reads) +
                " depth_reads=" + Value(counts.depth_reads) + " normal_reads=" +
                Value(counts.normal_reads) + " normal_mode=" +
                std::string(bounded ? "bounded_lookahead" : "eager_retained") +
                " bank_host=" + Value(settings.bank_charge.host) +
                " bank_device=" + Value(settings.bank_charge.device));
        // Explicit original job -> cohort -> bank layer mapping.
        for (size_t local = 0; local < count; ++local) {
          events.Emit(
              pair::PairEvents::kBatch,
              "cohort_bank_reference",
              "cohort=" + Value(c) + " bank_job_index=" +
                  Value(plan.cohorts.at(c).original_jobs.at(local)) +
                  " layer=" +
                  Value(static_cast<size_t>(host.ReferenceLayer(local))));
          const auto& layers = host.SourceLayers(local);
          for (size_t ordinal = 0; ordinal < layers.size(); ++ordinal)
            events.Emit(
                pair::PairEvents::kBatch,
                "cohort_bank_source",
                "cohort=" + Value(c) + " bank_job_index=" +
                    Value(plan.cohorts.at(c).original_jobs.at(local)) +
                    " bank_layer_ordinal=" + Value(static_cast<size_t>(local)) +
                    " ordinal=" + Value(ordinal) +
                    " layer=" + Value(static_cast<size_t>(layers[ordinal])));
        }
        auto execute = [&](JobOptions& job,
                           pair::PairCoordinator& coordinator,
                           size_t generation,
                           const auto& hooks) {
          mvs::NormalMap owned_normal;
          const mvs::NormalMap* owned_ref_normal = nullptr;
          if (normal_pipeline) {
            // Unique bounded FIFO handoff for exactly this cohort-local;
            // the owner frame keeps the map alive until upload/retirement.
            owned_normal = normal_pipeline->Consume(generation);
            owned_ref_normal = &owned_normal;
          }
          mvs::PatchMatchController controller(
              *job.patch_match_stereo,
              job.workspace,
              job.format,
              job.pmvs,
              job.config,
              mvs::PatchMatchExecution::kOwnedCuda);
          controller.RunResidentOwnedPairJob(host,
                                             gpu,
                                             coordinator,
                                             generation,
                                             generation,
                                             hooks,
                                             owned_ref_normal);
        };
        stage = "owners";
        // Fresh ledger per cohort; returns/throws only after owners drained.
        RunRollingFeed(
            jobs,
            cohort_settings,
            std::move(member_forecasts),
            execute,
            [bounded](const auto&,
                      const auto& problem,
                      pair::ByteCharge charge,
                      const auto&) {
              CheckResidentInputs(problem, charge, bounded);
            },
            &events,
            plan.cohorts.at(c).original_jobs,
            [&normal_pipeline](const pair::PairEvent& e) {
              if (normal_pipeline && e.event == "cancelled")
                normal_pipeline->Cancel();
            },
            first);
        if (normal_pipeline)
          normal_pipeline->Join();  // propagate loader errors
        // End of the bank-owning scope: host bank, resident bank and the
        // bounded pipeline's own buffers are destroyed here, strictly before
        // any destroy/release telemetry for this cohort.
      }
      bank_constructed = false;
      stage = "cohort_teardown";
      events.Emit(pair::PairEvents::kBatch,
                  "cohort_bank_destroyed",
                  "cohort=" + Value(c));
      bank_reserved = false;
      events.Emit(pair::PairEvents::kBatch,
                  "cohort_bank_release",
                  "cohort=" + Value(c) + " reserved_host=0 reserved_device=0");
    }
    events.Finished();
  } catch (...) {
    // Inner scopes have unwound; owners joined by RunRollingFeed. Cohort
    // bank destruction claims never precede actual destruction.
    try {
      if (bank_constructed)
        events.Emit(pair::PairEvents::kBatch, "cohort_bank_destroyed");
      if (bank_reserved)
        events.Emit(pair::PairEvents::kBatch,
                    "cohort_bank_release",
                    "reserved_host=0 reserved_device=0");
      events.Emit(pair::PairEvents::kBatch,
                  "batch_failed",
                  std::string("status=failure stage=") + stage);
    } catch (...) {
    }
    throw;
  }
}

int RunPair(int argc, char** argv) {
  return RunPairs(argc,
                  argv,
                  [](JobOptions& job,
                     pair::PairCoordinator& coordinator,
                     size_t slot,
                     const mvs::PatchMatchController::OwnedPairHooks& hooks,
                     std::unique_ptr<PreparedHostJob> prepared) {
                    mvs::PatchMatchController controller(
                        *job.patch_match_stereo,
                        job.workspace,
                        job.format,
                        job.pmvs,
                        job.config,
                        mvs::PatchMatchExecution::kOwnedCuda);
                    if (prepared) {
                      controller.RunPreparedOwnedPairJob(
                          std::move(prepared), coordinator, slot, hooks);
                    } else {
                      controller.RunOwnedPairJob(coordinator, slot, hooks);
                    }
                  });
}
}  // namespace
}  // namespace colmap

int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);
  try {
    return colmap::RunPair(argc, argv);
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return EXIT_FAILURE;
  } catch (...) {
    LOG(ERROR) << "Unknown owned pair failure";
    return EXIT_FAILURE;
  }
}
