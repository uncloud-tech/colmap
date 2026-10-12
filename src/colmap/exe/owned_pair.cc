// SPDX-License-Identifier: BSD-3-Clause
// Finite two-owner epochs or experimental closed static triples. Never a queue.
#include "colmap/controllers/option_manager.h"
#include "colmap/mvs/owned_host_lookahead.h"
#include "colmap/mvs/owned_pair_events.h"
#include "colmap/mvs/owned_pair_image.h"
#include "colmap/mvs/owned_pair_run.h"
#include "colmap/mvs/owned_pair_validation.h"
#include "colmap/mvs/owned_rotation_storage.h"
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
    return std::tie(seed,target,path,level,severity,color) ==
           std::tie(other.seed,other.target,other.path,other.level,other.severity,other.color);
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
  std::array<fs::path,2> outputs;
  std::array<fs::path,2> output_dirs;
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

void ValidateJob(JobOptions& job, pair::ByteCharge charge) {
  auto& o = *job.patch_match_stereo;
  StringToLower(&job.format);
  if (job.format != "colmap" || (o.geom_consistency && !o.filter) ||
      o.window_radius != 5 ||
      o.window_step != 1 || o.num_iterations != 5 || o.max_image_size != -1 ||
      o.allow_missing_files || o.write_consistency_graph) {
    throw std::invalid_argument("TWO prototype requires native COLMAP 11/1, five iterations, complete inputs, no graph; geometry requires filter=1");
  }
  const uint64_t device = pair::ParseBytes(o.gpu_index);
  if (device > INT_MAX) throw std::invalid_argument("device index overflow");
  if (pair::CacheBytes(o.cache_size) > charge.host) {
    throw std::invalid_argument("cache cap exceeds entire job host reservation");
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
      complete = complete && fs::is_regular_file(
          job.workspace / "sparse" / (std::string(stem) + extension));
    }
    complete_sparse = complete_sparse || complete;
  }
  if (!complete_sparse) throw std::invalid_argument("sparse model inputs missing");
  job.config = fs::canonical(job.config.empty()
                                ? job.workspace / "stereo/patch-match.cfg"
                                : job.config);
  std::vector<std::string> lines;
  for (auto line : ReadTextFileLines(job.config)) {
    StringTrim(&line);
    if (!line.empty() && line[0] != '#') lines.push_back(line);
  }
  if (lines.size() != 2) throw std::invalid_argument("exactly ONE reference per job required");
  job.reference_name = lines[0];
  fs::path reference(job.reference_name);
  if (reference.empty() || reference.is_absolute()) throw std::invalid_argument("invalid reference path");
  for (const auto& component : reference) {
    if (component == ".." || component == ".") throw std::invalid_argument("unsafe reference path");
  }
  if (!fs::is_regular_file(job.workspace / "images" / reference)) {
    throw std::invalid_argument("reference image missing");
  }
  const auto stereo = fs::weakly_canonical(job.workspace / "stereo");
  if (!pair::PathContains(job.workspace,stereo) || stereo==job.workspace) {
    throw std::invalid_argument("stereo output root escapes workspace");
  }
  for (size_t k=0;k<2;++k) {
    const auto dir = fs::weakly_canonical(job.workspace / "stereo" /
                                        (k==0 ? "depth_maps" : "normal_maps"));
    job.output_dirs[k] = dir;
    if (!pair::PathContains(stereo,dir) || dir==stereo) {
      throw std::invalid_argument("output directory escapes workspace stereo root");
    }
    const auto output = dir / (job.reference_name +
        (o.geom_consistency ? ".geometric.bin" : ".photometric.bin"));
    if (fs::exists(output) || fs::is_symlink(fs::symlink_status(output))) {
      throw std::invalid_argument("fresh outputs required (including partial results/symlinks)");
    }
    job.outputs[k] = fs::weakly_canonical(output);
    if (!pair::PathContains(dir,job.outputs[k])) throw std::invalid_argument("output path escapes map directory");
  }
  if (pair::PathsAlias(job.output_dirs[0],job.output_dirs[1])) {
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
        throw std::invalid_argument("invalid seed map header: " + path.string());
      }
      field.push_back(c);
    }
    if (!input || c != '&' || field.empty()) {
      throw std::invalid_argument("missing/invalid seed map header: " + path.string());
    }
    dimension = pair::ParseBytes(field);
  }
  if (dimensions != std::array<uint64_t, 3>{width, height, channels}) {
    throw std::invalid_argument("seed map/model dimensions mismatch: " + path.string());
  }
  const uint64_t payload = pair::CheckedMultiply(
      4, pair::CheckedMultiply(channels, pair::CheckedMultiply(width, height)));
  if (fs::file_size(path) != pair::CheckedAdd(
          static_cast<uint64_t>(input.tellg()), payload)) {
    throw std::invalid_argument("seed map payload size mismatch: " + path.string());
  }
}

// File-header/model-checked logical minima, not physical peak guarantees.
// Inputs must remain immutable after admission. Requiring the
// cache to retain all used grayscale inputs and geometry maps also avoids
// returning references after a concurrent cache eviction. No cache rewrite/pool.
void CheckInputs(const mvs::PatchMatchOptions& options,
                 const mvs::PatchMatch::Problem& problem,
                 pair::ByteCharge charge,
                 const fs::path& workspace) {
  const auto& images = *problem.images;
  const auto& ref = images.at(problem.ref_image_idx);
  const uint64_t a = pair::ReadPairBitmapPixels(
      ref.GetPath(),ref.GetWidth(),ref.GetHeight());
  const uint64_t s = problem.src_image_idxs.size();
  if (!a || !s) throw std::invalid_argument("empty reference/source dimensions");
  uint64_t q=a, max_w=0, max_h=0;
  std::set<int> used{problem.ref_image_idx};
  for (int idx : problem.src_image_idxs) {
    if (!used.insert(idx).second) throw std::invalid_argument("duplicate effective source");
    const auto& image=images.at(idx);
    if (!image.GetWidth() || !image.GetHeight()) throw std::invalid_argument("empty source dimensions");
    q=pair::CheckedAdd(q,pair::ReadPairBitmapPixels(
        image.GetPath(),image.GetWidth(),image.GetHeight()));
    max_w=std::max(max_w,static_cast<uint64_t>(image.GetWidth()));
    max_h=std::max(max_h,static_cast<uint64_t>(image.GetHeight()));
    if (!fs::is_regular_file(image.GetPath())) throw std::invalid_argument("source image missing");
  }
  const uint64_t cache=pair::CacheBytes(options.cache_size);
  if (options.geom_consistency) {
    for (const int idx : used) {
      const auto& image = images.at(idx);
      const auto name = image.GetPath().lexically_relative(workspace / "images").string();
      for (const auto* folder : {"depth_maps", "normal_maps"}) {
        CheckSeedMap(workspace / "stereo" / folder / (name + ".photometric.bin"),
                     image.GetWidth(), image.GetHeight(),
                     std::string(folder) == "depth_maps" ? 1 : 3);
      }
    }
  }
  const uint64_t retained = options.geom_consistency
      ? pair::CheckedMultiply(17, q) : q;
  if (cache<retained) throw std::invalid_argument("cache must retain complete TWO-job private input set per owner");
  const uint64_t u=pair::CheckedMultiply(s,pair::CheckedMultiply(max_w,max_h));
  // Local decoded inputs and the PatchMatch owned snapshot coexist during
  // construction: each retains grayscale q + depth 4q + normals 12q.
  // Source grayscale staging u and float-depth staging 4u are sequential;
  // preserve the existing 24a result/reference scratch allowance.
  const uint64_t staging = options.geom_consistency
      ? pair::CheckedMultiply(4, u) : u;
  const uint64_t host=pair::CheckedAdd(cache,pair::CheckedAdd(
      pair::CheckedMultiply(2,retained),
      std::max(staging,pair::CheckedMultiply(24,a))));
  // Persistent pitched layout plus source/pose arrays and both live reference
  // arrays during rebinding. This logical minimum is NOT a physical bound.
  uint64_t device=pair::CheckedAdd(pair::BuildOwnedRotationLayout(
      ref.GetWidth(),ref.GetHeight(),s).logical_bytes,u);
  if (options.geom_consistency) {
    device=pair::CheckedAdd(device,pair::CheckedMultiply(4,u));
  }
  device=pair::CheckedAdd(device,pair::CheckedMultiply(2,a));
  device=pair::CheckedAdd(device,pair::CheckedMultiply(688,s));
  device=pair::CheckedAdd(device,32);  // Retained calibration policy allowance.
  if (charge.host<host || charge.device<device) throw std::invalid_argument("job reservation below logical workload envelope");
}

template<class T> std::string Value(const T& value) {
  std::ostringstream s;s<<std::setprecision(std::numeric_limits<double>::max_digits10)<<value;return s.str();
}
void Metadata(pair::PairEvents& events, size_t job,
              const mvs::PatchMatchOptions& o,
              const mvs::PatchMatch::Problem& problem) {
  events.Identity(job,problem.ref_image_idx,static_cast<int>(pair::ParseBytes(o.gpu_index)));
  const auto& ref=problem.images->at(problem.ref_image_idx);
  events.Emit(job,"effective_metadata_begin","width="+Value(ref.GetWidth())+" height="+Value(ref.GetHeight())+
              " ref_path_hex="+pair::Hex(ref.GetPath().string())+" sources="+Value(problem.src_image_idxs.size()));
  for(size_t i=0;i<problem.src_image_idxs.size();++i) {
    const int idx=problem.src_image_idxs[i];
    events.Emit(job,"effective_source","ordinal="+Value(i)+" index="+Value(idx)+
                " path_hex="+pair::Hex(problem.images->at(idx).GetPath().string()));
  }
  size_t count=0;
#define OPTION(name) events.Emit(job,"effective_option","option=" #name " value_hex="+pair::Hex(Value(o.name)));++count
  OPTION(depth_min);OPTION(depth_max);OPTION(sigma_spatial);OPTION(sigma_color);
  OPTION(ncc_sigma);OPTION(min_triangulation_angle);OPTION(incident_angle_sigma);
  OPTION(geom_consistency_regularizer);OPTION(geom_consistency_max_cost);
  OPTION(filter_min_ncc);OPTION(filter_min_triangulation_angle);OPTION(filter_geom_consistency_max_cost);
  OPTION(cache_size);OPTION(gpu_index);OPTION(max_image_size);OPTION(window_radius);OPTION(window_step);
  OPTION(num_samples);OPTION(num_iterations);OPTION(filter_min_num_consistent);OPTION(num_threads);
  OPTION(geom_consistency);OPTION(filter);OPTION(allow_missing_files);OPTION(write_consistency_graph);
#undef OPTION
  events.Emit(job,"effective_metadata_end","option_count="+Value(count));
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
};

std::vector<fs::path> JobProjectPaths(
    const std::map<std::string, std::string>& args) {
  const auto list = args.find("job_project_list_path");
  if (list == args.end()) {
    std::vector<fs::path> paths{args.at("job_a_project_path"), args.at("job_b_project_path")};
    if (args.count("job_c_project_path")) paths.push_back(args.at("job_c_project_path"));
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
  if (!stream.eof() || paths.empty() || (args.count("static_three_solver_permits")
      ? paths.size() % 3 != 0 : paths.size() % 2 != 0)) {
    throw std::invalid_argument(args.count("static_three_solver_permits")
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
    ValidateJob(job, settings.charges[j % (settings.static_three_solver_permits ? 3 : 2)]);
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
    if (settings.static_three_solver_permits) {
      pair::ValidateIo(job.patch_match_stereo->num_threads,
                       job.patch_match_stereo->num_threads,
                       settings.io_cap, settings.affinity);
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
    events.Emit(pair::PairEvents::kBatch,"validated",
      "driver_executable=colmap_owned_pair"+(c ? " job_count=3 static_three_solver_permits="+Value(settings.static_three_solver_permits)+
      " io_c="+Value(c->patch_match_stereo->num_threads)+" hc="+Value(charges[2].host)+" dc="+Value(charges[2].device) : std::string{})+std::string(settings.geometry_early_permits ? " geometry_early_permits=1" : "")+" io_threads_limit="+Value(io_cap)+" affinity="+Value(affinity)+
      " io_a="+Value(jobs[0]->patch_match_stereo->num_threads)+" io_b="+Value(jobs[1]->patch_match_stereo->num_threads)+
      " host_budget="+Value(capacity.host)+" device_budget="+Value(capacity.device)+
      " hp="+Value(process.host)+" ha="+Value(charges[0].host)+" hb="+Value(charges[1].host)+
      " dp="+Value(process.device)+" da="+Value(charges[0].device)+" db="+Value(charges[1].device)+
      " default_random_seed="+Value(g.seed)+" log_target_hex="+pair::Hex(g.target)+
      " log_path_hex="+pair::Hex(g.path)+" log_level="+Value(g.level)+
      " log_severity="+Value(g.severity)+" log_color="+Value(g.color));
    std::unique_ptr<pair::PairCoordinator> coordinator;
    try {
      auto sink = [&](const pair::PairEvent& e) {
        events.Coordinator(e);
        if (e.event == "solver_release" && e.job == 0 && solver_released) {
          solver_released();
        }
      };
      const auto dispatch = settings.dispatch == "serial"
          ? pair::PairDispatch::kSerial : pair::PairDispatch::kPair;
      if (c) {
        coordinator = std::make_unique<pair::PairCoordinator>(dispatch,
            capacity, process, charges, std::chrono::milliseconds(300000),
            sink, settings.static_three_solver_permits);
      } else {
        const std::array<pair::ByteCharge, 2> two{{charges[0], charges[1]}};
        coordinator = std::make_unique<pair::PairCoordinator>(dispatch,
            capacity, process, two, std::chrono::milliseconds(300000), sink,
            settings.geometry_early_permits ? pair::EarlyGeometryPermits::kOn
                                            : pair::EarlyGeometryPermits::kOff);
      }
    } catch (...) {
      events.Emit(pair::PairEvents::kBatch,"admission_denied","reserved_host=0 reserved_device=0");
      throw;
    }
    const auto reserved = coordinator->ReservedBytes();
    events.Emit(pair::PairEvents::kBatch,"ledger_reserved",
                "reserved_host="+Value(reserved.host)+" reserved_device="+Value(reserved.device));
    if (admitted) admitted();
    mvs::PatchMatchController::OwnedPairHooks hooks;
    hooks.check_inputs=[&](size_t j,const auto& o,const auto& p){CheckInputs(o,p,charges[j],jobs[j]->workspace);};
    hooks.metadata=[&](size_t j,const auto& o,const auto& p){Metadata(events,j,o,p);};
    hooks.sweep=[&](size_t j,int iteration,int sweep){events.Emit(j,"sweep_complete","iteration="+Value(iteration)+" sweep="+Value(sweep));};
    hooks.output_phase=[&](size_t j,const char* phase){events.Emit(j,phase);};
    pair::RunPairOwners(*coordinator,[&](size_t job){
      events.Owner(job);
      run_job(*jobs[job], *coordinator, job, hooks);
    },[&](size_t job,const char* event){events.Emit(job,event);},
      pair::PairThreadFactory{}, [&](size_t job) {
        if (cleanup_inputs) cleanup_inputs(job);
      });
    const auto remaining = coordinator->ReservedBytes();
    if (remaining.host != 0 || remaining.device != 0) {
      throw std::logic_error("epoch finished with unreleased ledger credits");
    }
    events.Finished();
  } catch (const std::exception& e) {
    try {events.Emit(pair::PairEvents::kBatch,"batch_failed","status=failure error_hex="+pair::Hex(e.what()));}catch(...){}
    throw;
  } catch (...) {
    try {events.Emit(pair::PairEvents::kBatch,"batch_failed","status=failure error_hex="+pair::Hex("unknown exception"));}catch(...){}
    throw;
  }
}

using PreparedHostJob = mvs::PatchMatchController::OwnedPreparedJob;
using HostLookahead = pair::HostLookahead<PreparedHostJob>;

void HostLookaheadEvent(size_t target, bool enabled, const char* event,
                        int job, uint64_t reserved,
                        const char* reason = nullptr) {
  std::ostringstream line;
  line << "\nOWNED_HOST_LOOKAHEAD pid=" << getpid()
       << " monotonic_ns=" << pair::PairEvents::NowNs()
       << " target_epoch=" << target << " event=" << event
       << " enabled=" << (enabled ? 1 : 0)
       << " reserved_host=" << reserved;
  if (job >= 0) line << " job=" << (job == 0 ? "A" : "B");
  if (reason) line << " reason=" << reason;
  line << " io_threads=" << (enabled ? 1 : 0) << '\n';
  const auto text = line.str();
  if (text.size() > 4096) throw std::length_error("lookahead event too long");
  ssize_t written;
  do { written = write(STDOUT_FILENO, text.data(), text.size()); }
  while (written < 0 && errno == EINTR);
  if (written < 0 || static_cast<size_t>(written) != text.size()) {
    throw std::runtime_error("lookahead event write failed");
  }
}

std::unique_ptr<PreparedHostJob> PrepareFutureHostJob(
    JobOptions& job, size_t slot, pair::ByteCharge charge,
    const std::atomic<bool>& cancelled) {
  if (job.patch_match_stereo->geom_consistency) {
    throw std::invalid_argument("host lookahead is photometric-only");
  }
  mvs::PatchMatchController controller(*job.patch_match_stereo, job.workspace,
      job.format, job.pmvs, job.config, mvs::PatchMatchExecution::kOwnedCuda);
  mvs::PatchMatchController::OwnedPairHooks hooks;
  hooks.check_inputs = [&](size_t, const auto& options, const auto& problem) {
    if (cancelled.load(std::memory_order_relaxed)) {
      throw std::runtime_error("host preparation cancelled");
    }
    CheckInputs(options, problem, charge, job.workspace);
  };
  return controller.PrepareOwnedPairJob(slot, hooks, 1);
}

template <class RunJob, class PrepareJob = decltype(&PrepareFutureHostJob)>
int RunPairs(int argc, char** argv, RunJob run_job,
             PrepareJob prepare_job = &PrepareFutureHostJob) {
  std::set<std::string> keys = {
      "dispatch", "host_budget_bytes", "device_budget_bytes",
      "job_a_host_charge_bytes", "job_a_device_charge_bytes",
      "job_b_host_charge_bytes", "job_b_device_charge_bytes",
      "process_host_charge_bytes", "process_device_charge_bytes",
      "io_threads_limit"};
  if (argc == 2 && (std::string(argv[1]) == "-h" ||
                    std::string(argv[1]) == "--help")) {
    std::cout << "Finite even owned references; --dispatch serial|pair "
                 "(SOLVER serial only). Required:\n";
    for (const auto& key : keys) std::cout << "  --" << key << " <value>\n";
    std::cout << "  --job_a_project_path <INI> --job_b_project_path <INI>\n"
                 "OR --job_project_list_path <list: one INI path per line, "
                 "relative to list directory or absolute>\n"
                 "Optional: --host_lookahead 0|1 (default 0) "
                 "--host_lookahead_charge_bytes <separate host reservation>\n"
                 "  --geometry_early_permits 0|1 (experimental, default 0; "
                 "geometry-only, two owners per closed epoch, lookahead OFF; "
                 "SERIAL unchanged)\n"
                 "  --static_three_solver_permits 2|3 (closed geometry-only triples; pair dispatch; no lookahead)\n"
                 "  --job_c_project_path <INI> (or positive multiple-of-three list)\n"
                 "  --job_c_host_charge_bytes <bytes> --job_c_device_charge_bytes <bytes>\n";
    return EXIT_SUCCESS;
  }
  bool list_mode = false;
  for (int i = 1; i < argc; i += 2) {
    list_mode = list_mode || std::string(argv[i]) == "--job_project_list_path";
    if (std::string(argv[i]) == "--static_three_solver_permits") {
      keys.insert("static_three_solver_permits");
      keys.insert("job_c_host_charge_bytes");
      keys.insert("job_c_device_charge_bytes");
    }
    if (std::string(argv[i]) == "--geometry_early_permits") {
      keys.insert("geometry_early_permits");
    }
    if (std::string(argv[i]) == "--host_lookahead") keys.insert("host_lookahead");
    if (std::string(argv[i]) == "--host_lookahead_charge_bytes") {
      keys.insert("host_lookahead_charge_bytes");
    }
  }
  if (list_mode) {
    keys.insert("job_project_list_path");
  } else {
    keys.insert("job_a_project_path");
    keys.insert("job_b_project_path");
    if (keys.count("static_three_solver_permits")) keys.insert("job_c_project_path");
  }
  // Selecting one exact key set also rejects mixed legacy/list arguments.
  const auto args = pair::ParseStrictArguments(argc, argv, keys);
  if (args.at("dispatch") != "serial" && args.at("dispatch") != "pair") {
    throw std::invalid_argument("invalid dispatch");
  }
  if (args.count("geometry_early_permits") &&
      args.at("geometry_early_permits") != "0" &&
      args.at("geometry_early_permits") != "1") {
    throw std::invalid_argument("geometry_early_permits must be 0 or 1");
  }
  size_t static_three_permits = 0;
  if (args.count("static_three_solver_permits")) {
    if (args.at("static_three_solver_permits") != "2" &&
        args.at("static_three_solver_permits") != "3") {
      throw std::invalid_argument("static_three_solver_permits must be 2 or 3");
    }
    static_three_permits = pair::ParseBytes(args.at("static_three_solver_permits"));
    if (args.at("dispatch") != "pair" || args.count("host_lookahead") ||
        args.count("host_lookahead_charge_bytes") || args.count("geometry_early_permits")) {
      throw std::invalid_argument("static three requires pair, no lookahead/other permit flags");
    }
  }
  const bool early_permits_requested = args.count("geometry_early_permits") &&
                                      args.at("geometry_early_permits") == "1";
  const bool lookahead_requested = args.count("host_lookahead") &&
                                   args.at("host_lookahead") == "1";
  if (args.count("host_lookahead") && args.at("host_lookahead") != "0" &&
      args.at("host_lookahead") != "1") {
    throw std::invalid_argument("host_lookahead must be 0 or 1");
  }
  const uint64_t lookahead_charge = args.count("host_lookahead_charge_bytes")
      ? pair::ParseBytes(args.at("host_lookahead_charge_bytes")) : 0;
  if (lookahead_requested && lookahead_charge == 0) {
    throw std::invalid_argument("lookahead requires separate positive host charge");
  }
  const auto bytes = [&](const char* key) { return pair::ParseBytes(args.at(key)); };
  const PairSettings settings{
      args.at("dispatch"),
      {bytes("host_budget_bytes"), bytes("device_budget_bytes")},
      {bytes("process_host_charge_bytes"), bytes("process_device_charge_bytes")},
      {{{bytes("job_a_host_charge_bytes"), bytes("job_a_device_charge_bytes")},
        {bytes("job_b_host_charge_bytes"), bytes("job_b_device_charge_bytes")},
        {static_three_permits ? bytes("job_c_host_charge_bytes") : 0,
         static_three_permits ? bytes("job_c_device_charge_bytes") : 0}}},
      AffinityCount(), bytes("io_threads_limit"), early_permits_requested, static_three_permits};
  BaseOptionManager globals(false);
  auto jobs = LoadJobs(JobProjectPaths(args), settings);
  if (static_three_permits &&
      (jobs.empty() || jobs.size() % 3 != 0 ||
       !jobs.front()->patch_match_stereo->geom_consistency ||
       settings.affinity < 3)) {
    throw std::invalid_argument("static three requires complete geometry triples and three CPU equivalents");
  }
  if (early_permits_requested &&
      (!jobs.front()->patch_match_stereo->geom_consistency || lookahead_requested)) {
    throw std::invalid_argument(
        "early geometry permits require geometry jobs and lookahead OFF");
  }
  if (lookahead_requested && jobs.front()->patch_match_stereo->geom_consistency) {
    throw std::invalid_argument("geometry host lookahead is unsupported");
  }
  const auto& g = jobs.front()->globals;
  std::vector<std::string> common = {
      "colmap_owned_pair", "--default_random_seed", Value(g.seed),
      "--log_target", g.target, "--log_path", g.path,
      "--log_level", Value(g.level), "--log_severity", Value(g.severity)};
#if COLMAP_GLOG_HAS_COLOR_SUPPORT
  common.insert(common.end(), {"--log_color", Value(g.color)});
#endif
  std::vector<char*> common_argv;
  for (auto& value : common) common_argv.push_back(value.data());
  if (!globals.Parse(static_cast<int>(common_argv.size()), common_argv.data())) {
    throw std::invalid_argument("invalid global settings");
  }
  if (static_three_permits) {
    auto execute = [&](JobOptions& job, pair::PairCoordinator& coordinator,
                       size_t slot, const auto& hooks) {
      if constexpr (std::is_invocable_v<RunJob&, JobOptions&,
                        pair::PairCoordinator&, size_t, decltype(hooks),
                        std::unique_ptr<PreparedHostJob>>) {
        run_job(job, coordinator, slot, hooks, nullptr);
      } else {
        run_job(job, coordinator, slot, hooks);
      }
    };
    for (size_t first = 0; first < jobs.size(); first += 3) {
      // Each call joins and retires all owners, checks Finish credits, and
      // destroys the coordinator before the next independent admission.
      RunPairEpoch(*jobs[first], *jobs[first + 1], settings, execute,
                   {}, {}, {}, jobs[first + 2].get());
    }
    return EXIT_SUCCESS;
  }
  // The full additional transient charge is held even after CPU preparation
  // completes. Never lend current owner/process credits to future snapshots.
  const bool host_room = pair::CanReserveHostLookahead(
      settings.capacity.host, settings.process.host, settings.charges[0].host,
      settings.charges[1].host, lookahead_charge);
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
    auto execute = [&](JobOptions& job, pair::PairCoordinator& coordinator,
                       size_t slot, const auto& hooks) {
      if constexpr (std::is_invocable_v<RunJob&, JobOptions&,
                        pair::PairCoordinator&, size_t, decltype(hooks),
                        std::unique_ptr<PreparedHostJob>>) {
        run_job(job, coordinator, slot, hooks, std::move(prepared[slot]));
      } else {
        if (prepared[slot]) throw std::logic_error("caller cannot consume snapshot");
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
        HostLookaheadEvent(target, false, "fallback", -1, 0,
                           host_room ? "cpu_io_budget" : "host_budget");
        return;
      }
      next = std::make_unique<HostLookahead>(lookahead_charge,
          [target](const char* event, int slot, uint64_t reserved) {
            HostLookaheadEvent(target, true, event, slot, reserved);
          });
      next->Start([&, first](size_t slot, const std::atomic<bool>& cancelled) {
        return prepare_job(*jobs[first + 2 + slot], slot,
                           settings.charges[slot], cancelled);
      });
    };
    RunPairEpoch(*jobs[first], *jobs[first + 1], settings, execute,
                 admitted, solver_released,
                 [&](size_t slot) { prepared[slot].reset(); });
  }
  return EXIT_SUCCESS;
}

int RunPair(int argc, char** argv) {
  return RunPairs(argc, argv, [](JobOptions& job,
                                 pair::PairCoordinator& coordinator,
                                 size_t slot,
                                 const mvs::PatchMatchController::OwnedPairHooks& hooks,
                                 std::unique_ptr<PreparedHostJob> prepared) {
    mvs::PatchMatchController controller(*job.patch_match_stereo, job.workspace,
        job.format, job.pmvs, job.config, mvs::PatchMatchExecution::kOwnedCuda);
    if (prepared) {
      controller.RunPreparedOwnedPairJob(std::move(prepared), coordinator, slot, hooks);
    } else {
      controller.RunOwnedPairJob(coordinator, slot, hooks);
    }
  });
}
}  // namespace
}  // namespace colmap

int main(int argc,char** argv) {
  colmap::InitializeGlog(argv);
  try {return colmap::RunPair(argc,argv);}
  catch(const std::exception& e){LOG(ERROR)<<e.what();return EXIT_FAILURE;}
  catch(...){LOG(ERROR)<<"Unknown owned pair failure";return EXIT_FAILURE;}
}
