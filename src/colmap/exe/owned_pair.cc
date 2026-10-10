// SPDX-License-Identifier: BSD-3-Clause
// Optional finite TWO-owner experiment. Never a production queue scheduler.
#include "colmap/controllers/option_manager.h"
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <tuple>
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
  if (job.format != "colmap" || o.geom_consistency || o.window_radius != 5 ||
      o.window_step != 1 || o.num_iterations != 5 || o.max_image_size != -1 ||
      o.allow_missing_files || o.write_consistency_graph || !o.filter) {
    throw std::invalid_argument("TWO prototype requires native COLMAP photo11/1, five iterations, filtered, complete inputs, no graph");
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
    const auto output = dir / (job.reference_name + ".photometric.bin");
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

// File-header/model-checked logical minima, not physical peak guarantees.
// Inputs must remain immutable after admission. Requiring the
// cache to retain the entire used grayscale input set also avoids returning a
// bitmap reference after a concurrent cache eviction. No cache rewrite/pool.
void CheckInputs(const mvs::PatchMatchOptions& options,
                 const mvs::PatchMatch::Problem& problem,
                 pair::ByteCharge charge) {
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
  if (cache<q) throw std::invalid_argument("cache must retain complete TWO-job private input set per owner");
  const uint64_t u=pair::CheckedMultiply(s,pair::CheckedMultiply(max_w,max_h));
  const uint64_t host=pair::CheckedAdd(cache,pair::CheckedAdd(pair::CheckedMultiply(2,q),
                        std::max(u,pair::CheckedMultiply(24,a))));
  // Persistent pitched layout plus source/pose arrays and both live reference
  // arrays during rebinding. This logical minimum is NOT a physical bound.
  uint64_t device=pair::CheckedAdd(pair::BuildOwnedRotationLayout(
      ref.GetWidth(),ref.GetHeight(),s).logical_bytes,u);
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

int RunPair(int argc,char** argv) {
  const std::set<std::string> keys={"dispatch","job_a_project_path","job_b_project_path",
    "host_budget_bytes","device_budget_bytes","job_a_host_charge_bytes","job_a_device_charge_bytes",
    "job_b_host_charge_bytes","job_b_device_charge_bytes","process_host_charge_bytes","process_device_charge_bytes",
    "io_threads_limit"};
  if(argc==2 && (std::string(argv[1])=="-h" || std::string(argv[1])=="--help")) {
    std::cout<<"Finite TWO owned references; --dispatch serial|pair (SOLVER serial only). Required:\n";
    for(const auto& key:keys)std::cout<<"  --"<<key<<" <value>\n";
    return EXIT_SUCCESS;
  }
  const auto args=pair::ParseStrictArguments(argc,argv,keys);
  if(args.at("dispatch")!="serial" && args.at("dispatch")!="pair") throw std::invalid_argument("invalid dispatch");
  pair::PairEvents events(args.at("dispatch"));
  try {
    BaseOptionManager globals(false);
    std::array<JobOptions,2> jobs;
    if(!jobs[0].Read(args.at("job_a_project_path"),false) || !jobs[1].Read(args.at("job_b_project_path"),false)) {
      throw std::invalid_argument("invalid job INI");
    }
    if(!(jobs[0].globals==jobs[1].globals)) throw std::invalid_argument("conflicting process-global settings");
    auto bytes=[&](const char* key){return pair::ParseBytes(args.at(key));};
    const pair::ByteCharge capacity{bytes("host_budget_bytes"),bytes("device_budget_bytes")};
    const pair::ByteCharge process{bytes("process_host_charge_bytes"),bytes("process_device_charge_bytes")};
    const std::array<pair::ByteCharge,2> charges{{
      {bytes("job_a_host_charge_bytes"),bytes("job_a_device_charge_bytes")},
      {bytes("job_b_host_charge_bytes"),bytes("job_b_device_charge_bytes")}}};
    const uint64_t affinity=AffinityCount(),io_cap=bytes("io_threads_limit");
    pair::ValidateIo(jobs[0].patch_match_stereo->num_threads,jobs[1].patch_match_stereo->num_threads,io_cap,affinity);
    for(size_t j=0;j<2;++j)ValidateJob(jobs[j],charges[j]);
    if(jobs[0].reference_name==jobs[1].reference_name ||
       pair::ParseBytes(jobs[0].patch_match_stereo->gpu_index)!=pair::ParseBytes(jobs[1].patch_match_stereo->gpu_index)) {
      throw std::invalid_argument("duplicate reference or mixed device");
    }
    if(pair::PathsAlias(jobs[0].workspace,jobs[1].workspace)) throw std::invalid_argument("workspace/output roots alias");
    for(const auto& a:jobs[0].output_dirs)for(const auto& b:jobs[1].output_dirs) {
      if(pair::PathsAlias(a,b)) throw std::invalid_argument("resolved output directories alias");
    }
    const auto& g=jobs[0].globals;
    std::vector<std::string> common={"colmap_owned_pair","--default_random_seed",Value(g.seed),"--log_target",g.target,
      "--log_path",g.path,"--log_level",Value(g.level),"--log_severity",Value(g.severity)};
#if COLMAP_GLOG_HAS_COLOR_SUPPORT
    common.insert(common.end(),{"--log_color",Value(g.color)});
#endif
    std::vector<char*> common_argv;for(auto& value:common)common_argv.push_back(value.data());
    if(!globals.Parse(static_cast<int>(common_argv.size()),common_argv.data())) throw std::invalid_argument("invalid global settings");
    events.Emit(pair::PairEvents::kBatch,"validated",
      "driver_executable=colmap_owned_pair io_threads_limit="+Value(io_cap)+" affinity="+Value(affinity)+
      " io_a="+Value(jobs[0].patch_match_stereo->num_threads)+" io_b="+Value(jobs[1].patch_match_stereo->num_threads)+
      " host_budget="+Value(capacity.host)+" device_budget="+Value(capacity.device)+
      " hp="+Value(process.host)+" ha="+Value(charges[0].host)+" hb="+Value(charges[1].host)+
      " dp="+Value(process.device)+" da="+Value(charges[0].device)+" db="+Value(charges[1].device)+
      " default_random_seed="+Value(g.seed)+" log_target_hex="+pair::Hex(g.target)+
      " log_path_hex="+pair::Hex(g.path)+" log_level="+Value(g.level)+
      " log_severity="+Value(g.severity)+" log_color="+Value(g.color));
    std::unique_ptr<pair::PairCoordinator> coordinator;
    try {
      coordinator=std::make_unique<pair::PairCoordinator>(
        args.at("dispatch")=="serial"?pair::PairDispatch::kSerial:pair::PairDispatch::kPair,
        capacity,process,charges,std::chrono::milliseconds(300000),
        [&](const pair::PairEvent& e){events.Coordinator(e);});
    } catch (...) {
      events.Emit(pair::PairEvents::kBatch,"admission_denied","reserved_host=0 reserved_device=0");
      throw;
    }
    const auto reserved = coordinator->ReservedBytes();
    events.Emit(pair::PairEvents::kBatch,"ledger_reserved",
                "reserved_host="+Value(reserved.host)+" reserved_device="+Value(reserved.device));
    mvs::PatchMatchController::OwnedPairHooks hooks;
    hooks.check_inputs=[&](size_t j,const auto& o,const auto& p){CheckInputs(o,p,charges[j]);};
    hooks.metadata=[&](size_t j,const auto& o,const auto& p){Metadata(events,j,o,p);};
    hooks.sweep=[&](size_t j,int iteration,int sweep){events.Emit(j,"sweep_complete","iteration="+Value(iteration)+" sweep="+Value(sweep));};
    hooks.output_phase=[&](size_t j,const char* phase){events.Emit(j,phase);};
    pair::RunPairOwners(*coordinator,[&](size_t job){
      events.Owner(job);
      mvs::PatchMatchController controller(*jobs[job].patch_match_stereo,jobs[job].workspace,
          jobs[job].format,jobs[job].pmvs,jobs[job].config,mvs::PatchMatchExecution::kOwnedCuda);
      controller.RunOwnedPairJob(*coordinator,job,hooks);
    },[&](size_t job,const char* event){events.Emit(job,event);});
    events.Finished();
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    try {events.Emit(pair::PairEvents::kBatch,"batch_failed","status=failure error_hex="+pair::Hex(e.what()));}catch(...){}
    throw;
  } catch (...) {
    try {events.Emit(pair::PairEvents::kBatch,"batch_failed","status=failure error_hex="+pair::Hex("unknown exception"));}catch(...){}
    throw;
  }
}
}  // namespace
}  // namespace colmap

int main(int argc,char** argv) {
  colmap::InitializeGlog(argv);
  try {return colmap::RunPair(argc,argv);}
  catch(const std::exception& e){LOG(ERROR)<<e.what();return EXIT_FAILURE;}
  catch(...){LOG(ERROR)<<"Unknown owned pair failure";return EXIT_FAILURE;}
}
