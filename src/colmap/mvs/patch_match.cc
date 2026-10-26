// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"

#include "colmap/math/math.h"
#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/optimized_photo_paths.h"
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_input.h"
#include "colmap/mvs/owned_pair_output.h"
#include "colmap/mvs/patch_match_cuda.h"
#include "colmap/mvs/resident_contract.h"
#if !defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/patch_match_cuda_owned.h"
#endif
#include "colmap/mvs/workspace.h"
#include "colmap/util/cuda.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <set>
#include <stdexcept>

#define PrintOption(option) LOG(INFO) << #option ": " << option << std::endl

namespace colmap {
namespace mvs {

namespace {
// Shared by real input preparation, complete-bank capture, and the planned
// graph. Insertion and single iteration MUST NOT be sorted or repeated.
FlatHashSet<int> ResolveNativeUsedImages(const std::vector<int>& sources,
                                         int reference,
                                         PatchMatchOptions* options) {
  FlatHashSet<int> used(sources.begin(), sources.end());
  used.insert(reference);
  options->filter_min_num_consistent = std::min(
      static_cast<int>(used.size()) - 1, options->filter_min_num_consistent);
  return used;
}
std::vector<Image> SnapshotImages(const PatchMatch::Problem& problem) {
  if (!problem.images) return {};
  if (!problem.resident_bank) return *problem.images;
  std::vector<Image> images;
  images.reserve(problem.images->size());
  for (const auto& image : *problem.images) {
    images.emplace_back(image.GetPath(),
                        image.GetWidth(),
                        image.GetHeight(),
                        image.GetK(),
                        image.GetR(),
                        image.GetT());
  }
  return images;
}
PatchMatchExecution ValidateExecution(const PatchMatchOptions& options,
                                      const PatchMatchExecution execution) {
#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
  constexpr bool kOwnedCudaAvailable = true;
#else
  constexpr bool kOwnedCudaAvailable = false;
#endif
  return ValidatePatchMatchExecution(execution,
                                     options.window_radius,
                                     options.window_step,
                                     options.geom_consistency,
                                     kOwnedCudaAvailable,
                                     options.filter,
                                     options.write_consistency_graph,
                                     options.allow_missing_files);
}
}  // namespace

PatchMatch::PatchMatch(const PatchMatchOptions& options,
                       const Problem& problem,
                       const PatchMatchExecution execution)
    : options_(options),
      execution_(ValidateExecution(options, execution)),
      owned_images_(execution_ == PatchMatchExecution::kOwnedCuda &&
                            problem.images
                        ? SnapshotImages(problem)
                        : std::vector<Image>{}),
      owned_depth_maps_(execution_ == PatchMatchExecution::kOwnedCuda &&
                                options_.geom_consistency &&
                                !problem.resident_bank && problem.depth_maps
                            ? *problem.depth_maps
                            : std::vector<DepthMap>{}),
      owned_normal_maps_(execution_ == PatchMatchExecution::kOwnedCuda &&
                                 options_.geom_consistency &&
                                 !problem.resident_bank && problem.normal_maps
                             ? *problem.normal_maps
                             : std::vector<NormalMap>{}),
      problem_([&] {
        Problem result = problem;
        if (execution_ == PatchMatchExecution::kOwnedCuda) {
          result.images = &owned_images_;
          result.depth_maps = options_.geom_consistency &&
                                      !problem.resident_bank &&
                                      problem.depth_maps
                                  ? &owned_depth_maps_
                                  : nullptr;
          result.normal_maps = options_.geom_consistency &&
                                       !problem.resident_bank &&
                                       problem.normal_maps
                                   ? &owned_normal_maps_
                                   : nullptr;
        }
        return result;
      }()) {}

PatchMatch::~PatchMatch() {}

void PatchMatch::Problem::Print() const {
  LOG_HEADING2("PatchMatch::Problem");
  LOG(INFO) << "ref_image_idx: " << ref_image_idx;
  THROW_CHECK(!src_image_idxs.empty());
  std::ostringstream src_image_idxs_stream;
  for (size_t i = 0; i < src_image_idxs.size() - 1; ++i) {
    src_image_idxs_stream << src_image_idxs[i] << " ";
  }
  src_image_idxs_stream << src_image_idxs.back();
  LOG(INFO) << "src_image_idxs: " << src_image_idxs_stream.str();
}

void PatchMatch::Check() const {
  THROW_CHECK(options_.Check());

  THROW_CHECK(!options_.gpu_index.empty());
  const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
  THROW_CHECK_EQ(gpu_indices.size(), 1);
  THROW_CHECK_GE(gpu_indices[0], -1);

  THROW_CHECK_NOTNULL(problem_.images);
  if (options_.geom_consistency && !problem_.resident_bank) {
    THROW_CHECK_NOTNULL(problem_.depth_maps);
    THROW_CHECK_NOTNULL(problem_.normal_maps);
    THROW_CHECK_EQ(problem_.depth_maps->size(), problem_.images->size());
    THROW_CHECK_EQ(problem_.normal_maps->size(), problem_.images->size());
  }

  THROW_CHECK_GT(problem_.src_image_idxs.size(), 0);

  // Check that there are no duplicate images and that the reference image
  // is not defined as a source image.
  std::set<int> unique_image_idxs(problem_.src_image_idxs.begin(),
                                  problem_.src_image_idxs.end());
  unique_image_idxs.insert(problem_.ref_image_idx);
  THROW_CHECK_EQ(problem_.src_image_idxs.size() + 1, unique_image_idxs.size());

  // Check that input data is well-formed.
  for (const int image_idx : unique_image_idxs) {
    THROW_CHECK_GE(image_idx, 0) << image_idx;
    THROW_CHECK_LT(image_idx, problem_.images->size()) << image_idx;

    const Image& image = problem_.images->at(image_idx);
    if (problem_.resident_bank) {
#if defined(COLMAP_HIP_ENABLED)
      THROW_CHECK(false)
          << "Resident input banks require the owned CUDA backend";
#else
      THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
      THROW_CHECK(options_.geom_consistency);
      THROW_CHECK_EQ(gpu_indices[0], problem_.resident_bank->DeviceIndex());
      THROW_CHECK_EQ(problem_.resident_src_layers.size(),
                     problem_.src_image_idxs.size());
      int layer = problem_.resident_ref_layer;
      if (image_idx != problem_.ref_image_idx) {
        const auto it = std::find(problem_.src_image_idxs.begin(),
                                  problem_.src_image_idxs.end(),
                                  image_idx);
        layer = problem_.resident_src_layers.at(
            std::distance(problem_.src_image_idxs.begin(), it));
      }
      THROW_CHECK_GE(layer, 0);
      THROW_CHECK_LT(layer, problem_.resident_bank->NumLayers());
      THROW_CHECK_GT(image.GetWidth(), 0);
      THROW_CHECK_GT(image.GetHeight(), 0);
      THROW_CHECK_EQ(image.GetWidth(),
                     problem_.resident_bank->LayerWidth(layer));
      THROW_CHECK_EQ(image.GetHeight(),
                     problem_.resident_bank->LayerHeight(layer));
      THROW_CHECK(image.GetBitmap().IsEmpty());
#endif
    } else {
      THROW_CHECK_GT(image.GetBitmap().Width(), 0) << image_idx;
      THROW_CHECK_GT(image.GetBitmap().Height(), 0) << image_idx;
      THROW_CHECK(image.GetBitmap().IsGrey()) << image_idx;
      THROW_CHECK_EQ(image.GetWidth(), image.GetBitmap().Width()) << image_idx;
      THROW_CHECK_EQ(image.GetHeight(), image.GetBitmap().Height())
          << image_idx;
    }

    // Make sure, the calibration matrix only contains fx, fy, cx, cy.
    THROW_CHECK_LT(std::abs(image.GetK()[1] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[3] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[6] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[7] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[8] - 1.0f), 1e-6f) << image_idx;

    if (options_.geom_consistency && !problem_.resident_bank) {
      THROW_CHECK_LT(image_idx, problem_.depth_maps->size()) << image_idx;
      const DepthMap& depth_map = problem_.depth_maps->at(image_idx);
      THROW_CHECK_EQ(image.GetWidth(), depth_map.GetWidth()) << image_idx;
      THROW_CHECK_EQ(image.GetHeight(), depth_map.GetHeight()) << image_idx;
      if (execution_ == PatchMatchExecution::kOwnedCuda) {
        const NormalMap& normal_map = problem_.normal_maps->at(image_idx);
        THROW_CHECK_EQ(image.GetWidth(), normal_map.GetWidth()) << image_idx;
        THROW_CHECK_EQ(image.GetHeight(), normal_map.GetHeight()) << image_idx;
      }
    }
  }

  if (options_.geom_consistency) {
    const Image& ref_image = problem_.images->at(problem_.ref_image_idx);
    if (problem_.resident_bank) {
      THROW_CHECK_NOTNULL(problem_.resident_ref_normal);
      THROW_CHECK_EQ(problem_.resident_ref_normal->GetDepth(), 3);
    }
    const NormalMap& ref_normal_map =
        problem_.resident_bank
            ? *problem_.resident_ref_normal
            : problem_.normal_maps->at(problem_.ref_image_idx);
    THROW_CHECK_EQ(ref_image.GetWidth(), ref_normal_map.GetWidth());
    THROW_CHECK_EQ(ref_image.GetHeight(), ref_normal_map.GetHeight());
  }
}

void PatchMatch::Run() {
  LOG_HEADING2("PatchMatch::Run");

  Check();

#if !defined(COLMAP_HIP_ENABLED)
  if (execution_ == PatchMatchExecution::kOwnedCuda) {
    PrepareOwned();
    RunPreparedOwned();
    return;
  }
#endif
  patch_match_cuda_ = std::make_unique<PatchMatchCuda>(options_, problem_);
  patch_match_cuda_->Run();
}

void PatchMatch::PrepareOwned(
    const int owned_tile_threads,
    std::shared_ptr<owned::GraphLaunchControl> graph_coordinator,
    size_t graph_owner) {
#if !defined(COLMAP_HIP_ENABLED)
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  THROW_CHECK(!owned_patch_match_cuda_) << "Owned instances prepare only once";
  Check();
  owned_patch_match_cuda_ = std::make_unique<owned::PatchMatchCuda>(
      options_, problem_, owned_tile_threads);
  owned_patch_match_cuda_->SetGraphCoordinator(std::move(graph_coordinator),
                                               graph_owner);
  LOG(INFO) << "PATCH_MATCH_BACKEND actual=owned_cuda reference="
            << problem_.ref_image_idx;
#else
  THROW_CHECK(false) << "Owned lifecycle requires CUDA, not HIP";
#endif
}

void PatchMatch::RunPreparedOwned() {
#if !defined(COLMAP_HIP_ENABLED)
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  THROW_CHECK_NOTNULL(owned_patch_match_cuda_.get());
  THROW_CHECK(!owned_run_started_) << "Owned instances run only once";
  owned_run_started_ = true;
  owned_patch_match_cuda_->Run();
  owned_run_completed_ = true;
#else
  THROW_CHECK(false) << "Owned lifecycle requires CUDA, not HIP";
#endif
}

void PatchMatch::DrainOwned() noexcept {
#if !defined(COLMAP_HIP_ENABLED)
  if (owned_patch_match_cuda_) {
    owned_patch_match_cuda_->DrainForCleanup();
  }
#endif
}

void PatchMatch::SetOwnedSweepObserver(std::function<void(int, int)> observer) {
#if !defined(COLMAP_HIP_ENABLED)
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  THROW_CHECK_NOTNULL(owned_patch_match_cuda_.get());
  THROW_CHECK(!owned_run_started_);
  owned_patch_match_cuda_->SetSweepObserver(std::move(observer));
#else
  THROW_CHECK(false) << "Owned lifecycle requires CUDA, not HIP";
#endif
}

DepthMap PatchMatch::GetDepthMap() const {
#if !defined(COLMAP_HIP_ENABLED)
  if (execution_ == PatchMatchExecution::kOwnedCuda) {
    THROW_CHECK_NOTNULL(owned_patch_match_cuda_.get());
    THROW_CHECK(owned_run_completed_) << "Owned result is not complete";
    return owned_patch_match_cuda_->GetDepthMap();
  }
#endif
  return patch_match_cuda_->GetDepthMap();
}

NormalMap PatchMatch::GetNormalMap() const {
#if !defined(COLMAP_HIP_ENABLED)
  if (execution_ == PatchMatchExecution::kOwnedCuda) {
    THROW_CHECK_NOTNULL(owned_patch_match_cuda_.get());
    THROW_CHECK(owned_run_completed_) << "Owned result is not complete";
    return owned_patch_match_cuda_->GetNormalMap();
  }
#endif
  return patch_match_cuda_->GetNormalMap();
}

Mat<float> PatchMatch::GetSelProbMap() const {
#if !defined(COLMAP_HIP_ENABLED)
  if (execution_ == PatchMatchExecution::kOwnedCuda) {
    THROW_CHECK_NOTNULL(owned_patch_match_cuda_.get());
    THROW_CHECK(owned_run_completed_) << "Owned result is not complete";
    return owned_patch_match_cuda_->GetSelProbMap();
  }
#endif
  return patch_match_cuda_->GetSelProbMap();
}

ConsistencyGraph PatchMatch::GetConsistencyGraph() const {
  const auto& ref_image = problem_.images->at(problem_.ref_image_idx);
#if !defined(COLMAP_HIP_ENABLED)
  if (execution_ == PatchMatchExecution::kOwnedCuda) {
    THROW_CHECK_NOTNULL(owned_patch_match_cuda_.get());
    THROW_CHECK(owned_run_completed_) << "Owned result is not complete";
    return ConsistencyGraph(ref_image.GetWidth(),
                            ref_image.GetHeight(),
                            owned_patch_match_cuda_->GetConsistentImageIdxs());
  }
#endif
  return ConsistencyGraph(ref_image.GetWidth(),
                          ref_image.GetHeight(),
                          patch_match_cuda_->GetConsistentImageIdxs());
}

PatchMatchController::PatchMatchController(
    const PatchMatchOptions& options,
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const std::filesystem::path& config_path,
    const PatchMatchExecution execution)
    : options_(options),
      execution_(ValidateExecution(options, execution)),
      workspace_path_(workspace_path),
      workspace_format_(workspace_format),
      pmvs_option_name_(pmvs_option_name),
      config_path_(config_path) {
  std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
}

void PatchMatchController::Run() {
  Timer run_timer;
  run_timer.Start();
  ReadWorkspace();
  ReadProblems();
  if (execution_ == PatchMatchExecution::kOwnedCuda) {
    THROW_CHECK_EQ(problems_.size(), 1)
        << "Owned prototype accepts exactly one configured reference";
  }
  ReadGpuIndices();

  thread_pool_ = std::make_unique<ThreadPool>(gpu_indices_.size());
  io_thread_pool_ = std::make_unique<ThreadPool>(
      GetEffectiveNumThreads(options_.num_threads));

  // Stock computes (or reuses) photometric output first. Owned N1 geometry
  // requires existing seeds and transforms source order only in its geom pass.
  if (options_.geom_consistency && execution_ == PatchMatchExecution::kStock) {
    auto photometric_options = options_;
    photometric_options.geom_consistency = false;
    photometric_options.filter = false;

    for (size_t problem_idx = 0; problem_idx < problems_.size();
         ++problem_idx) {
      thread_pool_->AddTask(&PatchMatchController::ProcessProblem,
                            this,
                            photometric_options,
                            problem_idx);
    }

    thread_pool_->Wait();
  }

  for (size_t problem_idx = 0; problem_idx < problems_.size(); ++problem_idx) {
    thread_pool_->AddTask(
        &PatchMatchController::ProcessProblem, this, options_, problem_idx);
  }

  thread_pool_->Wait();

  run_timer.PrintMinutes();
}

PatchMatchController::DensePlanGraph
PatchMatchController::ResolveDensePlanGraph() {
  THROW_CHECK(execution_ == PatchMatchExecution::kStock);
  THROW_CHECK(!owned_bank_resolved_ && !owned_pair_started_);
  THROW_CHECK(!workspace_ && !io_thread_pool_ && !thread_pool_);
  THROW_CHECK_EQ(options_.max_image_size, -1);
  THROW_CHECK(!options_.allow_missing_files);
  THROW_CHECK(!options_.write_consistency_graph);
  auto format = workspace_format_;
  StringToLower(&format);
  THROW_CHECK_EQ(format, "colmap");
  const auto devices = CSVToVector<int>(options_.gpu_index);
  THROW_CHECK_EQ(devices.size(), 1);
  THROW_CHECK_GE(devices.front(), 0);
  THROW_CHECK_GT(options_.num_threads, 0);
  owned_bank_resolved_ = true;
  const auto release = [this] {
    workspace_.reset();
    problems_.clear();
    depth_ranges_.clear();
  };
  try {
    ReadWorkspace();
    ReadProblems();
    THROW_CHECK(!problems_.empty());
    DensePlanGraph graph;
    const auto& model = workspace_->GetModel();
    graph.images = model.images;
    for (size_t i = 0; i < model.images.size(); ++i)
      graph.names.push_back(model.GetImageName(i));
    FlatHashSet<int> references;
    for (const auto& problem : problems_) {
      THROW_CHECK(references.insert(problem.ref_image_idx).second)
          << "duplicate reference outputs";
    }
    const auto root = std::filesystem::canonical(workspace_path_);
    FlatHashSet<std::string> outputs;
    for (const auto& problem : problems_) {
      // Both optimized phases start fresh controllers from the SAME cfg,
      // unlike stock Run's PHOTO-mutated source vector in its second pass.
      auto effective = options_;
      const auto used = ResolveNativeUsedImages(
          problem.src_image_idxs, problem.ref_image_idx, &effective);
      if (effective.depth_min < 0 || effective.depth_max < 0) {
        effective.depth_min = depth_ranges_.at(problem.ref_image_idx).first;
        effective.depth_max = depth_ranges_.at(problem.ref_image_idx).second;
      }
      THROW_CHECK_GT(effective.depth_min, 0);
      THROW_CHECK_GT(effective.depth_max, effective.depth_min);
      if (effective.sigma_spatial <= 0)
        effective.sigma_spatial = effective.window_radius;
      THROW_CHECK(effective.Check());
      for (const bool geo : {false, true}) {
        owned::BankJob job;
        job.options = effective;
        job.options.geom_consistency = geo;
        job.options.filter = geo ? options_.filter : false;
        job.reference_image_idx = problem.ref_image_idx;
        // Metadata snapshots only. No Bitmap / maps are populated.
        job.images = graph.images;
        for (int idx : used) {
          THROW_CHECK(ExistsFile(workspace_->GetBitmapPath(idx)));
          THROW_CHECK(references.count(idx))
              << "PHOTO producer missing for GEO dependency " << idx;
          if (idx != problem.ref_image_idx)
            job.source_image_idxs.push_back(idx);
        }
        const auto photo_paths = ValidateOptimizedPhotoOutputs(
            root, model.GetImageName(problem.ref_image_idx));
        auto paths = photo_paths;
        if (geo) {
          for (size_t k = 0; k < paths.size(); ++k) {
            const auto folder =
                root / "stereo" / (k == 0 ? "depth_maps" : "normal_maps");
            const auto path =
                folder /
                (model.GetImageName(problem.ref_image_idx) + ".geometric.bin");
            paths[k] = std::filesystem::weakly_canonical(path);
            THROW_CHECK(owned::PathContains(folder, paths[k]));
            THROW_CHECK(!std::filesystem::exists(path) &&
                        !std::filesystem::is_symlink(
                            std::filesystem::symlink_status(path)));
          }
        }
        for (const auto& path : paths)
          THROW_CHECK(outputs.insert(path.string()).second)
              << "aliased outputs";
        job.depth_map_path = paths[0];
        job.normal_map_path = paths[1];
        (geo ? graph.geo : graph.photo).push_back(std::move(job));
      }
    }
    release();
    return graph;
  } catch (...) {
    release();
    throw;
  }
}

std::vector<owned::BankJob> PatchMatchController::ResolveOwnedBankJobs() {
  // Metadata-only resolution is a stock-dispatch CPU path; the one-reference
  // executor wrapper (kOwnedCuda) keeps ResolveOwnedBankJob's guards.
  THROW_CHECK(execution_ == PatchMatchExecution::kStock)
      << "Bank job metadata resolution is a stock-dispatch CPU path";
  return ResolveOwnedBankJobsImpl(nullptr, nullptr);
}

PatchMatchController::OwnedBankResolved
PatchMatchController::ResolveOwnedBankContractWithJobs() {
  // Same guards as the contract capture: the snapshot binds options the
  // owned execution path admits, resolved once (owned_bank_resolved_).
  THROW_CHECK(execution_ == PatchMatchExecution::kStock)
      << "Bank resolution with contract binding is a stock-dispatch CPU "
         "metadata path";
  ValidatePatchMatchExecution(PatchMatchExecution::kOwnedCuda,
                              options_.window_radius,
                              options_.window_step,
                              options_.geom_consistency,
                              /*owned_cuda_available=*/true,
                              options_.filter,
                              options_.write_consistency_graph,
                              options_.allow_missing_files);
  OwnedBankResolved resolved;
  resolved.jobs = ResolveOwnedBankJobsImpl(&resolved.contract, nullptr);
  THROW_CHECK_EQ(resolved.jobs.size(), resolved.contract.problems.size());
  return resolved;
}

OwnedBankContract PatchMatchController::ResolveOwnedBankContract() {
  THROW_CHECK(execution_ == PatchMatchExecution::kStock)
      << "Contract resolution is a stock-dispatch CPU metadata mode";
  // Reuse the exact owned-execution option admission as a CPU-only check so
  // the contract cannot bind options the owned path would reject. No CUDA
  // availability is queried here.
  ValidatePatchMatchExecution(PatchMatchExecution::kOwnedCuda,
                              options_.window_radius,
                              options_.window_step,
                              options_.geom_consistency,
                              /*owned_cuda_available=*/true,
                              options_.filter,
                              options_.write_consistency_graph,
                              options_.allow_missing_files);
  OwnedBankContract contract;
  const auto jobs = ResolveOwnedBankJobsImpl(&contract, nullptr);
  THROW_CHECK_EQ(jobs.size(), contract.problems.size());
  return contract;
}

std::vector<owned::BankJob> PatchMatchController::ResolveOwnedBankJobsImpl(
    OwnedBankContract* contract, const size_t* expected_problems) {
  THROW_CHECK(options_.geom_consistency);
  THROW_CHECK(!owned_pair_started_);
  THROW_CHECK(!workspace_ && !io_thread_pool_ && !thread_pool_);
  const auto devices = CSVToVector<int>(options_.gpu_index);
  THROW_CHECK_EQ(devices.size(), 1);
  THROW_CHECK_GE(devices.front(), 0);
  THROW_CHECK_GT(options_.num_threads, 0);
  // PMVS import writes workspace files; this metadata-only route must not.
  auto format = workspace_format_;
  StringToLower(&format);
  THROW_CHECK_EQ(format, "colmap")
      << "Resident input mode supports only COLMAP workspaces";
  THROW_CHECK(!owned_bank_resolved_) << "Resident job resolves only once";
  owned_bank_resolved_ = true;
  const auto release = [this] {
    problems_.clear();
    workspace_.reset();
    depth_ranges_.clear();
    gpu_indices_.clear();
  };
  try {
    ReadWorkspace();
    ReadProblems();
    if (expected_problems != nullptr) {
      // Early exact-size guard, checked BEFORE any per-job traversal.
      THROW_CHECK_EQ(problems_.size(), *expected_problems);
    }
    THROW_CHECK_GE(problems_.size(), 1);
    const auto& model = workspace_->GetModel();
    std::vector<std::string> image_names;
    image_names.reserve(model.images.size());
    for (size_t image_idx = 0; image_idx < model.images.size(); ++image_idx) {
      image_names.push_back(model.GetImageName(image_idx));
    }
    if (contract != nullptr) {
      *contract = MakeOwnedBankContract(options_,
                                        workspace_->GetOptions().stereo_folder);
      contract->problems.reserve(problems_.size());
    }
    // Native problem order is preserved; only actual duplicates are rejected:
    // two problems sharing one reference name resolve to the same output
    // paths, and output paths are derived deterministically, so duplicate
    // references and output aliases are caught together below. Alias keys
    // use resolved paths so `..`-bearing model names or symlinked output
    // parents cannot mask collisions; depth versus normal outputs of every
    // problem share one key space.
    FlatHashSet<std::string> seen_outputs;
    FlatHashSet<std::string> seen_reference_names;
    const auto output_key = [](const std::filesystem::path& path) {
      return std::filesystem::weakly_canonical(path).string();
    };
    std::vector<owned::BankJob> jobs;
    jobs.reserve(problems_.size());
    for (size_t problem_idx = 0; problem_idx < problems_.size();
         ++problem_idx) {
      OwnedBankContractProblem contract_problem;
      auto job = CaptureOwnedBankJobMetadata(
          problem_idx,
          image_names,
          contract == nullptr ? nullptr : &contract_problem);
      THROW_CHECK(
          seen_reference_names.insert(image_names.at(job.reference_image_idx))
              .second)
          << "Duplicate reference image in configured problems: "
          << image_names.at(job.reference_image_idx);
      THROW_CHECK(seen_outputs.insert(output_key(job.depth_map_path)).second)
          << "Duplicate owned output path across problems: "
          << job.depth_map_path;
      THROW_CHECK(seen_outputs.insert(output_key(job.normal_map_path)).second)
          << "Duplicate owned output path across problems: "
          << job.normal_map_path;
      jobs.push_back(std::move(job));
      if (contract != nullptr) {
        contract->problems.push_back(std::move(contract_problem));
      }
    }
    if (contract != nullptr) {
      // Repeated shared inputs must be captured consistently: a file used by
      // several problems carries exactly one invocation identity. All
      // retained identities are also collected for the invocation-wide
      // fresh revalidation below.
      FlatHashMap<std::string, owned::BankFileIdentity> identity_by_path;
      std::vector<const owned::BankFileIdentity*> retained_identities;
      for (const auto& problem_record : contract->problems) {
        for (const auto& input : problem_record.inputs) {
          const auto check = [&identity_by_path, &retained_identities](
                                 const owned::BankFileIdentity& identity,
                                 const char* kind) {
            retained_identities.push_back(&identity);
            const auto key = identity.path.string();
            const auto inserted = identity_by_path.emplace(key, identity);
            THROW_CHECK(inserted.second ||
                        owned::SameIdentity(inserted.first->second, identity))
                << "Inconsistent repeated " << kind
                << " input identity across problems: " << key;
          };
          check(input.image_identity, "image");
          check(input.depth_identity, "depth");
          check(input.normal_identity, "normal");
        }
      }
      // Invocation-wide fresh revalidation AFTER all jobs, stats and header
      // reads completed: an earlier input changing while later inputs were
      // being read cannot escape. FRESH filesystem checks against every
      // retained identity; the stored-record comparisons above remain the
      // cross-problem shared-path consistency contract.
      for (const owned::BankFileIdentity* identity : retained_identities) {
        owned::CheckFileIdentityUnchanged(*identity);
      }
    }
    release();
    return jobs;
  } catch (...) {
    release();
    throw;
  }
}

owned::BankJob PatchMatchController::ResolveOwnedBankJob() {
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  const size_t one_problem = 1;
  auto jobs = ResolveOwnedBankJobsImpl(nullptr, &one_problem);
  THROW_CHECK_EQ(jobs.size(), 1);
  return std::move(jobs.front());
}

owned::BankJob PatchMatchController::CaptureOwnedBankJobMetadata(
    const size_t problem_idx,
    const std::vector<std::string>& image_names,
    OwnedBankContractProblem* contract) {
  const auto& problem = problems_.at(problem_idx);
  owned::BankJob result;
  result.options = options_;
  result.images = workspace_->GetModel().images;
  result.reference_image_idx = problem.ref_image_idx;
  if (result.options.depth_min < 0 || result.options.depth_max < 0) {
    const auto range = depth_ranges_.at(problem.ref_image_idx);
    result.options.depth_min = range.first;
    result.options.depth_max = range.second;
    THROW_CHECK_GT(range.first, 0);
    THROW_CHECK_GT(range.second, 0);
  }
  if (result.options.sigma_spatial <= 0) {
    result.options.sigma_spatial = result.options.window_radius;
  }
  const auto& stereo = workspace_->GetOptions().stereo_folder;
  const auto file = StringPrintf(
      "%s.geometric.bin",
      workspace_->GetModel().GetImageName(problem.ref_image_idx).c_str());
  result.depth_map_path = workspace_path_ / stereo / "depth_maps" / file;
  result.normal_map_path = workspace_path_ / stereo / "normal_maps" / file;
  THROW_CHECK(!ExistsFile(result.depth_map_path));
  THROW_CHECK(!ExistsFile(result.normal_map_path));
  // Identical insertion and single iteration to ProcessProblemInputs. Never
  // rebuild this hash set from the resolved vector in the resident path.
  const auto used_image_idxs = ResolveNativeUsedImages(
      problem.src_image_idxs, problem.ref_image_idx, &result.options);
  for (const int idx : used_image_idxs) {
    const auto image = workspace_->GetBitmapPath(idx);
    const auto depth = workspace_->GetDepthMapPath(idx);
    const auto normal = workspace_->GetNormalMapPath(idx);
    THROW_CHECK(ExistsFile(image) && ExistsFile(depth) && ExistsFile(normal))
        << "Missing image or map dependency for image " << idx;
    owned::BankInput input;
    input.image_idx = idx;
    input.image = result.images.at(idx);
    input.depth_path = depth;
    input.normal_path = normal;
    input.max_image_size = options_.max_image_size;
    result.inputs.push_back(std::move(input));
    if (idx != problem.ref_image_idx) result.source_image_idxs.push_back(idx);
  }
  if (contract != nullptr) {
    *contract = CaptureOwnedBankContractProblem(result, image_names);
  }
  return result;
}

void PatchMatchController::RunResidentOwnedPairJob(
    const owned::HostInputBank& host,
    const owned::ResidentInputBank& resident,
    owned::PairCoordinator& coordinator,
    const size_t job,
    const OwnedPairHooks& hooks) {
  RunResidentOwnedPairJob(host, resident, coordinator, job, job, hooks);
}

void PatchMatchController::RunResidentOwnedPairJob(
    const owned::HostInputBank& host,
    const owned::ResidentInputBank& resident,
    owned::PairCoordinator& coordinator,
    const size_t job,
    const size_t bank_job_index,
    const OwnedPairHooks& hooks,
    const NormalMap* owned_ref_normal) {
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  THROW_CHECK(options_.geom_consistency);
  THROW_CHECK(!owned_pair_started_);
  owned_pair_started_ = true;
  try {
    THROW_CHECK(!workspace_ && !io_thread_pool_ && !thread_pool_);
    coordinator.BeginPreparation(job);
    const auto& resolved = host.Job(bank_job_index);
    THROW_CHECK(!ExistsFile(resolved.depth_map_path));
    THROW_CHECK(!ExistsFile(resolved.normal_map_path));
    // Mode-consistent reference normal ownership: eager banks borrow their
    // validated retained normal; bounded banks require the caller-owned
    // prepared normal, which outlives the GPU upload (kept until retire).
    if (host.NormalsMode() == owned::BankNormals::kBounded) {
      if (owned_ref_normal == nullptr) {
        throw std::invalid_argument(
            "bounded resident bank requires an owned prepared reference "
            "normal");
      }
    } else if (owned_ref_normal != nullptr) {
      throw std::invalid_argument(
          "eager resident bank retains its own reference normal");
    }
    auto images = resolved.images;
    PatchMatch::Problem problem;
    problem.images = &images;
    problem.ref_image_idx = resolved.reference_image_idx;
    problem.src_image_idxs = resolved.source_image_idxs;
    problem.resident_bank = &resident;
    problem.resident_ref_layer = host.ReferenceLayer(bank_job_index);
    problem.resident_src_layers = host.SourceLayers(bank_job_index);
    problem.resident_ref_normal =
        host.NormalsMode() == owned::BankNormals::kBounded
            ? owned_ref_normal
            : &host.ReferenceNormal(bank_job_index);
    if (hooks.check_inputs) hooks.check_inputs(job, resolved.options, problem);
    OwnedPreparedJob prepared;
    prepared.depth_map_path = resolved.depth_map_path;
    prepared.normal_map_path = resolved.normal_map_path;
    prepared.patch_match =
        std::make_unique<PatchMatch>(resolved.options, problem, execution_);
    prepared.patch_match->Check();
    ExecutePreparedOwnedPairJob(prepared, coordinator, job, hooks);
  } catch (...) {
    coordinator.Cancel(job, std::current_exception());
    throw;
  }
}

std::unique_ptr<PatchMatchController::OwnedPreparedJob>
PatchMatchController::PrepareOwnedPairJob(const size_t job,
                                          const OwnedPairHooks& hooks,
                                          const int preparation_threads) {
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  // `job` is an immutable generation/FIFO index, not a reusable slot number.
  // RunOwnedPairJob validates it with the coordinator before preparation.
  // Reject auto-selection before ReadGpuIndices can query CUDA devices.
  const auto explicit_gpu_indices = CSVToVector<int>(options_.gpu_index);
  THROW_CHECK_EQ(explicit_gpu_indices.size(), 1);
  THROW_CHECK_GE(explicit_gpu_indices.front(), 0);
  THROW_CHECK_GT(options_.num_threads, 0);
  THROW_CHECK_GE(preparation_threads, 0);
  THROW_CHECK_LE(preparation_threads, options_.num_threads);
  THROW_CHECK(!workspace_ && !io_thread_pool_ && !thread_pool_);

  const auto release_inputs = [this] {
    io_thread_pool_.reset();
    problems_.clear();
    workspace_.reset();
    depth_ranges_.clear();
    gpu_indices_.clear();
  };
  try {
    ReadWorkspace();
    ReadProblems();
    THROW_CHECK_EQ(problems_.size(), 1);
    ReadGpuIndices();
    // A lookahead worker may use fewer decode threads without changing the
    // frozen PatchMatch numerical/options metadata. Default synchronous
    // preparation retains the configured decode parallelism.
    io_thread_pool_ = std::make_unique<ThreadPool>(
        preparation_threads == 0 ? options_.num_threads : preparation_threads);
    auto prepared = std::make_unique<OwnedPreparedJob>();
    // Shared with stock dispatch: identical options, source ordering and
    // decode.
    ProcessProblemInputs(options_, 0, prepared.get(), job, hooks);
    THROW_CHECK_NOTNULL(prepared->patch_match.get());
    release_inputs();
    return prepared;
  } catch (...) {
    release_inputs();
    throw;
  }
}

std::unique_ptr<PatchMatchController::OwnedPreparedJob>
PatchMatchController::PrepareSharedOwnedPhotoJob(
    const std::filesystem::path& config,
    const size_t job,
    const OwnedPairHooks& hooks) {
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  THROW_CHECK(!options_.geom_consistency);
  THROW_CHECK(!owned_pair_started_ && !owned_bank_resolved_);
  config_path_ = config;
  return PrepareOwnedPairJob(job, hooks);
}

void PatchMatchController::RunOwnedPairJob(owned::PairCoordinator& coordinator,
                                           const size_t job,
                                           const OwnedPairHooks& hooks) {
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  THROW_CHECK(!owned_pair_started_);
  owned_pair_started_ = true;
  try {
    // Keep synchronous CPU preparation inside the existing single-prep lease.
    coordinator.BeginPreparation(job);
    auto prepared = PrepareOwnedPairJob(job, hooks);
    ExecutePreparedOwnedPairJob(*prepared, coordinator, job, hooks);
  } catch (...) {
    coordinator.Cancel(job, std::current_exception());
    throw;
  }
}

void PatchMatchController::RunPreparedOwnedPairJob(
    std::unique_ptr<OwnedPreparedJob> prepared,
    owned::PairCoordinator& coordinator,
    const size_t job,
    const OwnedPairHooks& hooks) {
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  THROW_CHECK(!options_.geom_consistency)
      << "Owned PAIR/preparation/lookahead is photometric-only";
  THROW_CHECK(!owned_pair_started_);
  owned_pair_started_ = true;
  try {
    THROW_CHECK(!workspace_ && !io_thread_pool_ && !thread_pool_);
    THROW_CHECK_NOTNULL(prepared.get());
    THROW_CHECK_NOTNULL(prepared->patch_match.get());
    THROW_CHECK(!prepared->patch_match->GetOptions().geom_consistency)
        << "Owned PAIR/preparation/lookahead is photometric-only";
    THROW_CHECK(!ExistsFile(prepared->depth_map_path));
    THROW_CHECK(!ExistsFile(prepared->normal_map_path));
    coordinator.BeginPreparation(job);
    ExecutePreparedOwnedPairJob(*prepared, coordinator, job, hooks);
  } catch (...) {
    coordinator.Cancel(job, std::current_exception());
    throw;
  }
}

void PatchMatchController::ExecutePreparedOwnedPairJob(
    OwnedPreparedJob& prepared,
    owned::PairCoordinator& coordinator,
    const size_t job,
    const OwnedPairHooks& hooks) {
  auto& patch_match = prepared.patch_match;
  try {
    if (CheckIfStopped()) {
      THROW_CHECK(false) << "Owned pair stopped before solver";
    }
    if (hooks.metadata) {
      hooks.metadata(job, patch_match->GetOptions(), patch_match->GetProblem());
    }
    using Clock = std::chrono::steady_clock;
    auto stage_start = Clock::now();
    const auto record_stage = [&](const char* stage) {
      const auto now = Clock::now();
      if (hooks.stage_timing) {
        hooks.stage_timing(
            job,
            stage,
            std::chrono::duration_cast<std::chrono::microseconds>(now -
                                                                  stage_start)
                .count());
      }
      stage_start = now;
    };
    patch_match->PrepareOwned(
        hooks.owned_tile_threads, hooks.graph_coordinator, hooks.graph_owner);
    record_stage("cuda_prepare");
    patch_match->SetOwnedSweepObserver([&hooks, job](int iteration, int sweep) {
      if (hooks.sweep) hooks.sweep(job, iteration, sweep);
    });
    coordinator.ReadyAndWait(job);
    record_stage("solver_wait");
    patch_match->RunPreparedOwned();
    record_stage("solve");
    coordinator.SolverDone(job);
    coordinator.BeginOutput(job);
    if (hooks.output_phase) hooks.output_phase(job, "depth_begin");
    owned::WriteCheckedMat(patch_match->GetDepthMap(), prepared.depth_map_path);
    if (hooks.output_phase) {
      hooks.output_phase(job, "depth_end");
      hooks.output_phase(job, "normal_begin");
    }
    owned::WriteCheckedMat(patch_match->GetNormalMap(),
                           prepared.normal_map_path);
    if (hooks.output_phase) hooks.output_phase(job, "normal_end");
    coordinator.EndOutput(job);
    coordinator.AwaitOutputs(job);
  } catch (...) {
    coordinator.CancelAndAwaitCleanup(
        job, std::current_exception(), [&] { patch_match->DrainOwned(); });
    throw;
  }
}

void PatchMatchController::ReadWorkspace() {
  LOG(INFO) << "Reading workspace...";

  Workspace::Options workspace_options;

  auto workspace_format_lower_case = workspace_format_;
  StringToLower(&workspace_format_lower_case);
  if (workspace_format_lower_case == "pmvs") {
    workspace_options.stereo_folder =
        StringPrintf("stereo-%s", pmvs_option_name_.c_str());
  }

  workspace_options.max_image_size = options_.max_image_size;
  workspace_options.image_as_rgb = false;
  workspace_options.cache_size = options_.cache_size;
  workspace_options.workspace_path = workspace_path_;
  workspace_options.workspace_format = workspace_format_;
  workspace_options.input_type = options_.geom_consistency ? "photometric" : "";

  workspace_ = std::make_unique<CachedWorkspace>(workspace_options);

  if (workspace_format_lower_case == "pmvs") {
    LOG(INFO) << StringPrintf("Importing PMVS workspace (option %s)...",
                              pmvs_option_name_.c_str());
    ImportPMVSWorkspace(*workspace_, pmvs_option_name_);
  }

  depth_ranges_ = workspace_->GetModel().ComputeDepthRanges();
}

void PatchMatchController::ReadProblems() {
  LOG(INFO) << "Reading configuration...";

  problems_.clear();

  const auto& model = workspace_->GetModel();

  const auto config_path = config_path_.empty()
                               ? workspace_path_ /
                                     workspace_->GetOptions().stereo_folder /
                                     "patch-match.cfg"
                               : config_path_;
  std::vector<std::string> config = ReadTextFileLines(config_path);

  std::vector<std::map<int, int>> shared_num_points;
  std::vector<std::map<int, float>> triangulation_angles;

  const float min_triangulation_angle_rad =
      DegToRad(options_.min_triangulation_angle);

  std::string ref_image_name;
  FlatHashSet<int> ref_image_idxs;

  struct ProblemConfig {
    std::string ref_image_name;
    std::vector<std::string> src_image_names;
  };
  std::vector<ProblemConfig> problem_configs;

  for (size_t i = 0; i < config.size(); ++i) {
    std::string& config_line = config[i];
    StringTrim(&config_line);

    if (config_line.empty() || config_line[0] == '#') {
      continue;
    }

    if (ref_image_name.empty()) {
      ref_image_name = config_line;
      continue;
    }

    ref_image_idxs.insert(model.GetImageIdx(ref_image_name));

    ProblemConfig problem_config;
    problem_config.ref_image_name = ref_image_name;
    problem_config.src_image_names = CSVToVector<std::string>(config_line);
    problem_configs.push_back(problem_config);

    ref_image_name.clear();
  }

  for (const auto& problem_config : problem_configs) {
    PatchMatch::Problem problem;

    problem.ref_image_idx = model.GetImageIdx(problem_config.ref_image_name);

    if (problem_config.src_image_names.size() == 1 &&
        problem_config.src_image_names[0] == "__all__") {
      // Use all images as source images.
      problem.src_image_idxs.clear();
      problem.src_image_idxs.reserve(model.images.size() - 1);
      for (size_t image_idx = 0; image_idx < model.images.size(); ++image_idx) {
        if (static_cast<int>(image_idx) != problem.ref_image_idx) {
          problem.src_image_idxs.push_back(image_idx);
        }
      }
    } else if (problem_config.src_image_names.size() == 2 &&
               problem_config.src_image_names[0] == "__auto__") {
      // Use maximum number of overlapping images as source images. Overlapping
      // will be sorted based on the number of shared points to the reference
      // image and the top ranked images are selected. Note that images are only
      // selected if some points have a sufficient triangulation angle.

      if (shared_num_points.empty()) {
        shared_num_points = model.ComputeSharedPoints();
      }
      if (triangulation_angles.empty()) {
        const float kTriangulationAnglePercentile = 75;
        triangulation_angles =
            model.ComputeTriangulationAngles(kTriangulationAnglePercentile);
      }

      const size_t max_num_src_images =
          std::stoll(problem_config.src_image_names[1]);

      const auto& overlapping_images =
          shared_num_points.at(problem.ref_image_idx);
      const auto& overlapping_triangulation_angles =
          triangulation_angles.at(problem.ref_image_idx);

      std::vector<std::pair<int, int>> src_images;
      src_images.reserve(overlapping_images.size());
      for (const auto& image : overlapping_images) {
        if (overlapping_triangulation_angles.at(image.first) >=
            min_triangulation_angle_rad) {
          src_images.emplace_back(image.first, image.second);
        }
      }

      const size_t eff_max_num_src_images =
          std::min(src_images.size(), max_num_src_images);

      std::partial_sort(src_images.begin(),
                        src_images.begin() + eff_max_num_src_images,
                        src_images.end(),
                        [](const std::pair<int, int>& image1,
                           const std::pair<int, int>& image2) {
                          return image1.second > image2.second;
                        });

      problem.src_image_idxs.reserve(eff_max_num_src_images);
      for (size_t i = 0; i < eff_max_num_src_images; ++i) {
        problem.src_image_idxs.push_back(src_images[i].first);
      }
    } else {
      problem.src_image_idxs.reserve(problem_config.src_image_names.size());
      for (const auto& src_image_name : problem_config.src_image_names) {
        problem.src_image_idxs.push_back(model.GetImageIdx(src_image_name));
      }
    }

    if (problem.src_image_idxs.empty()) {
      LOG(WARNING) << StringPrintf(
          "Ignoring reference image %s, because it has no "
          "source images.",
          problem_config.ref_image_name.c_str());
    } else {
      problems_.push_back(problem);
    }
  }

  LOG(INFO) << StringPrintf("Configuration has %d problems...",
                            problems_.size());
}

void PatchMatchController::ReadGpuIndices() {
  gpu_indices_ = CSVToVector<int>(options_.gpu_index);
  if (gpu_indices_.size() == 1 && gpu_indices_[0] == -1) {
    const int num_cuda_devices = GetNumCudaDevices();
    THROW_CHECK_GT(num_cuda_devices, 0);
    gpu_indices_.resize(num_cuda_devices);
    std::iota(gpu_indices_.begin(), gpu_indices_.end(), 0);
  }
}

void PatchMatchController::ProcessProblem(const PatchMatchOptions& options,
                                          const size_t problem_idx) {
  ProcessProblemInputs(options, problem_idx, nullptr, 0, OwnedPairHooks{});
}

void PatchMatchController::ProcessProblemInputs(
    const PatchMatchOptions& options,
    const size_t problem_idx,
    OwnedPreparedJob* prepared,
    const size_t job,
    const OwnedPairHooks& hooks) {
  if (CheckIfStopped()) {
    if (prepared) {
      THROW_CHECK(false) << "Owned pair stopped before solver";
    }
    return;
  }

  const auto& model = workspace_->GetModel();

  auto& problem = problems_.at(problem_idx);
  const int gpu_index = prepared
                            ? gpu_indices_.front()
                            : gpu_indices_.at(thread_pool_->GetThreadIndex());
  THROW_CHECK_GE(gpu_index, -1);

  const std::string& stereo_folder = workspace_->GetOptions().stereo_folder;
  const std::string output_type =
      options.geom_consistency ? "geometric" : "photometric";
  const std::string image_name = model.GetImageName(problem.ref_image_idx);
  const std::string file_name =
      StringPrintf("%s.%s.bin", image_name.c_str(), output_type.c_str());
  const auto depth_map_path =
      workspace_path_ / stereo_folder / "depth_maps" / file_name;
  const auto normal_map_path =
      workspace_path_ / stereo_folder / "normal_maps" / file_name;
  const auto consistency_graph_path =
      workspace_path_ / stereo_folder / "consistency_graphs" / file_name;

  if (execution_ == PatchMatchExecution::kOwnedCuda &&
      options.geom_consistency) {
    THROW_CHECK(!ExistsFile(depth_map_path) && !ExistsFile(normal_map_path))
        << "Owned geometry requires fresh output paths";
  }

  if (ExistsFile(depth_map_path) && ExistsFile(normal_map_path) &&
      (!options.write_consistency_graph ||
       ExistsFile(consistency_graph_path))) {
    THROW_CHECK(execution_ != PatchMatchExecution::kOwnedCuda)
        << "Owned prototype requires fresh output paths, not a skipped result";
    return;
  }

  LOG_HEADING1(StringPrintf("Processing view %d / %d for %s",
                            problem_idx + 1,
                            problems_.size(),
                            image_name.c_str()));

  auto patch_match_options = options;

  if (patch_match_options.depth_min < 0 || patch_match_options.depth_max < 0) {
    patch_match_options.depth_min =
        depth_ranges_.at(problem.ref_image_idx).first;
    patch_match_options.depth_max =
        depth_ranges_.at(problem.ref_image_idx).second;
    THROW_CHECK(patch_match_options.depth_min > 0 &&
                patch_match_options.depth_max > 0)
        << " - You must manually set the minimum and maximum depth, since no "
           "sparse model is provided in the workspace.";
  }

  patch_match_options.gpu_index = std::to_string(gpu_index);

  if (patch_match_options.sigma_spatial <= 0.0f) {
    patch_match_options.sigma_spatial = patch_match_options.window_radius;
  }

  std::vector<Image> images = model.images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  if (options.geom_consistency) {
    depth_maps.resize(model.images.size());
    normal_maps.resize(model.images.size());
  }

  problem.images = &images;
  problem.depth_maps = &depth_maps;
  problem.normal_maps = &normal_maps;

  {
    // Collect all used images in current problem.
    const auto used_image_idxs = ResolveNativeUsedImages(
        problem.src_image_idxs, problem.ref_image_idx, &patch_match_options);

    // Only access workspace from one thread at a time and only spawn resample
    // threads from one master thread at a time.
    std::unique_lock<std::mutex> lock(workspace_mutex_);

    LOG(INFO) << "Reading inputs...";

    // Filter by file existence first (fast, no heavy I/O).
    std::vector<int> src_image_idxs;
    std::vector<int> valid_image_idxs;
    for (const auto image_idx : used_image_idxs) {
      const auto image_path = workspace_->GetBitmapPath(image_idx);
      const auto depth_path = workspace_->GetDepthMapPath(image_idx);
      const auto normal_path = workspace_->GetNormalMapPath(image_idx);

      if (!ExistsFile(image_path) ||
          (options.geom_consistency && !ExistsFile(depth_path)) ||
          (options.geom_consistency && !ExistsFile(normal_path))) {
        if (options.allow_missing_files) {
          LOG(WARNING) << StringPrintf(
              "Skipping source image %d: %s for missing "
              "image or depth/normal map",
              image_idx,
              model.GetImageName(image_idx).c_str());
          continue;
        } else {
          if (execution_ == PatchMatchExecution::kOwnedCuda &&
              options.geom_consistency) {
            THROW_CHECK(false)
                << "Missing image or map dependency for image " << image_idx
                << ": " << model.GetImageName(image_idx);
          }
          LOG(ERROR) << StringPrintf(
              "Missing image or map dependency for image %d: %s",
              image_idx,
              model.GetImageName(image_idx).c_str());
        }
      }

      valid_image_idxs.push_back(image_idx);
      if (image_idx != problem.ref_image_idx) {
        src_image_idxs.push_back(image_idx);
      }
    }

    if (prepared && hooks.check_inputs) {
      auto budget_problem = problem;
      budget_problem.src_image_idxs = src_image_idxs;
      hooks.check_inputs(job, patch_match_options, budget_problem);
    }

    // Read images in parallel using the shared I/O thread pool.
    std::vector<std::shared_future<void>> futures;
    futures.reserve(valid_image_idxs.size());
    std::exception_ptr input_error;
    try {
      for (const auto image_idx : valid_image_idxs) {
        futures.push_back(io_thread_pool_->AddTask([&, image_idx]() {
          images.at(image_idx).SetBitmap(workspace_->GetBitmap(image_idx));
          if (options.geom_consistency) {
            depth_maps.at(image_idx) = workspace_->GetDepthMap(image_idx);
            normal_maps.at(image_idx) = workspace_->GetNormalMap(image_idx);
          }
        }));
      }
    } catch (...) {
      if (!prepared) throw;
      input_error = std::current_exception();
    }
    if (prepared) {
      // CPU preparation has no coordinator. Drain every launched task and the
      // pool's checker queue before local images or workspace can unwind.
      owned::DrainPairInputs(io_thread_pool_,
                             futures,
                             input_error,
                             [](std::exception_ptr) noexcept {});
    } else {
      for (auto& future : futures) future.get();
    }

    problem.src_image_idxs = src_image_idxs;
  }

  problem.Print();
  patch_match_options.Print();

  if (prepared) {
    prepared->depth_map_path = depth_map_path;
    prepared->normal_map_path = normal_map_path;
    prepared->patch_match =
        std::make_unique<PatchMatch>(patch_match_options, problem, execution_);
    prepared->patch_match->Check();
    prepared->prepared_by = this;
    prepared->prepared_job = job;
    return;
  }

  PatchMatch patch_match(patch_match_options, problem, execution_);
  patch_match.Run();

  LOG(INFO) << std::endl
            << StringPrintf("Writing %s output for %s",
                            output_type.c_str(),
                            image_name.c_str());

  patch_match.GetDepthMap().Write(depth_map_path);
  patch_match.GetNormalMap().Write(normal_map_path);
  if (options.write_consistency_graph) {
    patch_match.GetConsistencyGraph().Write(consistency_graph_path);
  }
}

}  // namespace mvs
}  // namespace colmap
