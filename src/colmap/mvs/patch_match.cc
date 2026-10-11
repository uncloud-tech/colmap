// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include "colmap/mvs/owned_pair_input.h"
#include "colmap/mvs/owned_pair_output.h"

#include "colmap/math/math.h"
#include "colmap/mvs/consistency_graph.h"
#include "colmap/mvs/patch_match_cuda.h"
#if !defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/patch_match_cuda_owned.h"
#endif
#include "colmap/mvs/workspace.h"
#include "colmap/util/cuda.h"
#include "colmap/util/file.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <numeric>
#include <set>

#define PrintOption(option) LOG(INFO) << #option ": " << option << std::endl

namespace colmap {
namespace mvs {

namespace {
PatchMatchExecution ValidateExecution(const PatchMatchOptions& options,
                                     const PatchMatchExecution execution) {
#if defined(COLMAP_CUDA_ENABLED) && !defined(COLMAP_HIP_ENABLED)
  constexpr bool kOwnedCudaAvailable = true;
#else
  constexpr bool kOwnedCudaAvailable = false;
#endif
  return ValidatePatchMatchExecution(execution, options.window_radius,
                                     options.window_step,
                                     options.geom_consistency,
                                     kOwnedCudaAvailable);
}
}  // namespace

PatchMatch::PatchMatch(const PatchMatchOptions& options,
                       const Problem& problem,
                       const PatchMatchExecution execution)
    : options_(options),
      execution_(ValidateExecution(options, execution)),
      owned_images_(execution_ == PatchMatchExecution::kOwnedCuda && problem.images
                        ? *problem.images : std::vector<Image>{}),
      problem_([&] {
        Problem result = problem;
        if (execution_ == PatchMatchExecution::kOwnedCuda) {
          result.images = &owned_images_;
          result.depth_maps = nullptr;
          result.normal_maps = nullptr;
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
  if (options_.geom_consistency) {
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
    THROW_CHECK_GT(image.GetBitmap().Width(), 0) << image_idx;
    THROW_CHECK_GT(image.GetBitmap().Height(), 0) << image_idx;
    THROW_CHECK(image.GetBitmap().IsGrey()) << image_idx;
    THROW_CHECK_EQ(image.GetWidth(), image.GetBitmap().Width()) << image_idx;
    THROW_CHECK_EQ(image.GetHeight(), image.GetBitmap().Height()) << image_idx;

    // Make sure, the calibration matrix only contains fx, fy, cx, cy.
    THROW_CHECK_LT(std::abs(image.GetK()[1] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[3] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[6] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[7] - 0.0f), 1e-6f) << image_idx;
    THROW_CHECK_LT(std::abs(image.GetK()[8] - 1.0f), 1e-6f) << image_idx;

    if (options_.geom_consistency) {
      THROW_CHECK_LT(image_idx, problem_.depth_maps->size()) << image_idx;
      const DepthMap& depth_map = problem_.depth_maps->at(image_idx);
      THROW_CHECK_EQ(image.GetWidth(), depth_map.GetWidth()) << image_idx;
      THROW_CHECK_EQ(image.GetHeight(), depth_map.GetHeight()) << image_idx;
    }
  }

  if (options_.geom_consistency) {
    const Image& ref_image = problem_.images->at(problem_.ref_image_idx);
    const NormalMap& ref_normal_map =
        problem_.normal_maps->at(problem_.ref_image_idx);
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

void PatchMatch::PrepareOwned() {
#if !defined(COLMAP_HIP_ENABLED)
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  THROW_CHECK(!owned_patch_match_cuda_) << "Owned instances prepare only once";
  Check();
  owned_patch_match_cuda_ =
      std::make_unique<owned::PatchMatchCuda>(options_, problem_);
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

void PatchMatch::SetOwnedSweepObserver(
    std::function<void(int, int)> observer) {
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
    return ConsistencyGraph(ref_image.GetWidth(), ref_image.GetHeight(),
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

  // If geometric consistency is enabled, then photometric output must be
  // computed first for all images without filtering.
  if (options_.geom_consistency) {
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

std::unique_ptr<PatchMatchController::OwnedPreparedJob>
PatchMatchController::PrepareOwnedPairJob(const size_t job,
                                        const OwnedPairHooks& hooks,
                                        const int preparation_threads) {
  THROW_CHECK(execution_ == PatchMatchExecution::kOwnedCuda);
  THROW_CHECK_LE(job, 1);
  THROW_CHECK(!options_.geom_consistency);
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
    // Shared with stock dispatch: identical options, source ordering and decode.
    ProcessProblemInputs(options_, 0, prepared.get(), job, hooks);
    THROW_CHECK_NOTNULL(prepared->patch_match.get());
    release_inputs();
    return prepared;
  } catch (...) {
    release_inputs();
    throw;
  }
}

void PatchMatchController::RunOwnedPairJob(
    owned::PairCoordinator& coordinator,
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
  THROW_CHECK(!owned_pair_started_);
  owned_pair_started_ = true;
  try {
    THROW_CHECK(!workspace_ && !io_thread_pool_ && !thread_pool_);
    THROW_CHECK_NOTNULL(prepared.get());
    THROW_CHECK_NOTNULL(prepared->patch_match.get());
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
    patch_match->PrepareOwned();
    patch_match->SetOwnedSweepObserver([&hooks, job](int iteration, int sweep) {
      if (hooks.sweep) hooks.sweep(job, iteration, sweep);
    });
    coordinator.ReadyAndWait(job);
    patch_match->RunPreparedOwned();
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
    coordinator.CancelAndAwaitCleanup(job, std::current_exception(), [&] {
      patch_match->DrainOwned();
    });
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
    FlatHashSet<int> used_image_idxs(problem.src_image_idxs.begin(),
                                     problem.src_image_idxs.end());
    used_image_idxs.insert(problem.ref_image_idx);

    patch_match_options.filter_min_num_consistent =
        std::min(static_cast<int>(used_image_idxs.size()) - 1,
                 patch_match_options.filter_min_num_consistent);

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
      owned::DrainPairInputs(io_thread_pool_, futures, input_error,
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
