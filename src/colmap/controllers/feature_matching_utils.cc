// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/controllers/feature_matching_utils.h"

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/util/cuda.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#if defined(COLMAP_MVS_ENABLED)
#include "colmap/mvs/owned_fixed_cohort_validation.h"
#endif

#include <numeric>

namespace colmap {

FeatureMatcherWorker::FeatureMatcherWorker(
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::shared_ptr<FeatureMatcherCache>& cache,
    JobQueue<Input>* input_queue,
    JobQueue<Output>* output_queue,
    std::shared_ptr<MatchingFailure> failure)
    : failure_(std::move(failure)),
      matching_options_(matching_options),
      geometry_options_(geometry_options),
      cache_(cache),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  THROW_CHECK(matching_options_.Check());

  if (matching_options_.RequiresOpenGL()) {
    opengl_context_ = std::make_unique<OpenGLContextManager>();
  }
}

void FeatureMatcherWorker::Run() {
  if (!failure_) {
    RunImpl();
    return;
  }
  try {
    RunImpl();
  } catch (...) {
    failure_->Fail(std::current_exception());
    if (!setup_notified_) SignalInvalidSetup();
  }
}

void FeatureMatcherWorker::RunImpl() {
  if (opengl_context_ != nullptr) {
    THROW_CHECK(opengl_context_->MakeCurrent());
  }

#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
  if (matching_options_.use_gpu) {
    // Initialize the GPU device for this worker thread
    const std::vector<int> gpu_indices =
        CSVToVector<int>(matching_options_.gpu_index);
    THROW_CHECK_EQ(gpu_indices.size(), 1)
        << "Each matching worker can only use one GPU";
    const int gpu_index = gpu_indices[0];

    if (gpu_index >= 0) {
      SetBestCudaDevice(gpu_index);
      LOG(INFO) << "Bind FeatureMatcherWorker to GPU device " << gpu_index;
    }
  }
#endif

  if (matching_options_.type == FeatureMatcherType::SIFT_BRUTEFORCE) {
    // TODO(jsch): This is a bit ugly, but currently cannot think of a better
    // way to inject the shared descriptor index cache.
    THROW_CHECK_NOTNULL(matching_options_.sift)->cpu_descriptor_index_cache =
        &cache_->GetFeatureDescriptorIndexCache();
    THROW_CHECK_NOTNULL(matching_options_.sift->cpu_descriptor_index_cache);
  }

  // Minimize the amount of allocated GPU memory by computing the maximum number
  // of descriptors for any image over the whole database.
  matching_options_.max_num_matches = std::min<int>(
      matching_options_.max_num_matches, cache_->MaxNumKeypoints());

  std::unique_ptr<FeatureMatcher> matcher =
      FeatureMatcher::Create(matching_options_);
  if (matcher == nullptr) {
    LOG(ERROR) << "Failed to create feature matcher.";
    SignalInvalidSetup();
    setup_notified_ = true;
    return;
  }

  SignalValidSetup();
  setup_notified_ = true;

  while (true) {
    if (IsStopped()) {
      if (failure_) throw std::runtime_error("matching worker cancelled");
      break;
    }

    auto input_job = input_queue_->Pop();
    if (!input_job.IsValid() && failure_) break;
    if (input_job.IsValid()) {
      auto& data = input_job.Data();
      if (failure_ && failure_->before_match) failure_->before_match();

      if (!cache_->ExistsDescriptors(data.image_id1) ||
          !cache_->ExistsDescriptors(data.image_id2)) {
        THROW_CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      const auto& camera1 =
          cache_->GetCamera(cache_->GetImage(data.image_id1).CameraId());
      const auto& camera2 =
          cache_->GetCamera(cache_->GetImage(data.image_id2).CameraId());

      if (matching_options_.guided_matching) {
        matcher->MatchGuided(
            geometry_options_.ransac_options.max_error,
            {
                data.image_id1,
                &camera1,
                cache_->GetKeypoints(data.image_id1),
                cache_->GetDescriptors(data.image_id1),
                cache_->FindImagePosePriorOrNull(data.image_id1),
            },
            {
                data.image_id2,
                &camera2,
                cache_->GetKeypoints(data.image_id2),
                cache_->GetDescriptors(data.image_id2),
                cache_->FindImagePosePriorOrNull(data.image_id2),
            },
            &data.two_view_geometry);
      } else {
        matcher->Match(
            {
                data.image_id1,
                &camera1,
                cache_->GetKeypoints(data.image_id1),
                cache_->GetDescriptors(data.image_id1),
                cache_->FindImagePosePriorOrNull(data.image_id1),
            },
            {
                data.image_id2,
                &camera2,
                cache_->GetKeypoints(data.image_id2),
                cache_->GetDescriptors(data.image_id2),
                cache_->FindImagePosePriorOrNull(data.image_id2),
            },
            &data.matches);
      }

      THROW_CHECK(output_queue_->Push(std::move(data)));
    }
  }
}

namespace {

class VerifierWorker : public Thread {
 public:
  using Input = FeatureMatcherData;
  using Output = FeatureMatcherData;

  VerifierWorker(const TwoViewGeometryOptions& options,
                 std::shared_ptr<FeatureMatcherCache> cache,
                 JobQueue<Input>* input_queue,
                 JobQueue<Output>* output_queue,
                 const bool use_existing_relative_pose = false,
                 std::shared_ptr<MatchingFailure> failure = nullptr)
      : failure_(std::move(failure)), options_(options),
        cache_(std::move(cache)),
        use_existing_relative_pose_(use_existing_relative_pose),
        input_queue_(input_queue),
        output_queue_(output_queue) {
    THROW_CHECK(options_.Check());
  }

 protected:
  void Run() override {
    if (!failure_) {
      RunImpl();
      return;
    }
    try {
      RunImpl();
    } catch (...) {
      failure_->Fail(std::current_exception());
    }
  }
  void RunImpl() {
    while (true) {
      if (IsStopped()) {
        if (failure_) throw std::runtime_error("matching verifier cancelled");
        break;
      }

      auto input_job = input_queue_->Pop();
      if (!input_job.IsValid() && failure_) break;
      if (input_job.IsValid()) {
        auto& data = input_job.Data();

        // Early abort pairs with too few matches to avoid data lookup,
        // labeling them DEGENERATE so the rejection is stored downstream.
        if (data.matches.size() <
            static_cast<size_t>(options_.min_num_inliers)) {
          data.two_view_geometry.config =
              TwoViewGeometry::ConfigurationType::DEGENERATE;
          THROW_CHECK(output_queue_->Push(std::move(data)));
          continue;
        }

        const auto& camera1 =
            cache_->GetCamera(cache_->GetImage(data.image_id1).CameraId());
        const auto& camera2 =
            cache_->GetCamera(cache_->GetImage(data.image_id2).CameraId());
        const auto keypoints1 = cache_->GetKeypoints(data.image_id1);
        const auto keypoints2 = cache_->GetKeypoints(data.image_id2);
        const std::vector<Eigen::Vector2d> points1 =
            FeatureKeypointsToPointsVector(*keypoints1);
        const std::vector<Eigen::Vector2d> points2 =
            FeatureKeypointsToPointsVector(*keypoints2);

        if (use_existing_relative_pose_ &&
            data.two_view_geometry.cam2_from_cam1.has_value()) {
          data.two_view_geometry = TwoViewGeometryFromKnownRelativePose(
              camera1,
              points1,
              camera2,
              points2,
              *data.two_view_geometry.cam2_from_cam1,
              data.matches,
              options_.min_num_inliers,
              options_.ransac_options.max_error);
        } else {
          data.two_view_geometry = EstimateTwoViewGeometry(
              camera1, points1, camera2, points2, data.matches, options_);
        }

        THROW_CHECK(output_queue_->Push(std::move(data)));
      }
    }
  }

 private:
  std::shared_ptr<MatchingFailure> failure_;
  const TwoViewGeometryOptions options_;
  std::shared_ptr<FeatureMatcherCache> cache_;
  const bool use_existing_relative_pose_;
  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

// Stores the result of verifying an image pair. Pairs that were rejected are
// stored with their DEGENERATE diagnosis but without any payload, so that a
// rejected pair never leaves models, a pose, or inlier matches behind. Pairs
// that bypassed verification (UNDEFINED) are not stored at all.
void WriteVerifiedTwoViewGeometry(FeatureMatcherCache& cache,
                                  image_t image_id1,
                                  image_t image_id2,
                                  TwoViewGeometry two_view_geometry,
                                  int min_num_inliers) {
  if (two_view_geometry.config ==
      TwoViewGeometry::ConfigurationType::UNDEFINED) {
    return;
  }

  if (two_view_geometry.config ==
          TwoViewGeometry::ConfigurationType::DEGENERATE ||
      two_view_geometry.inlier_matches.size() <
          static_cast<size_t>(min_num_inliers)) {
    two_view_geometry = TwoViewGeometry();
    two_view_geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
  }

  cache.WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
}

}  // namespace

FeatureMatcherController::FeatureMatcherController(
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    std::shared_ptr<FeatureMatcherCache> cache)
    : matching_options_(matching_options),
      geometry_options_(geometry_options),
      cache_(std::move(cache)),
      // If skip_geometric_verification, match directly to output_queue_.
      // Guided matching always requires a verification pass beforehand.
      skip_geometric_verification_(
          matching_options.skip_geometric_verification &&
          !matching_options.guided_matching),
      is_setup_(false) {
  THROW_CHECK(matching_options_.Check());
  THROW_CHECK(geometry_options_.Check());
  THROW_CHECK_EQ(geometry_options_.ransac_options.num_threads, 1)
      << "Parallel RANSAC is not supported inside multi-threaded matching";

  if (matching_options_.host_budget_bytes) {
    if (matching_options_.type != FeatureMatcherType::SIFT_BRUTEFORCE ||
        matching_options_.guided_matching || matching_options_.rig_verification ||
        geometry_options_.multiple_models)
      throw std::invalid_argument("budgeted matching supports exhaustive non-guided/non-rig SIFT without multiple models only");
    if (matching_options_.use_gpu && matching_options_.RequiresOpenGL())
      throw std::invalid_argument("budgeted GPU matching requires native CUDA backend");
#if !defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
    if (matching_options_.use_gpu)
      throw std::invalid_argument("budgeted GPU matching requires native CUDA backend");
#endif
    failure_ = std::make_shared<MatchingFailure>();
    failure_->stop_queues = [this] { StopQueues(); };
    resources_ = std::make_shared<MatchingResources>();
  }

  const int num_threads = GetEffectiveNumThreads(matching_options_.num_threads);
  THROW_CHECK_GT(num_threads, 0);

  std::vector<int> gpu_indices = CSVToVector<int>(matching_options_.gpu_index);
  THROW_CHECK_GT(gpu_indices.size(), 0);

#if defined(COLMAP_CUDA_ENABLED) || defined(COLMAP_HIP_ENABLED)
  if (matching_options_.use_gpu && gpu_indices.size() == 1 &&
      gpu_indices[0] == -1) {
    const int num_gpu_devices = GetNumCudaDevices();
    THROW_CHECK_GT(num_gpu_devices, 0);
    gpu_indices.resize(num_gpu_devices);
    std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
  }
#endif  // COLMAP_CUDA_ENABLED || COLMAP_HIP_ENABLED

  JobQueue<FeatureMatcherData>* matcher_output_queue =
      skip_geometric_verification_ ? &output_queue_ : &verifier_queue_;

  if (matching_options_.use_gpu) {
    auto worker_matching_options = matching_options_;
    // The first matching is always without guided matching.
    worker_matching_options.guided_matching = false;
    matchers_.reserve(gpu_indices.size());
    for (const auto& gpu_index : gpu_indices) {
      worker_matching_options.gpu_index = std::to_string(gpu_index);
      matchers_.emplace_back(
          std::make_unique<FeatureMatcherWorker>(worker_matching_options,
                                                 geometry_options_,
                                                 cache_,
                                                 &matcher_queue_,
                                                 matcher_output_queue, failure_));
    }
  } else {
    auto worker_matching_options = matching_options_;
    // Prevent nested threading.
    worker_matching_options.num_threads = 1;
    // The first matching is always without guided matching.
    worker_matching_options.guided_matching = false;
    matchers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      matchers_.emplace_back(
          std::make_unique<FeatureMatcherWorker>(worker_matching_options,
                                                 geometry_options_,
                                                 cache_,
                                                 &matcher_queue_,
                                                 matcher_output_queue, failure_));
    }
  }

  verifiers_.reserve(num_threads);
  if (matching_options_.guided_matching) {
    // Redirect the verification output to final round of guided matching.
    for (int i = 0; i < num_threads; ++i) {
      verifiers_.emplace_back(std::make_unique<VerifierWorker>(
          geometry_options_, cache_, &verifier_queue_, &guided_matcher_queue_, false, failure_));
    }

    if (matching_options_.use_gpu) {
      auto worker_matching_options = matching_options_;
      guided_matchers_.reserve(gpu_indices.size());
      for (const auto& gpu_index : gpu_indices) {
        worker_matching_options.gpu_index = std::to_string(gpu_index);
        guided_matchers_.emplace_back(
            std::make_unique<FeatureMatcherWorker>(worker_matching_options,
                                                   geometry_options_,
                                                   cache_,
                                                   &guided_matcher_queue_,
                                                   &output_queue_, failure_));
      }
    } else {
      auto worker_matching_options = matching_options_;
      // Prevent nested threading.
      worker_matching_options.num_threads = 1;
      guided_matchers_.reserve(num_threads);
      for (int i = 0; i < num_threads; ++i) {
        guided_matchers_.emplace_back(
            std::make_unique<FeatureMatcherWorker>(worker_matching_options,
                                                   geometry_options_,
                                                   cache_,
                                                   &guided_matcher_queue_,
                                                   &output_queue_, failure_));
      }
    }
  } else if (!matching_options.skip_geometric_verification) {
    for (int i = 0; i < num_threads; ++i) {
      verifiers_.emplace_back(std::make_unique<VerifierWorker>(
          geometry_options_, cache_, &verifier_queue_, &output_queue_, false, failure_));
    }
  }
}

FeatureMatcherController::~FeatureMatcherController() {
  if (resources_) {
    StopQueues();
    for (auto& worker : matchers_) worker->Stop();
    for (auto& worker : verifiers_) worker->Stop();
    for (auto& worker : matchers_) worker->Wait();
    for (auto& worker : verifiers_) worker->Wait();
    matcher_queue_.Clear();
    verifier_queue_.Clear();
    guided_matcher_queue_.Clear();
    output_queue_.Clear();
    // Destroy worker-local indexes/CUDA buffers before their reservations.
    matchers_.clear();
    verifiers_.clear();
    return;
  }
  matcher_queue_.Wait();
  verifier_queue_.Wait();
  guided_matcher_queue_.Wait();
  output_queue_.Wait();

  for (auto& matcher : matchers_) {
    matcher->Stop();
  }

  for (auto& verifier : verifiers_) {
    verifier->Stop();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->Stop();
  }

  matcher_queue_.Stop();
  verifier_queue_.Stop();
  guided_matcher_queue_.Stop();
  output_queue_.Stop();

  for (auto& matcher : matchers_) {
    matcher->Wait();
  }

  for (auto& verifier : verifiers_) {
    verifier->Wait();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->Wait();
  }
}

void FeatureMatcherController::StopQueues() {
  matcher_queue_.Stop();
  verifier_queue_.Stop();
  guided_matcher_queue_.Stop();
  output_queue_.Stop();
}

void FeatureMatcherController::InitializeBudget() {
#if defined(COLMAP_MVS_ENABLED)
  const auto host_capacity = std::min(matching_options_.host_budget_bytes,
      mvs::owned::ObserveFixedCohortHostAvailable());
  resources_->host = std::make_shared<MatchingLedger>(host_capacity,
      matching_options_.policy_host_bytes, matching_options_.reserve_host_bytes);
  cache_->AccessDatabase([&](Database& database) {
    resources_->inventory = MatchingInventory::Read(database, matching_options_);
  });
  cache_->SetMatchingShapeValidator([resources = resources_](Database& database, image_t id) {
    const auto actual = database.ReadMatchingFeatureShape(id);
    const auto& frozen = resources->inventory.images.at(id);
    if (actual.keypoint_rows != frozen.keypoint_rows || actual.keypoint_cols != frozen.keypoint_cols ||
        actual.keypoint_bytes != frozen.keypoint_bytes || actual.descriptor_rows != frozen.descriptor_rows ||
        actual.descriptor_cols != frozen.descriptor_cols || actual.descriptor_bytes != frozen.descriptor_bytes ||
        actual.type != frozen.type)
      throw std::runtime_error("matching DB feature shape changed after admission; workers must drain");
  });
  const auto persistent = resources_->inventory.HostPersistent(matching_options_, matchers_.size(), verifiers_.size());
  resources_->persistent_host = resources_->host->TryAcquire(persistent);
  LOG(INFO) << "Feature matching frozen logical host capacity: capacity_bytes=" << host_capacity
            << " host_budget_bytes=" << matching_options_.host_budget_bytes
            << " policy_host_bytes=" << matching_options_.policy_host_bytes
            << " reserve_host_bytes=" << matching_options_.reserve_host_bytes
            << " whole_db_cache_payload_bytes=" << resources_->inventory.cache_payload
            << " index_payload_bytes=" << resources_->inventory.index_payload
            << " persistent_known_bytes=" << persistent
            << "; whole-DB reservation may over-refuse; FAISS/RANSAC/SQLite/allocator workspace UNKNOWN; physical fit UNKNOWN";
  if (matching_options_.use_gpu) {
    if (!matching_options_.device_budget_bytes)
      throw std::invalid_argument("budgeted GPU matching requires observed device budget");
    auto indices = CSVToVector<int>(matching_options_.gpu_index);
    if (indices.size() == 1 && indices[0] == -1) {
      indices.resize(GetNumCudaDevices());
      std::iota(indices.begin(), indices.end(), 0);
    }
    FlatHashSet<int> seen;
    for (const auto index : indices) {
      if (index < 0 || !seen.insert(index).second)
        throw std::invalid_argument("budgeted matching requires distinct native GPU ordinals");
      const auto observation = mvs::owned::ObserveFixedCohortCapacities(index);
      // Context creation can lower host FREE too. It can only LOWER the
      // frozen host ceiling, never manufacture admission credit.
      resources_->host->LowerCapacity(observation.available.host);
      const auto capacity = std::min(matching_options_.device_budget_bytes, observation.available.device);
      auto ledger = std::make_shared<MatchingLedger>(capacity,
          matching_options_.policy_device_bytes, matching_options_.reserve_device_bytes);
      const auto bytes = resources_->inventory.DevicePersistent(matching_options_);
      resources_->persistent_devices.push_back(ledger->TryAcquire(bytes));
      resources_->devices.push_back(ledger);
      LOG(INFO) << "Feature matching frozen logical CUDA capacity: device=" << observation.device
                << " uuid=" << observation.device_uuid << " capacity_bytes=" << capacity
                << " device_budget_bytes=" << matching_options_.device_budget_bytes
                << " policy_device_bytes=" << matching_options_.policy_device_bytes
                << " reserve_device_bytes=" << matching_options_.reserve_device_bytes
                << " persistent_known_bytes=" << bytes
                << "; driver/allocator workspace UNKNOWN; physical fit UNKNOWN";
    }
  } else if (matching_options_.device_budget_bytes) {
    throw std::invalid_argument("device matching budget specified for CPU backend");
  }
  LOG(INFO) << "Feature matching effective logical host capacity after backend observation: capacity_bytes="
            << resources_->host->Capacity();
#else
  throw std::invalid_argument("budgeted matching requires native host observation (MVS build)");
#endif
}

void FeatureMatcherController::WriteOne() {
  if (stopped && stopped()) throw std::runtime_error("matching cancelled");
  failure_->Rethrow();
  auto job = output_queue_.Pop();
  failure_->Rethrow();
  if (!job.IsValid()) throw std::runtime_error("matching output cancelled");
  auto& output = job.Data();
  if (before_write) before_write();
  const auto actual_results = MatchingMultiply(
      MatchingAdd(output.matches.capacity(), output.two_view_geometry.inlier_matches.capacity()), sizeof(FeatureMatch));
  // Known result capacity, not merely vector size. Any unanticipated capacity
  // growth refuses immediately; never wait while workers retain input owners.
  if (actual_results > output.reserved_result_bytes)
    output.credit->Grow(actual_results - output.reserved_result_bytes);
  if (output.matches.size() < static_cast<size_t>(geometry_options_.min_num_inliers))
    output.matches = {};
  cache_->WriteMatches(output.image_id1, output.image_id2, output.matches);
  WriteVerifiedTwoViewGeometry(*cache_, output.image_id1, output.image_id2,
                              std::move(output.two_view_geometry), geometry_options_.min_num_inliers);
  --pending_;
  // Job's payloads die BEFORE its ticket. No release at matcher completion.
}

bool FeatureMatcherController::Setup() {
  if (resources_) InitializeBudget();
  for (auto& matcher : matchers_) {
    matcher->Start();
  }

  for (auto& verifier : verifiers_) {
    verifier->Start();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->Start();
  }

  for (auto& matcher : matchers_) {
    if (!matcher->CheckValidSetup()) {
      if (failure_) failure_->Rethrow();
      return false;
    }
  }

  for (auto& guided_matcher : guided_matchers_) {
    if (!guided_matcher->CheckValidSetup()) {
      return false;
    }
  }

  is_setup_ = true;

  return true;
}

void FeatureMatcherController::Match(
    const std::vector<std::pair<image_t, image_t>>& image_pairs) {
  THROW_CHECK_NOTNULL(cache_);
  THROW_CHECK(is_setup_);

  if (image_pairs.empty()) {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Match the image pairs
  //////////////////////////////////////////////////////////////////////////////

  FlatHashSet<image_pair_t> image_pair_ids;
  image_pair_ids.reserve(image_pairs.size());

  size_t num_outputs = 0;
  for (const auto& [image_id1, image_id2] : image_pairs) {
    // Avoid self-matches.
    if (image_id1 == image_id2) {
      continue;
    }

    // Avoid duplicate image pairs.
    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    if (!image_pair_ids.insert(pair_id).second) {
      continue;
    }

    // Avoid self-matches within a frame.
    if (matching_options_.skip_image_pairs_in_same_frame) {
      const Image& image1 = cache_->GetImage(image_id1);
      const Image& image2 = cache_->GetImage(image_id2);
      if (image1.HasFrameId() && image2.HasFrameId() &&
          image1.FrameId() == image2.FrameId()) {
        continue;
      }
    }

    const bool exists_matches = cache_->ExistsMatches(image_id1, image_id2);
    const bool exists_two_view_geometry =
        cache_->ExistsTwoViewGeometry(image_id1, image_id2);

    if (exists_matches && exists_two_view_geometry) {
      continue;
    }

    // Without geometric verification, no two-view geometry is ever stored, so
    // existing matches are all there is to compute for the pair. Pushing it to
    // the verifier queue below would block forever, because no verifier
    // workers exist in this configuration.
    if (exists_matches && skip_geometric_verification_) {
      continue;
    }

    std::shared_ptr<MatchingLedger::Ticket> credit;
    if (resources_) {
      if (stopped && stopped()) throw std::runtime_error("matching cancelled");
      failure_->Rethrow();
      const auto bytes = resources_->inventory.PairPayload(image_id1, image_id2, matching_options_);
      while (!(credit = resources_->host->TryAcquire(bytes))) {
        if (!pending_) throw std::runtime_error("matching logical singleton refused with persistent cache payload");
        LOG(INFO) << "Feature matching admission backpressure: pending_pairs=" << pending_;
        if (admission_wait) admission_wait();
        WriteOne();
      }
      LOG(INFO) << "Feature matching admitted pair: image_id1=" << image_id1
                << " image_id2=" << image_id2 << " known_payload_bytes=" << bytes
                << " ledger_used_bytes=" << resources_->host->used.load()
                << " ledger_peak_bytes=" << resources_->host->peak;
    }

    num_outputs += 1;

    // If only one of the matches or inlier matches exist, we recompute them
    // from scratch and delete the existing results. This must be done before
    // pushing the jobs to the queue, otherwise database constraints might fail
    // when writing an existing result into the database.
    if (exists_two_view_geometry) {
      cache_->DeleteTwoViewGeometry(image_id1, image_id2);
    }

    FeatureMatcherData data;
    data.credit = std::move(credit);
    if (resources_) {
      auto count = matching_options_.use_gpu ? resources_->inventory.MatchLimit(matching_options_)
          : resources_->inventory.images.at(image_id1).keypoint_rows;
      const auto existing = resources_->inventory.stored_matches.find(pair_id);
      if (existing != resources_->inventory.stored_matches.end()) count = std::max(count, existing->second);
      data.reserved_result_bytes = MatchingMultiply(count, 2 * sizeof(FeatureMatch));
    }
    data.image_id1 = image_id1;
    data.image_id2 = image_id2;

    if (exists_matches) {
      data.matches = cache_->GetMatches(image_id1, image_id2);
      cache_->DeleteMatches(image_id1, image_id2);
      THROW_CHECK(verifier_queue_.Push(std::move(data)));
    } else {
      THROW_CHECK(matcher_queue_.Push(std::move(data)));
    }
    if (resources_) ++pending_;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Write results to database
  //////////////////////////////////////////////////////////////////////////////

  if (resources_) {
    while (pending_) WriteOne();
    failure_->Rethrow();
    return;
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_job = output_queue_.Pop();
    THROW_CHECK(output_job.IsValid());
    auto& output = output_job.Data();

    if (output.matches.size() <
        static_cast<size_t>(geometry_options_.min_num_inliers)) {
      output.matches = {};
    }

    cache_->WriteMatches(output.image_id1, output.image_id2, output.matches);

    WriteVerifiedTwoViewGeometry(*cache_,
                                 output.image_id1,
                                 output.image_id2,
                                 std::move(output.two_view_geometry),
                                 geometry_options_.min_num_inliers);
  }

  THROW_CHECK_EQ(output_queue_.Size(), 0);
}

GeometricVerifierController::GeometricVerifierController(
    const GeometricVerifierOptions& options,
    const TwoViewGeometryOptions& geometry_options,
    std::shared_ptr<FeatureMatcherCache> cache)
    : geometry_options_(geometry_options),
      cache_(std::move(cache)),
      options_(options),
      is_setup_(false) {
  THROW_CHECK(geometry_options_.Check());

  const int num_threads = GetEffectiveNumThreads(options_.num_threads);

  // Run geometric verification
  for (int i = 0; i < num_threads; ++i) {
    verifiers_.emplace_back(
        std::make_unique<VerifierWorker>(geometry_options_,
                                         cache_,
                                         &verifier_queue_,
                                         &output_queue_,
                                         options_.use_existing_relative_pose));
  }
}

GeometricVerifierController::~GeometricVerifierController() {
  verifier_queue_.Wait();
  output_queue_.Wait();

  for (auto& verifier : verifiers_) {
    verifier->Stop();
  }

  verifier_queue_.Stop();
  output_queue_.Stop();

  for (auto& verifier : verifiers_) {
    verifier->Wait();
  }
}

const GeometricVerifierOptions& GeometricVerifierController::Options() const {
  return options_;
}

GeometricVerifierOptions& GeometricVerifierController::Options() {
  return options_;
}

bool GeometricVerifierController::Setup() {
  for (auto& verifier : verifiers_) {
    verifier->Start();
  }

  is_setup_ = true;
  return true;
}

void GeometricVerifierController::Verify(
    const std::vector<std::pair<image_t, image_t>>& image_pairs) {
  THROW_CHECK_NOTNULL(cache_);
  THROW_CHECK(is_setup_);

  if (image_pairs.empty()) {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Verify the matches from the image pairs
  //////////////////////////////////////////////////////////////////////////////

  FlatHashSet<image_pair_t> image_pair_ids;
  image_pair_ids.reserve(image_pairs.size());

  size_t num_outputs = 0;
  for (const auto& [image_id1, image_id2] : image_pairs) {
    // Avoid self-matches.
    if (image_id1 == image_id2) {
      continue;
    }

    // Avoid duplicate image pairs.
    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    if (!image_pair_ids.insert(pair_id).second) {
      continue;
    }

    const bool exists_matches = cache_->ExistsMatches(image_id1, image_id2);
    const bool exists_inlier_matches =
        cache_->ExistsInlierMatches(image_id1, image_id2);

    if (exists_matches && exists_inlier_matches) {
      continue;
    }

    num_outputs += 1;

    // If only one of the matches or inlier matches exist, we recompute them
    // from scratch and delete the existing results. This must be done before
    // pushing the jobs to the queue, otherwise database constraints might fail
    // when writing an existing result into the database.
    if (exists_inlier_matches) {
      cache_->DeleteTwoViewGeometry(image_id1, image_id2);
    }

    FeatureMatcherData data;
    data.image_id1 = image_id1;
    data.image_id2 = image_id2;

    if (exists_matches) {
      data.matches = cache_->GetMatches(image_id1, image_id2);
      // There exists a two view geometry without inlier matches.
      if (cache_->ExistsTwoViewGeometry(image_id1, image_id2)) {
        data.two_view_geometry =
            cache_->GetTwoViewGeometry(image_id1, image_id2);
      }
      THROW_CHECK(verifier_queue_.Push(std::move(data)));
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Write results to database
  //////////////////////////////////////////////////////////////////////////////

  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_job = output_queue_.Pop();
    THROW_CHECK(output_job.IsValid());
    auto& output = output_job.Data();

    if (output.matches.size() <
        static_cast<size_t>(geometry_options_.min_num_inliers)) {
      output.matches = {};
    }

    if (cache_->ExistsTwoViewGeometry(output.image_id1, output.image_id2)) {
      cache_->DeleteTwoViewGeometry(output.image_id1, output.image_id2);
    }

    WriteVerifiedTwoViewGeometry(*cache_,
                                 output.image_id1,
                                 output.image_id2,
                                 std::move(output.two_view_geometry),
                                 geometry_options_.min_num_inliers);
  }

  THROW_CHECK_EQ(output_queue_.Size(), 0);
}

}  // namespace colmap
