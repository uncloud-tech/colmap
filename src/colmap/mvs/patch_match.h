// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/patch_match_execution.h"
#include "colmap/mvs/resident_contract.h"
#ifndef __CUDACC__
#include "colmap/util/base_controller.h"
#include "colmap/util/threading.h"
#endif

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace colmap {
namespace mvs {

class ConsistencyGraph;
class PatchMatchCuda;
class Workspace;
struct OwnedBankContract;
struct OwnedBankContractProblem;
namespace owned {
class PairCoordinator;
class HostInputBank;
class ResidentInputBank;
struct BankJob;
}
#if !defined(COLMAP_HIP_ENABLED)
namespace owned {
class PatchMatchCuda;
}
#endif

// This is a wrapper class around the actual PatchMatchCuda implementation. This
// class is necessary to hide Cuda code from any boost or Eigen code, since
// NVCC/MSVC cannot compile complex C++ code.
class PatchMatch {
 public:
  struct Problem {
    // Index of the reference image.
    int ref_image_idx = -1;

    // Indices of the source images.
    std::vector<int> src_image_idxs;

    // Input images for the photometric consistency term.
    std::vector<Image>* images = nullptr;

    // Input depth maps for the geometric consistency term.
    std::vector<DepthMap>* depth_maps = nullptr;

    // Input normal maps for the geometric consistency term.
    std::vector<NormalMap>* normal_maps = nullptr;

    // Invocation-owned immutable GPU inputs; metadata stays per-engine.
    const owned::ResidentInputBank* resident_bank = nullptr;
    int resident_ref_layer = -1;
    std::vector<int> resident_src_layers;
    const NormalMap* resident_ref_normal = nullptr;

    // Print the configuration to stdout.
    void Print() const;
  };

  PatchMatch(const PatchMatchOptions& options,
             const Problem& problem,
             PatchMatchExecution execution = PatchMatchExecution::kStock);
  ~PatchMatch();

  // In owned mode, Problem pointers refer to this engine's input snapshots.
  const PatchMatchOptions& GetOptions() const { return options_; }
  const Problem& GetProblem() const { return problem_; }

  // Check the options and the problem for validity.
  void Check() const;

  // Run the patch match algorithm.
  void Run();

  // Experiment-only owned lifecycle. Prepare drains constructor work; solver
  // dispatch can then be coordinated without constructing another backend.
  void PrepareOwned(int owned_tile_threads = 32);
  void RunPreparedOwned();
  // Failure-path completion proof while this instance still owns its buffers.
  void DrainOwned() noexcept;
  void SetOwnedSweepObserver(std::function<void(int, int)> observer);

  // Get the computed values after running the algorithm.
  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  ConsistencyGraph GetConsistencyGraph() const;
  Mat<float> GetSelProbMap() const;

 private:
  const PatchMatchOptions options_;
  const PatchMatchExecution execution_;
  // Owned mode snapshots inputs before binding Problem pointers. Backend is
  // destroyed before this storage; stock mode retains its original borrowing.
  std::vector<Image> owned_images_;
  std::vector<DepthMap> owned_depth_maps_;
  std::vector<NormalMap> owned_normal_maps_;
  const Problem problem_;
  std::unique_ptr<PatchMatchCuda> patch_match_cuda_;
#if !defined(COLMAP_HIP_ENABLED)
  std::unique_ptr<owned::PatchMatchCuda> owned_patch_match_cuda_;
  bool owned_run_started_ = false;
  bool owned_run_completed_ = false;
#endif
};

// This thread processes all problems in a workspace. A workspace has the
// following file structure, if the workspace format is "COLMAP":
//
//    images/*
//    sparse/{cameras.txt, images.txt, points3D.txt}
//    stereo/
//      depth_maps/*
//      normal_maps/*
//      consistency_graphs/*
//      patch-match.cfg
//
// The `patch-match.cfg` file specifies the images to be processed as:
//
//    image_name1.jpg
//    __all__
//    image_name2.jpg
//    __auto__, 20
//    image_name3.jpg
//    image_name1.jpg, image_name2.jpg
//
// Two consecutive lines specify the images used to compute one patch match
// problem. The first line specifies the reference image and the second line the
// source images. Image names are relative to the `images` directory. In this
// example, the first reference image uses all other images as source images,
// the second reference image uses the 20 most connected images as source
// images, and the third reference image uses the first and second as source
// images. Note that all specified images must be reconstructed in the COLMAP
// reconstruction provided in the `sparse` folder.

#ifndef __CUDACC__

class PatchMatchController : public BaseController {
 public:
  struct OwnedPairHooks {
    // Copied from the immutable batch configuration into each CUDA owner.
    int owned_tile_threads = 32;
    std::function<void(size_t, const PatchMatchOptions&, const PatchMatch::Problem&)> check_inputs;
    std::function<void(size_t, const PatchMatchOptions&, const PatchMatch::Problem&)> metadata;
    std::function<void(size_t, int, int)> sweep;
    std::function<void(size_t, const char*)> output_phase;
  };
  struct OwnedPreparedJob {
    std::unique_ptr<PatchMatch> patch_match;
    std::filesystem::path depth_map_path;
    std::filesystem::path normal_map_path;
  };

  PatchMatchController(const PatchMatchOptions& options,
                       const std::filesystem::path& workspace_path,
                       const std::string& workspace_format,
                       const std::string& pmvs_option_name,
                       const std::filesystem::path& config_path = "",
                       PatchMatchExecution execution = PatchMatchExecution::kStock);
  void Run();
  // Synchronous owner-thread path; caller retains controller through join and
  // retires its reservation only after this controller has been destroyed.
  void RunOwnedPairJob(owned::PairCoordinator& coordinator,
                       size_t job,
                       const OwnedPairHooks& hooks);

  // Planned fresh-controller PHOTO / GEO graph before PHOTO maps exist.
  // This is NOT a bank contract or admission: actual GEO inputs/order must be
  // revalidated through ResolveOwnedBankContract after the PHOTO barrier.
  // No solver, device/context queries, payload decode, or workspace writes.
  struct DensePlanGraph {
    std::vector<Image> images;
    std::vector<std::string> names;
    std::vector<owned::BankJob> photo;
    std::vector<owned::BankJob> geo;
  };
  DensePlanGraph ResolveDensePlanGraph();

  // Resolve native effective order once, without decoding or touching CUDA.
  owned::BankJob ResolveOwnedBankJob();
  // Shared per-problem metadata helper: resolves EVERY configured problem's
  // effective invocation snapshot once, preserving the existing
  // FlatHashSet-produced insertion/iteration/source order without sorting or
  // canonical reorder. CPU-only metadata path on a stock-dispatch controller:
  // no payload decode, no CUDA, no workspace writes. Rejects duplicate
  // references/output aliases.
  std::vector<owned::BankJob> ResolveOwnedBankJobs();
  // Enriched contract capture for the CLI metadata export/verify modes.
  OwnedBankContract ResolveOwnedBankContract();
  // The single authoritative invocation snapshot for the resident runner:
  // ONE ResolveOwnedBankJobsImpl traversal produces the enriched contract
  // AND the resolved jobs together, so both the wire binding bytes and the
  // ledger minima derive from the same once-per-invocation resolution in
  // native problem order. Callers must serialize the contract once and
  // never re-resolve for a second "snapshot".
  struct OwnedBankResolved {
    OwnedBankContract contract;
    std::vector<owned::BankJob> jobs;
  };
  OwnedBankResolved ResolveOwnedBankContractWithJobs();
  void RunResidentOwnedPairJob(const owned::HostInputBank& host,
                               const owned::ResidentInputBank& resident,
                               owned::PairCoordinator& coordinator,
                               size_t job,
                               const OwnedPairHooks& hooks);

  // Compact owner generation may select a different immutable full-bank job.
  // `owned_ref_normal` is required for bounded-normal banks (the owned map is
  // handed off by the bounded lookahead and stays alive in the owner's frame
  // until upload and owner retirement) and must be null for eager banks,
  // which retain the validated reference normal internally. Problem pointers
  // refer to that caller-owned storage; no dangling pointer or early free.
  void RunResidentOwnedPairJob(const owned::HostInputBank& host,
                               const owned::ResidentInputBank& resident,
                               owned::PairCoordinator& coordinator,
                               size_t job,
                               size_t bank_job_index,
                               const OwnedPairHooks& hooks,
                               const NormalMap* owned_ref_normal = nullptr);

  // CPU-only preparation. No coordinator, events, or CUDA calls; all workspace
  // and decode owners are released before returning the owned engine.
  std::unique_ptr<OwnedPreparedJob> PrepareOwnedPairJob(
      size_t job, const OwnedPairHooks& hooks, int preparation_threads = 0);
  // Consumes a prefetched engine on a fresh owner-thread controller, acquiring
  // the normal preparation lease before the first CUDA backend construction.
  void RunPreparedOwnedPairJob(std::unique_ptr<OwnedPreparedJob> prepared,
                               owned::PairCoordinator& coordinator,
                               size_t job,
                               const OwnedPairHooks& hooks);

 private:
  void ExecutePreparedOwnedPairJob(OwnedPreparedJob& prepared,
                                   owned::PairCoordinator& coordinator,
                                   size_t job,
                                   const OwnedPairHooks& hooks);
  // Shared invocation setup guards and single-traversal resolution loop.
  // Execution-specific guards live in the public entry points. When
  // `contract` is non-null, the enriched per-problem contract records are
  // captured during the same, once-per-invocation traversal. When
  // `expected_problems` is non-null, the exact configured problem count is
  // enforced before any traversal (one-reference wrapper guard).
  std::vector<owned::BankJob> ResolveOwnedBankJobsImpl(
      OwnedBankContract* contract,
      const size_t* expected_problems = nullptr);
  owned::BankJob CaptureOwnedBankJobMetadata(
      size_t problem_idx,
      const std::vector<std::string>& image_names,
      OwnedBankContractProblem* contract);
  void ReadWorkspace();
  void ReadProblems();
  void ReadGpuIndices();
  void ProcessProblem(const PatchMatchOptions& options, size_t problem_idx);
  void ProcessProblemInputs(const PatchMatchOptions& options,
                            size_t problem_idx,
                            OwnedPreparedJob* prepared,
                            size_t job,
                            const OwnedPairHooks& hooks);

  const PatchMatchOptions options_;
  const PatchMatchExecution execution_;
  const std::filesystem::path workspace_path_;
  const std::string workspace_format_;
  const std::string pmvs_option_name_;
  const std::filesystem::path config_path_;

  bool owned_pair_started_ = false;
  bool owned_bank_resolved_ = false;

  std::unique_ptr<ThreadPool> thread_pool_;
  std::unique_ptr<ThreadPool> io_thread_pool_;
  std::mutex workspace_mutex_;
  std::unique_ptr<Workspace> workspace_;
  std::vector<PatchMatch::Problem> problems_;
  std::vector<int> gpu_indices_;
  std::vector<std::pair<float, float>> depth_ranges_;
};

#endif

}  // namespace mvs
}  // namespace colmap
