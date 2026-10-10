// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/image.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/mvs/patch_match_execution.h"
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
namespace owned {
class PairCoordinator;
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

    // Print the configuration to stdout.
    void Print() const;
  };

  PatchMatch(const PatchMatchOptions& options,
             const Problem& problem,
             PatchMatchExecution execution = PatchMatchExecution::kStock);
  ~PatchMatch();

  // Check the options and the problem for validity.
  void Check() const;

  // Run the patch match algorithm.
  void Run();

  // Experiment-only owned lifecycle. Prepare drains constructor work; solver
  // dispatch can then be coordinated without constructing another backend.
  void PrepareOwned();
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
    std::function<void(size_t, const PatchMatchOptions&, const PatchMatch::Problem&)> check_inputs;
    std::function<void(size_t, const PatchMatchOptions&, const PatchMatch::Problem&)> metadata;
    std::function<void(size_t, int, int)> sweep;
    std::function<void(size_t, const char*)> output_phase;
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

 private:
  void ReadWorkspace();
  void ReadProblems();
  void ReadGpuIndices();
  void ProcessProblem(const PatchMatchOptions& options, size_t problem_idx);

  const PatchMatchOptions options_;
  const PatchMatchExecution execution_;
  const std::filesystem::path workspace_path_;
  const std::string workspace_format_;
  const std::string pmvs_option_name_;
  const std::filesystem::path config_path_;

  owned::PairCoordinator* owned_pair_ = nullptr;
  size_t owned_pair_job_ = 0;
  OwnedPairHooks owned_pair_hooks_;

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
