// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#if !defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/owned_graph_coordinator.h"
#include "colmap/mvs/owned_graph_topology.h"

#include <exception>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace colmap::mvs::owned {
class GraphLaunchControl;
// One orientation, one owner. Resources/stream must outlive every launch.
// Caller fences before destruction/rebinding resources. Parameter updates
// affect subsequent launches only. No implicit device selection.
class GraphBuilder {
 public:
  GraphBuilder();
  ~GraphBuilder();
  GraphBuilder(const GraphBuilder&) = delete;
  GraphBuilder& operator=(const GraphBuilder&) = delete;

  template <typename Kernel, typename... Args>
  void Launch(Kernel kernel,
              dim3 grid,
              dim3 block,
              cudaStream_t stream,
              bool filtered,
              Args... args) {
    Prepare(kernel, grid, block, filtered, args...);
    LaunchPrepared(stream);
  }

  template <typename Kernel, typename... Args>
  void LaunchOrSubmit(Kernel kernel,
                      dim3 grid,
                      dim3 block,
                      cudaStream_t stream,
                      bool filtered,
                      GraphLaunchControl* coordinator,
                      size_t owner,
                      Args... args) {
    Prepare(kernel, grid, block, coordinator ? false : filtered, args...);
    if (coordinator) {
      SubmitPrepared(*coordinator, owner, stream);
    } else {
      LaunchPrepared(stream);
    }
  }

  // Owner-private parameter binding without submission. Rolling PHOTO uses
  // direct nodes even for the final filter specialization (no nested IF body).
  template <typename Kernel, typename... Args>
  void Prepare(
      Kernel kernel, dim3 grid, dim3 block, bool filtered, Args... args) {
    WithKernelArguments(
        [&](void** pointers) {
          cudaKernelNodeParams params{};
          params.func = reinterpret_cast<void*>(kernel);
          params.gridDim = grid;
          params.blockDim = block;
          params.kernelParams = pointers;
          Update(params, filtered);
        },
        args...);
  }

  // Standalone clone/composition utility ONLY; never called by chained PHOTO.
  // Refresh the source node before Get()/InstantiateCopies() when parameters
  // differ from the initial Prepare. Executable updates intentionally do not
  // maintain source snapshots. Refuses nested conditional bodies.
  void SetStandaloneSnapshot(const cudaKernelNodeParams& params) const;

  // Instantiates independent executables from the source-graph snapshot, not
  // the latest executable update. Refresh with SetStandaloneSnapshot first.
  // Rebind copies with SetCopyParams for real concurrent owners.
  std::vector<cudaGraphExec_t> InstantiateCopies(size_t count) const;
  // Bind a clone's distinct owner resources before concurrent launch. Clone
  // must originate from this builder; nested conditional bodies are refused.
  void SetCopyParams(cudaGraphExec_t copy,
                     const cudaKernelNodeParams& params) const;
  // Standalone composition snapshot (initial Prepare or explicit refresh).
  cudaGraph_t Get() const { return graph_; }
  static bool HasConditionalNodes();
  size_t Instantiations() const { return instantiations_; }
  size_t Updates() const { return updates_; }
  size_t Launches() const { return launches_; }

 private:
  void Update(const cudaKernelNodeParams&, bool);
  void LaunchPrepared(cudaStream_t);
  void SubmitPrepared(GraphLaunchControl&, size_t, cudaStream_t);
  cudaGraph_t graph_ = nullptr;
  cudaGraphNode_t kernel_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  bool conditional_ = false;
  size_t instantiations_ = 0, updates_ = 0, launches_ = 0;
};

// Shared cancellation state only. Launch immediately on the caller's stream;
// no cross-owner synchronization. Owner fences before destroying bindings.
class GraphLaunchControl {
 public:
  explicit GraphLaunchControl(size_t owners) : state_(owners) {}
  void Launch(size_t owner, cudaGraphExec_t exec, cudaStream_t stream);
  void Cancel(std::exception_ptr error) { state_.Cancel(error); }

 private:
  OwnerGraphLaunchState state_;
};

// Copies owner graphs as child nodes and adds dependency edges in the parent.
// All children must belong to the current device and use distinct mutable
// state. Caller owns returned graph and any executables instantiated from it.
cudaGraph_t ComposeOwnerGraphs(
    const std::vector<cudaGraph_t>& owners,
    const std::vector<OwnerDependency>& dependencies);
}  // namespace colmap::mvs::owned
#endif
