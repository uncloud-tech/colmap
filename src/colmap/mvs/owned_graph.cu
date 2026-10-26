// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_graph.h"
#include "colmap/mvs/owned_graph_coordinator.h"

#include <stdexcept>
#include <string>

namespace colmap::mvs::owned {
namespace {
void Check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}  // namespace

GraphBuilder::GraphBuilder() { Check(cudaGraphCreate(&graph_, 0)); }
GraphBuilder::~GraphBuilder() {
  if (exec_) (void)cudaGraphExecDestroy(exec_);
  if (graph_) (void)cudaGraphDestroy(graph_);
}
bool GraphBuilder::HasConditionalNodes() {
#if CUDART_VERSION >= 12040
  int driver = 0;
  Check(cudaDriverGetVersion(&driver));
  return driver >= 12040;
#else
  return false;
#endif
}

void GraphBuilder::Update(const cudaKernelNodeParams& params, bool filtered) {
  // Filter specialization is used once at the end of Run(). Rebuild on a
  // subsequent Run: nested conditional-body exec parameter update is not used.
  if (exec_ && conditional_) {
    Check(cudaGraphExecDestroy(exec_));
    exec_ = nullptr;
    Check(cudaGraphDestroy(graph_));
    graph_ = nullptr;
    Check(cudaGraphCreate(&graph_, 0));
    kernel_ = nullptr;
  }
  if (!exec_) {
    cudaGraph_t body = graph_;
    conditional_ = filtered && HasConditionalNodes();
#if CUDART_VERSION >= 12040
    if (conditional_) {
      cudaGraphConditionalHandle handle;
      Check(cudaGraphConditionalHandleCreate(
          &handle, graph_, 1, cudaGraphCondAssignDefault));
      cudaGraphNodeParams condition{};
      condition.type = cudaGraphNodeTypeConditional;
      condition.conditional.handle = handle;
      condition.conditional.type = cudaGraphCondTypeIf;
      condition.conditional.size = 1;
      cudaGraphNode_t node;
      Check(cudaGraphAddNode(&node, graph_, nullptr, 0, &condition));
      body = condition.conditional.phGraph_out[0];
    }
#endif
    Check(cudaGraphAddKernelNode(&kernel_, body, nullptr, 0, &params));
    Check(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
    ++instantiations_;
  } else {
    Check(cudaGraphExecKernelNodeSetParams(exec_, kernel_, &params));
    ++updates_;
    // Chained production sweeps update the executable ONLY. Source-graph
    // snapshots belong to the explicit standalone clone/composition helper.
  }
}

void GraphBuilder::LaunchPrepared(cudaStream_t stream) {
  Check(cudaGraphLaunch(exec_, stream));
  ++launches_;
}

void GraphBuilder::SubmitPrepared(GraphLaunchControl& coordinator,
                                  size_t owner,
                                  cudaStream_t stream) {
  coordinator.Launch(owner, exec_, stream);
  ++launches_;
}

void GraphBuilder::SetStandaloneSnapshot(
    const cudaKernelNodeParams& params) const {
  if (!exec_ || conditional_)
    throw std::invalid_argument("Unbuilt/conditional standalone graph");
  Check(cudaGraphKernelNodeSetParams(kernel_, &params));
}

std::vector<cudaGraphExec_t> GraphBuilder::InstantiateCopies(
    size_t count) const {
  if (!exec_ || count == 0) throw std::invalid_argument("Unbuilt/empty graph");
  std::vector<cudaGraphExec_t> copies(count, nullptr);
  try {
    for (auto& copy : copies) {
      Check(cudaGraphInstantiate(&copy, graph_, nullptr, nullptr, 0));
    }
  } catch (...) {
    for (auto copy : copies)
      if (copy) (void)cudaGraphExecDestroy(copy);
    throw;
  }
  return copies;
}

void GraphBuilder::SetCopyParams(cudaGraphExec_t copy,
                                 const cudaKernelNodeParams& params) const {
  if (!exec_ || conditional_ || !copy) {
    throw std::invalid_argument("Unbuilt/conditional graph or null clone");
  }
  Check(cudaGraphExecKernelNodeSetParams(copy, kernel_, &params));
}

cudaGraph_t ComposeOwnerGraphs(
    const std::vector<cudaGraph_t>& owners,
    const std::vector<OwnerDependency>& dependencies) {
  const auto topology = MakeOwnerGraphTopology(owners.size(), dependencies);
  cudaGraph_t graph = nullptr;
  Check(cudaGraphCreate(&graph, 0));
  try {
    std::vector<cudaGraphNode_t> children(owners.size());
    for (size_t i = 0; i < owners.size(); ++i) {
      Check(cudaGraphAddChildGraphNode(
          &children[i], graph, nullptr, 0, owners[i]));
    }
    for (const auto& edge : topology.dependencies) {
      Check(cudaGraphAddDependencies(
          graph, &children[edge.first], &children[edge.second], 1));
    }
  } catch (...) {
    (void)cudaGraphDestroy(graph);
    throw;
  }
  return graph;
}
void GraphLaunchControl::Launch(size_t owner,
                                cudaGraphExec_t exec,
                                cudaStream_t stream) {
  if (!exec) throw std::invalid_argument("Null graph executable");
  state_.Check(owner);
  // Cancellation racing this call may leave one final launch queued. The
  // caller always drains its own stream before releasing owner resources.
  Check(cudaGraphLaunch(exec, stream));
}
}  // namespace colmap::mvs::owned
