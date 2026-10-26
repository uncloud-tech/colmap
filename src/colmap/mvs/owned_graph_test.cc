// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_graph_topology.h"

#include <cassert>
#include <stdexcept>
#include <vector>

// Dependency-free CPU executable, also usable on hosts without CUDA/GTest.
int main() {
  using namespace colmap::mvs::owned;
  const auto plan = MakeOwnerGraphTopology(4, {{0, 2}, {1, 2}, {2, 3}});
  assert(plan.owners == 4);
  assert(plan.dependencies.size() == 3);
  assert(plan.dependencies[1].first == 1);
  assert(plan.dependencies[1].second == 2);
  assert(MakeOwnerGraphTopology(2, {}).dependencies.empty());
  for (const auto& edges : std::vector<std::vector<OwnerDependency>>{
           {{0, 0}}, {{0, 2}}, {{0, 1}, {1, 0}}, {{0, 1}, {0, 1}}}) {
    bool refused = false;
    try {
      MakeOwnerGraphTopology(2, edges);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused);
  }
  bool refused = false;
  try {
    MakeOwnerGraphTopology(0, {});
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  assert(refused);

  // CUDA's add/update calls synchronously copy the pointed-to argument values.
  // Exercise the same packing seam with a CPU consumer, including a pointer.
  int storage = 17;
  int calls = 0;
  auto consume = [&](void** args) {
    assert(*static_cast<int*>(args[0]) == (calls == 0 ? 3 : 9));
    assert(*static_cast<int**>(args[1]) == &storage);
    assert(*static_cast<float*>(args[2]) == (calls == 0 ? 0.5f : 0.25f));
    ++calls;
  };
  WithKernelArguments(consume, 3, &storage, 0.5f);
  WithKernelArguments(consume, 9, &storage, 0.25f);
  assert(calls == 2);
}
