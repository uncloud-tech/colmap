// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace colmap::mvs::owned {
// Consumer must copy values synchronously; pointers are valid only in the call.
template <typename Consumer, typename... Args>
void WithKernelArguments(Consumer&& consume, Args... args) {
  auto values = std::make_tuple(args...);
  std::apply(
      [&](auto&... value) {
        std::array<void*, sizeof...(Args)> pointers{
            static_cast<void*>(&value)...};
        consume(pointers.data());
      },
      values);
}

using OwnerDependency = std::pair<size_t, size_t>;
struct OwnerGraphTopology {
  size_t owners;
  std::vector<OwnerDependency> dependencies;
};

inline OwnerGraphTopology MakeOwnerGraphTopology(
    size_t owners, const std::vector<OwnerDependency>& dependencies) {
  if (owners == 0) throw std::invalid_argument("Empty owner graph");
  std::vector<size_t> indegree(owners, 0);
  std::vector<std::vector<size_t>> successors(owners);
  for (const auto& edge : dependencies) {
    if (edge.first >= owners || edge.second >= owners ||
        edge.first == edge.second) {
      throw std::invalid_argument("Invalid owner dependency");
    }
    for (size_t next : successors[edge.first]) {
      if (next == edge.second) throw std::invalid_argument("Duplicate edge");
    }
    successors[edge.first].push_back(edge.second);
    ++indegree[edge.second];
  }
  std::vector<size_t> ready;
  for (size_t i = 0; i < owners; ++i) {
    if (indegree[i] == 0) ready.push_back(i);
  }
  for (size_t i = 0; i < ready.size(); ++i) {
    for (size_t next : successors[ready[i]]) {
      if (--indegree[next] == 0) ready.push_back(next);
    }
  }
  if (ready.size() != owners) throw std::invalid_argument("Cyclic owner graph");
  return {owners, dependencies};
}
}  // namespace colmap::mvs::owned
