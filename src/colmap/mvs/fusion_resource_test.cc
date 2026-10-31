// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/fusion_resource.h"

#include <cassert>
#include <limits>

int main() {
  using namespace colmap::mvs;
  FusionInputInventory inputs;
  inputs.AddImage(2, 3);
  inputs.AddImage(1, 10);
  // First image: 30. Second: retained prefix 24 + transient 50.
  assert(inputs.retained_bytes == 64);
  assert(inputs.peak_bytes == 74);
  assert(inputs.mapped_bytes == 256);
  bool overflow = false;
  try {
    inputs.AddImage(std::numeric_limits<size_t>::max(), 2);
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  assert(overflow);
  assert(inputs.retained_bytes == 64);
  bool invalid = false;
  try {
    inputs.AddImage(0, 10);
  } catch (const std::invalid_argument&) {
    invalid = true;
  }
  assert(invalid);
  overflow = false;
  try {
    FusionLogicalBudget(std::numeric_limits<uint64_t>::max(), 1, 0)
        .Check(std::numeric_limits<uint64_t>::max(), 0);
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  assert(overflow);
  FusionLogicalBudget exact(114, 26, 11);
  exact.Check(74, 0);  // ceil(100 * 1.03) + 11 == 114.
  bool refused = false;
  try {
    FusionLogicalBudget(113, 26, 11).Check(74, 0);
  } catch (const FusionResourceError&) {
    refused = true;
  }
  assert(refused);

  // One int fits at 5 bytes; replacing capacity one by two must account
  // for both old 4-byte and new 8-byte buffers before reserve allocates.
  FusionOutputLedger ledger(FusionLogicalBudget(12, 0, 0), 0);
  std::vector<int> points;
  ledger.Reserve(points, 1);
  assert(ledger.Bytes() == 4);
  refused = false;
  try {
    ledger.Reserve(points, 2);
  } catch (const FusionResourceError&) {
    refused = true;
  }
  assert(refused);
  assert(points.capacity() == 1);
  assert(ledger.Bytes() == 4);
  // Merge must retain task storage while allocating final storage.
  FusionOutputLedger merge(FusionLogicalBudget(8, 0, 0), 0);
  std::vector<int> task;
  std::vector<int> final;
  merge.Reserve(task, 1);
  refused = false;
  try {
    merge.Reserve(final, 1);
  } catch (const FusionResourceError&) {
    refused = true;
  }
  assert(refused);
  assert(final.capacity() == 0);
  merge.Release(task);
  merge.Reserve(final, 1);
  assert(merge.Bytes() == 4);
  // Moved inner visibility allocations stay charged through final merge.
  FusionOutputLedger visibility(FusionLogicalBudget(1000, 0, 0), 0);
  std::vector<std::vector<int>> task_visibility;
  std::vector<std::vector<int>> final_visibility;
  std::vector<int> indices;
  visibility.Reserve(task_visibility, 1);
  visibility.Reserve(indices, 2);
  indices.push_back(3);
  indices.push_back(7);
  task_visibility.push_back(std::move(indices));
  visibility.Reserve(final_visibility, 1);
  final_visibility.push_back(std::move(task_visibility.front()));
  visibility.Release(task_visibility);
  assert(visibility.Bytes() == sizeof(std::vector<int>) + 8);
  assert(final_visibility.front() == std::vector<int>({3, 7}));
}
