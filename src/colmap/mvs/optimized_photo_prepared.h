// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>

namespace colmap::mvs {
// Pre-execution guard, deliberately independent of CUDA/solver construction.
// Compare against the effective options observed during THIS preparation, not
// controller defaults (depth bounds, GPU and spatial sigma may be resolved).
template <class Prepared, class Controller, class Serialize>
void CheckOptimizedPhotoPreparedJob(const Prepared& prepared,
                                    const Controller* controller,
                                    size_t job,
                                    const std::string& effective_options,
                                    Serialize&& serialize) {
  if (!prepared || !prepared->patch_match)
    throw std::invalid_argument("PHOTO prepared engine missing");
  if (prepared->prepared_by != controller || prepared->prepared_job != job)
    throw std::invalid_argument("PHOTO prepared job provenance mismatch");
  if (effective_options.empty() ||
      serialize(prepared->patch_match->GetOptions()) != effective_options)
    throw std::invalid_argument("PHOTO prepared effective options mismatch");
}
}  // namespace colmap::mvs
