// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/patch_match.h"

#include <functional>

namespace colmap::mvs {
// Diagnostic embedding seam: supplies resolved CPU inputs to the SAME native
// input check, after trusted delivery/admission but before controller/device
// creation. Never enables execution or a success receipt: the runner always
// refuses after the probe returns. No CLI flag or production adapter uses it.
using OptimizedPhotoInputCheck =
    std::function<void(size_t generation, const PatchMatch::Problem&)>;
using OptimizedPhotoInputProbe =
    std::function<void(const OptimizedPhotoInputCheck&)>;
}  // namespace colmap::mvs
