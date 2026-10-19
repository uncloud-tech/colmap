// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_source_cache.h"

#include <stdexcept>
#include <string>

namespace colmap::mvs {
// The same capability predicate used by the compiled CUDA dispatch. This is
// source support only, not hardware/performance/quality qualification.
inline bool OptimizedStereoCacheSupported(bool cuda, int tile, int samples) {
  if (!cuda || samples < 1) return false;
  switch (tile) {
    case 16:
      return owned::OwnedSourceCache<16>::Enabled(true, samples);
    case 32:
      return owned::OwnedSourceCache<32>::Enabled(true, samples);
    default:
      return false;
  }
}

// A supported explicit configuration, not a performance/quality qualification.
// Both lookahead modes use the owned preallocated-rotation path with cache ON.
inline void ValidateOptimizedStereo(bool cuda,
                                    int tile,
                                    int samples,
                                    bool cache,
                                    bool /*lookahead*/,
                                    int radius,
                                    int step,
                                    bool filter,
                                    bool graph,
                                    bool missing,
                                    bool geom_consistency) {
  if (!OptimizedStereoCacheSupported(cuda, tile, samples) || !cache ||
      radius != 5 || step != 1 || (geom_consistency && !filter) || graph ||
      missing) {
    throw std::invalid_argument(
        "optimized_only requires CUDA owned tile16|32, samples 1..15 (not "
        "source "
        "count), distinct-source cache=true, lookahead=false|true, "
        "window5/step1, GEO "
        "filter=true "
        "(PHOTO may be unfiltered), graph=false, allow_missing_files=false; "
        "no ordinary fallback");
  }
}

// Full-bank resident forecasts bind rolling_active_jobs EXACTLY to the CLI
// value before any payload allocation. Cohort CLI limits are only ceilings;
// per-cohort k may be lower, so N2 cohorts need a separate effective-k
// contract.
inline void ValidateOptimizedStereoConcurrency(bool geom_consistency,
                                               const std::string& concurrency,
                                               bool full_bank_forecast,
                                               bool cohort_forecast) {
  if (concurrency == "1") return;
  if (geom_consistency && concurrency == "2" && full_bank_forecast &&
      !cohort_forecast) {
    return;
  }
  throw std::invalid_argument(
      "optimized_only requires PHOTO concurrency1 or GEO concurrency1|2; "
      "GEO concurrency2 requires one full-bank owned_run_forecast with "
      "explicit active_jobs=2 and matching budgets; cohort N2 and N>2 "
      "are unsupported, no fallback");
}

inline void ValidateOptimizedFusion(const std::string& format,
                                    bool mmap,
                                    bool cache,
                                    int threads,
                                    int max_image_size) {
  if (format != "COLMAP" || !mmap || cache || threads != 1 ||
      max_image_size != -1) {
    throw std::invalid_argument(
        "optimized fusion requires COLMAP "
        "mmap=true/cache=false/threads1/native resolution");
  }
}

inline std::string OptimizedDispatchReceipt(const std::string& phase,
                                            int selected_tile = 0,
                                            int samples = 0,
                                            bool cuda = false,
                                            bool lookahead = true,
                                            int max_concurrent_solvers = 1) {
  if (max_concurrent_solvers != 1 &&
      !(phase == "geo" && max_concurrent_solvers == 2)) {
    throw std::invalid_argument("unsupported effective optimized concurrency");
  }
  if (phase == "fusion") {
    return "{\"schema_version\":1,\"phase\":\"fusion\",\"backend\":\"cpu\","
           "\"tile\":null,\"cache\":false,\"lookahead\":null,"
           "\"lifetime\":\"mapped_inputs\",\"fusion_mode\":\"mmap\"}\n";
  }
  if (phase != "photo" && phase != "geo") {
    throw std::invalid_argument("unknown optimized phase");
  }
  if (!OptimizedStereoCacheSupported(cuda, selected_tile, samples)) {
    throw std::invalid_argument("unsupported effective optimized dispatch");
  }
  return "{\"schema_version\":1,\"phase\":\"" + phase +
         "\",\"backend\":\"owned_cuda\",\"tile\":" +
         std::to_string(selected_tile) +
         ",\"cache\":true,"
         "\"lookahead\":" +
         std::string(lookahead ? "true" : "false") +
         ",\"lifetime\":\"preallocated_rotation\","
         "\"fusion_mode\":null" +
         std::string(max_concurrent_solvers == 2
                         ? ",\"max_concurrent_solvers\":2"
                         : "") +
         "}\n";
}
}  // namespace colmap::mvs
