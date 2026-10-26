// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/mvs/owned_source_cache.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

// Compiled rolling structural ceiling shared by the scheduler and coordinator
// of BOTH phases. It bounds in-process owners only: it is not qualification,
// not a forecast and not permission for any workload.
inline constexpr int kOptimizedRollingStructuralMax = 5;

// PHOTO concurrency actually qualified today. No PHOTO N>1 experiment has been
// run, so capability metadata keeps claiming 1 and a launch-contract ceiling
// never raises this value.
inline constexpr int kOptimizedPhotoQualifiedConcurrency = 1;

// Compiled-supported GEO ceiling only, never resource admission or
// qualification. Without an exclusive full-bank forecast, GEO remains
// singleton-only. PHOTO keeps its historical value of 1 here; PHOTO N>1 is
// admitted through OptimizedPhotoMaxConcurrency with a SUPPLIED ceiling.
inline int OptimizedStereoMaxConcurrency(bool geom_consistency,
                                         bool full_bank_forecast,
                                         bool cohort_forecast) {
  return geom_consistency && full_bank_forecast && !cohort_forecast ? 5 : 1;
}

// PHOTO ceiling from the supplied launch-contract k (photo forecast vs caps).
// `supplied_ceiling` is NEVER computed or estimated here: 0/1 means the launch
// contract supplied no PHOTO N>1 ceiling, so PHOTO stays singleton-only. A
// supplied value above the compiled rolling ceiling does not raise it.
inline int OptimizedPhotoMaxConcurrency(bool full_bank_forecast,
                                        bool cohort_forecast,
                                        int supplied_ceiling) {
  if (!full_bank_forecast || cohort_forecast || supplied_ceiling < 2) return 1;
  return std::min(supplied_ceiling, kOptimizedRollingStructuralMax);
}

// Exact canonical decimal CLI concurrency. No sign, whitespace, empty string,
// leading zero, suffix or out-of-range value is a concurrency.
inline int ParseOptimizedConcurrency(const std::string& concurrency) {
  if (concurrency.empty() || concurrency.front() == '0' ||
      concurrency.size() > 9) {
    throw std::invalid_argument(
        "owned concurrency requires a canonical positive decimal integer");
  }
  uint64_t value = 0;
  for (const unsigned char c : concurrency) {
    if (c < '0' || c > '9') {
      throw std::invalid_argument(
          "owned concurrency requires ASCII digits only, no sign/whitespace");
    }
    value = value * 10 + (c - '0');
  }
  if (!value ||
      value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("owned concurrency is outside int range");
  }
  return static_cast<int>(value);
}

// One admission decision for both phases. PHOTO N1..Nk runs in-process on ONE
// shared controller instance (rolling scheduler reuse); `k` is the launch
// contract's supplied ceiling, never derived here. Full-bank resident
// forecasts bind rolling_active_jobs EXACTLY to the CLI value before any
// payload allocation. Cohort CLI limits are only ceilings, so concurrent
// cohorts still need a separate effective-k contract. N is never clamped: an
// N above the supplied ceiling, above the compiled rolling ceiling, or above
// the number of distinct scheduled jobs refuses.
struct OptimizedConcurrencyLimits {
  bool geom_consistency = false;
  // Exact CLI value of the rolling concurrency (rolling_active_jobs).
  std::string concurrency;
  bool full_bank_forecast = false;
  bool cohort_forecast = false;
  // Supplied PHOTO ceiling k from the launch contract (photo forecast vs caps).
  // 0 = not supplied. Ignored for GEO.
  int photo_ceiling = 0;
  // Distinct scheduled jobs (references/sides). 0 = not bound at this site.
  size_t job_count = 0;
};

inline int AdmitOptimizedConcurrency(const OptimizedConcurrencyLimits& limits) {
  const int requested = ParseOptimizedConcurrency(limits.concurrency);
  const int maximum =
      limits.geom_consistency
          ? OptimizedStereoMaxConcurrency(
                true, limits.full_bank_forecast, limits.cohort_forecast)
          : OptimizedPhotoMaxConcurrency(limits.full_bank_forecast,
                                         limits.cohort_forecast,
                                         limits.photo_ceiling);
  if (requested > maximum) {
    if (limits.geom_consistency) {
      const int geo_maximum = OptimizedStereoMaxConcurrency(true, true, false);
      std::string geo_values = "1";
      for (int n = 2; n <= geo_maximum; ++n)
        geo_values += "|" + std::to_string(n);
      throw std::invalid_argument(
          "optimized_only requires PHOTO concurrency" +
          std::to_string(OptimizedPhotoMaxConcurrency(limits.full_bank_forecast,
                                                      limits.cohort_forecast,
                                                      limits.photo_ceiling)) +
          " for this launch or GEO concurrency" + geo_values +
          "; GEO concurrency2.." + std::to_string(geo_maximum) +
          " requires one full-bank owned_run_forecast with "
          "explicit active_jobs equal to concurrency and matching budgets; "
          "cohort N>1 and N>" +
          std::to_string(geo_maximum) + " are unsupported, no fallback");
    }
    throw std::invalid_argument(
        "optimized_only requires PHOTO concurrency" + std::to_string(maximum) +
        " for this launch: requested " + std::to_string(requested) +
        " exceeds the supplied launch-contract ceiling k=" +
        std::to_string(limits.photo_ceiling) +
        " (single shared in-process controller, refuse instead of clamping, "
        "no fallback); PHOTO N>1 requires one full-bank photo forecast, no "
        "cohort, a supplied ceiling k >= N and N<=" +
        std::to_string(kOptimizedRollingStructuralMax));
  }
  if (limits.job_count && static_cast<size_t>(requested) > limits.job_count) {
    throw std::invalid_argument(
        "optimized concurrency" + std::to_string(requested) + " exceeds the " +
        std::to_string(limits.job_count) +
        " distinct scheduled job(s); refuse instead of clamping");
  }
  return requested;
}

// Production entry point: same refusals, admitted value discarded.
inline void ValidateOptimizedStereoConcurrency(
    const OptimizedConcurrencyLimits& limits) {
  (void)AdmitOptimizedConcurrency(limits);
}

// Historical 4-argument GEO/PHOTO gate. PHOTO keeps ceiling 1 and no bound
// job count here, so its behavior is byte-for-byte the previous policy.
inline void ValidateOptimizedStereoConcurrency(bool geom_consistency,
                                               const std::string& concurrency,
                                               bool full_bank_forecast,
                                               bool cohort_forecast) {
  OptimizedConcurrencyLimits limits;
  limits.geom_consistency = geom_consistency;
  limits.concurrency = concurrency;
  limits.full_bank_forecast = full_bank_forecast;
  limits.cohort_forecast = cohort_forecast;
  limits.photo_ceiling = geom_consistency ? 0 : 1;
  ValidateOptimizedStereoConcurrency(limits);
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

// One admitted owner's live interval inside a shared-controller run.
// Descriptive telemetry only: it is not a resource bound, a forecast or a
// qualification.
struct OptimizedOwnerInterval {
  uint64_t begin_micros = 0;
  uint64_t end_micros = 0;
};

// In-process concurrency telemetry for an N>1 receipt. `requested_n` must equal
// the CLI rolling_active_jobs value and `admitted_n` the coordinator's admitted
// maximum, so the receipt cannot diverge from what was launched.
struct OptimizedConcurrencyObservation {
  int requested_n = 1;
  int admitted_n = 1;
  int observed_n = 1;
  std::vector<OptimizedOwnerInterval> owner_intervals;
};

inline std::string OwnerIntervals(
    const OptimizedConcurrencyObservation& concurrency) {
  std::string rendered = "[";
  for (size_t i = 0; i < concurrency.owner_intervals.size(); ++i) {
    const auto& interval = concurrency.owner_intervals[i];
    if (i) rendered += ",";
    rendered += "{\"begin_us\":" + std::to_string(interval.begin_micros) +
                ",\"end_us\":" + std::to_string(interval.end_micros) + "}";
  }
  return rendered + "]";
}

inline std::string OptimizedDispatchReceipt(
    const std::string& phase,
    int selected_tile = 0,
    int samples = 0,
    bool cuda = false,
    bool lookahead = true,
    int max_concurrent_solvers = 1,
    const OptimizedConcurrencyObservation* concurrency = nullptr) {
  if (max_concurrent_solvers < 1) {
    throw std::invalid_argument("unsupported effective optimized concurrency");
  }
  if (max_concurrent_solvers == 1) {
    // N1 keeps its historical bytes: telemetry that claims otherwise is a
    // divergence, not an acceptable extra field.
    if (concurrency != nullptr &&
        (concurrency->requested_n != 1 || concurrency->admitted_n != 1 ||
         concurrency->observed_n != 1 ||
         !concurrency->owner_intervals.empty())) {
      throw std::invalid_argument(
          "N1 optimized receipt cannot carry N>1 concurrency telemetry");
    }
  } else if (phase == "geo") {
    // GEO N2..N5 bytes are an existing wire; no field may be added here.
    if (concurrency != nullptr ||
        max_concurrent_solvers >
            OptimizedStereoMaxConcurrency(true, true, false)) {
      throw std::invalid_argument(
          "unsupported effective optimized concurrency");
    }
  } else if (phase == "photo") {
    // PHOTO N>1 must name the launched concurrency and the observed owners.
    if (concurrency == nullptr ||
        max_concurrent_solvers > kOptimizedRollingStructuralMax ||
        concurrency->requested_n != max_concurrent_solvers ||
        concurrency->admitted_n != max_concurrent_solvers) {
      throw std::invalid_argument(
          "PHOTO N>1 receipt requires rolling_active_jobs equal to concurrency "
          "and an admitted owner count that matches the CLI value");
    }
    // One interval per admitted OWNER GENERATION (a rolling generation is one
    // reference), so there are at least `admitted_n` of them.
    if (concurrency->observed_n < 1 ||
        concurrency->observed_n > concurrency->admitted_n ||
        concurrency->owner_intervals.size() <
            static_cast<size_t>(concurrency->admitted_n)) {
      throw std::invalid_argument(
          "PHOTO N>1 receipt requires one interval per admitted generation and "
          "1 <= observed N <= admitted N");
    }
    for (size_t i = 0; i < concurrency->owner_intervals.size(); ++i) {
      const auto& interval = concurrency->owner_intervals[i];
      if (interval.begin_micros > interval.end_micros ||
          (i && concurrency->owner_intervals[i - 1].begin_micros >
                    interval.begin_micros)) {
        throw std::invalid_argument(
            "PHOTO N>1 owner intervals require ordered begin <= end");
      }
    }
  } else {
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
         (max_concurrent_solvers > 1
              ? ",\"max_concurrent_solvers\":" +
                    std::to_string(max_concurrent_solvers)
              : "") +
         (max_concurrent_solvers > 1 && phase == "photo"
              ? ",\"rolling_active_jobs\":" +
                    std::to_string(concurrency->requested_n) +
                    ",\"requested_n\":" +
                    std::to_string(concurrency->requested_n) +
                    ",\"admitted_n\":" +
                    std::to_string(concurrency->admitted_n) +
                    ",\"observed_n\":" +
                    std::to_string(concurrency->observed_n) +
                    ",\"owner_intervals\":" + OwnerIntervals(*concurrency)
              : "") +
         "}\n";
}
}  // namespace colmap::mvs
