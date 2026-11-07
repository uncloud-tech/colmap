// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "colmap/controllers/extraction_resource.h"

#include <string>
#include <utility>

#include "CudaImageDomain.h"
#include "StreamedPlane.h"

namespace siftgpu {
inline uint64_t SiftRequired(uint64_t payload,
                             uint64_t policy,
                             uint64_t reserve) {
  const auto raw = colmap::ExtractionAdd(payload, policy);
  const auto margin = colmap::ExtractionAdd(
      colmap::ExtractionMultiply(raw / 100, 3), (raw % 100 * 3 + 99) / 100);
  return colmap::ExtractionAdd(colmap::ExtractionAdd(raw, margin), reserve);
}

// Opt-in, extractor-owned ledger for real CuTexImage capacities. CUDA textures
// free before growing. This is a payload ledger, not a driver/physical-fit cap.
class SiftDeviceBudget {
 public:
  SiftDeviceBudget(uint64_t cap, uint64_t policy, uint64_t reserve)
      : cap_(cap), policy_(policy), reserve_(reserve) {}
  void Replace(uint64_t old_bytes, uint64_t new_bytes) {
    if (old_bytes > used_)
      throw std::logic_error("SIFT device ledger underflow");
    const auto next = colmap::ExtractionAdd(used_ - old_bytes, new_bytes);
    if (SiftRequired(next, policy_, reserve_) > cap_)
      throw std::runtime_error(
          "SIFT actual device growth refused before allocation: payload=" +
          std::to_string(next) +
          " required=" + std::to_string(SiftRequired(next, policy_, reserve_)) +
          " capacity=" + std::to_string(cap_) +
          "; free device memory/increase cap; no mode switch or quality "
          "fallback");
    used_ = next;
    peak_ = std::max(peak_, used_);
  }
  void Release(uint64_t bytes) noexcept { used_ -= bytes; }
  uint64_t Used() const { return used_; }
  uint64_t Peak() const { return peak_; }

 private:
  uint64_t cap_, policy_, reserve_, used_ = 0, peak_ = 0;
};
inline thread_local std::shared_ptr<SiftDeviceBudget> active_sift_device_budget;
struct SiftDeviceBudgetScope {
  std::shared_ptr<SiftDeviceBudget> previous;
  explicit SiftDeviceBudgetScope(std::shared_ptr<SiftDeviceBudget> budget)
      : previous(std::move(active_sift_device_budget)) {
    active_sift_device_budget = std::move(budget);
  }
  ~SiftDeviceBudgetScope() { active_sift_device_budget = std::move(previous); }
};

inline uint64_t SiftPayloadCapacity(uint64_t cap,
                                    uint64_t policy,
                                    uint64_t reserve) {
  if (cap <= reserve) return 0;
  const auto usable = cap - reserve;
  const auto raw = (usable / 103) * 100 + (usable % 103) * 100 / 103;
  return raw > policy ? raw - policy : 0;
}

// Host growth charges travel with the existing image credit through the writer.
// Current persistent capacities are prepaid again on each input; every possible
// old/new vector relocation is charged before resize, without changing growth.
inline thread_local std::function<void(uint64_t)> active_sift_host_reserve;
struct SiftHostBudgetScope {
  std::function<void(uint64_t)> previous;
  explicit SiftHostBudgetScope(std::function<void(uint64_t)> reserve)
      : previous(std::move(active_sift_host_reserve)) {
    active_sift_host_reserve = std::move(reserve);
  }
  ~SiftHostBudgetScope() { active_sift_host_reserve = std::move(previous); }
};
inline void SiftReserveHost(uint64_t bytes) {
  if (active_sift_host_reserve) active_sift_host_reserve(bytes);
}
template <class T>
void SiftResizeHostVector(std::vector<T>& values, size_t count) {
  if (active_sift_host_reserve && count > values.capacity()) {
    const auto capacity = std::max<uint64_t>(
        count, colmap::ExtractionMultiply(values.capacity(), 2));
    SiftReserveHost(colmap::ExtractionMultiply(capacity, sizeof(T)));
  }
  values.resize(count);
}
inline uint64_t SiftStreamingMetadata(int octaves, int levels) {
  return sizeof(StreamInputHistory) + sizeof(StreamedPlane) +
         uint64_t(octaves) * levels *
             (4 * (sizeof(StreamKeyResult) + 2 * sizeof(int)) +
              4 * sizeof(StreamListResult) +
              4 * sizeof(StreamDescriptorResult) + 128 * sizeof(StreamedPlane) +
              256);
}

// Shared with ResizeFeatureStorage: per-slot maxima over every retained-octave
// fit, preserving the original histogram slot shift and unreduced row height.
inline std::vector<std::pair<int, int>> SiftHistogramCapacityShapes(
    int width, int height, int octaves, int levels) {
  std::vector<std::pair<int, int>> shapes(levels, {0, 0});
  for (int first = 0; first < octaves; ++first) {
    int w = (width + 3) / 4;
    for (int slot = levels - first / 2 - 1; slot >= 0; --slot) {
      auto& shape = shapes[slot];
      if (uint64_t(w) * height > uint64_t(shape.first) * shape.second)
        shape = {w, height};
      if (w == 1) break;
      w = (w + 3) / 4;
    }
    width = std::max(1, width / 2);
    height = std::max(1, height / 2);
  }
  return shapes;
}

struct SiftRequirements {
  bool stream_supported = true;
  uint64_t resident_device = 0, resident_host = 0;
  uint64_t stream_device = 0, stream_host = 0, stream_staging_host = 0;
  uint64_t scratch_bytes = 0, input_history_bytes = 0;
  uint64_t retained_width = 0, retained_height = 0;
};

// Installed photos CLI's scientific configuration: first octave -1, three DoG
// levels, two orientations, ordinary CUDA SIFT. Values follow ResizePyramid /
// ResizeFeatureStorage, not a learned bytes-per-pixel fit. No allocations or
// future-max prewarming: crossed aspect ratios affect only this simulation.
inline SiftRequirements SiftSequenceRequirements(
    const std::vector<std::pair<int, int>>& sequence) {
  using colmap::ExtractionAdd;
  using colmap::ExtractionMultiply;
  auto add = [](uint64_t& value, uint64_t bytes) {
    value = ExtractionAdd(value, bytes);
  };
  SiftRequirements result;
  uint64_t bitmap = 0;
  for (auto [width, height] : sequence) {
    if (width < 1 || height < 1 || width >= (1 << 22) || height >= (1 << 22) ||
        uint64_t(width) * height * 4 > INT32_MAX)
      throw std::invalid_argument(
          "SIFT auto unsupported dimensions/integer domain");
    result.retained_width =
        std::max(result.retained_width, uint64_t(width) * 2);
    result.retained_height =
        std::max(result.retained_height, uint64_t(height) * 2);
    result.input_history_bytes =
        std::max(result.input_history_bytes, uint64_t(width) * height * 4);
    bitmap = std::max(bitmap,
                      colmap::ExtractionBitmapPayload(
                          width, height, 4, std::max(width, height), false));
    if (width < 3 || height < 3) {
      result.stream_supported = false;
      continue;
    }
    const int w = width * 2, h = height * 2;
    // Known row/halo minima, not a promise that a data-dependent sampling
    // window fits. The original orientation/descriptor planners remain gates.
    const uint64_t gpu = std::max({uint64_t(2 * std::min(h, 33)) * w * 4 + 132,
                                   uint64_t(7 * std::min(h, 3)) * w * 4 + 132,
                                   uint64_t(4) * (w + (w + 3) / 4) * 4 + 132,
                                   uint64_t(w) * 8 + 788});
    const uint64_t host = std::max({uint64_t(std::min(h, 33)) * w * 4 + 132,
                                    uint64_t(4 * std::min(h, 3)) * w * 4,
                                    uint64_t(w) * 8 + 656});
    // Reuse the native stage contracts as load-bearing domain/minimum checks.
    (void)StripePlan(w, h, 33, gpu, host);
    (void)DogPlan(w, h, gpu, host);
    (void)KeyPlan(w, h, gpu, host);
    (void)ListPlan(gpu, host);
    (void)OrientationPlan(w, 1, gpu, host);
    (void)DescriptorPlan(w, 1, gpu, host);
    (void)SamplePlan(width, height, w, h, true, width % 4 != 0, gpu, host);
    result.stream_device = std::max(result.stream_device, gpu);
    result.stream_staging_host = std::max(result.stream_staging_host, host);
    // Retained Gaussian(6), DoG(5), gradients(5 float2), KEY(3 float4),
    // histogram trees plus transient Gaussian horizontal/seed planes.
    uint64_t disk = 0;
    int ow = w, oh = h;
    const int octaves =
        std::max(1, int(std::floor(std::log2(std::min(w, h)))) - 3);
    for (int octave = 0; octave < octaves; ++octave) {
      const int domain_width = LevelWidth(ow, width % 4 != 0);
      add(disk, uint64_t(domain_width) * oh * (6 + 5 + 10 + 12 + 2) * 4);
      for (int hw = (ow + 3) / 4;; hw = (hw + 3) / 4) {
        add(disk, uint64_t(hw) * oh * 16 * 3);
        if (hw == 1) break;
      }
      ow = std::max(1, ow / 2);
      oh = std::max(1, oh / 2);
    }
    result.scratch_bytes = std::max(result.scratch_bytes, disk);
  }
  if (sequence.empty()) throw std::invalid_argument("SIFT auto empty sequence");
  add(result.scratch_bytes, result.input_history_bytes);
  // The three capacity-one queues, three serialized workers and reader can
  // own seven decoded images. Existing host credits still gate actual overlap.
  const uint64_t decoded = ExtractionMultiply(bitmap, 7);
  result.resident_host = ExtractionAdd(decoded, result.input_history_bytes);
  result.stream_host = ExtractionAdd(
      ExtractionAdd(decoded, result.stream_staging_host),
      SiftStreamingMetadata(
          std::max(1,
                   int(std::floor(std::log2(std::min(
                       result.retained_width, result.retained_height)))) -
                       3),
          6));
  uint64_t w = result.retained_width, h = result.retained_height;
  const int octaves =
      std::max(1, int(std::floor(std::log2(std::min(w, h)))) - 3);
  const int histogram_levels = std::max(
      1, int(std::ceil(std::log(double(std::max(w, h))) / std::log(4.0))));
  uint64_t histogram = 0;
  for (auto [hw, hh] :
       SiftHistogramCapacityShapes(int(w), int(h), octaves, histogram_levels))
    add(histogram, uint64_t(hw) * hh * 16);
  for (int first = 0; first < octaves; ++first) {
    const uint64_t aligned = (w + 3) / 4 * 4;
    add(result.resident_device, aligned * h * (6 + 5 + 6) * 4);
    if (first == 0) add(result.resident_device, aligned * h * 16);
    // Native initial feature and orientation capacities; NOT count bounds.
    const uint64_t fmax = std::max<uint64_t>(
        32, std::min<uint64_t>(4096, uint64_t(float(aligned * h) * 0.005f)));
    add(result.resident_device, fmax * 16 * 3 * 2);
    if (first == 0) add(result.resident_device, fmax * 128 * 4);
    w = std::max<uint64_t>(1, w / 2);
    h = std::max<uint64_t>(1, h / 2);
  }
  add(result.resident_device, histogram);
  add(result.resident_device, result.input_history_bytes);
  add(result.resident_device, 132);  // live d_kernel
  return result;
}
struct SiftSelection {
  std::string mode, reason;
};
inline SiftSelection SiftSelect(const SiftRequirements& r,
                                uint64_t host,
                                uint64_t device,
                                uint64_t hp,
                                uint64_t dp,
                                uint64_t hr,
                                uint64_t dr,
                                uint64_t disk) {
  if (SiftRequired(r.resident_host, hp, hr) <= host &&
      SiftRequired(r.resident_device, dp, dr) <= device)
    return {"resident", "whole-sequence initial resident owners admitted"};
  if (r.stream_supported && SiftRequired(r.stream_host, hp, hr) <= host &&
      SiftRequired(r.stream_device, dp, dr) <= device &&
      r.scratch_bytes <= disk)
    return {"streamed",
            "resident estimate exceeds capacity; streamed minima admitted"};
  return {"refused",
          "neither mode admitted; free host/device/disk capacity or increase "
          "caps; no resize/fallback"};
}
}  // namespace siftgpu
