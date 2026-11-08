// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_stereo_admission.h"

#include <set>
#include <tuple>

namespace colmap::mvs::owned {
namespace {
uint64_t Pitch(uint64_t row, uint64_t alignment) {
  return CheckedMultiply(CheckedAdd(row, alignment - 1) / alignment, alignment);
}
ByteCharge Add(ByteCharge a, ByteCharge b) {
  return {CheckedAdd(a.host, b.host), CheckedAdd(a.device, b.device)};
}
ByteCharge Max(ByteCharge a, ByteCharge b) {
  return {std::max(a.host, b.host), std::max(a.device, b.device)};
}
StereoExecutionRecord Inventory(const StereoAdmissionRequest& request,
                                const StereoAdmissionReference& reference,
                                bool geometry,
                                bool strip = false) {
  StereoExecutionRecord record;
  record.geometry = geometry;
  record.reference = reference.reference;
  record.sources = reference.sources[geometry];
  record.policy = reference.policy[geometry];
  const auto shape = request.images.at(reference.reference);
  record.width = shape.width;
  record.height = shape.height;
  const auto area = CheckedMultiply(shape.width, shape.height);
  uint64_t width = 0, height = 0;
  for (int source : record.sources) {
    const auto& source_shape = request.images.at(source);
    width = std::max(width, source_shape.width);
    height = std::max(height, source_shape.height);
  }
  const auto sources = record.sources.size();
  record.padded_source_pixels =
      CheckedMultiply(CheckedMultiply(width, height), sources);
  const auto layout = BuildOwnedExecutionRotationLayout(
      shape.width, shape.height, sources, record.policy.retain_mask, strip);
  uint64_t initialization = 0, steady = 0;
  for (const auto& slot : layout.slot_layouts) {
    const auto row = CheckedMultiply(slot.width, slot.element_bytes);
    // SelectionCurrent is packed OwnedSelectionScratch, not cudaMallocPitch.
    const auto alignment =
        slot.role == RotationRole::kSelectionCurrent ? 1
        : slot.role == RotationRole::kWorkspace && record.policy.async_pool
            ? 128
            : request.estimated_pitch_alignment;
    const auto pitch = Pitch(row, alignment);
    const auto bytes =
        CheckedMultiply(CheckedMultiply(pitch, slot.height), slot.depth);
    record.rotation_planes.push_back(
        {slot.role, slot.parity, row, slot.height, slot.depth, pitch});
    steady = CheckedAdd(steady, bytes);
    if (!slot.parity) initialization = CheckedAdd(initialization, bytes);
  }
  record.initialization_prng_pitch = Pitch(CheckedMultiply(shape.width, 48),
                                           request.estimated_pitch_alignment);
  initialization = CheckedAdd(
      initialization,
      CheckedMultiply(record.initialization_prng_pitch, shape.height));
  // Both reference texture arrays coexist. Source gray/depth textures stay
  // resident, padded independently of reference shape and including duplicates.
  const auto device = CheckedAdd(
      CheckedAdd(CheckedMultiply(2, area), CheckedMultiply(692, sources)),
      CheckedAdd(
          32, CheckedMultiply(geometry ? 5 : 1, record.padded_source_pixels)));
  // Existing explicit output/reference scratch allowance plus host packing.
  // Conservatively retain gray packing while GEO's float packing is live.
  // InitTransforms packs 43 float parameters per source for one rotation at
  // a time. Its host vector is separate from the four retained device arrays.
  const auto host = CheckedAdd(
      CheckedAdd(CheckedMultiply(24, area), CheckedMultiply(172, sources)),
      CheckedMultiply(geometry ? 5 : 1, record.padded_source_pixels));
  ByteCharge extra{};
  if (strip) {
    record.mode = StereoVolumeMode::kStripFile;
    // Keep the globally aligned stepping width even for smaller images;
    // BuildStripStagingLayout clamps the actual allocation to the extent.
    record.columns = request.strip_columns;
    const auto staging = BuildStripStagingLayout(shape.width,
                                                 shape.height,
                                                 sources,
                                                 record.columns,
                                                 record.policy.retain_mask);
    record.staging_float_pitch = Pitch(CheckedMultiply(staging.columns, 4),
                                       request.estimated_pitch_alignment);
    record.staging_mask_pitch =
        record.policy.retain_mask
            ? Pitch(staging.columns, request.estimated_pitch_alignment)
            : 0;
    record.staging_host_bytes = staging.host_bytes;
    record.staging_device_bytes = CheckedMultiply(
        CheckedMultiply(
            CheckedAdd(CheckedMultiply(3, record.staging_float_pitch),
                       record.staging_mask_pitch),
            staging.height),
        sources);
    record.scratch_disk_bytes = CheckedMultiply(
        CheckedMultiply(area, sources), record.policy.retain_mask ? 9 : 8);
    if (record.scratch_disk_bytes >
        uint64_t(std::numeric_limits<int64_t>::max()))
      throw std::overflow_error("stereo scratch exceeds file offset range");
    // fadvise is not an eviction guarantee: initially charge the entire file.
    record.cache_host_bytes = record.scratch_disk_bytes;
    extra = {CheckedAdd(record.staging_host_bytes, record.cache_host_bytes),
             record.staging_device_bytes};
  }
  record.initialization =
      Add({host, CheckedAdd(initialization, device)}, extra);
  record.steady = Add({host, CheckedAdd(steady, device)}, extra);
  record.peak = Max(record.initialization, record.steady);
  record.reason = strip
                      ? "resident-private exceeds capacity; complete "
                        "strip-file estimate fits"
                      : "complete resident-private HOST-exchange envelope fits";
  return record;
}
}  // namespace

StereoAdmissionPlan PlanSingleDeviceStereoAdmission(
    const StereoAdmissionRequest& request) {
  if (request.active_owners != 1 || request.references.empty() ||
      !request.estimated_pitch_alignment)
    throw std::invalid_argument(
        "single-device stereo requires one owner, work and pitch policy");
  ValidateStripColumns(request.strip_columns, request.block_width);
  StereoAdmissionPlan plan;
  uint64_t pixels = 0, maps = 0;
  for (const auto& image : request.images) {
    detail::CheckDimension(image.width);
    detail::CheckDimension(image.height);
    pixels = CheckedAdd(pixels, CheckedMultiply(image.width, image.height));
  }
  std::set<int> producers;
  for (const auto& reference : request.references) {
    const auto& shape = request.images.at(reference.reference);
    if (!producers.insert(reference.reference).second)
      throw std::invalid_argument("duplicate stereo PHOTO producer");
    maps = CheckedAdd(
        maps, CheckedMultiply(16, CheckedMultiply(shape.width, shape.height)));
  }
  for (const auto& reference : request.references) {
    for (size_t phase = 0; phase < 2; ++phase) {
      if (reference.sources[phase].empty())
        throw std::invalid_argument("empty stereo sources");
      for (int source : reference.sources[phase]) {
        request.images.at(source);
        if (source == reference.reference)
          throw std::invalid_argument("stereo reference cannot source itself");
        if (phase && !producers.count(source))
          throw std::invalid_argument(
              "stereo GEO source has no PHOTO producer");
      }
      // Validate and inventory all jobs before interpreting any capacity issue.
      plan.records[phase].push_back(Inventory(request, reference, phase));
      Inventory(request, reference, phase, true);
    }
  }
  const auto fixed = Add(request.process, request.unknown);
  plan.output_disk_bytes = CheckedMultiply(2, maps);
  plan.common_host_raw =
      CheckedAdd(CheckedMultiply(2, pixels), CheckedMultiply(2, maps));
  plan.loader_host_raw = CheckedAdd(
      CheckedAdd(CheckedMultiply(2, pixels), request.loader_scratch_host),
      fixed.host);
  plan.peak_raw = {plan.loader_host_raw, fixed.device};
  const auto complete = [&](const StereoExecutionRecord& record) {
    auto peak = Add(fixed, record.peak);
    peak.host = std::max(CheckedAdd(peak.host, plan.common_host_raw),
                         plan.loader_host_raw);
    return peak;
  };
  const auto fits = [&](ByteCharge peak) {
    return CheckedAdd(MarginBytes(peak.host), request.reserve.host) <=
               request.capacity.host &&
           CheckedAdd(MarginBytes(peak.device), request.reserve.device) <=
               request.capacity.device;
  };
  for (size_t phase = 0; phase < 2; ++phase) {
    for (size_t job = 0; job < plan.records[phase].size(); ++job) {
      auto& record = plan.records[phase][job];
      if (!fits(complete(record))) {
        record = Inventory(request, request.references[job], phase, true);
        const auto peak = complete(record);
        if (!fits(peak))
          throw StereoAdmissionRefusal(
              CheckedAdd(MarginBytes(peak.host), request.reserve.host) >
                      request.capacity.host
                  ? StereoAdmissionResource::kHost
                  : StereoAdmissionResource::kDevice,
              phase,
              record.reference,
              "single-device stereo minimum HOST-exchange envelope does not "
              "fit");
        if (record.policy.cuda_graphs)
          throw StereoAdmissionRefusal(
              StereoAdmissionResource::kGraphConflict,
              phase,
              record.reference,
              "CUDA graphs conflict with selected strip-file reference");
      }
      plan.scratch_disk_bytes =
          std::max(plan.scratch_disk_bytes, record.scratch_disk_bytes);
      plan.peak_raw = Max(plan.peak_raw, complete(record));
    }
  }
  plan.admitted = {
      CheckedAdd(MarginBytes(plan.peak_raw.host), request.reserve.host),
      CheckedAdd(MarginBytes(plan.peak_raw.device), request.reserve.device)};
  if (plan.admitted.host > request.capacity.host ||
      plan.admitted.device > request.capacity.device)
    throw StereoAdmissionRefusal(
        plan.admitted.host > request.capacity.host
            ? StereoAdmissionResource::kHost
            : StereoAdmissionResource::kDevice,
        false,
        0,
        "single-device stereo complete envelope does not fit");
  plan.required_disk_bytes =
      CheckedAdd(plan.output_disk_bytes, plan.scratch_disk_bytes);
  if (plan.required_disk_bytes > request.disk_capacity)
    throw StereoAdmissionRefusal(StereoAdmissionResource::kDisk,
                                 false,
                                 0,
                                 "stereo output disk capacity unavailable");
  return plan;
}
bool SameStereoExecutionRecord(const StereoExecutionRecord& a,
                               const StereoExecutionRecord& b) {
  const auto fields = [](const StereoExecutionRecord& r) {
    return std::tie(r.geometry,
                    r.reference,
                    r.width,
                    r.height,
                    r.sources,
                    r.policy.retain_mask,
                    r.policy.cuda_graphs,
                    r.policy.async_pool,
                    r.mode,
                    r.columns,
                    r.initialization.host,
                    r.initialization.device,
                    r.steady.host,
                    r.steady.device,
                    r.peak.host,
                    r.peak.device,
                    r.initialization_prng_pitch,
                    r.staging_float_pitch,
                    r.staging_mask_pitch,
                    r.staging_host_bytes,
                    r.staging_device_bytes,
                    r.scratch_disk_bytes,
                    r.cache_host_bytes,
                    r.padded_source_pixels,
                    r.reason);
  };
  if (fields(a) != fields(b) ||
      a.rotation_planes.size() != b.rotation_planes.size())
    return false;
  for (size_t i = 0; i < a.rotation_planes.size(); ++i) {
    const auto& x = a.rotation_planes[i];
    const auto& y = b.rotation_planes[i];
    if (std::tie(
            x.role, x.parity, x.row_bytes, x.rows, x.depth, x.pitch_ceiling) !=
        std::tie(
            y.role, y.parity, y.row_bytes, y.rows, y.depth, y.pitch_ceiling))
      return false;
  }
  return true;
}
bool SameStereoAdmissionPlan(const StereoAdmissionPlan& a,
                             const StereoAdmissionPlan& b) {
  const auto fields = [](const StereoAdmissionPlan& p) {
    return std::tie(p.active_owners,
                    p.peak_raw.host,
                    p.peak_raw.device,
                    p.admitted.host,
                    p.admitted.device,
                    p.common_host_raw,
                    p.loader_host_raw,
                    p.output_disk_bytes,
                    p.scratch_disk_bytes,
                    p.required_disk_bytes);
  };
  if (fields(a) != fields(b)) return false;
  for (size_t phase = 0; phase < 2; ++phase) {
    if (a.records[phase].size() != b.records[phase].size()) return false;
    for (size_t job = 0; job < a.records[phase].size(); ++job)
      if (!SameStereoExecutionRecord(a.records[phase][job],
                                     b.records[phase][job]))
        return false;
  }
  return true;
}
void ValidateStereoExecutionBinding(const StereoExecutionRecord& record,
                                    int reference,
                                    uint64_t width,
                                    uint64_t height,
                                    const std::vector<int>& sources,
                                    bool geometry,
                                    StereoExecutionPolicy policy) {
  if (record.reference != reference || record.width != width ||
      record.height != height || record.sources != sources ||
      record.geometry != geometry ||
      record.policy.retain_mask != policy.retain_mask ||
      record.policy.cuda_graphs != policy.cuda_graphs ||
      record.policy.async_pool != policy.async_pool ||
      (record.mode != StereoVolumeMode::kResidentPrivate &&
       record.mode != StereoVolumeMode::kStripFile))
    throw std::invalid_argument("stereo factory execution binding changed");
  if (record.mode == StereoVolumeMode::kStripFile && policy.cuda_graphs)
    throw StereoAdmissionRefusal(
        StereoAdmissionResource::kGraphConflict,
        geometry,
        reference,
        "CUDA graphs conflict with bound strip-file reference");
}
void ValidateStereoReturnedPitches(
    const StereoExecutionRecord& record,
    const std::vector<RotationAllocationRecord>& rotation,
    uint64_t initialization_prng_pitch,
    const std::array<uint64_t, 4>& staging_pitches) {
  if (rotation.size() != record.rotation_planes.size() ||
      initialization_prng_pitch < CheckedMultiply(record.width, 48) ||
      initialization_prng_pitch > record.initialization_prng_pitch)
    throw std::invalid_argument(
        "stereo rotation count or initialization pitch changed");
  uint64_t spans = 0, admitted_spans = 0;
  for (size_t i = 0; i < rotation.size(); ++i) {
    const auto& actual = rotation[i];
    const auto& expected = record.rotation_planes[i];
    const auto span = CheckedMultiply(
        CheckedMultiply(actual.pitch_bytes, actual.height), actual.depth);
    if (actual.role != expected.role || actual.parity != expected.parity ||
        actual.element_bytes != detail::ElementBytes(expected.role) ||
        actual.requested_span_bytes !=
            CheckedMultiply(CheckedMultiply(expected.row_bytes, expected.rows),
                            expected.depth) ||
        CheckedMultiply(actual.width, actual.element_bytes) !=
            expected.row_bytes ||
        actual.height != expected.rows || actual.depth != expected.depth ||
        actual.pitch_bytes < expected.row_bytes ||
        actual.pitch_bytes > expected.pitch_ceiling ||
        span != actual.pitched_span_bytes)
      throw std::invalid_argument(
          "stereo returned rotation pitch exceeds admitted plane");
    spans = CheckedAdd(spans, span);
    admitted_spans = CheckedAdd(
        admitted_spans,
        CheckedMultiply(CheckedMultiply(expected.pitch_ceiling, expected.rows),
                        expected.depth));
  }
  if (spans > admitted_spans)
    throw std::invalid_argument(
        "stereo returned rotation spans exceed admission");
  if (record.mode == StereoVolumeMode::kResidentPrivate) {
    if (staging_pitches != std::array<uint64_t, 4>{})
      throw std::invalid_argument(
          "resident stereo unexpectedly allocated strip staging");
    return;
  }
  const auto staging = BuildStripStagingLayout(record.width,
                                               record.height,
                                               record.sources.size(),
                                               record.columns,
                                               record.policy.retain_mask);
  spans = 0;
  for (size_t i = 0; i < staging_pitches.size(); ++i) {
    const auto ceiling =
        i == 3 ? record.staging_mask_pitch : record.staging_float_pitch;
    const auto row = CheckedMultiply(staging.columns, i == 3 ? 1 : 4);
    if ((i == 3 && !record.policy.retain_mask)
            ? staging_pitches[i] != 0
            : staging_pitches[i] < row || staging_pitches[i] > ceiling)
      throw std::invalid_argument(
          "stereo returned staging pitch exceeds admitted plane");
    spans = CheckedAdd(
        spans,
        CheckedMultiply(CheckedMultiply(staging_pitches[i], staging.height),
                        record.sources.size()));
  }
  if (spans > record.staging_device_bytes)
    throw std::invalid_argument(
        "stereo returned staging spans exceed admission");
}
}  // namespace colmap::mvs::owned
