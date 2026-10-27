// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_input_bank.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include <OpenImageIO/imageio.h>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/stat.h>
#endif

namespace colmap::mvs::owned {
namespace {
uint64_t Add(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a) {
    throw std::overflow_error("Resident bank byte sum overflow");
  }
  return a + b;
}
uint64_t Mul(uint64_t a, uint64_t b) {
  if (a && b > std::numeric_limits<uint64_t>::max() / a) {
    throw std::overflow_error("Resident bank byte product overflow");
  }
  return a * b;
}
uint64_t Margin(uint64_t bytes) {
  // Single 3% forecast margin, rounded up; not a physical allocation bound.
  return Add(bytes, Add(Mul(bytes / 100, 3), (Mul(bytes % 100, 3) + 99) / 100));
}
// VLFeat gaussian filter charge (imopv.c _vl_new_gaussian_fitler_f):
// 4 bytes * (2*ceil(3*sigma)+1) floats, with vl_ceil_d semantics (plain ceil).
uint64_t GaussianFilterBytes(float sigma) {
  const double samples = std::ceil(static_cast<double>(sigma) * 3.0);
  return Mul(Add(Mul(static_cast<uint64_t>(samples), 2), 1), 4);
}
// FileIdentity resolves to the public invocation-snapshot record type.
using FileIdentity = BankFileIdentity;
BankFileIdentity Identify(const std::filesystem::path& path) {
  const auto resolved = std::filesystem::canonical(path);
#if defined(__linux__) || defined(__APPLE__)
  // Capture object identity and change metadata together. Size/mtime alone
  // cannot detect atomic replacement preserving timestamps, or an in-place
  // write followed by restoration of mtime. Do not truncate nanoseconds.
  struct stat info{};
  if (::stat(resolved.c_str(), &info) != 0) {
    throw std::runtime_error("Cannot stat resident input");
  }
  if (!S_ISREG(info.st_mode) || info.st_size < 0) {
    throw std::invalid_argument("Resident input is not a regular file");
  }
#if defined(__APPLE__)
  const auto modified = info.st_mtimespec;
  const auto changed = info.st_ctimespec;
#else
  const auto modified = info.st_mtim;
  const auto changed = info.st_ctim;
#endif
  return {resolved,
          static_cast<uintmax_t>(info.st_size),
          static_cast<uintmax_t>(info.st_dev),
          static_cast<uintmax_t>(info.st_ino),
          modified.tv_sec,
          modified.tv_nsec,
          changed.tv_sec,
          changed.tv_nsec};
#else
  // Resident mode must fail closed rather than silently use a weaker identity
  // on platforms without the checked file-object/change metadata above.
  throw std::runtime_error(
      "Resident input identity unsupported on this platform");
#endif
}
void CheckIdentity(const BankFileIdentity& identity) {
  const auto current = Identify(identity.path);
  if (!SameIdentity(current, identity)) {
    throw std::runtime_error("Resident input changed during bank construction");
  }
}
// Strict invocation-snapshot validation: captured identity must be unchanged
// and structurally equal to the expected layer triple.
void CheckSnapshotIdentity(const BankFileIdentity& expected) {
  const auto current = Identify(expected.path);
  if (!SameIdentity(current, expected)) {
    throw std::runtime_error(
        "Resident input does not match the captured invocation identity");
  }
}
Bitmap ReadGrayscale(const Image& image,
                     int max_image_size,
                     const BankFileIdentity& identity) {
  CheckIdentity(identity);
  Bitmap grayscale;
  THROW_CHECK(grayscale.Read(identity.path, false));
  if (max_image_size > 0) {
    grayscale.Rescale(image.GetWidth(), image.GetHeight());
  }
  THROW_CHECK(grayscale.IsGrey());
  THROW_CHECK_EQ(static_cast<size_t>(grayscale.Width()), image.GetWidth());
  THROW_CHECK_EQ(static_cast<size_t>(grayscale.Height()), image.GetHeight());
  CheckIdentity(identity);
  return grayscale;
}
bool SameMetadata(const BankInput& a, const BankInput& b) {
  return a.max_image_size == b.max_image_size &&
         a.image.GetWidth() == b.image.GetWidth() &&
         a.image.GetHeight() == b.image.GetHeight() &&
         std::equal(a.image.GetK(), a.image.GetK() + 9, b.image.GetK()) &&
         std::equal(a.image.GetR(), a.image.GetR() + 9, b.image.GetR()) &&
         std::equal(a.image.GetT(), a.image.GetT() + 3, b.image.GetT());
}
uint64_t MapBytes(const FileIdentity& file,
                  size_t channels,
                  uint64_t* width = nullptr,
                  uint64_t* height = nullptr) {
  std::ifstream stream(file.path, std::ios::binary);
  uint64_t w = 0, h = 0, d = 0;
  char a = 0, b = 0, c = 0;
  stream >> w >> a >> h >> b >> d >> c;
  if (!stream || a != '&' || b != '&' || c != '&' || !w || !h ||
      d != channels) {
    throw std::invalid_argument("Invalid resident map header");
  }
  const uint64_t bytes = Mul(Mul(Mul(w, h), d), sizeof(float));
  const auto begin = stream.tellg();
  if (begin < 0 || Add(static_cast<uint64_t>(begin), bytes) > file.size) {
    throw std::invalid_argument("Truncated resident map");
  }
  if (width) *width = w;
  if (height) *height = h;
  return bytes;
}
struct Layout {
  std::vector<BankLayer> layers;
  std::vector<std::vector<FileIdentity>> identities;
  // Per job, in job.inputs order: the layer holding each admitted input.
  std::vector<std::vector<int>> input_layers;
  std::vector<int> references;
  std::vector<std::vector<int>> sources;
  size_t width = 0, height = 0;
  BankLimits estimate;
  BankLimits estimate_bounded;
  uint64_t host_raw_bytes = 0;
  uint64_t device_raw_bytes = 0;
  // Removed exactly once from the raw eager total for the bounded estimate:
  // 12 bytes/texel per reference layer (3 float normal channels).
  uint64_t normal_retained_once = 0;
  // Explicit per-reference-layer normal allowances (raw, margin-free).
  uint64_t max_ref_owner_bytes = 0;
  uint64_t max_ref_transient_bytes = 0;
  size_t reference_layers = 0;
};
Layout Resolve(const std::vector<BankJob>& jobs) {
  if (jobs.empty()) throw std::invalid_argument("Empty resident bank");
  Layout result;
  for (const auto& job : jobs) {
    if (!job.options.geom_consistency || job.options.allow_missing_files ||
        job.source_image_idxs.empty()) {
      throw std::invalid_argument(
          "Resident bank requires complete geometry inputs");
    }
    for (const auto& image : job.images) {
      if (!image.GetBitmap().IsEmpty()) {
        throw std::invalid_argument("Resident job must contain metadata only");
      }
    }
    std::set<int> required(job.source_image_idxs.begin(),
                           job.source_image_idxs.end());
    required.insert(job.reference_image_idx);
    if (required.size() != job.source_image_idxs.size() + 1 ||
        job.inputs.size() != required.size()) {
      throw std::invalid_argument(
          "Duplicate or incomplete resident input mapping");
    }
    std::vector<int> mapping(job.images.size(), -1);
    // Input-ordinal -> layer mapping: one entry per job input, in job.inputs
    // order. Distinct from the image-index mapping below (image space is the
    // WHOLE workspace, not the job's admitted inputs); validation of the
    // invocation identity snapshot uses THIS space, never image space.
    std::vector<int> input_layers;
    input_layers.reserve(job.inputs.size());
    for (const auto& input : job.inputs) {
      if (input.image_idx < 0 ||
          static_cast<size_t>(input.image_idx) >= job.images.size() ||
          !required.count(input.image_idx) || mapping[input.image_idx] != -1 ||
          !input.image.GetBitmap().IsEmpty() || !input.image.GetWidth() ||
          !input.image.GetHeight()) {
        throw std::invalid_argument("Invalid resident input metadata");
      }
      BankInput expected = input;
      expected.image = job.images.at(input.image_idx);
      if (!SameMetadata(input, expected) ||
          input.max_image_size != job.options.max_image_size ||
          input.image.GetPath() != expected.image.GetPath()) {
        throw std::invalid_argument("Resident job/input metadata mismatch");
      }
      std::vector<FileIdentity> ids = {Identify(input.image.GetPath()),
                                       Identify(input.depth_path),
                                       Identify(input.normal_path)};
      size_t layer = 0;
      for (; layer < result.layers.size(); ++layer) {
        const auto& prior = result.identities[layer];
        if (ids[0].path == prior[0].path && ids[1].path == prior[1].path &&
            ids[2].path == prior[2].path &&
            SameMetadata(input, result.layers[layer].input)) {
          for (const auto& id : prior) CheckIdentity(id);
          break;
        }
      }
      if (layer == result.layers.size()) {
        if (layer >= static_cast<size_t>(std::numeric_limits<int>::max())) {
          throw std::overflow_error("Too many resident layers");
        }
        BankLayer entry;
        entry.input = input;
        entry.input.image = Image(ids[0].path,
                                  input.image.GetWidth(),
                                  input.image.GetHeight(),
                                  input.image.GetK(),
                                  input.image.GetR(),
                                  input.image.GetT());
        entry.input.depth_path = ids[1].path;
        entry.input.normal_path = ids[2].path;
        result.layers.push_back(std::move(entry));
        result.identities.push_back(std::move(ids));
      }
      mapping[input.image_idx] = static_cast<int>(layer);
      input_layers.push_back(static_cast<int>(layer));
      if (input.image_idx == job.reference_image_idx) {
        result.layers[layer].is_reference = true;
      }
    }
    result.input_layers.push_back(std::move(input_layers));
    result.references.push_back(mapping.at(job.reference_image_idx));
    result.sources.emplace_back();
    for (const int idx : job.source_image_idxs) {
      result.sources.back().push_back(mapping.at(idx));
    }
  }
  uint64_t retained = 0, staging = 0;
  uint64_t metadata = Mul(jobs.size(), sizeof(BankJob));
  for (const auto& job : jobs) {
    metadata = Add(metadata, Mul(job.images.size(), sizeof(Image)));
    metadata = Add(metadata, Mul(job.inputs.size(), sizeof(BankInput) + 512));
    metadata =
        Add(metadata, Mul(job.source_image_idxs.size(), 2 * sizeof(int)));
  }
  metadata = Add(metadata,
                 Mul(result.layers.size(),
                     sizeof(BankLayer) + 3 * sizeof(FileIdentity) + 512));
  for (size_t i = 0; i < result.layers.size(); ++i) {
    const auto& layer = result.layers[i];
    const auto& image = layer.input.image;
    const uint64_t area = Mul(image.GetWidth(), image.GetHeight());
    if (image.GetWidth() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        image.GetHeight() >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
          "Resident image exceeds bitmap dimension range");
    }
    result.width = std::max(result.width, image.GetWidth());
    result.height = std::max(result.height, image.GetHeight());
    retained = Add(retained, Mul(area, layer.is_reference ? 17 : 5));
    const auto& ids = result.identities[i];
    // Header-only OIIO inspection: never decode before the whole-bank check.
    auto reader = OIIO::ImageInput::open(ids[0].path.u8string());
    if (!reader)
      throw std::invalid_argument("Unreadable resident image header");
    const auto spec = reader->spec();
    reader->close();
    if (spec.width <= 0 || spec.height <= 0 || spec.nchannels <= 0) {
      throw std::invalid_argument("Invalid resident image dimensions");
    }
    const uint64_t decode = Mul(Mul(spec.width, spec.height),
                                Add(static_cast<uint64_t>(spec.nchannels), 1));
    uint64_t depth_w = 0, depth_h = 0;
    const uint64_t depth = MapBytes(ids[1], 1, &depth_w, &depth_h);
    uint64_t normal_w = 0, normal_h = 0;
    const uint64_t normal = MapBytes(ids[2], 3, &normal_w, &normal_h);
    const uint64_t src_texels = Mul(normal_w, normal_h);
    // SmoothImage sigmas replicate warp.cc DownsampleImage exactly: float
    // dimension ratios, max against float epsilon. vl_imsmooth_f then compares
    // the two sigmas as doubles and aliases filtery to filterx when equal.
    const float scale_x =
        static_cast<float>(normal_w) / static_cast<float>(image.GetWidth());
    const float scale_y =
        static_cast<float>(normal_h) / static_cast<float>(image.GetHeight());
    const float kSigmaScale = 0.5f;
    const float kEpsilon = std::numeric_limits<float>::epsilon();
    const float sigma_x = std::max(kEpsilon, kSigmaScale * (scale_x - 1.0f));
    const float sigma_y = std::max(kEpsilon, kSigmaScale * (scale_y - 1.0f));
    const uint64_t filters =
        sigma_x == sigma_y
            ? GaussianFilterBytes(sigma_x)
            : Add(GaussianFilterBytes(sigma_x), GaussianFilterBytes(sigma_y));
    // Serial loader allowance: original decode and float maps, resample
    // destinations/copies and upload staging. Decoder internals are estimates.
    staging = std::max(
        staging, Add(Add(Mul(decode, 4), depth), Add(normal, Mul(area, 34))));
    if (layer.is_reference) {
      // Explicit on-demand normal allowances, derived from the same header
      // validation as above. Owner charge: one retained 3-channel map at the
      // reference layer resolution. In-flight load charge: file payload; when
      // Downsize rescales, the checked actual overlap covers the Rescale
      // buffers (source capacity, new_data, shrink replacement) AND the
      // DownsampleImage per-channel full-source smoothing temp kept live
      // alongside new_data (see warp.cc DownsampleImage). Keep NormalMap
      // unchanged; charge only these observed allocations, never decoder
      // internals beyond the estimates already present in the bank staging.
      const uint64_t owner = Mul(area, 12);
      result.max_ref_owner_bytes = std::max(result.max_ref_owner_bytes, owner);
      result.reference_layers += 1;
      const uint64_t downsize_target_w =
          static_cast<uint64_t>(image.GetWidth());
      const uint64_t downsize_target_h =
          static_cast<uint64_t>(image.GetHeight());
      const bool downsize =
          layer.input.max_image_size > 0 &&
          (normal_w > downsize_target_w || normal_h > downsize_target_h);
      const uint64_t transient =
          downsize
              ? std::max(
                    // Rescale keeps the original payload capacity, new_data and
                    // the shrink-to-fit replacement simultaneously
                    // [normal_map.cc 28,43,46]: payload + 24*a.
                    Add(normal, Add(owner, owner)),
                    // Each smoothing pass keeps new_data plus TWO full-source
                    // float temps: warp.cc DownsampleImage's smoothed vector
                    // (4*S) and the VLFeat convolution buffer of vl_imsmooth_f
                    // (imopv.c:660, 4*S); plus input-derived Gaussian filters
                    // (_vl_new_gaussian_fitler_f, imopv.c:620-639),
                    // 4*(2*ceil(3*sigma)+1) bytes per axis, with filtery
                    // aliasing filterx exactly as in vl_imsmooth_f when the
                    // float sigmas match (then charged once). vl_imconvcol_v
                    // (plain and SSE2) and ResampleImageBilinear allocate
                    // nothing.
                    Add(Add(normal, owner), Add(Mul(src_texels, 8), filters)))
              : normal;
      result.max_ref_transient_bytes =
          std::max(result.max_ref_transient_bytes, transient);
      result.normal_retained_once = Add(result.normal_retained_once, owner);
    }
  }
  const uint64_t padded_area = Mul(result.width, result.height);
  const uint64_t padded = Mul(padded_area, result.layers.size());
  // GPU upload keeps all retained payloads alive and stages one padded gray
  // layer plus one padded float-depth layer at a time. Decode and upload are
  // sequential, so their temporary allowances are a maximum, not a sum.
  staging = std::max(staging, Mul(padded_area, 5));
  const uint64_t raw = Add(Add(retained, metadata), staging);
  result.host_raw_bytes = raw;
  result.estimate.host_bytes = Margin(raw);
  // Bounded bank: remove the retained reference normals exactly once from the
  // raw eager total; keep every non-normal construction/staging allowance and
  // apply the same single 3% margin to the reduced raw total.
  if (raw < result.normal_retained_once) {
    throw std::overflow_error("Resident bank bounded raw underflow");
  }
  result.estimate_bounded.host_bytes =
      Margin(raw - result.normal_retained_once);
  // Five bytes/texel (gray + float depth), plus one texture descriptor
  // allowance per layer and fixed handle allowance. CUDA physical padding is
  // not proven.
  result.device_raw_bytes =
      Add(Mul(padded, 5), Add(Mul(result.layers.size(), 256), 4096));
  result.estimate.device_bytes = Margin(result.device_raw_bytes);
  result.estimate_bounded.device_bytes = result.estimate.device_bytes;
  return result;
}
}  // namespace

// Public invocation-snapshot identity capture shared with the resident
// contract metadata export. Fails closed exactly like the internal capture.
BankFileIdentity CaptureFileIdentity(const std::filesystem::path& path) {
  return Identify(path);
}

// Public invocation-snapshot revalidation shared with the resident contract.
// A changed object, size, or change metadata between the capture and this
// check throws; contract binding must describe ONE stable invocation.
void CheckFileIdentityUnchanged(const BankFileIdentity& identity) {
  CheckIdentity(identity);
}

struct HostInputBank::Impl {
  std::vector<BankJob> jobs;
  Layout layout;
  BankWorkCounts work;
  BankNormals normals = BankNormals::kEager;
  // Explicit invocation snapshot for strict identity validation, per job in
  // job.inputs order; empty when the caller does not bind one (legacy
  // self-recheck behavior unchanged).
  std::vector<BankJobIdentities> snapshot;
  // Per-layer expected identity derived from the snapshot (reload reads and
  // bounded on-demand loads validate against the invocation, never a
  // recapture). Empty when unbound.
  std::vector<BankLayerIdentity> layer_expected;
};
HostGrayscaleBank::HostGrayscaleBank(
    const std::vector<std::vector<GrayscaleInput>>& jobs,
    const std::atomic<bool>* cancelled) {
  const auto check_cancelled = [cancelled] {
    if (cancelled && cancelled->load(std::memory_order_relaxed)) {
      throw std::runtime_error("Grayscale bank construction cancelled");
    }
  };
  check_cancelled();
  if (jobs.empty()) throw std::invalid_argument("Empty grayscale bank");
  std::vector<GrayscaleInput> inputs;
  for (const auto& job : jobs) {
    if (job.empty()) throw std::invalid_argument("Empty grayscale source job");
    std::vector<int> sources;
    for (const auto& input : job) {
      check_cancelled();
      const auto& image = input.image;
      if (!image.GetBitmap().IsEmpty() || !image.GetWidth() ||
          !image.GetHeight() ||
          image.GetWidth() >
              static_cast<size_t>(std::numeric_limits<int>::max()) ||
          image.GetHeight() >
              static_cast<size_t>(std::numeric_limits<int>::max()) ||
          input.identity.path.empty()) {
        throw std::invalid_argument("Invalid grayscale input metadata");
      }
      // The supplied admission snapshot is mandatory. Canonical aliases may
      // share, but a new object at the same path is never silently re-admitted.
      CheckSnapshotIdentity(input.identity);
      if (!SameIdentity(Identify(image.GetPath()), input.identity)) {
        throw std::invalid_argument("Grayscale input/snapshot mismatch");
      }
      size_t layer = 0;
      for (; layer < inputs.size(); ++layer) {
        if (!SameIdentity(inputs[layer].identity, input.identity)) continue;
        if (inputs[layer].image.GetWidth() != image.GetWidth() ||
            inputs[layer].image.GetHeight() != image.GetHeight() ||
            inputs[layer].max_image_size != input.max_image_size) {
          throw std::invalid_argument(
              "Inconsistent shared grayscale representation");
        }
        break;
      }
      if (layer == inputs.size()) {
        if (layer >= static_cast<size_t>(std::numeric_limits<int>::max())) {
          throw std::overflow_error("Too many grayscale layers");
        }
        // Check the requested effective dimensions against the actual header
        // and the private path's resizing policy before any cohort decode.
        auto reader = OIIO::ImageInput::open(input.identity.path.u8string());
        if (!reader) throw std::invalid_argument("Unreadable grayscale header");
        const auto spec = reader->spec();
        reader->close();
        if (spec.width <= 0 || spec.height <= 0 || spec.depth != 1 ||
            spec.nchannels < 1 || spec.nchannels > 4) {
          throw std::invalid_argument("Invalid grayscale image dimensions");
        }
        // Bitmap's decoder indexes width*height*channels using int before
        // resizing. Reject overflow in that actual allocation path, even if
        // the requested effective image is small.
        if (Mul(Mul(spec.width, spec.height), spec.nchannels) >
            static_cast<uint64_t>(std::numeric_limits<int>::max())) {
          throw std::overflow_error("Grayscale decode exceeds bitmap range");
        }
        Image effective(input.identity.path,
                        spec.width,
                        spec.height,
                        image.GetK(),
                        image.GetR(),
                        image.GetT());
        if (input.max_image_size > 0) {
          effective.Downsize(input.max_image_size, input.max_image_size);
        }
        if (effective.GetWidth() != image.GetWidth() ||
            effective.GetHeight() != image.GetHeight()) {
          throw std::invalid_argument(
              "Grayscale effective dimensions mismatch");
        }
        CheckSnapshotIdentity(input.identity);
        inputs.push_back(input);
        width_ = std::max(width_, image.GetWidth());
        height_ = std::max(height_, image.GetHeight());
      }
      if (std::find(sources.begin(), sources.end(), static_cast<int>(layer)) !=
          sources.end()) {
        throw std::invalid_argument("Duplicate grayscale source in job");
      }
      sources.push_back(static_cast<int>(layer));
    }
    sources_.push_back(std::move(sources));
  }
  const uint64_t layer_bytes = Mul(width_, height_);
  // Check the entire padded union, not just each individual allocation.
  Mul(layer_bytes, inputs.size());
  if (layer_bytes > std::vector<uint8_t>().max_size()) {
    throw std::overflow_error("Grayscale layer exceeds host vector range");
  }
  for (const auto& input : inputs) {
    check_cancelled();
    const auto grayscale =
        ReadGrayscale(input.image, input.max_image_size, input.identity);
    ++work_.grayscale_reads;
    check_cancelled();
    auto layer = std::make_shared<GrayscaleLayer>();
    layer->identity = input.identity;
    layer->width = input.image.GetWidth();
    layer->height = input.image.GetHeight();
    layer->row_stride = width_;
    layer->pixels.resize(static_cast<size_t>(layer_bytes), 0);
    for (size_t row = 0; row < layer->height; ++row) {
      std::copy_n(grayscale.RowMajorData().data() + row * layer->width,
                  layer->width,
                  layer->pixels.data() + row * width_);
    }
    layers_.push_back(std::move(layer));
  }
  // Also detect replacement of an earlier layer while later layers decoded.
  for (const auto& input : inputs) CheckSnapshotIdentity(input.identity);
  check_cancelled();
}
size_t HostGrayscaleBank::Width() const { return width_; }
size_t HostGrayscaleBank::Height() const { return height_; }
const std::vector<std::shared_ptr<const GrayscaleLayer>>&
HostGrayscaleBank::Layers() const {
  return layers_;
}
const std::vector<int>& HostGrayscaleBank::SourceLayers(size_t job) const {
  return sources_.at(job);
}
std::vector<std::shared_ptr<const GrayscaleLayer>> HostGrayscaleBank::BindJob(
    size_t job) const {
  std::vector<std::shared_ptr<const GrayscaleLayer>> bindings;
  for (const int layer : SourceLayers(job)) {
    bindings.push_back(layers_.at(layer));
  }
  return bindings;
}
const BankWorkCounts& HostGrayscaleBank::WorkCounts() const { return work_; }

BankLimits HostInputBank::Estimate(const std::vector<BankJob>& jobs) {
  return Resolve(jobs).estimate;
}
std::vector<BankJobIdentities> HostInputBank::CaptureJobIdentities(
    const std::vector<BankJob>& jobs) {
  // Metadata-only, per job in job.inputs order (role order image, depth,
  // normal); planner reads, reloads and on-demand loads validate against
  // this snapshot instead of silently recapturing identities.
  std::vector<BankJobIdentities> snapshot;
  snapshot.reserve(jobs.size());
  for (const auto& job : jobs) {
    BankJobIdentities entries;
    entries.reserve(job.inputs.size());
    for (const auto& input : job.inputs) {
      entries.push_back(BankLayerIdentity{Identify(input.image.GetPath()),
                                          Identify(input.depth_path),
                                          Identify(input.normal_path)});
    }
    snapshot.push_back(std::move(entries));
  }
  return snapshot;
}
BankLimits HostInputBank::EstimateBounded(const std::vector<BankJob>& jobs) {
  return Resolve(jobs).estimate_bounded;
}
BankRawLimits HostInputBank::RawEstimate(const std::vector<BankJob>& jobs) {
  // Margin-free raw totals of the same layout the margined estimates use.
  const auto layout = Resolve(jobs);
  BankRawLimits raw;
  raw.host_raw_bytes = layout.host_raw_bytes;
  if (layout.host_raw_bytes < layout.normal_retained_once) {
    throw std::overflow_error("Resident bank bounded raw underflow");
  }
  raw.bounded_host_raw_bytes =
      layout.host_raw_bytes - layout.normal_retained_once;
  raw.device_raw_bytes = layout.device_raw_bytes;
  return raw;
}
NormalAllowance HostInputBank::EstimateNormalAllowance(
    const std::vector<BankJob>& jobs, size_t max_future) {
  // The bounded FIFO contract permits at most one or two future buffers.
  if (max_future == 0 || max_future > 2) {
    throw std::invalid_argument(
        "bounded normal lookahead supports max_future of 1 or 2 only");
  }
  const auto layout = Resolve(jobs);
  NormalAllowance allowance;
  allowance.owner_bytes = layout.max_ref_owner_bytes;
  allowance.transient_bytes = layout.max_ref_transient_bytes;
  allowance.future_bytes = Mul(max_future, allowance.owner_bytes);
  allowance.reference_layers = layout.reference_layers;
  return allowance;
}
HostInputBank::HostInputBank(std::vector<BankJob> jobs,
                             BankLimits limits,
                             BankNormals normals,
                             const BankJobIdentities* identity_snapshot,
                             size_t snapshot_count)
    : impl_(std::make_unique<Impl>()) {
  impl_->normals = normals;
  // Reject inconsistent binding arguments: they must either bind a full
  // snapshot or leave it unset; silently ignoring a positive count with a
  // null pointer (or a non-null pointer with zero count) would quietly
  // disable invocation identity validation.
  const bool binds = identity_snapshot != nullptr;
  if (binds != (snapshot_count != 0)) {
    throw std::invalid_argument(
        "Resident bank identity snapshot binding requires a consistent "
        "non-null pointer and positive count");
  }
  if (binds) {
    impl_->snapshot.assign(identity_snapshot,
                           identity_snapshot + snapshot_count);
  }
  impl_->layout = Resolve(jobs);
  const auto estimate = normals == BankNormals::kBounded
                            ? impl_->layout.estimate_bounded
                            : impl_->layout.estimate;
  if (limits.host_bytes < estimate.host_bytes ||
      limits.device_bytes < estimate.device_bytes) {
    throw std::invalid_argument("Resident bank exceeds admitted reservation");
  }
  if (!impl_->snapshot.empty()) {
    // Snapshot-bound constructor: the snapshot covers THIS bank's jobs (per
    // job, in job.inputs role order) captured at invocation admission; every
    // later read validates against it and recapture is never proof.
    auto& layers = impl_->layout.layers;
    impl_->layer_expected.assign(layers.size(), BankLayerIdentity{});
    if (impl_->snapshot.size() != jobs.size()) {
      throw std::invalid_argument(
          "Resident bank identity snapshot must cover every bank job");
    }
    for (size_t i = 0; i < impl_->snapshot.size(); ++i) {
      const auto& expected = impl_->snapshot.at(i);
      const auto& mappings = impl_->layout.input_layers.at(i);
      if (expected.size() != jobs.at(i).inputs.size() ||
          mappings.size() != expected.size()) {
        throw std::invalid_argument(
            "Resident bank identity snapshot/job mapping mismatch");
      }
      for (size_t k = 0; k < expected.size(); ++k) {
        const auto layer = mappings.at(k);
        const auto& ids = impl_->layout.identities.at(layer);
        if (!SameIdentity(ids[0], expected.at(k).image) ||
            !SameIdentity(ids[1], expected.at(k).depth) ||
            !SameIdentity(ids[2], expected.at(k).normal)) {
          throw std::runtime_error(
              "Resident input no longer matches the invocation identity"
              " snapshot before reload");
        }
        auto& bound = impl_->layer_expected.at(layer);
        if (!bound.image.path.empty() &&
            (!SameIdentity(bound.image, expected.at(k).image) ||
             !SameIdentity(bound.depth, expected.at(k).depth) ||
             !SameIdentity(bound.normal, expected.at(k).normal))) {
          throw std::runtime_error(
              "Resident identity snapshot inconsistency across shared layers");
        }
        if (bound.image.path.empty()) bound = expected.at(k);
      }
    }
  }
  impl_->jobs = std::move(jobs);
  for (size_t i = 0; i < impl_->layout.layers.size(); ++i) {
    auto& layer = impl_->layout.layers[i];
    const auto& ids = impl_->layout.identities[i];
    for (const auto& id : ids) CheckIdentity(id);
    const auto& image = layer.input.image;
    layer.grayscale = ReadGrayscale(image, layer.input.max_image_size, ids[0]);
    ++impl_->work.grayscale_reads;
    layer.depth.Read(ids[1].path);
    ++impl_->work.depth_reads;
    THROW_CHECK_EQ(layer.depth.GetDepth(), 1);
    if (layer.input.max_image_size > 0) {
      layer.depth.Downsize(image.GetWidth(), image.GetHeight());
    }
    THROW_CHECK_EQ(layer.depth.GetWidth(), image.GetWidth());
    THROW_CHECK_EQ(layer.depth.GetHeight(), image.GetHeight());
    if (impl_->normals == BankNormals::kEager) {
      // Eager-only: bounded banks load reference normals on demand with
      // LoadReferenceNormal, which revalidates identity, header dims, and
      // this exact Downsize path per load.
      NormalMap normal;
      normal.Read(ids[2].path);
      ++impl_->work.normal_reads;
      THROW_CHECK_EQ(normal.GetDepth(), 3);
      if (layer.input.max_image_size > 0) {
        normal.Downsize(image.GetWidth(), image.GetHeight());
      }
      THROW_CHECK_EQ(normal.GetWidth(), image.GetWidth());
      THROW_CHECK_EQ(normal.GetHeight(), image.GetHeight());
      if (layer.is_reference) layer.reference_normal = std::move(normal);
    }
    for (const auto& id : ids) CheckIdentity(id);
  }
  // Final whole-bank revalidation: FRESH filesystem checks against the
  // invocation snapshot for every role file of every layer, after ALL loads
  // complete (stored-record equality is not a filesystem revalidation; a
  // mutation of a finished layer while later layers load must reject).
  if (!impl_->layer_expected.empty()) {
    for (const auto& expected : impl_->layer_expected) {
      CheckSnapshotIdentity(expected.image);
      CheckSnapshotIdentity(expected.depth);
      CheckSnapshotIdentity(expected.normal);
    }
  } else {
    for (const auto& ids : impl_->layout.identities) {
      for (const auto& id : ids) CheckIdentity(id);
    }
  }
}
HostInputBank::~HostInputBank() = default;
HostInputBank::HostInputBank(HostInputBank&&) noexcept = default;
HostInputBank& HostInputBank::operator=(HostInputBank&&) noexcept = default;
const BankJob& HostInputBank::Job(size_t job) const {
  return impl_->jobs.at(job);
}
size_t HostInputBank::NumJobs() const { return impl_->jobs.size(); }
const std::vector<BankLayer>& HostInputBank::Layers() const {
  return impl_->layout.layers;
}
size_t HostInputBank::Width() const { return impl_->layout.width; }
size_t HostInputBank::Height() const { return impl_->layout.height; }
int HostInputBank::ReferenceLayer(size_t job) const {
  return impl_->layout.references.at(job);
}
const std::vector<int>& HostInputBank::SourceLayers(size_t job) const {
  return impl_->layout.sources.at(job);
}
const NormalMap& HostInputBank::ReferenceNormal(size_t job) const {
  if (impl_->normals == BankNormals::kBounded) {
    throw std::logic_error("bounded resident bank retains no reference normal");
  }
  return impl_->layout.layers.at(ReferenceLayer(job)).reference_normal;
}
BankNormals HostInputBank::NormalsMode() const { return impl_->normals; }
NormalMap HostInputBank::LoadReferenceNormal(size_t layer) const {
  const auto& ids = impl_->layout.identities.at(layer);
  const auto& entry = impl_->layout.layers.at(layer);
  const auto& image = entry.input.image;
  // Identity immediately before and after the read, strictly against the
  // invocation snapshot when bound; immutable admitted files must not be
  // replaced or rewritten between validation and consumption.
  const auto check = [&](const BankFileIdentity& id) {
    const auto& expected_normal = impl_->layer_expected.empty()
                                      ? BankFileIdentity{}
                                      : impl_->layer_expected.at(layer).normal;
    if (!expected_normal.path.empty()) {
      CheckSnapshotIdentity(expected_normal);
    } else {
      CheckIdentity(id);
    }
  };
  check(ids[2]);
  NormalMap normal;
  normal.Read(ids[2].path);
  THROW_CHECK_EQ(normal.GetDepth(), 3);
  if (entry.input.max_image_size > 0) {
    // Exact eager-constructor Downsize semantics.
    normal.Downsize(image.GetWidth(), image.GetHeight());
  }
  THROW_CHECK_EQ(static_cast<size_t>(normal.GetWidth()), image.GetWidth());
  THROW_CHECK_EQ(static_cast<size_t>(normal.GetHeight()), image.GetHeight());
  check(ids[2]);
  return normal;
}
const BankWorkCounts& HostInputBank::WorkCounts() const { return impl_->work; }
void HostInputBank::ReleaseUploadPayloads() {
  for (auto& layer : impl_->layout.layers) {
    layer.grayscale = Bitmap();
    layer.depth = DepthMap();
  }
}
}  // namespace colmap::mvs::owned
