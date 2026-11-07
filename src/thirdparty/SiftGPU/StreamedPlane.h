// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/magic.h>
#include <sys/vfs.h>
#endif
#endif

namespace siftgpu {

using StreamCancellation = std::function<bool()>;
inline void CheckStreamCancellation(const StreamCancellation& cancelled) {
  if (cancelled && cancelled())
    throw std::runtime_error("SIFT streaming cancelled");
}

// Payload staging only, not whole-extractor admission. Two equal GPU planes,
// one host stripe, plus the 33-float coefficient buffer/constant on each side.
// Both passes reuse these allocations; no transfer overlaps another stripe.
struct StripePlan {
  size_t rows, staging_rows, gpu_bytes, host_bytes;
  StripePlan(int width,
             int height,
             int filter_width,
             size_t gpu_cap,
             size_t host_cap) {
    constexpr size_t coefficients = 33 * sizeof(float);
    if (width <= 0 || height <= 0 || width >= (1 << 23) || filter_width < 5 ||
        filter_width > 33 || filter_width % 2 == 0 || gpu_cap <= coefficients ||
        host_cap <= coefficients)
      throw std::invalid_argument("SIFT streaming dimensions/capacity");
    const size_t row_bytes = size_t(width) * sizeof(float);
    const size_t halo = size_t(filter_width - 1);
    const size_t available =
        std::min({(gpu_cap - coefficients) / row_bytes / 2,
                  (host_cap - coefficients) / row_bytes,
                  size_t(std::numeric_limits<int>::max()) / row_bytes,
                  size_t(65535)});
    // CUDA grid.y for FilterH is one block per row; retain its portable limit.
    const size_t minimum = std::min(size_t(height), halo + 1);
    if (available < minimum)
      throw std::runtime_error(
          "SIFT streaming needs two GPU row/halo planes and one host row/halo "
          "stripe");
    rows = available >= size_t(height) ? size_t(height) : available - halo;
    staging_rows = std::min(size_t(height), rows + halo);
    host_bytes = staging_rows * row_bytes + coefficients;
    gpu_bytes = 2 * staging_rows * row_bytes + coefficients;
  }
};

// One output row, two source rows plus the third-row first scalar for legacy
// flattened sampling. Alignment slack is live payload, not hidden capacity.
// Only final integer texture addresses are translated; float ops stay global.
struct SamplePlan {
  struct FlatWindow {
    size_t origin, values;
  };
  size_t source_rows, source_values, host_bytes, gpu_bytes;
  int source_width, source_height, destination_height;
  bool upsample, exact_domain;
  size_t alignment_values, capacity_values;
  SamplePlan(int sw,
             int sh,
             int dw,
             int dh,
             bool up,
             bool exact,
             size_t gpu_cap,
             size_t host_cap,
             size_t texture_alignment = sizeof(float),
             size_t capacity = 0)
      : source_width(sw),
        source_height(sh),
        destination_height(dh),
        upsample(up),
        exact_domain(exact) {
    if (texture_alignment == 0 || texture_alignment % sizeof(float) != 0 ||
        texture_alignment > size_t(INT32_MAX) || sw <= 0 || sh <= 0 ||
        dw <= 0 || dh <= 0 || sw >= (1 << 22) || dw >= (1 << 23) ||
        sh >= (1 << 22) || int64_t(sw) * sh > INT32_MAX ||
        int64_t(dw) * dh > INT32_MAX ||
        (up &&
         (int64_t(dw) != int64_t(sw) * 2 || int64_t(dh) != int64_t(sh) * 2)) ||
        (!up && (int64_t(dh - 1) * 2 >= sh)))
      throw std::invalid_argument(
          "SIFT streamed sampling domain: up requires 2x; down requires "
          "valid unclamped rows");
    capacity_values = capacity ? capacity : size_t(sw) * sh;
    if (capacity_values < size_t(sw) * sh || capacity_values > INT32_MAX)
      throw std::invalid_argument("SIFT sampling input capacity");
    source_rows = up ? std::min(2, sh) : 1;
    alignment_values = texture_alignment / sizeof(float);
    source_values = up && !exact ? std::min(capacity_values,
                                            size_t(sw) * 2 + alignment_values)
                                 : size_t(sw) * source_rows;
    if (source_values > size_t(INT32_MAX) / sizeof(float))
      throw std::overflow_error("SIFT sampling staging range");
    const size_t source = source_values * sizeof(float);
    const size_t destination = size_t(dw) * sizeof(float);
    host_bytes = std::max(source, destination);
    gpu_bytes =
        source + destination + 33 * sizeof(float);  // Live d_kernel constant.
    if (host_bytes > host_cap || gpu_bytes > gpu_cap)
      throw std::runtime_error(
          "SIFT streamed sampling needs source rows and one output row");
  }
  FlatWindow Window(int output_row) const {
    if (output_row < 0 || output_row >= destination_height)
      throw std::out_of_range("SIFT sampling output row");
    const size_t begin = upsample ? output_row / 2 : output_row * 2;
    const size_t start = begin * source_width;
    const size_t origin = upsample && !exact_domain
                              ? start / alignment_values * alignment_values
                              : start;
    const size_t end = std::min(upsample && !exact_domain
                                    ? capacity_values
                                    : size_t(source_width) * source_height,
                                (begin + (upsample ? 2 : 1)) * source_width +
                                    (upsample && !exact_domain ? 1 : 0));
    return {origin, end - origin};
  }
};

// Two scalar Gaussian inputs, scalar DoG and linear float2 gradient. All
// buffers include a one-row halo on each side, and reuse one float2 host
// stripe synchronously. Include the live Gaussian coefficient constant.
struct DogPlan {
  size_t rows, staging_rows, gpu_bytes, host_bytes;
  DogPlan(int width, int height, size_t gpu_cap, size_t host_cap) {
    constexpr size_t coefficients = 33 * sizeof(float);
    if (width <= 0 || height <= 0 || width >= (1 << 23) ||
        gpu_cap <= coefficients)
      throw std::invalid_argument("SIFT streamed DoG dimensions/capacity");
    const size_t row = size_t(width) * sizeof(float);
    const size_t available = std::min({(gpu_cap - coefficients) / row / 5,
                                       host_cap / row / 2,
                                       size_t(INT32_MAX) / row / 2,
                                       size_t(65535)});
    if (available < std::min(size_t(height), size_t(3)))
      throw std::runtime_error(
          "SIFT streamed DoG needs rows and one-row halos");
    rows = available >= size_t(height) ? size_t(height) : available - 2;
    staging_rows = std::min(size_t(height), rows + 2);
    gpu_bytes = 5 * staging_rows * row + coefficients;
    host_bytes = 2 * staging_rows * row;
  }
};

// Three scalar DoG stripes and one float4 KEY stripe, including +/-1 row
// halos. Histogram passes run afterwards, sizing their two float4 stripes
// within this same reservation. Host staging is reused.
struct KeyPlan {
  size_t rows, staging_rows, gpu_bytes, host_bytes;
  KeyPlan(int width, int height, size_t gpu_cap, size_t host_cap) {
    constexpr size_t coefficients = 33 * sizeof(float);
    if (width <= 0 || height <= 0 || width >= (1 << 23) ||
        gpu_cap <= coefficients)
      throw std::invalid_argument("SIFT streamed KEY dimensions/capacity");
    const size_t row = size_t(width) * sizeof(float);
    const size_t available = std::min({(gpu_cap - coefficients) / row / 7,
                                       host_cap / row / 4,
                                       size_t(INT32_MAX) / row / 4,
                                       size_t(65535)});
    if (available < std::min(size_t(height), size_t(3)))
      throw std::runtime_error(
          "SIFT streamed KEY needs rows and one-row halos");
    rows = available >= size_t(height) ? size_t(height) : available - 2;
    staging_rows = std::min(size_t(height), rows + 2);
    gpu_bytes = 7 * staging_rows * row + coefficients;
    host_bytes = 4 * staging_rows * row;
    // The height-one histogram pair can be larger than a tiny KEY stripe.
    const size_t histogram =
        4 * (size_t(width) + (size_t(width) + 3) / 4) * sizeof(float) +
        coefficients;
    gpu_bytes = std::max(gpu_bytes, histogram);
    if (gpu_bytes > gpu_cap)
      throw std::runtime_error("SIFT streamed histogram needs two rows");
  }
};

// Two GPU int4 arrays (positions and packed histogram records), two host
// arrays plus 64 bytes of root/position/count stack staging. Includes the
// live Gaussian constant.
struct ListPlan {
  size_t records, gpu_bytes, host_bytes;
  ListPlan(size_t gpu_cap, size_t host_cap) {
    if (gpu_cap < 164 || host_cap < 96)
      throw std::runtime_error("SIFT streamed list needs one record batch");
    records = std::min({(gpu_cap - 132) / 32,
                        (host_cap - 64) / 32,
                        size_t(65535) * 128,
                        size_t(INT32_MAX) / 16});
    gpu_bytes = records * 32 + 132;
    host_bytes = records * 32 + 64;
  }
};

// Original ReshapeFeatureListCPU arithmetic and orientation order. Capacity
// of output is two float4 records per input; packed 65535 is not an angle.
inline int ReshapePackedOrientations(const float* src, int count, float* des) {
  const static double factor = 2.0 * 3.14159265358979323846 / 65535.0;
  int retained = 0;
  for (int i = 0; i < count; ++i, src += 4) {
    unsigned short orientations[2];
    std::memcpy(orientations, src + 3, 4);
    if (orientations[0] != 65535) {
      des[0] = src[0];
      des[1] = src[1];
      des[2] = src[2];
      des[3] = float(factor * orientations[0]);
      ++retained;
      des += 4;
      if (orientations[1] != 65535 && orientations[1] != orientations[0]) {
        des[0] = src[0];
        des[1] = src[1];
        des[2] = src[2];
        des[3] = float(factor * orientations[1]);
        ++retained;
        des += 4;
      }
    }
  }
  return retained;
}

// One-feature batch: full original-width float2 sampling rows, three GPU
// float4 records (list, gathered KEY, bounds), plus the live filter constant.
// Host accounting includes list/KEY/bounds/output and the two-record reshape
// maximum, all coexisting with the gradient read staging. Never shrink sigma.
struct OrientationPlan {
  size_t gpu_bytes, host_bytes;
  OrientationPlan(int width, int rows, size_t gpu_cap, size_t host_cap) {
    if (width <= 0 || rows <= 0 || uint64_t(width) * rows * 8 > INT32_MAX)
      throw std::invalid_argument(
          "SIFT streamed orientation window dimensions");
    const size_t gradient = size_t(width) * rows * 8;
    gpu_bytes = 132 + 48 + gradient;
    host_bytes = 96 + gradient;
    if (gpu_bytes > gpu_cap || host_bytes > host_cap)
      throw std::runtime_error(
          "SIFT streamed orientation needs full sampling window");
  }
};

// One descriptor at a time: list float4, 16 int2 sampling bounds and
// 128-float descriptor, plus full-width gradient rows and live coefficients.
// Host records coexist with the synchronous gradient staging allocation.
struct DescriptorPlan {
  size_t gpu_bytes, host_bytes;
  DescriptorPlan(int width, int rows, size_t gpu_cap, size_t host_cap) {
    if (width <= 0 || rows <= 0 || uint64_t(width) * rows * 8 > INT32_MAX)
      throw std::invalid_argument("SIFT streamed descriptor window dimensions");
    const size_t gradient = size_t(width) * rows * 8;
    gpu_bytes = 132 + 16 + 128 + 512 + gradient;
    host_bytes = 16 + 128 + 512 + gradient;
    if (gpu_bytes > gpu_cap || host_bytes > host_cap)
      throw std::runtime_error(
          "SIFT streamed descriptor needs full sampling window");
  }
};

// Original LimitFeatureCount(0) over canonical octave-major counts. This
// boundary deliberately keeps whole levels and its strict > overshoot.
inline int LimitStreamedFeatureCounts(std::vector<int>& counts,
                                      int threshold,
                                      int method) {
  int64_t total = 0;
  for (int count : counts) {
    if (count < 0) throw std::invalid_argument("negative SIFT feature count");
    total += count;
  }
  if (total > INT32_MAX)
    throw std::overflow_error("SIFT feature total overflow");
  if (threshold <= 0) return int(total);
  if (method == 2) {
    size_t i = 0;
    int retained = 0;
    for (; retained < threshold && i < counts.size(); ++i)
      retained += counts[i];
    for (; i < counts.size(); ++i) counts[i] = 0;
    return retained;
  }
  size_t i = 0;
  while (i < counts.size() && total - counts[i] > threshold) {
    total -= counts[i];
    counts[i++] = 0;
  }
  return int(total);
}

// Private, exclusive, unlinked scratch. Writes are strictly sequential and
// immutable after Seal: failures never publish a partial plane as complete.
// fdatasync bounds dirty writeback per caller stripe; DONTNEED is advisory
// only. Cache charged to the job cgroup still needs measurement (not a RAM
// guarantee).
class StreamedPlane {
 public:
  StreamedPlane(const std::string& directory,
                int width,
                int height,
                int channels = 1)
      : width_(width),
        height_(height),
        channels_(channels),
        capacity_values_(uint64_t(width) * height) {
    if (directory.empty() || width <= 0 || height <= 0 ||
        (channels != 1 && channels != 2 && channels != 4 && channels != 128) ||
        uint64_t(width) * uint64_t(height) >
            uint64_t(std::numeric_limits<int64_t>::max()) /
                (sizeof(float) * channels))
      throw std::invalid_argument("SIFT scratch dimensions overflow");
#if !defined(_WIN32)
    if (Bytes() > uint64_t(std::numeric_limits<off_t>::max()))
      throw std::overflow_error("SIFT scratch offset overflow");
    std::string pattern = directory + "/sift-plane-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back(0);
#if defined(__linux__)
    fd_ = mkostemp(name.data(), O_CLOEXEC);  // atomic close-on-exec on Linux
#else
    fd_ = mkstemp(name.data());  // exclusive, mode 0600
#endif
    if (fd_ < 0) Fail("create scratch");
    if (unlink(name.data()) != 0) {
      const int saved = errno;
      close(fd_);
      fd_ = -1;
      errno = saved;
      Fail("unlink scratch");
    }
#if defined(__linux__)
    struct statfs filesystem;
    const int status = fstatfs(fd_, &filesystem);
    if (status != 0 || filesystem.f_type == TMPFS_MAGIC ||
        filesystem.f_type == RAMFS_MAGIC) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error(
          "SIFT scratch requires disk backing, not tmpfs/ramfs");
    }
    (void)posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
#endif
    if (fcntl(fd_, F_SETFD, FD_CLOEXEC) < 0) {
      const int saved = errno;
      close(fd_);
      fd_ = -1;
      errno = saved;
      Fail("scratch close-on-exec");
    }
#else
    (void)directory;
    throw std::runtime_error("SIFT file streaming requires POSIX");
#endif
  }
  ~StreamedPlane() {
#if !defined(_WIN32)
    if (fd_ >= 0) close(fd_);
#endif
  }
  StreamedPlane(const StreamedPlane&) = delete;
  StreamedPlane& operator=(const StreamedPlane&) = delete;
  int Width() const { return width_; }
  int Height() const { return height_; }
  int Channels() const { return channels_; }
  uint64_t RowBytes() const {
    return uint64_t(width_) * channels_ * sizeof(float);
  }
  uint64_t Bytes() const {
    return capacity_values_ * channels_ * sizeof(float);
  }
  uint64_t CapacityValues() const { return capacity_values_; }
 private:
  friend class StreamInputHistory;
  // Only the SIFT input owner may reopen a fully initialized generation.
  // Prefix writes leave the old flat tail untouched even when width changes.
  void BeginInputPrefix(int width, int height) {
    if (!complete_ || failed_ || channels_ != 1 || width <= 0 || height <= 0 ||
        uint64_t(width) * height > capacity_values_)
      throw std::runtime_error("SIFT invalid input history prefix");
    width_ = width;
    height_ = height;
    next_row_ = 0;
    complete_ = false;
  }
 public:
  uint64_t BytesRead() const { return read_; }
  uint64_t BytesWritten() const { return written_; }
  bool Complete() const { return complete_; }
  void Seal() {
    if (failed_ || next_row_ != height_)
      throw std::runtime_error("incomplete SIFT scratch plane");
    complete_ = true;
  }
  // Variable-length orientation expansion reserves its worst-case disk
  // capacity, then publishes only the sequentially written prefix. No RAM
  // buffering of a whole feature list and no readable unwritten tail.
  void SealPrefix(int rows) {
    if (complete_ || failed_ || rows <= 0 || rows != next_row_ ||
        rows > height_)
      throw std::runtime_error("incomplete SIFT scratch prefix");
#if !defined(_WIN32)
    if (ftruncate(fd_, off_t(RowBytes() * rows)) != 0) {
      failed_ = true;
      Fail("truncate scratch prefix");
    }
#endif
    height_ = rows;
    capacity_values_ = uint64_t(width_) * rows;
    Seal();
  }
  void ReadRows(int first,
                int count,
                float* data,
                const StreamCancellation& cancelled = {}) {
    if (!complete_) throw std::runtime_error("unpublished SIFT scratch plane");
    Transfer(first, count, data, false, cancelled);
  }
  // Scalar flat windows preserve allocation order even across row seams.
  void ReadFlat(size_t first,
                size_t count,
                float* data,
                const StreamCancellation& cancelled = {}) {
    if (!complete_) throw std::runtime_error("unpublished SIFT scratch plane");
    const uint64_t values = Bytes() / sizeof(float);
    if (channels_ != 1 || !count || first >= values || count > values - first)
      throw std::out_of_range("SIFT scratch flat range");
    Transfer(0, 1, data, false, cancelled, -1, first, count);
  }
  // A checked single interleaved record; no whole row allocation for sparse
  // histogram traversal. Uses the same exact-I/O/cancellation/cache policy.
  void ReadRecord(int x,
                  int y,
                  float* data,
                  const StreamCancellation& cancelled = {}) {
    if (!complete_) throw std::runtime_error("unpublished SIFT scratch plane");
    if (x < 0 || x >= width_) throw std::out_of_range("SIFT scratch column");
    Transfer(y, 1, data, false, cancelled, x);
  }
  // Reserve actual disk space before producing a variable-length list.
  void Reserve() {
#if defined(__linux__)
    const int status = posix_fallocate(fd_, 0, off_t(Bytes()));
    if (status) {
      errno = status;
      Fail("scratch reservation");
    }
#else
    throw std::runtime_error("SIFT list disk reservation requires Linux");
#endif
  }
  void WriteRows(int first,
                 int count,
                 const float* data,
                 const StreamCancellation& cancelled = {}) {
    if (complete_ || failed_ || first != next_row_)
      throw std::runtime_error(
          "SIFT scratch writes must be sequential and unpublished");
    try {
      Transfer(first, count, const_cast<float*>(data), true, cancelled);
      next_row_ += count;
    } catch (...) {
      failed_ = true;
      throw;
    }
  }

 private:
  [[noreturn]] static void Fail(const char* operation) {
    throw std::runtime_error(std::string("SIFT ") + operation + ": " +
                             std::strerror(errno));
  }
  void Transfer(int first,
                int count,
                float* data,
                bool write,
                const StreamCancellation& cancelled,
                int column = -1,
                size_t flat_first = 0,
                size_t flat_count = 0) {
    if (first < 0 || count <= 0 || first > height_ || count > height_ - first ||
        !data)
      throw std::out_of_range("SIFT scratch row range");
#if !defined(_WIN32)
    uint64_t offset = uint64_t(first) * RowBytes() +
                      (column < 0 ? 0 : uint64_t(column) * channels_ * 4);
    if (flat_count) offset = uint64_t(flat_first) * sizeof(float);
    const uint64_t transfer_bytes =
        flat_count ? uint64_t(flat_count) * 4
                   : (column < 0 ? uint64_t(count) * RowBytes()
                                 : uint64_t(channels_) * 4);
    const uint64_t transfer_begin = offset;
    if (transfer_bytes > std::numeric_limits<size_t>::max())
      throw std::overflow_error("SIFT scratch transfer overflow");
    size_t remaining = size_t(transfer_bytes);
    char* cursor = reinterpret_cast<char*>(data);
    while (remaining) {
      CheckStreamCancellation(cancelled);
      const size_t request = std::min(remaining, size_t(1 << 20));
      const ssize_t done = write ? pwrite(fd_, cursor, request, off_t(offset))
                                 : pread(fd_, cursor, request, off_t(offset));
      if (done < 0 && errno == EINTR) continue;
      if (done < 0) Fail(write ? "scratch write" : "scratch read");
      if (done == 0) throw std::runtime_error("SIFT scratch short/zero I/O");
      if (write)
        written_ += done;
      else
        read_ += done;
      remaining -= size_t(done);
      offset += done;
      cursor += done;
    }
    if (write) {
      int status;
      do {
#if defined(__linux__)
        status = fdatasync(fd_);
#else
        status = fsync(fd_);
#endif
      } while (status < 0 && errno == EINTR);
      if (status < 0) Fail("scratch writeback");
    }
#if defined(__linux__)
    // Include the previous stripe's partial page once it becomes a full page.
    // At most the final partial page per file is retained by this policy;
    // actual kernel cache accounting/eviction is still not guaranteed.
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size > 0) {
      const uint64_t page = uint64_t(page_size);
      const uint64_t begin = (transfer_begin / page) * page;
      const uint64_t end = (offset / page) * page;
      if (end > begin)
        (void)posix_fadvise(
            fd_, off_t(begin), off_t(end - begin), POSIX_FADV_DONTNEED);
    }
#endif
#else
    (void)cancelled;
    (void)column;
    (void)flat_first;
    (void)flat_count;
    (void)write;
    throw std::runtime_error("SIFT file streaming requires POSIX");
#endif
  }
  int width_, height_, channels_, fd_ = -1, next_row_ = 0;
  uint64_t capacity_values_;
  uint64_t read_ = 0, written_ = 0;
  bool complete_ = false, failed_ = false;
};

// One initialized grow-only allocation per extractor, never global image order.
// Mode is fixed today. Future automatic resident/streamed selection must carry
// this exact flat history across transitions, or refuse that transition.
class StreamInputHistory {
 public:
  void Invalidate() {
    valid_ = false;
    input_.reset();  // Release owned disk blocks immediately on failed extraction.
  }
  uint64_t Capacity() const { return input_ ? input_->CapacityValues() : 0; }
  const char* Case() const { return case_; }
  StreamedPlane& Begin(const std::string& directory, int width, int height) {
    if (!valid_ || (input_ && !input_->Complete()))
      throw std::runtime_error(
          "SIFT streaming input history invalid after failed extraction");
    if (width <= 0 || height <= 0 || int64_t(width) * height > INT32_MAX)
      throw std::invalid_argument("SIFT input history dimensions");
    const uint64_t active = uint64_t(width) * height;
    case_ = !input_               ? "fresh"
            : active > Capacity() ? "growth"
            : active < Capacity() ? "retained_tail"
                                  : "equal";
    // Any exception after starting a generation fails closed. No partial
    // prefix is ever a valid history, including failed growth/reservation.
    valid_ = false;
    if (active > Capacity()) {
      // Match InitTexture's free-before-grow reset. Failure refuses subsequent
      // history; no old image archive or second full-plane reservation remains.
      input_.reset();
      auto next = std::make_unique<StreamedPlane>(directory, width, height);
      next->Reserve();
      input_ = std::move(next);
    } else {
      input_->BeginInputPrefix(width, height);
    }
    valid_ =
        true;  // Complete() remains false until the entire prefix is sealed.
    return *input_;
  }

 private:
  std::unique_ptr<StreamedPlane> input_;
  const char* case_ = "fresh";
  bool valid_ = true;
};

struct StreamFilterOptions {
  size_t gpu_bytes = 0, host_bytes = 0;
  std::string scratch_directory;  // Explicit disk-backed location required.
  StreamCancellation cancelled;
};
struct StreamFilterStats {
  size_t gpu_staging_bytes = 0, host_staging_bytes = 0;
  uint64_t file_read_bytes = 0, file_written_bytes = 0;
  uint64_t h2d_bytes = 0, d2h_bytes = 0;
  uint64_t scratch_bytes = 0;  // Two new planes, in addition to input backing.
  double seconds = 0, file_seconds = 0, h2d_seconds = 0, d2h_seconds = 0,
         kernel_seconds = 0;
};
struct StreamFilterResult {
  std::unique_ptr<StreamedPlane> horizontal, output;
  StreamFilterStats stats;
};

struct StreamKeyResult {
  // Only interior KEY bytes are defined by ComputeKEY. Backing border bytes
  // are inert storage, never feature semantics. Histograms are int4 bit
  // patterns in the four-channel plane, bottom-up, with original per-row
  // grouping.
  std::unique_ptr<StreamedPlane> key;
  std::vector<std::unique_ptr<StreamedPlane>> histograms;
  StreamFilterStats stats;
};

// One float4-sized record per file row, null for zero features. GenerateList
// produces original int4 bits (x,y,residual,0); ComputeOrientationStreamed
// consumes those detected records and produces expanded float4 (x,y,s,o).
// The producer defines the phase/record interpretation, not the file owner.
struct StreamListResult {
  int count = 0;
  std::unique_ptr<StreamedPlane> records;
  StreamFilterStats stats;
};

// Same order and count as the supplied expanded xyso; one 128-F32 record
// per logical pixel (width remains 1, not a flattened 128-pixel image).
struct StreamDescriptorResult {
  int count = 0;
  std::unique_ptr<StreamedPlane> records;
  StreamFilterStats stats;
};

struct StreamDogResult {
  std::unique_ptr<StreamedPlane> dog, gradient;
  StreamFilterStats stats;
};

}  // namespace siftgpu
