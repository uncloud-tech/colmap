// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Test-executable-local GNU linker interception. Calls still reach the real
// CUDA allocator; no model counter, production allocation hook, or mock driver.
// Requires static COLMAP libraries so solver-object relocations are
// intercepted.
#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

namespace colmap::mvs::owned::strip_test {
enum class Kind { kPitched, kLinear, kArray };
struct Allocation {
  Kind kind;
  void* pointer;
  size_t row_bytes, rows, layers, pitch;
  size_t allocated, released = 0;
  cudaStream_t pending_free = nullptr;
  bool free_pending = false;
};
struct Capture {
  std::vector<Allocation> allocations;
  size_t sequence = 0, constructor_finished = 0, last_sweep = 0;
  bool invalid_release = false;
  bool fail_copy_after_submission = false, fault_submitted = false;
  bool fault_drained = false, release_before_fault_drain = false;
  cudaStream_t fault_stream = nullptr;
  cudaEvent_t fault_event = nullptr;

  void Add(Kind kind,
           void* pointer,
           size_t row,
           size_t rows,
           size_t layers,
           size_t pitch) {
    allocations.push_back(
        {kind, pointer, row, rows, layers, pitch, ++sequence});
  }
  void BeforeRelease() {
    if (fault_submitted && !fault_drained) release_before_fault_drain = true;
  }
  void Release(void* pointer,
               bool async = false,
               cudaStream_t stream = nullptr) {
    if (!pointer) return;
    for (auto i = allocations.rbegin(); i != allocations.rend(); ++i) {
      if (i->pointer != pointer || i->released) continue;
      if (i->free_pending) {
        invalid_release = true;
        return;
      }
      if (async) {
        i->free_pending = true;
        i->pending_free = stream;
      } else
        i->released = ++sequence;
      return;
    }
    invalid_release = true;
  }
  void Synchronized(cudaStream_t stream) {
    ++sequence;
    if (fault_submitted && stream == fault_stream) fault_drained = true;
    for (auto& allocation : allocations)
      if (allocation.free_pending && allocation.pending_free == stream) {
        allocation.released = sequence;
        allocation.free_pending = false;
      }
  }
  bool AllReleased() const {
    return !invalid_release &&
           std::all_of(
               allocations.begin(), allocations.end(), [](const auto& a) {
                 return a.released > a.allocated && !a.free_pending;
               });
  }
};
inline thread_local Capture* active = nullptr;
// Suppress any allocator-internal calls: record each application API call once.
struct Pause {
  Capture* capture = active;
  Pause() { active = nullptr; }
  ~Pause() { active = capture; }
};
struct Scope {
  explicit Scope(Capture& capture) {
    if (active) throw std::logic_error("nested strip allocation capture");
    active = &capture;
  }
  ~Scope() { active = nullptr; }
};
}  // namespace colmap::mvs::owned::strip_test

extern "C" {
cudaError_t CUDARTAPI __real_cudaMalloc(void**, size_t);
cudaError_t CUDARTAPI __real_cudaMallocPitch(void**, size_t*, size_t, size_t);
cudaError_t CUDARTAPI __real_cudaMalloc3DArray(cudaArray_t*,
                                               const cudaChannelFormatDesc*,
                                               cudaExtent,
                                               unsigned);
cudaError_t CUDARTAPI __real_cudaFree(void*);
cudaError_t CUDARTAPI __real_cudaFreeArray(cudaArray_t);
cudaError_t CUDARTAPI __real_cudaStreamSynchronize(cudaStream_t);
cudaError_t CUDARTAPI __real_cudaMemcpy2DAsync(void*,
                                               size_t,
                                               const void*,
                                               size_t,
                                               size_t,
                                               size_t,
                                               cudaMemcpyKind,
                                               cudaStream_t);
#if CUDART_VERSION >= 11020
cudaError_t CUDARTAPI __real_cudaMallocAsync(void**, size_t, cudaStream_t);
cudaError_t CUDARTAPI __real_cudaMallocFromPoolAsync(void**,
                                                     size_t,
                                                     cudaMemPool_t,
                                                     cudaStream_t);
cudaError_t CUDARTAPI __real_cudaFreeAsync(void*, cudaStream_t);
#endif

cudaError_t CUDARTAPI __wrap_cudaMalloc(void** pointer, size_t bytes) {
  using namespace colmap::mvs::owned::strip_test;
  Pause pause;
  const auto result = __real_cudaMalloc(pointer, bytes);
  if (result == cudaSuccess && pause.capture)
    pause.capture->Add(Kind::kLinear, *pointer, bytes, 1, 1, bytes);
  return result;
}
cudaError_t CUDARTAPI __wrap_cudaMallocPitch(void** pointer,
                                             size_t* pitch,
                                             size_t row,
                                             size_t rows) {
  using namespace colmap::mvs::owned::strip_test;
  Pause pause;
  const auto result = __real_cudaMallocPitch(pointer, pitch, row, rows);
  if (result == cudaSuccess && pause.capture)
    pause.capture->Add(Kind::kPitched, *pointer, row, rows, 1, *pitch);
  return result;
}
cudaError_t CUDARTAPI
__wrap_cudaMalloc3DArray(cudaArray_t* pointer,
                         const cudaChannelFormatDesc* format,
                         cudaExtent extent,
                         unsigned flags) {
  using namespace colmap::mvs::owned::strip_test;
  Pause pause;
  const auto result = __real_cudaMalloc3DArray(pointer, format, extent, flags);
  if (result == cudaSuccess && pause.capture) {
    const size_t bits = format->x + format->y + format->z + format->w;
    pause.capture->Add(Kind::kArray,
                       *pointer,
                       extent.width * bits / 8,
                       extent.height,
                       extent.depth,
                       0);
  }
  return result;
}
cudaError_t CUDARTAPI __wrap_cudaFree(void* pointer) {
  using namespace colmap::mvs::owned::strip_test;
  Pause pause;
  if (pause.capture) pause.capture->BeforeRelease();
  const auto result = __real_cudaFree(pointer);
  if (result == cudaSuccess && pause.capture) pause.capture->Release(pointer);
  return result;
}
cudaError_t CUDARTAPI __wrap_cudaFreeArray(cudaArray_t pointer) {
  using namespace colmap::mvs::owned::strip_test;
  Pause pause;
  if (pause.capture) pause.capture->BeforeRelease();
  const auto result = __real_cudaFreeArray(pointer);
  if (result == cudaSuccess && pause.capture) pause.capture->Release(pointer);
  return result;
}
cudaError_t CUDARTAPI __wrap_cudaStreamSynchronize(cudaStream_t stream) {
  using namespace colmap::mvs::owned::strip_test;
  Pause pause;
  const auto result = __real_cudaStreamSynchronize(stream);
  if (result == cudaSuccess && pause.capture)
    pause.capture->Synchronized(stream);
  return result;
}
cudaError_t CUDARTAPI __wrap_cudaMemcpy2DAsync(void* dst,
                                               size_t dst_pitch,
                                               const void* src,
                                               size_t src_pitch,
                                               size_t width,
                                               size_t height,
                                               cudaMemcpyKind kind,
                                               cudaStream_t stream) {
  using namespace colmap::mvs::owned::strip_test;
  Pause pause;
  const auto result = __real_cudaMemcpy2DAsync(
      dst, dst_pitch, src, src_pitch, width, height, kind, stream);
  if (result == cudaSuccess && pause.capture &&
      pause.capture->fail_copy_after_submission &&
      !pause.capture->fault_submitted) {
    // Record a real event behind real queued work, then make the caller throw.
    const auto recorded = cudaEventRecord(pause.capture->fault_event, stream);
    if (recorded != cudaSuccess) return recorded;
    pause.capture->fault_submitted = true;
    pause.capture->fault_stream = stream;
    return cudaErrorInvalidValue;
  }
  return result;
}
#if CUDART_VERSION >= 11020
cudaError_t CUDARTAPI __wrap_cudaMallocAsync(void** pointer,
                                             size_t bytes,
                                             cudaStream_t stream) {
  using namespace colmap::mvs::owned::strip_test;
  Pause pause;
  const auto result = __real_cudaMallocAsync(pointer, bytes, stream);
  if (result == cudaSuccess && pause.capture)
    pause.capture->Add(Kind::kLinear, *pointer, bytes, 1, 1, bytes);
  return result;
}
cudaError_t CUDARTAPI __wrap_cudaMallocFromPoolAsync(void** pointer,
                                                     size_t bytes,
                                                     cudaMemPool_t pool,
                                                     cudaStream_t stream) {
  using namespace colmap::mvs::owned::strip_test;
  Pause pause;
  const auto result =
      __real_cudaMallocFromPoolAsync(pointer, bytes, pool, stream);
  if (result == cudaSuccess && pause.capture)
    pause.capture->Add(Kind::kLinear, *pointer, bytes, 1, 1, bytes);
  return result;
}
cudaError_t CUDARTAPI __wrap_cudaFreeAsync(void* pointer, cudaStream_t stream) {
  using namespace colmap::mvs::owned::strip_test;
  Pause pause;
  if (pause.capture) pause.capture->BeforeRelease();
  const auto result = __real_cudaFreeAsync(pointer, stream);
  if (result == cudaSuccess && pause.capture)
    pause.capture->Release(pointer, true, stream);
  return result;
}
#endif
}  // extern "C"
