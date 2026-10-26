// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#if !defined(COLMAP_HIP_ENABLED)
#include <exception>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>

namespace colmap::mvs::owned {
// Explicit per-owner pool; never changes the device default pool. Owner stream
// must outlive this object and all allocated GpuMats. No implicit cross-stream
// use.
class AsyncPool {
 public:
  explicit AsyncPool(cudaStream_t stream) : stream_(stream) {
#if CUDART_VERSION >= 11020
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    Check(cudaGetDevice(&props.location.id));
    Check(cudaMemPoolCreate(&pool_, &props));
#else
    throw std::runtime_error("Async pool requires CUDA 11.2 headers");
#endif
  }
  ~AsyncPool() {
#if CUDART_VERSION >= 11020
    // FreeAsync submissions from GpuMat destructors must retire first.
    if (cudaStreamSynchronize(stream_) != cudaSuccess) std::terminate();
    if (pool_) (void)cudaMemPoolDestroy(pool_);
#endif
  }
  AsyncPool(const AsyncPool&) = delete;
  AsyncPool& operator=(const AsyncPool&) = delete;
  void* Allocate(size_t row_bytes, size_t rows, size_t* pitch) {
    const size_t max = std::numeric_limits<size_t>::max();
    if (row_bytes == 0 || rows == 0 || row_bytes > max - 127) {
      throw std::overflow_error("Invalid pool allocation dimensions");
    }
    *pitch = (row_bytes + 127) / 128 * 128;
    if (rows > max / *pitch)
      throw std::overflow_error("Pool allocation overflow");
    void* ptr = nullptr;
#if CUDART_VERSION >= 11020
    Check(cudaMallocFromPoolAsync(&ptr, *pitch * rows, pool_, stream_));
#endif
    return ptr;
  }
  cudaStream_t Stream() const { return stream_; }

 private:
  static void Check(cudaError_t error) {
    if (error != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(error));
  }
  cudaStream_t stream_;
#if CUDART_VERSION >= 11020
  cudaMemPool_t pool_ = nullptr;
#endif
};
}  // namespace colmap::mvs::owned
#endif
