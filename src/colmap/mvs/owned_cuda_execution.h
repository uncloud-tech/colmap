// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#if !defined(COLMAP_HIP_ENABLED)
#include "colmap/util/cudacc.h"
#include "colmap/util/logging.h"
#include "colmap/util/string.h"

#include <exception>

namespace colmap {
namespace mvs {
namespace owned {

inline void SynchronizeStream(const cudaStream_t stream) {
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
  CUDA_CHECK();
}

// Declare AFTER any local texture/array owner and before submitting work.
// On an API exception this fence runs before that local owner's destructor.
class ScopedStreamCompletion {
 public:
  explicit ScopedStreamCompletion(cudaStream_t stream) : stream_(stream) {}
  ~ScopedStreamCompletion() {
    if (pending_) {
      try {
        CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
      } catch (...) {
        std::terminate();
      }
    }
  }
  ScopedStreamCompletion(const ScopedStreamCompletion&) = delete;
  ScopedStreamCompletion& operator=(const ScopedStreamCompletion&) = delete;
  // wait=false transfers completion to the enclosing owner stream lifetime.
  // On enqueue failure the destructor still fences before buffers unwind.
  void Wait(bool wait = true) {
    if (wait) CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
    pending_ = false;
    CUDA_CHECK();
  }

 private:
  cudaStream_t stream_;
  bool pending_ = true;
};

class OwnedCudaStream {
 public:
  OwnedCudaStream() {
    CUDA_SAFE_CALL(cudaGetDevice(&device_));
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }
  ~OwnedCudaStream() {
    Activate();
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream_));
    CUDA_SAFE_CALL(cudaStreamDestroy(stream_));
  }
  OwnedCudaStream(const OwnedCudaStream&) = delete;
  OwnedCudaStream& operator=(const OwnedCudaStream&) = delete;

  void Activate() const { CUDA_SAFE_CALL(cudaSetDevice(device_)); }
  cudaStream_t Get() const { return stream_; }
  int DeviceIndex() const { return device_; }
  void Synchronize() const {
    Activate();
    SynchronizeStream(stream_);
  }
  void DrainBeforeDestruction() const noexcept {
    try {
      Synchronize();
    } catch (...) {
      // Never unwind live texture/buffer owners after an unconfirmed fence.
      // This failed run cannot produce a successful result receipt.
      std::terminate();
    }
  }

 private:
  int device_ = -1;
  cudaStream_t stream_ = nullptr;
};

class OwnedCudaTimer {
 public:
  explicit OwnedCudaTimer(const cudaStream_t stream) : stream_(stream) {
    CUDA_SAFE_CALL(cudaEventCreate(&start_));
    try {
      CUDA_SAFE_CALL(cudaEventCreate(&stop_));
      CUDA_SAFE_CALL(cudaEventRecord(start_, stream_));
    } catch (...) {
      if (stop_ != nullptr) {
        (void)cudaEventDestroy(stop_);
      }
      (void)cudaEventDestroy(start_);
      throw;
    }
  }
  ~OwnedCudaTimer() {
    CUDA_SAFE_CALL(cudaEventDestroy(start_));
    CUDA_SAFE_CALL(cudaEventDestroy(stop_));
  }
  OwnedCudaTimer(const OwnedCudaTimer&) = delete;
  OwnedCudaTimer& operator=(const OwnedCudaTimer&) = delete;

  void Print(const std::string& message) {
    CUDA_SAFE_CALL(cudaEventRecord(stop_, stream_));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_));
    float elapsed_time = 0.0f;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start_, stop_));
    LOG(INFO) << StringPrintf(
        "%s: %.4fs", message.c_str(), elapsed_time / 1000.0f);
  }

 private:
  cudaStream_t stream_;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace owned
}  // namespace mvs
}  // namespace colmap
#endif  // !defined(COLMAP_HIP_ENABLED)
