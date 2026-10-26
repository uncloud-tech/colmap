// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/mat.h"
#include "colmap/util/cudacc.h"
#include "colmap/util/endian.h"
#include "colmap/util/logging.h"
#include "colmap/util/string.h"

#include <climits>
#include <fstream>

#if !defined(COLMAP_HIP_ENABLED)
#include "colmap/mvs/owned_cuda_execution.h"
#include "colmap/mvs/owned_async_pool.h"
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "colmap/mvs/cuda_flip.h"
#include "colmap/mvs/cuda_rotate.h"
#include "colmap/mvs/cuda_transpose.h"
#endif

namespace colmap {
namespace mvs {

// Lightweight, trivially-copyable view of GPU device memory suitable for
// passing to CUDA/HIP kernels. GpuMat<T> is non-trivially copyable; passing it
// by value to kernels is undefined behavior per CUDA spec. Use GpuMat::View()
// to obtain a GpuMatView for kernel args.
template <typename T>
struct GpuMatView {
  T* const ptr;
  const size_t pitch;
  const size_t width;
  const size_t height;
  const size_t depth;

  __host__ __device__ const T* GetPtr() const { return ptr; }
  __host__ __device__ T* GetPtr() { return ptr; }

  __host__ __device__ size_t GetPitch() const { return pitch; }
  __host__ __device__ size_t GetWidth() const { return width; }
  __host__ __device__ size_t GetHeight() const { return height; }
  __host__ __device__ size_t GetDepth() const { return depth; }

  __device__ T Get(const size_t row,
                   const size_t col,
                   const size_t slice = 0) const {
    return *((T*)((char*)ptr + pitch * (slice * height + row)) + col);
  }

  __device__ void GetSlice(const size_t row,
                           const size_t col,
                           T* values) const {
    for (size_t s = 0; s < depth; ++s) {
      values[s] = Get(row, col, s);
    }
  }

  __device__ T& GetRef(const size_t row, const size_t col) {
    return GetRef(row, col, 0);
  }

  __device__ T& GetRef(const size_t row, const size_t col, const size_t slice) {
    return *((T*)((char*)ptr + pitch * (slice * height + row)) + col);
  }

  __device__ void Set(const size_t row, const size_t col, const T value) {
    Set(row, col, 0, value);
  }

  __device__ void Set(const size_t row,
                      const size_t col,
                      const size_t slice,
                      const T value) {
    GetRef(row, col, slice) = value;
  }

  __device__ void SetSlice(const size_t row,
                           const size_t col,
                           const T* values) {
    for (size_t s = 0; s < depth; ++s) {
      Set(row, col, s, values[s]);
    }
  }
};

template <typename T>
class GpuMat {
 public:
  GpuMat(const size_t width, const size_t height, const size_t depth = 1);
  ~GpuMat();

  GpuMat(const GpuMat&) = delete;
  GpuMat& operator=(const GpuMat&) = delete;
  GpuMat(GpuMat&&) = delete;
  GpuMat& operator=(GpuMat&&) = delete;

  const T* GetPtr() const;
  T* GetPtr();

  size_t GetPitch() const;
  size_t GetWidth() const;
  size_t GetHeight() const;
  size_t GetDepth() const;

  // Returns a lightweight, trivially-copyable view suitable for passing
  // to CUDA/HIP kernels.
  GpuMatView<T> View() const;

  void FillWithScalar(const T value);
  void FillWithVector(const T* values);
  void FillWithRandomNumbers(const T min_value,
                             const T max_value,
                             const GpuMat<curandState>& random_state);

  void CopyToDevice(const T* data, const size_t pitch);
  void CopyToHost(T* data, const size_t pitch) const;
  Mat<T> CopyToMat() const;

  // Transpose array by swapping x and y coordinates.
  void Transpose(GpuMat<T>* output);

  // Flip array along vertical axis.
  void FlipHorizontal(GpuMat<T>* output);

  // Rotate array in counter-clockwise direction.
  void Rotate(GpuMat<T>* output);

#if !defined(COLMAP_HIP_ENABLED)
  GpuMat(size_t width, size_t height, size_t depth, owned::AsyncPool& pool);
  // Explicit-stream overloads block by default. For fill/rotation only,
  // wait=false requires all buffers to outlive the enclosing owner-stream
  // drain.
  void CopyToDevice(const T* data, size_t pitch, cudaStream_t stream);
  void CopyToHost(T* data, size_t pitch, cudaStream_t stream) const;
  Mat<T> CopyToMat(cudaStream_t stream) const;
  void FillWithScalar(T value, cudaStream_t stream, bool wait = true);
  void FillWithRandomNumbers(T min_value,
                             T max_value,
                             const GpuMat<curandState>& random_state,
                             cudaStream_t stream);
  void Rotate(GpuMat<T>* output, cudaStream_t stream, bool wait = true);
  // Rotate a disjoint, possibly packed input view into this pitched owner.
  void RotateFrom(GpuMatView<T> input, cudaStream_t stream, bool wait = true);
#endif

  void Read(const std::filesystem::path& path);
  void Write(const std::filesystem::path& path);
  void Write(const std::filesystem::path& path, const size_t slice);

 protected:
  void ComputeCudaConfig();

  const static size_t kBlockDimX = 32;
  const static size_t kBlockDimY = 16;

#if !defined(COLMAP_HIP_ENABLED)
  owned::AsyncPool* async_pool_ = nullptr;
#endif
  T* array_ptr_;

  size_t pitch_;
  const size_t width_;
  const size_t height_;
  const size_t depth_;

  dim3 blockSize_;
  dim3 gridSize_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

// Host-only template implementations are outside #ifdef __CUDACC__ so they are
// available in all translation units (not just nvcc-compiled .cu files).
// This is necessary because patch_match.cc (compiled by g++, not nvcc) needs
// GpuMat<T> constructor/destructor through std::unique_ptr<PatchMatchCuda>.

template <typename T>
GpuMat<T>::GpuMat(const size_t width, const size_t height, const size_t depth)
    : array_ptr_(nullptr), width_(width), height_(height), depth_(depth) {
  CUDA_SAFE_CALL(cudaMallocPitch(
      (void**)&array_ptr_, &pitch_, width_ * sizeof(T), height_ * depth_));

  ComputeCudaConfig();
}

#if !defined(COLMAP_HIP_ENABLED)
template <typename T>
GpuMat<T>::GpuMat(size_t width,
                  size_t height,
                  size_t depth,
                  owned::AsyncPool& pool)
    : async_pool_(&pool),
      array_ptr_(nullptr),
      width_(width),
      height_(height),
      depth_(depth) {
  const auto max = std::numeric_limits<size_t>::max();
  if (width > max / sizeof(T) || (depth && height > max / depth)) {
    throw std::overflow_error("GpuMat dimensions overflow");
  }
  array_ptr_ = static_cast<T*>(
      pool.Allocate(width * sizeof(T), height * depth, &pitch_));
  ComputeCudaConfig();
}
#endif

template <typename T>
GpuMat<T>::~GpuMat() {
  // Errors during destruction would be invisible to callers anyway; explicitly
  // discard the return value so HIP's [[nodiscard]] cudaFree does not warn.
#if !defined(COLMAP_HIP_ENABLED) && CUDART_VERSION >= 11020
  if (async_pool_) {
    if (cudaFreeAsync(array_ptr_, async_pool_->Stream()) != cudaSuccess) {
      std::terminate();
    }
    return;
  }
#endif
  (void)cudaFree(array_ptr_);
}

template <typename T>
const T* GpuMat<T>::GetPtr() const {
  return array_ptr_;
}

template <typename T>
T* GpuMat<T>::GetPtr() {
  return array_ptr_;
}

template <typename T>
size_t GpuMat<T>::GetPitch() const {
  return pitch_;
}

template <typename T>
size_t GpuMat<T>::GetWidth() const {
  return width_;
}

template <typename T>
size_t GpuMat<T>::GetHeight() const {
  return height_;
}

template <typename T>
size_t GpuMat<T>::GetDepth() const {
  return depth_;
}

template <typename T>
GpuMatView<T> GpuMat<T>::View() const {
  return {array_ptr_, pitch_, width_, height_, depth_};
}

template <typename T>
void GpuMat<T>::CopyToDevice(const T* data, const size_t pitch) {
  CUDA_SAFE_CALL(cudaMemcpy2D((void*)array_ptr_,
                              (size_t)pitch_,
                              (void*)data,
                              pitch,
                              width_ * sizeof(T),
                              height_ * depth_,
                              cudaMemcpyHostToDevice));
}

template <typename T>
void GpuMat<T>::CopyToHost(T* data, const size_t pitch) const {
  CUDA_SAFE_CALL(cudaMemcpy2D((void*)data,
                              pitch,
                              (void*)array_ptr_,
                              (size_t)pitch_,
                              width_ * sizeof(T),
                              height_ * depth_,
                              cudaMemcpyDeviceToHost));
}

template <typename T>
Mat<T> GpuMat<T>::CopyToMat() const {
  Mat<T> mat(width_, height_, depth_);
  CopyToHost(mat.GetPtr(), mat.GetWidth() * sizeof(T));
  return mat;
}

#if !defined(COLMAP_HIP_ENABLED)
template <typename T>
void GpuMat<T>::CopyToDevice(const T* data,
                              const size_t pitch,
                              const cudaStream_t stream) {
  owned::ScopedStreamCompletion completion(stream);
  CUDA_SAFE_CALL(cudaMemcpy2DAsync(array_ptr_, pitch_, data, pitch,
                                  width_ * sizeof(T), height_ * depth_,
                                  cudaMemcpyHostToDevice, stream));
  completion.Wait();
}

template <typename T>
void GpuMat<T>::CopyToHost(T* data,
                            const size_t pitch,
                            const cudaStream_t stream) const {
  owned::ScopedStreamCompletion completion(stream);
  CUDA_SAFE_CALL(cudaMemcpy2DAsync(data, pitch, array_ptr_, pitch_,
                                  width_ * sizeof(T), height_ * depth_,
                                  cudaMemcpyDeviceToHost, stream));
  completion.Wait();
}

template <typename T>
Mat<T> GpuMat<T>::CopyToMat(const cudaStream_t stream) const {
  Mat<T> mat(width_, height_, depth_);
  CopyToHost(mat.GetPtr(), mat.GetWidth() * sizeof(T), stream);
  return mat;
}
#endif

template <typename T>
void GpuMat<T>::Read(const std::filesystem::path& path) {
  std::fstream text_file(path, std::ios::in | std::ios::binary);
  THROW_CHECK(text_file.is_open()) << "Could not open " << path;
  SetFullPrecTextStream(text_file);

  size_t width;
  size_t height;
  size_t depth;
  char unused_char;
  text_file >> width >> unused_char >> height >> unused_char >> depth >>
      unused_char;
  std::streampos pos = text_file.tellg();
  text_file.close();

  THROW_CHECK_EQ(width, width_) << "File width does not match GpuMat width";
  THROW_CHECK_EQ(height, height_) << "File height does not match GpuMat height";
  THROW_CHECK_EQ(depth, depth_) << "File depth does not match GpuMat depth";

  std::fstream binary_file(path, std::ios::in | std::ios::binary);
  binary_file.seekg(pos);

  std::vector<T> source(width_ * height_ * depth_);
  ReadBinaryLittleEndian<T>(&binary_file, &source);
  binary_file.close();

  CopyToDevice(source.data(), width_ * sizeof(T));
}

template <typename T>
void GpuMat<T>::Write(const std::filesystem::path& path) {
  std::vector<T> dest(width_ * height_ * depth_);
  CopyToHost(dest.data(), width_ * sizeof(T));

  std::fstream text_file(path, std::ios::out);
  THROW_CHECK(text_file.is_open()) << "Could not open " << path;
  SetFullPrecTextStream(text_file);
  text_file << width_ << "&" << height_ << "&" << depth_ << "&";
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  THROW_CHECK(binary_file.is_open()) << "Could not open " << path;
  WriteBinaryLittleEndian<T>(&binary_file, dest);
  binary_file.close();
}

template <typename T>
void GpuMat<T>::Write(const std::filesystem::path& path, const size_t slice) {
  std::vector<T> dest(width_ * height_);
  CUDA_SAFE_CALL(
      cudaMemcpy2D((void*)dest.data(),
                   width_ * sizeof(T),
                   (void*)(array_ptr_ + slice * height_ * pitch_ / sizeof(T)),
                   pitch_,
                   width_ * sizeof(T),
                   height_,
                   cudaMemcpyDeviceToHost));

  std::fstream text_file(path, std::ios::out);
  THROW_CHECK(text_file.is_open()) << "Could not open " << path;
  SetFullPrecTextStream(text_file);
  text_file << width_ << "&" << height_ << "&" << 1 << "&";
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  THROW_CHECK(binary_file.is_open()) << "Could not open " << path;
  WriteBinaryLittleEndian<T>(&binary_file, dest);
  binary_file.close();
}

template <typename T>
void GpuMat<T>::ComputeCudaConfig() {
  blockSize_.x = kBlockDimX;
  blockSize_.y = kBlockDimY;
  blockSize_.z = 1;

  gridSize_.x = (width_ - 1) / kBlockDimX + 1;
  gridSize_.y = (height_ - 1) / kBlockDimY + 1;
  gridSize_.z = 1;
}

// Methods that use CUDA/HIP kernel launch syntax (<<<>>>) or call functions
// defined only inside __CUDACC__/__HIPCC__ blocks must remain guarded.
#if defined(__CUDACC__) || defined(__HIPCC__)

namespace internal {

template <typename T>
__global__ void FillWithScalarKernel(GpuMatView<T> output, const T value) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output.GetHeight() && col < output.GetWidth()) {
    for (size_t slice = 0; slice < output.GetDepth(); ++slice) {
      output.Set(row, col, slice, value);
    }
  }
}

template <typename T>
__global__ void FillWithVectorKernel(const T* values, GpuMatView<T> output) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output.GetHeight() && col < output.GetWidth()) {
    for (size_t slice = 0; slice < output.GetDepth(); ++slice) {
      output.Set(row, col, slice, values[slice]);
    }
  }
}

template <typename T>
__global__ void FillWithRandomNumbersKernel(
    GpuMatView<T> output,
    GpuMatView<curandState> random_state,
    const T min_value,
    const T max_value) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output.GetHeight() && col < output.GetWidth()) {
    curandState local_state = random_state.Get(row, col);
    for (size_t slice = 0; slice < output.GetDepth(); ++slice) {
      const T random_value =
          curand_uniform(&local_state) * (max_value - min_value) + min_value;
      output.Set(row, col, slice, random_value);
    }
    random_state.Set(row, col, local_state);
  }
}

}  // namespace internal

template <typename T>
void GpuMat<T>::FillWithScalar(const T value) {
  internal::FillWithScalarKernel<T><<<gridSize_, blockSize_>>>(View(), value);
  CUDA_SYNC_AND_CHECK();
}

template <typename T>
void GpuMat<T>::FillWithVector(const T* values) {
  T* values_device;
  CUDA_SAFE_CALL(cudaMalloc((void**)&values_device, depth_ * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(
      values_device, values, depth_ * sizeof(T), cudaMemcpyHostToDevice));
  internal::FillWithVectorKernel<T>
      <<<gridSize_, blockSize_>>>(values_device, View());
  CUDA_SYNC_AND_CHECK();
  CUDA_SAFE_CALL(cudaFree(values_device));
}

template <typename T>
void GpuMat<T>::FillWithRandomNumbers(const T min_value,
                                      const T max_value,
                                      const GpuMat<curandState>& random_state) {
  internal::FillWithRandomNumbersKernel<T><<<gridSize_, blockSize_>>>(
      View(), random_state.View(), min_value, max_value);
  CUDA_SYNC_AND_CHECK();
}

template <typename T>
void GpuMat<T>::Transpose(GpuMat<T>* output) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaTranspose(array_ptr_ + slice * pitch_ / sizeof(T) * GetHeight(),
                  output->GetPtr() +
                      slice * output->pitch_ / sizeof(T) * output->GetHeight(),
                  width_,
                  height_,
                  pitch_,
                  output->pitch_);
  }
  CUDA_SYNC_AND_CHECK();
}

template <typename T>
void GpuMat<T>::FlipHorizontal(GpuMat<T>* output) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaFlipHorizontal(array_ptr_ + slice * pitch_ / sizeof(T) * GetHeight(),
                       output->GetPtr() + slice * output->pitch_ / sizeof(T) *
                                              output->GetHeight(),
                       width_,
                       height_,
                       pitch_,
                       output->pitch_);
  }
  CUDA_SYNC_AND_CHECK();
}

template <typename T>
void GpuMat<T>::Rotate(GpuMat<T>* output) {
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaRotate((T*)((char*)array_ptr_ + slice * pitch_ * GetHeight()),
               (T*)((char*)output->GetPtr() +
                    slice * output->pitch_ * output->GetHeight()),
               width_,
               height_,
               pitch_,
               output->pitch_);
  }
  CUDA_SYNC_AND_CHECK();
}

#if !defined(COLMAP_HIP_ENABLED)
template <typename T>
void GpuMat<T>::FillWithScalar(const T value,
                               const cudaStream_t stream,
                               bool wait) {
  owned::ScopedStreamCompletion completion(stream);
  internal::FillWithScalarKernel<T><<<gridSize_, blockSize_, 0, stream>>>(
      View(), value);
  completion.Wait(wait);
}

template <typename T>
void GpuMat<T>::FillWithRandomNumbers(
    const T min_value,
    const T max_value,
    const GpuMat<curandState>& random_state,
    const cudaStream_t stream) {
  owned::ScopedStreamCompletion completion(stream);
  internal::FillWithRandomNumbersKernel<T>
      <<<gridSize_, blockSize_, 0, stream>>>(
          View(), random_state.View(), min_value, max_value);
  completion.Wait();
}

template <typename T>
void GpuMat<T>::RotateFrom(const GpuMatView<T> input,
                           const cudaStream_t stream,
                           bool wait) {
  THROW_CHECK(input.GetPtr() != GetPtr());
  THROW_CHECK_EQ(input.GetWidth(), GetHeight());
  THROW_CHECK_EQ(input.GetHeight(), GetWidth());
  THROW_CHECK_EQ(input.GetDepth(), GetDepth());
  THROW_CHECK_GT(input.GetWidth(), 0);
  THROW_CHECK_GT(input.GetHeight(), 0);
  THROW_CHECK_LE(input.GetWidth(), INT_MAX);
  THROW_CHECK_LE(input.GetHeight(), INT_MAX);
  THROW_CHECK_LE(input.GetPitch(), INT_MAX);
  THROW_CHECK_LE(GetPitch(), INT_MAX);
  THROW_CHECK_LE(input.GetWidth() * sizeof(T), input.GetPitch());
  owned::ScopedStreamCompletion completion(stream);
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaRotate(reinterpret_cast<const T*>(
                   reinterpret_cast<const char*>(input.GetPtr()) +
                   slice * input.GetPitch() * input.GetHeight()),
               reinterpret_cast<T*>(reinterpret_cast<char*>(GetPtr()) +
                                    slice * GetPitch() * GetHeight()),
               input.GetWidth(), input.GetHeight(), input.GetPitch(),
               GetPitch(), stream);
  }
  completion.Wait(wait);
}

template <typename T>
void GpuMat<T>::Rotate(GpuMat<T>* output,
                       const cudaStream_t stream,
                       bool wait) {
  owned::ScopedStreamCompletion completion(stream);
  for (size_t slice = 0; slice < depth_; ++slice) {
    CudaRotate((T*)((char*)array_ptr_ + slice * pitch_ * GetHeight()),
               (T*)((char*)output->GetPtr() +
                    slice * output->pitch_ * output->GetHeight()),
               width_, height_, pitch_, output->pitch_, stream);
  }
  completion.Wait(wait);
}
#endif  // !defined(COLMAP_HIP_ENABLED)

#endif  // __CUDACC__ || __HIPCC__

}  // namespace mvs
}  // namespace colmap
