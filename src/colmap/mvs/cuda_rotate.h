// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/rotation_storage_plan.h"
#include "colmap/util/cuda_to_hip.h"

namespace colmap {
namespace mvs {

// Rotate the input matrix by 90 degrees in counter-clockwise direction.
template <typename T>
void CudaRotate(const T* input,
                T* output,
                const int width,
                const int height,
                const int pitch_input,
                const int pitch_output);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDACC__) || defined(__HIPCC__)

#define TILE_DIM_ROTATE 32

namespace internal {

template <typename T>
__global__ void CudaRotateKernel(T* output_data,
                                 const T* input_data,
                                 const int width,
                                 const int height,
                                 const int input_pitch,
                                 const int output_pitch) {
  int input_x = blockDim.x * blockIdx.x + threadIdx.x;
  int input_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (input_x >= width || input_y >= height) {
    return;
  }

  int output_x = input_y;
  int output_y = width - 1 - input_x;

  *((T*)((char*)output_data + static_cast<size_t>(output_y) * output_pitch) +
    output_x) =
      *((T*)((char*)input_data + static_cast<size_t>(input_y) * input_pitch) +
        input_x);
}

// AoS utility: no floating-point interpretation, including for float scratch.
// One launch gathers/rotates one opaque word; the caller scatters it back only
// after all reads complete in the same stream. Existing kernels are unchanged.
template <int = 0>
__global__ void CudaRotateWordKernel(const uint32_t* input,
                                    uint32_t* output,
                                    const size_t width,
                                    const size_t height,
                                    const size_t words_per_element,
                                    const size_t word) {
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y;
  if (x < width) {
    const size_t index = y * width + x;
    output[rotation_storage::RotatedIndex(index, width, height)] =
        input[index * words_per_element + word];
  }
}

}  // namespace internal

template <typename T>
void CudaRotate(const T* input,
                T* output,
                const int width,
                const int height,
                const int pitch_input,
                const int pitch_output) {
  dim3 block_dim(TILE_DIM_ROTATE, 1, 1);
  dim3 grid_dim;
  grid_dim.x = (width - 1) / TILE_DIM_ROTATE + 1;
  grid_dim.y = height;

  internal::CudaRotateKernel<<<grid_dim, block_dim>>>(
      output, input, width, height, pitch_input, pitch_output);
}

inline void CudaRotateWord(const void* input,
                           void* output,
                           size_t width,
                           size_t height,
                           size_t words_per_element,
                           size_t word) {
  dim3 block_dim(TILE_DIM_ROTATE, 1, 1);
  dim3 grid_dim((width - 1) / TILE_DIM_ROTATE + 1, height, 1);
  internal::CudaRotateWordKernel<><<<grid_dim, block_dim>>>(
      static_cast<const uint32_t*>(input), static_cast<uint32_t*>(output),
      width, height, words_per_element, word);
}

#undef TILE_DIM_ROTATE

#endif  // defined(__CUDACC__) || defined(__HIPCC__)

}  // namespace mvs
}  // namespace colmap
