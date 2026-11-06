////////////////////////////////////////////////////////////////////////////
//	File:		ProgramCU.cu
//	Author:		Changchang Wu
//	Description : implementation of ProgramCU and all CUDA kernels
//
//	Copyright (c) 2007 University of North Carolina at Chapel Hill
//	All Rights Reserved
//
//	Permission to use, copy, modify and distribute this software and its
//	documentation for educational, research and non-profit purposes, without
//	fee, and without a written agreement is hereby granted, provided that the
//	above copyright notice and the following paragraph appear in all copies.
//
//	The University of North Carolina at Chapel Hill make no representations
//	about the suitability of this software for any purpose. It is provided
//	'as is' without express or implied warranty.
//
//	Please send BUG REPORTS to ccwu@cs.unc.edu
//
////////////////////////////////////////////////////////////////////////////

#if defined(SIFTGPU_CUDA_ENABLED)

#include "GL/glew.h"
#include "stdio.h"

#include "colmap/util/cuda_to_hip.h"

#include "CuTexImage.h"
#include "ProgramCU.h"
#include "GlobalUtil.h"
#include "CudaImageDomain.h"
#include <chrono>
#include <cmath>

//----------------------------------------------------------------
//Begin SiftGPU setting section.
//////////////////////////////////////////////////////////
#define IMUL(X,Y) __mul24(X,Y)
//#define FDIV(X,Y) ((X)/(Y))
#define FDIV(X,Y) __fdividef(X,Y)

/////////////////////////////////////////////////////////
//filter kernel width range (don't change this)
#define KERNEL_MAX_WIDTH 33
#define KERNEL_MIN_WIDTH 5

//////////////////////////////////////////////////////////
//horizontal filter block size (32, 64, 128, 256, 512)
#define FILTERH_TILE_WIDTH 128
//thread block for vertical filter. FILTERV_BLOCK_WIDTH can be (4, 8 or 16)
#define FILTERV_BLOCK_WIDTH 16
#define FILTERV_BLOCK_HEIGHT 32
//The corresponding image patch for a thread block
#define FILTERV_PIXEL_PER_THREAD 4
#define FILTERV_TILE_WIDTH FILTERV_BLOCK_WIDTH
#define FILTERV_TILE_HEIGHT (FILTERV_PIXEL_PER_THREAD * FILTERV_BLOCK_HEIGHT)


//////////////////////////////////////////////////////////
//thread block size for computing Difference of Gaussian
#define DOG_BLOCK_LOG_DIMX 7
#define DOG_BLOCK_LOG_DIMY 0
#define DOG_BLOCK_DIMX (1 << DOG_BLOCK_LOG_DIMX)
#define DOG_BLOCK_DIMY (1 << DOG_BLOCK_LOG_DIMY)

//////////////////////////////////////////////////////////
//thread block size for keypoint detection
#define KEY_BLOCK_LOG_DIMX 3
#define KEY_BLOCK_LOG_DIMY 3
#define KEY_BLOCK_DIMX (1<<KEY_BLOCK_LOG_DIMX)
#define KEY_BLOCK_DIMY (1<<KEY_BLOCK_LOG_DIMY)
//#define KEY_OFFSET_ONE
//make KEY_BLOCK_LOG_DIMX 4 will make the write coalesced..
//but it seems uncoalesced writes don't affect the speed

//////////////////////////////////////////////////////////
//thread block size for initializing list generation (64, 128, 256, 512 ...)
#define HIST_INIT_WIDTH 128
//thread block size for generating feature list (32, 64, 128, 256, 512, ...)
#define LISTGEN_BLOCK_DIM 128


/////////////////////////////////////////////////////////
//how many keypoint orientations to compute in a block
#define ORIENTATION_COMPUTE_PER_BLOCK 64
//how many keypoint descriptor to compute in a block (2, 4, 8, 16, 32)
#define DESCRIPTOR_COMPUTE_PER_BLOCK	4
#define DESCRIPTOR_COMPUTE_BLOCK_SIZE	(16 * DESCRIPTOR_COMPUTE_PER_BLOCK)
//how many keypoint descriptor to normalized in a block (32, ...)
#define DESCRIPTOR_NORMALIZ_PER_BLOCK	32



///////////////////////////////////////////
//Thread block size for visualization
//(This doesn't affect the speed of computation)
#define BLOCK_LOG_DIM 4
#define BLOCK_DIM (1 << BLOCK_LOG_DIM)

//End SiftGPU setting section.
//----------------------------------------------------------------


__device__ __constant__ float d_kernel[KERNEL_MAX_WIDTH];

const static cudaTextureDesc texDataDesc = []() {
  cudaTextureDesc textureDesc;
  memset(&textureDesc, 0, sizeof(textureDesc));
  textureDesc.readMode = cudaReadModeElementType;
  textureDesc.addressMode[0] = cudaAddressModeClamp;
  textureDesc.addressMode[1] = cudaAddressModeClamp;
  textureDesc.addressMode[2] = cudaAddressModeClamp;
  textureDesc.filterMode = cudaFilterModePoint;
  textureDesc.normalizedCoords = false;
  return textureDesc;
}();

const static cudaTextureDesc texDataBDesc = []() {
  cudaTextureDesc textureDesc;
  memset(&textureDesc, 0, sizeof(textureDesc));
  textureDesc.readMode = cudaReadModeNormalizedFloat;
  textureDesc.addressMode[0] = cudaAddressModeClamp;
  textureDesc.addressMode[1] = cudaAddressModeClamp;
  textureDesc.addressMode[2] = cudaAddressModeClamp;
  textureDesc.filterMode = cudaFilterModePoint;
  textureDesc.normalizedCoords = false;
  return textureDesc;
}();

//////////////////////////////////////////////////////////////
template<int FW> __global__ void FilterH(cudaTextureObject_t texData, float* d_result, int width)
{

	const int HALF_WIDTH = FW >> 1;
	const int CACHE_WIDTH = FILTERH_TILE_WIDTH + FW -1;
	const int CACHE_COUNT = 2 + (CACHE_WIDTH - 2)/ FILTERH_TILE_WIDTH;
	__shared__ float data[CACHE_WIDTH];
	const int bcol = IMUL(blockIdx.x, FILTERH_TILE_WIDTH);
	const int col =  bcol + threadIdx.x;
	const int index_min = IMUL(blockIdx.y, width);
	const int index_max = index_min + width - 1;
	int src_index = index_min + bcol - HALF_WIDTH + threadIdx.x;
	int cache_index = threadIdx.x;
	float value = 0;
#pragma unroll
	for(int j = 0; j < CACHE_COUNT; ++j)
	{
		if(cache_index < CACHE_WIDTH)
		{
			int fetch_index = src_index < index_min? index_min : (src_index > index_max ? index_max : src_index);
			data[cache_index] = tex1Dfetch<float>(texData,fetch_index);
			src_index += FILTERH_TILE_WIDTH;
			cache_index += FILTERH_TILE_WIDTH;
		}
	}
	__syncthreads();
	if(col >= width) return;
#pragma unroll
	for(int i = 0; i < FW; ++i)
	{
		value += (data[threadIdx.x + i]* d_kernel[i]);
	}
//	value = Conv<FW-1>(data + threadIdx.x);
	d_result[index_min + col] = value;
}



////////////////////////////////////////////////////////////////////
template<int  FW>  __global__ void FilterV(cudaTextureObject_t texData, float* d_result, int width, int height)
{
	const int HALF_WIDTH = FW >> 1;
	const int CACHE_WIDTH = FW + FILTERV_TILE_HEIGHT - 1;
	const int TEMP = CACHE_WIDTH & 0xf;
//add some extra space to avoid bank conflict
#if FILTERV_TILE_WIDTH == 16
	//make the stride 16 * n +/- 1
	const int EXTRA = (TEMP == 1 || TEMP == 0) ? 1 - TEMP : 15 - TEMP;
#elif FILTERV_TILE_WIDTH == 8
	//make the stride 16 * n +/- 2
	const int EXTRA = (TEMP == 2 || TEMP == 1 || TEMP == 0) ? 2 - TEMP : (TEMP == 15? 3 : 14 - TEMP);
#elif FILTERV_TILE_WIDTH == 4
	//make the stride 16 * n +/- 4
	const int EXTRA = (TEMP >=0 && TEMP <=4) ? 4 - TEMP : (TEMP > 12? 20 - TEMP : 12 - TEMP);
#else
#error
#endif
	const int CACHE_TRUE_WIDTH = CACHE_WIDTH + EXTRA;
	const int CACHE_COUNT = (CACHE_WIDTH + FILTERV_BLOCK_HEIGHT - 1) / FILTERV_BLOCK_HEIGHT;
	const int WRITE_COUNT = (FILTERV_TILE_HEIGHT + FILTERV_BLOCK_HEIGHT -1) / FILTERV_BLOCK_HEIGHT;
	__shared__ float data[CACHE_TRUE_WIDTH * FILTERV_TILE_WIDTH];
	const int row_block_first = IMUL(blockIdx.y, FILTERV_TILE_HEIGHT);
	const int col = IMUL(blockIdx.x, FILTERV_TILE_WIDTH) + threadIdx.x;
	const int row_first = row_block_first - HALF_WIDTH;
	const int data_index_max = IMUL(height - 1, width) + col;
	const int cache_col_start = threadIdx.y;
	const int cache_row_start = IMUL(threadIdx.x, CACHE_TRUE_WIDTH);
	int cache_index = cache_col_start + cache_row_start;
	int data_index = IMUL(row_first + cache_col_start, width) + col;

	if(col < width)
	{
#pragma unroll
		for(int i = 0; i < CACHE_COUNT; ++i)
		{
			if(cache_col_start < CACHE_WIDTH - i * FILTERV_BLOCK_HEIGHT)
			{
				int fetch_index = data_index < col ? col : (data_index > data_index_max? data_index_max : data_index);
				data[cache_index + i * FILTERV_BLOCK_HEIGHT] = tex1Dfetch<float>(texData,fetch_index);
				data_index += IMUL(FILTERV_BLOCK_HEIGHT, width);
			}
		}
	}
	__syncthreads();

	if(col >= width) return;

	int row = row_block_first + threadIdx.y;
	int index_start = cache_row_start + threadIdx.y;
#pragma unroll
	for(int i = 0; i < WRITE_COUNT;		++i,
			row += FILTERV_BLOCK_HEIGHT, index_start += FILTERV_BLOCK_HEIGHT)
	{
		if(row < height)
		{
			int index_dest = IMUL(row, width) + col;
			float value = 0;
#pragma unroll
			for(int i = 0; i < FW; ++i)
			{
				value += (data[index_start + i] * d_kernel[i]);
			}
			d_result[index_dest] = value;
		}
	}
}

template <int LOG_SCALE>
__global__ void UpsampleKernel(cudaTextureObject_t texData,
                               float* d_result,
                               int width,
                               int height,
                               bool exact_domain,
                               int source_origin = 0,
                               int destination_origin = 0) {
  const int SCALE = (1 << LOG_SCALE), SCALE_MASK = (SCALE - 1);
  const float INV_SCALE = 1.0f / (float(SCALE));
  int col = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
  if (col >= width) return;

  const int absolute_row = int(blockIdx.y) + destination_origin;
  int row = absolute_row >> LOG_SCALE;
  int index = row * width + col;
  const int neighbor =
      siftgpu::UpsampleNeighbor(col, row, width, height, exact_domain);
  int dst_row = absolute_row;
  int dst_idx = (width * dst_row + col) * SCALE;
  dst_idx -= destination_origin * width * SCALE;
  int helper = absolute_row & SCALE_MASK;
  if (helper) {
    float v11 = tex1Dfetch<float>(texData, index - source_origin * width);
    float v12 = tex1Dfetch<float>(texData, neighbor - source_origin * width);
    const int next_row = siftgpu::UpsampleRow(row + 1, height, exact_domain);
    index = next_row * width + col;
    float v21 = tex1Dfetch<float>(texData, index - source_origin * width);
    float v22 = tex1Dfetch<float>(
        texData,
        siftgpu::UpsampleNeighbor(col, row + 1, width, height, exact_domain) -
            source_origin * width);
    float w1 = INV_SCALE * helper, w2 = 1.0 - w1;
    float v1 = (v21 * w1 + w2 * v11);
    float v2 = (v22 * w1 + w2 * v12);
    d_result[dst_idx] = v1;
#pragma unroll
    for (int i = 1; i < SCALE; ++i) {
      const float r2 = i * INV_SCALE;
      const float r1 = 1.0f - r2;
      d_result[dst_idx + i] = v1 * r1 + v2 * r2;
    }
  } else {
    float v1 = tex1Dfetch<float>(texData, index - source_origin * width);
    float v2 = tex1Dfetch<float>(texData, neighbor - source_origin * width);
    d_result[dst_idx] = v1;
#pragma unroll
    for (int i = 1; i < SCALE; ++i) {
      const float r2 = i * INV_SCALE;
      const float r1 = 1.0f - r2;
      d_result[dst_idx + i] = v1 * r1 + v2 * r2;
    }
  }
}

template <int LOG_SCALE>
__global__ void StreamedUpsampleKernel(cudaTextureObject_t texData,
                                       float* d_result,
                                       int width,
                                       int height,
                                       bool exact_domain,
                                       int source_origin = 0,
                                       int destination_origin = 0) {
  const int SCALE = (1 << LOG_SCALE), SCALE_MASK = (SCALE - 1);
  const float INV_SCALE = 1.0f / (float(SCALE));
  int col = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
  if (col >= width) return;

  const int absolute_row = int(blockIdx.y) + destination_origin;
  int row = absolute_row >> LOG_SCALE;
  int index = row * width + col;
  const int neighbor =
      siftgpu::UpsampleNeighbor(col, row, width, height, exact_domain);
  int dst_row = absolute_row;
  int dst_idx = (width * dst_row + col) * SCALE;
  dst_idx -= destination_origin * width * SCALE;
  int helper = absolute_row & SCALE_MASK;
  if (helper) {
    float v11 = tex1Dfetch<float>(texData, index - source_origin);
    float v12 = tex1Dfetch<float>(texData, neighbor - source_origin);
    const int next_row = siftgpu::UpsampleRow(row + 1, height, exact_domain);
    index = next_row * width + col;
    float v21 = tex1Dfetch<float>(texData, index - source_origin);
    float v22 = tex1Dfetch<float>(
        texData,
        siftgpu::UpsampleNeighbor(col, row + 1, width, height, exact_domain) -
            source_origin);
    float w1 = INV_SCALE * helper, w2 = 1.0 - w1;
    float v1 = (v21 * w1 + w2 * v11);
    float v2 = (v22 * w1 + w2 * v12);
    d_result[dst_idx] = v1;
#pragma unroll
    for (int i = 1; i < SCALE; ++i) {
      const float r2 = i * INV_SCALE;
      const float r1 = 1.0f - r2;
      d_result[dst_idx + i] = v1 * r1 + v2 * r2;
    }
  } else {
    float v1 = tex1Dfetch<float>(texData, index - source_origin);
    float v2 = tex1Dfetch<float>(texData, neighbor - source_origin);
    d_result[dst_idx] = v1;
#pragma unroll
    for (int i = 1; i < SCALE; ++i) {
      const float r2 = i * INV_SCALE;
      const float r1 = 1.0f - r2;
      d_result[dst_idx + i] = v1 * r1 + v2 * r2;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////
void ProgramCU::SampleImageU(CuTexImage *dst, CuTexImage *src, int log_scale)
{
	int width = src->GetImgWidth(), height = src->GetImgHeight();
	CuTexImage::CuTexObj srcTex = src->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	dim3 grid((width +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH, height << log_scale);
	dim3 block(FILTERH_TILE_WIDTH);
	switch(log_scale)
	{
	case 1 : 	UpsampleKernel<1> <<< grid, block>>> (srcTex.handle, (float*) dst->_cuData, width, height, src->_exactDomain);	break;
	case 2 : 	UpsampleKernel<2> <<< grid, block>>> (srcTex.handle, (float*) dst->_cuData, width, height, src->_exactDomain);	break;
	case 3 : 	UpsampleKernel<3> <<< grid, block>>> (srcTex.handle, (float*) dst->_cuData, width, height, src->_exactDomain);	break;
	default:	break;
	}
}

template <int LOG_SCALE>
__global__ void DownsampleKernel(cudaTextureObject_t texData,
                                 float* d_result,
                                 int src_width,
                                 int dst_width,
                                 int source_origin = 0,
                                 int destination_origin = 0) {
  const int dst_col = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
  if (dst_col >= dst_width) return;
  const int src_col = min((dst_col << LOG_SCALE), (src_width - 1));
  const int dst_row = blockIdx.y;
  const int src_row = (int(blockIdx.y) + destination_origin) << LOG_SCALE;
  const int src_idx = IMUL(src_row, src_width) + src_col;
  const int dst_idx = IMUL(dst_width, dst_row) + dst_col;
  d_result[dst_idx] =
      tex1Dfetch<float>(texData, src_idx - source_origin * src_width);
}

__global__ void DownsampleKernel(cudaTextureObject_t texData, float* d_result, int src_width, int dst_width, const int log_scale)
{
	const int dst_col = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
	if(dst_col >= dst_width) return;
	const int src_col = min((dst_col << log_scale), (src_width - 1));
	const int dst_row = blockIdx.y;
	const int src_row = blockIdx.y << log_scale;
	const int src_idx = IMUL(src_row, src_width) + src_col;
	const int dst_idx = IMUL(dst_width, dst_row) + dst_col;
	d_result[dst_idx] = tex1Dfetch<float>(texData, src_idx);

}

// Bounded sampler: global legacy arithmetic with final flat-index translation.
// Published extent, not reusable staging capacity, controls hardware fetches.
siftgpu::StreamFilterResult ProgramCU::SampleImageStreamed(
    siftgpu::StreamedPlane& input,
    int width,
    int height,
    bool up,
    bool exact,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  if (input.Channels() != 1 || !input.Complete())
    throw std::invalid_argument(
        "SIFT sampling requires published scalar input");
  int device;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    throw std::runtime_error("SIFT sampling CUDA properties");
  const SamplePlan plan(input.Width(),
                        input.Height(),
                        width,
                        height,
                        up,
                        exact,
                        options.gpu_bytes,
                        options.host_bytes,
                        properties.textureAlignment,
                        input.CapacityValues());
  if (plan.source_values > size_t(properties.maxTexture1DLinear))
    throw std::runtime_error("SIFT sampling CUDA linear texture limit");
  CheckStreamCancellation(options.cancelled);
  const auto start = std::chrono::steady_clock::now();
  auto tick = start;
  const auto elapsed = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         tick)
        .count();
  };
  StreamFilterResult result;
  result.stats.host_staging_bytes = plan.host_bytes;
  result.stats.gpu_staging_bytes = plan.gpu_bytes;
  result.output =
      std::make_unique<StreamedPlane>(options.scratch_directory, width, height);
  result.stats.scratch_bytes = result.output->Bytes();
  std::unique_ptr<float[]> host(new float[plan.host_bytes / sizeof(float)]);
  CuTexImage source, destination;
  struct Drain {
    ~Drain() { cudaDeviceSynchronize(); }
  } drain;
  if (!source.InitTexture(int(plan.source_values), 1) ||
      !destination.InitTexture(width, 1))
    throw std::runtime_error("SIFT sampling stripe allocation");
  result.stats.gpu_staging_bytes = size_t(source.GetDataSize()) +
                                   size_t(destination.GetDataSize()) +
                                   33 * sizeof(float);
  if (result.stats.gpu_staging_bytes > options.gpu_bytes)
    throw std::runtime_error("SIFT sampling allocation exceeds cap");
  for (int row = 0; row < height; ++row) {
    CheckStreamCancellation(options.cancelled);
    const int begin = up ? row / 2 : row * 2;
    const auto window = plan.Window(row);
    tick = std::chrono::steady_clock::now();
    input.ReadFlat(window.origin, window.values, host.get(), options.cancelled);
    result.stats.file_seconds += elapsed();
    result.stats.file_read_bytes += uint64_t(window.values) * sizeof(float);
    source.SetImageSize(int(window.values), 1);
    tick = std::chrono::steady_clock::now();
    if (source.CopyFromHost(host.get()) != cudaSuccess)
      throw std::runtime_error("SIFT sampling H2D");
    result.stats.h2d_seconds += elapsed();
    result.stats.h2d_bytes += uint64_t(window.values) * sizeof(float);
    tick = std::chrono::steady_clock::now();
    {
      auto texture = source.BindTexture(texDataDesc,
                                        cudaCreateChannelDesc<float>(),
                                        window.values * sizeof(float));
      if (!texture.handle)
        throw std::runtime_error("SIFT sampling texture binding");
      dim3 grid(((up ? input.Width() : width) + FILTERH_TILE_WIDTH - 1) /
                    FILTERH_TILE_WIDTH,
                1);
      dim3 block(FILTERH_TILE_WIDTH);
      if (up)
        StreamedUpsampleKernel<1>
            <<<grid, block>>>(texture.handle,
                              static_cast<float*>(destination._cuData),
                              input.Width(),
                              input.Height(),
                              exact,
                              int(window.origin),
                              row);
      else
        DownsampleKernel<1>
            <<<grid, block>>>(texture.handle,
                              static_cast<float*>(destination._cuData),
                              input.Width(),
                              width,
                              begin,
                              row);
      const auto launch = cudaGetLastError();
      const auto finish = cudaDeviceSynchronize();
      if (launch != cudaSuccess || finish != cudaSuccess)
        throw std::runtime_error("SIFT sampling kernel");
    }
    result.stats.kernel_seconds += elapsed();
    tick = std::chrono::steady_clock::now();
    if (destination.CopyToHost(host.get()) != cudaSuccess)
      throw std::runtime_error("SIFT sampling D2H");
    result.stats.d2h_seconds += elapsed();
    result.stats.d2h_bytes += uint64_t(width) * sizeof(float);
    tick = std::chrono::steady_clock::now();
    result.output->WriteRows(row, 1, host.get(), options.cancelled);
    result.stats.file_seconds += elapsed();
  }
  CheckStreamCancellation(options.cancelled);
  result.output->Seal();
  result.stats.file_written_bytes = result.output->BytesWritten();
  result.stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

void ProgramCU::SampleImageD(CuTexImage *dst, CuTexImage *src, int log_scale)
{
	int src_width = src->GetImgWidth(), dst_width = dst->GetImgWidth() ;

	CuTexImage::CuTexObj srcTex = src->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	dim3 grid((dst_width +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH, dst->GetImgHeight());
	dim3 block(FILTERH_TILE_WIDTH);
	switch(log_scale)
	{
	case 1 : 	DownsampleKernel<1> <<< grid, block>>> (srcTex.handle, (float*) dst->_cuData, src_width, dst_width);	break;
	case 2 :	DownsampleKernel<2> <<< grid, block>>> (srcTex.handle, (float*) dst->_cuData, src_width, dst_width);	break;
	case 3 : 	DownsampleKernel<3> <<< grid, block>>> (srcTex.handle, (float*) dst->_cuData, src_width, dst_width);	break;
	default:	DownsampleKernel    <<< grid, block>>> (srcTex.handle, (float*) dst->_cuData, src_width, dst_width, log_scale);
	}
}

__global__ void ChannelReduce_Kernel(cudaTextureObject_t texData, float* d_result)
{
	int index = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
	d_result[index] = tex1Dfetch<float>(texData, index*4);
}

__global__ void ChannelReduce_Convert_Kernel(cudaTextureObject_t texDataF4, float* d_result)
{
	int index = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
	float4 rgba = tex1Dfetch<float4>(texDataF4, index);
	d_result[index] = 0.299f * rgba.x + 0.587f* rgba.y + 0.114f * rgba.z;
}

void ProgramCU::ReduceToSingleChannel(CuTexImage* dst, CuTexImage* src, int convert_rgb)
{
	int width = src->GetImgWidth(), height = dst->GetImgHeight() ;

	dim3 grid((width * height +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH);
	dim3 block(FILTERH_TILE_WIDTH);
	if(convert_rgb)
	{
        CuTexImage::CuTexObj srcTex = src->BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
		ChannelReduce_Convert_Kernel<<<grid, block>>>(srcTex.handle, (float*)dst->_cuData);
	}else
	{
		CuTexImage::CuTexObj srcTex = src->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
		ChannelReduce_Kernel<<<grid, block>>>(srcTex.handle, (float*)dst->_cuData);
	}
}

__global__ void ConvertByteToFloat_Kernel(cudaTextureObject_t texDataB, float* d_result)
{
	int index = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
	d_result[index] = tex1Dfetch<float>(texDataB, index);
}

void ProgramCU::ConvertByteToFloat(CuTexImage*src, CuTexImage* dst)
{
	int width = src->GetImgWidth(), height = dst->GetImgHeight() ;
	dim3 grid((width * height +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH);
	dim3 block(FILTERH_TILE_WIDTH);
	CuTexImage::CuTexObj srcTex = src->BindTexture(texDataBDesc, cudaCreateChannelDesc<float>());
	ConvertByteToFloat_Kernel<<<grid, block>>>(srcTex.handle, (float*)dst->_cuData);
}

void ProgramCU::CreateFilterKernel(float sigma, float* kernel, int& width)
{
	int i, sz = int( ceil( GlobalUtil::_FilterWidthFactor * sigma -0.5) ) ;//
	width = 2*sz + 1;

	if(width > KERNEL_MAX_WIDTH)
	{
		//filter size truncation
		sz = KERNEL_MAX_WIDTH >> 1;
		width =KERNEL_MAX_WIDTH;
	}else if(width < KERNEL_MIN_WIDTH)
	{
		sz = KERNEL_MIN_WIDTH >> 1;
		width =KERNEL_MIN_WIDTH;
	}

	float   rv = 1.0f/(sigma*sigma), v, ksum =0;

	// pre-compute filter
	for( i = -sz ; i <= sz ; ++i)
	{
		kernel[i+sz] =  v = exp(-0.5f * i * i *rv) ;
		ksum += v;
	}

	//normalize the kernel
	rv = 1.0f/ksum;
	for(i = 0; i< width ;i++) kernel[i]*=rv;
}


template<int FW> void ProgramCU::FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf)
{
	int width = src->GetImgWidth(), height = src->GetImgHeight();

	//horizontal filtering
	CuTexImage::CuTexObj srcTex = src->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	dim3 gridh((width +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH, height);
	dim3 blockh(FILTERH_TILE_WIDTH);
	FilterH<FW><<<gridh, blockh>>>(srcTex.handle, (float*)buf->_cuData, width);
	CheckErrorCUDA("FilterH");

	///vertical filtering
	CuTexImage::CuTexObj bufTex = buf->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	dim3 gridv((width + FILTERV_TILE_WIDTH - 1)/ FILTERV_TILE_WIDTH,  (height + FILTERV_TILE_HEIGHT - 1)/FILTERV_TILE_HEIGHT);
	dim3 blockv(FILTERV_TILE_WIDTH, FILTERV_BLOCK_HEIGHT);
	FilterV<FW><<<gridv, blockv>>>(bufTex.handle, (float*)dst->_cuData, width, height);
	CheckErrorCUDA("FilterV");
}

//////////////////////////////////////////////////////////////////////
// tested on 2048x1500 image, the time on pyramid construction is
// OpenGL version : 18ms
// CUDA version: 28 ms
void ProgramCU::FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf, float sigma)
{
	float filter_kernel[KERNEL_MAX_WIDTH]; int width;
	CreateFilterKernel(sigma, filter_kernel, width);
	cudaMemcpyToSymbol(d_kernel, filter_kernel, width * sizeof(float), 0, cudaMemcpyHostToDevice);

	switch(width)
	{
		case 5:		FilterImage< 5>(dst, src, buf);	break;
		case 7:		FilterImage< 7>(dst, src, buf);	break;
		case 9:		FilterImage< 9>(dst, src, buf);	break;
		case 11:	FilterImage<11>(dst, src, buf);	break;
		case 13:	FilterImage<13>(dst, src, buf);	break;
		case 15:	FilterImage<15>(dst, src, buf);	break;
		case 17:	FilterImage<17>(dst, src, buf);	break;
		case 19:	FilterImage<19>(dst, src, buf);	break;
		case 21:	FilterImage<21>(dst, src, buf);	break;
		case 23:	FilterImage<23>(dst, src, buf);	break;
		case 25:	FilterImage<25>(dst, src, buf);	break;
		case 27:	FilterImage<27>(dst, src, buf);	break;
		case 29:	FilterImage<29>(dst, src, buf);	break;
		case 31:	FilterImage<31>(dst, src, buf);	break;
		case 33:	FilterImage<33>(dst, src, buf);	break;
		default:	break;
	}

}


// Reuse exactly the resident specializations and their accumulation order.
template <int FW>
static void StreamFilterPass(CuTexImage* source,
                             float* destination,
                             bool vertical) {
  const int width = source->GetImgWidth(), height = source->GetImgHeight();
  auto texture =
      source->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
  if (!texture.handle)
    throw std::runtime_error("SIFT streaming texture binding");
  if (vertical) {
    dim3 grid((width + FILTERV_TILE_WIDTH - 1) / FILTERV_TILE_WIDTH,
              (height + FILTERV_TILE_HEIGHT - 1) / FILTERV_TILE_HEIGHT);
    dim3 block(FILTERV_TILE_WIDTH, FILTERV_BLOCK_HEIGHT);
    FilterV<FW><<<grid, block>>>(texture.handle, destination, width, height);
  } else {
    dim3 grid((width + FILTERH_TILE_WIDTH - 1) / FILTERH_TILE_WIDTH, height);
    dim3 block(FILTERH_TILE_WIDTH);
    FilterH<FW><<<grid, block>>>(texture.handle, destination, width);
  }
  const auto launch = cudaGetLastError();
  const auto finish = cudaDeviceSynchronize();
  if (launch != cudaSuccess || finish != cudaSuccess)
    throw std::runtime_error("SIFT streaming Gaussian kernel failed");
}

siftgpu::StreamFilterResult ProgramCU::FilterImageStreamed(
    siftgpu::StreamedPlane& input,
    float sigma,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  const auto start = std::chrono::steady_clock::now();
  if (input.Channels() != 1 || !input.Complete() || !std::isfinite(sigma) ||
      sigma <= 0 || !std::isfinite(GlobalUtil::_FilterWidthFactor) ||
      GlobalUtil::_FilterWidthFactor <= 0 ||
      double(sigma) * GlobalUtil::_FilterWidthFactor >
          (std::numeric_limits<int>::max() - 1) / 2)
    throw std::invalid_argument("SIFT streaming input/sigma/filter factor");
  float coefficients[KERNEL_MAX_WIDTH];
  int fw;
  CreateFilterKernel(sigma, coefficients, fw);
  const int width = input.Width(), height = input.Height();
  const StripePlan plan(
      width, height, fw, options.gpu_bytes, options.host_bytes);
  CheckStreamCancellation(options.cancelled);
  StreamFilterResult result;
  result.stats.gpu_staging_bytes = plan.gpu_bytes;
  result.stats.host_staging_bytes = plan.host_bytes;
  result.stats.scratch_bytes = 2 * input.Bytes();
  result.horizontal =
      std::make_unique<StreamedPlane>(options.scratch_directory, width, height);
  result.output =
      std::make_unique<StreamedPlane>(options.scratch_directory, width, height);
  std::unique_ptr<float[]> host(new float[plan.staging_rows * size_t(width)]);
  CuTexImage source, destination;
  // Drain before these RAII owners unwind on ANY error or cancellation.
  struct Drain {
    ~Drain() { cudaDeviceSynchronize(); }
  } drain;
  if (!source.InitTexture(width, int(plan.staging_rows)) ||
      !destination.InitTexture(width, int(plan.staging_rows)))
    throw std::runtime_error("SIFT streaming GPU stripe allocation");
  result.stats.gpu_staging_bytes = size_t(source.GetDataSize()) +
                                   size_t(destination.GetDataSize()) +
                                   sizeof(coefficients);
  auto tick = std::chrono::steady_clock::now();
  const auto elapsed = [&tick]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         tick)
        .count();
  };
  if (cudaMemcpyToSymbol(d_kernel,
                         coefficients,
                         fw * sizeof(float),
                         0,
                         cudaMemcpyHostToDevice) != cudaSuccess)
    throw std::runtime_error("SIFT streaming coefficient upload");
  result.stats.h2d_bytes += fw * sizeof(float);
  result.stats.h2d_seconds += elapsed();
  const uint64_t input_reads = input.BytesRead();
  for (bool vertical : {false, true}) {
    StreamedPlane& read = vertical ? *result.horizontal : input;
    StreamedPlane& write = vertical ? *result.output : *result.horizontal;
    for (int first = 0; first < height;) {
      CheckStreamCancellation(options.cancelled);
      const int count = int(std::min(plan.rows, size_t(height - first)));
      const int begin = vertical ? std::max(0, first - fw / 2) : first;
      const int end =
          vertical
              ? int(std::min(int64_t(height), int64_t(first) + count + fw / 2))
              : first + count;
      const int loaded = end - begin;
      tick = std::chrono::steady_clock::now();
      read.ReadRows(begin, loaded, host.get(), options.cancelled);
      result.stats.file_seconds += elapsed();
      source.SetImageSize(width, loaded);
      destination.SetImageSize(width, loaded);
      tick = std::chrono::steady_clock::now();
      if (source.CopyFromHost(host.get()) != cudaSuccess)
        throw std::runtime_error("SIFT streaming H2D copy");
      result.stats.h2d_bytes += uint64_t(loaded) * width * sizeof(float);
      result.stats.h2d_seconds += elapsed();
      tick = std::chrono::steady_clock::now();
#define SIFT_STREAM_CASE(FW)                                          \
  case FW:                                                            \
    StreamFilterPass<FW>(                                             \
        &source, static_cast<float*>(destination._cuData), vertical); \
    break
      switch (fw) {
        SIFT_STREAM_CASE(5);
        SIFT_STREAM_CASE(7);
        SIFT_STREAM_CASE(9);
        SIFT_STREAM_CASE(11);
        SIFT_STREAM_CASE(13);
        SIFT_STREAM_CASE(15);
        SIFT_STREAM_CASE(17);
        SIFT_STREAM_CASE(19);
        SIFT_STREAM_CASE(21);
        SIFT_STREAM_CASE(23);
        SIFT_STREAM_CASE(25);
        SIFT_STREAM_CASE(27);
        SIFT_STREAM_CASE(29);
        SIFT_STREAM_CASE(31);
        SIFT_STREAM_CASE(33);
        default:
          throw std::runtime_error("SIFT streaming unsupported filter width");
      }
#undef SIFT_STREAM_CASE
      result.stats.kernel_seconds += elapsed();
      tick = std::chrono::steady_clock::now();
      if (destination.CopyToHost(host.get()) != cudaSuccess)
        throw std::runtime_error("SIFT streaming D2H copy");
      result.stats.d2h_bytes += uint64_t(loaded) * width * sizeof(float);
      result.stats.d2h_seconds += elapsed();
      tick = std::chrono::steady_clock::now();
      write.WriteRows(first,
                      count,
                      host.get() + size_t(first - begin) * width,
                      options.cancelled);
      result.stats.file_seconds += elapsed();
      first += count;
    }
    CheckStreamCancellation(options.cancelled);
    write.Seal();
  }
  result.stats.file_read_bytes =
      input.BytesRead() - input_reads + result.horizontal->BytesRead();
  result.stats.file_written_bytes =
      result.horizontal->BytesWritten() + result.output->BytesWritten();
  result.stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

void __global__ ComputeDOG_Kernel(cudaTextureObject_t texC, cudaTextureObject_t texP, float* d_dog, float2* d_got, int width, int height)
{
	int row = (blockIdx.y << DOG_BLOCK_LOG_DIMY) + threadIdx.y;
	int col = (blockIdx.x << DOG_BLOCK_LOG_DIMX) + threadIdx.x;
	if(col < width && row < height)
	{
		int index = IMUL(row, width) + col;
		float vp = tex1Dfetch<float>(texP, index);
		float v = tex1Dfetch<float>(texC, index);
		d_dog[index] = v - vp;
		// The gradient stencil reaches one pixel in each direction. Clamp it to
		// the image instead of letting the border wrap into the neighbouring row
		// or read past the end of the buffer: this is a linear texture binding,
		// where the address mode does not apply, so the corner reads really were
		// out of range.
		const int colp = max(col - 1, 0);
		const int coln = min(col + 1, width - 1);
		const int rowp = max(row - 1, 0);
		const int rown = min(row + 1, height - 1);
		float vxn = tex1Dfetch<float>(texC, IMUL(row, width) + coln);
		float vxp = tex1Dfetch<float>(texC, IMUL(row, width) + colp);
		float vyp = tex1Dfetch<float>(texC, IMUL(rowp, width) + col);
		float vyn = tex1Dfetch<float>(texC, IMUL(rown, width) + col);
		float dx = vxn - vxp, dy = vyn - vyp;
		float grd = 0.5f * sqrt(dx * dx  + dy * dy);
		float rot = (grd == 0.0f? 0.0f : atan2(dy, dx));
		d_got[index] = make_float2(grd, rot);
	}
}

void __global__ ComputeDOG_Kernel(cudaTextureObject_t texC, cudaTextureObject_t texP, float* d_dog, int width, int height)
{
	int row = (blockIdx.y << DOG_BLOCK_LOG_DIMY) + threadIdx.y;
	int col = (blockIdx.x << DOG_BLOCK_LOG_DIMX) + threadIdx.x;
	if(col < width && row < height)
	{
		int index = IMUL(row, width) + col;
		float vp = tex1Dfetch<float>(texP, index);
		float v = tex1Dfetch<float>(texC, index);
		d_dog[index] = v - vp;
	}
}

void ProgramCU::ComputeDOG(CuTexImage* gus, CuTexImage* dog, CuTexImage* got)
{
	int width = gus->GetImgWidth(), height = gus->GetImgHeight();
	dim3 grid((width + DOG_BLOCK_DIMX - 1)/ DOG_BLOCK_DIMX,  (height + DOG_BLOCK_DIMY - 1)/DOG_BLOCK_DIMY);
	dim3 block(DOG_BLOCK_DIMX, DOG_BLOCK_DIMY);
	CuTexImage::CuTexObj texCObj = gus->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	CuTexImage::CuTexObj texPObj = (gus-1)->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	if(got->_cuData)
		ComputeDOG_Kernel<<<grid, block>>>(texCObj.handle, texPObj.handle, (float*) dog->_cuData, (float2*) got->_cuData, width, height);
	else
		ComputeDOG_Kernel<<<grid, block>>>(texCObj.handle, texPObj.handle, (float*) dog->_cuData, width, height);
}

siftgpu::StreamDogResult ProgramCU::ComputeDOGStreamed(
    siftgpu::StreamedPlane& current,
    siftgpu::StreamedPlane& previous,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  const auto start = std::chrono::steady_clock::now();
  if (!current.Complete() || !previous.Complete() || current.Channels() != 1 ||
      previous.Channels() != 1 || current.Width() != previous.Width() ||
      current.Height() != previous.Height())
    throw std::invalid_argument(
        "SIFT streamed DoG requires matching scalar planes");
  const int width = current.Width(), height = current.Height();
  const DogPlan plan(width, height, options.gpu_bytes, options.host_bytes);
  CheckStreamCancellation(options.cancelled);
  StreamDogResult result;
  result.stats.gpu_staging_bytes = plan.gpu_bytes;
  result.stats.host_staging_bytes = plan.host_bytes;
  result.dog =
      std::make_unique<StreamedPlane>(options.scratch_directory, width, height);
  result.gradient = std::make_unique<StreamedPlane>(
      options.scratch_directory, width, height, 2);
  result.stats.scratch_bytes = result.dog->Bytes() + result.gradient->Bytes();
  std::unique_ptr<float[]> host(
      new float[2 * plan.staging_rows * size_t(width)]);
  // Preserve the original previous/current Gaussian pair correspondence.
  CuTexImage gaussian[2], dog, gradient;
  struct Drain {
    ~Drain() { cudaDeviceSynchronize(); }
  } drain;
  for (auto& plane : gaussian)
    if (!plane.InitTexture(width, int(plan.staging_rows)))
      throw std::runtime_error("SIFT streamed DoG source allocation");
  if (!dog.InitTexture(width, int(plan.staging_rows)) ||
      !gradient.InitTexture(width, int(plan.staging_rows), 2))
    throw std::runtime_error("SIFT streamed DoG output allocation");
  result.stats.gpu_staging_bytes =
      size_t(gaussian[0].GetDataSize()) + size_t(gaussian[1].GetDataSize()) +
      size_t(dog.GetDataSize()) + size_t(gradient.GetDataSize()) +
      33 * sizeof(float);
  auto tick = start;
  const auto elapsed = [&tick]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         tick)
        .count();
  };
  const uint64_t before_c = current.BytesRead(),
                 before_p = previous.BytesRead();
  for (int first = 0; first < height;) {
    CheckStreamCancellation(options.cancelled);
    const int count = int(std::min(plan.rows, size_t(height - first)));
    const int begin = std::max(0, first - 1);
    const int end = int(std::min(int64_t(height), int64_t(first) + count + 1));
    const int loaded = end - begin;
    for (int i = 0; i < 2; ++i) {
      tick = std::chrono::steady_clock::now();
      (i ? current : previous)
          .ReadRows(begin, loaded, host.get(), options.cancelled);
      result.stats.file_seconds += elapsed();
      gaussian[i].SetImageSize(width, loaded);
      tick = std::chrono::steady_clock::now();
      if (gaussian[i].CopyFromHost(host.get()) != cudaSuccess)
        throw std::runtime_error("SIFT streamed DoG H2D copy");
      result.stats.h2d_bytes += uint64_t(loaded) * current.RowBytes();
      result.stats.h2d_seconds += elapsed();
    }
    dog.SetImageSize(width, loaded);
    gradient.SetImageSize(width, loaded);
    tick = std::chrono::steady_clock::now();
    // No new math or coordinates: full-width row translation preserves every
    // integer fetch. Only discarded halo outputs can see a stripe boundary;
    // retained edge rows clamp iff they are true global image edges.
    auto texC =
        gaussian[1].BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
    auto texP =
        gaussian[0].BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
    // BindTexture's legacy logging clears errors. Check its actual handles so
    // a failed binding cannot be mistaken for successful streamed output.
    if (!texC.handle || !texP.handle)
      throw std::runtime_error("SIFT streamed DoG texture binding");
    dim3 grid((width + DOG_BLOCK_DIMX - 1) / DOG_BLOCK_DIMX,
              (loaded + DOG_BLOCK_DIMY - 1) / DOG_BLOCK_DIMY);
    dim3 block(DOG_BLOCK_DIMX, DOG_BLOCK_DIMY);
    ComputeDOG_Kernel<<<grid, block>>>(texC.handle,
                                       texP.handle,
                                       static_cast<float*>(dog._cuData),
                                       static_cast<float2*>(gradient._cuData),
                                       width,
                                       loaded);
    const auto launch = cudaGetLastError();
    const auto finish = cudaDeviceSynchronize();
    if (launch != cudaSuccess || finish != cudaSuccess)
      throw std::runtime_error("SIFT streamed DoG kernel failed");
    result.stats.kernel_seconds += elapsed();
    for (bool grad : {false, true}) {
      tick = std::chrono::steady_clock::now();
      if ((grad ? gradient : dog).CopyToHost(host.get()) != cudaSuccess)
        throw std::runtime_error("SIFT streamed DoG D2H copy");
      auto& output = grad ? *result.gradient : *result.dog;
      result.stats.d2h_bytes += uint64_t(loaded) * output.RowBytes();
      result.stats.d2h_seconds += elapsed();
      tick = std::chrono::steady_clock::now();
      output.WriteRows(
          first,
          count,
          host.get() + size_t(first - begin) * width * output.Channels(),
          options.cancelled);
      result.stats.file_seconds += elapsed();
    }
    first += count;
  }
  CheckStreamCancellation(options.cancelled);
  result.dog->Seal();
  result.gradient->Seal();
  result.stats.file_read_bytes = current.BytesRead() - before_c;
  if (&current != &previous)
    result.stats.file_read_bytes += previous.BytesRead() - before_p;
  result.stats.file_written_bytes =
      result.dog->BytesWritten() + result.gradient->BytesWritten();
  result.stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

#define READ_CMP_DOG_DATA(datai, tex, idx) \
		datai[0] = tex1Dfetch<float>(tex, idx - 1);\
		datai[1] = tex1Dfetch<float>(tex, idx);\
		datai[2] = tex1Dfetch<float>(tex, idx + 1);\
		if(v > nmax)\
		{\
			   nmax = max(nmax, datai[0]);\
			   nmax = max(nmax, datai[1]);\
			   nmax = max(nmax, datai[2]);\
			   if(v < nmax) goto key_finish;\
		}else\
		{\
			   nmin = min(nmin, datai[0]);\
			   nmin = min(nmin, datai[1]);\
			   nmin = min(nmin, datai[2]);\
			   if(v > nmin) goto key_finish;\
		}


void __global__ ComputeKEY_Kernel(cudaTextureObject_t texP, cudaTextureObject_t texC, cudaTextureObject_t texN, float4* d_key, int width, int colmax, int rowmax,
					float dog_threshold0,  float dog_threshold, float edge_threshold, int subpixel_localization)
{
       float data[3][3], v;
       float datap[3][3], datan[3][3];
#ifdef KEY_OFFSET_ONE
       int row = (blockIdx.y << KEY_BLOCK_LOG_DIMY) + threadIdx.y + 1;
       int col = (blockIdx.x << KEY_BLOCK_LOG_DIMX) + threadIdx.x + 1;
#else
       int row = (blockIdx.y << KEY_BLOCK_LOG_DIMY) + threadIdx.y;
       int col = (blockIdx.x << KEY_BLOCK_LOG_DIMX) + threadIdx.x;
#endif
       int index = IMUL(row, width) + col;
	   int idx[3] ={index - width, index, index + width};
       int in_image =0;
       float nmax, nmin, result = 0.0f;
	   float dx = 0, dy = 0, ds = 0;
	   bool offset_test_passed = true;
#ifdef KEY_OFFSET_ONE
       if(row < rowmax && col < colmax)
#else
       if(row > 0 && col > 0 && row < rowmax && col < colmax)
#endif
       {
			in_image = 1;
			data[1][1] = v = tex1Dfetch<float>(texC, idx[1]);
			if(fabs(v) <= dog_threshold0) goto key_finish;

			data[1][0] = tex1Dfetch<float>(texC, idx[1] - 1);
			data[1][2] = tex1Dfetch<float>(texC, idx[1] + 1);
			nmax = max(data[1][0], data[1][2]);
			nmin = min(data[1][0], data[1][2]);

			if(v <=nmax && v >= nmin) goto key_finish;
			//if((v > nmax && v < 0 )|| (v < nmin && v > 0)) goto key_finish;
			READ_CMP_DOG_DATA(data[0], texC, idx[0]);
			READ_CMP_DOG_DATA(data[2], texC, idx[2]);

			//edge supression
			float vx2 = v * 2.0f;
			float fxx = data[1][0] + data[1][2] - vx2;
			float fyy = data[0][1] + data[2][1] - vx2;
			float fxy = 0.25f * (data[2][2] + data[0][0] - data[2][0] - data[0][2]);
			float temp1 = fxx * fyy - fxy * fxy;
			float temp2 = (fxx + fyy) * (fxx + fyy);
			if(temp1 <=0 || temp2 > edge_threshold * temp1) goto key_finish;


			//read the previous level
			READ_CMP_DOG_DATA(datap[0], texP, idx[0]);
			READ_CMP_DOG_DATA(datap[1], texP, idx[1]);
			READ_CMP_DOG_DATA(datap[2], texP, idx[2]);


			//read the next level
			READ_CMP_DOG_DATA(datan[0], texN, idx[0]);
			READ_CMP_DOG_DATA(datan[1], texN, idx[1]);
			READ_CMP_DOG_DATA(datan[2], texN, idx[2]);

			if(subpixel_localization)
			{
				//subpixel localization
				float fx = 0.5f * (data[1][2] - data[1][0]);
				float fy = 0.5f * (data[2][1] - data[0][1]);
				float fs = 0.5f * (datan[1][1] - datap[1][1]);

				float fss = (datan[1][1] + datap[1][1] - vx2);
				float fxs = 0.25f* (datan[1][2] + datap[1][0] - datan[1][0] - datap[1][2]);
				float fys = 0.25f* (datan[2][1] + datap[0][1] - datan[0][1] - datap[2][1]);

				//need to solve dx, dy, ds;
				// |-fx|     | fxx fxy fxs |   |dx|
				// |-fy|  =  | fxy fyy fys | * |dy|
				// |-fs|     | fxs fys fss |   |ds|
				float4 A0 = fxx > 0? make_float4(fxx, fxy, fxs, -fx) : make_float4(-fxx, -fxy, -fxs, fx);
				float4 A1 = fxy > 0? make_float4(fxy, fyy, fys, -fy) : make_float4(-fxy, -fyy, -fys, fy);
				float4 A2 = fxs > 0? make_float4(fxs, fys, fss, -fs) : make_float4(-fxs, -fys, -fss, fs);
				float maxa = max(max(A0.x, A1.x), A2.x);
				if(maxa >= 1e-10)
				{
					if(maxa == A1.x)
					{
						float4 TEMP = A1; A1 = A0; A0 = TEMP;
					}else if(maxa == A2.x)
					{
						float4 TEMP = A2; A2 = A0; A0 = TEMP;
					}
					A0.y /= A0.x;	A0.z /= A0.x;	A0.w/= A0.x;
					A1.y -= A1.x * A0.y;	A1.z -= A1.x * A0.z;	A1.w -= A1.x * A0.w;
					A2.y -= A2.x * A0.y;	A2.z -= A2.x * A0.z;	A2.w -= A2.x * A0.w;
					if(abs(A2.y) > abs(A1.y))
					{
						float4 TEMP = A2;	A2 = A1; A1 = TEMP;
					}
					if(abs(A1.y) >= 1e-10)
					{
						A1.z /= A1.y;	A1.w /= A1.y;
						A2.z -= A2.y * A1.z;	A2.w -= A2.y * A1.w;
						if(abs(A2.z) >= 1e-10)
						{
							ds = A2.w / A2.z;
							dy = A1.w - ds * A1.z;
							dx = A0.w - ds * A0.z - dy * A0.y;

							offset_test_passed =
								fabs(data[1][1] + 0.5f * (dx * fx + dy * fy + ds * fs)) > dog_threshold
								&&fabs(ds) < 1.0f && fabs(dx) < 1.0f && fabs(dy) < 1.0f;
						}
					}
				}
			}
			if(offset_test_passed) result = v > nmax ? 1.0 : -1.0;
       }
key_finish:
       if(in_image) d_key[index] = make_float4(result, dx, dy, ds);
}


void ProgramCU::ComputeKEY(CuTexImage* dog, CuTexImage* key, float Tdog, float Tedge)
{
	int width = dog->GetImgWidth(), height = dog->GetImgHeight();
	float Tdog1 = (GlobalUtil::_SubpixelLocalization? 0.8f : 1.0f) * Tdog;
	CuTexImage* dogp = dog - 1;
	CuTexImage* dogn = dog + 1;
#ifdef KEY_OFFSET_ONE
	dim3 grid((width - 1 + KEY_BLOCK_DIMX - 1)/ KEY_BLOCK_DIMX,  (height - 1 + KEY_BLOCK_DIMY - 1)/KEY_BLOCK_DIMY);
#else
	dim3 grid((width + KEY_BLOCK_DIMX - 1)/ KEY_BLOCK_DIMX,  (height + KEY_BLOCK_DIMY - 1)/KEY_BLOCK_DIMY);
#endif
	dim3 block(KEY_BLOCK_DIMX, KEY_BLOCK_DIMY);
	CuTexImage::CuTexObj texPObj = dogp->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	CuTexImage::CuTexObj texCObj = dog->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	CuTexImage::CuTexObj texNObj = dogn->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	Tedge = (Tedge+1)*(Tedge+1)/Tedge;
	ComputeKEY_Kernel<<<grid, block>>>(texPObj.handle, texCObj.handle, texNObj.handle, (float4*) key->_cuData, width,
        width -1, height -1, Tdog1, Tdog, Tedge, GlobalUtil::_SubpixelLocalization);

}

void __global__ InitHist_Kernel(cudaTextureObject_t texDataF4, int4* hist, int ws, int wd, int height,
                                int row_origin = 0, int full_height = 0)
{
       int row = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
       int col = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	   if(row < height && col < wd)
	   {
			int hidx = IMUL(row, wd) + col;
			int scol = col << 2;
			int sidx = IMUL(row, ws) + scol;
			int v[4] = {0, 0, 0, 0};
			if(row + row_origin > 0 &&
               row + row_origin < (full_height ? full_height : height) - 1)
			{
#pragma unroll
				for(int i = 0; i < 4 ; ++i, ++scol)
				{
					if(siftgpu::IsFeatureColumn(scol, ws))
                    {
                        float4 temp = tex1Dfetch<float4>(texDataF4, sidx + i);
                        v[i] = temp.x != 0 ? 1 : 0;
                    }
				}
			}
			hist[hidx] = make_int4(v[0], v[1], v[2], v[3]);

	   }
}



void ProgramCU::InitHistogram(CuTexImage* key, CuTexImage* hist)
{
	int ws = key->GetImgWidth(), hs = key->GetImgHeight();
	int wd = hist->GetImgWidth(), hd = hist->GetImgHeight();
	dim3 grid((wd  + HIST_INIT_WIDTH - 1)/ HIST_INIT_WIDTH,  hd);
	dim3 block(HIST_INIT_WIDTH, 1);
    CuTexImage::CuTexObj keyTex = key->BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
	InitHist_Kernel<<<grid, block>>>(keyTex.handle, (int4*) hist->_cuData, ws, wd, hd);
}



void __global__ ReduceHist_Kernel(cudaTextureObject_t texDataI4, int4* d_hist, int ws, int wd, int height)
{
       int row = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;
       int col = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	   if(row < height && col < wd)
	   {
			int hidx = IMUL(row, wd) + col;
			int scol = col << 2;
			int sidx = IMUL(row, ws) + scol;
			int v[4] = {0, 0, 0, 0};
#pragma unroll
			for(int i = 0; i < 4 && scol < ws; ++i, ++scol)
			{
				int4 temp = tex1Dfetch<int4>(texDataI4, sidx + i);
				v[i] = temp.x + temp.y + temp.z + temp.w;
			}
			d_hist[hidx] = make_int4(v[0], v[1], v[2], v[3]);
	   }
}

void ProgramCU::ReduceHistogram(CuTexImage*hist1, CuTexImage* hist2)
{
	int ws = hist1->GetImgWidth(), hs = hist1->GetImgHeight();
	int wd = hist2->GetImgWidth(), hd = hist2->GetImgHeight();
	int temp = (int)floorf(logf(float(wd * 2/ 3)) / logf(2.0f));
	const int wi = min(7, max(temp , 0));
    CuTexImage::CuTexObj hist1Tex = hist1->BindTexture(texDataDesc, cudaCreateChannelDesc<int4>());

	const int BW = 1 << wi, BH =  1 << (7 - wi);
	dim3 grid((wd  + BW - 1)/ BW,  (hd + BH -1) / BH);
	dim3 block(BW, BH);
	ReduceHist_Kernel<<<grid, block>>>(hist1Tex.handle, (int4*)hist2->_cuData, ws, wd, hd);
}

siftgpu::StreamKeyResult ProgramCU::ComputeKEYStreamed(
    siftgpu::StreamedPlane& previous,
    siftgpu::StreamedPlane& current,
    siftgpu::StreamedPlane& next,
    float Tdog,
    float Tedge,
    bool exact_domain,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  const auto start = std::chrono::steady_clock::now();
  const int width = current.Width(), height = current.Height();
  for (auto* plane : {&previous, &current, &next})
    if (!plane->Complete() || plane->Channels() != 1 ||
        plane->Width() != width || plane->Height() != height)
      throw std::invalid_argument(
          "SIFT streamed KEY requires three scalar planes");
  // The rounded resident path has full four-column domains, never a partial
  // last group. Exact domains retain the odd-width ceil grouping.
  if (!exact_domain && width % 4)
    throw std::invalid_argument(
        "SIFT rounded histogram requires aligned width");
  const KeyPlan plan(width, height, options.gpu_bytes, options.host_bytes);
  CheckStreamCancellation(options.cancelled);
  StreamKeyResult result;
  auto& stats = result.stats;
  stats.gpu_staging_bytes = plan.gpu_bytes;
  stats.host_staging_bytes = plan.host_bytes;
  result.key = std::make_unique<StreamedPlane>(
      options.scratch_directory, width, height, 4);
  std::unique_ptr<float[]> host(
      new float[4 * plan.staging_rows * size_t(width)]);
  const auto timed = [](double& seconds, auto operation) {
    const auto begin = std::chrono::steady_clock::now();
    operation();
    seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
  };
  const auto finish = [] {
    const auto launch = cudaGetLastError();
    const auto sync = cudaDeviceSynchronize();
    if (launch != cudaSuccess || sync != cudaSuccess)
      throw std::runtime_error("SIFT streamed KEY/histogram kernel failed");
  };
  const auto upload =
      [&](CuTexImage& texture, StreamedPlane& plane, int row, int count) {
        CheckStreamCancellation(options.cancelled);
        timed(stats.file_seconds, [&] {
          plane.ReadRows(row, count, host.get(), options.cancelled);
        });
        stats.file_read_bytes += uint64_t(count) * plane.RowBytes();
        texture.SetImageSize(plane.Width(), count);
        timed(stats.h2d_seconds, [&] {
          if (texture.CopyFromHost(host.get()) != cudaSuccess)
            throw std::runtime_error("SIFT streamed KEY/histogram upload");
        });
        stats.h2d_bytes += uint64_t(count) * plane.RowBytes();
      };
  const auto download = [&](CuTexImage& texture,
                            StreamedPlane& plane,
                            int first,
                            int count,
                            int offset,
                            int loaded) {
    timed(stats.d2h_seconds, [&] {
      if (texture.CopyToHost(host.get()) != cudaSuccess)
        throw std::runtime_error("SIFT streamed KEY/histogram download");
    });
    stats.d2h_bytes += uint64_t(loaded) * plane.RowBytes();
    timed(stats.file_seconds, [&] {
      plane.WriteRows(first,
                      count,
                      host.get() + size_t(offset) * plane.Width() * 4,
                      options.cancelled);
    });
    stats.file_written_bytes += uint64_t(count) * plane.RowBytes();
  };
  {
    CuTexImage dogs[3], key;
    struct Drain {
      ~Drain() { cudaDeviceSynchronize(); }
    } drain;
    for (auto& dog : dogs)
      if (!dog.InitTexture(width, int(plan.staging_rows)))
        throw std::runtime_error("SIFT streamed KEY source allocation");
    if (!key.InitTexture(width, int(plan.staging_rows), 4))
      throw std::runtime_error("SIFT streamed KEY output allocation");
    stats.gpu_staging_bytes = size_t(dogs[0].GetDataSize()) +
                              size_t(dogs[1].GetDataSize()) +
                              size_t(dogs[2].GetDataSize()) +
                              size_t(key.GetDataSize()) + 33 * sizeof(float);
    for (int first = 0; first < height;) {
      const int count = int(std::min(plan.rows, size_t(height - first)));
      const int begin = std::max(0, first - 1);
      const int end =
          int(std::min(int64_t(height), int64_t(first) + count + 1));
      const int loaded = end - begin;
      upload(dogs[0], previous, begin, loaded);
      upload(dogs[1], current, begin, loaded);
      upload(dogs[2], next, begin, loaded);
      key.SetImageSize(width, loaded);
      timed(stats.kernel_seconds, [&] {
        // The resident scratch border is undefined. Initialize only inert
        // backing storage; InitHist excludes every such byte. Do NOT compare
        // these bytes to resident Gaussian scratch or interpret them as keys.
        if (cudaMemset(key._cuData, 0, size_t(loaded) * width * 16) !=
            cudaSuccess)
          throw std::runtime_error("SIFT streamed KEY inert storage");
        // All retained interior pixels have exactly +/-1 row of original
        // data. ComputeKEY has no floating coordinate operations: dx/dy/ds
        // are unchanged, and absolute row identity is the backing file row.
        // Bind explicitly here rather than the legacy wrapper, whose error
        // logging can clear a failed texture-binding status.
        auto p =
            dogs[0].BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
        auto c =
            dogs[1].BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
        auto n =
            dogs[2].BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
        if (!p.handle || !c.handle || !n.handle)
          throw std::runtime_error("SIFT streamed KEY texture binding");
        const float threshold0 =
            (GlobalUtil::_SubpixelLocalization ? 0.8f : 1.0f) * Tdog;
        const float edge = (Tedge + 1) * (Tedge + 1) / Tedge;
#ifdef KEY_OFFSET_ONE
        dim3 grid((width - 1 + KEY_BLOCK_DIMX - 1) / KEY_BLOCK_DIMX,
                  (loaded - 1 + KEY_BLOCK_DIMY - 1) / KEY_BLOCK_DIMY);
#else
        dim3 grid((width + KEY_BLOCK_DIMX - 1) / KEY_BLOCK_DIMX,
                  (loaded + KEY_BLOCK_DIMY - 1) / KEY_BLOCK_DIMY);
#endif
        dim3 block(KEY_BLOCK_DIMX, KEY_BLOCK_DIMY);
        ComputeKEY_Kernel<<<grid, block>>>(p.handle,
                                           c.handle,
                                           n.handle,
                                           static_cast<float4*>(key._cuData),
                                           width,
                                           width - 1,
                                           loaded - 1,
                                           threshold0,
                                           Tdog,
                                           edge,
                                           GlobalUtil::_SubpixelLocalization);
        finish();
      });
      download(key, *result.key, first, count, first - begin, loaded);
      first += count;
    }
  }  // Release every KEY GPU buffer before the histogram pair is allocated.
  result.key->Seal();
  int source_width = width;
  StreamedPlane* source = result.key.get();
  for (int destination_width = exact_domain ? HistogramWidth(width)
                                            : (width + 2) >> 2;
       ;) {
    auto output = std::make_unique<StreamedPlane>(
        options.scratch_directory, destination_width, height, 4);
    {
      const size_t rows =
          std::min({size_t(height),
                    size_t(65535),
                    (plan.gpu_bytes - 132) /
                        (16 * (size_t(source_width) + destination_width)),
                    plan.host_bytes / (16 * size_t(source_width))});
      CuTexImage input, hist;
      struct Drain {
        ~Drain() { cudaDeviceSynchronize(); }
      } drain;
      if (!input.InitTexture(source_width, int(rows), 4) ||
          !hist.InitTexture(destination_width, int(rows), 4))
        throw std::runtime_error("SIFT streamed histogram allocation");
      stats.gpu_staging_bytes =
          std::max(stats.gpu_staging_bytes,
                   size_t(input.GetDataSize()) + size_t(hist.GetDataSize()) +
                       33 * sizeof(float));
      for (int row = 0; row < height;) {
        const int count = int(std::min(rows, size_t(height - row)));
        upload(input, *source, row, count);
        hist.SetImageSize(destination_width, count);
        timed(stats.kernel_seconds, [&] {
          const bool initial = source == result.key.get();
          auto texture =
              input.BindTexture(texDataDesc,
                                initial ? cudaCreateChannelDesc<float4>()
                                        : cudaCreateChannelDesc<int4>());
          if (!texture.handle)
            throw std::runtime_error("SIFT streamed histogram texture binding");
          if (initial) {
            dim3 grid(
                (destination_width + HIST_INIT_WIDTH - 1) / HIST_INIT_WIDTH,
                count);
            dim3 block(HIST_INIT_WIDTH, 1);
            InitHist_Kernel<<<grid, block>>>(texture.handle,
                                             static_cast<int4*>(hist._cuData),
                                             source_width,
                                             destination_width,
                                             count,
                                             row,
                                             height);
          } else {
            // Reduction groups four adjacent int4s within the same absolute
            // row. Launch geometry does not change grouping or addition order.
            dim3 grid(
                (destination_width + HIST_INIT_WIDTH - 1) / HIST_INIT_WIDTH,
                count);
            dim3 block(HIST_INIT_WIDTH, 1);
            ReduceHist_Kernel<<<grid, block>>>(texture.handle,
                                               static_cast<int4*>(hist._cuData),
                                               source_width,
                                               destination_width,
                                               count);
          }
          finish();
        });
        download(hist, *output, row, count, 0, count);
        row += count;
      }
    }
    output->Seal();
    source = output.get();
    source_width = destination_width;
    result.histograms.push_back(std::move(output));
    if (destination_width == 1) break;  // Same TraverseHistogram stopping rule.
    destination_width = (destination_width + 3) >> 2;
  }
  stats.scratch_bytes = stats.file_written_bytes;
  CheckStreamCancellation(options.cancelled);
  stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

void __global__ ListGen_Kernel(cudaTextureObject_t texDataList,
                               cudaTextureObject_t texDataI4,
                               int4* d_list,
                               int list_len,
                               int width,
                               bool packed = false) {
  int idx1 = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
  if (idx1 >= list_len) return;
  int4 pos = tex1Dfetch<int4>(texDataList, idx1);
  int idx2 = packed ? idx1 : IMUL(pos.y, width) + pos.x;
  int4 temp = tex1Dfetch<int4>(texDataI4, idx2);
  int sum1 = temp.x + temp.y;
  int sum2 = sum1 + temp.z;
  pos.x <<= 2;
  if (pos.z >= sum2) {
    pos.x += 3;
    pos.z -= sum2;
  } else if (pos.z >= sum1) {
    pos.x += 2;
    pos.z -= sum1;
  } else if (pos.z >= temp.x) {
    pos.x += 1;
    pos.z -= temp.x;
  }
  if (idx1 < list_len) {
    d_list[idx1] = pos;
  }
}

//input list (x, y) (x, y) ....
void ProgramCU::GenerateList(CuTexImage* list, CuTexImage* hist)
{
	int len = list->GetImgWidth();
    CuTexImage::CuTexObj listTex = list->BindTexture(texDataDesc, cudaCreateChannelDesc<int4>());
    CuTexImage::CuTexObj histTex = hist->BindTexture(texDataDesc, cudaCreateChannelDesc<int4>());
	dim3  grid((len + LISTGEN_BLOCK_DIM -1) /LISTGEN_BLOCK_DIM);
	dim3  block(LISTGEN_BLOCK_DIM);
	ListGen_Kernel<<<grid, block>>>(listTex.handle, histTex.handle, (int4*) list->_cuData, len,
                                  hist->GetImgWidth());
}

siftgpu::StreamListResult ProgramCU::GenerateListStreamed(
    siftgpu::StreamKeyResult& detection,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  const auto start = std::chrono::steady_clock::now();
  CheckStreamCancellation(options.cancelled);
  ListPlan plan(options.gpu_bytes, options.host_bytes);
  if (!detection.key || !detection.key->Complete() ||
      detection.key->Channels() != 4 || detection.histograms.empty())
    throw std::invalid_argument("SIFT streamed list requires KEY/histograms");
  const int height = detection.key->Height();
  int width = detection.key->Width();
  if (height >= (1 << 23) || width >= (1 << 23) ||
      int64_t(width) * height > INT32_MAX)
    throw std::invalid_argument("SIFT streamed list integer domain");
  for (const auto& plane : detection.histograms) {
    // Exact ceil and legacy rounded first grouping coincide on supported
    // rounded widths; detection has already established border semantics.
    width = (width + 3) / 4;
    if (!plane || !plane->Complete() || plane->Channels() != 4 ||
        plane->Height() != height || plane->Width() != width)
      throw std::invalid_argument("SIFT streamed list histogram shape");
  }
  if (width != 1)
    throw std::invalid_argument("SIFT streamed list missing histogram root");
  StreamListResult result;
  auto& stats = result.stats;
  // These are int4 bit patterns throughout, including host float storage.
  // Never convert histogram values to float or change traversal arithmetic.
  std::vector<float> positions(plan.records * 4), packed(plan.records * 4);
  float cell[4];
  auto read = [&](StreamedPlane& plane, int x, int y, float* target) {
    const auto time = std::chrono::steady_clock::now();
    plane.ReadRecord(x, y, target, options.cancelled);
    stats.file_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    stats.file_read_bytes += 16;
  };
  auto& root = *detection.histograms.back();
  int64_t total = 0;
  for (int row = 0; row < height; ++row) {
    read(root, 0, row, cell);
    int counts[4];
    std::memcpy(counts, cell, 16);
    // Resident GenerateFeatureList clamps negative root counts to zero.
    for (int count : counts) total += std::max(0, count);
    if (total > INT32_MAX)
      throw std::overflow_error("SIFT streamed list count overflow");
  }
  // Resident unconditionally descends once even with a single root. Its
  // nonzero tiny-domain result is not qualified; do not silently fix it.
  if (total && detection.histograms.size() == 1)
    throw std::runtime_error("SIFT nonzero single-level list is unqualified");
  result.count = int(total);
  stats.host_staging_bytes = plan.host_bytes;
  if (total) {
    result.records = std::make_unique<StreamedPlane>(
        options.scratch_directory, 1, result.count, 4);
    result.records->Reserve();
    CheckStreamCancellation(options.cancelled);
    CuTexImage list, histogram;
    struct Drain {
      ~Drain() { cudaDeviceSynchronize(); }
    } drain;
    if (!list.InitTexture(int(plan.records), 1, 4) ||
        !histogram.InitTexture(int(plan.records), 1, 4))
      throw std::runtime_error("SIFT streamed list GPU allocation");
    stats.gpu_staging_bytes =
        size_t(list.GetDataSize()) + size_t(histogram.GetDataSize()) + 132;
    auto finish = [&] {
      if (cudaGetLastError() != cudaSuccess ||
          cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("SIFT streamed list CUDA failure");
    };
    auto upload = [&](CuTexImage& texture, const float* data, int n) {
      const auto time = std::chrono::steady_clock::now();
      texture.SetImageSize(n, 1);
      if (cudaMemcpy(
              texture._cuData, data, size_t(n) * 16, cudaMemcpyHostToDevice) !=
          cudaSuccess)
        throw std::runtime_error("SIFT streamed list upload");
      stats.h2d_bytes += size_t(n) * 16;
      stats.h2d_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
              .count();
    };
    auto download = [&](int n) {
      const auto time = std::chrono::steady_clock::now();
      if (cudaMemcpy(positions.data(),
                     list._cuData,
                     size_t(n) * 16,
                     cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("SIFT streamed list download");
      stats.d2h_bytes += size_t(n) * 16;
      stats.d2h_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
              .count();
    };
    // Seed cursor is row, channel, rank: precisely the resident ii/jj loop.
    int row = -1, channel = 3, rank = 0, count = 0;
    for (int first = 0; first < result.count;) {
      CheckStreamCancellation(options.cancelled);
      const int n = int(std::min(plan.records, size_t(result.count - first)));
      for (int i = 0; i < n; ++i) {
        while (rank == count) {
          ++channel;
          if (channel >= 4) {
            ++row;
            read(root, 0, row, cell);
            channel = 0;
          }
          int counts[4];
          std::memcpy(counts, cell, 16);
          count = std::max(0, counts[channel]);
          rank = 0;
        }
        const int seed[4] = {channel, row, rank++, 0};
        std::memcpy(positions.data() + 4 * i, seed, 16);
      }
      upload(list, positions.data(), n);
      for (int hi = int(detection.histograms.size()) - 2; hi >= 0; --hi) {
        auto& plane = *detection.histograms[hi];
        for (int i = 0; i < n; ++i) {
          int position[4];
          std::memcpy(position, positions.data() + 4 * i, 16);
          read(plane, position[0], position[1], packed.data() + 4 * i);
          int values[4];
          std::memcpy(values, packed.data() + 4 * i, 16);
          int64_t sum = 0;
          for (int value : values) {
            if (value < 0)
              throw std::runtime_error("SIFT streamed negative histogram");
            sum += value;
          }
          if (sum > INT32_MAX || position[2] < 0 || position[2] >= sum)
            throw std::runtime_error("SIFT streamed invalid histogram rank");
        }
        upload(histogram, packed.data(), n);
        auto list_tex =
            list.BindTexture(texDataDesc, cudaCreateChannelDesc<int4>());
        auto hist_tex =
            histogram.BindTexture(texDataDesc, cudaCreateChannelDesc<int4>());
        if (!list_tex.handle || !hist_tex.handle)
          throw std::runtime_error("SIFT streamed list texture binding");
        const auto time = std::chrono::steady_clock::now();
        ListGen_Kernel<<<(n + LISTGEN_BLOCK_DIM - 1) / LISTGEN_BLOCK_DIM,
                         LISTGEN_BLOCK_DIM>>>(list_tex.handle,
                                              hist_tex.handle,
                                              static_cast<int4*>(list._cuData),
                                              n,
                                              plane.Width(),
                                              true);
        finish();
        stats.kernel_seconds += std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - time)
                                    .count();
        download(n);
      }
      const auto time = std::chrono::steady_clock::now();
      result.records->WriteRows(first, n, positions.data(), options.cancelled);
      stats.file_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
              .count();
      stats.file_written_bytes += size_t(n) * 16;
      first += n;
    }
    result.records->Seal();
    stats.scratch_bytes = result.records->Bytes();
  }
  CheckStreamCancellation(options.cancelled);
  stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}
void __global__ ComputeOrientation_Kernel(cudaTextureObject_t texDataF2,
                                          cudaTextureObject_t texDataF4,
                                          cudaTextureObject_t texDataList,
                                          float4* d_list,
										  int list_len,
										  int width, int height,
										  float sigma, float sigma_step,
										  float gaussian_factor, float sample_factor,
										  int num_orientation,
										  int existing_keypoint,
										  int subpixel,
										  int keepsign,
                                          int gradient_first_row = 0,
                                          bool gathered_key = false,
                                          int4* window_bounds = nullptr)
{
	const float ten_degree_per_radius = 5.7295779513082320876798154814105;
	const float radius_per_ten_degrees = 1.0 / 5.7295779513082320876798154814105;
	int idx = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
	if(idx >= list_len) return;
	float4 key;
	if(existing_keypoint)
	{
		key = tex1Dfetch<float4>(texDataF4, idx);
	}else
	{
		int4 ikey = tex1Dfetch<int4>(texDataList, idx);
		key.x = ikey.x + 0.5f;
		key.y = ikey.y + 0.5f;
		key.z = sigma;
		if(subpixel || keepsign)
		{
			float4 offset = tex1Dfetch<float4>(texDataF4, gathered_key ? idx : IMUL(width, ikey.y) + ikey.x);
			if(subpixel)
			{
				key.x += offset.y;
				key.y += offset.z;
				key.z *= pow(sigma_step, offset.w);
			}
			if(keepsign) key.z *= offset.x;
		}
	}
  // Read-only sampling metadata pass: the SAME key arithmetic (including
  // device pow and subpixel/sign) defines the original float loop bounds.
  // No floating coordinates are translated to a local origin.
  if (window_bounds) {
    float win = fabs(key.z) * sample_factor;
    float ymin = max(1.5f, floorf(key.y - win) + 0.5f);
    float ymax = min(height - 1.5f, floorf(key.y + win) + 0.5f);
    if (!isfinite(key.x) || !isfinite(key.y) || !isfinite(key.z) ||
        !isfinite(win)) {
      window_bounds[idx] = make_int4(-1, -1, 0, 0);
    } else if (!num_orientation || ymin > ymax) {
      window_bounds[idx] = make_int4(0, 0, 0, 0);
    } else {
      window_bounds[idx] = make_int4(int(ymin), int(ymax), 0, 0);
    }
    return;
  }
	if(num_orientation == 0)
	{
		key.w = 0;
		d_list[idx] = key;
		return;
	}
	float vote[37];
	float gsigma = key.z * gaussian_factor;
	float win = fabs(key.z) * sample_factor;
	float dist_threshold = win * win + 0.5;
	float factor = -0.5f / (gsigma * gsigma);
	float xmin = max(1.5f, floorf(key.x - win) + 0.5f);
	float ymin = max(1.5f, floorf(key.y - win) + 0.5f);
	float xmax = min(width - 1.5f, floorf(key.x + win) + 0.5f);
	float ymax = min(height -1.5f, floorf(key.y + win) + 0.5f);
#pragma unroll
	for(int i = 0; i < 36; ++i) vote[i] = 0.0f;
	for(float y = ymin; y <= ymax; y += 1.0f)
	{
		for(float x = xmin; x <= xmax; x += 1.0f)
		{
			float dx = x - key.x;
			float dy = y - key.y;
			float sq_dist  = dx * dx + dy * dy;
			if(sq_dist >= dist_threshold) continue;
			float2 got = tex1Dfetch<float2>(texDataF2, IMUL(int(y), width) + int(x) - gradient_first_row * width);
			float weight = got.x * exp(sq_dist * factor);
			float fidx = floorf(got.y * ten_degree_per_radius);
			int oidx = fidx;
			if(oidx < 0) oidx += 36;
			vote[oidx] += weight;
		}
	}

	//filter the vote

	const float one_third = 1.0 /3.0;
#pragma unroll
	for(int i = 0; i < 6; ++i)
	{
		vote[36] = vote[0];
		float pre = vote[35];
#pragma unroll
		for(int j = 0; j < 36; ++j)
		{
			float temp = one_third * (pre + vote[j] + vote[j + 1]);
			pre = vote[j];			vote[j] = temp;
		}
	}

	vote[36] = vote[0];
	if(num_orientation == 1 || existing_keypoint)
	{
		int index_max = 0;
		float max_vote = vote[0];
#pragma unroll
		for(int i = 1; i < 36; ++i)
		{
			index_max =  vote[i] > max_vote? i : index_max;
			max_vote = max(max_vote, vote[i]);
		}
		float pre = vote[index_max == 0? 35 : index_max -1];
		float next = vote[index_max + 1];
		float weight = max_vote;
		float off =  0.5f * FDIV(next - pre, weight + weight - next - pre);
		key.w = radius_per_ten_degrees * (index_max + 0.5f + off);
		d_list[idx] = key;

	}else
	{
		float max_vote = vote[0];
#pragma unroll
		for(int i = 1; i < 36; ++i)		max_vote = max(max_vote, vote[i]);

		float vote_threshold = max_vote * 0.8f;
		float pre = vote[35];
		float max_rot[2], max_vot[2] = {0, 0};
		int  ocount = 0;
#pragma unroll
		for(int i =0; i < 36; ++i)
		{
			float next = vote[i + 1];
			if(vote[i] > vote_threshold && vote[i] > pre && vote[i] > next)
			{
				float di = 0.5f * FDIV(next - pre, vote[i] + vote[i] - next - pre);
				float rot = i + di + 0.5f;
				float weight = vote[i];
				///
				if(weight > max_vot[1])
				{
					if(weight > max_vot[0])
					{
						max_vot[1] = max_vot[0];
						max_rot[1] = max_rot[0];
						max_vot[0] = weight;
						max_rot[0] = rot;
					}
					else
					{
						max_vot[1] = weight;
						max_rot[1] = rot;
					}
					ocount ++;
				}
			}
			pre = vote[i];
		}
		float fr1 = max_rot[0] / 36.0f;
		if(fr1 < 0) fr1 += 1.0f;
		unsigned short us1 = ocount == 0? 65535 : ((unsigned short )floorf(fr1 * 65535.0f));
		unsigned short us2 = 65535;
		if(ocount > 1)
		{
			float fr2 = max_rot[1] / 36.0f;
			if(fr2 < 0) fr2 += 1.0f;
			us2 = (unsigned short ) floorf(fr2 * 65535.0f);
		}
		unsigned int uspack = (us2 << 16) | us1;
		key.w = __int_as_float(uspack);
		d_list[idx] = key;
	}

}




void ProgramCU::ComputeOrientation(CuTexImage* list, CuTexImage* got, CuTexImage*key,
								   float sigma, float sigma_step, int existing_keypoint)
{
	int len = list->GetImgWidth();
	if(len <= 0) return;
	int width = got->GetImgWidth(), height = got->GetImgHeight();
    CuTexImage::CuTexObj texObjF4;
    CuTexImage::CuTexObj texObjList;
	if(existing_keypoint)
	{
        texObjF4 = list->BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
	}else
	{
        texObjList = list->BindTexture(texDataDesc, cudaCreateChannelDesc<int4>());
		if(GlobalUtil::_SubpixelLocalization)
        {
            texObjF4 = key->BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
        }
	}

	// The gradient/orientation image is bound linearly and indexed explicitly
	// by the kernels rather than bound as a 2D pitched texture. The binding is
	// point-filtered and every consumer already clamps its coordinates to the
	// image, so the sampled values are the same either way, while a pitched
	// binding additionally requires the row pitch to be a multiple of the
	// device's texture alignment (256 bytes on AMD GPUs), which a tightly
	// packed pyramid level does not satisfy.
	CuTexImage::CuTexObj gotTex = got->BindTexture(texDataDesc, cudaCreateChannelDesc<float2>());

	const int block_width = len < ORIENTATION_COMPUTE_PER_BLOCK ? 16 : ORIENTATION_COMPUTE_PER_BLOCK;
	dim3 grid((len + block_width -1) / block_width);
	dim3 block(block_width);

	ComputeOrientation_Kernel<<<grid, block>>>(
        gotTex.handle,
        texObjF4.handle,
        texObjList.handle,
        (float4*) list->_cuData,
		len, width, height, sigma, sigma_step,
		GlobalUtil::_OrientationGaussianFactor,
		GlobalUtil::_OrientationGaussianFactor * GlobalUtil::_OrientationWindowFactor,
		GlobalUtil::_FixedOrientation? 0 : GlobalUtil::_MaxOrientation,
		existing_keypoint, GlobalUtil::_SubpixelLocalization, GlobalUtil::_KeepExtremumSign);

	ProgramCU::CheckErrorCUDA("ComputeOrientation");
}

siftgpu::StreamListResult ProgramCU::ComputeOrientationStreamed(
    siftgpu::StreamListResult& list,
    siftgpu::StreamedPlane& gradient,
    siftgpu::StreamedPlane& key,
    float sigma,
    float sigma_step,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  const auto start = std::chrono::steady_clock::now();
  CheckStreamCancellation(options.cancelled);
  const int width = gradient.Width(), height = gradient.Height();
  OrientationPlan minimum(width, 1, options.gpu_bytes, options.host_bytes);
  if (!gradient.Complete() || gradient.Channels() != 2 || !key.Complete() ||
      key.Channels() != 4 || key.Width() != width || key.Height() != height ||
      width >= (1 << 23) || height >= (1 << 23) ||
      int64_t(width) * height > INT32_MAX || list.count < 0 ||
      !std::isfinite(sigma) || sigma <= 0 || !std::isfinite(sigma_step) ||
      sigma_step <= 0 ||
      (GlobalUtil::_KeepExtremumSign && !GlobalUtil::_SubpixelLocalization))
    throw std::invalid_argument("SIFT streamed orientation domain");
  if (list.count &&
      (!list.records || !list.records->Complete() ||
       list.records->Width() != 1 || list.records->Channels() != 4 ||
       list.records->Height() < list.count))
    throw std::invalid_argument("SIFT streamed orientation list");
  const bool expand =
      GlobalUtil::_MaxOrientation > 1 && !GlobalUtil::_FixedOrientation;
  const int orientations =
      GlobalUtil::_FixedOrientation ? 0 : GlobalUtil::_MaxOrientation;
  if (int64_t(list.count) * (expand ? 2 : 1) > INT32_MAX)
    throw std::overflow_error("SIFT streamed orientation expansion");
  StreamListResult result;
  auto& stats = result.stats;
  if (!list.count) return result;
  result.records = std::make_unique<StreamedPlane>(
      options.scratch_directory, 1, list.count * (expand ? 2 : 1), 4);
  result.records->Reserve();
  stats.scratch_bytes = result.records->Bytes();
  CuTexImage device_list, device_key, device_bounds;
  struct Drain {
    ~Drain() { cudaDeviceSynchronize(); }
  } drain;
  if (!device_list.InitTexture(1, 1, 4) || !device_key.InitTexture(1, 1, 4) ||
      !device_bounds.InitTexture(1, 1, 4))
    throw std::runtime_error("SIFT streamed orientation GPU records");
  auto finish = [&] {
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      throw std::runtime_error("SIFT streamed orientation CUDA failure");
  };
  auto transfer = [&](void* device, void* host, size_t bytes, bool upload) {
    const auto time = std::chrono::steady_clock::now();
    if (cudaMemcpy(upload ? device : host,
                   upload ? host : device,
                   bytes,
                   upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost) !=
        cudaSuccess)
      throw std::runtime_error("SIFT streamed orientation transfer");
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    if (upload) {
      stats.h2d_bytes += bytes;
      stats.h2d_seconds += seconds;
    } else {
      stats.d2h_bytes += bytes;
      stats.d2h_seconds += seconds;
    }
  };
  for (int first = 0; first < list.count; ++first) {
    CheckStreamCancellation(options.cancelled);
    // Exactly 96 bytes of host records, including maximum reshape capacity.
    int4 position, bounds;
    float offset[4], packed[4], expanded[8];
    auto time = std::chrono::steady_clock::now();
    list.records->ReadRows(
        first, 1, reinterpret_cast<float*>(&position), options.cancelled);
    stats.file_read_bytes += 16;
    if (position.x < 0 || position.x >= width || position.y < 0 ||
        position.y >= height)
      throw std::runtime_error("SIFT streamed orientation invalid feature");
    key.ReadRecord(position.x, position.y, offset, options.cancelled);
    stats.file_read_bytes += 16;
    stats.file_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    transfer(device_list._cuData, &position, 16, true);
    transfer(device_key._cuData, offset, 16, true);
    auto list_tex =
        device_list.BindTexture(texDataDesc, cudaCreateChannelDesc<int4>());
    auto key_tex =
        device_key.BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
    if (!list_tex.handle || !key_tex.handle)
      throw std::runtime_error("SIFT streamed orientation texture binding");
    time = std::chrono::steady_clock::now();
    ComputeOrientation_Kernel<<<1, 16>>>(
        0,
        key_tex.handle,
        list_tex.handle,
        static_cast<float4*>(device_list._cuData),
        1,
        width,
        height,
        sigma,
        sigma_step,
        GlobalUtil::_OrientationGaussianFactor,
        GlobalUtil::_OrientationGaussianFactor *
            GlobalUtil::_OrientationWindowFactor,
        orientations,
        0,
        GlobalUtil::_SubpixelLocalization,
        GlobalUtil::_KeepExtremumSign,
        0,
        true,
        static_cast<int4*>(device_bounds._cuData));
    finish();
    stats.kernel_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    transfer(device_bounds._cuData, &bounds, 16, false);
    if (bounds.x < 0 || bounds.y < bounds.x || bounds.y >= height)
      throw std::runtime_error("SIFT streamed orientation invalid window");
    const int rows = bounds.y - bounds.x + 1;
    OrientationPlan plan(width, rows, options.gpu_bytes, options.host_bytes);
    std::vector<float> staging(size_t(width) * rows * 2);
    CuTexImage device_gradient;
    // Declared after its owner, so unwind drains BEFORE releasing this window.
    Drain window_drain;
    if (!device_gradient.InitTexture(width, rows, 2))
      throw std::runtime_error("SIFT streamed orientation GPU window");
    stats.gpu_staging_bytes = std::max(stats.gpu_staging_bytes, plan.gpu_bytes);
    stats.host_staging_bytes =
        std::max(stats.host_staging_bytes, plan.host_bytes);
    time = std::chrono::steady_clock::now();
    gradient.ReadRows(bounds.x, rows, staging.data(), options.cancelled);
    stats.file_read_bytes += staging.size() * sizeof(float);
    stats.file_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    transfer(device_gradient._cuData,
             staging.data(),
             staging.size() * sizeof(float),
             true);
    auto gradient_tex = device_gradient.BindTexture(
        texDataDesc, cudaCreateChannelDesc<float2>());
    if (!gradient_tex.handle)
      throw std::runtime_error("SIFT streamed orientation gradient binding");
    time = std::chrono::steady_clock::now();
    ComputeOrientation_Kernel<<<1, 16>>>(
        gradient_tex.handle,
        key_tex.handle,
        list_tex.handle,
        static_cast<float4*>(device_list._cuData),
        1,
        width,
        height,
        sigma,
        sigma_step,
        GlobalUtil::_OrientationGaussianFactor,
        GlobalUtil::_OrientationGaussianFactor *
            GlobalUtil::_OrientationWindowFactor,
        orientations,
        0,
        GlobalUtil::_SubpixelLocalization,
        GlobalUtil::_KeepExtremumSign,
        bounds.x,
        true);
    finish();
    stats.kernel_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    transfer(device_list._cuData, packed, 16, false);
    int produced = 1;
    if (expand)
      produced = ReshapePackedOrientations(packed, 1, expanded);
    else
      std::memcpy(expanded, packed, 16);
    if (produced) {
      time = std::chrono::steady_clock::now();
      result.records->WriteRows(
          result.count, produced, expanded, options.cancelled);
      stats.file_written_bytes += size_t(produced) * 16;
      result.count += produced;
      stats.file_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
              .count();
    }
  }
  CheckStreamCancellation(options.cancelled);
  if (result.count)
    result.records->SealPrefix(result.count);
  else
    result.records.reset();
  stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

template <bool DYNAMIC_INDEXING, bool STREAMED = false> void __global__ ComputeDescriptor_Kernel(cudaTextureObject_t texDataF2, cudaTextureObject_t texDataF4, float4* d_des, int num,
											 int width, int height, float window_factor, int gradient_origin = 0, int2* bounds = nullptr)
{
	const float rpi = 4.0/ 3.14159265358979323846;
	int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	int fidx = idx >> 4;
	if(fidx >= num) return;
	float4 key = tex1Dfetch<float4>(texDataF4, fidx);
	int bidx = idx& 0xf, ix = bidx & 0x3, iy = bidx >> 2;
	float spt = fabs(key.z * window_factor);
	float s, c; __sincosf(key.w, &s, &c);
	float anglef = key.w > 3.14159265358979323846? key.w - (2.0 * 3.14159265358979323846) : key.w ;
	float cspt = c * spt, sspt = s * spt;
	float crspt = c / spt, srspt = s / spt;
	float2 offsetpt, pt;
	float xmin, ymin, xmax, ymax, bsz;
	offsetpt.x = ix - 1.5f;
	offsetpt.y = iy - 1.5f;
	pt.x = cspt * offsetpt.x - sspt * offsetpt.y + key.x;
	pt.y = cspt * offsetpt.y + sspt * offsetpt.x + key.y;
	bsz =  fabs(cspt) + fabs(sspt);
	xmin = max(1.5f, floorf(pt.x - bsz) + 0.5f);
	ymin = max(1.5f, floorf(pt.y - bsz) + 0.5f);
	xmax = min(width - 1.5f, floorf(pt.x + bsz) + 0.5f);
	ymax = min(height - 1.5f, floorf(pt.y + bsz) + 0.5f);
	// Bounds are produced by the original per-bin floating-point expressions
	// above. Empty bins fetch nothing; include every possible loop row even
	// if the later rotated-bin predicate rejects it. No float translation.
	if(STREAMED && bounds) {
		bounds[bidx] = ymin <= ymax && xmin <= xmax
			? make_int2(int(ymin), int(ymax)) : make_int2(0, -1);
		return;
	}
	float des[9];
#pragma unroll
	for(int i =0; i < 9; ++i) des[i] = 0.0f;
	for(float y = ymin; y <= ymax; y += 1.0f)
	{
		for(float x = xmin; x <= xmax; x += 1.0f)
		{
			float dx = x - pt.x;
			float dy = y - pt.y;
			float nx = crspt * dx + srspt * dy;
			float ny = crspt * dy - srspt * dx;
			float nxn = fabs(nx);
			float nyn = fabs(ny);
			if(nxn < 1.0f && nyn < 1.0f)
			{
				float2 cc = tex1Dfetch<float2>(texDataF2, IMUL(STREAMED ? int(y) - gradient_origin : int(y), width) + int(x));
				float dnx = nx + offsetpt.x;
				float dny = ny + offsetpt.y;
				float ww = exp(-0.125f * (dnx * dnx + dny * dny));
				float wx = 1.0 - nxn;
				float wy = 1.0 - nyn;
				float weight = ww * wx * wy * cc.x;
				float theta = (anglef - cc.y) * rpi;
				if(theta < 0) theta += 8.0f;
				float fo = floorf(theta);
				int fidx = fo;
				float weight1 = fo + 1.0f  - theta;
				float weight2 = theta - fo;
				if(DYNAMIC_INDEXING)
				{
					des[fidx] += (weight1 * weight);
					des[fidx + 1] += (weight2 * weight);
					//this dynamic indexing part might be slow
				}else
				{
					#pragma unroll
					for(int k = 0; k < 8; ++k)
					{
						if(k == fidx)
						{
							des[k] += (weight1 * weight);
							des[k+1] += (weight2 * weight);
						}
					}
				}
			}
		}
	}
	des[0] += des[8];

	int didx = idx << 1;
	d_des[didx] = make_float4(des[0], des[1], des[2], des[3]);
	d_des[didx+1] = make_float4(des[4], des[5], des[6], des[7]);
}


template <bool DYNAMIC_INDEXING> void __global__ ComputeDescriptorRECT_Kernel(cudaTextureObject_t texDataF2, cudaTextureObject_t texDataF4, float4* d_des, int num,
											 int width, int height, float window_factor)
{
	const float rpi = 4.0/ 3.14159265358979323846;
	int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	int fidx = idx >> 4;
	if(fidx >= num) return;
	float4 key = tex1Dfetch<float4>(texDataF4, fidx);
	int bidx = idx& 0xf, ix = bidx & 0x3, iy = bidx >> 2;
    //float aspect_ratio = key.w / key.z;
    //float aspect_sq = aspect_ratio * aspect_ratio;
	float sptx = key.z * 0.25, spty = key.w * 0.25;
	float xmin, ymin, xmax, ymax; float2 pt;
	pt.x = sptx * (ix + 0.5f)  + key.x;
	pt.y = spty * (iy + 0.5f)  + key.y;
	xmin = max(1.5f, floorf(pt.x - sptx) + 0.5f);
	ymin = max(1.5f, floorf(pt.y - spty) + 0.5f);
	xmax = min(width - 1.5f, floorf(pt.x + sptx) + 0.5f);
	ymax = min(height - 1.5f, floorf(pt.y + spty) + 0.5f);
	float des[9];
#pragma unroll
	for(int i =0; i < 9; ++i) des[i] = 0.0f;
	for(float y = ymin; y <= ymax; y += 1.0f)
	{
		for(float x = xmin; x <= xmax; x += 1.0f)
		{
			float nx = (x - pt.x) / sptx;
			float ny = (y - pt.y) / spty;
			float nxn = fabs(nx);
			float nyn = fabs(ny);
			if(nxn < 1.0f && nyn < 1.0f)
			{
				float2 cc = tex1Dfetch<float2>(texDataF2, IMUL(int(y), width) + int(x));
				float wx = 1.0 - nxn;
				float wy = 1.0 - nyn;
				float weight =  wx * wy * cc.x;
				float theta = (- cc.y) * rpi;
				if(theta < 0) theta += 8.0f;
				float fo = floorf(theta);
				int fidx = fo;
				float weight1 = fo + 1.0f  - theta;
				float weight2 = theta - fo;
				if(DYNAMIC_INDEXING)
				{
					des[fidx] += (weight1 * weight);
					des[fidx + 1] += (weight2 * weight);
					//this dynamic indexing part might be slow
				}else
				{
					#pragma unroll
					for(int k = 0; k < 8; ++k)
					{
						if(k == fidx)
						{
							des[k] += (weight1 * weight);
							des[k+1] += (weight2 * weight);
						}
					}
				}
			}
		}
	}
	des[0] += des[8];

	int didx = idx << 1;
	d_des[didx] = make_float4(des[0], des[1], des[2], des[3]);
	d_des[didx+1] = make_float4(des[4], des[5], des[6], des[7]);
}

void __global__ NormalizeDescriptor_Kernel(cudaTextureObject_t texDataF4, float4* d_des, int num)
{
	float4 temp[32];
	int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if(idx >= num) return;
	int sidx = idx << 5;
	float norm1 = 0, norm2 = 0;
#pragma unroll
	for(int i = 0; i < 32; ++i)
	{
		temp[i] = tex1Dfetch<float4>(texDataF4, sidx +i);
		norm1 += (temp[i].x * temp[i].x + temp[i].y * temp[i].y +
				 temp[i].z * temp[i].z + temp[i].w * temp[i].w);
	}
	norm1 = rsqrt(norm1);

#pragma unroll
	for(int i = 0; i < 32; ++i)
	{
		temp[i].x = min(0.2f, temp[i].x * norm1);
		temp[i].y = min(0.2f, temp[i].y * norm1);
		temp[i].z = min(0.2f, temp[i].z * norm1);
		temp[i].w = min(0.2f, temp[i].w * norm1);
		norm2 += (temp[i].x * temp[i].x + temp[i].y * temp[i].y +
				 temp[i].z * temp[i].z + temp[i].w * temp[i].w);
	}

	norm2 = rsqrt(norm2);
#pragma unroll
	for(int i = 0; i < 32; ++i)
	{
		temp[i].x *= norm2;		temp[i].y *= norm2;
		temp[i].z *= norm2;		temp[i].w *= norm2;
		d_des[sidx + i] = temp[i];
	}
}

void ProgramCU::ComputeDescriptor(CuTexImage*list, CuTexImage* got, CuTexImage* dtex, int rect, int stream)
{
	int num = list->GetImgWidth();
	int width = got->GetImgWidth();
	int height = got->GetImgHeight();

    dtex->InitTexture(num * 128, 1, 1);
    CuTexImage::CuTexObj gotTex = got->BindTexture(texDataDesc, cudaCreateChannelDesc<float2>());
    CuTexImage::CuTexObj listTex = list->BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
	int block_width = DESCRIPTOR_COMPUTE_BLOCK_SIZE;
	dim3 grid((num * 16 + block_width -1) / block_width);
	dim3 block(block_width);

    if(rect)
    {
	    if(GlobalUtil::_UseDynamicIndexing)
	    	ComputeDescriptorRECT_Kernel<true><<<grid, block>>>(gotTex.handle, listTex.handle, (float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
	    else
	    	ComputeDescriptorRECT_Kernel<false><<<grid, block>>>(gotTex.handle, listTex.handle, (float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);

    }else
    {
	    if(GlobalUtil::_UseDynamicIndexing)
	    	ComputeDescriptor_Kernel<true><<<grid, block>>>(gotTex.handle, listTex.handle, (float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
	    else
	    	ComputeDescriptor_Kernel<false><<<grid, block>>>(gotTex.handle, listTex.handle, (float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
    }
	if(GlobalUtil::_NormalizedSIFT)
	{
        CuTexImage::CuTexObj dtexTex = dtex->BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
		const int block_width = DESCRIPTOR_NORMALIZ_PER_BLOCK;
		dim3 grid((num + block_width -1) / block_width);
		dim3 block(block_width);
		NormalizeDescriptor_Kernel<<<grid, block>>>(dtexTex.handle, (float4*) dtex->_cuData, num);
	}
	CheckErrorCUDA("ComputeDescriptor");
}

siftgpu::StreamDescriptorResult ProgramCU::ComputeDescriptorStreamed(
    siftgpu::StreamListResult& list,
    siftgpu::StreamedPlane& gradient,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  const auto start = std::chrono::steady_clock::now();
  CheckStreamCancellation(options.cancelled);
  const int width = gradient.Width(), height = gradient.Height();
  DescriptorPlan minimum(width, 1, options.gpu_bytes, options.host_bytes);
  if (!gradient.Complete() || gradient.Channels() != 2 || width < 3 ||
      height < 3 || width >= (1 << 23) || height >= (1 << 23) ||
      int64_t(width) * height > INT32_MAX || list.count < 0 ||
      !std::isfinite(GlobalUtil::_DescriptorWindowFactor) ||
      GlobalUtil::_DescriptorWindowFactor <= 0 ||
      (list.count &&
       (!list.records || !list.records->Complete() ||
        list.records->Width() != 1 || list.records->Channels() != 4 ||
        list.records->Height() < list.count)))
    throw std::invalid_argument("SIFT streamed descriptor domain");
  StreamDescriptorResult result;
  auto& stats = result.stats;
  if (!list.count) return result;
  result.records = std::make_unique<StreamedPlane>(
      options.scratch_directory, 1, list.count, 128);
  result.records->Reserve();
  stats.scratch_bytes = result.records->Bytes();
  CuTexImage device_list, device_bounds, device_descriptor;
  struct Drain {
    ~Drain() { cudaDeviceSynchronize(); }
  } drain;
  if (!device_list.InitTexture(1, 1, 4) ||
      !device_bounds.InitTexture(16, 1, 2) ||
      !device_descriptor.InitTexture(128, 1))
    throw std::runtime_error("SIFT streamed descriptor GPU records");
  auto finish = [&] {
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      throw std::runtime_error("SIFT streamed descriptor CUDA failure");
  };
  auto transfer = [&](void* device, void* host, size_t bytes, bool upload) {
    const auto time = std::chrono::steady_clock::now();
    if (cudaMemcpy(upload ? device : host,
                   upload ? host : device,
                   bytes,
                   upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost) !=
        cudaSuccess)
      throw std::runtime_error("SIFT streamed descriptor transfer");
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    if (upload) {
      stats.h2d_bytes += bytes;
      stats.h2d_seconds += seconds;
    } else {
      stats.d2h_bytes += bytes;
      stats.d2h_seconds += seconds;
    }
  };
  for (int first = 0; first < list.count; ++first) {
    CheckStreamCancellation(options.cancelled);
    float key[4], descriptor[128];
    int2 bounds[16];
    auto time = std::chrono::steady_clock::now();
    list.records->ReadRows(first, 1, key, options.cancelled);
    stats.file_read_bytes += 16;
    stats.file_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    if (!std::isfinite(key[0]) || !std::isfinite(key[1]) ||
        !std::isfinite(key[2]) || !std::isfinite(key[3]) || key[2] == 0 ||
        key[0] < 0 || key[0] >= width || key[1] < 0 || key[1] >= height ||
        !std::isfinite(key[2] * GlobalUtil::_DescriptorWindowFactor))
      throw std::invalid_argument("SIFT streamed descriptor feature");
    transfer(device_list._cuData, key, 16, true);
    auto list_tex =
        device_list.BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
    if (!list_tex.handle)
      throw std::runtime_error("SIFT streamed descriptor list binding");
    time = std::chrono::steady_clock::now();
    // Same block size and lane/bin mapping as the resident kernel. Only the
    // first feature is live; inactive tail threads return before any access.
    if (GlobalUtil::_UseDynamicIndexing)
      ComputeDescriptor_Kernel<true, true>
          <<<1, DESCRIPTOR_COMPUTE_BLOCK_SIZE>>>(
              0,
              list_tex.handle,
              nullptr,
              1,
              width,
              height,
              GlobalUtil::_DescriptorWindowFactor,
              0,
              static_cast<int2*>(device_bounds._cuData));
    else
      ComputeDescriptor_Kernel<false, true>
          <<<1, DESCRIPTOR_COMPUTE_BLOCK_SIZE>>>(
              0,
              list_tex.handle,
              nullptr,
              1,
              width,
              height,
              GlobalUtil::_DescriptorWindowFactor,
              0,
              static_cast<int2*>(device_bounds._cuData));
    finish();
    stats.kernel_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    transfer(device_bounds._cuData, bounds, 128, false);
    int low = height, high = -1;
    for (const auto& bound : bounds) {
      if (bound.y < bound.x) continue;
      if (bound.x < 0 || bound.y >= height)
        throw std::runtime_error("SIFT streamed descriptor invalid bounds");
      low = std::min(low, bound.x);
      high = std::max(high, bound.y);
    }
    // An entirely empty support still needs a valid bound texture; no fetch
    // occurs. Original zero-descriptor normalization is left unchanged.
    if (high < low) low = high = 0;
    const int rows = high - low + 1;
    DescriptorPlan plan(width, rows, options.gpu_bytes, options.host_bytes);
    stats.gpu_staging_bytes = std::max(stats.gpu_staging_bytes, plan.gpu_bytes);
    stats.host_staging_bytes =
        std::max(stats.host_staging_bytes, plan.host_bytes);
    std::vector<float> staging(size_t(width) * rows * 2);
    CuTexImage device_gradient;
    Drain window_drain;
    if (!device_gradient.InitTexture(width, rows, 2))
      throw std::runtime_error("SIFT streamed descriptor GPU window");
    time = std::chrono::steady_clock::now();
    gradient.ReadRows(low, rows, staging.data(), options.cancelled);
    const size_t bytes = staging.size() * sizeof(float);
    stats.file_read_bytes += bytes;
    stats.file_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    transfer(device_gradient._cuData, staging.data(), bytes, true);
    auto gradient_tex = device_gradient.BindTexture(
        texDataDesc, cudaCreateChannelDesc<float2>());
    if (!gradient_tex.handle)
      throw std::runtime_error("SIFT streamed descriptor gradient binding");
    time = std::chrono::steady_clock::now();
    if (GlobalUtil::_UseDynamicIndexing)
      ComputeDescriptor_Kernel<true, true>
          <<<1, DESCRIPTOR_COMPUTE_BLOCK_SIZE>>>(
              gradient_tex.handle,
              list_tex.handle,
              static_cast<float4*>(device_descriptor._cuData),
              1,
              width,
              height,
              GlobalUtil::_DescriptorWindowFactor,
              low);
    else
      ComputeDescriptor_Kernel<false, true>
          <<<1, DESCRIPTOR_COMPUTE_BLOCK_SIZE>>>(
              gradient_tex.handle,
              list_tex.handle,
              static_cast<float4*>(device_descriptor._cuData),
              1,
              width,
              height,
              GlobalUtil::_DescriptorWindowFactor,
              low);
    finish();
    if (GlobalUtil::_NormalizedSIFT) {
      auto descriptor_tex = device_descriptor.BindTexture(
          texDataDesc, cudaCreateChannelDesc<float4>());
      if (!descriptor_tex.handle)
        throw std::runtime_error(
            "SIFT streamed descriptor normalization binding");
      NormalizeDescriptor_Kernel<<<1, DESCRIPTOR_NORMALIZ_PER_BLOCK>>>(
          descriptor_tex.handle,
          static_cast<float4*>(device_descriptor._cuData),
          1);
      finish();
    }
    stats.kernel_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    transfer(device_descriptor._cuData, descriptor, 512, false);
    time = std::chrono::steady_clock::now();
    result.records->WriteRows(first, 1, descriptor, options.cancelled);
    stats.file_written_bytes += 512;
    stats.file_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
  }
  CheckStreamCancellation(options.cancelled);
  result.records->Seal();
  result.count = list.count;
  stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}
//////////////////////////////////////////////////////
void ProgramCU::FinishCUDA()
{
	cudaDeviceSynchronize();
}

int ProgramCU::CheckErrorCUDA(const char* location)
{
	cudaError_t e = cudaGetLastError();
	if(e)
	{
        if(location) fprintf(stderr, "%s:\t",  location);
		fprintf(stderr, "%s\n",  cudaGetErrorString(e));
		//assert(0);
        return 1;
	}else
    {
        return 0;
    }
}

void __global__ ConvertDOG_Kernel(cudaTextureObject_t texData, float* d_result, int width, int height)
{
	int row = (blockIdx.y << BLOCK_LOG_DIM) + threadIdx.y;
	int col = (blockIdx.x << BLOCK_LOG_DIM) + threadIdx.x;
	if(col < width && row < height)
	{
		int index = row * width  + col;
		float v = tex1Dfetch<float>(texData, index);
		d_result[index] = (col == 0 || row == 0 || col == width -1 || row == height -1)?
			0.5 : __saturatef(0.5+20.0*v);
	}
}
///
void ProgramCU::DisplayConvertDOG(CuTexImage* dog, CuTexImage* out)
{
	if(out->_cuData == NULL) return;
	int width = dog->GetImgWidth(), height = dog ->GetImgHeight();
	CuTexImage::CuTexObj dogTex = dog->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	dim3 grid((width + BLOCK_DIM - 1)/ BLOCK_DIM,  (height + BLOCK_DIM - 1)/BLOCK_DIM);
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	ConvertDOG_Kernel<<<grid, block>>>(dogTex.handle, (float*) out->_cuData, width, height);
	ProgramCU::CheckErrorCUDA("DisplayConvertDOG");
}

void __global__ ConvertGRD_Kernel(cudaTextureObject_t texData, float* d_result, int width, int height)
{
	int row = (blockIdx.y << BLOCK_LOG_DIM) + threadIdx.y;
	int col = (blockIdx.x << BLOCK_LOG_DIM) + threadIdx.x;
	if(col < width && row < height)
	{
		int index = row * width  + col;
		float v = tex1Dfetch<float>(texData, index << 1);
		d_result[index] = (col == 0 || row == 0 || col == width -1 || row == height -1)?
				0 : __saturatef(5 * v);

	}
}


void ProgramCU::DisplayConvertGRD(CuTexImage* got, CuTexImage* out)
{
	if(out->_cuData == NULL) return;
	int width = got->GetImgWidth(), height = got ->GetImgHeight();
	CuTexImage::CuTexObj gotTex = got->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	dim3 grid((width + BLOCK_DIM - 1)/ BLOCK_DIM,  (height + BLOCK_DIM - 1)/BLOCK_DIM);
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	ConvertGRD_Kernel<<<grid, block>>>(gotTex.handle, (float*) out->_cuData, width, height);
	ProgramCU::CheckErrorCUDA("DisplayConvertGRD");
}

void __global__ ConvertKEY_Kernel(cudaTextureObject_t texData, cudaTextureObject_t texDataF4, float4* d_result, int width, int height)
{

	int row = (blockIdx.y << BLOCK_LOG_DIM) + threadIdx.y;
	int col = (blockIdx.x << BLOCK_LOG_DIM) + threadIdx.x;
	if(col < width && row < height)
	{
		int index = row * width + col;
		float4 keyv = tex1Dfetch<float4>(texDataF4, index);
		int is_key = (keyv.x == 1.0f || keyv.x == -1.0f);
		int inside = col > 0 && row > 0 && row < height -1 && col < width - 1;
		float v = inside? __saturatef(0.5 + 20 * tex1Dfetch<float>(texData, index)) : 0.5;
		d_result[index] = is_key && inside ?
			(keyv.x > 0? make_float4(1.0f, 0, 0, 1.0f) : make_float4(0.0f, 1.0f, 0.0f, 1.0f)):
			make_float4(v, v, v, 1.0f) ;
	}
}
void ProgramCU::DisplayConvertKEY(CuTexImage* key, CuTexImage* dog, CuTexImage* out)
{
	if(out->_cuData == NULL) return;
	int width = key->GetImgWidth(), height = key ->GetImgHeight();
	CuTexImage::CuTexObj dogTex = dog->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
    CuTexImage::CuTexObj keyTex = key->BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
	dim3 grid((width + BLOCK_DIM - 1)/ BLOCK_DIM,  (height + BLOCK_DIM - 1)/BLOCK_DIM);
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	ConvertKEY_Kernel<<<grid, block>>>(dogTex.handle, keyTex.handle, (float4*) out->_cuData, width, height);
}


void __global__ DisplayKeyPoint_Kernel(cudaTextureObject_t texDataF4, float4 * d_result, int num)
{
	int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if(idx >= num) return;
	float4 v = tex1Dfetch<float4>(texDataF4, idx);
	d_result[idx] = make_float4(v.x, v.y, 0, 1.0f);
}

void ProgramCU::DisplayKeyPoint(CuTexImage* ftex, CuTexImage* out)
{
	int num = ftex->GetImgWidth();
	int block_width = 64;
	dim3 grid((num + block_width -1) /block_width);
	dim3 block(block_width);
    CuTexImage::CuTexObj ftexTex = ftex->BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
	DisplayKeyPoint_Kernel<<<grid, block>>>(ftexTex.handle, (float4*) out->_cuData, num);
	ProgramCU::CheckErrorCUDA("DisplayKeyPoint");
}

void __global__ DisplayKeyBox_Kernel(cudaTextureObject_t texDataF4, float4* d_result, int num)
{
	int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if(idx >= num) return;
	int  kidx = idx / 10, vidx = idx - IMUL(kidx , 10);
	float4 v = tex1Dfetch<float4>(texDataF4, kidx);
	float sz = fabs(v.z * 3.0f);
	///////////////////////
	float s, c;	__sincosf(v.w, &s, &c);
	///////////////////////
	float dx = vidx == 0? 0 : ((vidx <= 4 || vidx >= 9)? sz : -sz);
	float dy = vidx <= 1? 0 : ((vidx <= 2 || vidx >= 7)? -sz : sz);
	float4 pos;
	pos.x = v.x + c * dx - s * dy;
	pos.y = v.y + c * dy + s * dx;
	pos.z = 0;	pos.w = 1.0f;
	d_result[idx]  = pos;
}

void ProgramCU::DisplayKeyBox(CuTexImage* ftex, CuTexImage* out)
{
	int len = ftex->GetImgWidth();
	int block_width = 32;
	dim3 grid((len * 10 + block_width -1) / block_width);
	dim3 block(block_width);
    CuTexImage::CuTexObj ftexTex = ftex->BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
	DisplayKeyBox_Kernel<<<grid, block>>>(ftexTex.handle, (float4*) out->_cuData, len * 10);
}

int ProgramCU::CheckCudaDevice(int device)
{
    int count = 0, device_used;
    if(cudaGetDeviceCount(&count) != cudaSuccess  || count <= 0)
    {
        ProgramCU::CheckErrorCUDA("CheckCudaDevice");
        return 0;
    }else if(count == 1)
    {
        cudaDeviceProp deviceProp;
        if ( cudaGetDeviceProperties(&deviceProp, 0) != cudaSuccess  ||
		  (deviceProp.major == 9999 && deviceProp.minor == 9999))
        {
            fprintf(stderr, "CheckCudaDevice: no device supporting CUDA.\n");
            return 0;
        }else
		{
			GlobalUtil::_MemCapGPU = deviceProp.totalGlobalMem / 1024;
			GlobalUtil::_texMaxDimGL = 32768;
			if(GlobalUtil::_verbose)
				fprintf(stdout, "NOTE: changing maximum texture dimension to %d\n", GlobalUtil::_texMaxDimGL);

		}
    }
    if(device >= 0 && device < count)
    {
        cudaSetDevice(device);
        CheckErrorCUDA("cudaSetDevice\n");
    }
    cudaGetDevice(&device_used);
    if(device != device_used)
        fprintf(stderr,  "\nERROR:   Cannot set device to %d\n"
        "\nWARNING: Use # %d device instead (out of %d)\n", device, device_used, count);
    return 1;
}

////////////////////////////////////////////////////////////////////////////////////////
// siftmatch funtions
//////////////////////////////////////////////////////////////////////////////////////////

#define MULT_TBLOCK_DIMX 128
#define MULT_TBLOCK_DIMY 1
#define MULT_BLOCK_DIMX (MULT_TBLOCK_DIMX)
#define MULT_BLOCK_DIMY (8 * MULT_TBLOCK_DIMY)

void __global__ MultiplyDescriptor_Kernel(cudaTextureObject_t texDes1, cudaTextureObject_t texDes2, int* d_result, int num1, int num2, int3* d_temp)
{
	int idx01 = (blockIdx.y  * MULT_BLOCK_DIMY),  idx02 = (blockIdx.x  * MULT_BLOCK_DIMX);

	int idx1 = idx01 + threadIdx.y, idx2 = idx02 + threadIdx.x;
	__shared__ int data1[17 * 2 * MULT_BLOCK_DIMY];
	int read_idx1 = idx01 * 8 +  threadIdx.x, read_idx2 = idx2 * 8;
	int col4 = threadIdx.x & 0x3, row4 = threadIdx.x >> 2;
	int cache_idx1 = IMUL(row4, 17) + (col4 << 2);

	///////////////////////////////////////////////////////////////
	//Load feature descriptors
	///////////////////////////////////////////////////////////////
#if MULT_BLOCK_DIMY == 16
	uint4 v = tex1Dfetch<uint4>(texDes1, read_idx1);
	data1[cache_idx1]   = v.x;	data1[cache_idx1+1] = v.y;
	data1[cache_idx1+2] = v.z;	data1[cache_idx1+3] = v.w;
#elif MULT_BLOCK_DIMY == 8
	if(threadIdx.x < 64)
	{
		uint4 v = tex1Dfetch<uint4>(texDes1, read_idx1);
		data1[cache_idx1]   = v.x;		data1[cache_idx1+1] = v.y;
		data1[cache_idx1+2] = v.z;		data1[cache_idx1+3] = v.w;
	}
#else
#error
#endif
	__syncthreads();

	///
	if(idx2 >= num2) return;
	///////////////////////////////////////////////////////////////////////////
	//compare descriptors

	int results[MULT_BLOCK_DIMY];
#pragma unroll
	for(int i = 0; i < MULT_BLOCK_DIMY; ++i) results[i] = 0;

#pragma unroll
	for(int i = 0; i < 8; ++i)
	{
		uint4 v = tex1Dfetch<uint4>(texDes2, read_idx2 + i);
		unsigned char* p2 = (unsigned char*)(&v);
#pragma unroll
		for(int k = 0; k < MULT_BLOCK_DIMY; ++k)
		{
			unsigned char* p1 = (unsigned char*) (data1 + k * 34 + i *  4 + (i/4));
			results[k] += 	 ( IMUL(p1[0], p2[0])	+ IMUL(p1[1], p2[1])
							 + IMUL(p1[2], p2[2])  	+ IMUL(p1[3], p2[3])
							 + IMUL(p1[4], p2[4])  	+ IMUL(p1[5], p2[5])
							 + IMUL(p1[6], p2[6])  	+ IMUL(p1[7], p2[7])
							 + IMUL(p1[8], p2[8])  	+ IMUL(p1[9], p2[9])
							 + IMUL(p1[10], p2[10])	+ IMUL(p1[11], p2[11])
							 + IMUL(p1[12], p2[12])	+ IMUL(p1[13], p2[13])
							 + IMUL(p1[14], p2[14])	+ IMUL(p1[15], p2[15]));
		}
	}

	int dst_idx = IMUL(idx1, num2)  + idx2;
	if(d_temp)
	{
		int3 cmp_result = make_int3(0, -1, 0);

#pragma unroll
		for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1)
			{
				cmp_result = results[i] > cmp_result.x?
				make_int3(results[i], idx1 + i, cmp_result.x) :
				make_int3(cmp_result.x, cmp_result.y, max(cmp_result.z, results[i]));
				d_result[dst_idx + IMUL(i, num2)] = results[i];
			}
		}
		d_temp[ IMUL(blockIdx.y, num2) + idx2] = cmp_result;
	}else
	{
#pragma unroll
		for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1) d_result[dst_idx + IMUL(i, num2)] = results[i];
		}
	}

}


void ProgramCU::MultiplyDescriptor(CuTexImage* des1, CuTexImage* des2, CuTexImage* texDot, CuTexImage* texCRT)
{
	int num1 = des1->GetImgWidth() / 8;
	int num2 = des2->GetImgWidth() / 8;
	dim3 grid(	(num2 + MULT_BLOCK_DIMX - 1)/ MULT_BLOCK_DIMX,
		(num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY);
	dim3 block(MULT_TBLOCK_DIMX, MULT_TBLOCK_DIMY);
	texDot->InitTexture( num2,num1);
	if(texCRT) texCRT->InitTexture(num2, (num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY, 32);
	CuTexImage::CuTexObj des1Tex = des1->BindTexture(texDataDesc, cudaCreateChannelDesc<uint4>());
	CuTexImage::CuTexObj des2Tex = des2->BindTexture(texDataDesc, cudaCreateChannelDesc<uint4>());

	MultiplyDescriptor_Kernel<<<grid, block>>>(des1Tex.handle, des2Tex.handle, (int*)texDot->_cuData, num1, num2,
												(texCRT? (int3*)texCRT->_cuData : NULL));
}

struct Matrix33
{
	float mat[3][3];
};



void __global__ MultiplyDescriptorG_Kernel(cudaTextureObject_t texDes1, cudaTextureObject_t texDes2,
										   cudaTextureObject_t texLoc1, cudaTextureObject_t texLoc2, 
										   int* d_result, int num1, int num2, int3* d_temp,
										   Matrix33 H, float hdistmax, Matrix33 F, float fdistmax,
										   int use_h, int use_f)
{
	int idx01 = (blockIdx.y  * MULT_BLOCK_DIMY);
	int idx02 = (blockIdx.x  * MULT_BLOCK_DIMX);

	int idx1 = idx01 + threadIdx.y;
	int idx2 = idx02 + threadIdx.x;
	__shared__ int data1[17 * 2 * MULT_BLOCK_DIMY];
	__shared__ float loc1[MULT_BLOCK_DIMY * 2];
	int read_idx1 = idx01 * 8 +  threadIdx.x ;
	int read_idx2 = idx2 * 8;
	int col4 = threadIdx.x & 0x3, row4 = threadIdx.x >> 2;
	int cache_idx1 = IMUL(row4, 17) + (col4 << 2);
#if MULT_BLOCK_DIMY == 16
	uint4 v = tex1Dfetch<uint4>(texDes1, read_idx1);
	data1[cache_idx1]   = v.x;
	data1[cache_idx1+1] = v.y;
	data1[cache_idx1+2] = v.z;
	data1[cache_idx1+3] = v.w;
#elif MULT_BLOCK_DIMY == 8
	if(threadIdx.x < 64)
	{
		uint4 v = tex1Dfetch<uint4>(texDes1, read_idx1);
		data1[cache_idx1]   = v.x;
		data1[cache_idx1+1] = v.y;
		data1[cache_idx1+2] = v.z;
		data1[cache_idx1+3] = v.w;
	}
#else
#error
#endif
	__syncthreads();
	if(threadIdx.x < MULT_BLOCK_DIMY * 2)
	{
		loc1[threadIdx.x] = tex1Dfetch<float>(texLoc1, 2 * idx01 + threadIdx.x);
	}
	__syncthreads();
	if(idx2 >= num2) return;
	int results[MULT_BLOCK_DIMY];
	/////////////////////////////////////////////////////////////////////////////////////////////
	//geometric verification
	/////////////////////////////////////////////////////////////////////////////////////////////
	int good_count = 0;
	float2 loc2 = tex1Dfetch<float2>(texLoc2, idx2);
#pragma unroll
	for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
	{

		if(idx1 + i < num1)
		{
			float* loci = loc1 + i * 2;
			float locx = loci[0], locy = loci[1];
			//homography
			float x[3], diff[2];
			x[0] = H.mat[0][0] * locx + H.mat[0][1] * locy + H.mat[0][2];
			x[1] = H.mat[1][0] * locx + H.mat[1][1] * locy + H.mat[1][2];
			x[2] = H.mat[2][0] * locx + H.mat[2][1] * locy + H.mat[2][2];
			diff[0] = FDIV(x[0], x[2]) - loc2.x;
			diff[1] = FDIV(x[1], x[2]) - loc2.y;
      float hdist = diff[0] * diff[0] + diff[1] * diff[1];
			// An explicit flag rather than an identity H with a huge threshold:
			// the latter silently rejects whenever the residual is inf or NaN.
			if(!use_h || hdist < hdistmax)
			{
				//check fundamental matrix
				float fx1[3], ftx2[3], x2fx1, se;
				fx1[0] = F.mat[0][0] * locx + F.mat[0][1] * locy + F.mat[0][2];
				fx1[1] = F.mat[1][0] * locx + F.mat[1][1] * locy + F.mat[1][2];
				fx1[2] = F.mat[2][0] * locx + F.mat[2][1] * locy + F.mat[2][2];

				ftx2[0] = F.mat[0][0] * loc2.x + F.mat[1][0] * loc2.y + F.mat[2][0];
				ftx2[1] = F.mat[0][1] * loc2.x + F.mat[1][1] * loc2.y + F.mat[2][1];

				x2fx1 = loc2.x * fx1[0]  + loc2.y * fx1[1] + fx1[2];
				se = FDIV(x2fx1 * x2fx1, fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1]);
				results[i] = (!use_f || se < fdistmax)? 0: -262144;
			}else
			{
				results[i] = -262144;
			}
		}else
		{
			results[i] = -262144;
		}
		good_count += (results[i] >=0);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////
	///compare feature descriptors anyway
	/////////////////////////////////////////////////////////////////////////////////////////////
	if(good_count > 0)
	{
#pragma unroll
		for(int i = 0; i < 8; ++i)
		{
			uint4 v = tex1Dfetch<uint4>(texDes2, read_idx2 + i);
			unsigned char* p2 = (unsigned char*)(&v);
#pragma unroll
			for(int k = 0; k < MULT_BLOCK_DIMY; ++k)
			{
				unsigned char* p1 = (unsigned char*) (data1 + k * 34 + i *  4 + (i/4));
				results[k] += 	 ( IMUL(p1[0], p2[0])	+ IMUL(p1[1], p2[1])
								 + IMUL(p1[2], p2[2])  	+ IMUL(p1[3], p2[3])
								 + IMUL(p1[4], p2[4])  	+ IMUL(p1[5], p2[5])
								 + IMUL(p1[6], p2[6])  	+ IMUL(p1[7], p2[7])
								 + IMUL(p1[8], p2[8])  	+ IMUL(p1[9], p2[9])
								 + IMUL(p1[10], p2[10])	+ IMUL(p1[11], p2[11])
								 + IMUL(p1[12], p2[12])	+ IMUL(p1[13], p2[13])
								 + IMUL(p1[14], p2[14])	+ IMUL(p1[15], p2[15]));
			}
		}
	}
	int dst_idx = IMUL(idx1, num2)  + idx2;
	if(d_temp)
	{
		int3 cmp_result = make_int3(0, -1, 0);
#pragma unroll
		for(int i= 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1)
			{
				cmp_result = results[i] > cmp_result.x?
				make_int3(results[i], idx1 + i, cmp_result.x) :
				make_int3(cmp_result.x, cmp_result.y, max(cmp_result.z, results[i]));
				d_result[dst_idx + IMUL(i, num2)] = max(results[i], 0);
			}else
			{
				break;
			}
		}
		d_temp[ IMUL(blockIdx.y, num2) + idx2] = cmp_result;
	}else
	{
#pragma unroll
		for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1) d_result[dst_idx + IMUL(i, num2)] = max(results[i], 0);
			else break;
		}
	}

}


void ProgramCU::MultiplyDescriptorG(CuTexImage* des1, CuTexImage* des2,
		CuTexImage* loc1, CuTexImage* loc2, CuTexImage* texDot, CuTexImage* texCRT,
		float* H, float hdistmax, float* F, float fdistmax, int use_h, int use_f)
{
	int num1 = des1->GetImgWidth() / 8;
	int num2 = des2->GetImgWidth() / 8;
	Matrix33 MatF, MatH;
	//copy the matrix
	memcpy(MatF.mat, F, 9 * sizeof(float));
	memcpy(MatH.mat, H, 9 * sizeof(float));
	//thread blocks
	dim3 grid(	(num2 + MULT_BLOCK_DIMX - 1)/ MULT_BLOCK_DIMX,
		(num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY);
	dim3 block(MULT_TBLOCK_DIMX, MULT_TBLOCK_DIMY);
	//intermediate results
	texDot->InitTexture( num2,num1);
	if(texCRT) texCRT->InitTexture( num2, (num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY, 3);
	CuTexImage::CuTexObj loc1Tex = loc1->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	CuTexImage::CuTexObj loc2Tex = loc2->BindTexture(texDataDesc, cudaCreateChannelDesc<float2>());
	CuTexImage::CuTexObj des1Tex = des1->BindTexture(texDataDesc, cudaCreateChannelDesc<uint4>());
	CuTexImage::CuTexObj des2Tex = des2->BindTexture(texDataDesc, cudaCreateChannelDesc<uint4>());
	MultiplyDescriptorG_Kernel<<<grid, block>>>(des1Tex.handle, des2Tex.handle, loc1Tex.handle, loc2Tex.handle,
												(int*)texDot->_cuData, num1, num2,
												(texCRT? (int3*)texCRT->_cuData : NULL),
												MatH, hdistmax, MatF, fdistmax, use_h, use_f);
}

// Guided matching against an essential matrix scored by the tangent Sampson
// error, i.e. in pixel units for an arbitrary central camera model.
//
// Each feature carries a unit bearing vector r and the Jacobian J = d(r)/d(px)
// of that bearing with respect to its pixel, laid out as 12 floats
// (r.x r.y r.z | J00 J10 J20 J01 J11 J21 | pad pad pad) so that it can be
// fetched as three float4. With C = r2' E r1 the residual is
//
//     C^2 / (||J1' (E' r2)||^2 + ||J2' (E r1)||^2)
//
// which reduces to the classical Sampson error scaled by f^2 for a pinhole.
void __global__ MultiplyDescriptorGRay_Kernel(cudaTextureObject_t texDes1, cudaTextureObject_t texDes2,
										   cudaTextureObject_t texLoc1, cudaTextureObject_t texLoc2,
										   int* d_result, int num1, int num2, int3* d_temp,
										   Matrix33 E, float edistmax)
{
	int idx01 = (blockIdx.y  * MULT_BLOCK_DIMY);
	int idx02 = (blockIdx.x  * MULT_BLOCK_DIMX);

	int idx1 = idx01 + threadIdx.y;
	int idx2 = idx02 + threadIdx.x;
	__shared__ int data1[17 * 2 * MULT_BLOCK_DIMY];
	__shared__ float loc1[MULT_BLOCK_DIMY * 12];
	int read_idx1 = idx01 * 8 +  threadIdx.x ;
	int read_idx2 = idx2 * 8;
	int col4 = threadIdx.x & 0x3, row4 = threadIdx.x >> 2;
	int cache_idx1 = IMUL(row4, 17) + (col4 << 2);
#if MULT_BLOCK_DIMY == 16
	uint4 v = tex1Dfetch<uint4>(texDes1, read_idx1);
	data1[cache_idx1]   = v.x;
	data1[cache_idx1+1] = v.y;
	data1[cache_idx1+2] = v.z;
	data1[cache_idx1+3] = v.w;
#elif MULT_BLOCK_DIMY == 8
	if(threadIdx.x < 64)
	{
		uint4 v = tex1Dfetch<uint4>(texDes1, read_idx1);
		data1[cache_idx1]   = v.x;
		data1[cache_idx1+1] = v.y;
		data1[cache_idx1+2] = v.z;
		data1[cache_idx1+3] = v.w;
	}
#else
#error
#endif
	__syncthreads();
	// Strided fill: a single pass leaves a garbage tail once the tile exceeds the
	// block width (MULT_BLOCK_DIMY == 16 gives 192 > 128).
	for(int j = threadIdx.x; j < MULT_BLOCK_DIMY * 12; j += MULT_TBLOCK_DIMX)
	{
		loc1[j] = tex1Dfetch<float>(texLoc1, 12 * idx01 + j);
	}
	__syncthreads();
	if(idx2 >= num2) return;
	int results[MULT_BLOCK_DIMY];
	/////////////////////////////////////////////////////////////////////////////////////////////
	//geometric verification
	/////////////////////////////////////////////////////////////////////////////////////////////
	int good_count = 0;
	float4 loc2a = tex1Dfetch<float4>(texLoc2, 3 * idx2);
	float4 loc2b = tex1Dfetch<float4>(texLoc2, 3 * idx2 + 1);
	float4 loc2c = tex1Dfetch<float4>(texLoc2, 3 * idx2 + 2);
	float r2x = loc2a.x, r2y = loc2a.y, r2z = loc2a.z;
	// J2 columns, in camera-ray space.
	float J2c0x = loc2a.w, J2c0y = loc2b.x, J2c0z = loc2b.y;
	float J2c1x = loc2b.z, J2c1y = loc2b.w, J2c1z = loc2c.x;
	// E' * r2, the gradient of the constraint w.r.t. r1.
	float etr2[3];
	etr2[0] = E.mat[0][0] * r2x + E.mat[1][0] * r2y + E.mat[2][0] * r2z;
	etr2[1] = E.mat[0][1] * r2x + E.mat[1][1] * r2y + E.mat[2][1] * r2z;
	etr2[2] = E.mat[0][2] * r2x + E.mat[1][2] * r2y + E.mat[2][2] * r2z;
#pragma unroll
	for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
	{
		if(idx1 + i < num1)
		{
			float* loci = loc1 + i * 12;
			float r1x = loci[0], r1y = loci[1], r1z = loci[2];
			// E * r1, the gradient of the constraint w.r.t. r2.
			float er1[3];
			er1[0] = E.mat[0][0] * r1x + E.mat[0][1] * r1y + E.mat[0][2] * r1z;
			er1[1] = E.mat[1][0] * r1x + E.mat[1][1] * r1y + E.mat[1][2] * r1z;
			er1[2] = E.mat[2][0] * r1x + E.mat[2][1] * r1y + E.mat[2][2] * r1z;
			float num = r2x * er1[0] + r2y * er1[1] + r2z * er1[2];
			// Project both gradients into pixel space through the Jacobians.
			float a0 = loci[3] * etr2[0] + loci[4] * etr2[1] + loci[5] * etr2[2];
			float a1 = loci[6] * etr2[0] + loci[7] * etr2[1] + loci[8] * etr2[2];
			float b0 = J2c0x * er1[0] + J2c0y * er1[1] + J2c0z * er1[2];
			float b1 = J2c1x * er1[0] + J2c1y * er1[1] + J2c1z * er1[2];
			float denom = a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1;
			// A zero denominator means the correspondence carries no geometric
			// information (an invalid bearing or a degenerate Jacobian); reject.
			results[i] = (denom > 0.0f && num * num <= edistmax * denom)? 0: -262144;
		}else
		{
			results[i] = -262144;
		}
		good_count += (results[i] >=0);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////
	///compare feature descriptors anyway
	/////////////////////////////////////////////////////////////////////////////////////////////
	if(good_count > 0)
	{
#pragma unroll
		for(int i = 0; i < 8; ++i)
		{
			uint4 v = tex1Dfetch<uint4>(texDes2, read_idx2 + i);
			unsigned char* p2 = (unsigned char*)(&v);
#pragma unroll
			for(int k = 0; k < MULT_BLOCK_DIMY; ++k)
			{
				unsigned char* p1 = (unsigned char*) (data1 + k * 34 + i *  4 + (i/4));
				results[k] += 	 ( IMUL(p1[0], p2[0])	+ IMUL(p1[1], p2[1])
								 + IMUL(p1[2], p2[2])  	+ IMUL(p1[3], p2[3])
								 + IMUL(p1[4], p2[4])  	+ IMUL(p1[5], p2[5])
								 + IMUL(p1[6], p2[6])  	+ IMUL(p1[7], p2[7])
								 + IMUL(p1[8], p2[8])  	+ IMUL(p1[9], p2[9])
								 + IMUL(p1[10], p2[10])	+ IMUL(p1[11], p2[11])
								 + IMUL(p1[12], p2[12])	+ IMUL(p1[13], p2[13])
								 + IMUL(p1[14], p2[14])	+ IMUL(p1[15], p2[15]));
			}
		}
	}
	int dst_idx = IMUL(idx1, num2)  + idx2;
	if(d_temp)
	{
		int3 cmp_result = make_int3(0, -1, 0);
#pragma unroll
		for(int i= 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1)
			{
				cmp_result = results[i] > cmp_result.x?
				make_int3(results[i], idx1 + i, cmp_result.x) :
				make_int3(cmp_result.x, cmp_result.y, max(cmp_result.z, results[i]));
				d_result[dst_idx + IMUL(i, num2)] = max(results[i], 0);
			}else
			{
				break;
			}
		}
		d_temp[ IMUL(blockIdx.y, num2) + idx2] = cmp_result;
	}else
	{
#pragma unroll
		for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1) d_result[dst_idx + IMUL(i, num2)] = max(results[i], 0);
			else break;
		}
	}
}

void ProgramCU::MultiplyDescriptorGRay(CuTexImage* des1, CuTexImage* des2,
		CuTexImage* loc1, CuTexImage* loc2, CuTexImage* texDot, CuTexImage* texCRT,
		float* E, float edistmax)
{
	int num1 = des1->GetImgWidth() / 8;
	int num2 = des2->GetImgWidth() / 8;
	Matrix33 MatE;
	memcpy(MatE.mat, E, 9 * sizeof(float));
	dim3 grid(	(num2 + MULT_BLOCK_DIMX - 1)/ MULT_BLOCK_DIMX,
		(num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY);
	dim3 block(MULT_TBLOCK_DIMX, MULT_TBLOCK_DIMY);
	texDot->InitTexture( num2,num1);
	if(texCRT) texCRT->InitTexture( num2, (num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY, 3);
	CuTexImage::CuTexObj loc1Tex = loc1->BindTexture(texDataDesc, cudaCreateChannelDesc<float>());
	CuTexImage::CuTexObj loc2Tex = loc2->BindTexture(texDataDesc, cudaCreateChannelDesc<float4>());
	CuTexImage::CuTexObj des1Tex = des1->BindTexture(texDataDesc, cudaCreateChannelDesc<uint4>());
	CuTexImage::CuTexObj des2Tex = des2->BindTexture(texDataDesc, cudaCreateChannelDesc<uint4>());
	MultiplyDescriptorGRay_Kernel<<<grid, block>>>(des1Tex.handle, des2Tex.handle, loc1Tex.handle, loc2Tex.handle,
												(int*)texDot->_cuData, num1, num2,
												(texCRT? (int3*)texCRT->_cuData : NULL),
												MatE, edistmax);
}

#define ROWMATCH_BLOCK_WIDTH 32
#define ROWMATCH_BLOCK_HEIGHT 1

void __global__  RowMatch_Kernel(int*d_dot, int* d_result, int num2, float distmax, float ratiomax)
{
#if ROWMATCH_BLOCK_HEIGHT == 1
	__shared__ int dotmax[ROWMATCH_BLOCK_WIDTH];
	__shared__ int dotnxt[ROWMATCH_BLOCK_WIDTH];
	__shared__ int dotidx[ROWMATCH_BLOCK_WIDTH];
	int	row = blockIdx.y;
#else
	__shared__ int x_dotmax[ROWMATCH_BLOCK_HEIGHT][ROWMATCH_BLOCK_WIDTH];
	__shared__ int x_dotnxt[ROWMATCH_BLOCK_HEIGHT][ROWMATCH_BLOCK_WIDTH];
	__shared__ int x_dotidx[ROWMATCH_BLOCK_HEIGHT][ROWMATCH_BLOCK_WIDTH];
	int*	dotmax = x_dotmax[threadIdx.y];
	int*	dotnxt = x_dotnxt[threadIdx.y];
	int*	dotidx = x_dotidx[threadIdx.y];
	int row = IMUL(blockIdx.y, ROWMATCH_BLOCK_HEIGHT) + threadIdx.y;
#endif

	int base_address = IMUL(row , num2);
	int t_dotmax = 0, t_dotnxt = 0, t_dotidx = -1;
	for(int i = 0; i < num2; i += ROWMATCH_BLOCK_WIDTH)
	{
		if(threadIdx.x + i < num2)
		{
			int v = d_dot[base_address + threadIdx.x + i];  // tex1Dfetch(texDOT, base_address + threadIdx.x + i);
			bool test = v > t_dotmax;
			t_dotnxt = test? t_dotmax : max(t_dotnxt, v);
			t_dotidx = test? (threadIdx.x + i) : t_dotidx;
			t_dotmax = test? v: t_dotmax;
		}
		__syncthreads();
	}
	dotmax[threadIdx.x] = t_dotmax;
	dotnxt[threadIdx.x] = t_dotnxt;
	dotidx[threadIdx.x] = t_dotidx;
	__syncthreads();

#pragma unroll
	for(int step = ROWMATCH_BLOCK_WIDTH/2; step >0; step /= 2)
	{
		if(threadIdx.x < step)
		{
			int v1 = dotmax[threadIdx.x], v2 = dotmax[threadIdx.x + step];
			bool test =  v2 > v1;
			dotnxt[threadIdx.x] = test? max(v1, dotnxt[threadIdx.x + step]) :max(dotnxt[threadIdx.x], v2);
			dotidx[threadIdx.x] = test? dotidx[threadIdx.x + step] : dotidx[threadIdx.x];
			dotmax[threadIdx.x] = test? v2 : v1;
		}
		__syncthreads();
	}
	if(threadIdx.x == 0)
	{
		float dist =  acos(min(dotmax[0] * 0.000003814697265625f, 1.0));
		float distn = acos(min(dotnxt[0] * 0.000003814697265625f, 1.0));
		//float ratio = dist / distn;
		d_result[row] = (dist < distmax) && (dist < distn * ratiomax) ? dotidx[0] : -1;//?  : -1;
	}

}


void ProgramCU::GetRowMatch(CuTexImage* texDot, CuTexImage* texMatch, float distmax, float ratiomax)
{
	int num1 = texDot->GetImgHeight();
	int num2 = texDot->GetImgWidth();
	dim3 grid(1, num1/ROWMATCH_BLOCK_HEIGHT);
	dim3 block(ROWMATCH_BLOCK_WIDTH, ROWMATCH_BLOCK_HEIGHT);
	RowMatch_Kernel<<<grid, block>>>((int*)texDot->_cuData,
		(int*)texMatch->_cuData, num2, distmax, ratiomax);
}

#define COLMATCH_BLOCK_WIDTH 32

void __global__  ColMatch_Kernel(int3*d_crt, int* d_result, int height, int num2, float distmax, float ratiomax)
{
	int col = COLMATCH_BLOCK_WIDTH * blockIdx.x + threadIdx.x;
	if(col >= num2) return;
	int3 result = d_crt[col];//tex1Dfetch(texCT, col);
	int read_idx = col + num2;
	for(int i = 1; i < height; ++i, read_idx += num2)
	{
		int3 temp = d_crt[read_idx];//tex1Dfetch(texCT, read_idx);
		result = result.x < temp.x?
			make_int3(temp.x, temp.y, max(result.x, temp.z)) :
			make_int3(result.x, result.y, max(result.z, temp.x));
	}

	float dist =  acos(min(result.x * 0.000003814697265625f, 1.0));
	float distn = acos(min(result.z * 0.000003814697265625f, 1.0));
		//float ratio = dist / distn;
	d_result[col] = (dist < distmax) && (dist < distn * ratiomax) ? result.y : -1;//?  : -1;

}

void ProgramCU::GetColMatch(CuTexImage* texCRT, CuTexImage* texMatch, float distmax, float ratiomax)
{
	int height = texCRT->GetImgHeight();
	int num2 = texCRT->GetImgWidth();
	//texCRT->BindTexture(texCT);
    dim3 grid((num2 + COLMATCH_BLOCK_WIDTH -1) / COLMATCH_BLOCK_WIDTH);
    dim3 block(COLMATCH_BLOCK_WIDTH);
	ColMatch_Kernel<<<grid, block>>>((int3*)texCRT->_cuData, (int*) texMatch->_cuData, height, num2, distmax, ratiomax);
}

#endif
