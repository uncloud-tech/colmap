////////////////////////////////////////////////////////////////////////////
//	File:		ProgramCU.h
//	Author:		Changchang Wu
//	Description :	interface for the ProgramCU classes.
//					It is basically a wrapper around all the CUDA kernels
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

#ifndef _PROGRAM_CU_H
#define _PROGRAM_CU_H
#if defined(SIFTGPU_CUDA_ENABLED)
#include "StreamedPlane.h"

class CuTexImage;

class ProgramCU
{
public:
    //GPU FUNCTIONS
	static void FinishCUDA();
	static int  CheckErrorCUDA(const char* location);
    static int  CheckCudaDevice(int device);
public:
    ////SIFTGPU FUNCTIONS
	static void CreateFilterKernel(float sigma, float* kernel, int& width);
	template<int KWIDTH> static void FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf);
	static void FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf, float sigma);
    // Native Gaussian phase only. No extractor/CLI streaming selection yet.
    // Like FilterImage, serialized with other users of the constant d_kernel.
    static siftgpu::StreamFilterResult FilterImageStreamed(
        siftgpu::StreamedPlane& input, float sigma,
        const siftgpu::StreamFilterOptions& options);
    static siftgpu::StreamFilterResult SampleImageStreamed(
        siftgpu::StreamedPlane& input,
        int width,
        int height,
        bool up,
        bool exact_domain,
        const siftgpu::StreamFilterOptions& options);
    // Same ComputeDOG kernel over full-width stripes with one-row halos.
    static siftgpu::StreamDogResult ComputeDOGStreamed(
        siftgpu::StreamedPlane& current,
        siftgpu::StreamedPlane& previous,
        const siftgpu::StreamFilterOptions& options);
	static void ComputeDOG(CuTexImage* gus, CuTexImage* dog, CuTexImage* got);
    // Partial detection only: no lists, truncation, orientation or
    // descriptors. File row offsets retain absolute row identity; KEY
    // records remain the original (sign, dx, dy, ds), not tile-local
    // feature coordinates.
    static siftgpu::StreamKeyResult ComputeKEYStreamed(
        siftgpu::StreamedPlane& previous,
        siftgpu::StreamedPlane& current,
        siftgpu::StreamedPlane& next,
        float Tdog,
        float Tedge,
        bool exact_domain,
        const siftgpu::StreamFilterOptions& options);
	static void ComputeKEY(CuTexImage* dog, CuTexImage* key, float Tdog, float Tedge);
	static void InitHistogram(CuTexImage* key, CuTexImage* hist);
	static void ReduceHistogram(CuTexImage*hist1, CuTexImage* hist2);
	static void GenerateList(CuTexImage* list, CuTexImage* hist);
    // Ordered int4 detection records only; no orientation or supplied-key path.
    // Nonzero single-root tiny domains remain unqualified and fail closed.
    static siftgpu::StreamListResult GenerateListStreamed(
        siftgpu::StreamKeyResult& detection,
        const siftgpu::StreamFilterOptions& options);
    // Detected int4 lists only. One-feature batches retain full-width global
    // sampling windows, CPU ordered expansion, and disk-backed float4 output.
    // Descriptors and user-supplied/rectangular keys are deliberately absent.
    static siftgpu::StreamListResult ComputeOrientationStreamed(
        siftgpu::StreamListResult& list,
        siftgpu::StreamedPlane& gradient,
        siftgpu::StreamedPlane& key,
        float sigma,
        float sigma_step,
        const siftgpu::StreamFilterOptions& options);
	static void ComputeOrientation(CuTexImage*list, CuTexImage* got, CuTexImage*key,
		float sigma, float sigma_step, int existing_keypoint);
        static siftgpu::StreamDescriptorResult ComputeDescriptorStreamed(
            siftgpu::StreamListResult& list,
            siftgpu::StreamedPlane& gradient,
            const siftgpu::StreamFilterOptions& options);

        static void ComputeDescriptor(CuTexImage*list, CuTexImage* got, CuTexImage* dtex, int rect = 0, int stream = 0);

    //data conversion
	static void SampleImageU(CuTexImage *dst, CuTexImage *src, int log_scale);
	static void SampleImageD(CuTexImage *dst, CuTexImage *src, int log_scale = 1);
	static void ReduceToSingleChannel(CuTexImage* dst, CuTexImage* src, int convert_rgb);
    static void ConvertByteToFloat(CuTexImage*src, CuTexImage* dst);

    //visualization
	static void DisplayConvertDOG(CuTexImage* dog, CuTexImage* out);
	static void DisplayConvertGRD(CuTexImage* got, CuTexImage* out);
	static void DisplayConvertKEY(CuTexImage* key, CuTexImage* dog, CuTexImage* out);
	static void DisplayKeyPoint(CuTexImage* ftex, CuTexImage* out);
	static void DisplayKeyBox(CuTexImage* ftex, CuTexImage* out);

	//SIFTMATCH FUNCTIONS
	static void MultiplyDescriptor(CuTexImage* tex1, CuTexImage* tex2, CuTexImage* texDot, CuTexImage* texCRT);
	static void MultiplyDescriptorG(CuTexImage* texDes1, CuTexImage* texDes2,
		CuTexImage* texLoc1, CuTexImage* texLoc2, CuTexImage* texDot, CuTexImage* texCRT,
		float* H, float hdistmax, float* F, float fdistmax,
		int use_h, int use_f);
	// Guided matching against an essential matrix using per-feature bearing
	// vectors and unprojection Jacobians (12 floats per feature).
	static void MultiplyDescriptorGRay(CuTexImage* texDes1, CuTexImage* texDes2,
		CuTexImage* texLoc1, CuTexImage* texLoc2, CuTexImage* texDot, CuTexImage* texCRT,
		float* E, float edistmax);
	static void GetRowMatch(CuTexImage* texDot, CuTexImage* texMatch, float distmax, float ratiomax);
	static void GetColMatch(CuTexImage* texCRT, CuTexImage* texMatch, float distmax, float ratiomax);
};

#endif
#endif

