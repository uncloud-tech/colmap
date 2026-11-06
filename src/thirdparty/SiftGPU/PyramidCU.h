////////////////////////////////////////////////////////////////////////////
//	File:		PyramidCU.h
//	Author:		Changchang Wu
//	Description : interface for the PyramdCU
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



#ifndef _PYRAMID_CU_H
#define _PYRAMID_CU_H
#if defined(SIFTGPU_CUDA_ENABLED)

#include "StreamedPlane.h"

namespace siftgpu {
// Partial phase owner: no list/descriptor allocations. Levels are
// octave-major, in the resident construction order. Backing files remain owned
// until this result is retired. Staging is not a page-cache cap.
struct StreamPyramidResult {
  int first_octave, level_min, level_num;
  bool exact_domain;
  std::vector<std::unique_ptr<StreamedPlane>> levels;
  // Empty for Gaussian-only calls. Otherwise octave-major, current Gaussian
  // indices 1..level_num-1, each paired with its immediate predecessor.
  std::vector<std::unique_ptr<StreamedPlane>> dogs, gradients;
  // Staging peaks and summed transfers/times; scratch_bytes is peak backing
  // including caller input and transient seeds/H planes. File cache charged
  // by the OS is NOT certified by these payload counters (see StreamedPlane).
  StreamFilterStats stats;
};
struct StreamDetectionLevel {
  int octave, level;  // Absolute octave and original Gaussian level index.
  StreamKeyResult detection;
};
struct StreamDetectionPyramidResult {
  StreamPyramidResult pyramid;
  // Resident GenerateFeatureList octave/level traversal, not tile lists.
  // No early truncation/global limiter is applied at this partial boundary.
  std::vector<StreamDetectionLevel> detections;
};
struct StreamFeatureListsResult {
  // Canonical octave-major indexing, unlike detection traversal order.
  // Null files for empty, early-skipped, or globally discarded levels.
  std::vector<StreamListResult> levels;
  std::vector<int> level_counts;
  int count = 0;
  StreamFilterStats stats;
};
struct StreamExtractionResult {
  // Canonical octave/level order; corresponding xyso and descriptor rows.
  // xyso uses original DownloadKeypoints/GetFeatureVector coordinates.
  StreamFeatureListsResult features;
  std::vector<StreamDescriptorResult> descriptors;
  StreamFilterStats stats;
};
}  // namespace siftgpu

class GLTexImage;
class CuTexImage;
class SiftPyramid;
class PyramidCU:public SiftPyramid
{
	// Reset from each effective (post-rotation) input, never from capacity.
    bool _exact_domain = false;
	CuTexImage* _inputTex;
	CuTexImage* _allPyramid;
	CuTexImage* _denseKey; // Sole float4 owner, also Gaussian-filter scratch.
	CuTexImage* _histoPyramidTex;
	CuTexImage* _featureTex;
	CuTexImage* _descriptorTex;
	CuTexImage* _orientationTex;
	GLuint		_bufferPBO;
    GLTexImage* _bufferTEX;
public:
    // Does not call InitPyramid/ResizePyramid and cannot report extraction
    // success. Temporary constraint: first octave 0 or exact-domain -1 only. In
    // particular aligned 6048-wide/default -1 remains unsupported.
    static siftgpu::StreamPyramidResult BuildGaussianPyramidStreamed(
        SiftParam& parameters,
        siftgpu::StreamedPlane& input,
        int first_octave,
        int octave_count,
        const siftgpu::StreamFilterOptions& options);
    // Coherent partial Gaussian -> DoG/linear float2 gradient pipeline only.
    // Inherits Gaussian first-octave restrictions; NOT extraction success.
    static siftgpu::StreamPyramidResult BuildPyramidStreamed(
        SiftParam& parameters,
        siftgpu::StreamedPlane& input,
        int first_octave,
        int octave_count,
        const siftgpu::StreamFilterOptions& options);
    // Adds bounded KEY + complete per-row histogram trees to the partial
    // pyramid. No feature list, global limiting, orientation, or extractor
    // success: those consumers still require a separate bounded integration.
    static siftgpu::StreamDetectionPyramidResult BuildDetectionPyramidStreamed(
        SiftParam& parameters,
        siftgpu::StreamedPlane& input,
        int first_octave,
        int octave_count,
        const siftgpu::StreamFilterOptions& options);
    // Detection records only, original traversal/early stop followed by the
    // FIRST global limiter. Caller supplies complete detected KEY trees, not
    // user-supplied keypoints. Orientation expansion/second limiter and
    // descriptors remain pending; not an installed extractor route.
    static siftgpu::StreamFeatureListsResult GenerateFeatureListsStreamed(
        siftgpu::StreamDetectionPyramidResult& detection,
        int truncate_method,
        int threshold,
        const siftgpu::StreamFilterOptions& options);
    // Consumes SECOND-limited expanded xyso, emits descriptors and original
    // output-coordinate xyso. No installed SiftGPU route selection.
    static siftgpu::StreamExtractionResult GetFeatureDescriptorsStreamed(
        SiftParam& parameters,
        siftgpu::StreamDetectionPyramidResult& detection,
        siftgpu::StreamFeatureListsResult& orientations,
        const siftgpu::StreamFilterOptions& options);

    // Consumes FIRST-limited detected int4 lists, emits ordered xyso float4
    // files after original orientation expansion and SECOND global limiter.
    // Separate partial native phase, not installed extraction route selection.
    static siftgpu::StreamFeatureListsResult GetFeatureOrientationsStreamed(
        SiftParam& parameters,
        siftgpu::StreamDetectionPyramidResult& detection,
        siftgpu::StreamFeatureListsResult& lists,
        int truncate_method,
        int threshold,
        const siftgpu::StreamFilterOptions& options);
	virtual void GetFeatureDescriptors();
	virtual void GenerateFeatureListTex();
	virtual void ReshapeFeatureListCPU();
	virtual void GenerateFeatureDisplayVBO();
	virtual void DestroySharedData();
	virtual void DestroyPerLevelData();
	virtual void DestroyPyramidData();
	virtual void DownloadKeypoints();
	virtual void GenerateFeatureListCPU();
	virtual void GenerateFeatureList();
	virtual GLTexImage* GetLevelTexture(int octave, int level);
	virtual GLTexImage* GetLevelTexture(int octave, int level, int dataName);
	virtual void BuildPyramid(GLTexInput * input);
	virtual void DetectKeypointsEX();
	virtual void ComputeGradient();
	virtual void GetFeatureOrientations();
	virtual void GetSimplifiedOrientation();
	virtual void InitPyramid(int w, int h, int ds = 0);
	virtual void ResizePyramid(int w, int h);
    virtual int  IsUsingRectDescription(){return _existing_keypoints & SIFT_RECT_DESCRIPTION; }	
	//////////
	void FitPyramid(int w, int h);

    void InitializeContext();
	int ResizeFeatureStorage();
	int FitHistogramPyramid(CuTexImage* tex);
	void SetLevelFeatureNum(int idx, int fcount);
	void ConvertInputToCU(GLTexInput* input);
	GLTexImage* ConvertTexCU2GL(CuTexImage* tex, int dataName, CuTexImage* dog = NULL);
	CuTexImage* GetBaseLevel(int octave, int dataName = DATA_GAUSSIAN);
    void TruncateWidth(int& w) { w = GLTexInput::TruncateWidthCU(w); }
    //////////////////////////
    static int CheckCudaDevice(int device);
private:
	CuTexImage* FitDenseKey(int octave);
	CuTexImage* RegenerateDenseKey(int octave, int level);
	void GenerateFeatureList(int i, int j, int reduction_count, vector<int>& hbuffer);
public:
	PyramidCU(SiftParam& sp);
	virtual ~PyramidCU();
};



#endif
#endif
