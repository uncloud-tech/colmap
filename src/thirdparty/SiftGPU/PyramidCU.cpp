////////////////////////////////////////////////////////////////////////////
//	File:		PyramidCU.cpp
//	Author:		Changchang Wu
//	Description : implementation of the PyramidCU class.
//				CUDA-based implementation of SiftPyramid
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
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cmath>
using namespace std;

#include "GlobalUtil.h"
#include "GLTexImage.h"
#include "CuTexImage.h" 
#include "SiftGPU.h"
#include "SiftPyramid.h"
#include "ProgramCU.h"
#include "PyramidCU.h"
#include "CudaImageDomain.h"
#include "SiftRequirements.h"


namespace {
// Storage reservation and fitting must use the same slot order and reductions.
// Rows are not reduced: all kernels address packed row * width + column.
template <typename Visit>
int TraverseHistogram(int width, int height, bool exact_domain, int levels,
                      Visit visit) {
    int w = exact_domain ? siftgpu::HistogramWidth(width) : (width + 2) >> 2;
    int count = 0;
    for (int slot = levels - 1; slot >= 0; --slot) {
        visit(slot, w, height);
        ++count;
        if (w == 1) break;
        w = (w + 3) >> 2;
    }
    return count;
}
}  // namespace

//#include "imdebug/imdebuggl.h"
//#pragma comment (lib, "../lib/imdebug.lib")



#define USE_TIMING()		double t, t0, tt;
#define OCTAVE_START()		if(GlobalUtil::_timingO){	t = t0 = CLOCK();	cout<<"#"<<i+_down_sample_factor<<"\t";	}
#define LEVEL_FINISH()		if(GlobalUtil::_timingL){	ProgramCU::FinishCUDA();	tt = CLOCK();cout<<(tt-t)<<"\t";	t = CLOCK();}
#define OCTAVE_FINISH()		if(GlobalUtil::_timingO)cout<<"|\t"<<(CLOCK()-t0)<<endl;


PyramidCU::PyramidCU(SiftParam& sp) : SiftPyramid(sp)
{
	_allPyramid = NULL;
	_denseKey = NULL;
	_histoPyramidTex = NULL;
	_featureTex = NULL;
	_descriptorTex = NULL;
	_orientationTex = NULL;
	_bufferPBO = 0;
    _bufferTEX = NULL;
	_inputTex = new CuTexImage();

    /////////////////////////
    InitializeContext();
}

PyramidCU::~PyramidCU()
{
	DestroyPerLevelData();
	DestroySharedData();
	DestroyPyramidData();
	if(_inputTex) delete _inputTex;
    if(_bufferPBO) glDeleteBuffers(1, &_bufferPBO);
    if(_bufferTEX) delete _bufferTEX;
}

void PyramidCU::InitializeContext()
{
    GlobalUtil::InitGLParam(1);
    GlobalUtil::_GoodOpenGL = max(GlobalUtil::_GoodOpenGL, 1); 
}

void PyramidCU::InitPyramid(int w, int h, int ds)
{
	int wp, hp, toobig = 0;
    // GLTexInput has already applied ds to its CPU upload dimensions.
    _exact_domain = ((w >> ds) % 4) != 0;
	if(ds == 0)
	{
		//
		if(!_exact_domain) TruncateWidth(w);
		////
		_down_sample_factor = 0;
		if(GlobalUtil::_octave_min_default>=0)
		{
			wp = w >> _octave_min_default;
			hp = h >> _octave_min_default;
		}else
		{
			//can't upsample by more than 8
			_octave_min_default = max(-3, _octave_min_default);
			//
			wp = w << (-_octave_min_default);
			hp = h << (-_octave_min_default);
		}
		_octave_min = _octave_min_default;
	}else
	{
		//must use 0 as _octave_min; 
		_octave_min = 0;
		_down_sample_factor = ds;
		w >>= ds;
		h >>= ds;
		/////

		if(!_exact_domain) TruncateWidth(w);

		wp = w;
		hp = h; 

	}

    // A requested positive first octave on a tiny input has no nonempty
    // historical output to preserve. Keep one sample rather than a zero grid.
    wp = max(1, wp);
    hp = max(1, hp);

	while(wp > GlobalUtil::_texMaxDim  || hp > GlobalUtil::_texMaxDim )
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 1;
	}

	while(GlobalUtil::_MemCapGPU > 0 && GlobalUtil::_FitMemoryCap &&  (wp >_pyramid_width || hp > _pyramid_height)&& 
		max(max(wp, hp), max(_pyramid_width, _pyramid_height)) >  1024 * sqrt(GlobalUtil::_MemCapGPU / 110.0))
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 2;
	}


	if(toobig && GlobalUtil::_verbose && _octave_min > 0)
	{
		std::cout<<(toobig == 2 ? "[**SKIP OCTAVES**]:\tExceeding Memory Cap (-nomc)\n" :
					"[**SKIP OCTAVES**]:\tReaching the dimension limit(-maxd)!\n");
	}
	//ResizePyramid(wp, hp);
	if( wp == _pyramid_width && hp == _pyramid_height && _allocated )
	{
		FitPyramid(wp, hp);
	}else if(GlobalUtil::_ForceTightPyramid || _allocated ==0)
	{
		ResizePyramid(wp, hp);
	}
	else if( wp > _pyramid_width || hp > _pyramid_height )
	{
		ResizePyramid(max(wp, _pyramid_width), max(hp, _pyramid_height));
		if(wp < _pyramid_width || hp < _pyramid_height)  FitPyramid(wp, hp);
	}
	else
	{
		//try use the pyramid allocated for large image on small input images
		FitPyramid(wp, hp);
	}
}

void PyramidCU::ResizePyramid(int w, int h)
{
	//
	unsigned int totalkb = 0;
	int _octave_num_new, input_sz, i, j;
	//

	if(_pyramid_width == w && _pyramid_height == h && _allocated) return;

	if(w > GlobalUtil::_texMaxDim || h > GlobalUtil::_texMaxDim) return ;

	if(GlobalUtil::_verbose && GlobalUtil::_timingS) std::cout<<"[Allocate Pyramid]:\t" <<w<<"x"<<h<<endl;
	//first octave does not change
	_pyramid_octave_first = 0;

	
	//compute # of octaves

	input_sz = min(w,h) ;
	_pyramid_width =  w;
	_pyramid_height =  h;



	//reset to preset parameters

	_octave_num_new  = GlobalUtil::_octave_num_default;

	if(_octave_num_new < 1) 
	{
		_octave_num_new = (int) floor (log ( double(input_sz))/log(2.0)) -3 ;
		if(_octave_num_new<1 ) _octave_num_new = 1;
	}

	if(_pyramid_octave_num != _octave_num_new)
	{
		//destroy the original pyramid if the # of octave changes
		if(_octave_num >0) 
		{
			DestroyPerLevelData();
			DestroyPyramidData();
		}
		_pyramid_octave_num = _octave_num_new;
	}

	_octave_num = _pyramid_octave_num;

	int noct = _octave_num;
	int nlev = param._level_num;

	//	//initialize the pyramid
	if(_allPyramid==NULL)	_allPyramid = new CuTexImage[ noct* nlev * DATA_NUM];

	CuTexImage * gus =  GetBaseLevel(_octave_min, DATA_GAUSSIAN);
	CuTexImage * dog =  GetBaseLevel(_octave_min, DATA_DOG);
	CuTexImage * got =  GetBaseLevel(_octave_min, DATA_GRAD);
	CuTexImage * key =  GetBaseLevel(_octave_min, DATA_KEYPOINT);

	////////////there could be "out of memory" happening during the allocation

	for(i = 0; i< noct; i++)
	{
		int wa = ((w + 3) / 4) * 4;

		totalkb += (size_t(wa) * h * (nlev * 4 - 7) * 4 / 1024);
		for( j = 0; j< nlev; j++, gus++, dog++, got++, key++)
		{
			gus->InitTexture(wa, h);
            gus->SetImageSize(siftgpu::LevelWidth(w, _exact_domain), h); //nlev
			if(j==0)continue;
			dog->InitTexture(wa, h);
            dog->SetImageSize(siftgpu::LevelWidth(w, _exact_domain), h);  //nlev -1
			if(	j >= 1 && j < 1 + param._dog_level_num)
			{
				// Orientation, descriptors and display bind this linear float2 storage.
				got->InitTexture(wa, h, 2);
                got->SetImageSize(siftgpu::LevelWidth(w, _exact_domain), h); //2 * nlev - 6
			}
			if(j > 1 && j < nlev -1) {
                key->SetImageSize(siftgpu::LevelWidth(w, _exact_domain), h);
            } // Shape metadata only; the sole dense owner is _denseKey.
		}
		w>>=1;
		h>>=1;
        if(_exact_domain) { w = siftgpu::LevelWidth(w, true); h = max(1, h); }
	}

	// Reserve once at the largest octave; fits below only change logical shape.
	CuTexImage* base = GetBaseLevel(_octave_min);
	if(_denseKey == NULL) _denseKey = new CuTexImage;
	_denseKey->InitTexture((base->GetImgWidth() + 3) / 4 * 4,
	                       base->GetImgHeight(), 4);
	totalkb += _denseKey->GetDataSize() / 1024;
	totalkb += ResizeFeatureStorage();

	if(ProgramCU::CheckErrorCUDA("ResizePyramid")) SetFailStatus(); 

	_allocated = 1;

	if(GlobalUtil::_verbose && GlobalUtil::_timingS) std::cout<<"[Allocate Pyramid]:\t" <<(totalkb/1024)<<"MB\n";

}

void PyramidCU::FitPyramid(int w, int h)
{
	_pyramid_octave_first = 0;
	//
	_octave_num  = GlobalUtil::_octave_num_default;

	int _octave_num_max = max(1, (int) floor (log ( double(min(w, h)))/log(2.0))  -3 );

	if(_octave_num < 1 || _octave_num > _octave_num_max) 
	{
		_octave_num = _octave_num_max;
	}


	int pw = _pyramid_width>>1, ph = _pyramid_height>>1;
	while(_pyramid_octave_first + _octave_num < _pyramid_octave_num &&  
		pw >= w && ph >= h)
	{
		_pyramid_octave_first++;
		pw >>= 1;
		ph >>= 1;
	}

	//////////////////
	int nlev = param._level_num;
	CuTexImage * gus =  GetBaseLevel(_octave_min, DATA_GAUSSIAN);
	CuTexImage * dog =  GetBaseLevel(_octave_min, DATA_DOG);
	CuTexImage * got =  GetBaseLevel(_octave_min, DATA_GRAD);
	CuTexImage * key =  GetBaseLevel(_octave_min, DATA_KEYPOINT);
	for(int i = 0; i< _octave_num; i++)
	{
		int wa = ((w + 3) / 4) * 4;

		for(int j = 0; j< nlev; j++, gus++, dog++, got++, key++)
		{
			gus->InitTexture(wa, h);
            gus->SetImageSize(siftgpu::LevelWidth(w, _exact_domain), h); //nlev
			if(j==0)continue;
			dog->InitTexture(wa, h);
            dog->SetImageSize(siftgpu::LevelWidth(w, _exact_domain), h);  //nlev -1
			if(	j >= 1 && j < 1 + param._dog_level_num)
			{
				// Orientation, descriptors and display bind this linear float2 storage.
				got->InitTexture(wa, h, 2);
                got->SetImageSize(siftgpu::LevelWidth(w, _exact_domain), h); //2 * nlev - 6
			}
			if(j > 1 && j < nlev -1) {
                key->SetImageSize(siftgpu::LevelWidth(w, _exact_domain), h);
            } // Shape metadata only; the sole dense owner is _denseKey.
		}
		w>>=1;
		h>>=1;
        if(_exact_domain) { w = siftgpu::LevelWidth(w, true); h = max(1, h); }
	}
}

int PyramidCU::CheckCudaDevice(int device)
{
    return ProgramCU::CheckCudaDevice(device);
}

void PyramidCU::SetLevelFeatureNum(int idx, int fcount)
{
	_featureTex[idx].InitTexture(fcount, 1, 4);
	_levelFeatureNum[idx] = fcount;
}

int PyramidCU::ResizeFeatureStorage()
{
	int totalkb = 0;
	if(_levelFeatureNum==NULL)	_levelFeatureNum = new int[_octave_num * param._dog_level_num];
	std::fill(_levelFeatureNum, _levelFeatureNum+_octave_num * param._dog_level_num, 0); 

	int wmax = GetBaseLevel(_octave_min)->GetImgWidth();
	int hmax = GetBaseLevel(_octave_min)->GetImgHeight();
	int whmax = max(wmax, hmax);
	int i;

	//
	int num = max(1, (int)ceil(log(double(whmax))/log(4.0)));

	if( _hpLevelNum != num)
	{
		_hpLevelNum = num;
		if(_histoPyramidTex ) delete [] _histoPyramidTex;
		_histoPyramidTex = new CuTexImage[_hpLevelNum];
	}

    // A smaller input can reuse any retained octave as its first octave.
    // Fitting shifts the histogram's starting slot by floor(first / 2).
    // Bound each such traversal, including exact/rounded domain transitions:
    // ceil(raw_width / 4) also equals the legacy rounded width's first
    // reduction. Later active octaves are no larger than that first octave.
    const auto shapes=siftgpu::SiftHistogramCapacityShapes(
        _pyramid_width,_pyramid_height,_pyramid_octave_num,_hpLevelNum);
    for (i = 0; i < _hpLevelNum; ++i) {
        const auto shape = shapes[i];
        // InitTexture retains the existing overflow/index-limit checks.
        _histoPyramidTex[i].InitTexture(shape.first, shape.second, 4);
        totalkb += size_t(shape.first) * shape.second * 16 / 1024;
    }

	//initialize the feature texture
	int idx = 0, n = _octave_num * param._dog_level_num;
	if(_featureTex==NULL)	_featureTex = new CuTexImage[n];
	if(GlobalUtil::_MaxOrientation >1 && GlobalUtil::_OrientationPack2==0 && _orientationTex== NULL)
		_orientationTex = new CuTexImage[n];


	for(i = 0; i < _octave_num; i++)
	{
		CuTexImage * tex = GetBaseLevel(i+_octave_min);
		int fmax = int(tex->GetImgWidth() * tex->GetImgHeight()*GlobalUtil::_MaxFeaturePercent);
		//
		if(fmax > GlobalUtil::_MaxLevelFeatureNum) fmax = GlobalUtil::_MaxLevelFeatureNum;
		else if(fmax < 32) fmax = 32;	//give it at least a space of 32 feature

		for(int j = 0; j < param._dog_level_num; j++, idx++)
		{
			_featureTex[idx].InitTexture(fmax, 1, 4);
			totalkb += fmax * 16 /1024;
			//
			if(GlobalUtil::_MaxOrientation>1 && GlobalUtil::_OrientationPack2 == 0)
			{
				_orientationTex[idx].InitTexture(fmax, 1, 4);
				totalkb += fmax * 16 /1024;
			}
		}
	}


	//this just need be initialized once
	if(_descriptorTex==NULL)
	{
		//initialize feature texture pyramid
		int fmax = _featureTex->GetImgWidth();
		_descriptorTex = new CuTexImage;
		totalkb += ( fmax /2);
		_descriptorTex->InitTexture(fmax *128, 1, 1);
	}else
	{
		totalkb +=  _descriptorTex->GetDataSize()/1024;
	}
	return totalkb;
}

void PyramidCU::GetFeatureDescriptors() 
{
	//descriptors...
	float* pd =  &_descriptor_buffer[0];
	vector<float> descriptor_buffer2;

	//use another buffer if we need to re-order the descriptors
	if(_keypoint_index.size() > 0)
	{
		descriptor_buffer2.resize(_descriptor_buffer.size());
		pd = &descriptor_buffer2[0];
	}

	CuTexImage * got, * ftex= _featureTex;
	for(int i = 0, idx = 0; i < _octave_num; i++)
	{
		got = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;
		for(int j = 0; j < param._dog_level_num; j++, ftex++, idx++, got++)
		{
			if(_levelFeatureNum[idx]==0) continue;
            if (siftgpu::active_sift_device_budget &&
                uint64_t(_levelFeatureNum[idx])*128*sizeof(float)>uint64_t(_descriptorTex->GetDataSize()))
              throw std::runtime_error("SIFT resident descriptor capacity insufficient for actual count; no capacity/quality fallback");
            ProgramCU::ComputeDescriptor(ftex, got, _descriptorTex, IsUsingRectDescription());//process
			_descriptorTex->CopyToHost(pd); //readback descriptor
			pd += 128*_levelFeatureNum[idx];
		}
	}

	if(GlobalUtil::_timingS) ProgramCU::FinishCUDA();

	if(_keypoint_index.size() > 0)
	{
	    //put the descriptor back to the original order for keypoint list.
		for(int i = 0; i < _featureNum; ++i)
		{
			int index = _keypoint_index[i];
			memcpy(&_descriptor_buffer[index*128], &descriptor_buffer2[i*128], 128 * sizeof(float));
		}
	}

	if(ProgramCU::CheckErrorCUDA("PyramidCU::GetFeatureDescriptors")) SetFailStatus(); 
}

void PyramidCU::GenerateFeatureListTex() 
{

	vector<float> list;
	int idx = 0;
	const double twopi = 2.0*3.14159265358979323846;
	float sigma_half_step = powf(2.0f, 0.5f / param._dog_level_num);
	float octave_sigma = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f; 
	if(_down_sample_factor>0) octave_sigma *= float(1<<_down_sample_factor); 

	_keypoint_index.resize(0); // should already be 0
	for(int i = 0; i < _octave_num; i++, octave_sigma*= 2.0f)
	{
		for(int j = 0; j < param._dog_level_num; j++, idx++)
		{
			list.resize(0);
			float level_sigma = param.GetLevelSigma(j + param._level_min + 1) * octave_sigma;
			float sigma_min = level_sigma / sigma_half_step;
			float sigma_max = level_sigma * sigma_half_step;
			int fcount = 0 ;
			for(int k = 0; k < _featureNum; k++)
			{
				float * key = &_keypoint_buffer[k*4];
                float sigmak = key[2]; 
                //////////////////////////////////////
                if(IsUsingRectDescription()) sigmak = min(key[2], key[3]) / 12.0f; 

				if(   (sigmak >= sigma_min && sigmak < sigma_max)
					||(sigmak < sigma_min && i ==0 && j == 0)
					||(sigmak > sigma_max && i == _octave_num -1 && j == param._dog_level_num - 1))
				{
					//add this keypoint to the list
					list.push_back((key[0] - offset) / octave_sigma + 0.5f);
					list.push_back((key[1] - offset) / octave_sigma + 0.5f);
                    if(IsUsingRectDescription())
                    {
                        list.push_back(key[2] / octave_sigma);
                        list.push_back(key[3] / octave_sigma);
                    }else
                    {
					    list.push_back(key[2] / octave_sigma);
					    list.push_back((float)fmod(twopi-key[3], twopi));
                    }
					fcount ++;
					//save the index of keypoints
					_keypoint_index.push_back(k);
				}

			}

			_levelFeatureNum[idx] = fcount;
			if(fcount==0)continue;
			CuTexImage * ftex = _featureTex+idx;

			SetLevelFeatureNum(idx, fcount);
			ftex->CopyFromHost(&list[0]);
		}
	}

	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}

}

void PyramidCU::ReshapeFeatureListCPU() 
{
	int i, szmax =0, sz;
	int n = param._dog_level_num*_octave_num;
	for( i = 0; i < n; i++) 
	{
		sz = _levelFeatureNum[i];
		if(sz > szmax ) szmax = sz;
	}
	siftgpu::SiftReserveHost(uint64_t(szmax)*16*sizeof(float));
	std::unique_ptr<float[]> buffer_owner(new float[size_t(szmax)*16]);
    float* buffer=buffer_owner.get();
	float * buffer1 = buffer;
	float * buffer2 = buffer + szmax*4;



	_featureNum = 0;

#ifdef NO_DUPLICATE_DOWNLOAD
	const double twopi = 2.0*3.14159265358979323846;
	_keypoint_buffer.resize(0);
	float os = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	if(_down_sample_factor>0) os *= float(1<<_down_sample_factor); 
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f;
#endif


	for(i = 0; i < n; i++)
	{
		if(_levelFeatureNum[i]==0)continue;

		_featureTex[i].CopyToHost(buffer1);

		int fcount = siftgpu::ReshapePackedOrientations(
            buffer1, _levelFeatureNum[i], buffer2);
		//texture size
		SetLevelFeatureNum(i, fcount);
		_featureTex[i].CopyFromHost(buffer2);
		
		if(fcount == 0) continue;

#ifdef NO_DUPLICATE_DOWNLOAD
		float oss = os * (1 << (i / param._dog_level_num));
		siftgpu::SiftResizeHostVector(_keypoint_buffer,(uint64_t(_featureNum)+fcount)*4);
		float* ds = &_keypoint_buffer[_featureNum * 4];
		float* fs = buffer2;
		for(int k = 0;  k < fcount; k++, ds+=4, fs+=4)
		{
			ds[0] = oss*(fs[0]-0.5f) + offset;	//x
			ds[1] = oss*(fs[1]-0.5f) + offset;	//y
			ds[2] = oss*fs[2];  //scale
			ds[3] = (float)fmod(twopi-fs[3], twopi);	//orientation, mirrored
		}
#endif
		_featureNum += fcount;
	}

	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features MO:\t"<<_featureNum<<endl;
	}
}

void PyramidCU::GenerateFeatureDisplayVBO() 
{
	//it is weried that this part is very slow.
	//use a big VBO to save all the SIFT box vertices
	int nvbo = _octave_num * param._dog_level_num;
	if(_featureDisplayVBO==NULL)
	{
		//initialize the vbos
		_featureDisplayVBO = new GLuint[nvbo];
		_featurePointVBO = new GLuint[nvbo];
		glGenBuffers(nvbo, _featureDisplayVBO);	
		glGenBuffers(nvbo, _featurePointVBO);
	}
	for(int i = 0; i < nvbo; i++)
	{
		if(_levelFeatureNum[i]<=0)continue;
		CuTexImage * ftex  = _featureTex + i;
		CuTexImage texPBO1( _levelFeatureNum[i]* 10, 1, 4, _featureDisplayVBO[i]);
		CuTexImage texPBO2(_levelFeatureNum[i], 1, 4, _featurePointVBO[i]);
		ProgramCU::DisplayKeyBox(ftex, &texPBO1);
		ProgramCU::DisplayKeyPoint(ftex, &texPBO2);	
	}
}

void PyramidCU::DestroySharedData() 
{
	//histogram reduction
	if(_histoPyramidTex)
	{
		delete[]	_histoPyramidTex;
		_hpLevelNum = 0;
		_histoPyramidTex = NULL;
	}
	//descriptor storage shared by all levels
	if(_descriptorTex)
	{
		delete _descriptorTex;
		_descriptorTex = NULL;
	}
	//cpu reduction buffer.
	if(_histo_buffer)
	{
		delete[] _histo_buffer;
		_histo_buffer = 0;
	}
}

void PyramidCU::DestroyPerLevelData() 
{
	//integers vector to store the feature numbers.
	if(_levelFeatureNum)
	{
		delete [] _levelFeatureNum;
		_levelFeatureNum = NULL;
	}
	//texture used to store features
	if(	_featureTex)
	{
		delete [] _featureTex;
		_featureTex =	NULL;
	}
	//texture used for multi-orientation 
	if(_orientationTex)
	{
		delete [] _orientationTex;
		_orientationTex = NULL;
	}
	int no = _octave_num* param._dog_level_num;

	//two sets of vbos used to display the features
	if(_featureDisplayVBO)
	{
		glDeleteBuffers(no, _featureDisplayVBO);
		delete [] _featureDisplayVBO;
		_featureDisplayVBO = NULL;
	}
	if( _featurePointVBO)
	{
		glDeleteBuffers(no, _featurePointVBO);
		delete [] _featurePointVBO;
		_featurePointVBO = NULL;
	}
}

void PyramidCU::DestroyPyramidData()
{
	delete _denseKey;
	_denseKey = NULL;
	if(_allPyramid)
	{
		delete [] _allPyramid;
		_allPyramid = NULL;
	}
}

void PyramidCU::DownloadKeypoints() 
{
	const double twopi = 2.0*3.14159265358979323846;
	int idx = 0;
	float * buffer = &_keypoint_buffer[0];
	vector<float> keypoint_buffer2;
	//use a different keypoint buffer when processing with an exisint features list
	//without orientation information. 
	if(_keypoint_index.size() > 0)
	{
		keypoint_buffer2.resize(_keypoint_buffer.size());
		buffer = &keypoint_buffer2[0];
	}
	float * p = buffer, *ps;
	CuTexImage * ftex = _featureTex;
	/////////////////////
	float os = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	if(_down_sample_factor>0) os *= float(1<<_down_sample_factor); 
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f;
	/////////////////////
	for(int i = 0; i < _octave_num; i++, os *= 2.0f)
	{
	
		for(int j = 0; j  < param._dog_level_num; j++, idx++, ftex++)
		{

			if(_levelFeatureNum[idx]>0)
			{	
				ftex->CopyToHost(ps = p);
				for(int k = 0;  k < _levelFeatureNum[idx]; k++, ps+=4)
				{
					ps[0] = os*(ps[0]-0.5f) + offset;	//x
					ps[1] = os*(ps[1]-0.5f) + offset;	//y
					ps[2] = os*ps[2]; 
					ps[3] = (float)fmod(twopi-ps[3], twopi);	//orientation, mirrored
				}
				p+= 4* _levelFeatureNum[idx];
			}
		}
	}

	//put the feature into their original order for existing keypoint 
	if(_keypoint_index.size() > 0)
	{
		for(int i = 0; i < _featureNum; ++i)
		{
			int index = _keypoint_index[i];
			memcpy(&_keypoint_buffer[index*4], &keypoint_buffer2[i*4], 4 * sizeof(float));
		}
	}
}

void PyramidCU::GenerateFeatureListCPU()
{
	//no cpu version provided
	GenerateFeatureList();
}

void PyramidCU::GenerateFeatureList(int i, int j, int reduction_count, vector<int>& hbuffer)
{
    int fcount = 0, idx = i * param._dog_level_num  + j;
	int hist_level_num = _hpLevelNum - _pyramid_octave_first /2; 
	int ii, k, len; 

	CuTexImage * htex, * ftex, * tex, *got;
	ftex = _featureTex + idx;
	htex = _histoPyramidTex + hist_level_num -1;
	tex = RegenerateDenseKey(_octave_min + i, 2 + j);
	got = GetBaseLevel(_octave_min + i, DATA_GRAD) + 2 + j;

	ProgramCU::InitHistogram(tex, htex);

	for(k = 0; k < reduction_count - 1; k++, htex--)
	{
		ProgramCU::ReduceHistogram(htex, htex -1);	
	}
	
	//htex has the row reduction result
	len = htex->GetImgHeight() * 4;
	siftgpu::SiftResizeHostVector(hbuffer,len);
	ProgramCU::FinishCUDA();
	htex->CopyToHost(&hbuffer[0]);
	
    ////TO DO: track the error found here..
	for(ii = 0; ii < len; ++ii)     {if(!(hbuffer[ii]>= 0)) hbuffer[ii] = 0; }//?
	
    
    for(ii = 0; ii < len; ++ii)		fcount += hbuffer[ii];
	SetLevelFeatureNum(idx, fcount);
	
    //build the feature list
	if(fcount > 0)
	{
		_featureNum += fcount;
		siftgpu::SiftResizeHostVector(_keypoint_buffer,uint64_t(fcount)*4);
		//vector<int> ikbuf(fcount*4);
		int* ibuf = (int*) (&_keypoint_buffer[0]);

		for(ii = 0; ii < len; ++ii)
		{
			int x = ii%4, y = ii / 4;
			for(int jj = 0 ; jj < hbuffer[ii]; ++jj, ibuf+=4)
			{
				ibuf[0] = x; ibuf[1] = y; ibuf[2] = jj; ibuf[3] = 0;
			}
		}
		_featureTex[idx].CopyFromHost(&_keypoint_buffer[0]);
	
		////////////////////////////////////////////
		ProgramCU::GenerateList(_featureTex + idx, ++htex);
		for(k = 2; k < reduction_count; k++)
		{
			ProgramCU::GenerateList(_featureTex + idx, ++htex);
		}
	}
}

void PyramidCU::GenerateFeatureList()
{
	double t1, t2; 
	int ocount = 0, reduction_count;
    int reverse = (GlobalUtil::_TruncateMethod == 1);

	vector<int> hbuffer;
	_featureNum = 0;

	//for(int i = 0, idx = 0; i < _octave_num; i++)
    FOR_EACH_OCTAVE(i, reverse)
	{
        CuTexImage* tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + 2;
		reduction_count = FitHistogramPyramid(tex);

		if(GlobalUtil::_timingO)
		{
			t1 = CLOCK(); 
			ocount = 0;
			std::cout<<"#"<<i+_octave_min + _down_sample_factor<<":\t";
		}
		//for(int j = 0; j < param._dog_level_num; j++, idx++)
        FOR_EACH_LEVEL(j, reverse)
		{
            // (mgprt 20/06/2018) _levelFeatureNum can still contain old
            // values for these levels, so if we do not reset them the sum
            // of level features will not match the absolte number of features.
            if (GlobalUtil::_TruncateMethod && GlobalUtil::_FeatureCountThreshold > 0 && _featureNum > GlobalUtil::_FeatureCountThreshold) {
                int idx = i * param._dog_level_num + j;
                _levelFeatureNum[idx] = 0;
                continue;
            }

	        GenerateFeatureList(i, j, reduction_count, hbuffer);

			/////////////////////////////
			if(GlobalUtil::_timingO)
			{
                int idx = i * param._dog_level_num + j;
				ocount += _levelFeatureNum[idx];
				std::cout<< _levelFeatureNum[idx] <<"\t";
			}
		}
		if(GlobalUtil::_timingO)
		{	
			t2 = CLOCK(); 
			std::cout << "| \t" << int(ocount) << " :\t(" << (t2 - t1) << ")\n";
		}
	}
	if(GlobalUtil::_timingS)ProgramCU::FinishCUDA();

	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}

	if(ProgramCU::CheckErrorCUDA("PyramidCU::GenerateFeatureList")) SetFailStatus();
}

GLTexImage* PyramidCU::GetLevelTexture(int octave, int level)
{
	return GetLevelTexture(octave, level, DATA_GAUSSIAN);
}

GLTexImage* PyramidCU::ConvertTexCU2GL(CuTexImage* tex, int dataName, CuTexImage* dog)
{
	
	GLenum format = GL_LUMINANCE;
	int convert_done = 1;
    if(_bufferPBO == 0) glGenBuffers(1, &_bufferPBO);
    if(_bufferTEX == NULL) _bufferTEX = new GLTexImage;
	switch(dataName)
	{
	case DATA_GAUSSIAN:
		{
			convert_done = tex->CopyToPBO(_bufferPBO);
			break;
		}
	case DATA_DOG:
		{
			CuTexImage texPBO(tex->GetImgWidth(), tex->GetImgHeight(), 1, _bufferPBO);
			if(texPBO._cuData == 0 || tex->_cuData == NULL) convert_done = 0;
			else ProgramCU::DisplayConvertDOG(tex, &texPBO);
			break;
		}
	case DATA_GRAD:
		{
			CuTexImage texPBO(tex->GetImgWidth(), tex->GetImgHeight(), 1, _bufferPBO);
			if(texPBO._cuData == 0 || tex->_cuData == NULL) convert_done = 0;
			else ProgramCU::DisplayConvertGRD(tex, &texPBO);
			break;
		}
	case DATA_KEYPOINT:
		{
			format = GL_RGBA;
			CuTexImage texPBO(tex->GetImgWidth(), tex->GetImgHeight(), 4, _bufferPBO);
			if(texPBO._cuData == 0 || tex->_cuData == NULL) convert_done = 0;
			else ProgramCU::DisplayConvertKEY(tex, dog, &texPBO);
			break;
		}
	default:
			convert_done = 0;
			break;
	}

	if(convert_done)
	{
		_bufferTEX->InitTexture(max(_bufferTEX->GetTexWidth(), tex->GetImgWidth()), max(_bufferTEX->GetTexHeight(), tex->GetImgHeight()));
		_bufferTEX->CopyFromPBO(_bufferPBO, tex->GetImgWidth(), tex->GetImgHeight(), format);
	}else
	{
		_bufferTEX->SetImageSize(0, 0);
	}

	return _bufferTEX;
}

GLTexImage* PyramidCU::GetLevelTexture(int octave, int level, int dataName) 
{
	if(dataName == DATA_KEYPOINT && level - param._level_min > 1 &&
       level - param._level_min < param._level_num - 1)
    {
        // Display is a late dense-key consumer too. Do not infer the DoG
        // address from the reusable plane's unrelated storage address.
        CuTexImage* key = RegenerateDenseKey(octave, level - param._level_min);
        return ConvertTexCU2GL(key, dataName,
            GetBaseLevel(octave, DATA_DOG) + (level - param._level_min));
    }
	CuTexImage* tex = GetBaseLevel(octave, dataName) + (level - param._level_min);
	//CuTexImage* gus = GetBaseLevel(octave, DATA_GAUSSIAN) + (level - param._level_min); 
	return ConvertTexCU2GL(tex, dataName);
}

void PyramidCU::ConvertInputToCU(GLTexInput* input)
{
	int ws = input->GetImgWidth(), hs = input->GetImgHeight();
    _inputTex->_exactDomain = _exact_domain;
	//copy the input image to pixel buffer object
    if(input->_pixel_data)
    {
        _inputTex->InitTexture(ws, hs, 1);
        _inputTex->CopyFromHost(input->_pixel_data); 
    }else
    {
        if(_bufferPBO == 0) glGenBuffers(1, &_bufferPBO);
        if(input->_rgb_converted && input->CopyToPBO(_bufferPBO, ws, hs, GL_LUMINANCE))
        {
		    _inputTex->InitTexture(ws, hs, 1);
            _inputTex->CopyFromPBO(ws, hs, _bufferPBO); 
        }else if(input->CopyToPBO(_bufferPBO, ws, hs))
	    {
		    CuTexImage texPBO(ws, hs, 4, _bufferPBO);
		    _inputTex->InitTexture(ws, hs, 1);
		    ProgramCU::ReduceToSingleChannel(_inputTex, &texPBO, !input->_rgb_converted);
	    }else
	    {
		    std::cerr<< "Unable To Convert Intput\n";
	    }
    }
}

siftgpu::StreamPyramidResult PyramidCU::BuildGaussianPyramidStreamed(
    SiftParam& parameters,
    siftgpu::StreamedPlane& input,
    int first_octave,
    int octave_count,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  const bool exact = input.Width() % 4 != 0;
  if (input.Channels() != 1 || !input.Complete() ||
      (first_octave != 0 && first_octave != -1) || octave_count < 1 ||
      octave_count > 32 || parameters._level_num < 2 ||
      parameters._level_num > 64 ||
      int64_t(parameters._level_max) - parameters._level_min + 1 !=
          parameters._level_num ||
      parameters._sigma_num != parameters._level_num - 1 ||
      !parameters._sigma || parameters._level_ds < parameters._level_min ||
      parameters._level_ds > parameters._level_max)
    throw std::invalid_argument(
        "SIFT partial Gaussian phase requires parsed levels, octave 0 or "
        "-1, at most 32 octaves/64 levels");
  const float initial_sigma = parameters.GetInitialSmoothSigma(first_octave);
  auto validate_filter = [&](int w, int h, float sigma) {
    if (!std::isfinite(sigma) || sigma <= 0 ||
        !std::isfinite(GlobalUtil::_FilterWidthFactor) ||
        GlobalUtil::_FilterWidthFactor <= 0 ||
        double(sigma) * GlobalUtil::_FilterWidthFactor >
            (std::numeric_limits<int>::max() - 1) / 2)
      throw std::invalid_argument("SIFT partial Gaussian sigma");
    float coefficients[33];
    int fw;
    ProgramCU::CreateFilterKernel(sigma, coefficients, fw);
    StripePlan plan(w, h, fw, options.gpu_bytes, options.host_bytes);
  };
  // Validate every original domain and minimum live payload before creating
  // scratch or launching CUDA. Legacy rounded widths use the unrounded base
  // width at each shift, exactly as ResizePyramid, not the preceding texture.
  int base_w = input.Width(), base_h = input.Height();
  if (first_octave == -1) {
    if (base_w >= (1 << 21) || base_h >= (1 << 21))
      throw std::invalid_argument("SIFT partial Gaussian upsample range");
    base_w *= 2;
    base_h *= 2;
    SamplePlan plan(input.Width(),
                    input.Height(),
                    base_w,
                    base_h,
                    true,
                    exact,
                    options.gpu_bytes,
                    options.host_bytes);
  }
  int previous_w = input.Width(), previous_h = input.Height();
  int w = base_w, h = base_h;
  for (int octave = 0; octave < octave_count; ++octave) {
    const int domain_w = LevelWidth(w, exact);
    if (h <= 0 || domain_w <= 0)
      throw std::invalid_argument("SIFT partial Gaussian empty legacy octave");
    if (octave) {
      SamplePlan plan(previous_w,
                      previous_h,
                      domain_w,
                      h,
                      false,
                      exact,
                      options.gpu_bytes,
                      options.host_bytes);
    }
    if (!octave)
      validate_filter(domain_w, h, initial_sigma);
    else if (parameters._sigma_skip1 > 0)
      validate_filter(domain_w, h, parameters._sigma_skip1);
    for (int level = 0; level < parameters._sigma_num; ++level)
      validate_filter(domain_w, h, parameters._sigma[level]);
    previous_w = domain_w;
    previous_h = h;
    w >>= 1;
    h >>= 1;
    if (exact) {
      w = std::max(1, w);
      h = std::max(1, h);
    }
  }
  CheckStreamCancellation(options.cancelled);
  StreamPyramidResult result;
  result.first_octave = first_octave;
  result.exact_domain = exact;
  result.level_min = parameters._level_min;
  result.level_num = parameters._level_num;
  uint64_t retained = input.Bytes();
  auto collect = [&](const StreamFilterResult& stage, uint64_t extra) {
    auto& a = result.stats;
    const auto& b = stage.stats;
    a.gpu_staging_bytes = std::max(a.gpu_staging_bytes, b.gpu_staging_bytes);
    a.host_staging_bytes = std::max(a.host_staging_bytes, b.host_staging_bytes);
    a.scratch_bytes =
        std::max(a.scratch_bytes, retained + extra + b.scratch_bytes);
    a.file_read_bytes += b.file_read_bytes;
    a.file_written_bytes += b.file_written_bytes;
    a.h2d_bytes += b.h2d_bytes;
    a.d2h_bytes += b.d2h_bytes;
    a.file_seconds += b.file_seconds;
    a.h2d_seconds += b.h2d_seconds;
    a.d2h_seconds += b.d2h_seconds;
    a.kernel_seconds += b.kernel_seconds;
  };
  const auto start = std::chrono::steady_clock::now();
  w = base_w;
  h = base_h;
  for (int octave = 0; octave < octave_count; ++octave) {
    const int domain_w = LevelWidth(w, exact);
    std::unique_ptr<StreamedPlane> seed;
    if (octave || first_octave == -1) {
      StreamedPlane& source =
          octave ? *result.levels[(octave - 1) * parameters._level_num +
                                  parameters._level_ds - parameters._level_min]
                 : input;
      auto sampled = ProgramCU::SampleImageStreamed(
          source, domain_w, h, !octave, exact, options);
      collect(sampled, 0);
      seed = std::move(sampled.output);
    }
    if (!octave || parameters._sigma_skip1 > 0) {
      auto filtered = ProgramCU::FilterImageStreamed(
          seed ? *seed : input,
          octave ? parameters._sigma_skip1 : initial_sigma,
          options);
      collect(filtered, seed ? seed->Bytes() : 0);
      seed = std::move(filtered.output);
    }
    retained += seed->Bytes();
    result.levels.push_back(std::move(seed));
    for (int level = 0; level < parameters._sigma_num; ++level) {
      auto filtered = ProgramCU::FilterImageStreamed(
          *result.levels.back(), parameters._sigma[level], options);
      collect(filtered, 0);
      retained += filtered.output->Bytes();
      result.levels.push_back(std::move(filtered.output));
    }
    w >>= 1;
    h >>= 1;
    if (exact) {
      w = std::max(1, w);
      h = std::max(1, h);
    }
  }
  CheckStreamCancellation(options.cancelled);
  result.stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

siftgpu::StreamPyramidResult PyramidCU::BuildPyramidStreamed(
    SiftParam& parameters,
    siftgpu::StreamedPlane& input,
    int first_octave,
    int octave_count,
    const siftgpu::StreamFilterOptions& options) {
  const auto start = std::chrono::steady_clock::now();
  // Validate the largest DoG row/halo payload before Gaussian I/O or CUDA
  // work. Subsequent octave domains never exceed this domain; their tiny
  // heights reduce the minimum halo requirement rather than increasing it.
  const int64_t factor = first_octave == -1 ? 2 : 1;
  if (int64_t(input.Width()) * factor >= (1 << 23) ||
      int64_t(input.Height()) * factor > INT32_MAX)
    throw std::invalid_argument("SIFT streamed pyramid dimensions");
  siftgpu::DogPlan minimum(int(input.Width() * factor),
                           int(input.Height() * factor),
                           options.gpu_bytes,
                           options.host_bytes);
  auto result = BuildGaussianPyramidStreamed(
      parameters, input, first_octave, octave_count, options);
  uint64_t retained = input.Bytes();
  for (const auto& plane : result.levels) retained += plane->Bytes();
  for (int octave = 0; octave < octave_count; ++octave) {
    for (int level = 1; level < result.level_num; ++level) {
      const size_t index = size_t(octave) * result.level_num + level;
      auto stage = ProgramCU::ComputeDOGStreamed(
          *result.levels[index], *result.levels[index - 1], options);
      auto& a = result.stats;
      const auto& b = stage.stats;
      a.gpu_staging_bytes = std::max(a.gpu_staging_bytes, b.gpu_staging_bytes);
      a.host_staging_bytes =
          std::max(a.host_staging_bytes, b.host_staging_bytes);
      retained += b.scratch_bytes;
      a.scratch_bytes = std::max(a.scratch_bytes, retained);
      a.file_read_bytes += b.file_read_bytes;
      a.file_written_bytes += b.file_written_bytes;
      a.h2d_bytes += b.h2d_bytes;
      a.d2h_bytes += b.d2h_bytes;
      a.file_seconds += b.file_seconds;
      a.h2d_seconds += b.h2d_seconds;
      a.d2h_seconds += b.d2h_seconds;
      a.kernel_seconds += b.kernel_seconds;
      result.dogs.push_back(std::move(stage.dog));
      result.gradients.push_back(std::move(stage.gradient));
    }
  }
  siftgpu::CheckStreamCancellation(options.cancelled);
  result.stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

siftgpu::StreamDetectionPyramidResult PyramidCU::BuildDetectionPyramidStreamed(
    SiftParam& parameters,
    siftgpu::StreamedPlane& input,
    int first_octave,
    int octave_count,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  const auto start = std::chrono::steady_clock::now();
  if (parameters._dog_level_num < 1 || parameters._dog_level_num > 61 ||
      parameters._level_num < parameters._dog_level_num + 3)
    throw std::invalid_argument(
        "SIFT streamed detection requires DoG triplets");
  const int64_t factor = first_octave == -1 ? 2 : 1;
  if (int64_t(input.Width()) * factor >= (1 << 23) ||
      int64_t(input.Height()) * factor > INT32_MAX)
    throw std::invalid_argument("SIFT streamed detection dimensions");
  KeyPlan minimum(int(input.Width() * factor),
                  int(input.Height() * factor),
                  options.gpu_bytes,
                  options.host_bytes);
  StreamDetectionPyramidResult result;
  result.pyramid = BuildPyramidStreamed(
      parameters, input, first_octave, octave_count, options);
  auto& pyramid = result.pyramid;
  uint64_t retained = input.Bytes();
  for (const auto& plane : pyramid.levels) retained += plane->Bytes();
  for (const auto& plane : pyramid.dogs) retained += plane->Bytes();
  for (const auto& plane : pyramid.gradients) retained += plane->Bytes();
  const bool reverse = GlobalUtil::_TruncateMethod == 1;
  // Exactly FOR_EACH_OCTAVE / FOR_EACH_LEVEL, without constructing any lists
  // or applying an approximate tile-local/global feature-count limiter.
  for (int oi = 0; oi < octave_count; ++oi) {
    const int octave = reverse ? octave_count - 1 - oi : oi;
    for (int li = 0; li < parameters._dog_level_num; ++li) {
      const int level = reverse ? parameters._dog_level_num - 1 - li : li;
      const size_t previous = size_t(octave) * (pyramid.level_num - 1) + level;
      auto stage = ProgramCU::ComputeKEYStreamed(*pyramid.dogs[previous],
                                                 *pyramid.dogs[previous + 1],
                                                 *pyramid.dogs[previous + 2],
                                                 parameters._dog_threshold,
                                                 parameters._edge_threshold,
                                                 pyramid.exact_domain,
                                                 options);
      auto& a = pyramid.stats;
      const auto& b = stage.stats;
      a.gpu_staging_bytes = std::max(a.gpu_staging_bytes, b.gpu_staging_bytes);
      a.host_staging_bytes =
          std::max(a.host_staging_bytes, b.host_staging_bytes);
      retained += b.scratch_bytes;
      a.scratch_bytes = std::max(a.scratch_bytes, retained);
      a.file_read_bytes += b.file_read_bytes;
      a.file_written_bytes += b.file_written_bytes;
      a.h2d_bytes += b.h2d_bytes;
      a.d2h_bytes += b.d2h_bytes;
      a.file_seconds += b.file_seconds;
      a.h2d_seconds += b.h2d_seconds;
      a.d2h_seconds += b.d2h_seconds;
      a.kernel_seconds += b.kernel_seconds;
      result.detections.push_back(
          {first_octave + octave, 2 + level, std::move(stage)});
    }
  }
  CheckStreamCancellation(options.cancelled);
  pyramid.stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

siftgpu::StreamFeatureListsResult PyramidCU::GenerateFeatureListsStreamed(
    siftgpu::StreamDetectionPyramidResult& detection,
    int truncate_method,
    int threshold,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  CheckStreamCancellation(options.cancelled);
  ListPlan minimum(options.gpu_bytes, options.host_bytes);
  const int dog_levels = detection.pyramid.level_num - 3;
  if (dog_levels <= 0 || detection.detections.empty() ||
      detection.detections.size() % dog_levels)
    throw std::invalid_argument("SIFT streamed feature-list level metadata");
  const size_t octaves = detection.detections.size() / dog_levels;
  StreamFeatureListsResult result;
  result.levels.resize(detection.detections.size());
  result.level_counts.resize(detection.detections.size(), 0);
  const bool reverse = truncate_method == 1;
  int64_t generated = 0;
  for (size_t oi = 0; oi < octaves; ++oi) {
    const size_t octave = reverse ? octaves - 1 - oi : oi;
    for (int li = 0; li < dog_levels; ++li) {
      const int level = reverse ? dog_levels - 1 - li : li;
      const size_t index = octave * dog_levels + level;
      // Detection may have been built under a different traversal setting;
      // select by metadata without changing scientific feature order.
      StreamDetectionLevel* selected = nullptr;
      for (auto& record : detection.detections)
        if (record.octave == detection.pyramid.first_octave + int(octave) &&
            record.level == level + 2) {
          if (selected)
            throw std::invalid_argument("duplicate SIFT detection level");
          selected = &record;
        }
      if (!selected)
        throw std::invalid_argument("missing SIFT detection level");
      if (truncate_method && threshold > 0 && generated > threshold) continue;
      auto stage =
          ProgramCU::GenerateListStreamed(selected->detection, options);
      generated += stage.count;
      if (generated > INT32_MAX)
        throw std::overflow_error("SIFT streamed total feature count");
      result.level_counts[index] = stage.count;
      auto& a = result.stats;
      const auto& b = stage.stats;
      a.gpu_staging_bytes = std::max(a.gpu_staging_bytes, b.gpu_staging_bytes);
      a.host_staging_bytes =
          std::max(a.host_staging_bytes, b.host_staging_bytes);
      a.file_read_bytes += b.file_read_bytes;
      a.file_written_bytes += b.file_written_bytes;
      a.h2d_bytes += b.h2d_bytes;
      a.d2h_bytes += b.d2h_bytes;
      a.scratch_bytes += b.scratch_bytes;
      a.seconds += b.seconds;
      a.file_seconds += b.file_seconds;
      a.h2d_seconds += b.h2d_seconds;
      a.d2h_seconds += b.d2h_seconds;
      a.kernel_seconds += b.kernel_seconds;
      result.levels[index] = std::move(stage);
    }
  }
  result.count = LimitStreamedFeatureCounts(
      result.level_counts, threshold, truncate_method);
  for (size_t i = 0; i < result.levels.size(); ++i)
    if (!result.level_counts[i]) {
      result.levels[i].records.reset();
      result.levels[i].count = 0;
    }
  CheckStreamCancellation(options.cancelled);
  return result;
}

siftgpu::StreamFeatureListsResult PyramidCU::GetFeatureOrientationsStreamed(
    SiftParam& parameters,
    siftgpu::StreamDetectionPyramidResult& detection,
    siftgpu::StreamFeatureListsResult& lists,
    int truncate_method,
    int threshold,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  CheckStreamCancellation(options.cancelled);
  const auto start = std::chrono::steady_clock::now();
  auto& pyramid = detection.pyramid;
  const int dog_levels = parameters._dog_level_num;
  if (dog_levels <= 0 || pyramid.level_num != dog_levels + 3 ||
      pyramid.level_min != parameters._level_min || lists.levels.empty() ||
      lists.levels.size() % dog_levels ||
      lists.level_counts.size() != lists.levels.size() ||
      detection.detections.size() != lists.levels.size() ||
      pyramid.gradients.size() !=
          lists.levels.size() / dog_levels * (pyramid.level_num - 1))
    throw std::invalid_argument("SIFT streamed orientation level metadata");
  StreamFeatureListsResult result;
  result.levels.resize(lists.levels.size());
  result.level_counts.resize(lists.levels.size(), 0);
  const float sigma_step = powf(2.0f, 1.0f / dog_levels);
  uint64_t retained = 0;
  for (size_t index = 0; index < lists.levels.size(); ++index) {
    auto& source = lists.levels[index];
    if (source.count != lists.level_counts[index])
      throw std::invalid_argument("SIFT streamed orientation retained count");
    if (!source.count) continue;
    const int octave = int(index) / dog_levels;
    const int level = int(index) % dog_levels;
    StreamDetectionLevel* selected = nullptr;
    for (auto& record : detection.detections)
      if (record.octave == pyramid.first_octave + octave &&
          record.level == level + 2) {
        if (selected)
          throw std::invalid_argument("duplicate SIFT orientation KEY");
        selected = &record;
      }
    const size_t gradient_index =
        size_t(octave) * (pyramid.level_num - 1) + level;
    if (!selected || !selected->detection.key ||
        !pyramid.gradients[gradient_index])
      throw std::invalid_argument("missing SIFT orientation plane");
    const float sigma =
        parameters.GetLevelSigma(level + parameters._level_min + 1);
    auto stage = ProgramCU::ComputeOrientationStreamed(
        source,
        *pyramid.gradients[gradient_index],
        *selected->detection.key,
        sigma,
        sigma_step,
        options);
    auto& a = result.stats;
    const auto& b = stage.stats;
    a.gpu_staging_bytes = std::max(a.gpu_staging_bytes, b.gpu_staging_bytes);
    a.host_staging_bytes = std::max(a.host_staging_bytes, b.host_staging_bytes);
    a.file_read_bytes += b.file_read_bytes;
    a.file_written_bytes += b.file_written_bytes;
    a.h2d_bytes += b.h2d_bytes;
    a.d2h_bytes += b.d2h_bytes;
    a.scratch_bytes = std::max(a.scratch_bytes, retained + b.scratch_bytes);
    a.file_seconds += b.file_seconds;
    a.h2d_seconds += b.h2d_seconds;
    a.d2h_seconds += b.d2h_seconds;
    a.kernel_seconds += b.kernel_seconds;
    if (stage.records) retained += stage.records->Bytes();
    result.level_counts[index] = stage.count;
    result.levels[index] = std::move(stage);
  }
  // Original RunSIFT only performs its second limiter after multi-orientation
  // reshape; single/fixed orientation retains the first boundary unchanged.
  result.count = LimitStreamedFeatureCounts(
      result.level_counts,
      GlobalUtil::_MaxOrientation > 1 && !GlobalUtil::_FixedOrientation
          ? threshold
          : 0,
      truncate_method);
  for (size_t i = 0; i < result.levels.size(); ++i)
    if (!result.level_counts[i]) {
      result.levels[i].records.reset();
      result.levels[i].count = 0;
    }
  CheckStreamCancellation(options.cancelled);
  result.stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

siftgpu::StreamExtractionResult PyramidCU::GetFeatureDescriptorsStreamed(
    SiftParam& parameters,
    siftgpu::StreamDetectionPyramidResult& detection,
    siftgpu::StreamFeatureListsResult& orientations,
    const siftgpu::StreamFilterOptions& options) {
  using namespace siftgpu;
  CheckStreamCancellation(options.cancelled);
  const auto start = std::chrono::steady_clock::now();
  auto& pyramid = detection.pyramid;
  const int levels = parameters._dog_level_num;
  if (levels <= 0 || pyramid.level_num != levels + 3 ||
      pyramid.level_min != parameters._level_min ||
      orientations.levels.size() != orientations.level_counts.size() ||
      orientations.levels.size() % levels ||
      pyramid.gradients.size() !=
          orientations.levels.size() / levels * (pyramid.level_num - 1) ||
      pyramid.first_octave < -1 || pyramid.first_octave > 30 ||
      orientations.levels.size() / levels > size_t(31 - pyramid.first_octave))
    throw std::invalid_argument("SIFT streamed descriptor level metadata");
  StreamExtractionResult result;
  result.features.levels.resize(orientations.levels.size());
  result.features.level_counts = orientations.level_counts;
  result.descriptors.resize(orientations.levels.size());
  uint64_t retained = 0;
  int64_t total = 0;
  float os = pyramid.first_octave >= 0 ? float(1 << pyramid.first_octave)
                                       : 1.0f / (1 << (-pyramid.first_octave));
  const float offset = GlobalUtil::_LoweOrigin ? 0 : 0.5f;
  const double twopi = 2.0 * 3.14159265358979323846;
  for (size_t index = 0; index < orientations.levels.size(); ++index) {
    if (index && index % levels == 0) os *= 2.0f;
    auto& source = orientations.levels[index];
    if (source.count < 0 || source.count != orientations.level_counts[index])
      throw std::invalid_argument("SIFT streamed descriptor retained count");
    if (!source.count) continue;
    const size_t gradient_index =
        index / levels * (pyramid.level_num - 1) + index % levels;
    if (!pyramid.gradients[gradient_index])
      throw std::invalid_argument("missing SIFT descriptor gradient");
    auto stage = ProgramCU::ComputeDescriptorStreamed(
        source, *pyramid.gradients[gradient_index], options);
    auto& a = result.stats;
    const auto& b = stage.stats;
    a.gpu_staging_bytes = std::max(a.gpu_staging_bytes, b.gpu_staging_bytes);
    a.host_staging_bytes = std::max(a.host_staging_bytes, b.host_staging_bytes);
    a.file_read_bytes += b.file_read_bytes;
    a.file_written_bytes += b.file_written_bytes;
    a.h2d_bytes += b.h2d_bytes;
    a.d2h_bytes += b.d2h_bytes;
    a.file_seconds += b.file_seconds;
    a.h2d_seconds += b.h2d_seconds;
    a.d2h_seconds += b.d2h_seconds;
    a.kernel_seconds += b.kernel_seconds;
    retained += stage.records->Bytes();
    auto& converted = result.features.levels[index];
    converted.records = std::make_unique<StreamedPlane>(
        options.scratch_directory, 1, source.count, 4);
    converted.records->Reserve();
    retained += converted.records->Bytes();
    a.scratch_bytes = std::max(a.scratch_bytes, retained);
    const auto time = std::chrono::steady_clock::now();
    for (int row = 0; row < source.count; ++row) {
      float key[4];
      source.records->ReadRows(row, 1, key, options.cancelled);
      // Original DownloadKeypoints arithmetic, with no implicit image
      // downsampling in this native pipeline. Descriptor used unconverted xyso.
      key[0] = os * (key[0] - 0.5f) + offset;
      key[1] = os * (key[1] - 0.5f) + offset;
      key[2] = os * key[2];
      key[3] = (float)fmod(twopi - key[3], twopi);
      converted.records->WriteRows(row, 1, key, options.cancelled);
      a.file_read_bytes += 16;
      a.file_written_bytes += 16;
    }
    a.file_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - time)
            .count();
    converted.records->Seal();
    converted.count = source.count;
    total += source.count;
    if (total > INT32_MAX)
      throw std::overflow_error("SIFT streamed descriptor total overflow");
    result.descriptors[index] = std::move(stage);
  }
  if (total != orientations.count)
    throw std::invalid_argument("SIFT streamed descriptor total count");
  result.features.count = int(total);
  CheckStreamCancellation(options.cancelled);
  result.stats.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return result;
}

void PyramidCU::BuildPyramid(GLTexInput * input)
{

	USE_TIMING();

	int i, j;
	
	for ( i = _octave_min; i < _octave_min + _octave_num; i++)
	{

		float* filter_sigma = param._sigma;
		CuTexImage *tex = GetBaseLevel(i);
		CuTexImage *buf = FitDenseKey(i);
		j = param._level_min + 1;

		OCTAVE_START();

		if( i == _octave_min )
		{	
			ConvertInputToCU(input);

			if(i == 0)
			{
				ProgramCU::FilterImage(tex, _inputTex, buf, 
                    param.GetInitialSmoothSigma(_octave_min + _down_sample_factor));
			}else
			{
				if(i < 0)	ProgramCU::SampleImageU(tex, _inputTex, -i);			
				else		ProgramCU::SampleImageD(tex, _inputTex, i);
				ProgramCU::FilterImage(tex, tex, buf, 
                    param.GetInitialSmoothSigma(_octave_min + _down_sample_factor));
			}
		}else
		{
			ProgramCU::SampleImageD(tex, GetBaseLevel(i - 1) + param._level_ds - param._level_min); 
			if(param._sigma_skip1 > 0)
			{
				ProgramCU::FilterImage(tex, tex, buf, param._sigma_skip1);
			}
		}
		LEVEL_FINISH();
		for( ; j <=  param._level_max ; j++, tex++, filter_sigma++)
		{
			// filtering
			ProgramCU::FilterImage(tex + 1, tex, buf, *filter_sigma);
			LEVEL_FINISH();
		}
		OCTAVE_FINISH();
	}
	if(GlobalUtil::_timingS) ProgramCU::FinishCUDA();

	if(ProgramCU::CheckErrorCUDA("PyramidCU::BuildPyramid")) SetFailStatus();
}

void PyramidCU::DetectKeypointsEX()
{


	int i, j;
	double ts, t1, t2;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();

	for(i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		CuTexImage * gus = GetBaseLevel(i) + 1;
		CuTexImage * dog = GetBaseLevel(i, DATA_DOG) + 1;
		CuTexImage * got = GetBaseLevel(i, DATA_GRAD) + 1;
		//compute the gradient
		for(j = param._level_min +1; j <=  param._level_max ; j++, gus++, dog++, got++)
		{
			//input: gus and gus -1
			//output: gradient, dog, orientation
			ProgramCU::ComputeDOG(gus, dog, got);
		}
	}
	if(GlobalUtil::_timingS && GlobalUtil::_verbose)
	{
		ProgramCU::FinishCUDA();
		t1 = CLOCK();
	}

    // Dense keys are produced immediately before their consumers. All DoGs
    // and gradients remain immutable, including across the two global limits.

	if(GlobalUtil::_timingS)
	{
		ProgramCU::FinishCUDA();
		if(GlobalUtil::_verbose) 
		{	
			t2 = CLOCK();
			std::cout	<<"<Gradient, DOG  >\t"<<(t1-ts)<<"\n"
						<<"<Deferred Keys  >\t"<<(t2-t1)<<"\n";
		}				
	}
}

void PyramidCU::ComputeGradient() 
{

	int i, j;
	double ts, t1;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();

	for(i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		CuTexImage * gus = GetBaseLevel(i) +  1;
		CuTexImage * dog = GetBaseLevel(i, DATA_DOG) +  1;
		CuTexImage * got = GetBaseLevel(i, DATA_GRAD) +  1;

		//compute the gradient
		for(j = 0; j <  param._dog_level_num ; j++, gus++, dog++, got++)
		{
			ProgramCU::ComputeDOG(gus, dog, got);
		}
	}
	if(GlobalUtil::_timingS)
	{
		ProgramCU::FinishCUDA();
		if(GlobalUtil::_verbose)
		{
			t1 = CLOCK();
			std::cout	<<"<Gradient, DOG  >\t"<<(t1-ts)<<"\n";
		}
	}
}

int PyramidCU::FitHistogramPyramid(CuTexImage* tex)
{
    return TraverseHistogram(
        tex->GetImgWidth(), tex->GetImgHeight(), _exact_domain,
        _hpLevelNum - _pyramid_octave_first / 2,
        [&](int slot, int w, int h) {
            _histoPyramidTex[slot].InitTexture(w, h, 4);
        });
}

void PyramidCU::GetFeatureOrientations() 
{

	CuTexImage * ftex = _featureTex;
	int * count	 = _levelFeatureNum;
	float sigma, sigma_step = powf(2.0f, 1.0f/param._dog_level_num);

	for(int i = 0; i < _octave_num; i++)
	{
		CuTexImage* got = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;
		CuTexImage* key = FitDenseKey(i + _octave_min);

		for(int j = 0; j < param._dog_level_num; j++, ftex++, count++, got++)
		{
			if(*count<=0)continue;

			//if(ftex->GetImgWidth() < *count) ftex->InitTexture(*count, 1, 4);

            // Supplied/rectangular keys and non-subpixel orientation never
            // read the dense plane. Otherwise regenerate after global limiting.
            if(!_existing_keypoints && GlobalUtil::_SubpixelLocalization)
                key = RegenerateDenseKey(i + _octave_min, j + 2);
			sigma = param.GetLevelSigma(j+param._level_min+1);

			ProgramCU::ComputeOrientation(ftex, got, key, sigma, sigma_step, _existing_keypoints);		
		}
	}

	if(GlobalUtil::_timingS)ProgramCU::FinishCUDA();
	if(ProgramCU::CheckErrorCUDA("PyramidCU::GetFeatureOrientations")) SetFailStatus();

}

void PyramidCU::GetSimplifiedOrientation() 
{
	//no simplified orientation
	GetFeatureOrientations();
}

CuTexImage* PyramidCU::FitDenseKey(int octave)
{
    CuTexImage* base = GetBaseLevel(octave);
    _denseKey->SetImageSize(base->GetImgWidth(), base->GetImgHeight());
    return _denseKey;
}

CuTexImage* PyramidCU::RegenerateDenseKey(int octave, int level)
{
    CuTexImage* key = FitDenseKey(octave);
    ProgramCU::ComputeKEY(GetBaseLevel(octave, DATA_DOG) + level, key,
                          param._dog_threshold, param._edge_threshold);
    return key;
}

CuTexImage* PyramidCU::GetBaseLevel(int octave, int dataName)
{
	if(octave <_octave_min || octave > _octave_min + _octave_num) return NULL;
	int offset = (_pyramid_octave_first + octave - _octave_min) * param._level_num;
	int num = param._level_num * _pyramid_octave_num;
	if (dataName == DATA_ROT) dataName = DATA_GRAD;
	return _allPyramid + num * dataName + offset;
}

#endif
