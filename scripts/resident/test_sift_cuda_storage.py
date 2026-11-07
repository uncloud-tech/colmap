# SPDX-License-Identifier: BSD-3-Clause
"""Execute real SIFT CUDA host allocation code with test-only API doubles.

No kernels are emulated and this is not a numerical or GPU qualification.
"""

import pathlib
import re
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SIFT = ROOT / "src/thirdparty/SiftGPU"


class SiftCudaStorageTest(unittest.TestCase):
    def test_pyramid_storage_and_reuse(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            temp = pathlib.Path(directory)
            (temp / "GL").mkdir()
            (temp / "colmap/util").mkdir(parents=True)
            (temp / "GL/glew.h").write_text(r"""
#pragma once
#include <cstddef>
using GLuint = unsigned int;
using GLint = int;
using GLenum = unsigned int;
enum { GL_RGBA, GL_RGB, GL_BGR, GL_BGRA, GL_LUMINANCE,
       GL_LUMINANCE_ALPHA, GL_FLOAT, GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT,
       GL_PIXEL_PACK_BUFFER_ARB, GL_BUFFER_SIZE, GL_STATIC_DRAW_ARB };
inline void glBindBuffer(GLenum, GLuint) {}
inline void glBufferData(GLenum, size_t, const void*, GLenum) {}
inline void glGetBufferParameteriv(GLenum, GLenum, GLint* n) { *n = 0; }
inline void glDeleteBuffers(int, const GLuint*) {}
inline void glGenBuffers(int, GLuint*) {}
""")
            (temp / "colmap/util/cuda_to_hip.h").write_text(r"""
#pragma once
#include <cassert>
#include <climits>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
inline int array_allocations = 0, array_copies = 0, linear_allocations = 0;
inline std::map<void*, size_t> live;
inline bool reservation_only = false;
inline std::vector<size_t> requests;
using cudaError_t = int;
using cudaStream_t = size_t;
using cudaTextureObject_t = size_t;
struct cudaArray {};
struct cudaChannelFormatDesc { int x, y, z, w, f; };
struct int4 { int x, y, z, w; };
template<class T> cudaChannelFormatDesc cudaCreateChannelDesc() {
  return {32,32,32,32,0};
}
inline int cudaGetLastError() { return 0; }
inline int cudaDeviceSynchronize() { return 0; }
template<class T> T tex1Dfetch(cudaTextureObject_t h, int i) {
  return reinterpret_cast<T*>(h)[i];
}
struct cudaTextureDesc {};
struct cudaResourceDesc {
  int resType;
  struct { struct { void* devPtr; cudaChannelFormatDesc desc;
                   size_t sizeInBytes; } linear; } res;
};
enum { cudaSuccess, cudaResourceTypeLinear, cudaChannelFormatKindFloat,
       cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
       cudaMemcpyDeviceToDevice };
inline int cudaMalloc(void** p, size_t n) {
  requests.push_back(n);
  *p = std::calloc(1, reservation_only ? 1 : n); assert(*p); live[*p] = n;
  ++linear_allocations; return 0;
}
inline int cudaFree(void* p) {
  assert(live.erase(p) == 1); std::free(p); return 0;
}
inline int cudaMallocArray(
    cudaArray** p, const cudaChannelFormatDesc*, int, int) {
  ++array_allocations; *p = new cudaArray; return 0;
}
inline int cudaFreeArray(cudaArray* p) { delete p; return 0; }
inline int cudaMemcpy(void* d, const void* s, size_t n, int) {
  std::memcpy(d, s, n); return 0;
}
inline int cudaMemcpyAsync(
    void* d, const void* s, size_t n, int k, cudaStream_t) {
  return cudaMemcpy(d, s, n, k);
}
inline int cudaMemcpy2DToArray(
    cudaArray*, int, int, void*, size_t, size_t, int, int) {
  ++array_copies; return 0;
}
inline int cudaCreateTextureObject(
    cudaTextureObject_t* h, const cudaResourceDesc* r,
                                   const cudaTextureDesc*, void*) {
  assert(r->resType == cudaResourceTypeLinear);
  assert(live.at(r->res.linear.devPtr) == r->res.linear.sizeInBytes);
  *h = reinterpret_cast<size_t>(r->res.linear.devPtr); return 0;
}
inline int cudaDestroyTextureObject(cudaTextureObject_t) { return 0; }
""")
            (temp / "cuda_gl_interop.h").write_text(r"""
#pragma once
inline int cudaGLRegisterBufferObject(unsigned int) { return 0; }
inline int cudaGLUnregisterBufferObject(unsigned int) { return 0; }
inline int cudaGLMapBufferObject(void**, unsigned int) { return 0; }
inline int cudaGLUnmapBufferObject(unsigned int) { return 0; }
""")
            # Define only external class dependencies, using their real headers.
            # Generated declarations avoid duplicating the third-party ABI.
            globals_header = (SIFT / "GlobalUtil.h").read_text()
            globals_defs = "\n".join(
                f"{kind} GlobalParam::{name} = {{}};"
                for kind, name in re.findall(
                    r"static\s+(GLuint|int|float|const char\*)\s+(_\w+)\s*;",
                    globals_header,
                )
            )
            program_header = (SIFT / "ProgramCU.h").read_text()
            program_defs = []
            for result, name, args in re.findall(
                r"static\s+(void|int|siftgpu::Stream(?:Filter|Dog|Key|List|Descriptor)Result)\s+(\w+)\(([^;]*?)\);",
                program_header,
            ):
                if name == "GenerateListStreamed":
                    # Exercise real host batching with CPU API doubles.
                    continue
                if name == "FilterImage" and "sigma" not in args:
                    continue  # template declaration, not a linked host call
                args = re.sub(r"\s*=\s*\d+", "", args)
                body = "return 0;" if result == "int" else ""
                if result.startswith("siftgpu::Stream"):
                    # This resident ownership test must never pretend to run
                    # CUDA streaming. Supply link symbols, fail if dispatched.
                    body = (
                        'throw std::runtime_error("streaming requires '
                        'native CUDA test");'
                    )
                if name == "ComputeKEY":
                    body = "RecordDenseKey(dog, key, Tdog, Tedge);"
                if name == "ComputeOrientation":
                    body = (
                        "CheckOrientation(got, key, existing_keypoint); "
                        "if(GlobalParam::_MaxOrientation == 2) { "
                        "vector<float> oriented(list->GetImgWidth()*4, 1.0f); "
                        "unsigned int packed = 32768u << 16; "
                        "for(int n=0; n<list->GetImgWidth(); ++n) "
                        "memcpy(&oriented[n*4+3], &packed, 4); "
                        "list->CopyFromHost(oriented.data()); }"
                    )
                if name in {"InitHistogram", "ReduceHistogram"}:
                    target = "hist" if name == "InitHistogram" else "hist2"
                    prefix = (
                        "histogram_slot = test_pyramid->GetHistLevelNum() - "
                        "test_pyramid->_pyramid_octave_first / 2 - 1; "
                        "assert(hist->GetImgWidth() == "
                        "(key->GetImgWidth()+3)/4); "
                        "assert(hist->GetImgHeight() == key->GetImgHeight()); "
                        "assert(key == last_key); "
                        "list_order.push_back(last_level); "
                        if name == "InitHistogram"
                        else "--histogram_slot; "
                        "assert(hist2->GetImgWidth() == "
                        "(hist1->GetImgWidth()+3)/4); "
                        "assert(hist2->GetImgHeight() == "
                        "hist1->GetImgHeight()); "
                    )
                    body = prefix + (
                        f"CheckHistogram({target}); "
                        f"vector<int> counts({target}->GetImgWidth() * "
                        f"{target}->GetImgHeight() * 4, 0); "
                        f"assert({target}->GetDataSize() >= "
                        "counts.size() * sizeof(int)); "
                        f"histogram_bytes[{target}] = {target}->GetDataSize(); "
                        "counts[0] = test_features; "
                        f"{target}->CopyFromHost(counts.data());"
                    )
                program_defs.append(
                    f"{result} ProgramCU::{name}({args}) {{{body}}}"
                )
            # Unused GL display and base-class entrypoints are inert. They are
            # linked because PyramidCU has a vtable, not exercised as a backend.
            gl_defs = []
            header = (
                (SIFT / "GLTexImage.h")
                .read_text()
                .split("class GLTexInput:")[0]
            )
            for name, args in re.findall(
                r"virtual void\s+(\w+)\(([^;]*?)\)\s*;", header
            ):
                args = re.sub(r"\s*=\s*\d+", "", args)
                gl_defs.append(f"void GLTexImage::{name}({args}) {{}}")
            # Exercise the unchanged real global limiter at both call sites.
            base_source = (SIFT / "SiftPyramid.cpp").read_text()
            limiter = base_source.split(
                "void SiftPyramid::LimitFeatureCount(", 1
            )[1]
            limiter = (
                "void SiftPyramid::LimitFeatureCount("
                + limiter.split("void SiftPyramid::PrepareBuffer()", 1)[0]
            )
            cuda_source = (SIFT / "ProgramCU.cu").read_text()
            list_kernel = cuda_source.split(
                "void __global__ ListGen_Kernel", 1
            )[1]
            list_kernel = (
                "void ListGen_Kernel" + list_kernel.split("//input list", 1)[0]
            )
            list_host = cuda_source.split(
                "siftgpu::StreamListResult ProgramCU::GenerateListStreamed", 1
            )[1]
            list_host = (
                "siftgpu::StreamListResult ProgramCU::GenerateListStreamed"
                + list_host.split(
                    "void __global__ ComputeOrientation_Kernel", 1
                )[0]
            )
            # CPU loop executes the ORIGINAL kernel body, not a numerical GPU
            # qualification. Host cursor, I/O, staging and failure paths
            # are real.
            list_host = re.sub(
                r"ListGen_Kernel<<<.*?>>>",
                "RunListKernel",
                list_host,
                flags=re.S,
            )
            list_cpu = (
                r"""
#define IMUL(X,Y) ((X)*(Y))
struct { int x = 0; } blockIdx, blockDim, threadIdx;
static cudaTextureDesc texDataDesc;
"""
                + list_kernel
                + r"""
void RunListKernel(cudaTextureObject_t list, cudaTextureObject_t hist,
                   int4* output, int n, int width, bool packed) {
  blockDim.x = 1;
  threadIdx.x = 0;
  for (blockIdx.x = 0; blockIdx.x < n; ++blockIdx.x)
    ListGen_Kernel(list, hist, output, n, width, packed);
}
"""
                + list_host
            )
            source = (
                r"""
#include "GL/glew.h"
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;
#include "GlobalUtil.h"
#include "GLTexImage.h"
#include "CuTexImage.h"
#include "SiftGPU.h"
#include "SiftPyramid.h"
#include "ProgramCU.h"
#include "PyramidCU.h"
int test_features = 0;
map<CuTexImage*, size_t> histogram_bytes;
PyramidCU* test_pyramid = nullptr;
class SuppliedKeyPyramid : public PyramidCU {
 public:
  using PyramidCU::PyramidCU;
  void ExistingKeys(int flags) { _existing_keypoints = flags; }
};
int histogram_slot = 0;
int key_executions = 0;
CuTexImage* last_key = nullptr;
int last_level = -1;
vector<int> list_order;
map<PyramidCU*, CuTexImage*> dense_owners;
void RecordDenseKey(CuTexImage* dog, CuTexImage* key, float td, float te) {
  ++key_executions;
  assert(td == 0.02f && te == 10.0f);
  int level = -1;
  for (int o = 0; o < test_pyramid->_octave_num; ++o)
    for (int j = 0; j < 3; ++j)
      if (dog == test_pyramid->GetBaseLevel(test_pyramid->_octave_min + o,
                    SiftPyramid::DATA_DOG) + j + 2) level = o * 3 + j;
  assert(level >= 0);
  auto inserted = dense_owners.emplace(test_pyramid, key);
  assert(inserted.first->second == key);
  assert(key->GetImgWidth() == dog->GetImgWidth());
  assert(key->GetImgHeight() == dog->GetImgHeight());
  vector<float> tag(key->GetImgWidth() * key->GetImgHeight() * 4, float(level));
  key->CopyFromHost(tag.data());
  last_key = key; last_level = level;
}
void CheckOrientation(CuTexImage* got, CuTexImage* key, int existing) {
  if (existing || !GlobalParam::_SubpixelLocalization) return;
  int level = -1;
  for (int o = 0; o < test_pyramid->_octave_num; ++o)
    for (int j = 0; j < 3; ++j)
      if (got == test_pyramid->GetBaseLevel(test_pyramid->_octave_min + o,
                    SiftPyramid::DATA_GRAD) + j + 1) level = o * 3 + j;
  assert(level >= 0 && key == last_key && last_level == level);
  vector<float> tag(key->GetImgWidth() * key->GetImgHeight() * 4);
  key->CopyToHost(tag.data());
  for (float v : tag) assert(v == float(level));
}
void CheckHistogram(CuTexImage* hist) {
  const size_t old_width = size_t(1) << (2 * histogram_slot);
  const size_t old_height = max(test_pyramid->_pyramid_width,
                                test_pyramid->_pyramid_height);
  // Independent enumeration of packed int4 destination addresses, not a
  // second capacity planner. This covers every write and valid texture read
  // in the unchanged InitHist/ReduceHist/ListGen kernels' histogram domain.
  for (int row = 0; row < hist->GetImgHeight(); ++row)
    for (int col = 0; col < hist->GetImgWidth(); ++col)
      for (int lane = 0; lane < 4; ++lane) {
        size_t address = ((size_t(row) * hist->GetImgWidth() + col)*4 + lane)*4;
        assert(address + 4 <= size_t(hist->GetDataSize()));
        assert(address + 4 <= old_width * old_height * 16);
      }
}
"""
                + globals_defs
                + "\n"
                + "\n".join(program_defs + gl_defs)
                + "\n"
                + limiter
                + "\n"
                + list_cpu
                + r"""
void GlobalUtil::InitGLParam(int) {}
double ClockTimer::CLOCK() { return 0; }
GLTexImage::GLTexImage() {}
GLTexImage::~GLTexImage() {}
void GLTexImage::CopyFromPBO(GLuint, int, int, GLenum) {}
int GLTexImage::CopyToPBO(GLuint, int, int, GLenum) { return 0; }
void GLTexImage::InitTexture(int, int, int, GLuint) {}
void SiftPyramid::RunSIFT(GLTexInput*) {
  GenerateFeatureList();
  vector<int> selected(_levelFeatureNum,
      _levelFeatureNum + _octave_num * param._dog_level_num);
  const int selected_total = siftgpu::LimitStreamedFeatureCounts(
      selected, GlobalParam::_FeatureCountThreshold,
      GlobalParam::_TruncateMethod);
  LimitFeatureCount(0);
  assert(selected_total == _featureNum);
  assert(std::equal(selected.begin(), selected.end(), _levelFeatureNum));
  GetFeatureOrientations();
  if (GlobalParam::_MaxOrientation > 1) {
    ReshapeFeatureListCPU();
    LimitFeatureCount(1);
  }
}
void SiftPyramid::SaveSIFT(const char*) {}
void SiftPyramid::CopyFeatureVector(float*, float*) {}
void SiftPyramid::SetKeypointList(int, const float*, int, int) {}
SiftParam::SiftParam() {}
float SiftParam::GetLevelSigma(int) { return 1; }
float SiftParam::GetInitialSmoothSigma(int) { return 1; }
int main(int argc, char** argv) {
  assert(argc == 2);
  {
    // Independently specified nonzero/tail list, larger than BOTH caps.
    siftgpu::StreamKeyResult detection;
    detection.key = std::make_unique<siftgpu::StreamedPlane>(argv[1], 65, 4, 4);
    vector<float> key(65*4*4, 0);
    detection.key->WriteRows(0, 4, key.data());
    detection.key->Seal();
    for (int w : {17, 5, 2, 1}) {
      auto plane = std::make_unique<siftgpu::StreamedPlane>(argv[1], w, 4, 4);
      vector<int> values(w*4*4, 0);
      if (w == 17) {
        for (int y : {1, 2})
          for (int x = 1; x < 64; x += 2)
            values[(y*w + x/4)*4 + x%4] = 1;
      } else {
        auto& previous = *detection.histograms.back();
        vector<float> bytes(previous.Bytes()/4);
        previous.ReadRows(0, 4, bytes.data());
        vector<int> counts(bytes.size());
        std::memcpy(counts.data(), bytes.data(), previous.Bytes());
        for (int y = 0; y < 4; ++y)
          for (int x = 0; x < previous.Width(); ++x)
            for (int lane = 0; lane < 4; ++lane)
              values[(y*w+x/4)*4+x%4] += counts[(y*previous.Width()+x)*4+lane];
      }
      vector<float> bytes(values.size());
      std::memcpy(bytes.data(), values.data(), values.size()*4);
      plane->WriteRows(0, 4, bytes.data());
      plane->Seal();
      detection.histograms.push_back(std::move(plane));
    }
    siftgpu::StreamFilterOptions options;
    options.scratch_directory = argv[1];
    options.gpu_bytes = 164;
    options.host_bytes = 96;
    {
      siftgpu::StreamKeyResult tiny;
      tiny.key = std::make_unique<siftgpu::StreamedPlane>(argv[1], 3, 3, 4);
      vector<float> zero(3*3*4, 0);
      tiny.key->WriteRows(0, 3, zero.data()); tiny.key->Seal();
      for (int nonzero : {0, 1}) {
        tiny.histograms.clear();
        auto root = std::make_unique<siftgpu::StreamedPlane>(argv[1], 1, 3, 4);
        vector<int> counts(12, 0); counts[5] = nonzero;
        vector<float> bits(12); std::memcpy(bits.data(), counts.data(), 48);
        root->WriteRows(0, 3, bits.data()); root->Seal();
        tiny.histograms.push_back(std::move(root));
        if (nonzero) {
          bool refused = false;
          try { ProgramCU::GenerateListStreamed(tiny, options); }
          catch (const std::runtime_error& e) {
            refused = std::string(e.what()) ==
                "SIFT nonzero single-level list is unqualified";
          }
          assert(refused);
        } else {
          auto empty = ProgramCU::GenerateListStreamed(tiny, options);
          assert(empty.count == 0 && !empty.records);
        }
      }
    }
    for (int batch : {1, 7, 32, 7}) {
      options.gpu_bytes = 132 + 32*batch;
      options.host_bytes = 64 + 32*batch;
      uint64_t reads_before = 0;
      for (auto& plane : detection.histograms)
        reads_before += plane->BytesRead();
      auto list = ProgramCU::GenerateListStreamed(detection, options);
      uint64_t reads_after = 0;
      for (auto& plane : detection.histograms)
        reads_after += plane->BytesRead();
      assert(list.stats.file_read_bytes == reads_after - reads_before);
      assert(list.count == 64 && list.records->Bytes() == 1024);
      assert(list.stats.gpu_staging_bytes <= options.gpu_bytes);
      assert(list.stats.host_staging_bytes <= options.host_bytes);
      vector<float> output(list.count*4);
      list.records->ReadRows(0, list.count, output.data());
      for (int i = 0; i < list.count; ++i) {
        int point[4]; std::memcpy(point, output.data()+4*i, 16);
        assert(point[0] == 1 + 2*(i%32));
        assert(point[1] == 1 + i/32 && point[2] == 0 && point[3] == 0);
      }
      assert(list.stats.file_read_bytes > 0);
      assert(list.stats.file_written_bytes == 1024);
      auto reads = detection.histograms.back()->BytesRead();
      options.cancelled = [&] {
        return detection.histograms.back()->BytesRead() > reads + 64;
      };
      bool refused = false;
      try { ProgramCU::GenerateListStreamed(detection, options); }
      catch (const std::runtime_error&) { refused = true; }
      assert(refused);
      options.cancelled = {};
    }
  }
  GlobalParam::_texMaxDim = 4096;
  GlobalParam::_octave_min_default = 0;
  GlobalParam::_MaxLevelFeatureNum = 32;
  SiftParam param;
  param._level_num = 6; param._dog_level_num = 3;
  param._level_min = -1; param._level_max = 4;
  param._dog_threshold = 0.02f; param._edge_threshold = 10.0f;
  {
    PyramidCU rectangle(param);
    dense_owners.clear();
    test_pyramid = &rectangle;
    rectangle.InitPyramid(260, 130);
    rectangle.GenerateFeatureList();
    size_t bytes = 0;
    for (auto entry : histogram_bytes) bytes += entry.second;
    // Independent worked rectangular reduction: 65,17,5,2,1 columns,
    // 130 rows, four int lanes. Old square bound: 341*260*16 bytes.
    assert(bytes == 187200);
    assert(bytes < 1418560);
    histogram_bytes.clear();
  }
  {
    SuppliedKeyPyramid pyramid(param);
    dense_owners.clear();
    test_pyramid = &pyramid;
    // Odd/aligned rectangles, automatic octaves, large-small-large reuse,
    // and transitions between exact and legacy aligned input domains.
    for (auto shape : {pair<int,int>{259,129}, {65,33}, {259,129},
                       {260,130}, {64,32}, {260,130}, {513,257}}) {
      dense_owners.clear();
      pyramid.InitPyramid(shape.first, shape.second);
      assert(pyramid._octave_num > 1);
      for (int o = 0; o < pyramid._octave_num; ++o) {
        auto* gradient = pyramid.GetBaseLevel(o, SiftPyramid::DATA_GRAD) + 1;
        assert(gradient->GetDataSize() >= gradient->GetImgWidth() *
               gradient->GetImgHeight() * 2 * int(sizeof(float)));
        // Prove retained float2 storage still round-trips unchanged and binds
        // a linear resource, including after capacity reuse.
        vector<float> input(
            gradient->GetImgWidth()*gradient->GetImgHeight()*2, 1.25f);
        vector<float> output(input.size());
        gradient->CopyFromHost(input.data());
        gradient->CopyToHost(output.data());
        assert(input == output);
        gradient->BindTexture({}, {32,32,0,0,cudaChannelFormatKindFloat});
      }
      // Reservation must cover every active octave without hot-loop growth.
      int allocations = linear_allocations;
      for (int o = 0; o < pyramid._octave_num; ++o)
        pyramid.FitHistogramPyramid(
            pyramid.GetBaseLevel(o, SiftPyramid::DATA_KEYPOINT) + 2);
      assert(linear_allocations == allocations);
      // Controlled histogram results exercise both copy branches, without
      // simulating detection or claiming numerical SIFT correctness.
      test_features = 0;
      pyramid.GenerateFeatureList();
      assert(pyramid.GetFeatureNum() == 0);
      test_features = 1;
      pyramid.GenerateFeatureList();
      assert(pyramid.GetFeatureNum() == pyramid._octave_num * 3);
      int after_lists = linear_allocations;
      for (int subpixel : {0, 1}) {
        GlobalParam::_SubpixelLocalization = subpixel;
        pyramid.GetFeatureOrientations();
        int executed = key_executions;
        // Supplied and rectangular modes consume their feature lists, never
        // the dense keys; changing this must not add detection execution.
        for (int existing : {1, int(SiftPyramid::SIFT_RECT_DESCRIPTION)}) {
          pyramid.ExistingKeys(existing);
          pyramid.GetFeatureOrientations();
          assert(key_executions == executed);
        }
        pyramid.ExistingKeys(0);
      }
      assert(linear_allocations == after_lists);
      // Actual preorientation limiter versus streamed metadata, including
      // zero, equality, strict early stop and whole-level overshoot.
      for (int features : {0, 1, 2, 3, 40})
        for (int threshold : {0, 1, 2, 3, 4, 40, 41})
          for (int method : {0, 1, 2}) {
            test_features = features;
            GlobalParam::_FeatureCountThreshold = threshold;
            GlobalParam::_TruncateMethod = method;
            GlobalParam::_MaxOrientation = 0;
            pyramid.RunSIFT(nullptr);
          }
      GlobalParam::_FeatureCountThreshold = 0;
      GlobalParam::_TruncateMethod = 0;
      // Overflow the initial 32-feature reservation without touching dense
      // capacity or changing level ordering. These are API counts, not SIFT.
      test_features = 40;
      pyramid.GenerateFeatureList();
      assert(pyramid.GetFeatureNum() == pyramid._octave_num * 3 * 40);
      pyramid.GetFeatureOrientations();
    }
  }
  // Reuse into shifted slots, thin/tiny/odd domains, transposes, tight
  // allocations, explicit octaves, up/downsampling and traversal options.
  for (int first : {-1, 0, 2})
    for (int tight : {0, 1})
      for (int octaves : {0, 1}) {
        GlobalParam::_octave_min_default = first;
        GlobalParam::_ForceTightPyramid = tight;
        GlobalParam::_octave_num_default = octaves;
        PyramidCU pyramid(param);
        dense_owners.clear();
        test_pyramid = &pyramid;
        for (auto shape : {pair<int,int>{256,256}, {4,128}, {128,4},
                           {1,1}, {3,5}, {65,33}, {33,65}, {256,256}})
          for (int ds : {0, 1}) {
            dense_owners.clear();
            pyramid.InitPyramid(shape.first, shape.second, ds);
            int allocations = linear_allocations;
            for (int o = 0; o < pyramid._octave_num; ++o)
              pyramid.FitHistogramPyramid(pyramid.GetBaseLevel(
                  pyramid._octave_min + o, SiftPyramid::DATA_KEYPOINT) + 2);
            assert(linear_allocations == allocations);
            for (int truncate : {0, 1, 2}) {
              GlobalParam::_TruncateMethod = truncate;
              GlobalParam::_FeatureCountThreshold = 2;
              test_features = 1;
              list_order.clear();
              pyramid.GenerateFeatureList();
              // Independent original traversal with one feature per level:
              // strict > early stop retains three levels for threshold two.
              vector<int> expected;
              int levels = pyramid._octave_num * 3;
              for (int n = 0; n < levels; ++n) {
                if (truncate && n > 2) break;
                expected.push_back(truncate == 1 ? levels - 1 - n : n);
              }
              assert(list_order == expected);
              pyramid.GetFeatureOrientations();
              GlobalParam::_MaxOrientation = 2;
              pyramid.RunSIFT(nullptr);
              const int* final_counts = pyramid.GetLevelFeatureNum();
              // Independent worked original boundaries: method 2 retains
              // first two single features, expands to four, then keeps the
              // first two-orientation level. Others retain last three single
              // features, expand to six, then retain last two levels (four).
              assert(pyramid.GetFeatureNum() == (truncate == 2 ? 2 : 4));
              for (int n = 0; n < levels; ++n)
                assert(final_counts[n] == (truncate == 2 ?
                    (n == 0 ? 2 : 0) : (n >= levels - 2 ? 2 : 0)));
              GlobalParam::_MaxOrientation = 0;
            }
          }
      }
  GlobalParam::_octave_min_default = 0;
  GlobalParam::_octave_num_default = 3;
  GlobalParam::_ForceTightPyramid = 0;
  {
    PyramidCU shifted(param);
    dense_owners.clear();
    test_pyramid = &shifted;
    shifted.InitPyramid(4, 128);
    shifted.InitPyramid(1, 5);
    // The width-one reduction moves into a slot unused by the large
    // input's traversal. Reserving only the first rectangle is insufficient.
    assert(shifted._pyramid_octave_first == 2);
    int allocations = linear_allocations;
    shifted.FitHistogramPyramid(shifted.GetBaseLevel(
        shifted._octave_min, SiftPyramid::DATA_KEYPOINT) + 2);
    assert(linear_allocations == allocations);
    shifted.GenerateFeatureList();
  }
  GlobalParam::_octave_num_default = 8;
  {
    PyramidCU thin(param);
    dense_owners.clear();
    test_pyramid = &thin;
    for (auto shape : {pair<int,int>{1,257}, {1,5}, {3,129}, {1,257}}) {
      thin.InitPyramid(shape.first, shape.second);
      int allocations = linear_allocations;
      for (int o = 0; o < thin._octave_num; ++o)
        thin.FitHistogramPyramid(thin.GetBaseLevel(
            thin._octave_min + o, SiftPyramid::DATA_KEYPOINT) + 2);
      assert(linear_allocations == allocations);
      thin.GenerateFeatureList();
    }
  }
  // Real initialization request ledger only: do not allocate the audited
  // multi-GB payload on the CPU, run kernels, or claim physical residency.
  GlobalParam::_octave_min_default = -1;
  GlobalParam::_octave_num_default = 0;
  GlobalParam::_texMaxDim = 32768;
  reservation_only = true;
  requests.clear();
  {
    PyramidCU audited(param);
    audited.InitPyramid(6048, 4032);
    // Nine octaves, each with 6 Gaussian + 5 DoG + 3 gradient,
    // ONE maximum-sized float4 scratch/key allocation, then histograms.
    // No per-level dense key owner may retain a device allocation.
    for (int o = 0; o < audited._octave_num; ++o)
      for (int j = 0; j < param._level_num; ++j)
        assert((audited.GetBaseLevel(audited._octave_min + o,
                SiftPyramid::DATA_KEYPOINT) + j)->GetDataSize() == 0);
    assert(audited._octave_num == 9);
    assert(requests.at(9 * 14) == size_t(12096) * 8064 * 16);
    // Old independently audited dense payload across all nine octaves minus
    // the sole largest plane. Logical requests, not additive peak savings.
    assert(size_t(130056672) * 3 * 16 - requests.at(9 * 14) == 4682045952ULL);
    vector<size_t> expected = {129024, 387072, 1548288, 6193152,
                              24385536, 97542144, 390168576};
    size_t bytes = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      assert(requests.at(9 * 14 + 1 + i) == expected[i]);
      bytes += expected[i];
    }
    assert(bytes == 520353792);
    assert(1056900096 - bytes == 536546304);
    cerr << "audited histogram old=1056900096 new=" << bytes
         << " saving=536546304 bytes\n";
  }
  assert(live.empty()); // real resize/destructor must release linear owners
  assert(linear_allocations > 0);
  cerr << "array allocations=" << array_allocations
       << " array copies=" << array_copies << "\n";
  assert(array_allocations == 0);
  assert(array_copies == 0);
}
"""
            )
            (temp / "storage.cc").write_text(source)
            binary = temp / "storage"
            for flags in ([], ["-DSIFTGPU_ENABLE_LINEAR_TEX2D"]):
                with self.subTest(flags=flags):
                    subprocess.run(
                        [
                            "c++",
                            "-std=c++17",
                            "-DSIFTGPU_CUDA_ENABLED",
                            "-include",
                            "climits",
                            *flags,
                            "-I",
                            str(temp),
                            "-I",
                            str(SIFT),
                            str(temp / "storage.cc"),
                            str(SIFT / "PyramidCU.cpp"),
                            str(SIFT / "CuTexImage.cpp"),
                            "-o",
                            str(binary),
                        ],
                        check=True,
                    )
                    subprocess.run([str(binary), str(temp)], check=True)


if __name__ == "__main__":
    unittest.main()
