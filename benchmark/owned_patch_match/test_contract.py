# SPDX-License-Identifier: BSD-3-Clause
"""CPU source/ABI contracts. NOT CUDA compilation, ordering, or exactness proof."""
import hashlib
import os
import re
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASE = 'c6ab4f897c94f10b15bdecc442dcfce4688f1b76'
MVS = ROOT / 'src/colmap/mvs'


class Contract(unittest.TestCase):
    def source(self, name):
        path = MVS / name
        self.assertTrue(path.exists(), f'missing required owned implementation: {name}')
        return path.read_text()

    def compile_cpu(self, body, headers=None):
        # Standalone CPU executable, never links libcudart or executes CUDA.
        with tempfile.TemporaryDirectory(prefix='owned-contract-') as tmp:
            source = Path(tmp) / 'contract.cc'
            binary = Path(tmp) / 'contract'
            source.write_text(body)
            for relative, data in (headers or {}).items():
                path = Path(tmp) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)
            result = subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                     '-I', tmp, '-I', str(ROOT / 'src'), str(source), '-o', str(binary)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stock_hot_tu_unchanged(self):
        original = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                           f'{BASE}:src/colmap/mvs/patch_match_cuda.cu'])
        self.assertEqual((MVS / 'patch_match_cuda.cu').read_bytes(), original)

    def test_execution_policy_cpu(self):
        self.source('patch_match_execution.h')
        self.compile_cpu(r'''
#include "colmap/mvs/patch_match_execution.h"
#include <cassert>
#include <stdexcept>
using namespace colmap::mvs;
int main() {
  assert(ValidatePatchMatchExecution(PatchMatchExecution::kStock, 3, 2, true, false)
         == PatchMatchExecution::kStock);
  assert(ValidatePatchMatchExecution(PatchMatchExecution::kOwnedCuda, 5, 1, false, true)
         == PatchMatchExecution::kOwnedCuda);
  int rejected = 0;
  for (int i=0; i<5; ++i) {
    try { ValidatePatchMatchExecution(i==4 ? static_cast<PatchMatchExecution>(99)
                                         : PatchMatchExecution::kOwnedCuda,
                                     i==0 ? 4:5, i==1 ? 2:1, i==2, i!=3); }
    catch (const std::invalid_argument&) { ++rejected; }
  }
  assert(rejected == 5);
}
''')

    def test_calibration_snapshot_all_four_orientations_cpu(self):
        self.source('patch_match_calibration.h')
        self.compile_cpu(r'''
#include "colmap/mvs/patch_match_calibration.h"
#include <cassert>
#include <cstring>
#include <type_traits>
using namespace colmap::mvs::owned;
int main() {
  static_assert(sizeof(ReferenceCalibration)==32);
  static_assert(std::is_trivially_copyable<ReferenceCalibration>::value);
  const float K[4][4]={{1200,2010.5f,1190,1510.5f},
                       {1190,1510.5f,1200,4192.5f},
                       {1200,4192.5f,1190,2620.5f},
                       {1190,2620.5f,1200,2010.5f}};
  ReferenceCalibration saved[4];
  for(int r=0;r<4;++r) {
    float inverse[4]={1/K[r][0],-K[r][1]/K[r][0],1/K[r][2],-K[r][3]/K[r][2]};
    saved[r]=SnapshotReferenceCalibration(K[r],inverse);
    assert(std::memcmp(saved[r].K,K[r],16)==0);
    assert(std::memcmp(saved[r].inv_K,inverse,16)==0);
    inverse[0]=42;
    assert(saved[r].inv_K[0]!=42);
  }
  float other[4]={-0.0f,1,2,3};
  auto special=SnapshotReferenceCalibration(other,other);
  assert(std::memcmp(special.K,other,16)==0);
  auto changed=saved[0]; changed.K[0]=7;
  assert(saved[0].K[0]==1200 && changed.K[0]==7);
}
''')

    def test_default_dispatch_and_owned_input_lifetime(self):
        header = self.source('patch_match.h')
        source = self.source('patch_match.cc')
        self.assertIn('PatchMatchExecution execution = PatchMatchExecution::kStock', header)
        self.assertIn('owned_images_', header)
        self.assertLess(header.index('owned_images_'), header.index('const Problem problem_'))
        self.assertIn('result.images = &owned_images_', source)
        self.assertIn('std::make_unique<PatchMatchCuda>(options_, problem_)', source)
        self.assertIn('std::make_unique<owned::PatchMatchCuda>(options_, problem_)', source)
        self.assertIn('ValidatePatchMatchExecution', source)

    def test_actual_wrapper_snapshot_and_default_borrow_cpu(self):
        source = self.source('patch_match.cc')
        start = source.index('namespace {\nPatchMatchExecution ValidateExecution')
        end = source.index('void PatchMatch::Problem::Print()', start)
        constructors = source[start:end]
        headers = {
            'colmap/mvs/depth_map.h': '#pragma once\nnamespace colmap {namespace mvs {class DepthMap {}; template<class T> class Mat {};}}',
            'colmap/mvs/normal_map.h': '#pragma once\nnamespace colmap {namespace mvs {class NormalMap {};}}',
            'colmap/mvs/image.h': '#pragma once\n#include <vector>\nnamespace colmap {namespace mvs {struct Image {std::vector<unsigned char> bytes;};}}',
            'colmap/mvs/patch_match_options.h': '#pragma once\nnamespace colmap {namespace mvs {struct PatchMatchOptions {int window_radius=5; int window_step=1; bool geom_consistency=false;};}}',
        }
        program = r'''
#include <cassert>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>
#define __CUDACC__
#define COLMAP_CUDA_ENABLED
#define private public
#include "colmap/mvs/patch_match.h"
#undef private
namespace colmap {namespace mvs {
class PatchMatchCuda {};
namespace owned {class PatchMatchCuda {};}
// CONSTRUCTORS
}}
int main() {
 using namespace colmap::mvs;
 PatchMatchOptions options;
 std::unique_ptr<PatchMatch> owned;
 {
  std::vector<Image> input(1); input[0].bytes={7,8,9};
  PatchMatch::Problem problem; problem.images=&input; problem.ref_image_idx=0;
  problem.src_image_idxs={2,1};
  PatchMatch stock(options,problem);
  assert(stock.execution_==PatchMatchExecution::kStock);
  assert(stock.problem_.images==&input && stock.owned_images_.empty());
  owned=std::make_unique<PatchMatch>(options,problem,PatchMatchExecution::kOwnedCuda);
  assert(owned->problem_.images==&owned->owned_images_);
  assert(owned->problem_.images!=&input);
  input[0].bytes[0]=42;
  assert(owned->problem_.images->at(0).bytes[0]==7);
  assert(stock.problem_.images->at(0).bytes[0]==42);
 }
 assert(owned->problem_.images->at(0).bytes[0]==7);
 assert((owned->problem_.src_image_idxs==std::vector<int>{2,1}));
 assert(owned->problem_.depth_maps==nullptr && owned->problem_.normal_maps==nullptr);
 options.window_radius=4;
 bool caught=false;
 try { PatchMatch invalid(options,PatchMatch::Problem{},PatchMatchExecution::kOwnedCuda); }
 catch(const std::invalid_argument&) {caught=true;}
 assert(caught);
}
'''
        self.compile_cpu(program.replace('// CONSTRUCTORS', constructors), headers)
        # Actual production Image contains a by-value Bitmap; Bitmap's copy owns
        # a vector and clones metadata. The CPU wrapper test above stubs pixels.
        bitmap = (ROOT / 'src/colmap/sensor/bitmap.cc').read_text()
        copy = bitmap[bitmap.index('Bitmap::Bitmap(const Bitmap& other)'):bitmap.index('Bitmap::Bitmap(Bitmap&&')]
        self.assertIn('data_ = other.data_', copy)
        self.assertIn('OIIOMetaData::Clone(other.meta_data_)', copy)
        self.assertIn('Bitmap bitmap_', self.source('image.h'))

    def test_one_reference_controller_validation_seam(self):
        source = self.source('patch_match.cc')
        self.assertIn('THROW_CHECK_EQ(problems_.size(), 1)', source)
        self.assertIn('PatchMatch patch_match(patch_match_options, problem, execution_)', source)
        marker = 'PATCH_MATCH_BACKEND actual=owned_cuda reference='
        self.assertIn(marker, source)
        self.assertLess(source.index('std::make_unique<owned::PatchMatchCuda>'),
                        source.index(marker))
        driver = ROOT / 'src/colmap/exe/owned_reference.cc'
        self.assertTrue(driver.exists(), 'missing optional one-reference driver')
        driver_text = driver.read_text()
        self.assertIn('RunPatchMatchStereo', driver_text)
        self.assertIn('PatchMatchExecution::kOwnedCuda', driver_text)
        self.assertNotIn('OptionManager', driver_text)
        cmake = (ROOT / 'src/colmap/exe/CMakeLists.txt').read_text()
        self.assertIn('OWNED_PATCH_MATCH_DRIVER_ENABLED', cmake)
        self.assertIn('colmap_owned_reference', cmake)

    def test_constructor_failure_fenced_before_unwind(self):
        source = self.source('patch_match_cuda_owned.cu')
        constructor = source[source.index('PatchMatchCuda::PatchMatchCuda('):
                             source.index('PatchMatchCuda::~PatchMatchCuda()')]
        self.assertIn('catch (...)', constructor)
        self.assertIn('stream_->DrainBeforeDestruction();', constructor)
        primitive = self.source('owned_cuda_execution.h')
        self.assertIn('void DrainBeforeDestruction() const noexcept', primitive)
        self.assertIn('std::terminate()', primitive)

    def test_owned_camera_no_global_writes(self):
        source = self.source('patch_match_cuda_owned.cu')
        self.assertIn('namespace owned {', source)
        self.assertNotIn('__constant__', source)
        self.assertNotIn('cudaMemcpyToSymbol', source)
        self.assertIn('const ReferenceCalibration calibration', source)
        self.assertIn('calibration.inv_K', source)
        self.assertIn('calibration.K', source)
        self.assertNotIn('COLMAP_C_BACKWARD_MODE', source)
        self.assertNotIn('staged_src', source)

    def test_owned_streams_and_lifetime(self):
        source = self.source('patch_match_cuda_owned.cu')
        header = self.source('patch_match_cuda_owned.h')
        self.assertNotIn('CUDA_SYNC_AND_CHECK()', source)
        self.assertIsNone(re.search(r'\bCudaTimer\s', source))
        self.assertIn('OwnedCudaTimer ', source)
        self.assertIn('PatchMatchCuda::~PatchMatchCuda()', source)
        self.assertIn('catch (...)', source)
        self.assertIn('stream_->Synchronize()', source)
        self.assertLess(header.index('stream_'), header.index('ref_image_texture_'))
        self.assertNotIn('cudaDeviceSynchronize', source)

    def test_stream_helper_surfaces(self):
        for name in ['gpu_mat.h','cuda_rotate.h','gpu_mat_prng.h',
                     'gpu_mat_prng.cu','gpu_mat_ref_image.h',
                     'gpu_mat_ref_image.cu','cuda_texture.h']:
            source=self.source(name)
            self.assertIn('cudaStream_t stream', source, name)
            self.assertIn('!defined(COLMAP_HIP_ENABLED)', source, name)
        self.assertIn('cudaMemcpy2DAsync', self.source('gpu_mat.h'))
        self.assertIn('cudaMemcpy3DAsync', self.source('cuda_texture.h'))
        self.assertIn('0, stream>>>', self.source('cuda_rotate.h'))

    def test_device_algorithm_only_calibration_plumbing_differs(self):
        original = self.source('patch_match_cuda.cu')
        owned = self.source('patch_match_cuda_owned.cu')
        def device(text):
            return text[text.index('__device__ inline void Mat33DotVec3('):
                        text.index('PatchMatchCuda::PatchMatchCuda(')]
        a, b = device(original), device(owned)
        b = b.replace('  const ReferenceCalibration calibration;\n', '')
        b = b.replace('      : calibration(calibration),\n        local_ref_image',
                      '      : local_ref_image')
        b = re.sub(r'const ReferenceCalibration(?:&)? calibration,\s*', '', b)
        b = re.sub(r'\(calibration,\s*', '(', b)
        b = b.replace('calibration.inv_K[', 'ref_inv_K[').replace('calibration.K[', 'ref_K[')
        self.assertEqual(re.sub(r'\s+', '', a), re.sub(r'\s+', '', b))

    def test_all_owned_launch_and_helper_calls_forward_stream(self):
        source = self.source('patch_match_cuda_owned.cu')
        launches = re.findall(r'<<<(.*?)>>>', source, re.S)
        self.assertEqual(len(launches), 4)
        self.assertTrue(all('stream_->Get()' in x for x in launches))
        for name in ['Rotate', 'Filter', 'CopyToDevice', 'FillWithScalar',
                     'FillWithRandomNumbers', 'CopyToMat']:
            calls = re.findall(r'->'+name+r'\(([^;]+)\);', source)
            self.assertTrue(calls, name)
            self.assertTrue(all('stream_->Get()' in x for x in calls), name)
        for name in ['FromGpuMat', 'FromHostArray']:
            calls = re.findall(r'::'+name+r'\(([^;]+)\);', source)
            self.assertTrue(calls, name)
            self.assertTrue(all('stream_->Get()' in x for x in calls), name)

    def test_legacy_helper_tokens_unchanged(self):
        for name in ['gpu_mat.h','cuda_rotate.h','gpu_mat_prng.h',
                     'gpu_mat_prng.cu','gpu_mat_ref_image.h',
                     'gpu_mat_ref_image.cu','cuda_texture.h']:
            original = subprocess.check_output(['git','-C',str(ROOT),'show',
                                                f'{BASE}:src/colmap/mvs/{name}'],text=True)
            current = self.source(name)
            current = re.sub(r'#if !defined\(COLMAP_HIP_ENABLED\).*?#endif[^\n]*\n?',
                             '',current,flags=re.S)
            self.assertEqual(re.sub(r'\s+','',original), re.sub(r'\s+','',current),name)

    def test_actual_four_rotation_host_arithmetic_cpu(self):
        original = self.source('patch_match_cuda.cu')
        owned = self.source('patch_match_cuda_owned.cu')
        def block(text):
            start = text.index('  for (size_t i = 0; i < 4; ++i)',
                               text.index('void PatchMatchCuda::InitTransforms()'))
            end = text.index('    ref_inv_K_host_[i][3]', start)
            return text[start:text.index('\n  }',end)+4]
        self.assertEqual(block(original),block(owned))
        body = r'''
#include "colmap/mvs/patch_match_calibration.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
struct Image { float K[9]={1200,0,2011,0,1190,1511,0,0,1}; const float* GetK() const {return K;} };
int main() {
  Image ref_image;
  size_t ref_width_=6204,ref_height_=4132;
  float ref_K_host_[4][4],ref_inv_K_host_[4][4];
  // ORIGINAL_HOST_ARITHMETIC
  const float expected[4][4]={{1200,2010.5f,1190,1510.5f},
                            {1190,1510.5f,1200,4192.5f},
                            {1200,4192.5f,1190,2620.5f},
                            {1190,2620.5f,1200,2010.5f}};
  for(int r=0;r<4;++r) {
    auto c=colmap::mvs::owned::SnapshotReferenceCalibration(ref_K_host_[r],ref_inv_K_host_[r]);
    assert(std::memcmp(c.K,expected[r],16)==0);
    assert(c.inv_K[0]==1.0f/expected[r][0]);
    assert(c.inv_K[1]==-expected[r][1]/expected[r][0]);
    assert(c.inv_K[2]==1.0f/expected[r][2]);
    assert(c.inv_K[3]==-expected[r][3]/expected[r][2]);
  }
}
'''
        self.compile_cpu(body.replace('// ORIGINAL_HOST_ARITHMETIC',block(owned)))

    def test_local_async_resource_unwind_fences(self):
        self.assertIn('class ScopedStreamCompletion', self.source('owned_cuda_execution.h'))
        for name in ['gpu_mat.h', 'gpu_mat_prng.cu', 'gpu_mat_ref_image.cu', 'cuda_texture.h']:
            self.assertIn('owned::ScopedStreamCompletion completion(stream)', self.source(name), name)
        texture = self.source('cuda_texture.h')
        self.assertGreaterEqual(texture.count('completion.Wait();'), 2)

    def test_owned_timer_explicit_stream(self):
        source=self.source('owned_cuda_execution.h')
        self.assertIn('cudaStreamNonBlocking', source)
        self.assertIn('cudaEventRecord(start_, stream_)', source)
        self.assertIn('cudaEventRecord(stop_, stream_)', source)
        self.assertIn('cudaEventSynchronize(stop_)', source)
        self.assertIn('cudaStreamSynchronize(stream_)', source)
        self.assertNotIn('cudaDeviceSynchronize', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
