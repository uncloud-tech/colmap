# SPDX-License-Identifier: BSD-3-Clause
"CPU source/ABI contracts. NOT CUDA compilation, ordering, or exactness proof."

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "c6ab4f897c94f10b15bdecc442dcfce4688f1b76"
MVS = ROOT / "src/colmap/mvs"


class Contract(unittest.TestCase):
    def source(self, name):
        path = MVS / name
        self.assertTrue(
            path.exists(), f"missing required owned implementation: {name}"
        )
        return path.read_text()

    def compile_cpu(self, body, headers=None):
        # Standalone CPU executable, never links libcudart or executes CUDA.
        with tempfile.TemporaryDirectory(prefix="owned-contract-") as tmp:
            source = Path(tmp) / "contract.cc"
            binary = Path(tmp) / "contract"
            source.write_text(body)
            for relative, data in (headers or {}).items():
                path = Path(tmp) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)
            result = subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    tmp,
                    "-I",
                    str(ROOT / "src"),
                    str(source),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr
            )
            result = subprocess.run(
                [str(binary)], capture_output=True, text=True, timeout=10
            )
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr
            )

    def test_stock_hot_tu_unchanged(self):
        original = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{BASE}:src/colmap/mvs/patch_match_cuda.cu",
            ]
        )
        self.assertEqual((MVS / "patch_match_cuda.cu").read_bytes(), original)

    def test_execution_policy_cpu(self):
        self.source("patch_match_execution.h")
        self.compile_cpu(
            "\n"
            '#include "colmap/mvs/patch_match_execution.h"\n'
            "#include <cassert>\n"
            "#include <stdexcept>\n"
            "using namespace colmap::mvs;\n"
            "int main() {\n"
            "  assert(ValidatePatchMatchExecution(PatchMatchExecution::kS"
            "tock, 3, 2, true, false)\n"
            "         == PatchMatchExecution::kStock);\n"
            "  assert(ValidatePatchMatchExecution(PatchMatchExecution::kO"
            "wnedCuda, 5, 1, false, true)\n"
            "         == PatchMatchExecution::kOwnedCuda);\n"
            "  assert(ValidatePatchMatchExecution(PatchMatchExecution::kO"
            "wnedCuda, 5, 1, true, true)\n"
            "         == PatchMatchExecution::kOwnedCuda);\n"
            "  int rejected = 0;\n"
            "  for (int i=0; i<4; ++i) {\n"
            "    try { ValidatePatchMatchExecution(i==2 ? static_cast<Pat"
            "chMatchExecution>(99)\n"
            "                                         : PatchMatchExecuti"
            "on::kOwnedCuda,\n"
            "                                     i==0 ? 4:5, i==1 ? 2:1,"
            " i==2, i!=3); }\n"
            "    catch (const std::invalid_argument&) { ++rejected; }\n"
            "  }\n"
            "  assert(rejected == 4);\n"
            "  for (int i=0; i<3; ++i) {\n"
            "    bool caught=false;\n"
            "    try { ValidatePatchMatchExecution(PatchMatchExecution::k"
            "OwnedCuda, 5, 1,\n"
            "                                     true, true, i!=0, i==1,"
            " i==2); }\n"
            "    catch (const std::invalid_argument&) { caught=true; }\n"
            "    assert(caught);\n"
            "  }\n"
            "  // These restrictions must not change photometric or stock"
            " policy.\n"
            "  ValidatePatchMatchExecution(PatchMatchExecution::kOwnedCud"
            "a, 5, 1,\n"
            "                             false, true, false, true, true)"
            ";\n"
            "}\n"
        )

    def test_calibration_snapshot_all_four_orientations_cpu(self):
        self.source("patch_match_calibration.h")
        self.compile_cpu(r"""
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
""")

    def test_default_dispatch_and_owned_input_lifetime(self):
        header = self.source("patch_match.h")
        source = self.source("patch_match.cc")
        self.assertIn(
            "PatchMatchExecution execution = PatchMatchExecution::kStock",
            header,
        )
        self.assertIn("owned_images_", header)
        for member in [
            "owned_images_",
            "owned_depth_maps_",
            "owned_normal_maps_",
        ]:
            self.assertLess(
                header.index(member), header.index("const Problem problem_")
            )
        self.assertIn("result.images = &owned_images_", source)
        self.assertIn(
            "std::make_unique<PatchMatchCuda>(options_, problem_)", source
        )
        self.assertIn(
            "std::make_unique<owned::PatchMatchCuda>(options_, problem_)",
            source,
        )
        self.assertIn("ValidatePatchMatchExecution", source)

    def test_actual_wrapper_snapshot_and_default_borrow_cpu(self):
        source = self.source("patch_match.cc")
        start = source.index(
            "namespace {\nPatchMatchExecution ValidateExecution"
        )
        end = source.index("void PatchMatch::Problem::Print()", start)
        constructors = source[start:end]
        headers = {
            ("colmap/mvs/depth_map.h"): (
                "#pragma once\n"
                "#include <vector>\n"
                "namespace colmap {namespace mvs {struct DepthMap {std::vecto"
                "r<float> bytes;}; template<class T> class Mat {};}}"
            ),
            ("colmap/mvs/normal_map.h"): (
                "#pragma once\n"
                "#include <vector>\n"
                "namespace colmap {namespace mvs {struct NormalMap {std::vect"
                "or<float> bytes;};}}"
            ),
            ("colmap/mvs/image.h"): (
                "#pragma once\n"
                "#include <vector>\n"
                "namespace colmap {namespace mvs {struct Image {std::vector<u"
                "nsigned char> bytes;};}}"
            ),
            ("colmap/mvs/patch_match_options.h"): (
                "#pragma once\n"
                "namespace colmap {namespace mvs {struct PatchMatchOptions {i"
                "nt window_radius=5; int window_step=1; bool geom_consistency"
                "=false; bool filter=true; bool write_consistency_graph=false"
                "; bool allow_missing_files=false;};}}"
            ),
        }
        program = (
            "\n"
            "#include <cassert>\n"
            "#include <filesystem>\n"
            "#include <memory>\n"
            "#include <stdexcept>\n"
            "#include <vector>\n"
            "#define __CUDACC__\n"
            "#define COLMAP_CUDA_ENABLED\n"
            "#define private public\n"
            '#include "colmap/mvs/patch_match.h"\n'
            "#undef private\n"
            "namespace colmap {namespace mvs {\n"
            "class PatchMatchCuda {};\n"
            "namespace owned {class PatchMatchCuda {};}\n"
            "// CONSTRUCTORS\n"
            "}}\n"
            "int main() {\n"
            " using namespace colmap::mvs;\n"
            " PatchMatchOptions options;\n"
            " std::unique_ptr<PatchMatch> owned;\n"
            " {\n"
            "  std::vector<Image> input(1); input[0].bytes={7,8,9};\n"
            "  PatchMatch::Problem problem; problem.images=&input; proble"
            "m.ref_image_idx=0;\n"
            "  problem.src_image_idxs={2,1};\n"
            "  PatchMatch stock(options,problem);\n"
            "  assert(stock.execution_==PatchMatchExecution::kStock);\n"
            "  assert(stock.problem_.images==&input && stock.owned_images"
            "_.empty());\n"
            "  owned=std::make_unique<PatchMatch>(options,problem,PatchMa"
            "tchExecution::kOwnedCuda);\n"
            "  assert(owned->problem_.images==&owned->owned_images_);\n"
            "  assert(owned->problem_.images!=&input);\n"
            "  input[0].bytes[0]=42;\n"
            "  assert(owned->problem_.images->at(0).bytes[0]==7);\n"
            "  assert(stock.problem_.images->at(0).bytes[0]==42);\n"
            " }\n"
            " assert(owned->problem_.images->at(0).bytes[0]==7);\n"
            " assert((owned->problem_.src_image_idxs==std::vector<int>{2,"
            "1}));\n"
            " assert(owned->problem_.depth_maps==nullptr && owned->proble"
            "m_.normal_maps==nullptr);\n"
            " options.geom_consistency=true;\n"
            " std::unique_ptr<PatchMatch> geometry;\n"
            " {\n"
            "  std::vector<Image> images(4);\n"
            "  std::vector<DepthMap> depths(4);\n"
            "  std::vector<NormalMap> normals(4);\n"
            "  depths[2].bytes={1,2}; normals[2].bytes={3,4,5};\n"
            "  PatchMatch::Problem problem;\n"
            "  problem.images=&images; problem.depth_maps=&depths; proble"
            "m.normal_maps=&normals;\n"
            "  problem.ref_image_idx=2; problem.src_image_idxs={3,1};\n"
            "  geometry=std::make_unique<PatchMatch>(options,problem,Patc"
            "hMatchExecution::kOwnedCuda);\n"
            "  assert(geometry->GetProblem().depth_maps!=&depths);\n"
            "  assert(geometry->GetProblem().normal_maps!=&normals);\n"
            "  depths[2].bytes[0]=99; normals.clear();\n"
            " }\n"
            " assert(geometry->GetProblem().depth_maps->size()==4);\n"
            " assert(geometry->GetProblem().depth_maps->at(0).bytes.empty"
            "());\n"
            " assert(geometry->GetProblem().depth_maps->at(2).bytes[0]==1"
            ");\n"
            " assert(geometry->GetProblem().normal_maps->at(2).bytes[0]=="
            "3);\n"
            " assert((geometry->GetProblem().src_image_idxs==std::vector<"
            "int>{3,1}));\n"
            " options.window_radius=4;\n"
            " bool caught=false;\n"
            " try { PatchMatch invalid(options,PatchMatch::Problem{},Patc"
            "hMatchExecution::kOwnedCuda); }\n"
            " catch(const std::invalid_argument&) {caught=true;}\n"
            " assert(caught);\n"
            "}\n"
        )
        self.compile_cpu(
            program.replace("// CONSTRUCTORS", constructors), headers
        )
        # Actual production Image contains a by-value Bitmap; Bitmap's copy owns
        # a vector and clones metadata. The CPU wrapper test above stubs pixels.
        bitmap = (ROOT / "src/colmap/sensor/bitmap.cc").read_text()
        copy = bitmap[
            bitmap.index("Bitmap::Bitmap(const Bitmap& other)") : bitmap.index(
                "Bitmap::Bitmap(Bitmap&&"
            )
        ]
        self.assertIn("data_ = other.data_", copy)
        self.assertIn("OIIOMetaData::Clone(other.meta_data_)", copy)
        self.assertIn("Bitmap bitmap_", self.source("image.h"))

    def test_one_reference_controller_validation_seam(self):
        source = self.source("patch_match.cc")
        self.assertIn("THROW_CHECK_EQ(problems_.size(), 1)", source)
        self.assertIn(
            "PatchMatch patch_match(patch_match_options, problem, execution_)",
            source,
        )
        marker = "PATCH_MATCH_BACKEND actual=owned_cuda reference="
        self.assertIn(marker, source)
        self.assertLess(
            source.index("std::make_unique<owned::PatchMatchCuda>"),
            source.index(marker),
        )
        driver = ROOT / "src/colmap/exe/owned_reference.cc"
        self.assertTrue(
            driver.exists(), "missing optional one-reference driver"
        )
        driver_text = driver.read_text()
        self.assertIn("RunPatchMatchStereo", driver_text)
        self.assertIn("PatchMatchExecution::kOwnedCuda", driver_text)
        self.assertNotIn("OptionManager", driver_text)
        cmake = (ROOT / "src/colmap/exe/CMakeLists.txt").read_text()
        self.assertIn("OWNED_PATCH_MATCH_DRIVER_ENABLED", cmake)
        self.assertIn("colmap_owned_reference", cmake)

    def test_geometry_route_guards_and_effective_order(self):
        source = self.source("patch_match.cc")
        run = source[
            source.index("void PatchMatchController::Run()") : source.index(
                "PatchMatchController::PrepareOwnedPairJob"
            )
        ]
        self.assertIn("execution_ == PatchMatchExecution::kStock", run)
        self.assertIn("if (options_.geom_consistency &&", run)
        inputs = source[
            source.index("void PatchMatchController::ProcessProblemInputs") :
        ]
        # Native reused-photo skips before mutating configured source order;
        # owned geometry bypasses that prepass and reaches the same
        # transform once.
        skip = inputs.index(
            "if (ExistsFile(depth_map_path) && ExistsFile(normal_map_path)"
        )
        collect = inputs.index("FlatHashSet<int> used_image_idxs")
        transform = inputs.index("problem.src_image_idxs = src_image_idxs;")
        self.assertLess(skip, collect)
        self.assertLess(collect, transform)
        self.assertEqual(
            inputs.count("    problem.src_image_idxs = src_image_idxs;"), 1
        )
        self.assertNotIn("sort(", inputs)
        self.assertIn(
            "!ExistsFile(depth_map_path) && !ExistsFile(normal_map_path)",
            inputs,
        )
        self.assertIn("Missing image or map dependency for image ", inputs)
        self.assertIn(
            "!prepared->patch_match->GetOptions().geom_consistency", source
        )
        for entry, end in [
            ("PrepareOwnedPairJob", "RunOwnedPairJob"),
            ("RunOwnedPairJob", "RunPreparedOwnedPairJob"),
            ("RunPreparedOwnedPairJob", "ExecutePreparedOwnedPairJob"),
        ]:
            start = source.index("PatchMatchController::" + entry)
            block = source[
                start : source.index("PatchMatchController::" + end, start)
            ]
            if entry == "RunPreparedOwnedPairJob":
                self.assertIn("THROW_CHECK(!options_.geom_consistency)", block)
            else:
                self.assertNotIn(
                    "THROW_CHECK(!options_.geom_consistency)", block
                )

    def test_paired_geometry_admission_and_accounting(self):
        source = (ROOT / "src/colmap/exe/owned_pair.cc").read_text()
        self.assertIn("(o.geom_consistency && !o.filter)", source)
        self.assertIn("geometry requires exactly TWO jobs", source)
        self.assertIn("mixed photometric/geometric phases", source)
        self.assertIn("geometry host lookahead is unsupported", source)
        future = source[
            source.index("PrepareFutureHostJob(") : source.index(
                "template <class RunJob, class PrepareJob"
            )
        ]
        self.assertLess(
            future.index("if (job.patch_match_stereo->geom_consistency)"),
            future.index("mvs::PatchMatchController controller"),
        )
        self.assertIn("host lookahead is photometric-only", future)
        check = source[
            source.index("void CheckInputs(") : source.index(
                "template<class T>"
            )
        ]
        self.assertIn("CheckSeedMap(", check)
        self.assertIn("pair::CheckedMultiply(17, q) : q", check)
        self.assertIn("pair::CheckedMultiply(2,retained)", check)
        self.assertIn("pair::CheckedMultiply(4, u) : u", check)
        self.assertIn("pair::CheckedMultiply(4,u)", check)
        self.assertIn("pair::CheckedMultiply(max_w,max_h)", check)
        self.assertIn("std::max(staging,pair::CheckedMultiply(24,a))", check)

    def test_paired_geometry_envelope_cpu(self):
        # Compile the actual runner preflight, with image/rotation stubs only.
        # Native bitmap decoding and controller lifetime are covered by GTest.
        source = (ROOT / "src/colmap/exe/owned_pair.cc").read_text()
        preflight = source[
            source.index("void CheckSeedMap(") : source.index(
                "template<class T>"
            )
        ]
        self.compile_cpu(
            (
                "\n"
                '#include "colmap/mvs/owned_pair_validation.h"\n'
                "#include <algorithm>\n"
                "#include <array>\n"
                "#include <cassert>\n"
                "#include <filesystem>\n"
                "#include <fstream>\n"
                "#include <set>\n"
                "#include <vector>\n"
                "namespace fs = std::filesystem;\n"
                "namespace colmap::mvs::owned {\n"
                "struct ByteCharge { uint64_t host, device; };\n"
                "struct Layout { uint64_t logical_bytes; };\n"
                "inline Layout BuildOwnedRotationLayout(uint64_t, uint64_t, u"
                "int64_t) { return {42}; }\n"
                "inline uint64_t ReadPairBitmapPixels(const fs::path&, uint64"
                "_t w, uint64_t h) {\n"
                "  return CheckedMultiply(w, h);\n"
                "}\n"
                "}\n"
                "namespace pair = colmap::mvs::owned;\n"
                "namespace mvs {\n"
                "struct Image {\n"
                "  fs::path path; uint64_t width, height;\n"
                "  const fs::path& GetPath() const { return path; }\n"
                "  uint64_t GetWidth() const { return width; }\n"
                "  uint64_t GetHeight() const { return height; }\n"
                "};\n"
                "struct PatchMatchOptions { bool geom_consistency; double cac"
                "he_size; };\n"
                "struct PatchMatch { struct Problem {\n"
                "  const std::vector<Image>* images; int ref_image_idx;\n"
                "  std::vector<int> src_image_idxs;\n"
                "}; };\n"
                "}\n"
            )
            + preflight
            + r"""
void Seed(const fs::path& path, uint64_t w, uint64_t h, uint64_t channels) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  out << w << "&" << h << "&" << channels << "&";
  std::string payload(w*h*channels*4, 0);
  out.write(payload.data(), payload.size());
}
int main(int argc, char** argv) {
  assert(argc == 1);
  const auto root = fs::path(argv[0]).parent_path() / "workspace";
  fs::create_directories(root / "images");
  std::vector<mvs::Image> images{{root/"images/a",2,2},
                               {root/"images/b",10,3},
                               {root/"images/c",3,11}};
  for (const auto& image : images) {
    std::ofstream(image.path).put(0);
    const auto name = image.path.filename().string() + ".photometric.bin";
    Seed(root/"stereo/depth_maps"/name, image.width, image.height, 1);
    Seed(root/"stereo/normal_maps"/name, image.width, image.height, 3);
  }
  mvs::PatchMatch::Problem p{&images, 0, {2, 1}};
  mvs::PatchMatchOptions o{true, 1139/1073741824.0};
  const uint64_t host = 1139+2278+880;
  const uint64_t device = 42+1100+8+1376+32;
  CheckInputs(o, p, {host, device}, root);
  auto rejects = [&](pair::ByteCharge charge) {
    bool caught=false;
    try { CheckInputs(o, p, charge, root); }
    catch(const std::invalid_argument&) { caught=true; }
    assert(caught);
  };
  rejects({host-1,device}); rejects({host,device-1});
  o.cache_size=1138/1073741824.0; rejects({host,device});
  o.cache_size=67/1073741824.0; rejects({host,device});
  o.geom_consistency=false;
  fs::remove_all(root/"stereo");
  CheckInputs(o,p,{67+134+220,42+220+8+1376+32},root);
  rejects({67+134+220-1,device});
  o.geom_consistency=true; o.cache_size=1139/1073741824.0;
  rejects({host,device});  // Missing seeds, not a photo estimate.
  Seed(root/"stereo/depth_maps/a.photometric.bin",2,2,1);
  CheckSeedMap(root/"stereo/depth_maps/a.photometric.bin",2,2,1);
  Seed(root/"stereo/depth_maps/a.photometric.bin",3,2,1);
  rejects({host,device});
  fs::remove_all(root);
}
"""
        )

    def test_constructor_failure_fenced_before_unwind(self):
        source = self.source("patch_match_cuda_owned.cu")
        constructor = source[
            source.index("PatchMatchCuda::PatchMatchCuda(") : source.index(
                "PatchMatchCuda::~PatchMatchCuda()"
            )
        ]
        self.assertIn("catch (...)", constructor)
        self.assertIn("stream_->DrainBeforeDestruction();", constructor)
        primitive = self.source("owned_cuda_execution.h")
        self.assertIn("void DrainBeforeDestruction() const noexcept", primitive)
        self.assertIn("std::terminate()", primitive)

    def test_owned_camera_no_global_writes(self):
        source = self.source("patch_match_cuda_owned.cu")
        self.assertIn("namespace owned {", source)
        self.assertNotIn("__constant__", source)
        self.assertNotIn("cudaMemcpyToSymbol", source)
        self.assertIn("const ReferenceCalibration calibration", source)
        self.assertIn("calibration.inv_K", source)
        self.assertIn("calibration.K", source)
        self.assertNotIn("COLMAP_C_BACKWARD_MODE", source)
        self.assertNotIn("staged_src", source)

    def test_owned_streams_and_lifetime(self):
        source = self.source("patch_match_cuda_owned.cu")
        header = self.source("patch_match_cuda_owned.h")
        self.assertNotIn("CUDA_SYNC_AND_CHECK()", source)
        self.assertIsNone(re.search(r"\bCudaTimer\s", source))
        self.assertIn("OwnedCudaTimer ", source)
        self.assertIn("PatchMatchCuda::~PatchMatchCuda()", source)
        self.assertIn("catch (...)", source)
        self.assertIn("stream_->Synchronize()", source)
        self.assertLess(
            header.index("stream_"), header.index("ref_image_texture_")
        )
        self.assertNotIn("cudaDeviceSynchronize", source)

    def test_stream_helper_surfaces(self):
        for name in [
            "gpu_mat.h",
            "cuda_rotate.h",
            "gpu_mat_prng.h",
            "gpu_mat_prng.cu",
            "gpu_mat_ref_image.h",
            "gpu_mat_ref_image.cu",
            "cuda_texture.h",
        ]:
            source = self.source(name)
            self.assertIn("cudaStream_t stream", source, name)
            self.assertIn("!defined(COLMAP_HIP_ENABLED)", source, name)
        self.assertIn("cudaMemcpy2DAsync", self.source("gpu_mat.h"))
        self.assertIn("cudaMemcpy3DAsync", self.source("cuda_texture.h"))
        self.assertIn("0, stream>>>", self.source("cuda_rotate.h"))

    def test_device_algorithm_only_calibration_plumbing_differs(self):
        original = self.source("patch_match_cuda.cu")
        owned = self.source("patch_match_cuda_owned.cu")

        def device(text):
            return text[
                text.index("__device__ inline void Mat33DotVec3(") : text.index(
                    "PatchMatchCuda::PatchMatchCuda("
                )
            ]

        a, b = device(original), device(owned)
        b = b.replace("  const ReferenceCalibration calibration;\n", "")
        b = b.replace(
            "      : calibration(calibration),\n        local_ref_image",
            "      : local_ref_image",
        )
        b = re.sub(r"const ReferenceCalibration(?:&)? calibration,\s*", "", b)
        b = re.sub(r"\(calibration,\s*", "(", b)
        b = b.replace("calibration.inv_K[", "ref_inv_K[").replace(
            "calibration.K[", "ref_K["
        )
        self.assertEqual(re.sub(r"\s+", "", a), re.sub(r"\s+", "", b))

    def test_all_owned_launch_and_helper_calls_forward_stream(self):
        source = self.source("patch_match_cuda_owned.cu")
        launches = re.findall(r"<<<(.*?)>>>", source, re.S)
        self.assertEqual(len(launches), 4)
        self.assertTrue(all("stream_->Get()" in x for x in launches))
        for name in [
            "Rotate",
            "Filter",
            "CopyToDevice",
            "FillWithScalar",
            "FillWithRandomNumbers",
            "CopyToMat",
            "CopyFromGpuMat",
        ]:
            calls = re.findall(r"->" + name + r"\(([^;]+)\);", source)
            self.assertTrue(calls, name)
            self.assertTrue(all("stream_->Get()" in x for x in calls), name)
        for name in ["FromGpuMat", "FromHostArray"]:
            calls = re.findall(r"::" + name + r"\(([^;]+)\);", source)
            # Reference textures now refresh in place; validate any remaining
            # factory callers without requiring an obsolete ref allocation.
            if name == "FromHostArray":
                self.assertTrue(calls, name)
            self.assertTrue(all("stream_->Get()" in x for x in calls), name)
        self.assertIn(
            "array->CopyFromGpuMat(mat, stream);", self.source("cuda_texture.h")
        )

    def test_legacy_helper_tokens_unchanged(self):
        for name in [
            "gpu_mat.h",
            "cuda_rotate.h",
            "gpu_mat_prng.h",
            "gpu_mat_prng.cu",
            "gpu_mat_ref_image.h",
            "gpu_mat_ref_image.cu",
            "cuda_texture.h",
        ]:
            original = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "show",
                    f"{BASE}:src/colmap/mvs/{name}",
                ],
                text=True,
            )
            current = self.source(name)
            if name == "cuda_texture.h":
                # Permit only this exact constructor-failure cleanup seam;
                # all other legacy tokens still compare against the stock base.
                guarded = """  try {
    CUDA_SAFE_CALL(cudaCreateTextureObject(
        &texture_, &resource_desc_, &texture_desc_, nullptr));
  } catch (...) {
    // The destructor is not called if texture creation fails after allocation.
    (void)cudaFreeArray(array_);
    throw;
  }"""
                legacy = """  CUDA_SAFE_CALL(cudaCreateTextureObject(
      &texture_, &resource_desc_, &texture_desc_, nullptr));"""
                self.assertEqual(current.count(guarded), 1)
                current = current.replace(guarded, legacy)
            current = re.sub(
                r"#if !defined\(COLMAP_HIP_ENABLED\).*?#endif[^\n]*\n?",
                "",
                current,
                flags=re.S,
            )
            self.assertEqual(
                re.sub(r"\s+", "", original), re.sub(r"\s+", "", current), name
            )

    def test_actual_four_rotation_host_arithmetic_cpu(self):
        original = self.source("patch_match_cuda.cu")
        owned = self.source("patch_match_cuda_owned.cu")

        def block(text):
            start = text.index(
                "  for (size_t i = 0; i < 4; ++i)",
                text.index("void PatchMatchCuda::InitTransforms()"),
            )
            end = text.index("    ref_inv_K_host_[i][3]", start)
            return text[start : text.index("\n  }", end) + 4]

        self.assertEqual(block(original), block(owned))
        body = (
            "\n"
            '#include "colmap/mvs/patch_match_calibration.h"\n'
            "#include <algorithm>\n"
            "#include <cassert>\n"
            "#include <cstddef>\n"
            "#include <cstring>\n"
            "struct Image { float K[9]={1200,0,2011,0,1190,1511,0,0,1}; c"
            "onst float* GetK() const {return K;} };\n"
            "int main() {\n"
            "  Image ref_image;\n"
            "  size_t ref_width_=6204,ref_height_=4132;\n"
            "  float ref_K_host_[4][4],ref_inv_K_host_[4][4];\n"
            "  // ORIGINAL_HOST_ARITHMETIC\n"
            "  const float expected[4][4]={{1200,2010.5f,1190,1510.5f},\n"
            "                            {1190,1510.5f,1200,4192.5f},\n"
            "                            {1200,4192.5f,1190,2620.5f},\n"
            "                            {1190,2620.5f,1200,2010.5f}};\n"
            "  for(int r=0;r<4;++r) {\n"
            "    auto c=colmap::mvs::owned::SnapshotReferenceCalibration("
            "ref_K_host_[r],ref_inv_K_host_[r]);\n"
            "    assert(std::memcmp(c.K,expected[r],16)==0);\n"
            "    assert(c.inv_K[0]==1.0f/expected[r][0]);\n"
            "    assert(c.inv_K[1]==-expected[r][1]/expected[r][0]);\n"
            "    assert(c.inv_K[2]==1.0f/expected[r][2]);\n"
            "    assert(c.inv_K[3]==-expected[r][3]/expected[r][2]);\n"
            "  }\n"
            "}\n"
        )
        self.compile_cpu(
            body.replace("// ORIGINAL_HOST_ARITHMETIC", block(owned))
        )

    def test_local_async_resource_unwind_fences(self):
        self.assertIn(
            "class ScopedStreamCompletion",
            self.source("owned_cuda_execution.h"),
        )
        for name in [
            "gpu_mat.h",
            "gpu_mat_prng.cu",
            "gpu_mat_ref_image.cu",
            "cuda_texture.h",
        ]:
            self.assertIn(
                "owned::ScopedStreamCompletion completion(stream)",
                self.source(name),
                name,
            )
        texture = self.source("cuda_texture.h")
        self.assertGreaterEqual(texture.count("completion.Wait();"), 2)

    def test_owned_timer_explicit_stream(self):
        source = self.source("owned_cuda_execution.h")
        self.assertIn("cudaStreamNonBlocking", source)
        self.assertIn("cudaEventRecord(start_, stream_)", source)
        self.assertIn("cudaEventRecord(stop_, stream_)", source)
        self.assertIn("cudaEventSynchronize(stop_)", source)
        self.assertIn("cudaStreamSynchronize(stream_)", source)
        self.assertNotIn("cudaDeviceSynchronize", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
