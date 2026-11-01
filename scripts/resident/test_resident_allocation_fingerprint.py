# SPDX-License-Identifier: BSD-3-Clause
"""Canonical build binding must change with phase allocation behavior."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "resident_generator",
    Path(__file__).with_name("generate_resident_allocation_fingerprint.py"),
)
assert spec is not None and spec.loader is not None
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)

FUSION_ALLOCATION_SOURCES = (
    "src/colmap/mvs/fusion.h",
    "src/colmap/mvs/fusion.cc",
    "src/colmap/mvs/fusion_resource.h",
    "src/colmap/mvs/fusion_mapped_mat.h",
    "src/colmap/mvs/fusion_mapped_mat.cc",
    "src/colmap/mvs/fusion_dependencies.h",
    "src/colmap/util/ply.h",
    "src/colmap/util/eigen_alignment.h",
    "src/colmap/util/threading.h",
    "src/colmap/util/threading.cc",
    "src/colmap/util/oiio_utils.h",
    "src/colmap/util/oiio_utils.cc",
)


UNDISTORTION_ALLOCATION_SOURCES = (
    "src/colmap/controllers/undistorters.h",
    "src/colmap/controllers/undistorters.cc",
    "src/colmap/controllers/undistortion_resource.h",
    "src/colmap/image/undistortion.h",
    "src/colmap/image/undistortion.cc",
    "src/colmap/exe/image.cc",
    "src/colmap/exe/colmap.cc",
    "src/colmap/util/cancellation.h",
    "src/colmap/util/cancellation.cc",
)


EXTRACTION_ALLOCATION_SOURCES = (
    "src/colmap/controllers/base_option_manager.h",
    "src/colmap/controllers/base_option_manager.cc",
    "src/colmap/controllers/extraction_resource.h",
    "src/colmap/controllers/feature_extraction.h",
    "src/colmap/controllers/feature_extraction.cc",
    "src/colmap/controllers/image_reader.cc",
    "src/colmap/feature/extractor.h",
    "src/colmap/feature/extractor.cc",
    "src/colmap/feature/sift.h",
    "src/colmap/feature/sift.cc",
    "src/colmap/feature/types.h",
    "src/colmap/feature/utils.cc",
    "src/colmap/scene/database_sqlite.cc",
    "src/thirdparty/VLFeat/sift.c",
    "src/thirdparty/VLFeat/covdet.c",
    "src/thirdparty/SiftGPU/SiftGPU.cpp",
    "src/thirdparty/SiftGPU/SiftPyramid.cpp",
    "src/thirdparty/SiftGPU/PyramidCU.cpp",
    "src/thirdparty/SiftGPU/ProgramCU.cu",
    "src/colmap/exe/feature.cc",
)


MATCHING_ALLOCATION_SOURCES = (
    "src/colmap/scene/database_blob.h",
    "src/colmap/scene/database.h",
    "src/colmap/scene/database.cc",
    "src/colmap/scene/database_sqlite.cc",
    "src/colmap/controllers/matching_resource.h",
    "src/colmap/controllers/feature_matching.h",
    "src/colmap/controllers/feature_matching.cc",
    "src/colmap/controllers/feature_matching_utils.h",
    "src/colmap/controllers/feature_matching_utils.cc",
    "src/colmap/controllers/matcher_cache.h",
    "src/colmap/controllers/matcher_cache.cc",
    "src/colmap/controllers/pairing.h",
    "src/colmap/controllers/pairing.cc",
    "src/colmap/feature/matcher.h",
    "src/colmap/feature/matcher.cc",
    "src/colmap/feature/index.h",
    "src/colmap/feature/index.cc",
    "src/colmap/estimators/two_view_geometry.h",
    "src/colmap/estimators/two_view_geometry.cc",
    "src/colmap/optim/ransac.h",
    "src/colmap/optim/loransac.h",
    "src/colmap/util/cache.h",
    "src/colmap/feature/utils.h",
    "src/colmap/scene/two_view_geometry.h",
    "src/colmap/scene/two_view_geometry.cc",
    "src/thirdparty/SiftGPU/SiftMatchCU.cpp",
    "src/thirdparty/SiftGPU/SiftMatchCU.h",
    "src/thirdparty/SiftGPU/SiftMatch.cpp",
    "src/colmap/geometry/bbox.cc",
    "src/colmap/geometry/bbox.h",
    "src/colmap/geometry/essential_matrix.cc",
    "src/colmap/geometry/essential_matrix.h",
    "src/colmap/geometry/gps.cc",
    "src/colmap/geometry/gps.h",
    "src/colmap/geometry/homography_matrix.cc",
    "src/colmap/geometry/homography_matrix.h",
    "src/colmap/geometry/normalization.cc",
    "src/colmap/geometry/normalization.h",
    "src/colmap/geometry/pose.cc",
    "src/colmap/geometry/pose.h",
    "src/colmap/geometry/pose_prior.cc",
    "src/colmap/geometry/pose_prior.h",
    "src/colmap/geometry/rigid3.cc",
    "src/colmap/geometry/rigid3.h",
    "src/colmap/geometry/sim3.cc",
    "src/colmap/geometry/sim3.h",
    "src/colmap/geometry/triangulation.cc",
    "src/colmap/geometry/triangulation.h",
    "src/colmap/estimators/solvers/absolute_pose.cc",
    "src/colmap/estimators/solvers/absolute_pose.h",
    "src/colmap/estimators/solvers/affine_transform.cc",
    "src/colmap/estimators/solvers/affine_transform.h",
    "src/colmap/estimators/solvers/essential_matrix.cc",
    "src/colmap/estimators/solvers/essential_matrix.h",
    "src/colmap/estimators/solvers/essential_matrix_coeffs.h",
    "src/colmap/estimators/solvers/essential_matrix_poly.h",
    "src/colmap/estimators/solvers/euclidean_transform.h",
    "src/colmap/estimators/solvers/fundamental_matrix.cc",
    "src/colmap/estimators/solvers/fundamental_matrix.h",
    "src/colmap/estimators/solvers/generalized_absolute_pose.cc",
    "src/colmap/estimators/solvers/generalized_absolute_pose.h",
    "src/colmap/estimators/solvers/generalized_relative_pose.cc",
    "src/colmap/estimators/solvers/generalized_relative_pose.h",
    "src/colmap/estimators/solvers/homography_matrix.cc",
    "src/colmap/estimators/solvers/homography_matrix.h",
    "src/colmap/estimators/solvers/poselib_utils.cc",
    "src/colmap/estimators/solvers/poselib_utils.h",
    "src/colmap/estimators/solvers/relpose_one_sided_focal.cc",
    "src/colmap/estimators/solvers/relpose_one_sided_focal.h",
    "src/colmap/estimators/solvers/relpose_shared_focal.cc",
    "src/colmap/estimators/solvers/relpose_shared_focal.h",
    "src/colmap/estimators/solvers/similarity_transform.cc",
    "src/colmap/estimators/solvers/similarity_transform.h",
    "src/colmap/estimators/solvers/translation_transform.h",
    "src/colmap/estimators/solvers/utils.cc",
    "src/colmap/estimators/solvers/utils.h",
    "src/colmap/optim/combination_sampler.cc",
    "src/colmap/optim/combination_sampler.h",
    "src/colmap/optim/least_absolute_deviations.cc",
    "src/colmap/optim/least_absolute_deviations.h",
    "src/colmap/optim/progressive_sampler.cc",
    "src/colmap/optim/progressive_sampler.h",
    "src/colmap/optim/random_sampler.cc",
    "src/colmap/optim/random_sampler.h",
    "src/colmap/optim/sampler.h",
    "src/colmap/optim/sparse_cholesky.cc",
    "src/colmap/optim/sparse_cholesky.h",
    "src/colmap/optim/sprt.cc",
    "src/colmap/optim/sprt.h",
    "src/colmap/optim/support_measurement.cc",
    "src/colmap/optim/support_measurement.h",
    "src/colmap/optim/tiny_solver.h",
    "src/colmap/optim/two_group_sampler.cc",
    "src/colmap/optim/two_group_sampler.h",
    "src/colmap/estimators/fundamental_matrix_degensac.h",
    "src/colmap/estimators/fundamental_matrix_degensac.cc",
    "src/colmap/estimators/generalized_pose.h",
    "src/colmap/estimators/generalized_pose.cc",
    "src/colmap/estimators/bundle_adjustment_ceres.h",
    "src/colmap/estimators/cost_functions/manifold.h",
    "src/colmap/estimators/cost_functions/pose_prior.h",
    "src/colmap/estimators/cost_functions/reprojection_error.h",
    "src/colmap/estimators/cost_functions/utils.h",
    "src/colmap/estimators/pose.h",
    "src/colmap/util/logging.h",
    "src/colmap/util/glog_macros.h",
    "src/colmap/util/string.h",
    "src/colmap/estimators/cost_functions/quaternion_utils.h",
    "src/colmap/estimators/bundle_adjustment.h",
    "src/colmap/estimators/ceres_loss_function.h",
    "src/colmap/util/enum_utils.h",
    "src/colmap/math/random.h",
    "src/colmap/util/misc.h",
    "src/colmap/estimators/cost_functions/tiny_manifold.h",
    "src/colmap/estimators/cost_functions/tiny_sampson_error.h",
    "src/colmap/estimators/cost_functions/sampson_error.h",
    "src/colmap/math/polynomial.h",
    "src/colmap/math/matrix.h",
    "src/colmap/util/cuda_to_hip.h",
    "src/colmap/util/timer.h",
    "src/colmap/scene/database_cache.h",
    "src/colmap/scene/correspondence_graph.h",
    "src/colmap/feature/aliked.h",
    "src/colmap/feature/loma.h",
    "src/colmap/feature/onnx_matchers.h",
    "src/colmap/feature/resources.h",
    "src/colmap/feature/lightglue.h",
    "src/colmap/retrieval/resources.h",
    "src/colmap/retrieval/visual_index.h",
    "src/colmap/retrieval/utils.h",
)

MAPPING_ALLOCATION_SOURCES = (
    "src/colmap/scene/mapping_resource.h",
    "src/colmap/controllers/incremental_pipeline.h",
    "src/colmap/controllers/incremental_pipeline.cc",
    "src/colmap/scene/database.h",
    "src/colmap/scene/database.cc",
    "src/colmap/scene/database_sqlite.h",
    "src/colmap/scene/database_sqlite.cc",
    "src/colmap/scene/database_cache.h",
    "src/colmap/scene/database_cache.cc",
    "src/colmap/scene/correspondence_graph.h",
    "src/colmap/scene/correspondence_graph.cc",
    "src/colmap/scene/reconstruction.h",
    "src/colmap/scene/reconstruction.cc",
    "src/colmap/scene/reconstruction_manager.h",
    "src/colmap/scene/reconstruction_manager.cc",
    "src/colmap/scene/reconstruction_io_binary.h",
    "src/colmap/scene/reconstruction_io_binary.cc",
    "src/colmap/scene/reconstruction_io_utils.h",
    "src/colmap/scene/reconstruction_io_utils.cc",
    "src/colmap/scene/camera.h",
    "src/colmap/scene/camera.cc",
    "src/colmap/scene/image.h",
    "src/colmap/scene/image.cc",
    "src/colmap/scene/frame.h",
    "src/colmap/scene/frame.cc",
    "src/colmap/scene/point2d.h",
    "src/colmap/scene/point2d.cc",
    "src/colmap/scene/point3d.h",
    "src/colmap/scene/point3d.cc",
    "src/colmap/scene/track.h",
    "src/colmap/scene/track.cc",
    "src/colmap/scene/visibility_pyramid.h",
    "src/colmap/scene/visibility_pyramid.cc",
    "src/colmap/sensor/rig.h",
    "src/colmap/sensor/rig.cc",
    "src/colmap/sfm/incremental_mapper.h",
    "src/colmap/sfm/incremental_mapper.cc",
    "src/colmap/sfm/incremental_mapper_impl.h",
    "src/colmap/sfm/incremental_mapper_impl.cc",
    "src/colmap/sfm/incremental_triangulator.h",
    "src/colmap/sfm/incremental_triangulator.cc",
    "src/colmap/sfm/observation_manager.h",
    "src/colmap/sfm/observation_manager.cc",
    "src/colmap/estimators/bundle_adjustment.h",
    "src/colmap/estimators/bundle_adjustment.cc",
    "src/colmap/estimators/bundle_adjustment_ceres.h",
    "src/colmap/estimators/bundle_adjustment_ceres.cc",
    "src/colmap/exe/sfm.h",
    "src/colmap/exe/sfm.cc",
    "src/colmap/util/base_controller.h",
    "src/colmap/util/base_controller.cc",
    "src/colmap/util/cancellation.h",
    "src/colmap/util/cancellation.cc",
    "src/colmap/controllers/option_manager.h",
    "src/colmap/controllers/option_manager.cc",
    "src/colmap/sensor/models/runtime.h",
    "src/colmap/sensor/models.cc",
)

class ResidentAllocationFingerprintTest(unittest.TestCase):
    def test_phase_mutations_change_compiled_binding_and_rebuild_dependencies(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Copy real source bytes, including phase files even when a
            # regression omits them from the generator's declared closure.
            sources = dict.fromkeys(
                (*generator.CLOSURE, *FUSION_ALLOCATION_SOURCES,
                 *UNDISTORTION_ALLOCATION_SOURCES, *EXTRACTION_ALLOCATION_SOURCES, *MATCHING_ALLOCATION_SOURCES, *MAPPING_ALLOCATION_SOURCES)
            )
            for name in sources:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((repo / name).read_bytes())
            output = root / "generated.h"
            depfile = root / "generated.d"

            def generate():
                subprocess.run(
                    [
                        sys.executable,
                        str(repo / "scripts/resident" /
                            "generate_resident_allocation_fingerprint.py"),
                        "--repo-root", str(root), "--output", str(output),
                        "--depfile", str(depfile),
                    ],
                    check=True,
                )
                return output.read_text()

            baseline = generate()
            for name in (*FUSION_ALLOCATION_SOURCES,
                         *UNDISTORTION_ALLOCATION_SOURCES, *EXTRACTION_ALLOCATION_SOURCES, *MATCHING_ALLOCATION_SOURCES, *MAPPING_ALLOCATION_SOURCES):
                with self.subTest(source=name):
                    source = root / name
                    original = source.read_bytes()
                    try:
                        source.write_bytes(
                            original + b"\n// allocation mutation test\n"
                        )
                        self.assertNotEqual(baseline, generate())
                        self.assertIn(str(source.resolve()), depfile.read_text())
                    finally:
                        source.write_bytes(original)
            self.assertEqual(baseline, generate())


if __name__ == "__main__":
    unittest.main()
