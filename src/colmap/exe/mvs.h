// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/fusion.h"
#include "colmap/mvs/optimized_photo_delivery.h"
#include "colmap/mvs/optimized_photo_input_probe.h"
#include "colmap/mvs/patch_match_execution.h"
#include "colmap/mvs/patch_match_options.h"
#include "colmap/scene/reconstruction.h"

#include <filesystem>
#include <functional>

namespace colmap {

void RunPatchMatchStereoImpl(const std::filesystem::path& workspace_path,
                             const std::string& workspace_format,
                             const std::string& pmvs_option_name,
                             const mvs::PatchMatchOptions& options,
                             const std::filesystem::path& config_path);

// Shared parser/controller path for the optional one-reference experiment.
// Ordinary CLI callers supply typed per-job bindings in the V3 wire.
// Native RUNNER callers may supply staged-plan objects via runner_delivery.
// phase_delivery remains the already-attested embedding seam; do not supply
// both.
int RunPatchMatchStereoWithExecution(
    int argc,
    char** argv,
    mvs::PatchMatchExecution execution,
    const mvs::TrustedPhotoPhaseV1Delivery& phase_delivery = {},
    const mvs::OptimizedPhotoInputProbe& input_probe = {},
    const mvs::PhotoPhaseV1RunnerDelivery& runner_delivery = {});
void RunPatchMatchStereoImplWithExecution(
    const std::filesystem::path& workspace_path,
    const std::string& workspace_format,
    const std::string& pmvs_option_name,
    const mvs::PatchMatchOptions& options,
    const std::filesystem::path& config_path,
    mvs::PatchMatchExecution execution);

Reconstruction RunStereoFuserImpl(const std::filesystem::path& output_path,
                                  const std::filesystem::path& workspace_path,
                                  std::string workspace_format,
                                  const std::string& pmvs_option_name,
                                  std::string input_type,
                                  const mvs::StereoFusionOptions& options,
                                  std::string output_type,
                                  std::function<bool()> check_if_stopped = {});

int RunAdvancingFrontMesher(int argc, char** argv);
int RunDelaunayMesher(int argc, char** argv);
int RunMeshSimplifier(int argc, char** argv);
int RunMeshTexturer(int argc, char** argv);
int RunPatchMatchStereo(int argc, char** argv);
int RunPoissonMesher(int argc, char** argv);
int RunStereoFuser(int argc, char** argv);

}  // namespace colmap
