// SPDX-License-Identifier: BSD-3-Clause
// Real-CLI/regression tests for the CPU-only owned-bank metadata modes.
// Every path exercises the actual controller resolution and the actual
// standard CLI argument parser; contract bytes are never mock-derived.
#include "colmap/exe/mvs.h"

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/owned_input_bank.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/resident_contract.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/misc.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <fstream>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <gtest/gtest.h>

namespace colmap {
namespace {

mvs::PatchMatchOptions CliOptions() {
  mvs::PatchMatchOptions options;
  options.gpu_index = "0";
  options.num_threads = 2;
  options.geom_consistency = true;
  options.window_radius = 5;
  options.window_step = 1;
  // Explicit valid depth range: the synthesized sparse model's auto range
  // is invalid in this tiny fixture, so controller-based tests pass a
  // known-good range matching the photometric map payload (0.5..10). The
  // product's auto-range guard is untouched.
  options.depth_min = 0.5;
  options.depth_max = 10.0;
  return options;
}

int RunArgs(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  return RunPatchMatchStereoWithExecution(
      static_cast<int>(argv.size()),
      argv.data(),
      mvs::PatchMatchExecution::kStock);
}

// Real COLMAP workspace with synthetic poses/points, photometric inputs for
// every image, and an explicit patch-match.cfg given per test.
class ResidentCliContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = CreateTestDir();
    CreateDirIfNotExists(root_ / "images");
    // All required parents explicitly: the dir-requiring writers (sparse
    // reconstruction write) and the helper's non-recursive default.
    CreateDirIfNotExists(root_ / "sparse", /*recursive=*/true);
    CreateDirIfNotExists(root_ / "stereo", /*recursive=*/true);
    CreateDirIfNotExists(root_ / "stereo" / "depth_maps");
    CreateDirIfNotExists(root_ / "stereo" / "normal_maps");

    SyntheticDatasetOptions synth_options;
    synth_options.num_rigs = 1;
    synth_options.num_cameras_per_rig = 1;
    synth_options.num_frames_per_rig = 5;
    synth_options.camera_width = 10;
    synth_options.camera_height = 5;
    Reconstruction reconstruction;
    SynthesizeDataset(synth_options, &reconstruction);
    reconstruction.Write(root_ / "sparse");

    for (const auto& image_entry : reconstruction.Images()) {
      const std::string name = image_entry.second.Name();
      names_.push_back(name);
      Bitmap bitmap(10, 5, true);
      bitmap.Fill(BitmapColor<uint8_t>(10, 20, 30));
      ASSERT_TRUE(bitmap.Write(root_ / "images" / name));
      mvs::DepthMap depth(10, 5, 0.5f, 10.0f);
      depth.Fill(1.0f);
      depth.Write(root_ / "stereo" / "depth_maps" /
                  (name + ".photometric.bin"));
      mvs::NormalMap normal(10, 5);
      normal.Fill(0.5f);
      normal.Write(root_ / "stereo" / "normal_maps" /
                   (name + ".photometric.bin"));
    }
    ASSERT_EQ(names_.size(), 5);
  }

  std::filesystem::path WriteConfig(
      const std::vector<std::pair<int, std::vector<int>>>& problems) {
    std::ofstream cfg_file(root_ / "stereo" / "patch-match.cfg");
    for (const auto& problem : problems) {
      cfg_file << names_.at(problem.first) << "\n";
      std::string sources;
      for (const int source : problem.second) {
        sources += (sources.empty() ? "" : ",");
        sources += names_.at(source);
      }
      cfg_file << sources << "\n";
    }
    cfg_file.close();
    return root_;
  }

  const char* ModeFlag(const bool verify) const {
    return verify ? "--owned_metadata_verify" : "--owned_metadata_export";
  }

  int RunCli(const std::filesystem::path& workspace,
             const std::filesystem::path& contract_path,
             const bool verify,
             const std::vector<std::string>& extra_options = {}) {
    // Valid explicit depth range for the synthetic fixture (the synthesized
    // sparse model's auto depth range is invalid, so every valid-baseline
    // export passes it explicitly, matching the photometric input maps). No
    // product weakening: the real range guard stays active.
    std::vector<std::string> args = {
        "colmap",
        "--workspace_path",
        workspace.string(),
        "--PatchMatchStereo.gpu_index",
        "0",
        "--PatchMatchStereo.num_threads",
        "2",
        "--PatchMatchStereo.depth_min",
        "0.5",
        "--PatchMatchStereo.depth_max",
        "10",
        ModeFlag(verify),
        contract_path.string()};
    args.insert(args.end(), extra_options.begin(), extra_options.end());
    return RunArgs(args);
  }

  std::string ReadContractBytes(const std::filesystem::path& path) {
    std::vector<char> data;
    ReadBinaryBlob(path, &data);
    return std::string(data.begin(), data.end());
  }

  // Runs the CLI with an EXPLICIT (non-duplicate) option list, for tests
  // that need option overrides without boost repeated-option errors.
  int RunCliArgs(
      const std::filesystem::path& contract_path,
      const bool verify,
      const std::vector<std::pair<std::string, std::string>>& option_pairs) {
    std::vector<std::string> args = {"colmap",
                                     "--workspace_path",
                                     root_.string()};
    for (const auto& option : option_pairs) {
      args.push_back(option.first);
      args.push_back(option.second);
    }
    args.push_back(ModeFlag(verify));
    args.push_back(contract_path.string());
    return RunArgs(args);
  }

  // Renames one model image to an exact custom name and rewrites the sparse
  // reconstruction, so a fixture can bind problem names/outputs to crafted
  // path forms. Returns the new name.
  std::string RenameImage(const std::string& old_name,
                          const std::string& new_name) {
    Reconstruction reconstruction;
    reconstruction.Read(root_ / "sparse");
    image_t renamed_id = kInvalidImageId;
    for (const auto& image_entry : reconstruction.Images()) {
      if (image_entry.second.Name() == old_name) {
        renamed_id = image_entry.first;
      }
    }
    THROW_CHECK_NE(renamed_id, kInvalidImageId)
        << "No image named " << old_name << " in the sparse model";
    reconstruction.Image(renamed_id).SetName(new_name);
    reconstruction.Write(root_ / "sparse");
    return new_name;
  }

  // Converts the sparse reconstruction to the text format and doubles the
  // first camera parameter (focal length), forcing a REAL calibration change
  // that the model rereads on re-resolution.
  void DoubleSparseFocalLength() {
    const auto sparse = root_ / "sparse";
    Reconstruction reconstruction;
    reconstruction.Read(sparse);
    reconstruction.WriteText(sparse);
    for (const char* file : {"cameras.bin", "images.bin", "points3D.bin",
                             "rigs.bin", "frames.bin"}) {
      std::filesystem::remove(sparse / file);
    }
    std::ifstream in(sparse / "cameras.txt");
    THROW_CHECK_FILE_OPEN(in, sparse / "cameras.txt");
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
      lines.push_back(line);
    }
    in.close();
    std::ofstream out(sparse / "cameras.txt");
    bool first_camera_line = true;
    for (auto& to_edit : lines) {
      const auto first = to_edit.find_first_not_of(" \t\r");
      if (first == std::string::npos || to_edit[first] == '#') {
        out << to_edit << '\n';
        continue;
      }
      std::vector<std::string> tokens;
      size_t cursor = 0;
      while (true) {
        const size_t end = to_edit.find(' ', cursor);
        tokens.push_back(to_edit.substr(
            cursor, end == std::string::npos ? end : end - cursor));
        if (end == std::string::npos) break;
        cursor = end + 1;
      }
      // cameras.txt line: CAMERA_ID MODEL WIDTH HEIGHT params...
      if (first_camera_line && tokens.size() >= 5) {
        const double focal = std::stod(tokens.at(4));
        tokens[4] = StringPrintf("%.17g", focal * 2.0);
        first_camera_line = false;
      }
      std::string edited;
      for (size_t i = 0; i < tokens.size(); ++i) {
        edited += (i == 0 ? "" : " ");
        edited += tokens.at(i);
      }
      out << edited << "\n";
    }
    out.close();
    THROW_CHECK(!first_camera_line) << "No camera line found in cameras.txt";
  }

  std::filesystem::path root_;
  std::vector<std::string> names_;
};

TEST_F(ResidentCliContractTest, ExportAndVerifyTwoReferences) {
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  const auto contract_path = root_ / "two_ref_contract.rc";

  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/false), EXIT_SUCCESS);
  THROW_CHECK(ExistsFile(contract_path));

  const auto export_bytes = ReadContractBytes(contract_path);
  const auto parsed = mvs::ParseOwnedBankContract(export_bytes);
  ASSERT_EQ(parsed.problems.size(), 2);
  EXPECT_EQ(parsed.problems[0].reference_name, names_.at(0));
  EXPECT_EQ(parsed.problems[1].reference_name, names_.at(1));
  for (const auto& problem : parsed.problems) {
    ASSERT_GE(problem.inputs.size(), 2u);
    size_t reference_count = 0;
    for (const auto& input : problem.inputs) {
      reference_count += input.is_reference ? 1 : 0;
    }
    EXPECT_EQ(reference_count, 1u);
    // Effective per-problem geometry options are bound from the invocation.
    EXPECT_TRUE(problem.effective_options.geom_consistency);
    EXPECT_EQ(problem.effective_options.window_radius, 5);
    EXPECT_EQ(problem.effective_options.window_step, 1);
    // Map headers are captured from the real photometric files.
    EXPECT_EQ(problem.inputs.front().depth_header_width, 10u);
    EXPECT_EQ(problem.inputs.front().depth_header_height, 5u);
    EXPECT_EQ(problem.inputs.front().depth_header_depth, 1u);
    EXPECT_EQ(problem.inputs.front().normal_header_depth, 3u);
    // Invocation snapshot identities are real captured file records.
    EXPECT_GT(problem.inputs.front().image_identity.size, 0u);
  }

  // Fresh same-native-resolution verification binds exactly.
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/true), EXIT_SUCCESS);
}

TEST_F(ResidentCliContractTest, ExportAndVerifyFourReferences) {
  WriteConfig({{0, {1, 2, 3, 4}},
               {1, {0, 2}},
               {2, {0, 1, 3, 4}},
               {3, {0, 1, 2, 4}}});
  const auto contract_path = root_ / "four_ref_contract.rc";

  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/false), EXIT_SUCCESS);
  const auto parsed = mvs::ParseOwnedBankContract(ReadContractBytes(contract_path));
  ASSERT_EQ(parsed.problems.size(), 4);
  for (size_t i = 0; i < parsed.problems.size(); ++i) {
    const auto& problem = parsed.problems[i];
    EXPECT_EQ(problem.reference_name, names_.at(i));
    // Exactly one ordered input is the reference, at a native position.
    const auto reference_it = std::find_if(problem.inputs.begin(),
                                           problem.inputs.end(),
                                           [](const auto& input) {
                                             return input.is_reference;
                                           });
    ASSERT_FALSE(reference_it == problem.inputs.end());
    EXPECT_EQ(reference_it->image_name, problem.reference_name);
    // Ordered inputs are the reference plus the configured dedup sources.
    const size_t source_count = problem.inputs.size() - 1;
    EXPECT_EQ(problem.effective_options.filter_min_num_consistent,
              std::min(2, static_cast<int>(source_count)));
  }
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/true), EXIT_SUCCESS);
}

TEST_F(ResidentCliContractTest, OneReferenceWrapperRejectsBeforeTraversal) {
  // Two problems where the SECOND would fail per-job validation (occupied
  // output). The size==1 wrapper must reject on the configured count BEFORE
  // any per-job traversal, so the failure stays the size guard, never the
  // later per-job error.
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  {
    std::ofstream occupied(root_ / "stereo" / "depth_maps" /
                           (names_.at(1) + ".geometric.bin"));
    occupied << "junk";
  }
  auto owned_controller = mvs::PatchMatchController(
      CliOptions(), root_, "colmap", "option-all", "",
      mvs::PatchMatchExecution::kOwnedCuda);
  try {
    owned_controller.ResolveOwnedBankJob();
    FAIL() << "expected the configured problem count guard to reject";
  } catch (const std::exception& exc) {
    EXPECT_THAT(exc.what(), testing::HasSubstr("problems_.size()"));
    EXPECT_THAT(exc.what(), ::testing::Not(::testing::HasSubstr("Owned geometry")));
  }
}

TEST_F(ResidentCliContractTest, OneReferenceWrapperCompatibility) {
  const auto workspace = WriteConfig({{2, {0, 1}}});
  auto stock_controller = mvs::PatchMatchController(
      CliOptions(), workspace, "colmap", "option-all", "",
      mvs::PatchMatchExecution::kStock);
  const auto all_jobs = stock_controller.ResolveOwnedBankJobs();
  ASSERT_EQ(all_jobs.size(), 1);

  auto owned_controller = mvs::PatchMatchController(
      CliOptions(), workspace, "colmap", "option-all", "",
      mvs::PatchMatchExecution::kOwnedCuda);
  const auto wrapper_job = owned_controller.ResolveOwnedBankJob();
  // The size-one wrapper must resolve exactly the shared helper's snapshot.
  EXPECT_EQ(wrapper_job.reference_image_idx,
            all_jobs.front().reference_image_idx);
  EXPECT_EQ(wrapper_job.source_image_idxs, all_jobs.front().source_image_idxs);
  ASSERT_EQ(wrapper_job.inputs.size(), all_jobs.front().inputs.size());
  for (size_t i = 0; i < wrapper_job.inputs.size(); ++i) {
    EXPECT_EQ(wrapper_job.inputs[i].image_idx,
              all_jobs.front().inputs[i].image_idx);
    EXPECT_EQ(wrapper_job.inputs[i].depth_path.string(),
              all_jobs.front().inputs[i].depth_path.string());
    EXPECT_EQ(wrapper_job.inputs[i].normal_path.string(),
              all_jobs.front().inputs[i].normal_path.string());
  }
  EXPECT_EQ(wrapper_job.depth_map_path, all_jobs.front().depth_map_path);
  EXPECT_EQ(wrapper_job.normal_map_path, all_jobs.front().normal_map_path);
  // Resolved effective options are identical on both paths.
  EXPECT_EQ(wrapper_job.options.depth_min, all_jobs.front().options.depth_min);
  EXPECT_EQ(wrapper_job.options.depth_max, all_jobs.front().options.depth_max);
  EXPECT_EQ(wrapper_job.options.sigma_spatial,
            all_jobs.front().options.sigma_spatial);
  EXPECT_EQ(wrapper_job.options.filter_min_num_consistent,
            all_jobs.front().options.filter_min_num_consistent);
  EXPECT_EQ(wrapper_job.options.num_threads,
            all_jobs.front().options.num_threads);
}

TEST_F(ResidentCliContractTest, ProjectPathWithMetadataFlagsRefusedBeforeStock) {
  // Real safety regression: an explicit metadata-mode request combined with
  // --project_path must be REFUSED up front. The project-file reread
  // discards CLI option values, so without the refusal the mode flag would
  // be silently lost and the STOCK solver would start (decode + CUDA).
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  {
    std::ofstream project_file(root_ / "project.ini");
    project_file << "[PatchMatchStereo]\nwindow_radius=5\n";
  }
  const auto contract_path = root_ / "project_refused.rc";

  const int exit_code = RunArgs({
      "colmap",
      "--workspace_path",
      root_.string(),
      "--project_path",
      (root_ / "project.ini").string(),
      "--PatchMatchStereo.gpu_index",
      "0",
      "--PatchMatchStereo.num_threads",
      "2",
      "--PatchMatchStereo.depth_min",
      "0.5",
      "--PatchMatchStereo.depth_max",
      "10",
      "--owned_metadata_export",
      contract_path.string()});
  EXPECT_EQ(exit_code, EXIT_FAILURE);
  // No contract file and NO stock decode: the stock pipeline would have
  // written geometric outputs before any CUDA stage; the refusal must leave
  // the workspace untouched (all fixture outputs are photometric only).
  EXPECT_FALSE(ExistsFile(contract_path));
  for (const auto& entry :
       std::filesystem::directory_iterator(root_ / "stereo" /
                                           "depth_maps")) {
    const auto name = entry.path().filename().string();
    EXPECT_EQ(name.find("geometric"), std::string::npos)
        << "stock pipeline ran despite the refusal: " << name;
  }

  // The verify mode gets the identical fail-closed treatment against a
  // prior contract; nothing runs and nothing changes.
  EXPECT_EQ(RunCli(root_, root_ / "prior.rc", /*verify=*/false),
            EXIT_SUCCESS);
  const auto prior_bytes = ReadContractBytes(root_ / "prior.rc");
  EXPECT_EQ(RunArgs({
                "colmap",
                "--workspace_path",
                root_.string(),
                "--project_path",
                (root_ / "project.ini").string(),
                "--PatchMatchStereo.gpu_index",
                "0",
                "--PatchMatchStereo.num_threads",
                "2",
                "--PatchMatchStereo.depth_min",
                "0.5",
                "--PatchMatchStereo.depth_max",
                "10",
                "--owned_metadata_verify",
                (root_ / "prior.rc").string()}),
            EXIT_FAILURE);
  EXPECT_EQ(ReadContractBytes(root_ / "prior.rc"), prior_bytes);
}

TEST_F(ResidentCliContractTest, RejectDuplicateReferencesAndAliases) {
  // Same reference listed twice: derived output aliases collide.
  WriteConfig({{0, {1, 2}}, {0, {1}}});
  EXPECT_EQ(RunCli(root_, root_ / "dup.rc", /*verify=*/false), EXIT_FAILURE);
}

TEST_F(ResidentCliContractTest, RejectsTraversalNameOutputAliases) {
  // Deterministic output-ALIAS case (replaces the old symlink-parent test,
  // whose removal of a non-empty seed directory was unsafe and whose early
  // failures could have a wrong reason): the reference name built with a
  // traversal component resolves onto the OTHER reference's planned output
  // path canonical form, while every input (bitmap/depth/normal) still
  // resolves validly (the traversal target files are the shared seeds of
  // the base-named reference). Both problems must capture their inputs fine
  // and then hit EXACTLY the duplicate-output-alias rejection. No directory
  // removals ever happen.
  const auto base_name = names_.at(1);
  const auto source_name = names_.at(2);
  const auto alias_name = "aliasdir/../" + base_name;
  RenameImage(names_.at(0), alias_name);
  // Intermediate traversal component must exist as a directory wherever the
  // model name becomes a path (bitmap and depth/normal consumer dirs).
  for (const char* sub : {"images", "stereo/depth_maps",
                          "stereo/normal_maps"}) {
    CreateDirIfNotExists(root_ / sub / "aliasdir", /*recursive=*/true);
  }
  std::ofstream cfg_file(root_ / "stereo" / "patch-match.cfg");
  cfg_file << alias_name << "\n" << source_name << "\n";
  cfg_file << base_name << "\n" << source_name << "\n";
  cfg_file.close();
  const auto contract_path = root_ / "alias.rc";
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/false), EXIT_FAILURE);
  EXPECT_FALSE(ExistsFile(contract_path));
}

TEST_F(ResidentCliContractTest, ExportTargetCollidesWithPlannedOutputsRejected) {
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  // Targeting a planned (not yet occupied) solver output directly.
  const auto planned_output = root_ / "stereo" / "depth_maps" /
                              (names_.at(1) + ".geometric.bin");
  EXPECT_EQ(RunCli(root_, planned_output, /*verify=*/false), EXIT_FAILURE);
  EXPECT_FALSE(ExistsFile(planned_output));
}

TEST_F(ResidentCliContractTest, ExportRefusesExistingFileWithoutOverwrite) {
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  const auto existing = root_ / "existing.rc";
  {
    std::ofstream existing_file(existing);
    existing_file << "precious-content";
  }
  EXPECT_EQ(RunCli(root_, existing, /*verify=*/false), EXIT_FAILURE);
  std::vector<char> content;
  ReadBinaryBlob(existing, &content);
  EXPECT_EQ(std::string(content.begin(), content.end()), "precious-content");
}

TEST_F(ResidentCliContractTest, ExportRefusesUnwritableParentFailClosed) {
  // Privilege-sensitive: a root process can create files in read-only
  // directories, making this scenario non-deterministic. Skip the PERMISSION
  // aspect only, with a clear reason; semantic tests above are never masked.
  if (::geteuid() == 0) {
    GTEST_SKIP() << "running as root: read-only-parent creation is "
                    " privilege-sensitive and skipped under root";
  }
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  // Exclusive creation in an unwritable parent fails closed at open time;
  // a pre-existing target elsewhere is untouched and no file is created.
  CreateDirIfNotExists(root_ / "readonly", /*recursive=*/true);
  std::filesystem::permissions(root_ / "readonly",
                               std::filesystem::perms::owner_read |
                                   std::filesystem::perms::owner_exec,
                               std::filesystem::perm_options::replace);
  EXPECT_EQ(RunCli(root_, root_ / "readonly" / "contract.rc",
                   /*verify=*/false),
            EXIT_FAILURE);
  EXPECT_FALSE(ExistsFile(root_ / "readonly" / "contract.rc"));
  std::filesystem::permissions(root_ / "readonly",
                               std::filesystem::perms::owner_all,
                               std::filesystem::perm_options::replace);
}

TEST_F(ResidentCliContractTest, SharedInputsCarryConsistentIdentities) {
  // Two problems sharing image 2 as a source: the shared image's three
  // input files must be captured with exactly one identity across problems.
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  const auto contract_path = root_ / "shared.rc";
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/false), EXIT_SUCCESS);
  std::vector<char> bytes_vec;
  ReadBinaryBlob(contract_path, &bytes_vec);
  const auto parsed =
      mvs::ParseOwnedBankContract(std::string(bytes_vec.begin(), bytes_vec.end()));
  ASSERT_EQ(parsed.problems.size(), 2);
  std::map<std::string, std::pair<uintmax_t, uintmax_t>> identities;
  auto record_input = [&identities](
                            const mvs::OwnedBankContractInput& input) {
    for (const auto* identity : {&input.image_identity, &input.depth_identity,
                                 &input.normal_identity}) {
      const std::pair<uintmax_t, uintmax_t> key = {identity->size,
                                                   identity->inode};
      const auto inserted = identities.emplace(identity->path.string(), key);
      EXPECT_TRUE(inserted.second || inserted.first->second == key)
          << "Inconsistent identity for shared input " << identity->path;
    }
  };
  for (const auto& problem : parsed.problems) {
    for (const auto& input : problem.inputs) {
      record_input(input);
    }
  }
  // 3 shared images (0, 1, 2) x 3 files each = 9 unique file identities.
  EXPECT_EQ(identities.size(), 9u);
}

TEST_F(ResidentCliContractTest, VerifyFailsOnContentOnlySameShapeMutation) {
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  const auto contract_path = root_ / "same_shape.rc";
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/false), EXIT_SUCCESS);
  // Rewrite one photometric map with the SAME dims but different content:
  // only the file object changes; verification must fail before payload.
  mvs::DepthMap depth(10, 5, 0.5f, 10.0f);
  depth.Fill(7.0f);
  depth.Write(root_ / "stereo" / "depth_maps" /
              (names_.at(2) + ".photometric.bin"));
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/true), EXIT_FAILURE);
}

TEST_F(ResidentCliContractTest, RejectOccupiedOutputs) {
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  // Pre-existing output for a configured reference.
  std::ofstream occupied(root_ / "stereo" / "depth_maps" /
                         (names_.at(0) + ".geometric.bin"));
  occupied << "junk";
  occupied.close();
  EXPECT_EQ(RunCli(root_, root_ / "occupied.rc", /*verify=*/false),
            EXIT_FAILURE);
}

TEST_F(ResidentCliContractTest, VerifyFailsOnChangedOptionsOrInputs) {
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  const auto contract_path = root_ / "changed.rc";
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/false), EXIT_SUCCESS);

  // A changed invocation option must reject the old contract, with an
  // explicit (non-duplicate) argument list so the failure is the CONTRACT
  // difference, not a boost repeated-option parse error.
  EXPECT_EQ(RunCliArgs(contract_path,
                       /*verify=*/true,
                       {{"--PatchMatchStereo.gpu_index", "0"},
                        {"--PatchMatchStereo.num_threads", "3"},
                        {"--PatchMatchStereo.depth_min", "0.5"},
                        {"--PatchMatchStereo.depth_max", "10"}}),
            EXIT_FAILURE);

  // A changed input file object (rewritten image, same dimensions) changes
  // the captured invocation identity: verify must fail before any payload.
  Bitmap bitmap(10, 5, true);
  bitmap.Fill(BitmapColor<uint8_t>(200, 200, 200));
  ASSERT_TRUE(bitmap.Write(root_ / "images" / names_.at(2)));
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/true), EXIT_FAILURE);
}

TEST_F(ResidentCliContractTest, VerifyFailsOnChangedCalibration) {
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  const auto contract_path = root_ / "calib.rc";
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/false), EXIT_SUCCESS);

  DoubleSparseFocalLength();
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/true), EXIT_FAILURE);
}

TEST_F(ResidentCliContractTest, VerifyFailsOnChangedMapHeader) {
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
  const auto contract_path = root_ / "header.rc";
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/false), EXIT_SUCCESS);

  // Rewrite one photometric input map with different dimensions: header and
  // identity both change; verification must fail before any payload use.
  mvs::DepthMap depth(8, 5, 0.5f, 10.0f);
  depth.Fill(2.0f);
  depth.Write(root_ / "stereo" / "depth_maps" /
              (names_.at(1) + ".photometric.bin"));
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/true), EXIT_FAILURE);
}

TEST_F(ResidentCliContractTest, RejectUnsupportedNegativeFailureReasons) {
  // Negative CLI results must fail for the SPECIFIC intended reason, not an
  // accidental early failure (invalid auto depth range, missing seeds,
  // boost duplicate-option parse errors). The controller API asserts the
  // exact rejection message for each negative class.
  const auto expect_message = [](const std::function<void()>& action,
                                 const std::string& expected) {
    try {
      action();
      FAIL() << "expected rejection containing: " << expected;
    } catch (const std::exception& exc) {
      EXPECT_NE(std::string(exc.what()).find(expected), std::string::npos)
          << "expected message containing \"" << expected << "\", got: "
          << exc.what();
    }
  };

  // Duplicate reference name (output aliases collide together with it).
  {
    WriteConfig({{0, {1, 2}}, {0, {1}}});
    auto controller = mvs::PatchMatchController(
        CliOptions(), root_, "colmap", "option-all", "",
        mvs::PatchMatchExecution::kStock);
    expect_message(
        [&controller] { controller.ResolveOwnedBankJobs(); },
        "Duplicate reference image in configured problems");
  }

  // Occupied output (pre-existing output for a configured reference) with
  // an otherwise-valid configuration.
  {
    WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
    std::ofstream occupied(root_ / "stereo" / "depth_maps" /
                           (names_.at(0) + ".geometric.bin"));
    occupied << "junk";
    occupied.close();
    auto controller = mvs::PatchMatchController(
        CliOptions(), root_, "colmap", "option-all", "",
        mvs::PatchMatchExecution::kStock);
    expect_message(
        [&controller] { controller.ResolveOwnedBankContract(); },
        "!ExistsFile(result.depth_map_path)");
  }

  // Traversal-name output alias: valid inputs must capture, the rejection
  // must be the resolved-path alias check, not a capture/header/range error.
  {
    const auto base_name = names_.at(1);
    const auto source_name = names_.at(2);
    const auto alias_name = "aliasdir/../" + base_name;
    RenameImage(names_.at(0), alias_name);
    for (const char* sub : {"images", "stereo/depth_maps",
                            "stereo/normal_maps"}) {
      CreateDirIfNotExists(root_ / sub / "aliasdir", /*recursive=*/true);
    }
    std::ofstream cfg_file(root_ / "stereo" / "patch-match.cfg");
    cfg_file << alias_name << "\n" << source_name << "\n";
    cfg_file << base_name << "\n" << source_name << "\n";
    cfg_file.close();
    auto controller = mvs::PatchMatchController(
        CliOptions(), root_, "colmap", "option-all", "",
        mvs::PatchMatchExecution::kStock);
    expect_message(
        [&controller] { controller.ResolveOwnedBankJobs(); },
        "Duplicate owned output path across problems");
  }

  // Missing geometry.
  {
    WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
    auto options = CliOptions();
    options.geom_consistency = false;
    auto controller = mvs::PatchMatchController(
        options, root_, "colmap", "option-all", "",
        mvs::PatchMatchExecution::kStock);
    expect_message(
        [&controller] { controller.ResolveOwnedBankContract(); },
        "options_.geom_consistency");
  }

  // Unsupported solver-window options.
  {
    WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
    auto options = CliOptions();
    options.window_radius = 2;
    auto controller = mvs::PatchMatchController(
        options, root_, "colmap", "option-all", "",
        mvs::PatchMatchExecution::kStock);
    expect_message(
        [&controller] { controller.ResolveOwnedBankContract(); },
        "Owned PatchMatch requires CUDA and window radius 5 / step 1");
  }

  // PMVS workspace format.
  {
    WriteConfig({{0, {1, 2}}, {1, {0, 2}}});
    auto controller = mvs::PatchMatchController(
        CliOptions(), root_, "pmvs", "option-all", "",
        mvs::PatchMatchExecution::kStock);
    expect_message(
        [&controller] { controller.ResolveOwnedBankJobs(); },
        "Resident input mode supports only COLMAP workspaces");
  }
}

TEST_F(ResidentCliContractTest, RejectsUnsupportedModesAndMalformed) {
  WriteConfig({{0, {1, 2}}, {1, {0, 2}}});

  // Both mode flags at once are rejected outright.
  EXPECT_EQ(RunArgs({"colmap",
                     "--workspace_path",
                     root_.string(),
                     "--owned_metadata_export",
                     (root_ / "x.rc").string(),
                     "--owned_metadata_verify",
                     (root_ / "x.rc").string()}),
            EXIT_FAILURE);

  // Unknown/misnamed flags fail the standard parser.
  EXPECT_EQ(RunArgs({"colmap",
                     "--workspace_path",
                     root_.string(),
                     "--owned_metadata_expo",
                     (root_ / "y.rc").string()}),
            EXIT_FAILURE);

  // Geometry is required for the owned metadata contract.
  EXPECT_EQ(RunCliArgs((root_ / "nogeom.rc").string(),
                       /*verify=*/false,
                       {{"--PatchMatchStereo.gpu_index", "0"},
                        {"--PatchMatchStereo.num_threads", "2"},
                        {"--PatchMatchStereo.geom_consistency", "0"},
                        {"--PatchMatchStereo.depth_min", "0.5"},
                        {"--PatchMatchStereo.depth_max", "10"}}),
            EXIT_FAILURE);

  // Unsupported solver-window options are rejected before any binding (the
  // full explicit argument list avoids boost repeated-option errors).
  EXPECT_EQ(RunCliArgs((root_ / "window.rc").string(),
                       /*verify=*/false,
                       {{"--PatchMatchStereo.gpu_index", "0"},
                        {"--PatchMatchStereo.num_threads", "2"},
                        {"--PatchMatchStereo.window_radius", "2"},
                        {"--PatchMatchStereo.depth_min", "0.5"},
                        {"--PatchMatchStereo.depth_max", "10"}}),
            EXIT_FAILURE);

  // PMVS workspaces are unsupported for this metadata-only route.
  EXPECT_EQ(RunCliArgs((root_ / "pmvs.rc").string(),
                       /*verify=*/false,
                       {{"--workspace_format", "PMVS"},
                        {"--PatchMatchStereo.gpu_index", "0"},
                        {"--PatchMatchStereo.num_threads", "2"},
                        {"--PatchMatchStereo.depth_min", "0.5"},
                        {"--PatchMatchStereo.depth_max", "10"}}),
            EXIT_FAILURE);

  // A malformed/truncated stored contract is rejected, never trusted.
  const auto contract_path = root_ / "truncated.rc";
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/false), EXIT_SUCCESS);
  std::vector<char> bytes;
  ReadBinaryBlob(contract_path, &bytes);
  bytes.pop_back();
  WriteBinaryBlob(contract_path, {bytes.data(), bytes.size()});
  EXPECT_EQ(RunCli(root_, contract_path, /*verify=*/true), EXIT_FAILURE);
}

}  // namespace
}  // namespace colmap