// SPDX-License-Identifier: BSD-3-Clause
// CPU callbacks exercise the real parser, epoch loop and owner retirement.
// Linked with the optional driver, but never construct a CUDA controller.
#define main renamed_owned_pair_test_main
#include "colmap/exe/owned_pair.cc"
#undef main

#include "colmap/util/testing.h"

#include <atomic>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>

#include <gtest/gtest.h>

namespace colmap::mvs::owned {

// No event instances or owner threads run while this test-only guard exchanges
// the token. Restore even on a fatal GTest assertion; never modify the clock.
struct PairEventsTestAccess {
  explicit PairEventsTestAccess(uint64_t initial)
      : saved(PairEvents::last_batch_token_.exchange(initial)) {}
  ~PairEventsTestAccess() { PairEvents::last_batch_token_.store(saved); }
  static uint64_t Next(uint64_t now) { return PairEvents::NextBatchToken(now); }
  static uint64_t Current() { return PairEvents::last_batch_token_.load(); }
  uint64_t saved;
};

}  // namespace colmap::mvs::owned

namespace colmap {
namespace {

// PairEvents writes directly to fd 1, not std::cout. A file avoids a bounded
// pipe filling while synchronous RunPairs waits for its owners to join.
class EventCapture {
 public:
  EventCapture() {
    std::cout.flush();
    file_ = std::tmpfile();
    if (!file_) throw std::runtime_error("tmpfile failed");
    saved_ = dup(STDOUT_FILENO);
    if (saved_ < 0 || dup2(fileno(file_), STDOUT_FILENO) < 0) {
      if (saved_ >= 0) close(saved_);
      std::fclose(file_);
      throw std::runtime_error("stdout capture failed");
    }
  }
  ~EventCapture() {
    std::cout.flush();
    dup2(saved_, STDOUT_FILENO);
    close(saved_);
    std::fclose(file_);
  }
  std::string Read() {
    std::cout.flush();
    std::rewind(file_);
    std::string result;
    char buffer[4096];
    while (const size_t n = std::fread(buffer, 1, sizeof(buffer), file_)) {
      result.append(buffer, n);
    }
    return result;
  }

 private:
  FILE* file_ = nullptr;
  int saved_ = -1;
};

using Record = std::map<std::string, std::string>;
std::vector<Record> Records(const std::string& text) {
  std::vector<Record> records;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.rfind("OWNED_PAIR_EVENT ", 0) != 0) continue;
    Record record;
    std::istringstream fields(line);
    std::string field;
    while (fields >> field) {
      const size_t separator = field.find('=');
      if (separator != std::string::npos) {
        record.emplace(field.substr(0, separator), field.substr(separator + 1));
      }
    }
    records.push_back(std::move(record));
  }
  return records;
}

void Put(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream file(path);
  file << text;
  if (!file) throw std::runtime_error("test fixture write failed");
}

class OwnedPairTest : public testing::Test {
 protected:
  void SetUp() override {
    root_ = CreateTestDir();
    for (size_t j = 0; j < 4; ++j) WriteJob(j);
  }
  void TearDown() override { fs::remove_all(root_); }

  fs::path Ini(size_t j) const {
    return root_ / "list with spaces" / ("job " + Value(j) + ".ini");
  }
  fs::path Workspace(size_t j) const {
    return root_ / ("workspace " + Value(j));
  }
  void WriteJob(size_t j,
                const std::string& extra = "",
                fs::path workspace = {},
                std::string reference = "") {
    if (workspace.empty()) workspace = Workspace(j);
    if (reference.empty()) reference = "reference " + Value(j) + ".jpg";
    fs::create_directories(workspace / "images");
    fs::create_directories(workspace / "sparse");
    for (const auto* stem : {"cameras", "images", "points3D"}) {
      Put(workspace / "sparse" / (std::string(stem) + ".txt"), "");
    }
    // Prevalidation checks existence only; the CPU callback supplies metadata
    // instead of decoding these placeholders or reading a sparse model.
    Put(workspace / "images" / reference, "placeholder");
    const auto config = root_ / ("config " + Value(j) + ".cfg");
    Put(config, reference + "\n__all__\n");
    Put(Ini(j),
        "workspace_path=" + workspace.string() + "\nconfig_path=" +
            config.string() + "\n" + extra +
            "[PatchMatchStereo]\nnum_threads=1\ncache_size=0.000001\n"
            "geom_consistency=false\ngpu_index=0\n");
  }
  void SetPatchMatchOption(size_t j,
                           const std::string& name,
                           const std::string& value) {
    std::ifstream input(Ini(j));
    if (!input) throw std::runtime_error("test INI read failed");
    std::string text((std::istreambuf_iterator<char>(input)), {});
    const std::string prefix = "\n" + name + "=";
    const auto start = text.find(prefix);
    if (start == std::string::npos) {
      text += name + "=" + value + "\n";
    } else {
      const auto end = text.find('\n', start + 1);
      text.replace(start, end - start, prefix + value);
    }
    Put(Ini(j), text);
  }
  fs::path List(const std::vector<size_t>& jobs) {
    const auto path = root_ / "list with spaces" / "projects.txt";
    std::string text;
    for (size_t j : jobs) {
      // Mix relative entries containing spaces and absolute entries.
      text += (j % 2 ? Ini(j).string() : Ini(j).filename().string()) + "\n";
    }
    Put(path, text);
    return path;
  }
  std::vector<std::string> Args() const {
    return {"owned_pair_test", "--dispatch", "pair",
            "--host_budget_bytes", "4000000",
            "--device_budget_bytes", "1024",
            "--job_a_host_charge_bytes", "1048576",
            "--job_b_host_charge_bytes", "2097152",
            "--job_a_device_charge_bytes", "128",
            "--job_b_device_charge_bytes", "256",
            "--process_host_charge_bytes", "16",
            "--process_device_charge_bytes", "16",
            "--io_threads_limit", "1"};
  }
  std::vector<std::string> ListArgs(const std::vector<size_t>& jobs) {
    auto args = Args();
    args.insert(args.end(), {"--job_project_list_path", List(jobs).string()});
    return args;
  }
  template <class RunJob>
  int Invoke(std::vector<std::string> args, RunJob run_job) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return RunPairs(static_cast<int>(argv.size()), argv.data(), run_job);
  }
  template <class RunJob, class PrepareJob>
  int InvokePrepared(std::vector<std::string> args, RunJob run_job,
                     PrepareJob prepare_job) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return RunPairs(static_cast<int>(argv.size()), argv.data(),
                    run_job, prepare_job);
  }
  void Reject(std::vector<std::string> args) {
    std::atomic<size_t> calls{0};
    EventCapture capture;
    EXPECT_THROW(Invoke(std::move(args), [&](auto&, auto&, size_t, const auto&) {
      ++calls;
    }), std::exception);
    EXPECT_EQ(calls.load(), 0);
    for (const auto& record : Records(capture.Read())) {
      EXPECT_NE(record.at("event"), "admitted");
    }
  }

  static void Complete(JobOptions& job,
                       pair::PairCoordinator& coordinator,
                       size_t slot,
                       const mvs::PatchMatchController::OwnedPairHooks& hooks) {
    coordinator.BeginPreparation(slot);
    const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float translation[3] = {0, 0, 0};
    // Use distinct reference indices for the two owners; preceding synthetic
    // entries are unused and never decoded or reported as sources.
    // Do not relocate default Image placeholders: their unused matrix arrays
    // are uninitialized. Reserve before creating them, then append in place.
    std::vector<mvs::Image> images;
    images.reserve(slot + 1);
    images.resize(slot);
    images.emplace_back(job.workspace / "images" / job.reference_name,
                        1, 1, identity, identity, translation);
    mvs::PatchMatch::Problem problem;
    problem.ref_image_idx = static_cast<int>(slot);
    problem.images = &images;
    hooks.metadata(slot, *job.patch_match_stereo, problem);
    coordinator.ReadyAndWait(slot);
    coordinator.SolverDone(slot);
    coordinator.BeginOutput(slot);
    hooks.output_phase(slot, "depth_begin");
    hooks.output_phase(slot, "depth_end");
    hooks.output_phase(slot, "normal_begin");
    hooks.output_phase(slot, "normal_end");
    coordinator.EndOutput(slot);
    coordinator.AwaitOutputs(slot);
    // images and all synthetic resources remain in this owner scope.
  }

  fs::path root_;
};

TEST_F(OwnedPairTest, StaticThreeRealParserOwnerAndEventContract) {
  for (size_t j = 0; j < 3; ++j) {
    SetPatchMatchOption(j, "geom_consistency", "true");
  }
  for (const auto* permits : {"2", "3"}) {
    auto args = ListArgs({0, 1, 2});
    args.insert(args.end(), {"--static_three_solver_permits", permits,
                            "--job_c_host_charge_bytes", "524288",
                            "--job_c_device_charge_bytes", "128"});
    EventCapture capture;
    std::atomic<size_t> calls{0};
    EXPECT_EQ(Invoke(args, [&](auto& job, auto& coordinator, size_t slot,
                              const auto& hooks) {
      ++calls;
      EXPECT_EQ(coordinator.JobCount(), 3);
      Complete(job, coordinator, slot, hooks);
    }), EXIT_SUCCESS);
    EXPECT_EQ(calls.load(), 3);
    const auto records = Records(capture.Read());
    std::set<std::string> ready;
    size_t all_ready = 0, release = 0, retired = 0, joined = 0;
    for (const auto& record : records) {
      const auto& event = record.at("event");
      if (event == "validated") {
        EXPECT_EQ(record.at("job_count"), "3");
        EXPECT_EQ(record.at("static_three_solver_permits"), permits);
        EXPECT_EQ(record.at("hc"), "524288");
        EXPECT_EQ(record.at("dc"), "128");
        EXPECT_EQ(record.at("io_c"), "1");
      }
      if (event == "ready") ready.insert(record.at("job"));
      if (event == "all_ready") ++all_ready;
      EXPECT_NE(event, "both_ready");
      if (event == "retired") ++retired;
      if (event == "owners_joined") ++joined;
      if (event == "ledger_release") {
        ++release;
        EXPECT_EQ(retired, 3);
        EXPECT_EQ(joined, 1);
      }
    }
    EXPECT_EQ(ready, (std::set<std::string>{"A", "B", "C"}));
    EXPECT_EQ(all_ready, 1);
    EXPECT_EQ(release, 1);
  }
}

TEST_F(OwnedPairTest, StaticThreeTwoEpochsCloseBeforeSuccessorAdmission) {
  for (size_t j = 0; j < 6; ++j) {
    if (j >= 4) WriteJob(j);
    SetPatchMatchOption(j, "geom_consistency", "true");
  }
  for (const auto* permits : {"2", "3"}) {
    auto args = ListArgs({0, 1, 2, 3, 4, 5});
    args.insert(args.end(), {"--static_three_solver_permits", permits,
                            "--job_c_host_charge_bytes", "524288",
                            "--job_c_device_charge_bytes", "128"});
    EventCapture capture;
    std::atomic<size_t> calls{0};
    EXPECT_EQ(Invoke(args, [&](auto& job, auto& coordinator, size_t slot,
                              const auto& hooks) {
      const size_t ordinal = calls.fetch_add(1);
      EXPECT_EQ(job.workspace, Workspace(slot + (ordinal >= 3 ? 3 : 0)));
      Complete(job, coordinator, slot, hooks);
    }), EXIT_SUCCESS);
    EXPECT_EQ(calls.load(), 6);
    const auto records = Records(capture.Read());
    std::string batch;
    size_t retired = 0, joined = 0, released = 0, finished = 0, epochs = 0;
    for (const auto& record : records) {
      const auto& event = record.at("event");
      if (event == "validated") {
        if (epochs) {
          EXPECT_EQ(retired, 3);
          EXPECT_EQ(joined, 1);
          EXPECT_EQ(released, 1);
          EXPECT_EQ(finished, 1);
        }
        EXPECT_NE(batch, record.at("batch"));
        batch = record.at("batch");
        retired = joined = released = finished = 0;
        ++epochs;
      }
      EXPECT_EQ(record.at("batch"), batch);
      if (event == "retired") ++retired;
      if (event == "owners_joined") ++joined;
      if (event == "ledger_release") {
        EXPECT_EQ(retired, 3);
        EXPECT_EQ(joined, 1);
        EXPECT_EQ(record.at("reserved_host"), "0");
        EXPECT_EQ(record.at("reserved_device"), "0");
        ++released;
      }
      if (event == "batch_finished") {
        EXPECT_EQ(released, 1);
        ++finished;
      }
    }
    EXPECT_EQ(epochs, 2);
    EXPECT_EQ(finished, 1);
  }
}

TEST_F(OwnedPairTest, StaticThreeFailureClosesAndStopsSuccessor) {
  for (size_t j = 0; j < 6; ++j) {
    if (j >= 4) WriteJob(j);
    SetPatchMatchOption(j, "geom_consistency", "true");
  }
  for (size_t failing : {0, 2, 3, 5}) {
    auto args = ListArgs({0, 1, 2, 3, 4, 5});
    args.insert(args.end(), {"--static_three_solver_permits", "3",
                            "--job_c_host_charge_bytes", "524288",
                            "--job_c_device_charge_bytes", "128"});
    EventCapture capture;
    std::atomic<size_t> calls{0};
    EXPECT_THROW(Invoke(args, [&](auto& job, auto& coordinator, size_t slot,
                                 const auto& hooks) {
      ++calls;
      if (job.workspace == Workspace(failing)) {
        throw std::runtime_error("injected triple failure");
      }
      Complete(job, coordinator, slot, hooks);
    }), std::runtime_error);
    const size_t epochs = failing < 3 ? 1 : 2;
    EXPECT_EQ(calls.load(), epochs * 3);
    size_t admitted = 0, retired = 0, joined = 0, released = 0, failed = 0;
    for (const auto& record : Records(capture.Read())) {
      const auto& event = record.at("event");
      if (event == "admitted") ++admitted;
      if (event == "retired") ++retired;
      if (event == "owners_joined") ++joined;
      if (event == "ledger_release") {
        EXPECT_EQ(retired, joined * 3);
        EXPECT_EQ(record.at("reserved_host"), "0");
        EXPECT_EQ(record.at("reserved_device"), "0");
        ++released;
      }
      if (event == "batch_failed") ++failed;
    }
    EXPECT_EQ(admitted, epochs * 3);
    EXPECT_EQ(retired, epochs * 3);
    EXPECT_EQ(joined, epochs);
    EXPECT_EQ(released, epochs);
    EXPECT_EQ(failed, 1);
  }
}

TEST_F(OwnedPairTest, StaticThreeRejectsUnsupportedModesBeforeOwners) {
  auto triple = [&](std::vector<size_t> slots) {
    auto args = ListArgs(slots);
    args.insert(args.end(), {"--static_three_solver_permits", "2",
                            "--job_c_host_charge_bytes", "524288",
                            "--job_c_device_charge_bytes", "128"});
    return args;
  };
  Reject(triple({0, 1, 2})); // photometric
  for (size_t j = 0; j < 4; ++j) SetPatchMatchOption(j, "geom_consistency", "true");
  Reject(triple({0, 1}));
  Reject(triple({0, 1, 2, 3}));
  auto args = triple({0, 1, 2});
  args[2] = "serial";
  Reject(args);
  args = triple({0, 1, 2});
  args.insert(args.end(), {"--host_lookahead", "0"});
  Reject(args);
  args = triple({0, 1, 2});
  args.insert(args.end(), {"--geometry_early_permits", "1"});
  Reject(args);
  SetPatchMatchOption(2, "geom_consistency", "false");
  Reject(triple({0, 1, 2}));
}

TEST_F(OwnedPairTest, RejectsEmptyOddAndDuplicateLists) {
  Reject(ListArgs({}));
  Reject(ListArgs({0, 1, 2}));
  Reject(ListArgs({0, 1, 0, 3}));
  auto args = ListArgs({0, 1});
  Put(List({0, 1}), "job 0.ini\n\njob 1.ini\n");
  Reject(args);
  Put(List({0, 1}), std::string("job 0.ini") + '\0' + "suffix\njob 1.ini\n");
  Reject(args);
}

TEST_F(OwnedPairTest, RejectsMixedMissingAndAmbiguousArguments) {
  auto mixed = ListArgs({0, 1});
  mixed.insert(mixed.end(), {"--job_a_project_path", Ini(0).string(),
                             "--job_b_project_path", Ini(1).string()});
  Reject(mixed);
  auto partial = Args();
  partial.insert(partial.end(), {"--job_a_project_path", Ini(0).string()});
  Reject(partial);
  Reject(Args());
  auto duplicate = ListArgs({0, 1});
  duplicate.insert(duplicate.end(), {"--dispatch", "pair"});
  Reject(duplicate);
  auto equals = ListArgs({0, 1});
  equals[1] = "--dispatch=pair";
  Reject(equals);
  auto unknown = ListArgs({0, 1});
  unknown.insert(unknown.end(), {"--unknown", "1"});
  Reject(unknown);
}

TEST_F(OwnedPairTest, PreservesLiteralEdgeSpacesAndCrLfListPaths) {
  const auto spaced = Ini(0).parent_path() / " leading and trailing.ini ";
  fs::copy_file(Ini(0), spaced);
  auto args = ListArgs({0, 1});
  Put(List({0, 1}), spaced.filename().string() + "\r\n" +
                        Ini(1).string() + "\r\n");
  std::atomic<size_t> calls{0};
  EventCapture capture;
  EXPECT_EQ(Invoke(args, [&](auto& job, auto& coordinator, size_t slot,
                             const auto& hooks) {
    ++calls;
    Complete(job, coordinator, slot, hooks);
  }), EXIT_SUCCESS);
  EXPECT_EQ(calls.load(), 2);
}

TEST_F(OwnedPairTest, InvalidLaterIniPreventsAllOwners) {
  Put(Ini(3), "workspace_path=" + Workspace(3).string() +
                  "\n[PatchMatchStereo]\nnot_an_option=1\n");
  Reject(ListArgs({0, 1, 2, 3}));
}

TEST_F(OwnedPairTest, LaterConflictingGlobalsDeviceAndReferencePreventAllOwners) {
  WriteJob(3, "default_random_seed=123\n");
  Reject(ListArgs({0, 1, 2, 3}));
  WriteJob(3);
  std::ifstream file(Ini(3));
  std::string text((std::istreambuf_iterator<char>(file)), {});
  text.replace(text.find("gpu_index=0"), std::string("gpu_index=0").size(),
               "gpu_index=1");
  file.close();
  Put(Ini(3), text);
  Reject(ListArgs({0, 1, 2, 3}));
  WriteJob(3, "", {}, "reference 0.jpg");
  Reject(ListArgs({0, 1, 2, 3}));
}

TEST_F(OwnedPairTest, LaterWorkspaceAndOutputAliasesPreventAllOwners) {
  WriteJob(3, "", Workspace(0));
  Reject(ListArgs({0, 1, 2, 3}));
  WriteJob(3);
  fs::create_directories(Workspace(0) / "stereo/depth_maps");
  fs::create_directories(Workspace(3) / "stereo");
  fs::create_directory_symlink(Workspace(0) / "stereo/depth_maps",
                               Workspace(3) / "stereo/depth_maps");
  Reject(ListArgs({0, 1, 2, 3}));
}

TEST_F(OwnedPairTest, CoordinatorConstructionFailurePreventsOwners) {
  auto args = ListArgs({0, 1, 2, 3});
  args[4] = "1000000";  // Less than the atomic pair reservation.
  Reject(args);
}

TEST_F(OwnedPairTest, LegacyTwoIniStillUsesProductionOwners) {
  auto args = Args();
  args.insert(args.end(), {"--job_a_project_path", Ini(0).string(),
                           "--job_b_project_path", Ini(1).string()});
  std::atomic<size_t> calls{0};
  EventCapture capture;
  EXPECT_EQ(Invoke(args, [&](auto& job, auto& coordinator, size_t slot,
                             const auto& hooks) {
    EXPECT_TRUE(job.patch_match_stereo->filter);
    ++calls;
    Complete(job, coordinator, slot, hooks);
  }), EXIT_SUCCESS);
  EXPECT_EQ(calls.load(), 2);
  size_t filter_options = 0;
  for (const auto& record : Records(capture.Read())) {
    if (record.at("event") == "effective_option" &&
        record.at("option") == "filter") {
      EXPECT_EQ(record.at("value_hex"), "31");  // Printed bool: 1.
      ++filter_options;
    }
  }
  EXPECT_EQ(filter_options, 2);
}

TEST_F(OwnedPairTest, ExplicitUnfilteredPhotoReachesBothEpochsAndMetadata) {
  for (size_t j = 0; j < 4; ++j) SetPatchMatchOption(j, "filter", "false");
  for (const std::string dispatch : {"serial", "pair"}) {
    SCOPED_TRACE(dispatch);
    auto args = ListArgs({0, 1, 2, 3});
    args[2] = dispatch;
    std::atomic<size_t> calls{0};
    EventCapture capture;
    EXPECT_EQ(Invoke(args, [&](auto& job, auto& coordinator, size_t slot,
                               const auto& hooks) {
      EXPECT_FALSE(job.patch_match_stereo->filter);
      EXPECT_FALSE(job.patch_match_stereo->geom_consistency);
      ++calls;
      Complete(job, coordinator, slot, hooks);
    }), EXIT_SUCCESS);
    EXPECT_EQ(calls.load(), 4);
    size_t filter_options = 0, finished = 0;
    for (const auto& record : Records(capture.Read())) {
      if (record.at("event") == "effective_option" &&
          record.at("option") == "filter") {
        EXPECT_EQ(record.at("value_hex"), "30");  // Printed bool: 0.
        ++filter_options;
      }
      finished += record.at("event") == "batch_finished";
    }
    EXPECT_EQ(filter_options, 4);
    EXPECT_EQ(finished, 2);
  }
}

TEST_F(OwnedPairTest, UnfilteredStillRejectsGeometryAndConsistencyGraphs) {
  for (size_t j = 0; j < 4; ++j) SetPatchMatchOption(j, "filter", "false");
  SetPatchMatchOption(3, "geom_consistency", "true");
  Reject(ListArgs({0, 1, 2, 3}));
  SetPatchMatchOption(3, "geom_consistency", "false");
  SetPatchMatchOption(3, "write_consistency_graph", "true");
  Reject(ListArgs({0, 1, 2, 3}));
}

TEST_F(OwnedPairTest, GeometryRequiresSamePhaseJobsWithoutLookahead) {
  SetPatchMatchOption(0, "geom_consistency", "true");
  Reject(ListArgs({0, 1}));
  SetPatchMatchOption(1, "geom_consistency", "true");
  for (const std::string dispatch : {"serial", "pair"}) {
    auto args = ListArgs({0, 1});
    args[2] = dispatch;
    std::atomic<size_t> calls{0};
    EventCapture capture;
    EXPECT_EQ(Invoke(args, [&](auto& job, auto& coordinator, size_t slot,
                               const auto& hooks) {
      EXPECT_TRUE(job.patch_match_stereo->geom_consistency);
      EXPECT_TRUE(job.patch_match_stereo->filter);
      EXPECT_EQ(job.outputs[0].filename(), job.reference_name + ".geometric.bin");
      ++calls;
      Complete(job, coordinator, slot, hooks);
    }), EXIT_SUCCESS);
    EXPECT_EQ(calls.load(), 2);
  }
  auto args = ListArgs({0, 1});
  args.insert(args.end(), {"--host_lookahead", "1",
                          "--host_lookahead_charge_bytes", "100"});
  Reject(args);  // Even the no-next-epoch fallback must not permit geometry.
  for (size_t j : {2, 3}) SetPatchMatchOption(j, "geom_consistency", "true");
  {
    EventCapture capture;
    EXPECT_EQ(Invoke(ListArgs({0, 1, 2, 3}), Complete), EXIT_SUCCESS);
  }
  Reject(ListArgs({0, 1, 2}));
  SetPatchMatchOption(3, "geom_consistency", "false");
  Reject(ListArgs({0, 1, 2, 3}));
  SetPatchMatchOption(0, "geom_consistency", "false");
  Reject(ListArgs({0, 1}));  // Opposite mixed-phase direction.
}

TEST_F(OwnedPairTest, EarlyGeometryPermitsAdmissionAndMetadata) {
  auto args = ListArgs({0, 1});
  args.insert(args.end(), {"--geometry_early_permits", "invalid"});
  Reject(args);
  args.back() = "1";
  Reject(args);  // Photometric jobs cannot enable geometry permits.
  for (size_t j : {0, 1}) SetPatchMatchOption(j, "geom_consistency", "true");
  {
    EventCapture capture;
    EXPECT_EQ(Invoke(args, Complete), EXIT_SUCCESS);
    bool validated = false;
    for (const auto& record : Records(capture.Read())) {
      if (record.at("event") == "validated") {
        validated = true;
        EXPECT_EQ(record.at("geometry_early_permits"), "1");
      }
    }
    EXPECT_TRUE(validated);
  }
  args.insert(args.end(), {"--host_lookahead", "1",
                           "--host_lookahead_charge_bytes", "1"});
  Reject(args);
  auto photo = ListArgs({2, 3});
  photo.insert(photo.end(), {"--geometry_early_permits", "0"});
  EventCapture capture;
  EXPECT_EQ(Invoke(photo, Complete), EXIT_SUCCESS);
}

TEST_F(OwnedPairTest, GeometryBadFlagsAndExistingOutputRejectBeforeDispatch) {
  for (size_t j : {0, 1}) SetPatchMatchOption(j, "geom_consistency", "true");
  for (const auto& option : std::vector<std::pair<std::string, std::string>>{
           {"filter", "false"}, {"write_consistency_graph", "true"},
           {"allow_missing_files", "true"}, {"window_radius", "4"},
           {"window_step", "2"}}) {
    SetPatchMatchOption(1, option.first, option.second);
    Reject(ListArgs({0, 1}));
    WriteJob(1);
    SetPatchMatchOption(1, "geom_consistency", "true");
  }
  for (const auto* folder : {"depth_maps", "normal_maps"}) {
    const auto path = Workspace(1) / "stereo" / folder /
                      "reference 1.jpg.geometric.bin";
    Put(path, "partial output");
    Reject(ListArgs({0, 1}));
    fs::remove(path);
  }
  JobOptions job;
  job.patch_match_stereo->geom_consistency = true;
  std::atomic<bool> cancelled{false};
  EXPECT_THROW(PrepareFutureHostJob(job, 0, {1, 1}, cancelled), std::invalid_argument);
}

TEST_F(OwnedPairTest, GeometryEnvelopeUsesActualDimensionsAndPhotoIsUnchanged) {
  const auto workspace = Workspace(0);
  const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float translation[3] = {0, 0, 0};
  std::vector<mvs::Image> images;
  images.reserve(3);
  const std::array<std::pair<int, int>, 3> shapes{{{2, 2}, {10, 3}, {3, 11}}};
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto [width, height] = shapes[i];
    const auto name = "input" + Value(i) + ".png";
    const auto path = workspace / "images" / name;
    Bitmap bitmap(width, height, false);
    bitmap.Fill(BitmapColor<uint8_t>(100));
    ASSERT_TRUE(bitmap.Write(path));
    images.emplace_back(path, width, height, identity, identity, translation);
    mvs::DepthMap depth(width, height, 1, 8);
    mvs::NormalMap normal(width, height);
    for (const auto* folder : {"depth_maps", "normal_maps"}) {
      fs::create_directories(workspace / "stereo" / folder);
    }
    depth.Write(workspace / "stereo/depth_maps" / (name + ".photometric.bin"));
    normal.Write(workspace / "stereo/normal_maps" / (name + ".photometric.bin"));
  }
  mvs::PatchMatch::Problem problem;
  problem.images = &images;
  problem.ref_image_idx = 0;
  problem.src_image_idxs = {2, 1};
  // q=4+30+33=67, u=2*10*11=220, a=4; do not use reference
  // dimensions or the sum of source pixels in place of padded upload layers.
  const uint64_t q = 67, u = 220, a = 4;
  const uint64_t rotation = pair::BuildOwnedRotationLayout(2, 2, 2).logical_bytes;
  mvs::PatchMatchOptions options;
  options.geom_consistency = true;
  options.cache_size = (17 * q) / 1073741824.0;
  const uint64_t host = 17 * q + 34 * q + 4 * u;
  const uint64_t device = rotation + 5 * u + 2 * a + 688 * 2 + 32;
  EXPECT_NO_THROW(CheckInputs(options, problem, {host, device}, workspace));
  EXPECT_THROW(CheckInputs(options, problem, {host - 1, device}, workspace),
               std::invalid_argument);
  EXPECT_THROW(CheckInputs(options, problem, {host, device - 1}, workspace),
               std::invalid_argument);
  options.cache_size = (17 * q - 1) / 1073741824.0;
  EXPECT_THROW(CheckInputs(options, problem, {host, device}, workspace),
               std::invalid_argument);
  options.cache_size = q / 1073741824.0;
  EXPECT_THROW(CheckInputs(options, problem, {host, device}, workspace),
               std::invalid_argument);
  options.geom_consistency = false;
  const uint64_t photo_host = q + 2 * q + u;
  const uint64_t photo_device = rotation + u + 2 * a + 688 * 2 + 32;
  fs::remove_all(workspace / "stereo");  // Photo needs no seeds.
  EXPECT_NO_THROW(CheckInputs(options, problem, {photo_host, photo_device}, workspace));
  EXPECT_THROW(CheckInputs(options, problem, {photo_host - 1, photo_device}, workspace),
               std::invalid_argument);
  EXPECT_THROW(CheckInputs(options, problem, {photo_host, photo_device - 1}, workspace),
               std::invalid_argument);
  options.cache_size = (q - 1) / 1073741824.0;
  EXPECT_THROW(CheckInputs(options, problem, {photo_host, photo_device}, workspace),
               std::invalid_argument);
}

TEST_F(OwnedPairTest, SeedHeaderPreflightRejectsMissingMalformedAndMismatchedMaps) {
  const auto path = root_ / "seed.bin";
  EXPECT_THROW(CheckSeedMap(path, 2, 3, 1), std::invalid_argument);
  for (const std::string header : {"", "2&3&", "-2&3&1&", "2x3&1&",
                                  "2&3&3&", "3&2&1&", "2&3&1&"}) {
    Put(path, header);
    EXPECT_THROW(CheckSeedMap(path, 2, 3, 1), std::invalid_argument);
  }
  Put(path, "2&3&1&" + std::string(24, '\0'));
  EXPECT_NO_THROW(CheckSeedMap(path, 2, 3, 1));
  Put(path, "2&3&1&" + std::string(25, '\0'));
  EXPECT_THROW(CheckSeedMap(path, 2, 3, 1), std::invalid_argument);
  Put(path, "18446744073709551616&3&1&");
  EXPECT_THROW(CheckSeedMap(path, 2, 3, 1), std::overflow_error);
}

TEST_F(OwnedPairTest, InvalidFilterDoesNotSilentlyUseDefault) {
  SetPatchMatchOption(3, "filter", "not-a-bool");
  Reject(ListArgs({0, 1, 2, 3}));
}

TEST_F(OwnedPairTest, TwoEpochsJoinReleaseAndFinishBeforeNextAdmission) {
  for (const auto& mode : {
           std::make_tuple(false, false, "serial"),
           std::make_tuple(false, false, "pair"),
           std::make_tuple(true, false, "serial"),
           std::make_tuple(true, false, "pair"),
           std::make_tuple(true, true, "serial"),
           std::make_tuple(true, true, "pair")}) {
    const bool geometry = std::get<0>(mode);
    const bool early = std::get<1>(mode);
    const std::string dispatch = std::get<2>(mode);
    SCOPED_TRACE(dispatch);
    SCOPED_TRACE(geometry);
    SCOPED_TRACE(early);
    for (size_t j = 0; j < 4; ++j) {
      SetPatchMatchOption(j, "geom_consistency", geometry ? "true" : "false");
    }
    auto args = ListArgs({0, 1, 2, 3});
    args[2] = dispatch;
    args.insert(args.end(), {"--geometry_early_permits", early ? "1" : "0"});
    std::atomic<size_t> destroyed{0};
    std::mutex mutex;
    std::array<std::vector<std::string>, 2> seen;
    struct Resource {
      std::atomic<size_t>& destroyed;
      ~Resource() { ++destroyed; }
    };
    EventCapture capture;
    ASSERT_EQ(Invoke(args, [&](auto& job, auto& coordinator, size_t slot,
                               const auto& hooks) {
      Resource resource{destroyed};
      if (job.reference_name == "reference 2.jpg" ||
          job.reference_name == "reference 3.jpg") {
        EXPECT_GE(destroyed.load(), 2);
      }
      EXPECT_EQ(coordinator.ReservedBytes().host, 3145744);
      EXPECT_EQ(coordinator.ReservedBytes().device, 400);
      {
        std::lock_guard<std::mutex> lock(mutex);
        seen[slot].push_back(job.reference_name);
      }
      Complete(job, coordinator, slot, hooks);
    }), EXIT_SUCCESS);
    EXPECT_EQ(destroyed.load(), 4);
    EXPECT_EQ(seen[0], (std::vector<std::string>{"reference 0.jpg", "reference 2.jpg"}));
    EXPECT_EQ(seen[1], (std::vector<std::string>{"reference 1.jpg", "reference 3.jpg"}));
    const auto records = Records(capture.Read());
    size_t admitted = 0, joined = 0, released = 0, finished = 0;
    std::vector<std::string> batches;
    for (const auto& record : records) {
      EXPECT_EQ(record.at("pid"), Value(getpid()));
      const auto& event = record.at("event");
      if (event == "admitted") {
        EXPECT_EQ(record.at("coord_seq"), admitted % 2 == 0 ? "1" : "2");
        if (admitted >= 2) {
          EXPECT_EQ(joined, 1);
          EXPECT_EQ(released, 1);
          EXPECT_EQ(finished, 1);
        }
        ++admitted;
      } else if (event == "owners_joined") {
        ++joined;
      } else if (event == "ledger_release") {
        EXPECT_EQ(joined, released + 1);
        EXPECT_EQ(record.at("reserved_host"), "0");
        EXPECT_EQ(record.at("reserved_device"), "0");
        ++released;
      } else if (event == "batch_finished") {
        EXPECT_EQ(released, finished + 1);
        ++finished;
        batches.push_back(record.at("batch"));
      }
    }
    EXPECT_EQ(admitted, 4);
    EXPECT_EQ(joined, 2);
    EXPECT_EQ(released, 2);
    ASSERT_EQ(batches.size(), 2);
    EXPECT_NE(batches[0], batches[1]);
  }
}

TEST_F(OwnedPairTest, CallbackFailuresAbortLaterEpochsAndDestroyOwnerResources) {
  for (size_t j = 0; j < 4; ++j) SetPatchMatchOption(j, "geom_consistency", "true");
  // Thread-factory injection is intentionally not added to RunPairs. These
  // faults cover controller construction, owner work, output and cleanup via
  // its supported callback seam, not OS thread-construction failure.
  for (const std::string phase : {"construction", "owner", "output", "cleanup"}) {
    SCOPED_TRACE(phase);
    auto args = ListArgs({0, 1, 2, 3});
    args.insert(args.end(), {"--geometry_early_permits", "1"});
    std::atomic<size_t> calls{0}, destroyed{0};
    struct Resource {
      std::atomic<size_t>& destroyed;
      ~Resource() { ++destroyed; }
    };
    EventCapture capture;
    EXPECT_THROW(Invoke(args, [&](auto& job, auto& coordinator,
                                 size_t slot, const auto& hooks) {
      ++calls;
      Resource resource{destroyed};
      EXPECT_NE(job.reference_name, "reference 2.jpg");
      EXPECT_NE(job.reference_name, "reference 3.jpg");
      if (slot == 0 && phase == "construction") {
        throw std::runtime_error("injected construction failure");
      }
      if (slot == 0 && phase == "owner") {
        coordinator.BeginPreparation(slot);
        throw std::runtime_error("injected owner failure");
      }
      if (slot == 0 && phase == "output") {
        auto failing_hooks = hooks;
        failing_hooks.output_phase = [](size_t, const char*) {
          throw std::runtime_error("injected output failure");
        };
        Complete(job, coordinator, slot, failing_hooks);
      } else {
        Complete(job, coordinator, slot, hooks);
      }
      if (slot == 0 && phase == "cleanup") {
        throw std::runtime_error("injected cleanup failure");
      }
    }), std::runtime_error);
    EXPECT_LE(calls.load(), 2);
    EXPECT_EQ(destroyed.load(), calls.load());
    size_t admissions = 0, joins = 0, finishes = 0;
    for (const auto& record : Records(capture.Read())) {
      admissions += record.at("event") == "admitted";
      joins += record.at("event") == "owners_joined";
      finishes += record.at("event") == "batch_finished";
    }
    EXPECT_EQ(admissions, 2);
    EXPECT_EQ(joins, 1);
    EXPECT_EQ(finishes, 0);
  }
}

TEST_F(OwnedPairTest, HostLookaheadParserRejectsInvalidOrMissingReservation) {
  auto args = ListArgs({0, 1, 2, 3});
  args.insert(args.end(), {"--host_lookahead", "2"});
  Reject(args);
  args.back() = "1";
  Reject(args);
  args.insert(args.end(), {"--host_lookahead_charge_bytes", "0"});
  Reject(args);
}

TEST_F(OwnedPairTest, HostLookaheadInsufficientBudgetFallsBackSynchronously) {
  auto args = ListArgs({0, 1, 2, 3});
  args.insert(args.end(), {"--host_lookahead", "1",
                           "--host_lookahead_charge_bytes", "4000000"});
  EventCapture capture;
  EXPECT_EQ(Invoke(args, Complete), EXIT_SUCCESS);
  const auto log = capture.Read();
  EXPECT_NE(log.find("event=fallback enabled=0 reserved_host=0 reason=host_budget"),
            std::string::npos);
  std::istringstream lines(log);
  for (std::string line; std::getline(lines, line);) {
    if (line.find("OWNED_HOST_LOOKAHEAD ") != std::string::npos) {
      EXPECT_EQ(line.find("event=prep_start"), std::string::npos);
    }
  }
}

TEST_F(OwnedPairTest, HostLookaheadRealDriverConsumesOnlyAfterNextAdmission) {
  if (AffinityCount() < 2) GTEST_SKIP() << "lookahead requires two CPU slots";
  auto args = ListArgs({0, 1, 2, 3});
  args.insert(args.end(), {"--host_lookahead", "1",
                           "--host_lookahead_charge_bytes", "100"});
  std::atomic<size_t> preparations{0}, consumed{0};
  EventCapture capture;
  EXPECT_EQ(InvokePrepared(args,
      [&](auto& job, auto& coordinator, size_t slot, const auto& hooks,
          std::unique_ptr<PreparedHostJob> prepared) {
        if (job.reference_name == "reference 2.jpg" ||
            job.reference_name == "reference 3.jpg") {
          EXPECT_NE(prepared, nullptr);
          ++consumed;
        } else EXPECT_EQ(prepared, nullptr);
        Complete(job, coordinator, slot, hooks);
      },
      [&](auto&, size_t, pair::ByteCharge, const std::atomic<bool>& cancelled) {
        EXPECT_FALSE(cancelled.load());
        ++preparations;
        return std::make_unique<PreparedHostJob>();
      }), EXIT_SUCCESS);
  EXPECT_EQ(preparations.load(), 2);
  EXPECT_EQ(consumed.load(), 2);
  const auto log = capture.Read();
  const auto consume = log.find("event=consume");
  ASSERT_NE(consume, std::string::npos);
  EXPECT_LT(log.find("event=batch_finished"), consume);
  EXPECT_EQ(log.substr(0, consume).find("event=admitted"),
            log.find("event=admitted"));
  size_t admissions = 0;
  for (const auto& record : Records(log.substr(0, consume))) {
    admissions += record.at("event") == "admitted";
  }
  EXPECT_EQ(admissions, 4);
  EXPECT_NE(log.find("event=release enabled=1 reserved_host=0"), std::string::npos);
}

TEST_F(OwnedPairTest, HostPreparationFailureStopsNextEpochBeforeAdmission) {
  if (AffinityCount() < 2) GTEST_SKIP() << "lookahead requires two CPU slots";
  auto args = ListArgs({0, 1, 2, 3});
  args.insert(args.end(), {"--host_lookahead", "1",
                           "--host_lookahead_charge_bytes", "100"});
  EventCapture capture;
  EXPECT_THROW(InvokePrepared(args, Complete,
      [](auto&, size_t slot, pair::ByteCharge, const std::atomic<bool>&) {
        if (slot == 1) throw std::runtime_error("injected CPU preparation error");
        return std::make_unique<PreparedHostJob>();
      }), std::runtime_error);
  const auto log = capture.Read();
  size_t admissions = 0;
  for (const auto& record : Records(log)) admissions += record.at("event") == "admitted";
  EXPECT_EQ(admissions, 2);
  EXPECT_NE(log.find("event=failed"), std::string::npos);
  EXPECT_NE(log.find("event=release enabled=1 reserved_host=0"), std::string::npos);
}

TEST(OwnedPairEventsTest, SameClockTickTokensAreStrictlyIncreasing) {
  pair::PairEventsTestAccess restore(0);
  EXPECT_EQ(pair::PairEventsTestAccess::Next(0), 1);
  for (uint64_t i = 0; i < 128; ++i) {
    EXPECT_EQ(pair::PairEventsTestAccess::Next(7), 7 + i);
  }
}

TEST(OwnedPairEventsTest, ConcurrentSameClockTickTokensAreUnique) {
  pair::PairEventsTestAccess restore(0);
  std::array<std::array<uint64_t, 128>, 4> tokens;
  std::array<std::thread, 4> threads;
  for (size_t t = 0; t < threads.size(); ++t) {
    threads[t] = std::thread([&, t] {
      for (auto& token : tokens[t]) {
        token = pair::PairEventsTestAccess::Next(7);
      }
    });
  }
  for (auto& thread : threads) thread.join();
  std::set<uint64_t> unique;
  for (const auto& row : tokens) unique.insert(row.begin(), row.end());
  ASSERT_EQ(unique.size(), 512);
  EXPECT_EQ(*unique.begin(), 7);
  EXPECT_EQ(*unique.rbegin(), 518);
}

TEST(OwnedPairEventsTest, TokenOverflowFailsClosedWithoutWrapping) {
  const auto maximum = std::numeric_limits<uint64_t>::max();
  pair::PairEventsTestAccess restore(maximum - 1);
  EXPECT_EQ(pair::PairEventsTestAccess::Next(0), maximum);
  EXPECT_THROW(pair::PairEventsTestAccess::Next(maximum), std::overflow_error);
  EXPECT_EQ(pair::PairEventsTestAccess::Current(), maximum);
  EXPECT_THROW(pair::PairEventsTestAccess::Next(0), std::overflow_error);
  EXPECT_EQ(pair::PairEventsTestAccess::Current(), maximum);
}

TEST(OwnedPairEventsTest, ConsecutiveAndConcurrentBatchIdentitiesAreUnique) {
  EventCapture capture;
  auto emit = [] {
    for (size_t i = 0; i < 128; ++i) {
      pair::PairEvents events("pair");
      events.Emit(pair::PairEvents::kBatch, "identity_probe");
    }
  };
  emit();
  std::array<std::thread, 4> threads;
  for (auto& thread : threads) thread = std::thread(emit);
  for (auto& thread : threads) thread.join();
  const auto records = Records(capture.Read());
  ASSERT_EQ(records.size(), 640);
  std::set<std::string> batches;
  for (const auto& record : records) {
    EXPECT_EQ(record.at("pid"), Value(getpid()));
    EXPECT_TRUE(batches.insert(record.at("batch")).second);
  }
}

}  // namespace
}  // namespace colmap
