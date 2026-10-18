// SPDX-License-Identifier: BSD-3-Clause
// Story17 photo_directory_summary CLI and serializer tests. These exercise
// the REAL command surface (RunPhotoDirectorySummary through raw argv for the
// actual OptionManager/collector path) and the actual typed JSON serializer
// for the frozen V1 contract. They intentionally do not repeat the collector
// coverage in controllers/reconstruction_input_summary_test.cc.
#include "colmap/exe/photo_directory_summary.h"

#include "colmap/controllers/reconstruction_input_summary.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"
#include "colmap/util/testing.h"
#include "colmap/util/version.h"

#include <algorithm>
#include <filesystem>
#include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void WriteTestImage(const std::filesystem::path& path,
                    const int width,
                    const int height) {
  Bitmap bitmap(width, height, /*as_rgb=*/true);
  bitmap.Fill(BitmapColor<uint8_t>(128));
  ASSERT_TRUE(bitmap.Write(path));
}

void WriteTextFile(const std::filesystem::path& path, const std::string& text) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file << text;
}

// Reads the byte content of every regular file under the directory, used to
// verify that the command does not mutate any input.
std::map<std::string, std::string> ReadAllFileContents(
    const std::filesystem::path& directory) {
  std::map<std::string, std::string> contents;
  for (const auto& file_path : GetRecursiveFileList(directory)) {
    std::ifstream file(file_path, std::ios::binary);
    EXPECT_TRUE(file.is_open());
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    contents[GetNormalizedRelativePath(file_path, directory)] =
        std::move(content);
  }
  return contents;
}

// Collects every directory entry (files and directories) under `root` as
// normalized relative paths, sorted; used to prove that the command does not
// create log files, log directories or any other side-effect entries.
std::vector<std::string> SnapshotTreeEntries(
    const std::filesystem::path& root) {
  std::vector<std::string> entries;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(
           root, std::filesystem::directory_options::skip_permission_denied)) {
    const std::string relative =
        std::filesystem::relative(entry.path(), root).generic_string();
    entries.push_back(relative);
  }
  std::sort(entries.begin(), entries.end());
  return entries;
}

std::vector<char*> ArgvFromStrings(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (const std::string& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);
  return argv;
}

struct CliResult {
  int return_code = -1;
  std::string stdout_text;
};

CliResult RunPhotoDirectorySummaryCli(
    const std::vector<std::string>& command_line) {
  std::vector<std::string> args = {"colmap"};
  args.insert(args.end(), command_line.begin(), command_line.end());
  // Must stay mutable: RunPhotoDirectorySummary takes `char** argv`, so a
  // const vector's data() (char* const*) is not convertible to char**.
  std::vector<char*> argv = ArgvFromStrings(args);

  CliResult result;
  std::ostringstream captured;
  std::streambuf* const saved = std::cout.rdbuf(captured.rdbuf());
  result.return_code =
      RunPhotoDirectorySummary(static_cast<int>(args.size()), argv.data());
  std::cout.rdbuf(saved);
  result.stdout_text = captured.str();
  return result;
}

PhotoDirectorySummaryJsonBuild TestBuild() {
  PhotoDirectorySummaryJsonBuild build;
  build.revision = "f64591f";
  build.version = "COLMAP test";
  build.cuda_compiled = false;
  return build;
}

TEST(PhotoDirectorySummaryCli, MissingDirectoryFailsWithoutJson) {
  const CliResult result = RunPhotoDirectorySummaryCli(
      {"--image_path", (CreateTestDir() / "does-not-exist").string()});
  EXPECT_EQ(result.return_code, EXIT_FAILURE);
  // No partially serialized document may appear on stdout.
  EXPECT_EQ(result.stdout_text, "");
}

TEST(PhotoDirectorySummaryCli, MixedPhotosProblemsAndSidecar) {
  const auto image_path = CreateTestDir() / "photos";
  CreateDirIfNotExists(image_path, /*recursive=*/true);
  WriteTestImage(image_path / "photo.png", /*width=*/4, /*height=*/3);
  CreateDirIfNotExists(image_path / "sub", /*recursive=*/true);
  WriteTestImage(image_path / "sub" / "beta.png", /*width=*/2, /*height=*/5);
  WriteTextFile(image_path / "corrupt.JPG", "not an image\n");
  WriteTextFile(image_path / "notes.txt", "sidecar\n");
  const std::map<std::string, std::string> contents_before =
      ReadAllFileContents(image_path);

  const CliResult result =
      RunPhotoDirectorySummaryCli({"--image_path", image_path.string()});

  EXPECT_EQ(result.return_code, EXIT_SUCCESS);
  // Exactly one complete JSON document, one line, on stdout.
  ASSERT_FALSE(result.stdout_text.empty());
  EXPECT_EQ(result.stdout_text.back(), '\n');
  EXPECT_EQ(
      std::count(result.stdout_text.begin(), result.stdout_text.end(), '\n'),
      1);
  const std::string& json = result.stdout_text;
  EXPECT_NE(json.find("\"schema_version\":1"), std::string::npos);
  EXPECT_NE(json.find("\"image_path\":\"" + image_path.string() + "\""),
            std::string::npos);
  EXPECT_NE(json.find("\"photo_count\":2,"), std::string::npos);
  EXPECT_NE(json.find("{\"path\":\"photo.png\",\"width\":4,\"height\":3,"
                      "\"orientation\":null}"),
            std::string::npos);
  EXPECT_NE(json.find("{\"path\":\"sub/beta.png\",\"width\":2,\"height\":5,"
                      "\"orientation\":null}"),
            std::string::npos);
  EXPECT_NE(json.find("\"largest_photo\":{\"path\":\"photo.png\",\"width\":4,"
                      "\"height\":3}"),
            std::string::npos);
  EXPECT_NE(
      json.find("\"dimension_envelope\":{\"max_width\":4,\"max_height\":5}"),
      std::string::npos);
  EXPECT_NE(json.find("\"kind\":\"corrupt_image\""), std::string::npos);
  EXPECT_NE(json.find("\"path\":\"corrupt.JPG\""), std::string::npos);
  EXPECT_NE(json.find("\"non_image_files\":[\"notes.txt\"]"),
            std::string::npos);
  // Build identity: the compiled accessor value is emitted verbatim, and a
  // metadata-free ("unversioned") build serializes its empty revision as
  // JSON null.
  const std::string compiled_revision = GetCommitRevision();
  if (compiled_revision.empty()) {
    EXPECT_NE(json.find(",\"build\":{\"revision\":null,"),
              std::string::npos);
  } else {
    EXPECT_NE(
        json.find(",\"build\":{\"revision\":\"" + compiled_revision + "\","),
        std::string::npos);
  }
  // The `build` object is closed before the frozen trailing claim fields.
  EXPECT_NE(json.find("\"stock_memory_fingerprint\":null},\"fit_claim\":"
                      "false,\"header_only\":true,\"orientation_applied\":"
                      "false}"),
            std::string::npos);

#if defined(COLMAP_CUDA_ENABLED) && defined(COLMAP_MVS_ENABLED) && \
    !defined(COLMAP_HIP_ENABLED)
  const auto layout = mvs::GetStockPrngStateLayout();
  EXPECT_GT(layout.bytes, 0);
  EXPECT_GT(layout.alignment, 0);
  EXPECT_NE(json.find("\"stock_prng_state\":{\"bytes\":" +
                      std::to_string(layout.bytes) + ",\"alignment\":" +
                      std::to_string(layout.alignment) +
                      ",\"type\":\"curandState\",\"provenance\":"
                      "\"cuda_translation_unit_compile_time\"}"),
            std::string::npos);
#else
  EXPECT_NE(json.find("\"stock_prng_state\":null,"), std::string::npos);
#endif

  // The command must not mutate any input file.
  EXPECT_EQ(ReadAllFileContents(image_path), contents_before);
}

TEST(PhotoDirectorySummaryCli, EmptyDirectoryNullAndEmptyShapes) {
  const auto image_path = CreateTestDir() / "empty";
  CreateDirIfNotExists(image_path, /*recursive=*/true);

  const CliResult result =
      RunPhotoDirectorySummaryCli({"--image_path", image_path.string()});

  EXPECT_EQ(result.return_code, EXIT_SUCCESS);
  const std::string& json = result.stdout_text;
  EXPECT_NE(json.find("\"photo_count\":0,\"photos\":[],\"largest_photo\":null,"
                      "\"dimension_envelope\":null,\"problems\":[],"
                      "\"non_image_files\":[]"),
            std::string::npos);
}

TEST(PhotoDirectorySummaryJsonSerializer, EmptySummaryExactTypedDocument) {
  const std::string json = WritePhotoDirectorySummaryJson(
      PhotoDirectorySummary(), "/photos", TestBuild());
  EXPECT_EQ(
      json,
      "{\"schema_version\":1,\"image_path\":\"/photos\",\"photo_count\":0,"
      "\"photos\":[],\"largest_photo\":null,\"dimension_envelope\":null,"
      "\"problems\":[],\"non_image_files\":[],"
      "\"build\":{\"revision\":\"f64591f\",\"version\":\"COLMAP test\","
      "\"cuda_compiled\":false,\"stock_prng_state\":null,"
      "\"stock_memory_fingerprint\":null},"
      "\"fit_claim\":false,\"header_only\":true,\"orientation_applied\":"
      "false}");
}

TEST(PhotoDirectorySummaryJsonSerializer, SuppliedCudaStateHasTypedLayout) {
  PhotoDirectorySummaryJsonBuild build = TestBuild();
  build.cuda_compiled = true;
  // Deliberately not a toolkit-specific layout: serialize supplied values,
  // never a guessed fallback or a hard-coded size.
  build.stock_prng_state = mvs::StockPrngStateLayout{123, 16};
  const std::string json =
      WritePhotoDirectorySummaryJson(PhotoDirectorySummary(), "/photos", build);
  EXPECT_NE(json.find("\"stock_prng_state\":{\"bytes\":123,\"alignment\":16,"
                      "\"type\":\"curandState\",\"provenance\":"
                      "\"cuda_translation_unit_compile_time\"},"
                      "\"stock_memory_fingerprint\":null},\"fit_claim\":false"),
            std::string::npos);
}

TEST(PhotoDirectorySummaryJsonSerializer, MissingCudaStateHasNoFallback) {
  PhotoDirectorySummaryJsonBuild build = TestBuild();
  build.cuda_compiled = true;
  const std::string json =
      WritePhotoDirectorySummaryJson(PhotoDirectorySummary(), "/photos", build);
  EXPECT_NE(json.find("\"cuda_compiled\":true,\"stock_prng_state\":null,"),
            std::string::npos);
}

TEST(PhotoDirectorySummaryJsonSerializer, InvalidStateDoesNotInventLayout) {
  PhotoDirectorySummaryJsonBuild build = TestBuild();
  build.cuda_compiled = true;
  for (const auto layout : {mvs::StockPrngStateLayout{0, 8},
                            mvs::StockPrngStateLayout{64, 0}}) {
    build.stock_prng_state = layout;
    const std::string json = WritePhotoDirectorySummaryJson(
        PhotoDirectorySummary(), "/photos", build);
    EXPECT_NE(json.find("\"stock_prng_state\":null,"), std::string::npos);
  }
}

TEST(PhotoDirectorySummaryJsonSerializer, CpuBuildCannotClaimCudaState) {
  PhotoDirectorySummaryJsonBuild build = TestBuild();
  build.stock_prng_state = mvs::StockPrngStateLayout{123, 16};
  const std::string json =
      WritePhotoDirectorySummaryJson(PhotoDirectorySummary(), "/photos", build);
  EXPECT_NE(json.find("\"cuda_compiled\":false,\"stock_prng_state\":null,"),
            std::string::npos);
}

TEST(PhotoDirectorySummaryJsonSerializer, EmptyRevisionSerializesAsNull) {
  PhotoDirectorySummaryJsonBuild build = TestBuild();
  build.revision = "";
  const std::string json =
      WritePhotoDirectorySummaryJson(PhotoDirectorySummary(), "/photos", build);
  EXPECT_NE(json.find("\"revision\":null,"), std::string::npos);
  EXPECT_EQ(json.find("\"revision\":\"\""), std::string::npos);
}

TEST(PhotoDirectorySummaryJsonSerializer,
     QuoteBackslashAndControlPathEscaping) {
  PhotoDirectorySummary summary;
  PhotoScanEntry photo;
  photo.relative_path =
      std::string("a.png") + '\t' + '\\' + '\n' + '\x01' + "b\"c";
  photo.encoded_width = 4;
  photo.encoded_height = 3;
  photo.exif_orientation = 6;
  summary.photos.push_back(photo);
  summary.largest_photo = photo;
  summary.dimension_envelope.max_encoded_width = 4;
  summary.dimension_envelope.max_encoded_height = 3;

  const std::string json =
      WritePhotoDirectorySummaryJson(summary, "/photos", TestBuild());
  EXPECT_EQ(
      json,
      "{\"schema_version\":1,\"image_path\":\"/photos\",\"photo_count\":1,"
      "\"photos\":[{\"path\":\"a.png\\t\\\\\\n\\u0001b\\\"c\",\"width\":4,"
      "\"height\":3,\"orientation\":6}],"
      "\"largest_photo\":{\"path\":\"a.png\\t\\\\\\n\\u0001b\\\"c\","
      "\"width\":4,\"height\":3},"
      "\"dimension_envelope\":{\"max_width\":4,\"max_height\":3},"
      "\"problems\":[],\"non_image_files\":[],"
      "\"build\":{\"revision\":\"f64591f\",\"version\":\"COLMAP test\","
      "\"cuda_compiled\":false,\"stock_prng_state\":null,"
      "\"stock_memory_fingerprint\":null},"
      "\"fit_claim\":false,\"header_only\":true,\"orientation_applied\":"
      "false}");
}

TEST(PhotoDirectorySummaryJsonSerializer, MultiplePhotosAndTiedEnvelope) {
  PhotoDirectorySummary summary;
  PhotoScanEntry first;
  first.relative_path = "one.png";
  first.encoded_width = 7;
  first.encoded_height = 2;
  PhotoScanEntry second;
  second.relative_path = "sub/two.png";
  second.encoded_width = 2;
  second.encoded_height = 7;
  summary.photos = {first, second};
  summary.dimension_envelope.max_encoded_width = 7;
  summary.dimension_envelope.max_encoded_height = 7;
  PhotoDirectorySummaryJsonBuild build = TestBuild();
  build.version = "version \"with\" quotes\\";
  build.cuda_compiled = true;

  const std::string json =
      WritePhotoDirectorySummaryJson(summary, "/photos", build);
  EXPECT_EQ(
      json,
      "{\"schema_version\":1,\"image_path\":\"/photos\",\"photo_count\":2,"
      "\"photos\":[{\"path\":\"one.png\",\"width\":7,\"height\":2,"
      "\"orientation\":null},{\"path\":\"sub/two.png\",\"width\":2,"
      "\"height\":7,\"orientation\":null}],"
      "\"largest_photo\":null,"
      "\"dimension_envelope\":{\"max_width\":7,\"max_height\":7},"
      "\"problems\":[],\"non_image_files\":[],"
      "\"build\":{\"revision\":\"f64591f\",\"version\":\"version \\\"with\\\" "
      "quotes\\\\\",\"cuda_compiled\":true,"
      "\"stock_prng_state\":null,\"stock_memory_fingerprint\":null},"
      "\"fit_claim\":false,\"header_only\":true,\"orientation_applied\":"
      "false}");
}

TEST(PhotoDirectorySummaryJsonSerializer, BothProblemKinds) {
  PhotoDirectorySummary summary;
  summary.problems.push_back(
      PhotoScanProblem{PhotoScanProblem::Kind::kCorruptImage, "a.JPG"});
  summary.problems.push_back(
      PhotoScanProblem{PhotoScanProblem::Kind::kUnknownDimensions, "b.png"});
  const std::string json =
      WritePhotoDirectorySummaryJson(summary, "/photos", TestBuild());
  EXPECT_EQ(
      json,
      "{\"schema_version\":1,\"image_path\":\"/photos\",\"photo_count\":0,"
      "\"photos\":[],\"largest_photo\":null,\"dimension_envelope\":null,"
      "\"problems\":["
      "{\"path\":\"a.JPG\",\"kind\":\"corrupt_image\",\"message\":\"Image "
      "header could not be parsed; the file appears corrupt or its format is "
      "unavailable.\"},"
      "{\"path\":\"b.png\",\"kind\":\"unknown_dimensions\",\"message\":"
      "\"Image header reports missing or invalid encoded dimensions.\"}],"
      "\"non_image_files\":[],"
      "\"build\":{\"revision\":\"f64591f\",\"version\":\"COLMAP test\","
      "\"cuda_compiled\":false,\"stock_prng_state\":null,"
      "\"stock_memory_fingerprint\":null},"
      "\"fit_claim\":false,\"header_only\":true,\"orientation_applied\":"
      "false}");
}

// The compiled identity flows into the CLI build section without any runtime
// Git lookup: the emitted revision equals the compiled accessor value.
TEST(PhotoDirectorySummaryCli, BuildSectionMatchesCompiledIdentity) {
  const auto image_path = CreateTestDir() / "empty";
  CreateDirIfNotExists(image_path, /*recursive=*/true);

  const CliResult result =
      RunPhotoDirectorySummaryCli({"--image_path", image_path.string()});

  EXPECT_EQ(result.return_code, EXIT_SUCCESS);
  const std::string& json = result.stdout_text;
  const std::string compiled_revision = GetCommitRevision();
  if (compiled_revision.empty()) {
    EXPECT_NE(json.find("\"revision\":null"), std::string::npos);
  } else {
    EXPECT_NE(json.find("\"revision\":\"" + compiled_revision + "\""),
              std::string::npos);
  }
}

// Regression for the inspection contract's stderr-only, non-writing
// diagnostics: logging override options must be rejected before any side
// effect, and neither the failure nor the success path may create log files
// or log directories; the only stdout content is the single JSON line on
// success.
TEST(PhotoDirectorySummaryCli,
     RejectsLoggingOverridesBeforeSideEffects) {
  const auto sandbox = CreateTestDir();
  const auto log_dir = sandbox / "glog-should-stay-empty";
  CreateDirIfNotExists(log_dir, /*recursive=*/true);
  const auto log_path_dir = sandbox / "created-by-log-path";
  const std::string saved_log_dir = FLAGS_log_dir;
  FLAGS_log_dir = log_dir.string();

  const CliResult result = RunPhotoDirectorySummaryCli(
      {"--log_target", "stdout",
       "--log_path", log_path_dir.string(),
       "--image_path", (sandbox / "unused").string()});

  // Rejected nonzero, and no stdout output: a rejected logging override
  // fails before any JSON could have been produced.
  EXPECT_EQ(result.return_code, EXIT_FAILURE);
  EXPECT_EQ(result.stdout_text, "");
  // The forbidden --log_path value never materialized as a directory and no
  // log file was written to the process log directory.
  EXPECT_FALSE(ExistsDir(log_path_dir));
  EXPECT_TRUE(GetRecursiveFileList(log_dir).empty());
  // Whole-run enforcement: the command leaves the process in a stderr-only,
  // non-writing glog configuration, so any log emitted by code reached from
  // this command cannot write files or contaminate stdout.
  EXPECT_TRUE(FLAGS_logtostderr);
  EXPECT_FALSE(FLAGS_alsologtostderr);
#if COLMAP_GLOG_HAS_STDOUT_SUPPORT
  EXPECT_FALSE(FLAGS_logtostdout);
#endif
  FLAGS_log_dir = saved_log_dir;
}

TEST(PhotoDirectorySummaryCli, RejectsUnknownOptionsWithoutSideEffects) {
  const auto sandbox = CreateTestDir();
  const auto log_dir = sandbox / "glog-should-stay-empty";
  CreateDirIfNotExists(log_dir, /*recursive=*/true);
  const std::string saved_log_dir = FLAGS_log_dir;
  FLAGS_log_dir = log_dir.string();

  const CliResult result = RunPhotoDirectorySummaryCli(
      {"--log_level", "3",
       "--image_path", (sandbox / "unused").string()});

  EXPECT_EQ(result.return_code, EXIT_FAILURE);
  EXPECT_EQ(result.stdout_text, "");
  EXPECT_TRUE(GetRecursiveFileList(log_dir).empty());
  EXPECT_TRUE(FLAGS_logtostderr);
  EXPECT_FALSE(FLAGS_alsologtostderr);
  FLAGS_log_dir = saved_log_dir;
}

// Success path: exactly one JSON line on stdout, and no log files or new
// directory entries anywhere in the sandbox.
TEST(PhotoDirectorySummaryCli,
     SuccessWritesOnlySingleJsonWithNoNewEntries) {
  const auto sandbox = CreateTestDir();
  const auto image_path = sandbox / "photos";
  CreateDirIfNotExists(image_path, /*recursive=*/true);
  WriteTestImage(image_path / "only.png", /*width=*/3, /*height=*/2);
  const auto log_dir = sandbox / "glog-should-stay-empty";
  CreateDirIfNotExists(log_dir, /*recursive=*/true);
  const std::vector<std::string> entries_before = SnapshotTreeEntries(sandbox);
  const std::string saved_log_dir = FLAGS_log_dir;
  FLAGS_log_dir = log_dir.string();

  const CliResult result =
      RunPhotoDirectorySummaryCli({"--image_path", image_path.string()});

  EXPECT_EQ(result.return_code, EXIT_SUCCESS);
  // Exactly one complete JSON line on stdout.
  ASSERT_FALSE(result.stdout_text.empty());
  EXPECT_EQ(result.stdout_text.back(), '\n');
  EXPECT_EQ(
      std::count(result.stdout_text.begin(), result.stdout_text.end(), '\n'),
      1);
  EXPECT_NE(result.stdout_text.find("only.png"), std::string::npos);
  // No log file written and no new files, directories or log artifacts.
  EXPECT_TRUE(GetRecursiveFileList(log_dir).empty());
  const std::vector<std::string> entries_after = SnapshotTreeEntries(sandbox);
  EXPECT_EQ(entries_after, entries_before);
  EXPECT_TRUE(FLAGS_logtostderr);
  EXPECT_FALSE(FLAGS_alsologtostderr);
  FLAGS_log_dir = saved_log_dir;
}

}  // namespace
}  // namespace colmap