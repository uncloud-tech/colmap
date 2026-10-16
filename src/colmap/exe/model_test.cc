// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/model.h"

#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <boost/property_tree/json_parser.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Restore stdout even when the native model reader throws.
class CaptureStdout {
 public:
  CaptureStdout() : saved_(std::cout.rdbuf(stream_.rdbuf())) {}
  ~CaptureStdout() { std::cout.rdbuf(saved_); }
  std::string Text() const { return stream_.str(); }

 private:
  std::ostringstream stream_;
  std::streambuf* saved_;
};

int Analyze(std::vector<std::string> args) {
  args.insert(args.begin(), "model_analyzer");
  std::vector<char*> argv;
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);
  return RunModelAnalyzer(static_cast<int>(args.size()), argv.data());
}

std::map<std::filesystem::path, std::string> Snapshot(
    const std::filesystem::path& root);

TEST(ModelAnalyzer, TextAndBinaryHaveSameTypedJsonCounts) {
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 2;
  options.num_frames_per_rig = 3;
  options.num_points3D = 5;
  // All unit-sphere points are visible from the radius-5 camera positions.
  options.camera_params = {100, 512, 384, 0};
  options.sensor_from_rig_translation_stddev = 0;
  options.sensor_from_rig_rotation_stddev = 0;
  SynthesizeDataset(options, &reconstruction);

  const auto root = CreateTestDir();
  const auto text_path = root / "text";
  const auto binary_path = root / "binary";
  CreateDirIfNotExists(text_path);
  CreateDirIfNotExists(binary_path);
  reconstruction.WriteText(text_path);
  reconstruction.WriteBinary(binary_path);
  const auto before = Snapshot(root);

  // Literal bytes also enforce JSON integer types (PropertyTree by itself
  // cannot distinguish a JSON number from a quoted string).
  const std::string expected =
      "{\"schema_version\":1,\"num_rigs\":1,\"num_cameras\":2,"
      "\"num_frames\":3,\"num_registered_frames\":3,\"num_images\":6,"
      "\"num_registered_images\":6,\"num_points3D\":5,"
      "\"num_observations\":30}\n";
  for (const auto& path : {text_path, binary_path}) {
    CaptureStdout capture;
    ASSERT_EQ(Analyze({"--path", path.string(), "--output_format", "JSON"}),
              EXIT_SUCCESS);
    EXPECT_EQ(capture.Text(), expected);
    std::istringstream stream(capture.Text());
    boost::property_tree::ptree json;
    ASSERT_NO_THROW(boost::property_tree::read_json(stream, json));
    EXPECT_EQ(json.size(), 9);
    EXPECT_EQ(json.get<int>("schema_version"), 1);
    EXPECT_EQ(json.get<int>("num_observations"), 30);
    for (const auto& field : json) {
      EXPECT_TRUE(field.second.empty());
    }
  }
  EXPECT_EQ(Snapshot(root), before);
}

TEST(ModelAnalyzer, EmptyModelIsValidJson) {
  const auto root = CreateTestDir();
  for (const bool binary : {false, true}) {
    const auto path = root / (binary ? "binary" : "text");
    CreateDirIfNotExists(path);
    const Reconstruction reconstruction;
    if (binary) {
      reconstruction.WriteBinary(path);
    } else {
      reconstruction.WriteText(path);
    }
    CaptureStdout capture;
    ASSERT_EQ(Analyze({"--path", path.string(), "--output_format", "JSON"}),
              EXIT_SUCCESS);
    EXPECT_EQ(capture.Text(),
              "{\"schema_version\":1,\"num_rigs\":0,\"num_cameras\":0,"
              "\"num_frames\":0,\"num_registered_frames\":0,\"num_images\":0,"
              "\"num_registered_images\":0,\"num_points3D\":0,"
              "\"num_observations\":0}\n");
  }
}

TEST(ModelAnalyzer, RejectsUnsupportedFormatAndJsonVerboseBeforeReading) {
  const auto path = CreateTestDir() / "missing";
  const std::vector<std::vector<std::string>> cases = {
      {"--output_format", "XML"},
      {"--output_format", "json"},
      {"--output_format", ""},
      {"--output_format", "JSON", "--verbose", "1"}};
  for (auto args : cases) {
    args.insert(args.end(), {"--path", path.string()});
    CaptureStdout capture;
    // A read of this missing model would throw instead of returning failure.
    EXPECT_EQ(Analyze(args), EXIT_FAILURE);
    EXPECT_TRUE(capture.Text().empty());
  }
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(ModelAnalyzer, FailedModelReadThrowsWithoutPartialJson) {
  const auto root = CreateTestDir();
  const auto empty_directory = root / "empty";
  const auto corrupt = root / "corrupt";
  const auto file_path = root / "not-a-directory";
  CreateDirIfNotExists(empty_directory);
  CreateDirIfNotExists(corrupt);
  Reconstruction().WriteText(corrupt);
  {
    std::ofstream stream(corrupt / "cameras.txt");
    stream << "not a camera\n";
  }
  { std::ofstream stream(file_path); }
  const auto before = Snapshot(root);
  for (const auto& path :
       {root / "missing", empty_directory, file_path, corrupt}) {
    CaptureStdout capture;
    EXPECT_ANY_THROW(
        Analyze({"--path", path.string(), "--output_format", "JSON"}));
    EXPECT_TRUE(capture.Text().empty());
  }
  EXPECT_EQ(Snapshot(root), before);
}

// Include directories as well as file bytes, so newly created empty output
// directories cannot escape the nonmutation check.
std::map<std::filesystem::path, std::string> Snapshot(
    const std::filesystem::path& root) {
  std::map<std::filesystem::path, std::string> entries;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(root)) {
    const auto path = std::filesystem::relative(entry.path(), root);
    if (entry.is_directory()) {
      entries[path] = "directory";
    } else {
      std::ifstream stream(entry.path(), std::ios::binary);
      entries[path] = std::string(std::istreambuf_iterator<char>(stream),
                                  std::istreambuf_iterator<char>());
    }
  }
  return entries;
}

class CaptureLog : public google::LogSink {
 public:
  CaptureLog() { google::AddLogSink(this); }
  ~CaptureLog() override { google::RemoveLogSink(this); }
  void send(google::LogSeverity,
            const char*,
            const char*,
            int,
            const struct ::tm*,
            const char* message,
            size_t length) override {
    messages.append(message, length);
    messages.push_back('\n');
  }
  std::string messages;
};

TEST(ModelAnalyzer, DefaultTextAndVerboseRemainUnchangedAndReadOnly) {
  const auto root = CreateTestDir();
  const auto path = root / "model";
  CreateDirIfNotExists(path);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 2;
  options.num_points3D = 0;
  SynthesizeDataset(options, &reconstruction);
  reconstruction.WriteBinary(path);
  const auto before = Snapshot(root);

  CaptureStdout capture;
  CaptureLog log;
  ASSERT_EQ(Analyze({"--path", path.string()}), EXIT_SUCCESS);
  const auto default_text = log.messages;
  EXPECT_NE(default_text.find("Rigs: 1\n"), std::string::npos);
  EXPECT_NE(default_text.find("Images: 2\n"), std::string::npos);
  EXPECT_NE(default_text.find("Observations: 0\n"), std::string::npos);
  EXPECT_NE(default_text.find("Mean reprojection error:"), std::string::npos);
  EXPECT_EQ(default_text.find("Camera Id:"), std::string::npos);
  EXPECT_TRUE(capture.Text().empty());

  log.messages.clear();
  ASSERT_EQ(Analyze({"--path", path.string(), "--output_format", "TEXT"}),
            EXIT_SUCCESS);
  EXPECT_EQ(log.messages, default_text);
  log.messages.clear();
  ASSERT_EQ(Analyze({"--path", path.string(), "--verbose", "1"}), EXIT_SUCCESS);
  EXPECT_NE(log.messages.find("Camera Id:"), std::string::npos);
  EXPECT_NE(log.messages.find("Registered Image Id:"), std::string::npos);
  EXPECT_TRUE(capture.Text().empty());

  log.messages.clear();
  ASSERT_EQ(Analyze({"--path",
                     path.string(),
                     "--output_format",
                     "JSON",
                     "--verbose",
                     "0"}),
            EXIT_SUCCESS);
  const auto json = capture.Text();
  EXPECT_FALSE(json.empty());
  EXPECT_TRUE(log.messages.empty());
  ASSERT_EQ(Analyze({"--path", path.string(), "--output_format", "JSON"}),
            EXIT_SUCCESS);
  EXPECT_EQ(capture.Text(), json + json);
  EXPECT_EQ(Snapshot(root), before);
}

}  // namespace
}  // namespace colmap
