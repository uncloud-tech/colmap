// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/controllers/image_reader.h"
#include "colmap/feature/extractor.h"
#include "colmap/util/threading.h"

#include <filesystem>

namespace colmap {

// Read only after the controller has joined. Budgeted failures leave honest
// partial database state and must not be reported as successful extraction.
struct FeatureExtractionResult {
  std::exception_ptr error;
  bool success = false;
};

// Reads images from a folder, extracts features, and writes them to database.
std::unique_ptr<Thread> CreateFeatureExtractorController(
    const std::filesystem::path& database_path,
    const ImageReaderOptions& reader_options,
    const FeatureExtractionOptions& extraction_options,
    std::shared_ptr<FeatureExtractionResult> result = nullptr);

// Import features from text files. Each image must have a corresponding text
// file with the same name and an additional ".txt" suffix.
std::unique_ptr<Thread> CreateFeatureImporterController(
    const std::filesystem::path& database_path,
    const ImageReaderOptions& reader_options,
    const std::filesystem::path& import_path);

}  // namespace colmap
