// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/exe/photo_directory_summary.h"

#include "colmap/util/logging.h"
#include "colmap/util/version.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

namespace colmap {

namespace {

// The V1 inspection contract (docs/designs/photo-cli-interface.md) accepts
// exactly one option, `--image_path DIR`. Parsing it here with a minimal
// dedicated scan—instead of the general OptionManager, whose logging
// defaults write log files and whose accepted logging override options can
// create log directories or contaminate stdout—guarantees the stderr-only,
// non-writing diagnostics contract without changing any global OptionManager
// defaults or introducing a generic parser framework: every other option,
// including `--log_target`, `--log_path`, `--log_level`, `--log_severity`
// and `--log_color`, is rejected before any side effect. Diagnostics go to
// stderr only.
bool ParseImagePathArgument(int argc, char** argv, std::string* image_path) {
  constexpr std::string_view kImagePathOption = "--image_path";
  constexpr std::string_view kImagePathAssignment = "--image_path=";
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    std::string_view value;
    if (arg == kImagePathOption) {
      if (i + 1 >= argc) {
        std::cerr << "Option `--image_path` requires a directory argument."
                  << std::endl;
        return false;
      }
      value = std::string_view(argv[++i]);
    } else if (arg.compare(
                   0, kImagePathAssignment.size(), kImagePathAssignment) == 0) {
      value = arg.substr(kImagePathAssignment.size());
    } else {
      std::cerr << "Unsupported option `" << arg
                << "`: this inspection command only accepts `--image_path "
                   "DIR`; logging override options are not accepted."
                << std::endl;
      return false;
    }
    if (!image_path->empty()) {
      std::cerr << "Option `--image_path` given more than once." << std::endl;
      return false;
    }
    if (value.empty()) {
      std::cerr << "Option `--image_path` requires a non-empty directory "
                   "argument."
                << std::endl;
      return false;
    }
    *image_path = std::string(value);
  }
  if (image_path->empty()) {
    std::cerr << "Required option `--image_path DIR` was not provided."
              << std::endl;
    return false;
  }
  return true;
}

}  // namespace

// V1 inspection command (docs/designs/photo-cli-interface.md): writes exactly
// one typed JSON object to stdout and diagnostics to stderr. The command
// performs no reconstruction, GPU-context creation, hardware admission,
// output-directory creation or source mutation: the accepted metadata-only
// collector is reused with no second image parser. An invalid directory
// fails nonzero; a valid directory returns success even when scan problems
// exist, because the consumer must inspect the `problems`, `non_image_files`
// and `build` fields itself rather than relying on the exit status.
int RunPhotoDirectorySummary(int argc, char** argv) {
  // Enforce the stderr-only, non-writing diagnostics contract for this
  // command without changing the global OptionManager defaults (which write
  // log files and can create log directories): any log reached from this
  // command, including warnings from library code, goes to stderr only and
  // never creates log files or directories.
  FLAGS_logtostderr = true;
  FLAGS_alsologtostderr = false;
#if COLMAP_GLOG_HAS_STDOUT_SUPPORT
  FLAGS_logtostdout = false;
#endif
  FLAGS_log_dir = "";

  std::string image_path_arg;
  if (!ParseImagePathArgument(argc, argv, &image_path_arg)) {
    return EXIT_FAILURE;
  }

  const std::filesystem::path image_path = image_path_arg;

  PhotoDirectorySummary summary;
  try {
    summary = SummarizePhotoDirectory(image_path);
  } catch (const std::exception& exc) {
    std::cerr << "Failed to summarize photo directory " << image_path_arg
              << ": " << exc.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Failed to summarize photo directory " << image_path_arg
              << " for an unknown reason." << std::endl;
    return EXIT_FAILURE;
  }

  // Compiled-only identity: no runtime Git lookup and no runtime
  // fingerprint qualification in this slice.
  PhotoDirectorySummaryJsonBuild build;
  build.revision = GetCommitRevision();
  build.version = GetVersionInfo();
#if defined(COLMAP_CUDA_ENABLED)
  // The contract's `cuda_compiled` means the CUDA backend specifically, not
  // generic GPU availability: a HIP-only compilation reports false.
  build.cuda_compiled = true;
#else
  build.cuda_compiled = false;
#endif
#if defined(COLMAP_CUDA_ENABLED) && defined(COLMAP_MVS_ENABLED) && \
    !defined(COLMAP_HIP_ENABLED)
  build.stock_prng_state = mvs::GetStockPrngStateLayout();
#endif

  std::cout << WritePhotoDirectorySummaryJson(summary, image_path_arg, build)
            << '\n';
  return EXIT_SUCCESS;
}

}  // namespace colmap