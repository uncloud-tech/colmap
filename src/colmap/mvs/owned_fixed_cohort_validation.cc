// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_fixed_cohort_validation.h"

#include "colmap/mvs/owned_fixed_cohort_inventory.h"
#include "colmap/mvs/patch_match_execution.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace colmap::mvs::owned {
namespace {
void Require(bool valid, const char* message) {
  if (!valid) throw std::invalid_argument(message);
}
}  // namespace

// Shared native Linux/cgroup-v2 observation, not a resource library/hard cap.
uint64_t ObserveFixedCohortHostAvailable() {
#if defined(__linux__)
  uint64_t available = 0;
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  while (std::getline(meminfo, line)) {
    std::istringstream fields(line);
    std::string key, unit;
    uint64_t value;
    if (fields >> key >> value >> unit && key == "MemAvailable:") {
      Require(unit == "kB", "fixed-cohort unexpected MemAvailable units");
      available = CheckedMultiply(value, 1024);
      break;
    }
  }
  Require(available > 0, "fixed-cohort RAM headroom query unavailable");
  std::ifstream membership("/proc/self/cgroup");
  std::filesystem::path group;
  while (std::getline(membership, line)) {
    if (line.rfind("0::/", 0) == 0) group = line.substr(3);
  }
  Require(!group.empty() && group.is_absolute(),
          "fixed-cohort cgroup-v2 membership unavailable");
  for (const auto& component : group)
    Require(component != "..",
            "fixed-cohort unsupported cgroup namespace path");
  std::filesystem::path root, mount;
  bool hierarchy_root = false;
  std::ifstream mounts("/proc/self/mountinfo");
  while (std::getline(mounts, line)) {
    if (line.find(" - cgroup2 ") == std::string::npos) continue;
    std::istringstream fields(line);
    std::string id, parent, dev, mount_root, mount_point;
    fields >> id >> parent >> dev >> mount_root >> mount_point;
    const auto relative = group.lexically_relative(mount_root);
    if (!relative.empty() && *relative.begin() != ".." &&
        mount_point.find('\\') == std::string::npos) {
      mount = mount_point;
      hierarchy_root = mount_root == "/";
      root = relative == "." ? mount : (mount / relative).lexically_normal();
      break;
    }
  }
  Require(!root.empty() && !mount.empty(),
          "fixed-cohort cgroup-v2 mount mapping unavailable");
  // Ancestor limits also constrain the leaf. Bound traversal inside this one
  // cgroup mount (never walk a filesystem root or inspect unrelated groups).
  for (size_t depth = 0;; ++depth) {
    Require(depth < 64, "fixed-cohort cgroup hierarchy too deep");
    std::string maximum, current;
    std::ifstream max_file(root / "memory.max");
    std::ifstream current_file(root / "memory.current");
    // Linux does not expose memory.max on the true hierarchy root: it has
    // no ancestor controller cap; MemAvailable still constrains this process.
    std::error_code error;
    const bool has_max = std::filesystem::exists(root / "memory.max", error);
    if (root == mount && hierarchy_root && !error && !has_max) break;
    Require(static_cast<bool>(max_file >> maximum) &&
                static_cast<bool>(current_file >> current),
            "fixed-cohort cgroup-v2 memory headroom query unavailable");
    const auto used = ParseBytes(current);
    if (maximum != "max") {
      const auto cap = ParseBytes(maximum);
      available = std::min(available, cap > used ? cap - used : 0);
    }
    if (root == mount) break;
    const auto parent = root.parent_path();
    Require(parent != root && parent != root.root_path(),
            "fixed-cohort cgroup mount traversal invalid");
    root = parent;
  }
  return available;
#else
  throw std::runtime_error("fixed-cohort RAM/cgroup headroom requires Linux");
#endif
}

int FixedCohortDevice(const PatchMatchOptions& options) {
  Require(!options.gpu_index.empty(), "fixed-cohort device missing");
  const auto device = ParseBytes(options.gpu_index);
  Require(device <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
          "fixed-cohort device out of range");
  Require(options.gpu_index == std::to_string(device),
          "fixed-cohort requires one explicit canonical device");
  return static_cast<int>(device);
}

void ValidateFixedCohortShape(const std::vector<Image>& images,
                              const std::vector<FixedCohortReference>& refs,
                              const FixedCohortBudget& budget) {
  Require(!images.empty(), "fixed-cohort images missing");
  ValidateFixedCohortJobShape(budget);
  Require(!refs.empty() && refs.size() == budget.photo_private.size(),
          "fixed-cohort reference/budget cardinality mismatch");
  const int device = FixedCohortDevice(refs[0].photo_options);
  std::set<int> producers;
  for (const auto& ref : refs) {
    Require(ref.reference_image_idx >= 0 &&
                static_cast<size_t>(ref.reference_image_idx) < images.size() &&
                producers.insert(ref.reference_image_idx).second,
            "fixed-cohort invalid/duplicate reference");
  }
  std::vector<FixedCohortImageShape> shapes;
  std::vector<FixedCohortJobShape> jobs;
  for (const auto& image : images) {
    Require(image.GetWidth() && image.GetHeight(),
            "fixed-cohort empty image shape");
    for (size_t k = 0; k < 9; ++k)
      Require(std::isfinite(image.GetK()[k]) && std::isfinite(image.GetR()[k]),
              "fixed-cohort nonfinite calibration");
    for (size_t k = 0; k < 3; ++k)
      Require(std::isfinite(image.GetT()[k]), "fixed-cohort nonfinite pose");
    Require(std::abs(image.GetK()[1]) < 1e-6f &&
                std::abs(image.GetK()[3]) < 1e-6f &&
                std::abs(image.GetK()[6]) < 1e-6f &&
                std::abs(image.GetK()[7]) < 1e-6f &&
                std::abs(image.GetK()[8] - 1) < 1e-6f && image.GetK()[0] > 0 &&
                image.GetK()[4] > 0,
            "fixed-cohort unsupported intrinsics");
    shapes.push_back({image.GetWidth(), image.GetHeight()});
  }
  for (size_t i = 0; i < refs.size(); ++i) {
    const auto& ref = refs[i];
    jobs.push_back(
        {ref.reference_image_idx, ref.photo_sources, ref.geo_sources});
    for (const bool geo : {false, true}) {
      const auto& options = geo ? ref.geo_options : ref.photo_options;
      const auto& sources = geo ? ref.geo_sources : ref.photo_sources;
      Require(options.Check() && std::isfinite(options.sigma_spatial) &&
                  options.sigma_spatial > 0 && options.depth_min > 0 &&
                  options.depth_max > options.depth_min &&
                  std::isfinite(options.depth_max) &&
                  options.geom_consistency == geo &&
                  FixedCohortDevice(options) == device &&
                  options.max_image_size == -1 && options.num_threads > 0 &&
                  !options.write_consistency_graph &&
                  !options.allow_missing_files && !sources.empty(),
              "fixed-cohort invalid phase options");
      ValidatePatchMatchExecution(PatchMatchExecution::kOwnedCuda,
                                  options.window_radius,
                                  options.window_step,
                                  geo,
                                  true,
                                  options.filter,
                                  options.write_consistency_graph,
                                  options.allow_missing_files);
      for (int source : sources) {
        Require(source >= 0 && static_cast<size_t>(source) < images.size() &&
                    source != ref.reference_image_idx &&
                    (!geo || producers.count(source)),
                "fixed-cohort missing source producer or invalid ordinal");
      }
    }
  }
  const auto minimum = BuildFixedCohortLogicalInventory(shapes, jobs, budget);
  Require(budget.caller_inputs.host >= minimum.caller_inputs.host &&
              budget.frozen_inputs.host >= minimum.frozen_inputs.host,
          "fixed-cohort input copy bound insufficient");
  for (size_t i = 0; i < refs.size(); ++i) {
    Require(budget.geo_staging_device[i] >= minimum.geo_staging_device[i],
            "fixed-cohort GEO staging bound below logical inventory");
    Require(
        budget.photo_private[i].host >= minimum.photo_private[i].host &&
            budget.photo_private[i].device >= minimum.photo_private[i].device &&
            budget.geo_private[i].host >= minimum.geo_private[i].host &&
            budget.geo_private[i].device >= minimum.geo_private[i].device,
        "fixed-cohort private bound below known logical inventory");
  }
  Require(budget.snapshots.device >= minimum.snapshots.device &&
              budget.host_outputs.host >= minimum.host_outputs.host,
          "fixed-cohort snapshot/output bound below logical maps");
}
void ValidateFixedCohortPixels(const std::vector<Image>& images,
                               const std::vector<FixedCohortReference>& refs,
                               const FixedCohortBudget& budget) {
  ValidateFixedCohortShape(images, refs, budget);
  uint64_t bytes = 0;
  for (const auto& image : images) {
    Require(image.GetBitmap().IsGrey() &&
                static_cast<size_t>(image.GetBitmap().Width()) ==
                    image.GetWidth() &&
                static_cast<size_t>(image.GetBitmap().Height()) ==
                    image.GetHeight(),
            "fixed-cohort requires loaded grayscale pixels");
    bytes = CheckedAdd(bytes, image.GetBitmap().NumBytes());
  }
  Require(
      budget.caller_inputs.host >= bytes && budget.frozen_inputs.host >= bytes,
      "fixed-cohort input copy bound insufficient");
}

}  // namespace colmap::mvs::owned
