// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/multi_gpu_phase_plan.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace colmap::mvs::owned {
namespace {
void Require(bool valid, const char* message) {
  if (!valid) throw std::invalid_argument(message);
}
template <typename T>
void RequireUnique(std::vector<T> values, const char* message) {
  std::sort(values.begin(), values.end());
  Require(std::adjacent_find(values.begin(), values.end()) == values.end(),
          message);
}
uint64_t Add(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("multi-GPU phase addition overflow");
  return a + b;
}
uint64_t Mul(uint64_t a, uint64_t b) {
  if (a && b > std::numeric_limits<uint64_t>::max() / a)
    throw std::overflow_error("multi-GPU phase multiplication overflow");
  return a * b;
}
PhasePlanCharge Add(PhasePlanCharge a, PhasePlanCharge b) {
  return {Add(a.host, b.host), Add(a.device, b.device)};
}
PhasePlanCharge Max(PhasePlanCharge a, PhasePlanCharge b) {
  return {std::max(a.host, b.host), std::max(a.device, b.device)};
}
uint64_t Margin(uint64_t raw) {
  // ceil(103 * raw / 100), avoiding overflow of an intermediate product.
  return Add(raw, Add(Mul(raw / 100, 3), (raw % 100 * 3 + 99) / 100));
}
bool FitsDevice(uint64_t raw, const PhasePlanDevice& d) {
  return Margin(raw) <= d.capacity_bytes - d.reserve_bytes;
}

struct CohortState {
  std::vector<size_t> members;  // Request indices, not global IDs.
  PhasePlanCohort plan;
};

CohortState MakeCohort(const PhasePlanRequest& r,
                         const PhasePlanDevice& d,
                         std::vector<size_t> members) {
  CohortState result;
  result.members = std::move(members);
  std::vector<size_t> sources;  // Placement-only union; inputs stay untouched.
  uint64_t width = 0, height = 0;
  std::vector<uint64_t> host, device;
  for (const size_t index : result.members) {
    const auto& job = r.jobs[index];
    result.plan.jobs.push_back(job.id);
    host.push_back(job.private_raw.host);
    device.push_back(job.private_raw.device);
    if (!r.geometry)
      sources.insert(sources.end(), job.sources.begin(), job.sources.end());
  }
  std::sort(sources.begin(), sources.end());
  sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
  for (const auto source : sources) {
    width = std::max(width, r.images[source].width);
    height = std::max(height, r.images[source].height);
  }
  const uint64_t bytes = Mul(Mul(width, height), sources.size());
  result.plan.bank_raw = {bytes, bytes};
  std::sort(host.begin(), host.end(), std::greater<uint64_t>());
  std::sort(device.begin(), device.end(), std::greater<uint64_t>());
  PhasePlanCharge private_peak;
  const auto limit = std::min(r.active_owners_ceiling, result.members.size());
  for (size_t i = 0; i < limit; ++i) {
    private_peak = Add(private_peak, {host[i], device[i]});
    const auto peak = Add(d.fixed_raw, Add(result.plan.bank_raw, private_peak));
    if (peak.host > d.host_share_raw || !FitsDevice(peak.device, d)) break;
    result.plan.active_owners = i + 1;
    result.plan.private_peak_raw = private_peak;
  }
  return result;
}
PhasePlanCharge Peak(const PhasePlanDevice& d, const PhasePlanCohort& c) {
  return Add(d.fixed_raw, Add(c.bank_raw, c.private_peak_raw));
}
}  // namespace

PhasePlan BuildMultiGpuPhasePlan(const PhasePlanRequest& r) {
  Require(!r.devices.empty(), "multi-GPU phase requires approved devices");
  Require(r.active_owners_ceiling > 0, "multi-GPU phase owner ceiling is zero");
  Require(r.host_reserve_bytes <= r.host_capacity_bytes,
          "multi-GPU phase host reserve exceeds capacity");
  std::vector<std::string> identities, uuids;
  std::vector<size_t> ids;
  std::vector<int> ordinals;
  for (const auto& image : r.images) {
    Require(!image.identity.empty(), "multi-GPU phase empty image identity");
    identities.push_back(image.identity);
    Require(image.width && image.height, "multi-GPU phase empty image");
    Mul(image.width, image.height);
  }
  RequireUnique(identities, "multi-GPU phase duplicate image identity");
  uint64_t total_work = 0;
  for (const auto& job : r.jobs) {
    ids.push_back(job.id);
    Require(job.reference < r.images.size(),
            "multi-GPU phase invalid reference");
    for (const auto source : job.sources)
      Require(source < r.images.size(), "multi-GPU phase invalid source");
    total_work = Add(total_work, job.work);
  }
  RequireUnique(ids, "multi-GPU phase duplicate job ID");
  PhasePlan plan;
  plan.geometry = r.geometry;
  plan.host_peak_raw = r.common_host_raw;
  for (const auto& d : r.devices) {
    Require(d.ordinal >= 0 && !d.uuid.empty(),
            "multi-GPU phase invalid device identity");
    ordinals.push_back(d.ordinal);
    uuids.push_back(d.uuid);
    Require(d.reserve_bytes <= d.capacity_bytes,
            "multi-GPU phase device reserve exceeds capacity");
    Require(d.fixed_raw.host <= d.host_share_raw &&
                FitsDevice(d.fixed_raw.device, d),
            "multi-GPU phase fixed allowance exceeds device envelope");
    plan.groups.push_back({d, {}, d.fixed_raw, 0});
    plan.host_peak_raw = Add(plan.host_peak_raw, d.fixed_raw.host);
  }
  RequireUnique(ordinals, "multi-GPU phase duplicate device ordinal");
  RequireUnique(uuids, "multi-GPU phase duplicate device UUID");
  Require(Margin(plan.host_peak_raw) <=
              r.host_capacity_bytes - r.host_reserve_bytes,
          "multi-GPU phase common/fixed host capacity exceeded");

  std::vector<std::vector<CohortState>> states(r.devices.size());
  for (size_t index = 0; index < r.jobs.size(); ++index) {
    struct Candidate {
      size_t group;
      bool fresh;
      CohortState cohort;
      PhasePlanCharge peak;
      uint64_t host_peak;
      uint64_t incremental;
    };
    std::vector<Candidate> candidates;
    for (size_t g = 0; g < plan.groups.size(); ++g) {
      const auto& group = plan.groups[g];
      // Prefer extending the live cohort. If its complete union cannot admit
      // even one owner, drain it and construct a new independent bank.
      for (const bool fresh : {false, true}) {
        if (!fresh && states[g].empty()) continue;
        auto members = fresh ? std::vector<size_t>{} : states[g].back().members;
        members.push_back(index);
        auto cohort = MakeCohort(r, group.device, std::move(members));
        if (!cohort.plan.active_owners) continue;
        auto peak = Peak(group.device, cohort.plan);
        const size_t retained = states[g].size() - (fresh ? 0 : 1);
        for (size_t c = 0; c < retained; ++c)
          peak = Max(peak, Peak(group.device, states[g][c].plan));
        const auto host_peak =
            Add(plan.host_peak_raw - group.peak_raw.host, peak.host);
        if (Margin(host_peak) > r.host_capacity_bytes - r.host_reserve_bytes)
          continue;
        const uint64_t previous =
            fresh ? 0 : states[g].back().plan.bank_raw.device;
        const auto incremental = cohort.plan.bank_raw.device - previous;
        candidates.push_back(
            {g, fresh, std::move(cohort), peak, host_peak, incremental});
        break;
      }
    }
    Require(!candidates.empty(),
            "multi-GPU phase complete singleton cannot fit");
    // Seed every feasible idle GPU first. Thereafter admit affinity candidates
    // no more than one current job's work above the least-loaded feasible GPU.
    // This bounds locality preference; GEO's zero bank cost reduces to balance.
    uint64_t least_work = std::numeric_limits<uint64_t>::max();
    bool idle = false;
    for (const auto& c : candidates) {
      least_work = std::min(least_work, plan.groups[c.group].assigned_work);
      idle = idle || states[c.group].empty();
    }
    const auto balanced_limit = Add(least_work, r.jobs[index].work);
    size_t best = candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i) {
      const auto& c = candidates[i];
      const auto work = plan.groups[c.group].assigned_work;
      if ((idle && !states[c.group].empty()) || work > balanced_limit) continue;
      if (best == candidates.size() ||
          std::make_tuple(c.incremental, work, c.group) <
              std::make_tuple(candidates[best].incremental,
                              plan.groups[candidates[best].group].assigned_work,
                              candidates[best].group)) {
        best = i;
      }
    }
    Require(best != candidates.size(),
            "multi-GPU phase no useful owner progress");
    auto& chosen = candidates[best];
    auto& group = plan.groups[chosen.group];
    group.assigned_work = Add(group.assigned_work, r.jobs[index].work);
    group.peak_raw = chosen.peak;
    plan.host_peak_raw = chosen.host_peak;
    if (chosen.fresh) {
      states[chosen.group].push_back(std::move(chosen.cohort));
    } else {
      states[chosen.group].back() = std::move(chosen.cohort);
    }
  }
  for (size_t g = 0; g < states.size(); ++g)
    for (auto& cohort : states[g])
      plan.groups[g].cohorts.push_back(std::move(cohort.plan));
  return plan;
}
}  // namespace colmap::mvs::owned
