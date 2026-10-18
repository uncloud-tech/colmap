// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>

namespace colmap {

std::string GetVersionInfo();

std::string GetBuildInfo();

// Returns the Git commit revision compiled into this binary through the
// build configuration. Returns an empty string when the build was produced
// without commit metadata. Never performs a runtime Git lookup.
std::string GetCommitRevision();

// Computes database version number from version components.
// Format: major * 1000000 + minor * 10000 + patch * 100 + revision
// This gives 100 migration slots between consecutive releases.
int MakeDatabaseVersionNumber(int major, int minor, int patch, int revision);

// Returns the database version number for the current COLMAP version.
int GetDatabaseVersionNumber();

}  // namespace colmap
