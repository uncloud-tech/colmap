// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/mvs/owned_strip_test_identity.h"

#include <cassert>
#include <sstream>
#include <string>

int main() {
  const std::string commit(40, 'a'), fingerprint(64, 'b');
  const char* identity[] = {"same-harness", "--stereo-test-identity"};
  std::ostringstream output, error;
  const auto result = colmap::mvs::owned::StripTestIdentity(
      2, identity, commit, fingerprint, output, error);
  assert(result && *result == 0);
  assert(error.str().empty());
  assert(output.str() == "{\"schema_version\":1,\"source_commit\":\"" + commit +
                             "\",\"allocation_fingerprint\":\"" + fingerprint +
                             "\"}\n");
  for (const char* argument : {"reconstruction_info",
                               "--stereo-test-identity=x",
                               "--identity",
                               "--unexpected"}) {
    const char* args[] = {"same-harness", argument};
    std::ostringstream out, err;
    const auto refused = colmap::mvs::owned::StripTestIdentity(
        2, args, commit, fingerprint, out, err);
    assert(refused && *refused != 0);
    assert(out.str().empty());
  }
  const char* mixed[] = {
      "same-harness", "--stereo-test-identity", "--gtest_list_tests"};
  std::ostringstream out, err;
  assert(colmap::mvs::owned::StripTestIdentity(
             3, mixed, commit, fingerprint, out, err)
             .value() != 0);
  assert(out.str().empty());
  for (const std::string& bad_commit : {std::string{}, std::string(40, 'z')}) {
    std::ostringstream bad_out, bad_err;
    assert(colmap::mvs::owned::StripTestIdentity(
               2, identity, bad_commit, fingerprint, bad_out, bad_err)
               .value() != 0);
    assert(bad_out.str().empty());
  }
  {
    std::ostringstream bad_out, bad_err;
    assert(colmap::mvs::owned::StripTestIdentity(
               2, identity, commit, "unknown", bad_out, bad_err)
               .value() != 0);
    assert(bad_out.str().empty());
  }
  const char* tests[] = {"same-harness", "--gtest_filter=*.identity*"};
  std::ostringstream test_out, test_err;
  assert(!colmap::mvs::owned::StripTestIdentity(
      2, tests, commit, fingerprint, test_out, test_err));
  assert(!colmap::mvs::owned::StripTestIdentity(
      1, tests, commit, fingerprint, test_out, test_err));
  assert(test_out.str().empty());
}
