// SPDX-License-Identifier: BSD-3-Clause
// Synthetic Driver API seam ONLY: never hardware identity evidence.
#include <stdlib.h>
#include <string.h>

int cuInit(unsigned int flags) { return 0; }
int cuDeviceGet(int* device, int ordinal) {
  const char* visible = getenv("CUDA_VISIBLE_DEVICES");
  if (ordinal < 0) return 101;
  if (!visible) { *device = ordinal; return 0; }
  for (int i = 0; i < ordinal; ++i) {
    visible = strchr(visible, ',');
    if (!visible) return 101;
    ++visible;
  }
  if (!*visible) return 101;
  *device = atoi(visible);
  return 0;
}
int cuDeviceGetUuid_v2(unsigned char* uuid, int device) {
  memset(uuid, 0, 16);
  uuid[15] = device + 1;
  return 0;
}
