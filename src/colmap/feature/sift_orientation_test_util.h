// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace colmap::test {

// Frozen independent oracle: PyramidCU::ReshapeFeatureListCPU from 07fa9c090,
// before eb2c26967 extracted the candidate helper. Only the per-level count is
// passed explicitly instead of reading _levelFeatureNum[i]. Keep this body
// independent: expected results must never call ReshapePackedOrientations.
inline int ReshapeOriginalResidentOrientations(float* buffer1,
                                               int count,
                                               float* buffer2) {
  int fcount = 0;
  float* src = buffer1;
  float* des = buffer2;
  const static double factor = 2.0 * 3.14159265358979323846 / 65535.0;
  for (int j = 0; j < count; j++, src += 4) {
    unsigned short* orientations = (unsigned short*)(&src[3]);
    if (orientations[0] != 65535) {
      des[0] = src[0];
      des[1] = src[1];
      des[2] = src[2];
      des[3] = float(factor * orientations[0]);
      fcount++;
      des += 4;
      if (orientations[1] != 65535 && orientations[1] != orientations[0]) {
        des[0] = src[0];
        des[1] = src[1];
        des[2] = src[2];
        des[3] = float(factor * orientations[1]);
        fcount++;
        des += 4;
      }
    }
  }
  return fcount;
}

}  // namespace colmap::test
