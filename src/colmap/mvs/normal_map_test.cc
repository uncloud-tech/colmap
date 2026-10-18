// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/normal_map.h"

#include <tuple>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

Mat<float> MakeTempNormalMat(const size_t width,
                             const size_t height,
                             const size_t depth,
                             const float value) {
  Mat<float> mat(width, height, depth);
  mat.Fill(value);
  return mat;
}

TEST(NormalMap, CopyConstructorPreservesSource) {
  Mat<float> source(2, 2, 3);
  source.Fill(0.5f);
  const float* source_ptr = source.GetPtr();
  const NormalMap normal_map(source);
  EXPECT_NE(normal_map.GetPtr(), source_ptr);
  EXPECT_EQ(normal_map.Get(1, 1, 2), 0.5f);
  EXPECT_EQ(source.Get(0, 0, 0), 0.5f);
  EXPECT_EQ(source.GetPtr(), source_ptr);
}

TEST(NormalMap, MovesOwnershipFromRvalueMat) {
  const size_t width = 3;
  const size_t height = 4;
  Mat<float> mat(width, height, 3);
  for (size_t slice = 0; slice < 3; ++slice) {
    for (size_t row = 0; row < height; ++row) {
      for (size_t col = 0; col < width; ++col) {
        mat.Set(row, col, slice,
                static_cast<float>(slice * 100 + row * width + col));
      }
    }
  }
  const float* source_ptr = mat.GetPtr();
  const NormalMap normal_map(std::move(mat));
  // Ownership of the downloaded buffer transferred without a second copy.
  EXPECT_EQ(normal_map.GetPtr(), source_ptr);
  EXPECT_EQ(normal_map.GetWidth(), width);
  EXPECT_EQ(normal_map.GetHeight(), height);
  EXPECT_EQ(normal_map.GetDepth(), 3);
  EXPECT_EQ(normal_map.Get(2, 1, 1),
            static_cast<float>(1 * 100 + 2 * width + 1));
  EXPECT_EQ(normal_map.GetNumBytes(), width * height * 3 * sizeof(float));
  // Moved-from object is safe: no buffer, and the source is not reused.
  EXPECT_TRUE(mat.GetData().empty());
  EXPECT_EQ(mat.GetPtr(), nullptr);
}

TEST(NormalMap, MovesOwnershipFromTemporaryMat) {
  const NormalMap normal_map(MakeTempNormalMat(2, 3, 3, 1.25f));
  EXPECT_EQ(normal_map.GetWidth(), 2);
  EXPECT_EQ(normal_map.GetHeight(), 3);
  EXPECT_EQ(normal_map.GetDepth(), 3);
  EXPECT_EQ(normal_map.Get(1, 1, 1), 1.25f);
  EXPECT_EQ(normal_map.GetNumBytes(), 18 * sizeof(float));
}

TEST(NormalMap, MoveConstructionChecksDepthBeforeMoving) {
  Mat<float> mat(2, 2, 1);
  mat.Fill(4.5f);
  const float* source_ptr = mat.GetPtr();
  EXPECT_THROW(std::ignore = NormalMap(std::move(mat)), std::exception);
  // The depth check must happen before the move, so the source stays intact.
  EXPECT_EQ(mat.GetPtr(), source_ptr);
  EXPECT_EQ(mat.Get(0, 0, 0), 4.5f);
}

TEST(NormalMap, Empty) {
  NormalMap normal_map;
  EXPECT_EQ(normal_map.GetWidth(), 0);
  EXPECT_EQ(normal_map.GetHeight(), 0);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, NonEmpty) {
  NormalMap normal_map(1, 2);
  EXPECT_EQ(normal_map.GetWidth(), 1);
  EXPECT_EQ(normal_map.GetHeight(), 2);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, Rescale) {
  NormalMap normal_map(6, 7);
  normal_map.Rescale(0.5);
  EXPECT_EQ(normal_map.GetWidth(), 3);
  EXPECT_EQ(normal_map.GetHeight(), 4);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, Downsize) {
  NormalMap normal_map(6, 7);
  normal_map.Downsize(2, 4);
  EXPECT_EQ(normal_map.GetWidth(), 2);
  EXPECT_EQ(normal_map.GetHeight(), 2);
  EXPECT_EQ(normal_map.GetDepth(), 3);
}

TEST(NormalMap, ToBitmap) {
  NormalMap normal_map(2, 2);
  normal_map.Set(0, 0, 0, 0);
  normal_map.Set(0, 0, 1, 0);
  normal_map.Set(0, 0, 2, 1);
  normal_map.Set(0, 1, 0, 0);
  normal_map.Set(0, 1, 1, 1);
  normal_map.Set(0, 1, 2, 0);
  normal_map.Set(1, 0, 0, 1);
  normal_map.Set(1, 0, 1, 0);
  normal_map.Set(1, 0, 2, 0);
  normal_map.Set(1, 1, 0, 1 / std::sqrt(2.0f));
  normal_map.Set(1, 1, 1, 1 / std::sqrt(2.0f));
  normal_map.Set(1, 1, 2, 0);
  const Bitmap bitmap = normal_map.ToBitmap();
  EXPECT_EQ(bitmap.Width(), normal_map.GetWidth());
  EXPECT_EQ(bitmap.Height(), normal_map.GetHeight());
  EXPECT_TRUE(bitmap.IsRGB());
  EXPECT_EQ(bitmap.GetPixel(0, 0).value(), BitmapColor<uint8_t>(128, 128, 0));
  EXPECT_EQ(bitmap.GetPixel(0, 1).value(), BitmapColor<uint8_t>(0, 128, 0));
  EXPECT_EQ(bitmap.GetPixel(1, 0).value(), BitmapColor<uint8_t>(128, 0, 0));
  EXPECT_EQ(bitmap.GetPixel(1, 1).value(), BitmapColor<uint8_t>(37, 37, 0));
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
