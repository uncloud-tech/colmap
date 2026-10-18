// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/depth_map.h"

#include <tuple>

#include <gtest/gtest.h>

namespace colmap {
namespace mvs {
namespace {

Mat<float> MakeTempMat(const size_t width,
                       const size_t height,
                       const size_t depth,
                       const float value) {
  Mat<float> mat(width, height, depth);
  mat.Fill(value);
  return mat;
}

TEST(DepthMap, CopyConstructorPreservesSource) {
  Mat<float> source(2, 2, 1);
  source.Fill(3.5f);
  const float* source_ptr = source.GetPtr();
  const DepthMap depth_map(source, 1.0f, 2.0f);
  EXPECT_NE(depth_map.GetPtr(), source_ptr);
  EXPECT_EQ(depth_map.Get(0, 0), 3.5f);
  EXPECT_EQ(depth_map.GetDepthMin(), 1.0f);
  EXPECT_EQ(depth_map.GetDepthMax(), 2.0f);
  EXPECT_EQ(source.Get(0, 0), 3.5f);
  EXPECT_EQ(source.GetPtr(), source_ptr);
}

TEST(DepthMap, MovesOwnershipFromRvalueMat) {
  const size_t width = 3;
  const size_t height = 4;
  Mat<float> mat(width, height, 1);
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      mat.Set(row, col, 0, 1.5f + static_cast<float>(row * width + col));
    }
  }
  const float* source_ptr = mat.GetPtr();
  const DepthMap depth_map(std::move(mat), 0.25f, 7.75f);
  // Ownership of the downloaded buffer transferred without a second copy.
  EXPECT_EQ(depth_map.GetPtr(), source_ptr);
  EXPECT_EQ(depth_map.GetWidth(), width);
  EXPECT_EQ(depth_map.GetHeight(), height);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), 0.25f);
  EXPECT_EQ(depth_map.GetDepthMax(), 7.75f);
  EXPECT_EQ(depth_map.Get(0, 0), 1.5f);
  EXPECT_EQ(depth_map.Get(height - 1, width - 1),
            1.5f + static_cast<float>(height * width - 1));
  EXPECT_EQ(depth_map.GetNumBytes(), width * height * sizeof(float));
  // Moved-from object is safe: no buffer, and the source is not reused.
  EXPECT_TRUE(mat.GetData().empty());
  EXPECT_EQ(mat.GetPtr(), nullptr);
}

TEST(DepthMap, MovesOwnershipFromTemporaryMat) {
  const DepthMap depth_map(MakeTempMat(2, 3, 1, 2.5f), 1.0f, 9.0f);
  EXPECT_EQ(depth_map.GetWidth(), 2);
  EXPECT_EQ(depth_map.GetHeight(), 3);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.Get(1, 1), 2.5f);
  EXPECT_EQ(depth_map.GetDepthMin(), 1.0f);
  EXPECT_EQ(depth_map.GetDepthMax(), 9.0f);
  EXPECT_EQ(depth_map.GetNumBytes(), 6 * sizeof(float));
}

TEST(DepthMap, MoveConstructionChecksDepthBeforeMoving) {
  Mat<float> mat(2, 2, 2);
  mat.Fill(4.5f);
  const float* source_ptr = mat.GetPtr();
  EXPECT_THROW(std::ignore = DepthMap(std::move(mat), 1.0f, 2.0f),
               std::exception);
  // The depth check must happen before the move, so the source stays intact.
  EXPECT_EQ(mat.GetPtr(), source_ptr);
  EXPECT_EQ(mat.Get(0, 0, 0), 4.5f);
}

TEST(DepthMap, Empty) {
  DepthMap depth_map;
  EXPECT_EQ(depth_map.GetWidth(), 0);
  EXPECT_EQ(depth_map.GetHeight(), 0);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), -1);
  EXPECT_EQ(depth_map.GetDepthMax(), -1);
}

TEST(DepthMap, NonEmpty) {
  DepthMap depth_map(1, 2, 0, 1);
  EXPECT_EQ(depth_map.GetWidth(), 1);
  EXPECT_EQ(depth_map.GetHeight(), 2);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), 0);
  EXPECT_EQ(depth_map.GetDepthMax(), 1);
}

TEST(DepthMap, Rescale) {
  DepthMap depth_map(6, 7, 0, 1);
  depth_map.Rescale(0.5);
  EXPECT_EQ(depth_map.GetWidth(), 3);
  EXPECT_EQ(depth_map.GetHeight(), 4);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), 0);
  EXPECT_EQ(depth_map.GetDepthMax(), 1);
}

TEST(DepthMap, Downsize) {
  DepthMap depth_map(6, 7, 0, 1);
  depth_map.Downsize(2, 4);
  EXPECT_EQ(depth_map.GetWidth(), 2);
  EXPECT_EQ(depth_map.GetHeight(), 2);
  EXPECT_EQ(depth_map.GetDepth(), 1);
  EXPECT_EQ(depth_map.GetDepthMin(), 0);
  EXPECT_EQ(depth_map.GetDepthMax(), 1);
}

TEST(DepthMap, ToBitmap) {
  DepthMap depth_map(2, 2, 0.1, 0.9);
  depth_map.Fill(0.9);
  depth_map.Set(0, 0, 0, 0.1);
  depth_map.Set(0, 1, 0, 0.5);
  const Bitmap bitmap = depth_map.ToBitmap(0, 100);
  EXPECT_EQ(bitmap.Width(), depth_map.GetWidth());
  EXPECT_EQ(bitmap.Height(), depth_map.GetHeight());
  EXPECT_TRUE(bitmap.IsRGB());
  EXPECT_EQ(bitmap.GetPixel(0, 0).value(), BitmapColor<uint8_t>(0, 0, 128));
  EXPECT_EQ(bitmap.GetPixel(0, 1).value(), BitmapColor<uint8_t>(128, 0, 0));
  EXPECT_EQ(bitmap.GetPixel(1, 0).value(), BitmapColor<uint8_t>(128, 255, 127));
  EXPECT_EQ(bitmap.GetPixel(1, 1).value(), BitmapColor<uint8_t>(128, 0, 0));
}

}  // namespace
}  // namespace mvs
}  // namespace colmap
