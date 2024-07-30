/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <cuda_runtime.h>

#include "nvblox/core/color.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"

namespace nvblox {

/// Row-major access to images (on host and device)
namespace image {
// 通用的行优先数据访问函数，命名空间为image，提高可读性，表明用于image访问
template <typename ElementType>
__host__ __device__ inline const ElementType& access(int row_idx, int col_idx,
                                                     int cols,
                                                     const ElementType* data) {
  return data[row_idx * cols + col_idx]; // 传入y, x, cols得到数据指针
}
template <typename ElementType>
__host__ __device__ inline ElementType& access(int row_idx, int col_idx,
                                               int cols, ElementType* data) {
  return data[row_idx * cols + col_idx];
}
template <typename ElementType>
__host__ __device__ inline const ElementType& access(int linear_idx,
                                                     const ElementType* data) {
  return data[linear_idx];
}
template <typename ElementType>
__host__ __device__ inline ElementType& access(int linear_idx,
                                               ElementType* data) {
  return data[linear_idx];
}

}  // namespace image

// 常量表达式，默认的图像内存类型为GPU
constexpr MemoryType kDefaultImageMemoryType = MemoryType::kDevice;

// 模板编程
// ImageBase无法通过设备访问？提供的更像是一个基础类
template <typename _ElementType>
class ImageBase {
 public:
  // "Save" the element type so it's queryable as Image::ElementType. 保留元素类型，可以通过Image::ElementType访问
  typedef _ElementType ElementType; 

  /// Destructor
  virtual ~ImageBase() = default;

  /// Attributes
  inline int cols() const { return cols_; }
  inline int rows() const { return rows_; }
  inline int numel() const { return cols_ * rows_; }
  inline int width() const { return cols_; }
  inline int height() const { return rows_; }

  /// Access
  /// NOTE(alexmillane): The guard-rails are off here. If you declare a kDevice
  /// Image and try to access its data, you will get undefined behaviour. 如果声明了一个kDevice并访问，会得到一个未定义行为
  inline const ElementType& operator()(const int row_idx,
                                       const int col_idx) const {
    return image::access(row_idx, col_idx, cols_, data_); // image::access是一个 __host__ __device__函数，主机和设备都可以调用，但是operator()没有__device__
  }
  inline ElementType& operator()(const int row_idx, const int col_idx) {
    return image::access(row_idx, col_idx, cols_, data_);
  }
  inline const ElementType& operator()(const int linear_idx) const {
    return image::access(linear_idx, data_);
  }
  inline ElementType& operator()(const int linear_idx) {
    return image::access(linear_idx, data_);
  }

  /// Raw pointer access
  inline ElementType* dataPtr() { return data_; }
  inline const ElementType* dataConstPtr() const { return data_; }

 protected:
  /// Constructors protected. Only callable from the child classes. 构造函数通过protected保护，只能通过派生类调用
  ImageBase(int rows, int cols, ElementType* data = nullptr)
      : rows_(rows), cols_(cols), data_(data) {}
  ImageBase() : rows_(0), cols_(0), data_(nullptr) {}

  // Delete copying
  ImageBase(const ImageBase& other) = delete;
  ImageBase& operator=(const ImageBase& other) = delete;

  // Delete moving
  ImageBase(ImageBase&& other) = delete;
  ImageBase& operator=(ImageBase&& other) = delete;

  int rows_;
  int cols_;
  ElementType* data_; // 实际存放数据的内存地址
};

/// Row-major image.
/// - Note that in a row-major image, rows follow one another in linear memory,
/// which means the column index varied between subsequent elements.
/// - Images use corner based indexing such that the pixel with index (0,0) is
/// centered at (0.5px,0.5px) and spans (0.0px,0.0px) to (1.0px,1.0px). 像素 (0, 1) 就是 (x, y)， 对应image[y=1][x=0], 像素中心为(0.5, 1.5) ，像素范围为(0, 1) to (0, 2)
/// - Points on the image plane are defined as: u_px = (u_px.x(), u_px.y()) =
/// (col_idx, row_idx) in pixels.
template <typename _ElementType>
class Image : public ImageBase<_ElementType> {
 public:
  // "Save" the element type so it's queryable as Image::ElementType.
  typedef _ElementType ElementType;

  explicit Image(MemoryType memory_type = kDefaultImageMemoryType)
      : memory_type_(memory_type) {} // 显式构造函数，防止隐式类型转换，只能通过显式转换 static_cast<type>

  Image(int rows, int cols, MemoryType memory_type = kDefaultImageMemoryType);

  virtual ~Image() = default;

  /// Move constructor and assignment 移动构造函数和赋值
  Image(Image&& other);
  Image& operator=(Image&& other);

  /// Deep copy constructor (second can be used to transition memory type)
  explicit Image(const Image& other);
  Image(const Image& other, MemoryType memory_type); // 可用于转换内存类型
  Image& operator=(const Image& other);

  // Attributes
  inline MemoryType memory_type() const { return memory_type_; }

  /// Set the image to 0.
  void setZero();

  /// Reset the contents of the image. Reallocate if the image got larger.
  ///@param rows Number of rows in the new image.
  ///@param colsNumber of columns in the new image.
  ///@param buffer A pointer to a memory buffer containing pixel data.
  ///@param memory_type Where the newly allocated image should reside.
  void populateFromBuffer(int rows, int cols, const ElementType* buffer,
                          MemoryType memory_type = kDefaultImageMemoryType);

  /// @brief Allocates a new image and populates it. Always allocates.
  /// @param rows Number of rows in the new image.
  /// @param cols Number of columns in the new image.
  /// @param buffer A pointer to a memory buffer containing pixel data.
  /// @param memory_type Where the newly allocated image should reside.
  /// @return A newly allocated image.
  static inline Image fromBuffer(
      int rows, int cols, const ElementType* buffer,
      MemoryType memory_type = kDefaultImageMemoryType);

 protected:
  MemoryType memory_type_;
  unified_ptr<ElementType[]> owned_data_;//unified_ptr 统一内存指针
};

template <typename _ElementType>
class ImageView : public ImageBase<_ElementType> {
 public:
  // "Save" the element type so it's queryable as Image::ElementType.
  typedef _ElementType ElementType;
  typedef typename std::remove_cv<_ElementType>::type ElementType_nonconst;

  /// Wraps an existing buffer
  /// Does not control memory lifetime. It is the users responsibility to ensure
  /// that the underlying memory exists throughout use.
  /// @param rows rows in image buffer
  /// @param cols cols in image buffer
  /// @param data memory buffer
  ImageView(int rows, int cols, ElementType* data = nullptr);

  /// Construct from an (memory-owning) Image
  /// Note that it is the users responsiblity to ensure the underlying image
  /// outlives the constructed view.
  /// @param image The (memory-owning) Image
  ImageView(Image<ElementType>& image);
  ImageView(const Image<ElementType_nonconst>& image);

  /// Destructor
  virtual ~ImageView() = default;

  /// (Shallow) Copy
  ImageView(const ImageView& other);
  ImageView& operator=(const ImageView& other);

  /// Move (ImageView is a shallow copy so a move-construction is the same as
  /// copy-construction)
  ImageView(ImageView&& other);
  ImageView& operator=(ImageView&& other);
};

/// Common Names
using DepthImage = Image<float>;
using ColorImage = Image<Color>;
using MonoImage = Image<uint8_t>;
using DepthImageView = ImageView<float>;
using ColorImageView = ImageView<Color>;
using MonoImageView = ImageView<uint8_t>;
using DepthImageConstView = ImageView<const float>;
using ColorImageConstView = ImageView<const Color>;
using MonoImageConstView = ImageView<const uint8_t>;

// Image Operations
namespace image {

float maxGPU(const DepthImage& image);
float minGPU(const DepthImage& image);
uint8_t maxGPU(const MonoImage& image);
uint8_t minGPU(const MonoImage& image);
std::pair<float, float> minmaxGPU(const DepthImage& image);

void elementWiseMinInPlaceGPU(const float constant, DepthImage* image);
void elementWiseMaxInPlaceGPU(const float constant, DepthImage* image);

void elementWiseMaxInPlaceGPU(const DepthImage& image_1, DepthImage* image_2);
void elementWiseMaxInPlaceGPU(const MonoImage& image_1, MonoImage* image_2);
void elementWiseMinInPlaceGPU(const DepthImage& image_1, DepthImage* image_2);
void elementWiseMinInPlaceGPU(const MonoImage& image_1, MonoImage* image_2);

void elementWiseMultiplicationInPlaceGPU(const float constant,
                                         DepthImage* image);

void getDifferenceImageGPU(const DepthImage& image_1, const DepthImage& image_2,
                           DepthImage* diff_image_ptr);
void getDifferenceImageGPU(const ColorImage& image_1, const ColorImage& image_2,
                           ColorImage* diff_image_ptr);
void getDifferenceImageGPU(const MonoImage& image_1, const MonoImage& image_2,
                           MonoImage* diff_image_ptr);

void castGPU(const DepthImage& image_in, MonoImage* image_out_ptr);

}  // namespace image
}  // namespace nvblox

#include "nvblox/sensors/internal/impl/image_impl.h"
