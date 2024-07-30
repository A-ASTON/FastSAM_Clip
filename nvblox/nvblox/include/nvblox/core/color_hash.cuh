#pragma once
#include <cuda_runtime.h>
#include "nvblox/core/color.h"
#include <nvblox/core/types.h>
#include <stdgpu/cstddef.h>
#include <stdgpu/unordered_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <stdio.h>
namespace nvblox {

// HashableColor既为Key也为Value
struct HashableColor : public Color {
  __host__ __device__ HashableColor(const Color& color) : Color(color) {}
  __host__ __device__ HashableColor() : Color() {}
  __host__ __device__ HashableColor(uint8_t r, uint8_t g, uint8_t b)
      : HashableColor(r, g, b, 255) {}
  __host__ __device__ HashableColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
      : Color(r, g, b, a) {}

  __host__ __device__ bool operator==(const HashableColor& other) const {
    return (r == other.r && g == other.g && b == other.b && a == other.a);
  }

  __host__ __device__ bool equal(const HashableColor& color) const {
    return r == color.r && g == color.g && b == color.b && a == color.a;
  }
};

struct ColorHasher {
  __host__ __device__ std::size_t operator()(const HashableColor& k) const {
    return static_cast<unsigned int>((stdgpu::hash<uint8_t>()(k.r) ^ (stdgpu::hash<uint8_t>()(k.g) << 1)) >> 1) ^
            (stdgpu::hash<uint8_t>()(k.b) << 1);
  }
};

// 直接通过下标和id对应就可以了，使用thrust::device_vector，下标为id即label
// 可以便捷地通过label获得color，如何通过color获取label呢？先通过遍历寻找，后续再优化
// 错误的，thrust::device_vector数据虽然在GPU上，但是其函数是在主机端访问的
// 直接用HashableColor数组

// class 
class SemanticLabel2Color {
 public:
 // CPU上调用的，因为在cpu上读取
  SemanticLabel2Color(const std::string& filename);
  ~SemanticLabel2Color();
  
  // color label map, device ptr
  HashableColor* color_map_device_;
  int color_map_size_;
  std::vector<HashableColor> color_map_host_;
};

__device__ inline bool getSemanticLabelFromColor(
    HashableColor* color_map_device,
    int color_map_size,
    const HashableColor color,
    SemanticLabel* semantic_label) {
  
  for (int i = 0; i < color_map_size; i++) {
    if (color_map_device[i] == color) {
      *semantic_label = i;
      return true;
    }
  }
  
  *semantic_label = kUnknownSemanticLabelId;
  return false;
}

__device__ inline bool getColorFromSemanticLabel(
    HashableColor* color_map_device,
    int color_map_size,
    const SemanticLabel semantic_label,
    HashableColor* color) {
      
    // get color from label
    if (semantic_label >= 0 && semantic_label < color_map_size) {
      *color = color_map_device[semantic_label];
      return true;
    } else {
      *color = HashableColor();
      return false;
    }

}

}