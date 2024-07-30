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

#include <iostream>

#include <cuda_runtime.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace nvblox {

/// Whether the storage or processing is happening on CPU, GPU, or any future
/// amazing hardware- accelerated platform.
enum class DeviceType { kCPU, kGPU };

/// How GPU data is stored, either in Device-only or unified (both) memory.
/// NOTE(alexmillane): tag: c++17, switch to constexpr when we move to c++17.
enum class MemoryType { kDevice, kUnified, kHost };
inline std::string toString(MemoryType memory_type) {
  switch (memory_type) {
    case MemoryType::kDevice:
      return "kDevice";
      break;
    case MemoryType::kUnified:
      return "kUnified";
      break;
    default:
      return "kHost";
      break;
  }
}

typedef Eigen::Vector3i Index3D;
typedef Eigen::Vector2i Index2D;

typedef Eigen::Vector3f Vector3f;
typedef Eigen::Vector2f Vector2f;

typedef Eigen::AlignedBox3f AxisAlignedBoundingBox;

typedef Eigen::Isometry3f Transform;

/// This can be replaced with std::byte once we go to C++17.
typedef uint8_t Byte;

/// Aligned Eigen containers
template <typename Type>
using AlignedVector = std::vector<Type, Eigen::aligned_allocator<Type>>;

enum class InterpolationType { kNearestNeighbor, kLinear };

typedef Eigen::ParametrizedLine<float, 3> Ray;

// semantic relate
typedef uint16_t SemanticLabel;
typedef AlignedVector<SemanticLabel> SemanticLabels;

static constexpr uint8_t kUnknownSemanticLabelId = 0u;
// The size of this array determines how many semantic labels SemanticVoxblox
// supports.
// TODO(Toni): parametrize this, although that means it becomes unknown at
// compile time...
static constexpr size_t kTotalNumberOfLabels = 21; // 指明了21种标签类型
typedef float SemanticProbability; 
typedef Eigen::Matrix<SemanticProbability, kTotalNumberOfLabels, 1>
    SemanticProbabilities;
typedef Eigen::
    Matrix<SemanticProbability, kTotalNumberOfLabels, kTotalNumberOfLabels>
        SemanticLikelihoodFunction;

}  // namespace nvblox
