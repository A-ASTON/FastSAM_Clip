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
#include "nvblox/core/color.h"
#include "nvblox/sensors/csv_iterator.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>

namespace nvblox {

Color Color::blendTwoColors(const Color& first_color, float first_weight,
                            const Color& second_color, float second_weight) {
  float total_weight = first_weight + second_weight;

  first_weight /= total_weight;
  second_weight /= total_weight;

  Color new_color;
  new_color.r = static_cast<uint8_t>(std::round(
      first_color.r * first_weight + second_color.r * second_weight));
  new_color.g = static_cast<uint8_t>(std::round(
      first_color.g * first_weight + second_color.g * second_weight));
  new_color.b = static_cast<uint8_t>(std::round(
      first_color.b * first_weight + second_color.b * second_weight));
  new_color.a = static_cast<uint8_t>(std::round(
      first_color.a * first_weight + second_color.a * second_weight));

  return new_color;
}

// __host__ __device__ HashableColor::HashableColor(const Color& color) : Color(color) {}
// __host__ __device__ HashableColor::HashableColor() : Color() {}
// __host__ __device__ HashableColor::HashableColor(uint8_t r, uint8_t g, uint8_t b)
//     : HashableColor(r, g, b, 255) {}
// __host__ __device__ HashableColor::HashableColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
//     : Color(r, g, b, a) {}

// bool HashableColor::operator==(const HashableColor& other) const {
//   return (r == other.r && g == other.g && b == other.b && a == other.a);
// }

// bool HashableColor::equal(const HashableColor& color) const {
//   return r == color.r && g == color.g && b == color.b && a == color.a;
// }

// // 哈希函数，通过Color计算hash值
// size_t ColorHasher::operator()(const HashableColor& k) const {
//   // Compute individual hash values for first,
//   // second and third and combine them using XOR
//   // and bit shifting:
//   // TODO(Toni): use alpha value as well!!
//   return ((std::hash<uint8_t>()(k.r) ^ (std::hash<uint8_t>()(k.g) << 1)) >> 1) ^
//          (std::hash<uint8_t>()(k.b) << 1);
// }

// SemanticLabel2Color::SemanticLabel2Color(const std::string& filename)
//     : color_to_semantic_label_(), semantic_label_to_color_map_() {
//       // 构造函数，初始化在CPU上进行，然后将得到的color_to_semantic_label_ 以及 semantic_label_to_color_map_存放到GPU上？
//   std::ifstream file(filename.c_str());
//   CHECK(file.good()) << "Couldn't open file: " << filename.c_str();
//   size_t row_number = 1;
//   ColorToSemanticLabelMap color_to_semantic_label;
//   SemanticLabelToColorMap semantic_label_to_color_map;
//   for (CSVIterator loop(file); loop != CSVIterator(); ++loop) {
//     // We expect the CSV to have header:
//     // 0   , 1  , 2    , 3   , 4    , 5
//     // name, red, green, blue, alpha, id
//     CHECK_EQ(loop->size(), 6) << "Row " << row_number << " is invalid.";
//     uint8_t r = std::atoi((*loop)[1].c_str());
//     uint8_t g = std::atoi((*loop)[2].c_str());
//     uint8_t b = std::atoi((*loop)[3].c_str());
//     uint8_t a = std::atoi((*loop)[4].c_str());
//     uint8_t id = std::atoi((*loop)[5].c_str());
//     HashableColor rgba = HashableColor(r, g, b, a);
//     semantic_label_to_color_map[id] = rgba;
//     color_to_semantic_label[rgba] = id;
//     row_number++;
//   }
//   // TODO(Toni): remove
//   // Assign color 255,255,255 to unknown object 0u
//   semantic_label_to_color_map[kUnknownSemanticLabelId] =
//       HashableColor::White();
//   color_to_semantic_label[HashableColor::White()] = kUnknownSemanticLabelId;
//   cudaMemcpy(color_to_semantic_label_, *color_to_semantic_label, sizeof(std::pair<>))
// }

// __host__ __device__ SemanticLabel SemanticLabel2Color::getSemanticLabelFromColor(
//     const HashableColor& color) const {
//   const auto& it = color_to_semantic_label_->find(color);
//   if (it != color_to_semantic_label_->end()) {
//     return it->second;
//   } else {
//     LOG(ERROR) << "Caught an unknown color: \n"
//                << "RGBA: " << std::to_string(color.r) << ' '
//                <<  std::to_string(color.g) << ' '
//                <<  std::to_string(color.b) << ' '
//                <<  std::to_string(color.a);
//     return kUnknownSemanticLabelId; // Assign unknown label for now...
//   }
// }

// __host__ __device__ HashableColor SemanticLabel2Color::getColorFromSemanticLabel(
//     const SemanticLabel& semantic_label) const {
//   const auto& it = semantic_label_to_color_map_.find(semantic_label);
//   if (it != semantic_label_to_color_map_.end()) {
//     return it->second;
//   } else {
//     LOG(ERROR) << "Caught an unknown semantic label: \n"
//                << "Label: " << std::to_string(semantic_label);
//     return HashableColor(); // Assign unknown color for now...
//   }
// }

}  // namespace nvblox