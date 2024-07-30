#include <cuda_runtime.h>
#include "nvblox/core/color.h"
#include "nvblox/core/color_hash.cuh"
#include "nvblox/sensors/csv_iterator.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
namespace nvblox {
SemanticLabel2Color::~SemanticLabel2Color() {
  cudaFree(color_map_device_);
}

SemanticLabel2Color::SemanticLabel2Color(const std::string& filename) {
  // 构造函数，初始化在CPU上进行，然后将得到的color_to_semantic_label_ 以及 semantic_label_to_color_map_存放到GPU上？
  std::ifstream file(filename.c_str());
  CHECK(file.good()) << "Couldn't open file: " << filename.c_str();
  size_t row_number = 1;

  color_map_host_.push_back(HashableColor(0, 0, 0));
  color_map_size_ += 1;

  std::cout << " start generate semanticlabel2color map" << std::endl;
  // reserve，但是不知道行数
  for (CSVIterator loop(file); loop != CSVIterator(); ++loop) {
    // We expect the CSV to have header:
    // 0   , 1  , 2    , 3   , 4    , 5
    // name, red, green, blue, alpha, id
    CHECK_EQ(loop->size(), 6) << "Row " << row_number << " is invalid.";
    if (row_number++ == 1) {
      continue;
    }
    uint8_t r = std::atoi((*loop)[1].c_str());
    uint8_t g = std::atoi((*loop)[2].c_str());
    uint8_t b = std::atoi((*loop)[3].c_str());
    uint8_t a = std::atoi((*loop)[4].c_str());
    uint8_t id = std::atoi((*loop)[5].c_str());
    HashableColor rgba = HashableColor(r, g, b, a);

    if (std::find(color_map_host_.begin(), color_map_host_.end(), rgba) != color_map_host_.end()) {
      // 出现过了
      continue;
    } 
    color_map_host_.push_back(rgba);
    color_map_size_ += 1;
  }

  color_map_host_.resize(color_map_size_);

  //  // Assign color 255,255,255 to unknown object 0u
  // color_map_host_[kUnknownSemanticLabelId] = HashableColor::White();
  checkCudaErrors(cudaMalloc((void **)&color_map_device_, sizeof(HashableColor) * color_map_host_.size()));
  checkCudaErrors(cudaMemcpyAsync(color_map_device_, color_map_host_.data(), sizeof(HashableColor) * color_map_host_.size(), cudaMemcpyHostToDevice));

  std::cout << " finish generate semanticlabel2color map, size:" << color_map_host_.size() << std::endl;
}


}
