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
#include "nvblox/core/internal/warmup_cuda.h"

namespace nvblox {

__global__ void warm_up_gpu() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; // 
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

void warmupCuda() {
  warm_up_gpu<<<64, 128>>>(); //<<<grid_size, block_size>>> 64 * 128核函数 中总的线程数就等于网格大小乘以线程块大小，而三括号中的两个数字分别就是网格 大小和线程块大小，即 <<<网格大小, 线程块大小>>>
  cudaDeviceSynchronize();
}

}  // namespace nvblox