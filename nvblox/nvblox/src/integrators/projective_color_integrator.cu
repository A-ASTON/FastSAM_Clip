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
#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

ProjectiveColorIntegrator::ProjectiveColorIntegrator() {
  sphere_tracer_.maximum_ray_length_m(max_integration_distance_m_);
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

ProjectiveColorIntegrator::~ProjectiveColorIntegrator() {
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

// integrateFrame 插入单个RGB帧
void ProjectiveColorIntegrator::integrateFrame(
    const ColorImage& color_frame, const Transform& T_L_C, const Camera& camera,
    const TsdfLayer& tsdf_layer, ColorLayer* color_layer,
    std::vector<Index3D>* updated_blocks) {
  timing::Timer color_timer("color/integrate");
  CHECK_NOTNULL(color_layer);
  CHECK_EQ(tsdf_layer.block_size(), color_layer->block_size());

  // Metric truncation distance for this layer
  const float voxel_size =
      color_layer->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size; // 截断距离

  timing::Timer blocks_in_view_timer("color/integrate/get_blocks_in_view");
  std::vector<Index3D> block_indices = view_calculator_.getBlocksInViewPlanes(
      T_L_C, camera, color_layer->block_size(),
      max_integration_distance_m_ + truncation_distance_m); // 获取block的index
  blocks_in_view_timer.Stop();

  // Check which of these blocks are:
  // - Allocated in the TSDF, and
  // - have at least a single voxel within the truncation band
  // This is because:
  // - We don't allocate new geometry here, we just color existing geometry
  // - We don't color freespace.
  // 检查在截断距离band内是否至少有一个体素，我们不会创建新的几何，只是对已有几何进行着色，不会对空白空间着色
  timing::Timer blocks_in_band_timer(
      "color/integrate/reduce_to_blocks_in_band");
  block_indices = reduceBlocksToThoseInTruncationBand(block_indices, tsdf_layer,
                                                      truncation_distance_m);
  blocks_in_band_timer.Stop();

  // Allocate blocks (CPU)
  // We allocate color blocks where
  // - there are allocated TSDF blocks, AND
  // - these blocks are within the truncation band
  timing::Timer allocate_blocks_timer("color/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, color_layer); // 在color_layer分配需要的blocks，实际上在GPU上也分配了
  allocate_blocks_timer.Stop();

  // Create a synthetic depth image 创建合成深度图
  timing::Timer sphere_trace_timer("color/integrate/sphere_trace");
  sphere_tracer_.renderImageOnGPU(
      camera, T_L_C, tsdf_layer, truncation_distance_m, &synthetic_depth_image_,
      MemoryType::kDevice, sphere_tracing_ray_subsampling_factor_); // 
  sphere_trace_timer.Stop();

  // Update identified blocks
  // Calls out to the child-class implementing the integation (GPU)
  timing::Timer update_blocks_timer("color/integrate/update_blocks");
  updateBlocks(block_indices, color_frame, synthetic_depth_image_, T_L_C,
               camera, truncation_distance_m, color_layer); // 在GPU上更新Blocks
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

void ProjectiveColorIntegrator::sphere_tracing_ray_subsampling_factor(
    int sphere_tracing_ray_subsampling_factor) {
  CHECK_GT(sphere_tracing_ray_subsampling_factor, 0);
  sphere_tracing_ray_subsampling_factor_ =
      sphere_tracing_ray_subsampling_factor;
}

int ProjectiveColorIntegrator::sphere_tracing_ray_subsampling_factor() const {
  return sphere_tracing_ray_subsampling_factor_;
}

float ProjectiveColorIntegrator::truncation_distance_vox() const {
  return truncation_distance_vox_;
}

float ProjectiveColorIntegrator::max_weight() const { return max_weight_; }

float ProjectiveColorIntegrator::max_integration_distance_m() const {
  return max_integration_distance_m_;
}

void ProjectiveColorIntegrator::truncation_distance_vox(
    float truncation_distance_vox) {
  CHECK_GT(truncation_distance_vox, 0.0f);
  truncation_distance_vox_ = truncation_distance_vox;
}

void ProjectiveColorIntegrator::max_weight(float max_weight) {
  CHECK_GT(max_weight, 0.0f);
  max_weight_ = max_weight;
}

void ProjectiveColorIntegrator::max_integration_distance_m(
    float max_integration_distance_m) {
  CHECK_GT(max_integration_distance_m, 0.0f);
  max_integration_distance_m_ = max_integration_distance_m;
}

float ProjectiveColorIntegrator::get_truncation_distance_m(
    float voxel_size) const {
  return truncation_distance_vox_ * voxel_size;
}

WeightingFunctionType ProjectiveColorIntegrator::weighting_function_type()
    const {
  return weighting_function_type_;
}

void ProjectiveColorIntegrator::weighting_function_type(
    WeightingFunctionType weighting_function_type) {
  weighting_function_type_ = weighting_function_type;
}

const ViewCalculator& ProjectiveColorIntegrator::view_calculator() const {
  return view_calculator_;
}

/// Returns the object used to calculate the blocks in camera views.
ViewCalculator& ProjectiveColorIntegrator::view_calculator() {
  return view_calculator_;
}

__device__ inline Color blendTwoColors(const Color& first_color,
                                       float first_weight,
                                       const Color& second_color,
                                       float second_weight) {
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

  return new_color;
}

__device__ inline bool updateVoxel(const Color color_measured,
                                   const float measured_depth_m,
                                   const float voxel_depth_m,
                                   const float max_weight,
                                   const float truncation_distance_m,
                                   const WeightingFunction& weighting_function,
                                   ColorVoxel* voxel_ptr) {
  // Read CURRENT voxel values (from global GPU memory)
  const Color voxel_color_current = voxel_ptr->color;
  const float voxel_weight_current = voxel_ptr->weight;
  // Fuse
  const float measurement_weight = weighting_function(
      measured_depth_m, voxel_depth_m, truncation_distance_m);
  const Color fused_color =
      blendTwoColors(voxel_color_current, voxel_weight_current, color_measured,
                     measurement_weight);
  const float weight =
      fmin(measurement_weight + voxel_weight_current, max_weight);
  // Write NEW voxel values (to global GPU memory)
  voxel_ptr->color = fused_color;
  voxel_ptr->weight = weight;
  return true;
}

// 核函数 integrateBlocks
__global__ void integrateBlocks(
    const Index3D* block_indices_device_ptr, const Camera camera,
    const Color* color_image, const int color_rows, const int color_cols,
    const float* depth_image, const int depth_rows, const int depth_cols,
    const Transform T_C_L, const float block_size,
    const float truncation_distance_m, const float max_weight,
    const float max_integration_distance, const int depth_subsample_factor,
    const WeightingFunction weighting_function,
    ColorBlock** block_device_ptrs) {
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  // 获取像素平面上的投影像素坐标，体素深度，体素中心点
  if (!projectThreadVoxel<Camera>(block_indices_device_ptr, camera, T_C_L,
                                  block_size, &u_px, &voxel_depth_m,
                                  &p_voxel_center_C)) {
    return;
  }

  // If voxel further away than the limit, skip this voxel
  if (max_integration_distance > 0.0f) {
    if (voxel_depth_m > max_integration_distance) {
      return;
    }
  }

  const Eigen::Vector2f u_px_depth =
      u_px / static_cast<float>(depth_subsample_factor);
  float surface_depth_m;
  if (!interpolation::interpolate2DLinear<float>(
          depth_image, u_px_depth, depth_rows, depth_cols, &surface_depth_m)) {
    return;
  }

  // Occlusion testing
  // Get the distance of the voxel from the rendered surface. If outside
  // truncation band, skip.
  const float voxel_distance_from_surface = surface_depth_m - voxel_depth_m;
  if (fabsf(voxel_distance_from_surface) > truncation_distance_m) {
    return;
  }

  Color image_value;
  if (!interpolation::interpolate2DLinear<
          Color, interpolation::checkers::ColorPixelAlphaGreaterThanZero>(
          color_image, u_px, color_rows, color_cols, &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order such
  // that adjacent threads (x-major) access adjacent memory locations in the
  // block (z-major). 获取在该thread需要更新的voxel
  ColorVoxel* voxel_ptr =
      &(block_device_ptrs[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the voxel using the update rule for this layer type 更新voxel
  updateVoxel(image_value, surface_depth_m, voxel_depth_m, max_weight,
              truncation_distance_m, weighting_function, voxel_ptr);
}

// 更新Blocks，block拥有8*8*8的voxel
void ProjectiveColorIntegrator::updateBlocks(
    const std::vector<Index3D>& block_indices, const ColorImage& color_frame,
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    const float truncation_distance_m, ColorLayer* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);
  CHECK_EQ(color_frame.rows() % depth_frame.rows(), 0);
  CHECK_EQ(color_frame.cols() % depth_frame.cols(), 0);

  if (block_indices.empty()) {
    return;
  }
  const int num_blocks = block_indices.size(); // 有多少个block
  const int depth_subsampling_factor = color_frame.rows() / depth_frame.rows(); // 深度下采样系数？RGB / 深度
  CHECK_EQ(color_frame.cols() / depth_frame.cols(), depth_subsampling_factor);

  // Expand the buffers when needed 如果block数大于block_indices_device_则扩容
  if (num_blocks > block_indices_device_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks); // 扩充为num_blocks的1.5倍
    block_indices_device_.reserve(new_size);
    block_ptrs_device_.reserve(new_size);
    block_indices_host_.reserve(new_size);
    block_ptrs_host_.reserve(new_size);
  }

  // Stage on the host pinned memory; pinned memory指的是固定内存，这是一种可以被主机和设备直接访问的特殊内存
  block_indices_host_ = block_indices;
  block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, layer_ptr); // 通过索引获取block，得到主机上的block的指针

  // Transfer to the device
  block_indices_device_ = block_indices_host_; // 在pinned memory里，通过直接赋值实现复制到GPU上？理由是实现了相关类并重载了 = 
  block_ptrs_device_ = block_ptrs_host_;

  // We need the inverse transform in the kernel
  const Transform T_C_L = T_L_C.inverse(); // 需要取逆

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;// Block每条边有多少个体素 8个
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);// 线程数是多维形式 8*8*8 ，刚好一个线程对应一个voxel
  const int num_thread_blocks = block_indices.size(); //线程块数目就是block数，刚好匹配上了
  // clang-format off 调用kernel函数
  integrateBlocks<<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      block_indices_device_.data(),
      camera,
      color_frame.dataConstPtr(),
      color_frame.rows(),
      color_frame.cols(),
      depth_frame.dataConstPtr(),
      depth_frame.rows(),
      depth_frame.cols(),
      T_C_L,
      layer_ptr->block_size(),
      truncation_distance_m,
      max_weight_,
      max_integration_distance_m_,
      depth_subsampling_factor,
      WeightingFunction(weighting_function_type_),
      block_ptrs_device_.data());
  // clang-format on
  cudaStreamSynchronize(integration_stream_);
  checkCudaErrors(cudaPeekAtLastError());
}

__global__ void checkBlocksInTruncationBand(
    const VoxelBlock<TsdfVoxel>** block_device_ptrs,
    const float truncation_distance_m,
    bool* contains_truncation_band_device_ptr) {
  // A single thread in each block initializes the output to 0
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    contains_truncation_band_device_ptr[blockIdx.x] = 0;
  }
  __syncthreads();

  // Get the Voxel we'll check in this thread
  const TsdfVoxel voxel = block_device_ptrs[blockIdx.x]
                              ->voxels[threadIdx.z][threadIdx.y][threadIdx.x];

  // If this voxel in the truncation band, write the flag to say that the block
  // should be processed.
  // NOTE(alexmillane): There will be collision on write here. However, from my
  // reading, all threads' writes will result in a single write to global
  // memory. Because we only write a single value (1) it doesn't matter which
  // thread "wins".
  if (std::abs(voxel.distance) <= truncation_distance_m) {
    contains_truncation_band_device_ptr[blockIdx.x] = true;
  }
}

std::vector<Index3D>
ProjectiveColorIntegrator::reduceBlocksToThoseInTruncationBand(
    const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
    const float truncation_distance_m) {
  // Check 1) Are the blocks allocated
  // - performed on the CPU because the hash-map is on the CPU
  std::vector<Index3D> block_indices_check_1;
  block_indices_check_1.reserve(block_indices.size());
  for (const Index3D& block_idx : block_indices) {
    if (tsdf_layer.isBlockAllocated(block_idx)) {
      block_indices_check_1.push_back(block_idx);
    }
  }

  if (block_indices_check_1.empty()) {
    return block_indices_check_1;
  }

  // Check 2) Does each of the blocks have a voxel within the truncation band
  // - performed on the GPU because the blocks are there
  // Get the blocks we need to check
  std::vector<const TsdfBlock*> block_ptrs =
      getBlockPtrsFromIndices(block_indices_check_1, tsdf_layer);

  const int num_blocks = block_ptrs.size();

  // Expand the buffers when needed
  if (num_blocks > truncation_band_block_ptrs_device_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    truncation_band_block_ptrs_host_.reserve(new_size);
    truncation_band_block_ptrs_device_.reserve(new_size);
    block_in_truncation_band_device_.reserve(new_size);
    block_in_truncation_band_host_.reserve(new_size);
  }

  // Host -> Device
  truncation_band_block_ptrs_host_ = block_ptrs;
  truncation_band_block_ptrs_device_ = truncation_band_block_ptrs_host_;

  // Prepare output space
  block_in_truncation_band_device_.resize(num_blocks);

  // Do the check on GPU
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;
  // clang-format off
  checkBlocksInTruncationBand<<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      truncation_band_block_ptrs_device_.data(),
      truncation_distance_m,
      block_in_truncation_band_device_.data());
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(integration_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back
  block_in_truncation_band_host_ = block_in_truncation_band_device_;

  // Filter the indices using the result
  std::vector<Index3D> block_indices_check_2;
  block_indices_check_2.reserve(block_indices_check_1.size());
  for (int i = 0; i < block_indices_check_1.size(); i++) {
    if (block_in_truncation_band_host_[i] == true) {
      block_indices_check_2.push_back(block_indices_check_1[i]);
    }
  }

  return block_indices_check_2;
}

}  // namespace nvblox
