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
#include <cuda_runtime.h>

#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"
#include "nvblox/mesh/internal/impl/marching_cubes_table.h"
#include "nvblox/mesh/internal/marching_cubes.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/utils/timing.h"

namespace nvblox {
void MeshIntegrator::semanticMesh(const SemanticLayer& semantic_layer,
                               BlockLayer<MeshBlock>* mesh_layer) {
  semanticMesh(semantic_layer, mesh_layer->getAllBlockIndices(), mesh_layer);
}
void MeshIntegrator::semanticMesh(const SemanticLayer& semantic_layer,
                               const std::vector<Index3D>& block_indices,
                               BlockLayer<MeshBlock>* mesh_layer) {
  // Default choice is GPU
  semanticMeshGPU(semantic_layer, block_indices, mesh_layer);
}

//
/* Color Mesh blocks on the GPU
 *
 * Call with
 * - one ThreadBlock per VoxelBlock, GridDim 1D
 * - BlockDim 1D, any size: we implement a stridded access pattern over
 *   MeshBlock verticies
 *
 * @param: color_blocks:     a list of color blocks which correspond in position
 *                           to mesh_blocks
 * @param: block_indices:    a list of blocks indices.
 * @param: cuda_mesh_blocks: a list of mesh_blocks to be colored.
 */
__global__ void semanticMeshBlockByClosestSemanticVoxel(
    const SemanticBlock** semantic_blocks, const Index3D* block_indices,
    const float block_size, const float voxel_size,
    CudaMeshBlock* cuda_mesh_blocks) {
  // Block
  const SemanticBlock* semantic_block_ptr = semantic_blocks[blockIdx.x]; // 直接用了线程块ID，因为刚好线程块对应block，线程对应voxel
  const Index3D block_index = block_indices[blockIdx.x];
  CudaMeshBlock cuda_mesh_block = cuda_mesh_blocks[blockIdx.x];

  // The position of this block in the layer
  const Vector3f p_L_B_m = getPositionFromBlockIndex(block_size, block_index);

  // Interate through MeshBlock vertices - Stidded access pattern
  for (int i = threadIdx.x; i < cuda_mesh_block.vertices_size;
       i += blockDim.x) { //i用了当前线程id，然后通过+blockDim.x加上block大小，一个核函数负责处理一个block的多个voxel
    // The position of this vertex in the layer
    const Vector3f p_L_V_m = cuda_mesh_block.vertices[i];

    // The position of this vertex in the block
    const Vector3f p_B_V_m = p_L_V_m - p_L_B_m;

    // Convert this to a voxel index
    Index3D voxel_idx_in_block = (p_B_V_m.array() / voxel_size).cast<int>();

    // NOTE(alexmillane): Here we make some assumptions.
    // - We assume that the closest voxel to p_L_V is in the ColorBlock
    //   co-located with the MeshBlock from which p_L_V was drawn.
    // - This is will (very?) occationally be incorrect when mesh vertices
    //   escape block boundaries. However, making this assumption saves us any
    //   neighbour calculations.
    constexpr size_t KVoxelsPerSizeMinusOne =
        VoxelBlock<ColorVoxel>::kVoxelsPerSide - 1;
    voxel_idx_in_block =
        voxel_idx_in_block.array().min(KVoxelsPerSizeMinusOne).max(0);

    // Get the color voxel 获取颜色体素
    const SemanticVoxel semantic_voxel =
        semantic_block_ptr->voxels[voxel_idx_in_block.x()]  // NOLINT
                               [voxel_idx_in_block.y()]  // NOLINT
                               [voxel_idx_in_block.z()];

    // Write the color out to global memory 赋予颜色
    cuda_mesh_block.colors[i] = semantic_voxel.color;
  }
}

__global__ void semanticMeshBlocksConstant(Color color,
                                        CudaMeshBlock* cuda_mesh_blocks) {
  // Each threadBlock operates on a single MeshBlock
  CudaMeshBlock cuda_mesh_block = cuda_mesh_blocks[blockIdx.x];
  // Interate through MeshBlock vertices - Stidded access pattern
  for (int i = threadIdx.x; i < cuda_mesh_block.vertices_size;
       i += blockDim.x) {
    cuda_mesh_block.colors[i] = color;
  }
}

void semanticMeshBlocksConstantGPU(const std::vector<Index3D>& block_indices,
                                const Color& color, MeshLayer* mesh_layer,
                                cudaStream_t cuda_stream) {
  CHECK_NOTNULL(mesh_layer);
  if (block_indices.size() == 0) {
    return;
  }

  // Prepare CudaMeshBlocks, which are effectively containers of device pointers
  std::vector<CudaMeshBlock> cuda_mesh_blocks;
  cuda_mesh_blocks.resize(block_indices.size());
  for (int i = 0; i < block_indices.size(); i++) {
    cuda_mesh_blocks[i] =
        CudaMeshBlock(mesh_layer->getBlockAtIndex(block_indices[i]).get());
  }

  // Allocate
  CudaMeshBlock* cuda_mesh_block_device_ptrs;
  checkCudaErrors(cudaMalloc(&cuda_mesh_block_device_ptrs,
                             cuda_mesh_blocks.size() * sizeof(CudaMeshBlock)));

  // Host -> GPU
  checkCudaErrors(
      cudaMemcpyAsync(cuda_mesh_block_device_ptrs, cuda_mesh_blocks.data(),
                      cuda_mesh_blocks.size() * sizeof(CudaMeshBlock),
                      cudaMemcpyHostToDevice, cuda_stream));

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kThreadsPerBlock = 8 * 32;  // Chosen at random
  const int num_blocks = block_indices.size();
  semanticMeshBlocksConstant<<<num_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
      Color::Gray(),  // NOLINT
      cuda_mesh_block_device_ptrs);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
  checkCudaErrors(cudaPeekAtLastError());

  // Deallocate
  checkCudaErrors(cudaFree(cuda_mesh_block_device_ptrs));
}

// 通过此方法进行着色
void semanticMeshBlockByClosestSemanticVoxelGPU(
    const SemanticLayer& semantic_layer, const std::vector<Index3D>& block_indices,
    MeshLayer* mesh_layer, cudaStream_t cuda_stream) {
  CHECK_NOTNULL(mesh_layer);
  if (block_indices.size() == 0) {
    return;
  }

  // Get the locations (on device) of the color blocks
  // NOTE(alexmillane): This function assumes that all block_indices have been
  // checked to exist in color_layer.
  std::vector<const SemanticBlock*> semantic_blocks =
      getBlockPtrsFromIndices(block_indices, semantic_layer);

  // Prepare CudaMeshBlocks, which are effectively containers of device pointers 包含设备指针的容器
  std::vector<CudaMeshBlock> cuda_mesh_blocks;
  cuda_mesh_blocks.resize(block_indices.size());
  for (int i = 0; i < block_indices.size(); i++) {
    cuda_mesh_blocks[i] =
        CudaMeshBlock(mesh_layer->getBlockAtIndex(block_indices[i]).get());
  }

  // Allocate 在GPU上分配空间 实际上是将指向设备内存的指针存放在host端，然后现在在设备的分配内存，将这批指针转移到device端
  const SemanticBlock** semantic_block_device_ptrs;
  checkCudaErrors(cudaMalloc(&semantic_block_device_ptrs,
                             semantic_blocks.size() * sizeof(ColorBlock*)));
  Index3D* block_indices_device_ptr;
  checkCudaErrors(cudaMalloc(&block_indices_device_ptr,
                             block_indices.size() * sizeof(Index3D)));
  CudaMeshBlock* cuda_mesh_block_device_ptrs; // 该指针指向设备内存（全局内存）
  checkCudaErrors(cudaMalloc(&cuda_mesh_block_device_ptrs,
                             cuda_mesh_blocks.size() * sizeof(CudaMeshBlock)));

  // Host -> GPU transfers
  checkCudaErrors(cudaMemcpyAsync(semantic_block_device_ptrs, semantic_blocks.data(),
                                  semantic_blocks.size() * sizeof(ColorBlock*),
                                  cudaMemcpyHostToDevice, cuda_stream));
  checkCudaErrors(cudaMemcpyAsync(block_indices_device_ptr,
                                  block_indices.data(),
                                  block_indices.size() * sizeof(Index3D),
                                  cudaMemcpyHostToDevice, cuda_stream));
  checkCudaErrors(
      cudaMemcpyAsync(cuda_mesh_block_device_ptrs, cuda_mesh_blocks.data(),
                      cuda_mesh_blocks.size() * sizeof(CudaMeshBlock),
                      cudaMemcpyHostToDevice, cuda_stream));

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kThreadsPerBlock = 8 * 32;  // Chosen at random
  const int num_blocks = block_indices.size();
  const float voxel_size =
      mesh_layer->block_size() / VoxelBlock<TsdfVoxel>::kVoxelsPerSide;
  semanticMeshBlockByClosestSemanticVoxel<<<num_blocks, kThreadsPerBlock, 0,
                                      cuda_stream>>>(
      semantic_block_device_ptrs,   // NOLINT
      block_indices_device_ptr,  // NOLINT
      mesh_layer->block_size(),  // NOLINT
      voxel_size,                // NOLINT
      cuda_mesh_block_device_ptrs);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
  checkCudaErrors(cudaPeekAtLastError());

  // Deallocate
  checkCudaErrors(cudaFree(semantic_block_device_ptrs));
  checkCudaErrors(cudaFree(block_indices_device_ptr));
  checkCudaErrors(cudaFree(cuda_mesh_block_device_ptrs));
}

void MeshIntegrator::semanticMeshGPU(const SemanticLayer& semantic_layer,
                                  MeshLayer* mesh_layer) {
  semanticMeshGPU(semantic_layer, mesh_layer->getAllBlockIndices(), mesh_layer);
}

// mapper中调用的方法，视作color mesh的入口，这是我们需要修改的，接受SemanticLayer作为输入
void MeshIntegrator::semanticMeshGPU(
    const SemanticLayer& semantic_layer,
    const std::vector<Index3D>& requested_block_indices,
    MeshLayer* mesh_layer) {
  CHECK_NOTNULL(mesh_layer);
  CHECK_EQ(semantic_layer.block_size(), mesh_layer->block_size());

  // NOTE(alexmillane): Generally, some of the MeshBlocks which we are
  // "coloring" will not have data in the color layer. HOWEVER, for colored
  // MeshBlocks (ie with non-empty color members), the size of the colors must
  // match vertices. Therefore we "color" all requested block_indices in two
  // parts:
  // - The first part using the color layer, and
  // - the second part a constant color.

  // Check for each index, that the MeshBlock exists, and if it does
  // allocate space for color.
  std::vector<Index3D> block_indices;
  block_indices.reserve(requested_block_indices.size());
  std::for_each(
      requested_block_indices.begin(), requested_block_indices.end(),
      [&mesh_layer, &block_indices](const Index3D& block_idx) {
        if (mesh_layer->isBlockAllocated(block_idx)) {
          mesh_layer->getBlockAtIndex(block_idx)->expandColorsToMatchVertices();
          block_indices.push_back(block_idx);
        }
      });

  // Split block indices into two groups, one group containing indices with
  // corresponding ColorBlocks, and one without.
  std::vector<Index3D> block_indices_in_semantic_layer;
  std::vector<Index3D> block_indices_not_in_semantic_layer;
  block_indices_in_semantic_layer.reserve(block_indices.size());
  block_indices_not_in_semantic_layer.reserve(block_indices.size());
  // 通过isBlockAllocated判断指定的semantic_layer是否分配了该block，将block_indices分为两部分，一部分根据semantic layer着色，另一部分是默认颜色
  for (const Index3D& block_idx : block_indices) {
    if (semantic_layer.isBlockAllocated(block_idx)) {
      block_indices_in_semantic_layer.push_back(block_idx);
    } else {
      block_indices_not_in_semantic_layer.push_back(block_idx);
    }
  }

  // Color 两部分着色，一部分是对应的color voxel有值的，另一部分用的constant color
  semanticMeshBlockByClosestSemanticVoxelGPU(
      semantic_layer, block_indices_in_semantic_layer, mesh_layer, cuda_stream_);
  semanticMeshBlocksConstantGPU(block_indices_not_in_semantic_layer,
                             default_mesh_color_, mesh_layer, cuda_stream_);
}

void MeshIntegrator::semanticMeshCPU(const SemanticLayer& semantic_layer,
                                  BlockLayer<MeshBlock>* mesh_layer) {
  semanticMeshCPU(semantic_layer, mesh_layer->getAllBlockIndices(), mesh_layer);
}

void MeshIntegrator::semanticMeshCPU(const SemanticLayer& semantic_layer,
                                  const std::vector<Index3D>& block_indices,
                                  BlockLayer<MeshBlock>* mesh_layer) {
  // For each vertex just grab the closest color
  for (const Index3D& block_idx : block_indices) {
    MeshBlock::Ptr block = mesh_layer->getBlockAtIndex(block_idx);
    if (block == nullptr) {
      continue;
    }
    block->colors.resize(block->vertices.size());
    for (int i = 0; i < block->vertices.size(); i++) {
      const Vector3f& vertex = block->vertices[i];
      const SemanticVoxel* semantic_voxel;
      if (getVoxelAtPosition<SemanticVoxel>(semantic_layer, vertex, &semantic_voxel)) {
        block->colors[i] = semantic_voxel->color;
      } else {
        block->colors[i] = Color::Gray();
      }
    }
  }
}

}  // namespace nvblox