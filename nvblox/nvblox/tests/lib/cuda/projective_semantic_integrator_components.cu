#include "nvblox/tests/projective_semantic_integrator_components.h"
#include "nvblox/core/types.h"
#include "nvblox/core/color_hash.cuh"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/interpolation/interpolation_2d.h"
namespace nvblox {
namespace test_utils {
// 提供一个CPP用的接口，然后再在这里实现GPU上的查询
__global__ void readImageLabelOnGPU(
    HashableColor* color_map_device,
    int color_map_size,
    const HashableColor color, SemanticLabel* semantic_label);

__global__ void readColorFromLabel(
    HashableColor* color_map_device,
    int color_map_size,
    const SemanticLabel semantic_label, HashableColor* color);

__global__ void readLabelFromImage(
    HashableColor* color_map_device,
    int color_map_size,
    const Color* frame, const int rows,
    const int cols);

ProjectiveSemanticIntegrator::SemanticConfig getSemanticConfig() {
    ProjectiveSemanticIntegrator::SemanticConfig semantic_config;
    std::string path = "/home/catkin_ws/src/nvblox_ros1/nvblox_ros/cfg/tesse_multiscene_archviz1_segmentation_mapping.csv";
    // Get semantic meas prob
    double semantic_measurement_probability =
        semantic_config.semantic_measurement_probability_;
    
    semantic_config.semantic_measurement_probability_ =
        static_cast<SemanticProbability>(semantic_measurement_probability);

    semantic_config.filename = path;

    std::vector<int> dynamic_labels = {133};
    semantic_config.dynamic_labels_.clear();
    for (const auto& label : dynamic_labels) {
        semantic_config.dynamic_labels_.push_back(label);
    }

    return semantic_config;
}


__global__ void readImageLabelOnGPU(
    HashableColor* color_map_device,
    int color_map_size,
    const HashableColor color, SemanticLabel* semantic_label) {

    if(!getSemanticLabelFromColor(
        color_map_device,
        color_map_size,
        color,
        semantic_label)) {
        return;
    }
    // std::cout << semantic_label + '0' << std::endl;
}

void readImageLabelOnGPU(
    const ProjectiveSemanticIntegrator::SemanticConfig& semantic_config,
    const Color& color) {
    HashableColor c(color);
    std::shared_ptr<SemanticLabel2Color> semantic_color_tool = semantic_config.semantic_color_tool_;
    // 调用

    SemanticLabel* label;
    SemanticLabel* local = (SemanticLabel*)malloc(sizeof(SemanticLabel));
    checkCudaErrors(cudaMalloc(&label, sizeof(SemanticLabel)));

    for (auto color : semantic_color_tool->color_map_host_) {
        readImageLabelOnGPU<<<1, 1>>>(semantic_color_tool->color_map_device_,
                                    semantic_color_tool->color_map_size_, color,label);
        checkCudaErrors(cudaMemcpy(local, label, sizeof(SemanticLabel), cudaMemcpyDeviceToHost));
    
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // std::cout<< "label:" << *local<< std::endl;
        printf("read label: %d from color r:%d, g:%d, b:%d\n",*local, color.r, color.g, color.b);
    }
    
    
    cudaFree(label);
    free(local);
}

__global__ void readColorFromLabel(
    HashableColor* color_map_device,
    int color_map_size,
    const SemanticLabel semantic_label, HashableColor* color) {
    if(!getColorFromSemanticLabel(
        color_map_device,
        color_map_size,
        semantic_label,
        color)) {
        return;
    }       
}



void readColorFromLabel(
    const ProjectiveSemanticIntegrator::SemanticConfig& semantic_config,
    const SemanticLabel& label) {
    
    std::shared_ptr<SemanticLabel2Color> semantic_color_tool = semantic_config.semantic_color_tool_;
    // 调用

    HashableColor* color;
    HashableColor* local = (HashableColor*)malloc(sizeof(HashableColor));

    checkCudaErrors(cudaMalloc(&color, sizeof(HashableColor)));

    for (int index = 0; index < semantic_color_tool->color_map_size_; index++) {
        readColorFromLabel<<<1, 1>>>(semantic_color_tool->color_map_device_, semantic_color_tool->color_map_size_,index, color);

        checkCudaErrors(cudaMemcpy(local, color, sizeof(HashableColor), cudaMemcpyDeviceToHost));
        
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // std::cout<< "label:" << *local<< std::endl;
        printf("read color from id: %d\n",index);
        printf("Color r:%d, g:%d, b:%d\n", local->r, local->g, local->b);
    }
    
    cudaFree(color);
    free(local);
}

__global__ void readLabelFromImage(
    HashableColor* color_map_device,
    int color_map_size,
    const Color* frame, const int rows,
    const int cols) {
    Eigen::Vector2f u_px(201.174438, 116.939835);
    Color image_value;

    if (!interpolation::interpolate2DClosest<
            Color, interpolation::checkers::ColorPixelAlphaGreaterThanZero>(
            frame, u_px, rows, cols, &image_value)) {
        return;
    }
    // if (!interpolation::interpolate2DLinear<
    //       Color, interpolation::checkers::ColorPixelAlphaGreaterThanZero>(
    //       frame.dataConstPtr(), u_px, frame.rows(), frame.cols(), &image_value)) {
    //     return;
    // }   

    printf("Read Color r:%d, g:%d, b:%d on u_px:(%f, %f)\n", image_value.r, image_value.g, image_value.b, u_px.x(), u_px.y());
    SemanticLabel label;
    if(!getSemanticLabelFromColor(
        color_map_device,
        color_map_size,
        HashableColor(image_value),
        &label)) {
        printf("No label found");
        return;
    }
    
    printf("Get Label:%d\n", label);
}

void readLabelFromImage(
    const ProjectiveSemanticIntegrator::SemanticConfig& semantic_config,
    const ColorImage& frame) {
    // 通过该函数调用核函数，实现读取指定pixel的label
    // 最近邻插值查询Color
    std::shared_ptr<SemanticLabel2Color> semantic_color_tool = semantic_config.semantic_color_tool_;
    readLabelFromImage<<<1,1>>>(semantic_color_tool->color_map_device_, semantic_color_tool->color_map_size_,
    frame.dataConstPtr(), frame.rows(), frame.cols());


    // SemanticLabels semantic_labels(semantic_frame.rows() * semantic_frame.cols());
    // for (int y = 0; y < semantic_frame.rows(); y++) {
    //     for (int x = 0; x < semantic_frame.cols(); x++) {
    //     Color color = image::access(y, x, semantic_frame.cols(), semantic_frame.dataConstPtr());
    //     semantic_labels[y * semantic_frame.cols() + x] = semantic_config.semantic_color_tool_->getSemanticLabelFromColor(
    //         HashableColor(color.r, color.g, color.b, 255u));
    //     std::cout << semantic_labels[y * semantic_frame.cols() + x] << std::endl;
    //     }
    // }
}


}
}