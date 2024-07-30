#include <gtest/gtest.h>
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/integrators/projective_semantic_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/interpolation/interpolation_3d.h"
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/core/color.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/tests/gpu_image_routines.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"
#include "nvblox/tests/projective_semantic_integrator_components.h"
#include"opencv2/opencv.hpp"

using namespace nvblox;

class SemanticIntegrationTest : public ::testing::Test {
 protected:
  SemanticIntegrationTest()
      : kSphereCenter(Vector3f(0.0f, 0.0f, 2.0f)),
        gt_layer_(voxel_size_m_, MemoryType::kUnified),
        camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {
    // Maximum distance to consider for scene generation.
    constexpr float kMaxDist = 10.0;
    constexpr float kMinWeight = 1.0;

    // Tolerance for error.
    constexpr float kDistanceErrorTolerance = truncation_distance_m_;

    // Scene is bounded to -5, -5, 0 to 5, 5, 5.
    scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, 0.0f),
                                          Vector3f(5.0f, 5.0f, 5.0f));
    // Create a scene with a ground plane and a sphere.
    scene_.addGroundLevel(0.0f);
    scene_.addCeiling(5.0f);
    scene_.addPrimitive(
        std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 2.0f), 2.0f));
    // Add bounding planes at 5 meters. Basically makes it sphere in a box.
    scene_.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);

    // Get the ground truth SDF for it.
    scene_.generateLayerFromScene(truncation_distance_m_, &gt_layer_);

    semantic_config_ = test_utils::getSemanticConfig();
  }

  // Scenes
  constexpr static float kSphereRadius = 2.0f;
  const Vector3f kSphereCenter;

  // Test layer
  constexpr static float voxel_size_m_ = 0.05;
  constexpr static float block_size_m_ =
      VoxelBlock<TsdfVoxel>::kVoxelsPerSide * voxel_size_m_;
  TsdfLayer gt_layer_;

  // Truncation distance
  constexpr static float truncation_distance_vox_ = 4;
  constexpr static float truncation_distance_m_ =
      truncation_distance_vox_ * voxel_size_m_;

  // Test camera 测试用相机
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;

  // Test Scene
  primitives::Scene scene_;

  // semantic config
  ProjectiveSemanticIntegrator::SemanticConfig semantic_config_;
};

class TestProjectiveSemanticIntegratorGPU : public ProjectiveSemanticIntegrator {
 public:
  TestProjectiveSemanticIntegratorGPU() : ProjectiveSemanticIntegrator() {}
  FRIEND_TEST(SemanticIntegrationTest, TruncationBandTest);
};


ColorImage generateSolidColorImage(const Color& color, const int height,
                                   const int width) {
  // Generate a random color for this scene
  ColorImage image(height, width);
  nvblox::test_utils::setImageConstantOnGpu(color, &image); // 辅助测试的函数
  return image;
}


bool colorsEqualIgnoreAlpha(const Color& color_1, const Color& color_2) {
  return (color_1.r == color_2.r) && (color_1.g == color_2.g) &&
         (color_1.b == color_2.b);
}

std::vector<Eigen::Vector3f> getPointsOnASphere(const float radius,
                                                const Eigen::Vector3f& center,
                                                const int points_per_rad = 10) {
  std::vector<Eigen::Vector3f> sphere_points;
  for (int azimuth_idx = 0; azimuth_idx < 2 * points_per_rad; azimuth_idx++) {
    for (int elevation_idx = 0; elevation_idx < points_per_rad;
         elevation_idx++) {
      const float azimuth = azimuth_idx * M_PI / points_per_rad - M_PI;
      const float elevation =
          elevation_idx * M_PI / points_per_rad - M_PI / 2.0f;
      Eigen::Vector3f p =
          radius * Eigen::Vector3f(cos(azimuth) * sin(elevation),
                                   sin(azimuth) * sin(elevation),
                                   cos(elevation));
      p += center;
      sphere_points.push_back(p);
    }
  }
  return sphere_points;
}

// 检查球面的颜色
float checkSphereColor(const SemanticLayer& semantic_layer, const Vector3f& center,
                       const float radius, const Color& color) {
  // Check that each sphere is colored appropriately (if observed)
  int num_observed = 0;
  int num_tested = 0;
  auto check_color = [&num_tested, &num_observed](
                         const SemanticVoxel& voxel,
                         const Color& color_2) -> void {
    ++num_tested;
    constexpr float kMinVoxelWeight = 1e-3;
    if (voxel.weight >= kMinVoxelWeight) {
      EXPECT_TRUE(colorsEqualIgnoreAlpha(voxel.color, color_2));
      ++num_observed;
    }
  };

  const std::vector<Eigen::Vector3f> sphere_points =
      getPointsOnASphere(radius, center);
  for (const Vector3f p : sphere_points) {
    const SemanticVoxel* color_voxel;
    EXPECT_TRUE(getVoxelAtPosition<SemanticVoxel>(semantic_layer, p, &color_voxel));
    check_color(*color_voxel, color);
  }

  const float ratio_observed_points =
      static_cast<float>(num_observed) / static_cast<float>(num_tested);
  return ratio_observed_points;
}

TEST_F(SemanticIntegrationTest, Color2LabelTest) {
    ProjectiveSemanticIntegrator integrator(semantic_config_);

    int height = 480;
    int weight = 640;

    // std::cout << toString(image.memory_type()) << std::endl;
    const auto color_1 = Color(255, 20, 127, 255);
    const auto color_2 = Color(92,136,89);
    ColorImage semantic_frame = generateSolidColorImage(color_2, height, weight); // ColorImage是在GPU上的，怎么可以直接通过赋值获取GPU上的数据呢

    cv::Mat image = cv::imread("/home/catkin_ws/segmentation/frame1000.jpg", 1);
    cv::Mat rgba_image;
    // 如果图片是 BGR 或 BGRA 格式，可以直接使用 cv::cvtColor，无法知道是RGB还是BGR，只能通过上游输出确定
    if (image.channels() == 3) {
        cv::cvtColor(image, rgba_image, cv::COLOR_RGB2RGBA);
        std::cout << "type: RGB" << std::endl;
    } else if (image.channels() == 4) {
        cv::cvtColor(image, rgba_image, cv::COLOR_BGRA2RGBA);
    } else {
        // 如果图片已经是 RGBA 格式，可以直接复制
        rgba_image = image.clone();
    }
    std::cout << "type: " << (rgba_image.type() == CV_8UC4) << std::endl;
    // cv::imshow("semantic",image);
    // cv::waitKey();

    ColorImage image_frame;
    const Color* temp_image = reinterpret_cast<const Color*>(rgba_image.ptr());
    (&image_frame)->populateFromBuffer(
    rgba_image.rows, rgba_image.cols,
    temp_image,
    MemoryType::kDevice); // 数据拷贝到GPU上，位于owned_data_


    std::cout << "before test_utils::readImageLabelOnGPU" << std::endl;
    test_utils::readImageLabelOnGPU(integrator.semantic_config_, color_2);

    // std::cout << "before test_utils::readColorFromLabel" << std::endl;
    // test_utils::readColorFromLabel(integrator.semantic_config_, 10);
    // 正确做法应该是动态获取label，一次查询而已，或者在GPU上获取所有颜色再转移到CPU上  
    
    std::cout << "test_utils::readLabelFromImage" << std::endl;
    test_utils::readLabelFromImage(integrator.semantic_config_, image_frame);
    
}

TEST_F(SemanticIntegrationTest, GetterAndSetter) {
  ProjectiveSemanticIntegrator semantic_integrator(semantic_config_);
  semantic_integrator.truncation_distance_vox(1); 
  EXPECT_EQ(semantic_integrator.truncation_distance_vox(), 1);
  semantic_integrator.sphere_tracing_ray_subsampling_factor(2);
  EXPECT_EQ(semantic_integrator.sphere_tracing_ray_subsampling_factor(), 2);
  semantic_integrator.max_weight(3.0f);
  EXPECT_EQ(semantic_integrator.max_weight(), 3.0f);
  semantic_integrator.max_integration_distance_m(4.0f);
  EXPECT_EQ(semantic_integrator.max_integration_distance_m(), 4.0f);
  semantic_integrator.weighting_function_type(
      WeightingFunctionType::kInverseSquareWeight);
  EXPECT_EQ(semantic_integrator.weighting_function_type(),
            WeightingFunctionType::kInverseSquareWeight);
}


TEST_F(SemanticIntegrationTest, TruncationBandTest) {
  // Check the GPU version against a hand-rolled CPU implementation.
  TestProjectiveSemanticIntegratorGPU integrator;

  // The distance from the surface that we "pass" blocks within.
  constexpr float kTestDistance = voxel_size_m_;

  std::vector<Index3D> all_indices = gt_layer_.getAllBlockIndices(); // 由于测试用例类为ColorIntegrationTest
  // 所以在这些测试宏中均能访问得到

  std::vector<Index3D> valid_indices =
      integrator.reduceBlocksToThoseInTruncationBand(all_indices, gt_layer_,
                                                     kTestDistance);

  // Horrible N^2 complexity set_difference implementation. But easy to write :)
  std::vector<Index3D> not_valid_indices;
  for (const Index3D& idx : all_indices) {
    if (std::find(valid_indices.begin(), valid_indices.end(), idx) ==
        valid_indices.end()) {
      not_valid_indices.push_back(idx);
    }
  }

  // Check indices touching band
  for (const Index3D& idx : valid_indices) {
    const auto block_ptr = gt_layer_.getBlockAtIndex(idx);
    bool touches_band = false;
    auto touches_band_lambda = [&touches_band, kTestDistance](
                                   const Index3D& voxel_index,
                                   const TsdfVoxel* voxel) -> void {
      if (std::abs(voxel->distance) <= kTestDistance) {
        touches_band = true;
      }
    };
    callFunctionOnAllVoxels<TsdfVoxel>(*block_ptr, touches_band_lambda);
    EXPECT_TRUE(touches_band);
  }

  // Check indices NOT touching band
  for (const Index3D& idx : not_valid_indices) {
    const auto block_ptr = gt_layer_.getBlockAtIndex(idx);
    bool touches_band = false;
    auto touches_band_lambda = [&touches_band, kTestDistance](
                                   const Index3D& voxel_index,
                                   const TsdfVoxel* voxel) -> void {
      if (std::abs(voxel->distance) <= kTestDistance) {
        touches_band = true;
      }
    };
    callFunctionOnAllVoxels<TsdfVoxel>(*block_ptr, touches_band_lambda);
    EXPECT_FALSE(touches_band);
  }
}

// TEST_F(SemanticIntegrationTest, IntegrateColorToGroundTruthDistanceField) {
//   // Create an integrator.
//   ProjectiveSemanticIntegrator color_integrator(semantic_config_);

//   // Simulate a trajectory of the requisite amount of points, on the circle
//   // around the sphere.
//   constexpr float kTrajectoryRadius = 4.0f; // 轨迹半径
//   constexpr float kTrajectoryHeight = 2.0f; // 轨迹高度
//   constexpr int kNumTrajectoryPoints = 80; // 轨迹点数目
//   const float radians_increment = 2 * M_PI / (kNumTrajectoryPoints); 

//   // Color layer
//   SemanticLayer color_layer(voxel_size_m_, MemoryType::kDevice);// Layer

//   // Generate a random color for this scene
//   const Color color = Color::Red();
//   const ColorImage image = generateSolidColorImage(color, height_, width_); // 生成图片
  
//   // Set keeping track of which blocks were touched during the test
//   Index3DSet touched_blocks; // 测试过程中触碰到的block

//   // 遍历轨迹点
//   for (size_t i = 0; i < kNumTrajectoryPoints; i++) {
//     const float theta = radians_increment * i; // 偏移角度，实际上就是将2pi分为轨迹点数这么多份，然后得到每个轨迹点所对应的角度
//     // Convert polar to cartesian coordinates.
//     Vector3f cartesian_coordinates(kTrajectoryRadius * std::cos(theta), // (4*cos(2pi * (1/80)))
//                                    kTrajectoryRadius * std::sin(theta),
//                                    kTrajectoryHeight);
//     // The camera has its z axis pointing towards the origin.
//     Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
//     Eigen::Quaternionf rotation_theta(
//         Eigen::AngleAxisf(M_PI + theta, Vector3f::UnitZ()));

//     // Construct a transform from camera to scene with this.
//     Transform T_S_C = Transform::Identity();
//     T_S_C.prerotate(rotation_theta * rotation_base);
//     T_S_C.pretranslate(cartesian_coordinates);

//     // Generate an image with a single color
//     std::vector<Index3D> updated_blocks;
//     color_integrator.integrateFrame(image, T_S_C, camera_, gt_layer_,
//                                     &color_layer, &updated_blocks);
//     // Accumulate touched block indices
//     std::copy(updated_blocks.begin(), updated_blocks.end(),
//               std::inserter(touched_blocks, touched_blocks.end()));
//   }

//   // Create a host copy of the layer.
//   SemanticLayer color_layer_host(color_layer, MemoryType::kHost);

//   // Lambda that checks if voxels have the passed color (if they have weight >
//   // 0)
//   auto color_check_lambda = [&color](const Index3D& voxel_idx,
//                                      const SemanticVoxel* voxel) -> void {
//     if (voxel->weight > 0.0f) {
//       EXPECT_TRUE(colorsEqualIgnoreAlpha(voxel->color, color));
//     }
//   };

//   // Check that all touched blocks are the color we chose
//   for (const Index3D& block_idx : touched_blocks) {
//     callFunctionOnAllVoxels<SemanticVoxel>(
//         *color_layer_host.getBlockAtIndex(block_idx), color_check_lambda);
//   }

//   // Check that most points on the surface of the sphere have been observed
//   int num_points_on_sphere_surface_observed = 0;
//   const std::vector<Eigen::Vector3f> sphere_points =
//       getPointsOnASphere(kSphereRadius, kSphereCenter);
//   const int num_surface_points_tested = sphere_points.size();
//   for (const Vector3f p : sphere_points) {
//     const SemanticVoxel* color_voxel;
//     EXPECT_TRUE(
//         getVoxelAtPosition<SemanticVoxel>(color_layer_host, p, &color_voxel));
//     if (color_voxel->weight >= 1.0f) {
//       ++num_points_on_sphere_surface_observed;
//     }
//   }
//   const float ratio_observed_surface_points =
//       static_cast<float>(num_points_on_sphere_surface_observed) /
//       static_cast<float>(num_surface_points_tested);
//   std::cout << "num_points_on_sphere_surface_observed: "
//             << num_points_on_sphere_surface_observed << std::endl;
//   std::cout << "num_surface_points_tested: " << num_surface_points_tested
//             << std::endl;
//   std::cout << "ratio_observed_surface_points: "
//             << ratio_observed_surface_points << std::endl;
//   EXPECT_GT(ratio_observed_surface_points, 0.5);

//   // Check that all color blocks have a corresponding block in the tsdf layer
//   for (const Index3D block_idx : color_layer_host.getAllBlockIndices()) {
//     EXPECT_NE(gt_layer_.getBlockAtIndex(block_idx), nullptr);
//   }

//   // Generate a mesh from the "reconstruction"
//   MeshIntegrator mesh_integrator;
//   BlockLayer<MeshBlock> mesh_layer(block_size_m_, MemoryType::kDevice);
//   EXPECT_TRUE(
//       mesh_integrator.integrateMeshFromDistanceField(gt_layer_, &mesh_layer));
//   mesh_integrator.semanticMesh(color_layer, &mesh_layer);

//   // Write to file
//   if (FLAGS_nvblox_test_file_output) {
//     io::outputMeshLayerToPly(mesh_layer, "color_sphere_mesh.ply");
//   }
// }

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
