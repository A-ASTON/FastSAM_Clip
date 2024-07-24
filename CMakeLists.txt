project(sam_to_clip)
cmake_minimum_required(VERSION 3.12)

set(BUILD_TEST ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
# CUDA
# set(CMAKE_BUILD_TYPE Debug)
# 由于find_package找到的不完全，故仍需手动set
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_LIBRARIES /usr/local/cuda/lib64)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")
# OpenCV
find_package(OpenCV REQUIRED)
# message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /home/tools/TensorRT-8.4.1.5/include)
set(TensorRT_LIBRARIES /home/tools/TensorRT-8.4.1.5/lib)


message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")


add_subdirectory(clip.cpp)
add_subdirectory(yaml-cpp)


# fastsam_instance
set(fastsam_INCLUDE_DIR)
list(APPEND fastsam_INCLUDE_DIR
${CUDA_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
${TensorRT_INCLUDE_DIRS}
)

list(APPEND ALL_LIBS
${CUDA_LIBRARIES}
${OpenCV_LIBRARIES}
${TensorRT_LIBRARIES}
)


add_library(fastsam_instance
src/fastsam_instance/fastsam_instance.cpp
src/fastsam_instance/fastsam.cpp)

target_include_directories(fastsam_instance PUBLIC ${fastsam_INCLUDE_DIR} include .)
target_link_directories(fastsam_instance PUBLIC ${ALL_LIBS}) # 查找库文件时应该查找的目录，库搜索路径
target_link_libraries(fastsam_instance nvinfer nvinfer_plugin cudart ${OpenCV_LIBS})

# clip_instance
add_library(clip_instance 
src/clip_instance/clip_instance.cpp
src/clip_instance/common-clip.cpp)
target_include_directories(clip_instance PUBLIC include . yaml-cpp/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(clip_instance clip ggml yaml-cpp ${OpenCV_LIBRARIES})

# build the test program
if (BUILD_TEST)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/test)
add_executable(test_fastsam test/test_fastsam.cpp)
target_include_directories(test_fastsam PUBLIC ${fastsam_INCLUDE_DIR} include .)
target_link_libraries(test_fastsam fastsam_instance)

add_executable(test_clip test/test_clip.cpp)
target_include_directories(test_clip PUBLIC include .)
target_link_libraries(test_clip clip_instance)
endif (BUILD_TEST)

add_executable(sam_to_clip src/sam_to_clip.cpp)
# target_include_directories(SamToClip PUBLIC include clip)   
target_include_directories(sam_to_clip PRIVATE include .)
target_link_libraries(sam_to_clip clip ggml clip_instance fastsam_instance)