// main example to demonstrate usage of the API

#include "clip.cpp/clip.h"
#include "clip_instance/clip_instance.h"
#include "clip_instance/common-clip.h"
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
namespace fs = std::filesystem;
int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    app_params params;
    if (!app_params_parse(argc, argv, params, 1, 1)) {
        print_help(argc, argv, params, 1, 1);
        return 1;
    }

    // test image load
    int64_t t_load_us = ggml_time_us();
    clip_image_u8 img0;
    clip_image_load_from_file(params.image_paths[0].c_str(), &img0);
    int64_t t_image_load_us = ggml_time_us();
    printf("Image loaded in %8.2f ms\n",  (t_image_load_us - t_load_us) / 1000.0);
    // std::flush(std::cout);

    std::string file("/home/catkin_ws/projects/clip_sam_ws/SamToClip/config/clip.yaml");
    SamToClip::ClipInstance clip_instance(file);

    clip_image_u8 img1;
    cv::Mat cv_img = cv::imread(params.image_paths[0]);

    t_load_us = ggml_time_us();
    clip_instance.to_clip_image_u8(img1, cv_img);
    t_image_load_us = ggml_time_us();
    printf("Image loaded in %8.2f ms\n",(t_image_load_us - t_load_us) / 1000.0);
    // std::cout << "channels: " << cv::imread(params.image_paths[0]).channels() << std::endl;
    int cnt = 0;
    for (int i = 0; i < img0.size; i++) {
        if (img0.data[i] != img1.data[i]) {
            cnt++;
            std::cout << "deferent: " << i << std::endl;
            std::cout << std::to_string(img0.data[i])<< std::endl;
            std::cout << std::to_string(img1.data[i])<< std::endl;
            std::cout << "data not equal " << cnt << std::endl;
        }
    }

    return 0;
}
