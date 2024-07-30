// main example to demonstrate usage of the API

#include "clip.cpp/clip.h"
#include "clip_instance/clip_instance.h"
#include "clip_instance/common-clip.h"
#include "common.h"
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

    const int64_t t_load_us = ggml_time_us();

    auto ctx = clip_model_load(params.model.c_str(), params.verbose); // 初始化ctx，已有
    if (!ctx) {
        printf("%s: Unable  to load model from %s", __func__, params.model.c_str());
        return 1;
    }

    const int64_t t_image_load_us = ggml_time_us();

    fs::path folder_path("/home/catkin_ws/projects/clip_sam_ws/SamToClip/seg_output/");

    std::vector<std::pair<std::string, float>> score_map;
    const char * text = params.texts[0].c_str(); // 文本内容
    // 遍历文件夹中的所有文件
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        fs::path file_path = entry.path();
        if (!entry.is_directory()) { // 确保只处理文件，而不是子目录
            
            // 检查文件是否为图像文件
            if (file_path.extension() == ".png" || file_path.extension() == ".jpg" ||
                file_path.extension() == ".jpeg" || file_path.extension() == ".bmp") {
                std::string strPath = file_path.string();
                const char * img_path = strPath.c_str();
                
                clip_image_u8 img0;
                if (!clip_image_load_from_file(img_path, &img0)) {
                    fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path);
                    return 1;
                }

                const int64_t t_similarity_score = ggml_time_us(); // 获取相似度的起始时间

                
                float* vec = new float[768](); // 得分
                clip_image_f32 img1;
                clip_image_preprocess(ctx, &img0, &img1);
                clip_image_encode(ctx, params.n_threads, &img1, vec, true);
                // if (!clip_compare_text_and_image(ctx, params.n_threads, text, &img0, &score)) {
                //     printf("Unable to compare text and image\n");
                //     clip_free(ctx); // 释放ctx
                //     return 1;
                // }
                // score_map.push_back({strPath, score});

                // const int64_t t_main_end_us = ggml_time_us(); // 获取结束时间

                // std::cout << img_path << std::endl;
                // printf("%s: Similarity score = %8.2f\n", __func__, score);
                // if (params.verbose >= 1) {
                //     printf("\n\nTimings\n");
                //     printf("%s: Model loaded in %8.2f ms\n", __func__, (t_image_load_us - t_load_us) / 1000.0);
                //     printf("%s: Image loaded in %8.2f ms\n", __func__, (t_similarity_score - t_image_load_us) / 1000.0);
                //     printf("%s: Similarity score calculated in %8.2f ms\n", __func__, (t_main_end_us - t_similarity_score) / 1000.0);
                //     printf("%s: Total time: %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0);
                // }
            }
        }
    }

    // // 排序
    // std::sort(score_map.begin(), score_map.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b){
    //     return a.second > b.second;
    // });

    // std::cout << "max score: " << score_map[0].second << std::endl;
    // std::cout << "max path: " << score_map[0].first << std::endl;
    
    // std::cout << text << std::endl;

    // for (int i = 0; i < 4; i++) {
    //     cv::imshow("max jpg", cv::imread(score_map[i].first));
    //     std::cout << "score " << i << " " << score_map[i].second << std::endl;
    //     cv::waitKey(0);
    // }
    
    
    

    // Usage of the individual functions that make up clip_compare_text_and_image is demonstrated in the
    // `extract` example.

    clip_free(ctx);

    return 0;
}
