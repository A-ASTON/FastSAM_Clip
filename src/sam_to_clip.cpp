#include <iostream>
#include "clip.cpp/clip.h"
#include "clip_instance/clip_instance.h"
#include "fastsam_instance/fastam_instance.h"
#include "common.h"
#include <filesystem>
namespace fs = std::filesystem;
int main(int argc, char *argv[]) {
    std::string file("/home/catkin_ws/projects/clip_sam_ws/SamToClip/config/clip.yaml");
    SamToClip::ClipInstance clip_instance(file);

    std::string engine_file("/home/catkin_ws/projects/clip_sam_ws/SamToClip/models/FastSAM.engine");
    SamToClip::FastSamInstance fastsam_instance(engine_file);

    fs::path folder_path("/home/catkin_ws/projects/clip_sam_ws/SamToClip/seg_output/");

    const int64_t t_main_start_us = ggml_time_us();
    const char *text = clip_instance.params_.texts[0].c_str();
    float *vec = clip_instance.generate_text_feature(text);  
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        fs::path file_path = entry.path();
        if (!entry.is_directory()) { // 确保只处理文件，而不是子目录
            
            // 检查文件是否为图像文件
            if (file_path.extension() == ".png" || file_path.extension() == ".jpg" ||
                file_path.extension() == ".jpeg" || file_path.extension() == ".bmp") {
                std::string strPath = file_path.string();
                const char * img_path = strPath.c_str();
                cv::Mat img = cv::imread(img_path);
                const int64_t t_similarity_score = ggml_time_us(); // 获取相似度的起始时间

                fastsam_instance.infer(img);
                // 得到的objs包含了目标框和物体的mask
                fastsam_instance.crop_images(true); // 执行裁剪，生成segments_

                std::vector<SamToClip::Segment*> segments = fastsam_instance.get_segments();
                // clip_instance.generate_segments_feature(segments);
                clip_instance.generate_segments_feature_batch(segments);
                
                float max_score = 0;
                int max_idx = 0;
                int idx = 0;

                for (auto& seg : segments) {
                    if (seg->embedding_dim != 0) {
                        float similarity_score = clip_similarity_score(seg->embedding, vec, seg->embedding_dim);
                        if (similarity_score > max_score) {
                            max_score = similarity_score;
                            max_idx = idx;
                        }
                    }
                    idx++;
                } 

                std::cout << "max_score: " << max_score << std::endl;
                cv::imshow("Highly Realted to text " + std::string(text), segments[max_idx]->img);
                cv::waitKey(0);
                
                const int64_t t_similarity_score_end_us = ggml_time_us(); 
                if (1) {
                    printf("\n\nTimings\n");
                    printf("%s:Image in %8.2f ms\n", __func__, (t_similarity_score_end_us - t_similarity_score) / 1000.0);
                }
            }
        }
    }
    const int64_t t_main_end_us = ggml_time_us(); // 获取结束时间
    printf("%s: Total time: %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0);

    
    

    return 0;
}


