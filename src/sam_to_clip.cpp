#include <iostream>
#include "clip.cpp/clip.h"
#include "clip_instance/clip_instance.h"
#include "fastsam_instance/fastam_instance.h"
#include "common.h"

int main(int argc, char *argv[]) {
    std::string file("/home/catkin_ws/projects/clip_sam_ws/SamToClip/config/clip.yaml");
    SamToClip::ClipInstance clip_instance(file);

    std::string engine_file("/home/catkin_ws/projects/clip_sam_ws/SamToClip/models/FastSAM.engine");
    SamToClip::FastSamInstance fastsam_instance(engine_file);

    cv::Mat img = cv::imread(clip_instance.params_.image_paths[0]);
    fastsam_instance.infer(img);
    // 得到的objs包含了目标框和物体的mask
    fastsam_instance.crop_images(true);

    std::vector<SamToClip::Segment> segments = fastsam_instance.get_segments();
    clip_instance.generate_segments_feature(segments);

    const char *text = clip_instance.params_.texts[0].c_str();
    float *vec = clip_instance.generate_text_feature(text);  
    
    float max_score = 0;
    int max_idx = 0;
    int idx = 0;

    for (auto& seg : segments) {
        if (seg.embedding_dim != 0) {
            float similarity_score = clip_similarity_score(seg.embedding, vec, seg.embedding_dim);
            if (similarity_score > max_score) {
                max_score = similarity_score;
                max_idx = idx;
            }
        }
        idx++;
    } 

    std::cout << "max_score: " << max_score << std::endl;
    cv::imshow("Highly Realted to text " + std::string(text), segments[max_idx].img);
    cv::waitKey(0);

    return 0;
}


