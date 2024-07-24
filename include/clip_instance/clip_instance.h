#pragma once
#include "clip.cpp/clip.h"
#include "clip_instance/common-clip.h"
#include "common.h"
#include <string>

// Clip单例类，实际上是对clip.cpp的再封装，用于加载Clip模型，接收mask类型输入，然后输出该mask的特征向量

namespace SamToClip {
class ClipInstance {
    // clip 参数
public:
    ClipInstance(std::string yaml_file);
    ~ClipInstance();
    bool load_model();
    bool generate_mask_feature(Segment& seg);
    // bool generate_masks_feature(std::vector<CropImg>& crop_imgs);
    bool to_clip_image_f32(clip_image_f32& res, const cv::Mat& img);
    bool generate_segments_feature(std::vector<Segment>& segments);
    float* generate_text_feature(const char* text);
    void compute_image_text_similarity(clip_image_u8& res, const char* text);
    app_params params_;

private:
    clip_ctx* clip_ctx_;
    int64_t time_infer_sum_;
    int cnt_;
    

};
}


